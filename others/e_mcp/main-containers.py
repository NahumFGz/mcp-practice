from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel


class MCPQueryProcessor:
    """
    Clase para procesar consultas utilizando MultiServerMCPClient con servidores SSE.
    """

    def __init__(self):
        # Configuración de servidores SSE
        self.server_configs = {
            "assistance": {
                "url": "http://localhost:9001/sse",
                "transport": "sse",
            },
            "contracting": {
                "url": "http://localhost:9002/sse",
                "transport": "sse",
            },
            "voting": {
                "url": "http://localhost:9003/sse",
                "transport": "sse",
            },
        }

        self.client = None
        self.graph = None
        self.model = init_chat_model("openai:gpt-4.1")

    async def __aenter__(self):
        self.client = MultiServerMCPClient(self.server_configs)
        await self.client.__aenter__()

        # Obtener herramientas del cliente
        tools = self.client.get_tools()

        # Definir la función de llamada al modelo
        def call_model(state: MessagesState):
            response = self.model.bind_tools(tools).invoke(state["messages"])
            return {"messages": response}

        # Construir el grafo
        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_node(ToolNode(tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            tools_condition,
        )
        builder.add_edge("tools", "call_model")
        self.graph = builder.compile()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def process_query(self, query: str):
        if not self.graph:
            raise RuntimeError("El procesador no está inicializado. Usa 'async with' primero.")

        response = await self.graph.ainvoke({"messages": query})
        return response


# Clase singleton para mantener una instancia global del procesador
class MCPProcessorManager:
    _instance = None
    _initialized = False

    @classmethod
    async def get_instance(cls):
        if cls._instance is None or not cls._initialized:
            if cls._instance:
                # Si existe una instancia previa pero no está inicializada, cerramos
                try:
                    await cls._instance.__aexit__(None, None, None)
                except:
                    pass

            cls._instance = MCPQueryProcessor()
            try:
                await cls._instance.__aenter__()
                cls._initialized = True
            except Exception as e:
                cls._instance = None
                cls._initialized = False
                raise e

        return cls._instance

    @classmethod
    async def close(cls):
        if cls._instance and cls._initialized:
            await cls._instance.__aexit__(None, None, None)
            cls._initialized = False


# Modelos de Pydantic para la API
class QueryRequest(BaseModel):
    query: str
    options: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    result: Any
    status: str = "success"


# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Consultas MCP con servidores SSE",
    description="API para procesar consultas usando MCP Adapters con servidores SSE",
)


# Dependencia para obtener el procesador
async def get_processor():
    processor = await MCPProcessorManager.get_instance()
    return processor


# Único endpoint para procesar consultas
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, processor: MCPQueryProcessor = Depends(get_processor)
):
    try:
        # Procesar la consulta
        result = await processor.process_query(request.query)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")


# Evento de inicio de la aplicación
@app.on_event("startup")
async def startup_event():
    # Inicializar el procesador al iniciar la aplicación
    try:
        await MCPProcessorManager.get_instance()
        print("Procesador MCP inicializado correctamente. Conectado a servidores SSE.")
    except Exception as e:
        print(f"Error al inicializar el procesador MCP: {str(e)}")


# Evento de cierre de la aplicación
@app.on_event("shutdown")
async def shutdown_event():
    # Cerrar el procesador al cerrar la aplicación
    await MCPProcessorManager.close()
    print("Procesador MCP cerrado correctamente.")
