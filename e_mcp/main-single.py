# app.py
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

# ---------- Procesador MCP -------------------------------------------------- #


class MCPQueryProcessor:
    """
    Conecta a varios servidores MCP (vía SSE) y expone `run(query)` para
    obtener la respuesta del grafo de LangGraph.
    """

    SERVERS = {
        "assistance": {"url": "http://localhost:9001/sse", "transport": "sse"},
        "contracting": {"url": "http://localhost:9002/sse", "transport": "sse"},
        "voting": {"url": "http://localhost:9003/sse", "transport": "sse"},
    }

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._graph = None

    async def start(self) -> None:
        """Abre la conexión y construye el grafo una sola vez."""
        self._client = MultiServerMCPClient(self.SERVERS)
        await self._client.__aenter__()  # abre todas las conexiones

        tools = self._client.get_tools()
        model = init_chat_model("openai:gpt-4.1")

        def call_model(state: MessagesState):
            resp = model.bind_tools(tools).invoke(state["messages"])
            return {"messages": resp}

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")
        self._graph = builder.compile()

    async def stop(self) -> None:
        """Cierra la conexión limpia‑mente."""
        if self._client:
            await self._client.__aexit__(None, None, None)

    async def run(self, query: str):
        if not self._graph:
            raise RuntimeError("MCPQueryProcessor no ha sido inicializado.")
        return await self._graph.ainvoke({"messages": query})


# ---------- FastAPI --------------------------------------------------------- #


class QueryRequest(BaseModel):
    query: str
    options: Optional[Dict[str, Any]] = None  # por si quieres filtros futuros


class QueryResponse(BaseModel):
    result: Any
    status: str = "success"


# Lifespan: crea y destruye el procesador automáticamente
@asynccontextmanager
async def lifespan(app: FastAPI):
    processor = MCPQueryProcessor()
    await processor.start()
    app.state.processor = processor
    print("✅ Procesador MCP listo 🚀")
    try:
        yield
    finally:
        await processor.stop()
        print("👋 Procesador MCP detenido")


app = FastAPI(
    title="API de Consultas MCP con servidores SSE",
    description="Procesa consultas con MultiServerMCPClient y LangGraph",
    lifespan=lifespan,
)


# Dependencia sencilla: sólo lee de app.state
def get_processor() -> MCPQueryProcessor:
    return app.state.processor  # type: ignore[attr-defined]


@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest, proc: MCPQueryProcessor = Depends(get_processor)):
    try:
        result = await proc.run(req.query)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {e}")
