# main.py
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, Optional
from uuid import uuid4

from decouple import config
from fastapi import Depends, FastAPI, HTTPException, Query
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from typing_extensions import TypedDict

# ---------- Estado del Grafo ------------------------------------------------ #


class ChatState(TypedDict):
    """
    - messages: historial (HumanMessage, AIMessage, etc.)
    - topic_decision: 'yes'/'no' indica si la última pregunta
      se considera 'respondible' en el contexto la transparencia gubernamental
    """

    messages: Annotated[list, add_messages]
    topic_decision: str


# ---------- Procesador MCP con Memoria -------------------------------------- #


class MCPQueryProcessor:
    """
    Conecta a varios servidores MCP (vía SSE) y expone `run(query, thread_id)`
    para obtener la respuesta del grafo de LangGraph con memoria persistente.
    """

    SERVERS = {
        "assistance": {"url": "http://localhost:9001/sse", "transport": "sse"},
        "contracting": {"url": "http://localhost:9002/sse", "transport": "sse"},
        "voting": {"url": "http://localhost:9003/sse", "transport": "sse"},
    }

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._graph = None
        self._memory_saver = None

        # Configuración del LLM para clasificación y respuesta
        self._llm_classifier = None
        self._llm_main = None
        self._llm_fallback = None

    async def start(self) -> None:
        """Abre la conexión y construye el grafo una sola vez."""
        # 1. Inicializar conexión MCP
        self._client = MultiServerMCPClient(self.SERVERS)
        await self._client.__aenter__()  # abre todas las conexiones

        # 2. Obtener herramientas MCP
        tools = self._client.get_tools()

        # 3. Inicializar modelos
        self._llm_classifier = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
        self._llm_main = init_chat_model("openai:gpt-4o", temperature=0.7)
        self._llm_fallback = init_chat_model("openai:gpt-4o-mini", temperature=0.7)

        # 4. Configurar prompts del sistema
        self._system_message = SystemMessage(
            content=(
                "Eres un asistente conversacional especializado en temas de transparencia gubernamental del Estado peruano. "
                "Respondes de forma clara, respetuosa y profesional, utilizando fuentes confiables cuando es necesario. "
                "Puedes ayudar con información sobre contrataciones públicas, empresas proveedoras del Estado, "
                "asistencia y votaciones del Congreso entre los años 2006 y 2024. "
                "Tu objetivo es facilitar el acceso a datos públicos relevantes para la ciudadanía."
            )
        )

        self._fallback_system_message = SystemMessage(
            content=(
                "Eres un asistente cordial y profesional. Aunque no puedes responder preguntas "
                "fuera del dominio de transparencia gubernamental del Estado peruano, debes explicar "
                "educadamente cuál es tu función y sugerir temas válidos, como contrataciones públicas o votaciones del Congreso. "
                "Si el usuario simplemente saluda, responde con cortesía e invita a hacer una consulta sobre esos temas."
            )
        )

        # 5. Configurar conexión a Postgres
        DB_HOST = config("DB_HOST")
        DB_PORT = config("DB_PORT")
        DB_NAME = config("DB_NAME")
        DB_USER = config("DB_USER")
        DB_PASSWORD = config("DB_PASSWORD")

        DB_URI = (
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
        )

        # Crear el pool sin abrirlo en el constructor
        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,  # evita errores con statements cacheados
            },
            open=False,  # Importante: no abrir en el constructor
        )
        # Abrir el pool explícitamente
        await pool.open()
        self._memory_saver = AsyncPostgresSaver(pool)

        # 6. Construir el grafo
        self._build_graph(tools)

    def _build_graph(self, tools):
        """Construye el grafo con los nodos necesarios."""

        # Definir nodos
        async def topic_classifier_node(state: ChatState):
            """
            Analiza todo el historial y decide si la última pregunta del usuario
            se puede responder en el contexto de transparencia gubernamental.

            Devuelve {"topic_decision": "yes"} o {"topic_decision": "no"}.
            """
            if not state["messages"]:
                return {"topic_decision": "no"}

            conversation_text = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    conversation_text += f"Usuario: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    conversation_text += f"Asistente: {msg.content}\n"

            classification_prompt = f"""
        Eres un verificador que decide si la última pregunta del usuario puede
        ser respondida en el contexto de transparencia gubernamental del Estado
        peruano, considerando toda la conversación previa.

        - Si la conversación está relacionada con contrataciones públicas, 
          empresas proveedoras del Estado, o actividad parlamentaria (asistencia,
          votaciones, licencias), responde 'yes'.
        - Si no está relacionada con eso, responde 'no'.

        Responde únicamente con 'yes' o 'no'.

        --- CONVERSACIÓN ---
        {conversation_text}
        --- FIN ---
        ¿Se puede responder esta última pregunta dentro del contexto de transparencia gubernamental?
        """

            decision_msg = await self._llm_classifier.ainvoke(
                [HumanMessage(content=classification_prompt)]
            )
            decision_str = decision_msg.content.strip().lower()

            if decision_str.startswith("y"):
                return {"topic_decision": "yes"}
            else:
                return {"topic_decision": "no"}

        def route_topic(state: ChatState) -> str:
            """
            Lee state["topic_decision"] y retorna 'chatbot' o 'fallback'.
            """
            return "chatbot" if state.get("topic_decision", "no").startswith("y") else "fallback"

        async def chatbot_node(state: ChatState):
            """
            Nodo principal:
            - Usa todo el historial + system_message
            - Invoca llm_with_tools (con acceso a herramientas como MCP).
            """
            messages = [self._system_message] + state["messages"]
            llm_with_tools = self._llm_main.bind_tools(tools)
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        async def fallback_node(state: ChatState):
            """
            Usa un modelo especializado para responder amablemente cuando la consulta
            está fuera de dominio. Informa al usuario sobre el propósito del asistente.
            """
            messages = [self._fallback_system_message] + state["messages"]
            response = await self._llm_fallback.ainvoke(messages)
            return {"messages": [response]}

        # Crear el grafo
        graph_builder = StateGraph(ChatState)

        # Añadir nodos
        graph_builder.add_node("topic_classifier", topic_classifier_node)
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_node("fallback", fallback_node)
        graph_builder.add_node("tools", ToolNode(tools))

        # Conectar nodos
        graph_builder.add_edge(START, "topic_classifier")
        graph_builder.add_conditional_edges(
            "topic_classifier",
            route_topic,
            {"chatbot": "chatbot", "fallback": "fallback"},
        )
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("fallback", END)

        # Compilar con memoria
        self._graph = graph_builder.compile(checkpointer=self._memory_saver)

    async def stop(self) -> None:
        """Cierra la conexión limpia‑mente."""
        if self._client:
            await self._client.__aexit__(None, None, None)

    async def run(self, query: str, thread_id: str = None):
        """
        Ejecuta una consulta con el grafo y mantiene la memoria en PostgreSQL.

        Args:
            query: Texto de la consulta del usuario
            thread_id: Identificador de la conversación para mantener el contexto
                    (si es None, se genera uno aleatorio)
        """
        if not self._graph:
            raise RuntimeError("MCPQueryProcessor no ha sido inicializado.")

        # Si no hay thread_id, generar uno nuevo
        if not thread_id:
            thread_id = str(uuid4())

        # Preparar entrada y configuración
        config = {"configurable": {"thread_id": thread_id}}
        human_message = HumanMessage(content=query)
        entrada = {"messages": [human_message]}

        # Ejecutar el grafo con memoria
        resultado = await self._graph.ainvoke(entrada, config=config)

        # Devolver el resultado y el thread_id
        return {
            "thread_id": thread_id,
            "response": (
                resultado["messages"][-1].content
                if "messages" in resultado
                else "No se pudo generar una respuesta."
            ),
        }


# ---------- FastAPI --------------------------------------------------------- #


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None  # por si quieres filtros futuros


class QueryResponse(BaseModel):
    result: Dict[str, Any]
    status: str = "success"


# Lifespan: crea y destruye el procesador automáticamente
@asynccontextmanager
async def lifespan(app: FastAPI):
    processor = MCPQueryProcessor()
    await processor.start()
    app.state.processor = processor
    print("✅ Procesador MCP con memoria PostgreSQL listo 🚀")
    try:
        yield
    finally:
        await processor.stop()
        print("👋 Procesador MCP detenido")


app = FastAPI(
    title="API de Consultas MCP con Memoria PostgreSQL",
    description="Procesa consultas con MultiServerMCPClient, LangGraph y memoria persistente",
    lifespan=lifespan,
)


# Dependencia sencilla: sólo lee de app.state
def get_processor() -> MCPQueryProcessor:
    return app.state.processor  # type: ignore[attr-defined]


@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest, proc: MCPQueryProcessor = Depends(get_processor)):
    """
    Procesa una consulta y mantiene el contexto de la conversación usando
    el thread_id proporcionado (o generando uno nuevo).
    """
    try:
        result = await proc.run(req.query, req.thread_id)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {e}")


@app.get("/conversations/{thread_id}", response_model=Dict[str, Any])
async def get_conversation_context(
    thread_id: str, proc: MCPQueryProcessor = Depends(get_processor)
):
    """
    Endpoint para obtener el historial de una conversación específica
    según su thread_id. Esto podría implementarse posteriormente.
    """
    # Nota: Esto es un placeholder para una futura implementación
    # que podría recuperar el historial desde PostgreSQL
    return {"thread_id": thread_id, "status": "not_implemented_yet"}
