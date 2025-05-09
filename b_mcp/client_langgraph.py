# client_langgraph.py
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o-mini")  # Asegúrate de tener tu OPENAI_API_KEY exportada


async def main():
    async with MultiServerMCPClient(
        {
            "transparencia": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)

        pregunta1 = "¿Cuántos contratos tiene la empresa con RUC 20512345678?"
        respuesta1 = await agent.ainvoke({"messages": pregunta1})
        print(f"\n🧾 Pregunta: {pregunta1}\n🔍 Respuesta: {respuesta1}\n")

        pregunta2 = "¿Qué asistencias tiene el congresista Alejandro Soto?"
        respuesta2 = await agent.ainvoke({"messages": pregunta2})
        print(f"\n📋 Pregunta: {pregunta2}\n🔍 Respuesta: {respuesta2}\n")

        pregunta3 = "¿Qué votaciones hubo sobre salud mental?"
        respuesta3 = await agent.ainvoke({"messages": pregunta3})
        print(f"\n🗳️ Pregunta: {pregunta3}\n🔍 Respuesta: {respuesta3}\n")


asyncio.run(main())
