import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["/Volumes/Projects/TemporalProjects/McpDemo/a_mcp/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    ) as client:
        agent = create_react_agent(
            "openai:gpt-4o-mini",  # puedes reemplazarlo con otro modelo si usas OpenAI u otro
            client.get_tools(),
        )

        # Prueba math
        math_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
        )
        print("Math response:", math_response)

        # Prueba weather
        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
        )
        print("Weather response:", weather_response)


if __name__ == "__main__":
    asyncio.run(main())
