import asyncio
from typing import Optional, Callable
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from mcp.client.streamable_http import streamable_http_client
import httpx

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self._get_session_id: list[Optional[Callable[[], Optional[str]]]] = [None]
        self._streams_context = None
        self._session_context = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def connect_to_server(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        # is_python = server_script_path.endswith('.py')
        # is_js = server_script_path.endswith('.js')
        # if not (is_python or is_js):
        #     raise ValueError("Server script must be a .py or .js file")

        # command = "python" if is_python else "node"
        # server_params = StdioServerParameters(
        #     command=command,
        #     args=[server_script_path],
        #     env=os.environ.copy()
        # )

        # stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        # self.stdio, self.write = stdio_transport
        # self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # await self.session.initialize()

        # """Connect to an MCP server running with HTTP Streamable transport"""

        async def add_headers(request: httpx.Request):
            request.headers["x-db-user"] = "user_full_access"
            request.headers["x-db-password"] = "yugabyte"
            request.headers["x-db-host"] = "localhost"
            request.headers["x-db-port"] = "5433"
            request.headers["x-db-name"] = "yugabyte"

            get_sid = self._get_session_id[0]
            if get_sid:
                session_id = get_sid()
                if session_id:
                    request.headers["mcp-session-id"] = session_id

        self._http_client = httpx.AsyncClient(
            event_hooks={"request": [add_headers]}
        )
        self._streams_context = streamable_http_client(
            "http://localhost:8000/mcp",
            http_client=self._http_client,
            terminate_on_close=True,
        )
        read_stream, write_stream, get_session_id = await self._streams_context.__aenter__()
        self._get_session_id[0] = get_session_id

        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()

        await self.session.initialize()

        # async with streamable_http_client("http://localhost:8000/mcp/", http_client=http_client_obj) as (
        # read_stream,
        # write_stream,
        # _,):
        #     # Create a session using the client streams
        #     async with ClientSession(read_stream, write_stream) as self.session:
        #         # Initialize the connection
        #         await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        while True:
            response = self.openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=available_tools,
            )

            message = response.choices[0].message

            # Normal assistant reply
            if not message.tool_calls:
                return message.content or ""

            # Tool calls
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = await self.session.call_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content,
                })

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources and close the MCP session (sends DELETE so server closes DB connection)."""
        # Exit session first, then streams context (streams exit sends DELETE to server)
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing session: {e}", flush=True)
            self._session_context = None
            self.session = None
        if self._streams_context is not None:
            try:
                await self._streams_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing streams: {e}", flush=True)
            self._streams_context = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        await self.exit_stack.aclose()


async def main():

    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())