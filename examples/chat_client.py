"""
Interactive chat client that connects to the YugabyteDB MCP server over HTTP.

Uses OpenAI for conversation and the MCP server's mem0 tools for persistent
memory -- every exchange is stored and prior context is retrieved before
each response.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/chat_client.py --user-id alice

    # Custom server URL:
    python examples/chat_client.py --url http://remote-host:8000/mcp --user-id alice
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI


DEFAULT_MCP_URL = "http://localhost:8000/mcp"
DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"

SYSTEM_PROMPT = """\
You are a helpful assistant with access to long-term memory.
Before answering, you will be given relevant memories from previous conversations.
Use them to personalise your responses.  If the user corrects you or shares new
preferences, those will be remembered for next time."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat client with MCP memory")
    parser.add_argument(
        "--url",
        default=os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_URL),
        help=f"MCP server URL (default: {DEFAULT_MCP_URL})",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Your user ID (passed to mem0 for memory scoping).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI chat model (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


async def call_tool(session: ClientSession, name: str, args: dict[str, Any]) -> str:
    result = await session.call_tool(name, args)
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


async def store_memory(session: ClientSession, text: str, user_id: str, role: str) -> None:
    await call_tool(session, "add_memory", {
        "text": text,
        "user_id": user_id,
        "metadata": json.dumps({"role": role}),
    })


async def recall_memories(session: ClientSession, query: str, user_id: str) -> str:
    raw = await call_tool(session, "search_memories", {
        "query": query,
        "user_id": user_id,
        "limit": 5,
    })
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""

    results = data.get("results", data) if isinstance(data, dict) else data
    if not results:
        return ""

    lines = []
    for item in results:
        mem = item.get("memory", "") if isinstance(item, dict) else str(item)
        if mem:
            lines.append(f"- {mem}")
    return "\n".join(lines)


async def chat_loop(url: str, user_id: str, model: str) -> None:
    openai = OpenAI()

    print(f"Connecting to MCP server at {url} ...")
    async with streamable_http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Connected. {len(tool_names)} tools available: {', '.join(tool_names)}")
            print(f"User: {user_id}")
            print("Type 'quit' or 'exit' to end the session.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                memory_context = await recall_memories(session, user_input, user_id)

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if memory_context:
                    messages.append({
                        "role": "system",
                        "content": f"Relevant memories from past conversations:\n{memory_context}",
                    })
                messages.append({"role": "user", "content": user_input})

                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                answer = response.choices[0].message.content or ""
                print(f"\nAssistant: {answer}\n")

                await store_memory(session, user_input, user_id, role="user")
                await store_memory(session, answer, user_id, role="assistant")


def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(chat_loop(args.url, args.user_id, args.model))


if __name__ == "__main__":
    main()
