"""
LangGraph multi-agent chatbot with per-agent MCP memory isolation.

Users dynamically create "writer" and "reader" agents via natural language,
then issue database tasks that get routed to the appropriate agent.  Each
agent uses a distinct ``agent_id`` so its memories land in isolated pgvector
collections and Apache AGE graphs on the MCP server.

Prerequisites:
    - YugabyteDB MCP server running in HTTP mode
    - OPENAI_API_KEY set in the environment

Usage:
    cd examples/langgraph_chatbot
    uv run python chatbot.py --url http://localhost:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from typing_extensions import TypedDict

DEFAULT_MCP_URL = "http://localhost:8000/mcp"
DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"

SUPERVISOR_PROMPT = """\
You are a routing supervisor for a multi-agent database system.

Currently registered agents: {agents}

Classify the user's latest message into exactly ONE intent and respond
with a single JSON object (no markdown fences, no extra text):

1. The user wants to CREATE a new agent:
   {{"intent": "create_agent", "agent_name": "<name>", "agent_type": "writer|reader"}}

2. The user wants an agent to REMEMBER a fact, preference, or piece of info:
   {{"intent": "agent_remember", "agent_name": "<name>", "text": "<what to remember>"}}

3. The user wants to RECALL or ask what an agent remembers:
   {{"intent": "agent_recall", "agent_name": "<name>", "query": "<what to search for>"}}

4. The user wants to USE an existing agent for a database SQL task:
   {{"intent": "agent_task", "agent_name": "<name>", "task": "<what to do>"}}

5. General conversation (none of the above):
   {{"intent": "general", "response": "<your reply>"}}

Rules:
- agent_type MUST be exactly "writer" or "reader".
- For intents 2-4 the agent_name MUST be one of the registered agents.
- When the user says "remember", "store", "note", "save" -> use agent_remember.
- When the user says "recall", "what do you remember", "what do you know" -> use agent_recall.
- When the user asks to run SQL, query a table, insert/update/delete rows -> use agent_task.
- Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    agents: dict[str, dict]
    current_agent: str | None
    tool_output: str | None


# ---------------------------------------------------------------------------
# MCP helper
# ---------------------------------------------------------------------------

async def call_tool(session: ClientSession, name: str, args: dict[str, Any]) -> str:
    result = await session.call_tool(name, args)
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(mcp_session: ClientSession, llm: ChatOpenAI):
    """Build and compile the LangGraph state machine."""

    # -- nodes ---------------------------------------------------------------

    async def supervisor(state: ChatState) -> dict:
        agents_desc = json.dumps(state.get("agents", {})) or "none registered yet"
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT.format(agents=agents_desc)),
        ] + state["messages"]

        response = await llm.ainvoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"intent": "general", "response": content}

        intent = parsed.get("intent", "general")

        if intent == "create_agent":
            return {
                "current_agent": parsed.get("agent_name", ""),
                "tool_output": json.dumps(parsed),
            }

        if intent in ("agent_task", "agent_remember", "agent_recall"):
            agent_name = parsed.get("agent_name", "")
            if agent_name not in state.get("agents", {}):
                return {
                    "messages": [AIMessage(
                        content=f"Agent '{agent_name}' is not registered. "
                                "Please create it first.",
                    )],
                    "current_agent": None,
                    "tool_output": None,
                }
            return {
                "current_agent": agent_name,
                "tool_output": json.dumps(parsed),
            }

        # general
        return {
            "messages": [AIMessage(content=parsed.get("response", content))],
            "current_agent": None,
            "tool_output": None,
        }

    async def create_agent(state: ChatState) -> dict:
        parsed = json.loads(state["tool_output"])
        agent_name = parsed["agent_name"]
        agent_type = parsed["agent_type"]

        agents = dict(state.get("agents", {}))
        agents[agent_name] = {"type": agent_type}

        return {
            "agents": agents,
            "messages": [AIMessage(
                content=f"Agent **{agent_name}** created (type: {agent_type}). "
                        f"Its memories will be stored in the "
                        f"`{agent_name}_mem0_graph` graph and "
                        f"`{agent_name}_mem0_collection` vector collection.",
            )],
            "current_agent": None,
            "tool_output": None,
        }

    async def agent_remember(state: ChatState) -> dict:
        parsed = json.loads(state["tool_output"])
        agent_name = state["current_agent"]
        text = parsed["text"]

        try:
            await call_tool(mcp_session, "add_memory", {
                "text": text,
                "agent_id": agent_name,
            })
            msg = f"[{agent_name}] Stored in memory: \"{text}\""
        except Exception as e:
            msg = f"[{agent_name}] Failed to store memory: {e}"

        return {
            "messages": [AIMessage(content=msg)],
            "current_agent": None,
            "tool_output": None,
        }

    async def agent_recall(state: ChatState) -> dict:
        parsed = json.loads(state["tool_output"])
        agent_name = state["current_agent"]
        query = parsed["query"]

        try:
            raw = await call_tool(mcp_session, "search_memories", {
                "query": query,
                "agent_id": agent_name,
                "limit": 10,
            })
            mem_data = json.loads(raw)
            results = (
                mem_data.get("results", mem_data)
                if isinstance(mem_data, dict) else mem_data
            )
            if results:
                lines = []
                for item in results:
                    m = item.get("memory", "") if isinstance(item, dict) else str(item)
                    if m:
                        lines.append(f"  - {m}")
                if lines:
                    msg = f"[{agent_name}] Memories matching \"{query}\":\n" + "\n".join(lines)
                else:
                    msg = f"[{agent_name}] No memories found for \"{query}\"."
            else:
                msg = f"[{agent_name}] No memories found for \"{query}\"."

            relations = mem_data.get("relations") if isinstance(mem_data, dict) else None
            if relations:
                rel_lines = []
                for rel in relations:
                    src = rel.get("source", "?")
                    tgt = rel.get("target", rel.get("destination", "?"))
                    label = rel.get("relationship", rel.get("label", "?"))
                    rel_lines.append(f"  - {src} --[{label}]--> {tgt}")
                if rel_lines:
                    msg += "\n  Graph relations:\n" + "\n".join(rel_lines)
        except Exception as e:
            msg = f"[{agent_name}] Error recalling memories: {e}"

        return {
            "messages": [AIMessage(content=msg)],
            "current_agent": None,
            "tool_output": None,
        }

    async def agent_executor(state: ChatState) -> dict:
        parsed = json.loads(state["tool_output"])
        agent_name = state["current_agent"]
        agent_info = state["agents"][agent_name]
        agent_type = agent_info["type"]
        task = parsed["task"]

        # 1. Recall prior memories for this agent
        memories = ""
        try:
            raw = await call_tool(mcp_session, "search_memories", {
                "query": task,
                "agent_id": agent_name,
                "limit": 5,
            })
            mem_data = json.loads(raw)
            results = (
                mem_data.get("results", mem_data)
                if isinstance(mem_data, dict) else mem_data
            )
            if results:
                lines = []
                for item in results:
                    m = item.get("memory", "") if isinstance(item, dict) else str(item)
                    if m:
                        lines.append(f"- {m}")
                memories = "\n".join(lines)
        except Exception:
            pass

        # 2. Generate SQL using the LLM
        role_hint = (
            "You write INSERT, UPDATE, or DELETE statements."
            if agent_type == "writer"
            else "You write SELECT statements only."
        )
        sql_prompt = (
            f"You are a database {agent_type} agent. {role_hint}\n\n"
            f"Task: {task}\n\n"
        )
        if memories:
            sql_prompt += f"Previous context from your memory:\n{memories}\n\n"
        sql_prompt += "Output ONLY the raw SQL query, no explanation."

        sql_response = await llm.ainvoke([HumanMessage(content=sql_prompt)])
        sql = sql_response.content.strip()

        if sql.startswith("```"):
            sql = sql.split("\n", 1)[-1]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()

        # 3. Execute via MCP
        tool_name = "run_write_query" if agent_type == "writer" else "run_read_only_query"
        try:
            result = await call_tool(mcp_session, tool_name, {"query": sql})
        except Exception as e:
            result = f"Error: {e}"

        # 4. Store this interaction in the agent's isolated memory
        memory_text = f"Task: {task} | SQL: {sql} | Result: {result}"
        try:
            await call_tool(mcp_session, "add_memory", {
                "text": memory_text,
                "agent_id": agent_name,
            })
        except Exception:
            pass

        # 5. Summarise for the user
        summary_prompt = (
            f"The user asked: {task}\n"
            f"SQL executed: {sql}\n"
            f"Raw result: {result}\n\n"
            "Give a clear, concise summary."
        )
        final = await llm.ainvoke([HumanMessage(content=summary_prompt)])

        return {
            "messages": [AIMessage(content=f"[{agent_name}] {final.content}")],
            "current_agent": None,
            "tool_output": None,
        }

    # -- routing -------------------------------------------------------------

    def route(state: ChatState) -> str:
        output = state.get("tool_output")
        if output:
            try:
                parsed = json.loads(output)
                intent = parsed.get("intent")
                if intent == "create_agent":
                    return "create_agent"
                if intent == "agent_remember":
                    return "agent_remember"
                if intent == "agent_recall":
                    return "agent_recall"
                if intent == "agent_task":
                    return "agent_executor"
            except (json.JSONDecodeError, KeyError):
                pass
        return "__end__"

    # -- wiring --------------------------------------------------------------

    graph = StateGraph(ChatState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("create_agent", create_agent)
    graph.add_node("agent_remember", agent_remember)
    graph.add_node("agent_recall", agent_recall)
    graph.add_node("agent_executor", agent_executor)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, {
        "create_agent": "create_agent",
        "agent_remember": "agent_remember",
        "agent_recall": "agent_recall",
        "agent_executor": "agent_executor",
        "__end__": END,
    })
    graph.add_edge("create_agent", END)
    graph.add_edge("agent_remember", END)
    graph.add_edge("agent_recall", END)
    graph.add_edge("agent_executor", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

async def chat_loop(url: str, model: str) -> None:
    llm = ChatOpenAI(model=model)

    print(f"Connecting to MCP server at {url} ...")
    async with streamable_http_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Connected. {len(tool_names)} MCP tools available.")
            print(
                "Commands (natural language):\n"
                '  Create agents:  "Create a reader agent called db_reader"\n'
                '  DB tasks:       "Using db_reader, show all customers"\n'
                '  Remember:       "db_reader remember that I like ice cream"\n'
                '  Recall:         "db_reader what do you remember?"\n'
            )
            print("Type 'quit' or 'exit' to end.\n")

            app = build_graph(session, llm)
            state: dict[str, Any] = {
                "messages": [],
                "agents": {},
                "current_agent": None,
                "tool_output": None,
            }

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

                state["messages"] = list(state["messages"]) + [
                    HumanMessage(content=user_input),
                ]

                result = await app.ainvoke(state)
                state = result

                last_msg = state["messages"][-1]
                print(f"\nAssistant: {last_msg.content}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph multi-agent chatbot with per-agent MCP memory",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_URL),
        help=f"MCP server URL (default: {DEFAULT_MCP_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI chat model (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(chat_loop(args.url, args.model))


if __name__ == "__main__":
    main()
