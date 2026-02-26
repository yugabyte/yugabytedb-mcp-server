# LangGraph Multi-Agent Chatbot

A LangGraph chatbot that lets you dynamically create **writer** and **reader**
database agents via natural language. Each agent gets its own isolated memory
store (separate pgvector collection and Apache AGE graph) on the YugabyteDB MCP
server.

## Prerequisites

1. **YugabyteDB MCP server** running in HTTP mode:

       # from the repo root
       cd src
       python server.py --transport http

2. **OPENAI_API_KEY** set in your environment.

## Setup

    cd examples/langgraph_chatbot
    uv sync

## Run

    uv run python chatbot.py --url http://localhost:8000/mcp

Or with a custom model:

    uv run python chatbot.py --url http://localhost:8000/mcp --model gpt-4.1-mini

## Example session

    You: Create a writer agent called db_writer

    Assistant: Agent **db_writer** created (type: writer). Its memories will be
    stored in the `db_writer_mem0_graph` graph and `db_writer_mem0_collection`
    vector collection.

    You: Create a reader agent called db_reader

    Assistant: Agent **db_reader** created (type: reader). Its memories will be
    stored in the `db_reader_mem0_graph` graph and `db_reader_mem0_collection`
    vector collection.

    You: Using db_writer, create a users table with columns id, name, and age

    Assistant: [db_writer] Created the `users` table with columns id (serial),
    name (text), and age (integer).

    You: Using db_writer, insert a user Alice aged 30

    Assistant: [db_writer] Inserted user Alice (age 30) into the users table.
    1 row affected.

    You: Using db_reader, show all users

    Assistant: [db_reader] Found 1 user in the table:
    | id | name  | age |
    |----|-------|-----|
    | 1  | Alice | 30  |

## How it works

The chatbot uses a LangGraph state machine with three nodes:

- **supervisor** -- Classifies user input into `create_agent`, `agent_task`,
  or `general` conversation using an LLM.
- **create_agent** -- Registers a new agent (writer or reader) in the graph
  state.
- **agent_executor** -- Recalls memories for the target agent, generates SQL
  via the LLM, executes it through the MCP server, and stores the interaction
  back as a memory.

Each agent's `agent_id` is passed to the MCP server's mem0 tools, which
automatically routes to isolated storage:

- Graph: `{agent_id}_mem0_graph`
- Vector collection: `{agent_id}_mem0_collection`
