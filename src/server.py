# server.py
import json
import os
import sys
import argparse
import psycopg2
from typing import List, Dict, Any, AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager, AsyncExitStack
from mcp.server.fastmcp import FastMCP, Context
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse

@dataclass
class AppContext:
    conn: psycopg2.extensions.connection

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    database_url = os.getenv("YUGABYTEDB_URL")
    if not database_url:
        print("YUGABYTEDB_URL is not set", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...", file=sys.stderr)
    conn = psycopg2.connect(database_url)
    try:
        yield AppContext(conn=conn)
    finally:
        print("Closing database connection...", file=sys.stderr)
        conn.close()


def summarize_database(ctx: Context, schema: str = "public") -> List[Dict[str, Any]]:
    """
    Summarize the database: list tables with schema and row counts.
    """
    summary = []
    conn = ctx.request_context.lifespan_context.conn
    with conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
            """, (schema,))
            tables = [row[0] for row in cur.fetchall()]

            for table in tables:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema, table,))
                schema_info = [{"column_name": col, "data_type": dtype} for col, dtype in cur.fetchall()]

                cur.execute(f"SELECT COUNT(*) FROM {schema}.\"{table}\"")
                row_count = cur.fetchone()[0]

                summary.append({
                    "table": table,
                    "row_count": row_count,
                    "schema": schema_info
                })

        except Exception as e:
            summary.append({"error": str(e)})

    return summary


def run_read_only_query(ctx: Context, query: str) -> str:
    """
    Run a read-only SQL query and return the results as JSON.
    """
    conn = ctx.request_context.lifespan_context.conn
    with conn.cursor() as cur:
        try:
            cur.execute("BEGIN READ ONLY")
            cur.execute(query)
            rows = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            result = [dict(zip(column_names, row)) for row in rows]
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing query: {e}"
        finally:
            try:
                cur.execute("ROLLBACK")
            except Exception as e:
                return f"Couldn't ROLLBACK transaction: {e}"


def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        default=os.environ.get("YB_MCP_TRANSPORT", "stdio"),
        help="stdio | http (env: YB_MCP_TRANSPORT)",
    )
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        default=os.environ.get("YB_MCP_STATELESS_HTTP", "").lower() == "true",
        help="Enable stateless HTTP mode (env: YB_MCP_STATELESS_HTTP=true)",
    )
    parser.add_argument(
        "--yugabytedb-url",
        default=os.environ.get("YUGABYTEDB_URL"),
        help="YugabyteDB connection string (env: YUGABYTEDB_URL)",
    )
    return parser.parse_args()


class YugabyteDBMCPServer:
    def __init__(self, transport: str, stateless_http: bool):
        self.transport = transport
        self.stateless_http = stateless_http

        self.mcp = FastMCP(
            "yugabytedb-mcp",
            lifespan=app_lifespan,
            json_response=True,
            stateless_http=stateless_http,
        )

        self._register_tools()

    def _register_tools(self):
        self.mcp.add_tool(summarize_database)
        self.mcp.add_tool(run_read_only_query)

    def run(self, host="0.0.0.0", port=8000):
        if self.transport == "http":
            self._run_http(host, port)
        else:
            self.mcp.run(transport="stdio")

    def _run_http(self, host, port):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(self.mcp.session_manager.run())
                yield

        app = FastAPI(lifespan=lifespan)

        @app.get("/ping")
        async def ping():
            return JSONResponse({"status": "ok"})

        app.mount("/", self.mcp.streamable_http_app())

        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cfg = parse_config()

    server = YugabyteDBMCPServer(
        transport=cfg.transport,
        stateless_http=cfg.stateless_http,
    )

    server.run()
