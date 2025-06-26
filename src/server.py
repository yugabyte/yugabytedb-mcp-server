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


# Initialize MCP server
mcp = FastMCP("yugabytedb-mcp", lifespan=app_lifespan, json_response=True)


@mcp.tool()
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


@mcp.tool()
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


def run_stdio():
    mcp.run(transport="stdio")


def run_http():
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp.session_manager.run())
            yield
    app = FastAPI(lifespan=lifespan)
    @app.get("/ping")
    async def ping():
        return JSONResponse({"status": "ok"})
    
    app.mount("/invocations", mcp.streamable_http_app())
    
    uvicorn.run(app, host="0.0.0.0", port=8080) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", help="stdio | http")
    args = parser.parse_args()

    print(f"Starting server with transport: {args.transport}", file=sys.stderr)

    if args.transport == "http":
        run_http()
    else:
        run_stdio()
