# server.py
import json
import os
import sys
import psycopg2
from typing import List, Dict, Any
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from typing import AsyncIterator

@dataclass
class AppContext:
    conn: psycopg2.extensions.connection

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    database_url = os.getenv("YUGABYTEDB_URL")
    if not database_url:
        print("YUGABYTEDB_URL is not set")
        sys.exit(1)

    print("Connecting to database...")
    conn = psycopg2.connect(database_url)
    try:
        yield AppContext(conn=conn)
    finally:
        print("Closing database connection...")
        conn.close()


# Create an MCP server
mcp = FastMCP("yugabytedb-mcp", lifespan=app_lifespan)


# Add an addition tool
@mcp.tool()
def summarize_database(ctx:Context) -> List[Dict[str, Any]]:
    """
    Summarize the database: list tables with schema and row counts.
    """
    summary = []
    conn = ctx.request_context.lifespan_context.conn
    with conn.cursor() as cur:
        try:
            # Get list of tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]

            for table in tables:
                # Get schema for each table
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                schema = [{"column_name": col, "data_type": dtype} for col, dtype in cur.fetchall()]

                # Get row count
                cur.execute(f"SELECT COUNT(*) FROM public.\"{table}\"")
                row_count = cur.fetchone()[0]

                summary.append({
                    "table": table,
                    "row_count": row_count,
                    "schema": schema
                })

        except Exception as e:
            summary.append({"error": str(e)})

    return summary

@mcp.tool()
def run_read_only_query(ctx:Context, query: str) -> str:
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
            cur.execute("ROLLBACK")
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing query: {e}"


if __name__ == "__main__":
    mcp.run()