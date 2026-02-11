# server.py
import asyncio
import json
import logging
import os
import sys
import argparse
import threading
import time
import psycopg2
from typing import List, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager, AsyncExitStack
from mcp.server.fastmcp import FastMCP, Context
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
import boto3
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.http import set_http_request

logger = logging.getLogger("yugabytedb-mcp")

# Session-scoped DB connections: one connection per MCP session, keyed by session_id.
# Credentials come from HTTP headers on first use; the same connection is reused for the session.
#
# Isolation (multi-tenant safety): Each client gets a unique mcp-session-id from the server.
# Connections are keyed only by that id. Credentials are never stored; they are read from
# the current request's headers when creating a connection. Lookup always uses the current
# request's mcp-session-id. So User A's credentials and connection are never visible to
# User B, and vice versa.
_session_connections: Dict[str, psycopg2.extensions.connection] = {}
_session_last_used: Dict[str, float] = {}  # session_id -> monotonic time of last use
_session_connections_lock = threading.Lock()

@dataclass
class ServerConfig:
    yugabytedb_url: str
    transport: str
    stateless_http: bool
    ssl_root_cert_secret_arn: str | None
    ssl_root_cert_key: str | None
    ssl_root_cert_path: str
    ssl_root_cert_secret_region: str
    session_idle_timeout_seconds: float = 0.0  # 0 = disabled

def normalize_pem(pem: str) -> str:
    # Remove surrounding spaces
    pem = pem.strip()

    # Fix cases where newlines were replaced by spaces
    pem = pem.replace("-----BEGIN CERTIFICATE----- ", "-----BEGIN CERTIFICATE-----\n")
    pem = pem.replace(" -----END CERTIFICATE-----", "\n-----END CERTIFICATE-----")

    # Also fix intermediate blocks
    pem = pem.replace("-----END CERTIFICATE-----  -----BEGIN CERTIFICATE-----",
                      "-----END CERTIFICATE-----\n\n-----BEGIN CERTIFICATE-----")

    return pem + "\n"


def write_root_cert():
    if not CONFIG.ssl_root_cert_secret_arn:
        return None

    try:
        sm = boto3.client("secretsmanager", region_name=CONFIG.ssl_root_cert_secret_region)
        resp = sm.get_secret_value(SecretId=CONFIG.ssl_root_cert_secret_arn)
        secret_string = resp["SecretString"]

        # If raw PEM, just use it
        if "BEGIN CERTIFICATE" in secret_string and not secret_string.strip().startswith("{"):
            pem = secret_string
        else:
            data = json.loads(secret_string)

            if CONFIG.ssl_root_cert_key:
                if CONFIG.ssl_root_cert_key not in data:
                    raise RuntimeError(f"Certificate key '{CONFIG.ssl_root_cert_key}' not found in secret")
                pem = data[CONFIG.ssl_root_cert_key]
            else:
                # Backward-compatible: allow exactly one entry
                if len(data) != 1:
                    raise RuntimeError(
                        "Multiple certificates found in secret; set YB_AWS_SSL_ROOT_CERT_KEY to select one"
                    )
                pem = next(iter(data.values()))
        
        pem = normalize_pem(pem)
        with open(CONFIG.ssl_root_cert_path, "w") as f:
            f.write(pem.strip() + "\n")

        return CONFIG.ssl_root_cert_path

    except Exception as e:
        print(f"Failed to load root cert from Secrets Manager: {e}", file=sys.stderr)
        raise



def _conn_from_request(request: Request) -> psycopg2.extensions.connection:
    """Build a new DB connection from an HTTP request's headers (call from middleware)."""
    headers = request.headers
    db_user = headers.get("x-db-user")
    db_password = headers.get("x-db-password")
    db_host = headers.get("x-db-host", "localhost")
    db_port = headers.get("x-db-port", "5433")
    db_name = headers.get("x-db-name", "yugabyte")

    if not db_user or not db_password:
        raise RuntimeError(
            "Missing DB credentials in request headers. "
            "Send x-db-user and x-db-password (and optionally x-db-host, x-db-port, x-db-name)."
        )

    conn_str = (
        f"dbname={db_name} "
        f"user={db_user} "
        f"password={db_password} "
        f"host={db_host} "
        f"port={db_port}"
    )
    return psycopg2.connect(conn_str)


def ensure_session_connection(request: Request) -> None:
    """
    Ensure a DB connection exists for this request's session. Call from middleware
    so the connection is created in the request thread (tools may run in another thread).
    """
    session_id = request.headers.get("mcp-session-id")
    if not session_id:
        return
    if not request.headers.get("x-db-user") or not request.headers.get("x-db-password"):
        return
    with _session_connections_lock:
        if session_id in _session_connections:
            _session_last_used[session_id] = time.monotonic()
            return
        try:
            conn = _conn_from_request(request)
            _session_connections[session_id] = conn
            _session_last_used[session_id] = time.monotonic()
            logger.info("Opened DB connection for session %s...", session_id[:8])
        except Exception as e:
            logger.exception("Failed to open DB connection for session %s: %s", session_id[:8], e)
            raise


def close_session_connection(session_id: str) -> None:
    """
    Close and remove the DB connection for the given session. Call when a session
    ends (e.g. client sends DELETE) so connections do not leak.
    """
    with _session_connections_lock:
        conn = _session_connections.pop(session_id, None)
        _session_last_used.pop(session_id, None)
    if conn is not None:
        try:
            conn.close()
            logger.info("Closed DB connection for session %s...", session_id[:8])
        except Exception as e:
            logger.exception("Error closing DB connection for session %s: %s", session_id[:8], e)


def close_idle_sessions(idle_timeout_seconds: float) -> int:
    """
    Close DB connections for sessions that have been idle longer than idle_timeout_seconds.
    Returns the number of sessions closed.
    """
    if idle_timeout_seconds <= 0:
        return 0
    now = time.monotonic()
    threshold = now - idle_timeout_seconds
    to_close: List[tuple[str, psycopg2.extensions.connection]] = []
    with _session_connections_lock:
        for session_id in list(_session_last_used.keys()):
            if _session_last_used[session_id] < threshold:
                conn = _session_connections.pop(session_id, None)
                _session_last_used.pop(session_id, None)
                if conn is not None:
                    to_close.append((session_id, conn))
    closed = 0
    for session_id, conn in to_close:
        try:
            conn.close()
            closed += 1
            logger.info("Closed idle DB connection for session %s... (idle > %.0fs)", session_id[:8], idle_timeout_seconds)
        except Exception as e:
            logger.exception("Error closing idle connection for session %s: %s", session_id[:8], e)
    return closed


def get_connection(ctx: Context) -> psycopg2.extensions.connection:
    """
    Return the session-scoped DB connection. The connection must already exist for this
    session (created by middleware from request headers). One connection per session.
    Uses the mcp-session-id HTTP header (MCP SDK Context does not expose session_id).
    """
    try:
        request = get_http_request()
        session_id = request.headers.get("mcp-session-id")
    except RuntimeError:
        logger.warning("get_connection: no HTTP request in context (e.g. tool ran in another thread)")
        raise RuntimeError(
            "No database connection: request context not available. "
            "Ensure the server is run with middleware that sets the HTTP request."
        ) from None
    if not session_id:
        raise RuntimeError(
            "No database connection: missing mcp-session-id header. "
            "Ensure your MCP client sends the session id on every request."
        )
    with _session_connections_lock:
        conn = _session_connections.get(session_id)
        known_ids = list(_session_connections.keys())
        if conn is not None:
            _session_last_used[session_id] = time.monotonic()
    if conn is None:
        logger.warning(
            "get_connection: no connection for session_id=%r; known sessions: %s",
            session_id,
            [s[:12] + "..." for s in known_ids],
        )
        raise RuntimeError(
            "No database connection for this session. Ensure you send x-db-user and "
            "x-db-password (and optionally x-db-host, x-db-port, x-db-name) on every "
            "request, including the initial connection."
        )
    return conn


def summarize_database(ctx: Context, schema: str = "public") -> List[Dict[str, Any]]:
    """
    Summarize the database: list tables with schema and row counts.
    """
    try:
        logger.info("summarize_database called schema=%s", schema)
        summary = []
        conn = get_connection(ctx)
        logger.info("get_connection succeeded")
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
                logger.exception("summarize_database SQL error: %s", e)
                summary.append({"error": str(e)})

        logger.info("summarize_database returning %d items", len(summary))
        return summary
    except Exception as e:
        logger.exception("summarize_database failed: %s", e)
        raise


def run_read_only_query(ctx: Context, query: str) -> str:
    """
    Run a read-only SQL query and return the results as JSON.
    """
    conn = get_connection(ctx)
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
    parser.add_argument(
        "--yb-aws-ssl-root-cert-secret-arn", 
        default=os.getenv("YB_AWS_SSL_ROOT_CERT_SECRET_ARN"),
        help="ARN of the AWS Secrets Manager secret containing the TLS root certificate",
    )
    parser.add_argument(
        "--yb-aws-ssl-root-cert-key", 
        default=os.getenv("YB_AWS_SSL_ROOT_CERT_KEY"),
        help="Key inside the secret JSON that selects which certificate to use",
    )
    parser.add_argument(
        "--yb-ssl-root-cert-path",
          default=os.getenv("YB_SSL_ROOT_CERT_PATH","/tmp/yb-root.crt"),
          help="Filesystem path where the root certificate will be written (default: `/tmp/yb-root.crt`)"
    )
    parser.add_argument(
        "--yb-aws-ssl-root-cert-secret-region",
          default=os.getenv("YB_AWS_SSL_ROOT_CERT_SECRET_REGION"),
          help="Region of the AWS Secrets Manager secret containing the TLS root certificate",
    )
    parser.add_argument(
        "--session-idle-timeout",
        type=float,
        default=float(os.environ.get("YB_SESSION_IDLE_TIMEOUT", "0")),
        help="Close DB connections idle longer than this many seconds (0=disabled). Default: 0 (env: YB_SESSION_IDLE_TIMEOUT)",
    )

    args = parser.parse_args()
    return ServerConfig(
        yugabytedb_url=args.yugabytedb_url,
        transport=args.transport,
        stateless_http=args.stateless_http,
        ssl_root_cert_secret_arn=args.yb_aws_ssl_root_cert_secret_arn,
        ssl_root_cert_key=args.yb_aws_ssl_root_cert_key,
        ssl_root_cert_path=args.yb_ssl_root_cert_path,
        ssl_root_cert_secret_region=args.yb_aws_ssl_root_cert_secret_region,
        session_idle_timeout_seconds=args.session_idle_timeout,
    )


class YugabyteDBMCPServer:
    def __init__(self):

        print("Initializing YugabyteDB MCP server")

        # stateless_http=False so one MCP session keeps one DB connection for the whole session
        self.mcp = FastMCP(
            "yugabytedb-mcp",
            json_response=True,
            stateless_http=CONFIG.stateless_http,
        )
        print("Server Initialized. Registering tools")
        print("HTTP mode: stateful (one DB connection per session, credentials via headers)")

        self._register_tools()

    def _register_tools(self):
        self.mcp.add_tool(summarize_database)
        self.mcp.add_tool(run_read_only_query)

    def run(self, host="0.0.0.0", port=8000):
        if CONFIG.transport == "http":
            self._run_http(host, port)
        else:
            self.mcp.run(transport="stdio")

    def _run_http(self, host, port):
        async def idle_cleanup_loop():
            """Background task: close DB connections idle longer than session_idle_timeout_seconds."""
            timeout = CONFIG.session_idle_timeout_seconds
            if timeout <= 0:
                return
            while True:
                await asyncio.sleep(60)
                n = close_idle_sessions(timeout)
                if n > 0:
                    logger.info("Closed %d idle session(s)", n)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(self.mcp.session_manager.run())
                cleanup_task = None
                if CONFIG.session_idle_timeout_seconds > 0:
                    cleanup_task = asyncio.create_task(idle_cleanup_loop())
                    logger.info("Session idle timeout: %.0fs", CONFIG.session_idle_timeout_seconds)
                try:
                    yield
                finally:
                    if cleanup_task is not None:
                        cleanup_task.cancel()
                        try:
                            await cleanup_task
                        except asyncio.CancelledError:
                            pass

        app = FastAPI(lifespan=lifespan)

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            print("\n==============================")
            print("Incoming HTTP Request")
            print("Method:", request.method)
            print("URL:", request.url)
            print("Headers:")
            for k, v in request.headers.items():
                print(f"  {k}: {v}")

            # Do not read request.body() here: it consumes the stream and breaks the MCP app.
            print("==============================\n")

            # Create/reuse session DB connection from headers (in request thread; tools may run in another thread)
            try:
                ensure_session_connection(request)
            except Exception as e:
                logger.exception("Session connection error: %s", e)
            # Set request in FastMCP context for any code that uses get_http_headers()
            with set_http_request(request):
                response = await call_next(request)

            # When client closes the session (DELETE), close our DB connection for that session
            if request.method == "DELETE":
                session_id = request.headers.get("mcp-session-id")
                if session_id:
                    close_session_connection(session_id)

            print("Response status:", response.status_code)
            print("==============================\n")

            return response

        @app.get("/ping")
        async def ping():
            return JSONResponse({"status": "ok"})

        app.mount("/", self.mcp.streamable_http_app())

        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    CONFIG = parse_config()
    server = YugabyteDBMCPServer()
    server.run()
