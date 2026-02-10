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
import boto3
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi import Request

import time

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [YugabyteDB MCP Server] {msg}",file=sys.stderr, flush=True)


@dataclass
class AppContext:
    conn: psycopg2.extensions.connection

@dataclass
class ServerConfig:
    yugabytedb_url: str
    transport: str
    stateless_http: bool
    ssl_root_cert_secret_arn: str | None
    ssl_root_cert_key: str | None
    ssl_root_cert_path: str
    ssl_root_cert_secret_region: str
    allowed_hosts: list[str]

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


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    log("App lifespan entered")
    if not CONFIG.yugabytedb_url:
        print("YUGABYTEDB_URL is not set", file=sys.stderr)
        sys.exit(1)

    log("Connecting to database")
    database_url = CONFIG.yugabytedb_url
    cert_path = write_root_cert()
    log(f"SSL cert path = {cert_path}")
    if cert_path and "sslrootcert" not in database_url:
        database_url += f" sslrootcert={cert_path}"
    conn = psycopg2.connect(database_url)
    log("Database connection established")
    try:
        yield AppContext(conn=conn)
    finally:
        log("Closing database connection")
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
    log("run_read_only_query called")
    log(f"Query={query}")
    conn = ctx.request_context.lifespan_context.conn
    log(f"Connection open={not conn.closed}")
    with conn.cursor() as cur:
        try:
            log("Starting transaction")
            cur.execute("BEGIN READ ONLY")
            log("Executing query")
            cur.execute(query)
            rows = cur.fetchall()
            log(f"Fetched {len(rows)} rows")
            column_names = [desc[0] for desc in cur.description]
            result = [dict(zip(column_names, row)) for row in rows]
            log("Query successful")
            return json.dumps(result, indent=2)
        except Exception as e:
            log(f"Error executing query: {repr(e)}")
            return f"Error executing query: {e}"
        finally:
            try:
                log("Rolling back transaction")
                cur.execute("ROLLBACK")
            except Exception as e:
                log(f"Couldn't ROLLBACK transaction: {repr(e)}")
                return f"Couldn't ROLLBACK transaction: {e}"


def parse_config() -> argparse.Namespace:

    log("Parsing configuration")
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
    "--yb-allowed-hosts",
    default=os.getenv("YB_ALLOWED_HOSTS", ""),
    help="Comma-separated list of allowed Host headers (public HTTP mode)",
)

    args = parser.parse_args()
    is_bedrock = os.getenv("ALLOW_ALL_HOST_HEADERS", "").lower() == "true"
    log(f"ALLOW_ALL_HOST_HEADERS={is_bedrock}")

    if is_bedrock:
        allowed_hosts = ["*"]
    else:
        allowed_hosts = [
            h.strip()
            for h in args.yb_allowed_hosts.split(",")
            if h.strip()
        ] or ["localhost"]

    log(f"Resolved allowed_hosts={allowed_hosts}")
    log(f"Transport={args.transport}, Stateless={args.stateless_http}")

    return ServerConfig(
        yugabytedb_url=args.yugabytedb_url,
        transport=args.transport,
        stateless_http=args.stateless_http,
        ssl_root_cert_secret_arn=args.yb_aws_ssl_root_cert_secret_arn,
        ssl_root_cert_key=args.yb_aws_ssl_root_cert_key,
        ssl_root_cert_path=args.yb_ssl_root_cert_path,
        ssl_root_cert_secret_region=args.yb_aws_ssl_root_cert_secret_region,
        allowed_hosts=allowed_hosts,
    )


class YugabyteDBMCPServer:
    def __init__(self):

        log("Initializing YugabyteDBMCPServer")
        log(f"FastMCP stateless_http={CONFIG.stateless_http}")

        self.mcp = FastMCP(
            "yugabytedb-mcp",
            lifespan=app_lifespan,
            json_response=True,
            stateless_http=CONFIG.stateless_http,
        )

        self._register_tools()

    def _register_tools(self):
        log("Registering MCP tools")
        self.mcp.add_tool(summarize_database)
        self.mcp.add_tool(run_read_only_query)

    def run(self, host="0.0.0.0", port=8000):

        log(f"Server run() called with transport={CONFIG.transport}")

        if CONFIG.transport == "http":
            self._run_http(host, port)
        else:
            self.mcp.run(transport="stdio")

    def _run_http(self, host, port):

        log("Starting HTTP server")
        log(f"Allowed hosts = {CONFIG.allowed_hosts}")
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            log("FastAPI lifespan starting")
            async with AsyncExitStack() as stack:
                log("Entering MCP session manager")
                await stack.enter_async_context(self.mcp.session_manager.run())
                yield
            log("FastAPI lifespan ending")

        app = FastAPI(lifespan=lifespan)

        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=CONFIG.allowed_hosts
        )

        log("TrustedHostMiddleware installed")

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            log(
                f"Incoming request: method={request.method} "
                f"path={request.url.path} "
                f"host={request.headers.get('host')} "
                f"user-agent={request.headers.get('user-agent')}"
                )
            response = await call_next(request)
            log(f"Response status={response.status_code}")
            return response

        @app.get("/ping")
        async def ping():
            return JSONResponse({"status": "ok"})
        
        log("Mounting MCP Streamable HTTP app")
        app.mount("/", self.mcp.streamable_http_app())

        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    CONFIG = parse_config()
    server = YugabyteDBMCPServer()
    server.run()
