# server.py
import json
import os
import sys
import argparse
import psycopg2
from typing import AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager, AsyncExitStack
from mcp.server.fastmcp import FastMCP
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse
import boto3

from tools import (
    summarize_database,
    run_read_only_query,
    run_write_query,
    # add_document_source,
    # init_vector_index,
    # trigger_index_build,
    check_index_status,
    add_source_to_index,
    trigger_knowledge_base_build,
)

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
    if not CONFIG.yugabytedb_url:
        print("YUGABYTEDB_URL is not set", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...", file=sys.stderr)
    database_url = CONFIG.yugabytedb_url
    cert_path = write_root_cert()
    if cert_path and "sslrootcert" not in database_url:
        database_url += f" sslrootcert={cert_path}"
    conn = psycopg2.connect(database_url)
    try:
        yield AppContext(conn=conn)
    finally:
        print("Closing database connection...", file=sys.stderr)
        conn.close()


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

    args = parser.parse_args()
    return ServerConfig(
        yugabytedb_url=args.yugabytedb_url,
        transport=args.transport,
        stateless_http=args.stateless_http,
        ssl_root_cert_secret_arn=args.yb_aws_ssl_root_cert_secret_arn,
        ssl_root_cert_key=args.yb_aws_ssl_root_cert_key,
        ssl_root_cert_path=args.yb_ssl_root_cert_path,
        ssl_root_cert_secret_region=args.yb_aws_ssl_root_cert_secret_region,
    )


class YugabyteDBMCPServer:
    def __init__(self):

        self.mcp = FastMCP(
            "yugabytedb-mcp",
            lifespan=app_lifespan,
            json_response=True,
            stateless_http=CONFIG.stateless_http,
        )

        self._register_tools()

    def _register_tools(self):
        self.mcp.add_tool(summarize_database)
        self.mcp.add_tool(run_read_only_query)
        self.mcp.add_tool(run_write_query)
        # self.mcp.add_tool(add_document_source)
        # self.mcp.add_tool(init_vector_index)
        # self.mcp.add_tool(trigger_index_build)
        self.mcp.add_tool(check_index_status)
        self.mcp.add_tool(add_source_to_index)
        self.mcp.add_tool(trigger_knowledge_base_build)

    def run(self, host="0.0.0.0", port=8000):
        if CONFIG.transport == "http":
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
    CONFIG = parse_config()
    server = YugabyteDBMCPServer()
    server.run()
