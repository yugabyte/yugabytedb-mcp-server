# server.py
import json
import logging
import os
import sys
import argparse
from typing import AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from psycopg_pool import ConnectionPool
import uvicorn
from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
import boto3

from .guardrails import GuardrailConfig
from .auth import create_auth_provider, cognito_password_login, CognitoLoginError
from .tools import summarize_database, run_read_only_query, run_write_query

logger = logging.getLogger("yugabytedb-mcp.server")


@dataclass
class ServerConfig:
    yugabytedb_url: str
    transport: str
    stateless_http: bool
    ssl_root_cert_secret_arn: str | None
    ssl_root_cert_key: str | None
    ssl_root_cert_path: str
    ssl_root_cert_secret_region: str
    max_insert_rows: int
    require_where_on_update: bool
    require_where_on_delete: bool
    auth_provider: str | None


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
        logger.error("Failed to load root cert from Secrets Manager: %s", e)
        raise


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    if not CONFIG.yugabytedb_url:
        logger.critical("YUGABYTEDB_URL is not set")
        sys.exit(1)

    logger.info("Connecting to database...")
    database_url = CONFIG.yugabytedb_url
    cert_path = write_root_cert()
    if cert_path:
        logger.debug("Wrote TLS root cert to %s", cert_path)
        if "sslrootcert" not in database_url:
            database_url += f" sslrootcert={cert_path}"
            logger.debug("Appended sslrootcert to connection string")

    # Connection string can contain a password — log only structural info.
    logger.debug(
        "Opening psycopg ConnectionPool (min_size=1, max_size=5, "
        "check=ConnectionPool.check_connection)"
    )
    pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=5,
        open=True,
        check=ConnectionPool.check_connection,
    )
    logger.debug("ConnectionPool opened successfully")

    guardrail_config = GuardrailConfig(
        max_insert_rows=CONFIG.max_insert_rows,
        require_where_on_update=CONFIG.require_where_on_update,
        require_where_on_delete=CONFIG.require_where_on_delete,
    )
    logger.debug(
        "GuardrailConfig: max_insert_rows=%d, require_where_on_update=%s, "
        "require_where_on_delete=%s",
        guardrail_config.max_insert_rows,
        guardrail_config.require_where_on_update,
        guardrail_config.require_where_on_delete,
    )
    try:
        yield {"pool": pool, "guardrail_config": guardrail_config}
    finally:
        logger.info("Closing database connections")
        pool.close()
        logger.debug("ConnectionPool closed")


def parse_config() -> ServerConfig:
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
        default=os.getenv("YB_SSL_ROOT_CERT_PATH", "/tmp/yb-root.crt"),
        help="Filesystem path where the root certificate will be written (default: `/tmp/yb-root.crt`)",
    )
    parser.add_argument(
        "--yb-aws-ssl-root-cert-secret-region",
        default=os.getenv("YB_AWS_SSL_ROOT_CERT_SECRET_REGION"),
        help="Region of the AWS Secrets Manager secret containing the TLS root certificate",
    )
    parser.add_argument(
        "--max-insert-rows",
        type=int,
        default=int(os.environ.get("YB_MCP_MAX_INSERT_ROWS", "1000")),
        help="Maximum rows allowed per INSERT VALUES statement (env: YB_MCP_MAX_INSERT_ROWS)",
    )
    parser.add_argument(
        "--require-where-on-update",
        action="store_true",
        default=os.environ.get("YB_MCP_REQUIRE_WHERE_ON_UPDATE", "").lower() == "true",
        help="Reject UPDATE without WHERE clause (env: YB_MCP_REQUIRE_WHERE_ON_UPDATE=true)",
    )
    parser.add_argument(
        "--require-where-on-delete",
        action="store_true",
        default=os.environ.get("YB_MCP_REQUIRE_WHERE_ON_DELETE", "").lower() == "true",
        help="Reject DELETE without WHERE clause (env: YB_MCP_REQUIRE_WHERE_ON_DELETE=true)",
    )
    parser.add_argument(
        "--mcp-auth-provider",
        default=os.environ.get("MCP_AUTH_PROVIDER"),
        help="Auth provider for the MCP server: 'cognito' or 'oidc'. Leave unset to disable auth (env: MCP_AUTH_PROVIDER)",
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
        max_insert_rows=args.max_insert_rows,
        require_where_on_update=args.require_where_on_update,
        require_where_on_delete=args.require_where_on_delete,
        auth_provider=args.mcp_auth_provider,
    )


class YugabyteDBMCPServer:
    def __init__(self):
        auth = create_auth_provider(CONFIG.auth_provider)
        self.mcp = FastMCP(
            "yugabytedb-mcp",
            lifespan=app_lifespan,
            auth=auth,
        )

        self._register_tools()

    def _register_tools(self):
        _ro = {"readOnlyHint": True, "destructiveHint": False}
        _dest = {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False}

        self.mcp.tool(
            summarize_database,
            annotations={**_ro, "title": "Summarize database schema and row counts"},
        )
        self.mcp.tool(
            run_read_only_query,
            annotations={**_ro, "title": "Run a read-only SQL query"},
        )
        self.mcp.tool(
            run_write_query,
            annotations={**_dest, "title": "Run a write SQL query (with guardrails)"},
        )

    def run(self, host="0.0.0.0", port=8000):
        if CONFIG.transport == "http":
            self._run_http(host, port)
        else:
            self.mcp.run(transport="stdio")

    def _run_http(self, host, port):
        # Note: json_response is intentionally NOT set here. The MCP spec
        # (Streamable HTTP §2.1 #5) requires the server to be able to return
        # text/event-stream as well as application/json. Forcing json_response
        # silently drops intermediate SSE messages and relaxes Accept header
        # validation.
        mcp_app = self.mcp.http_app(
            path="/mcp",
            stateless_http=CONFIG.stateless_http,
        )

        app = FastAPI(lifespan=mcp_app.lifespan)

        # Middleware stack — request flow is OUTERMOST-first
        # (Starlette/FastAPI wraps each add_middleware around the previous):
        #
        #   request → reject_null_id → WWWAuthScope → OriginValidation → CORS? → MCP app
        #   response ← reject_null_id ← WWWAuthScope ← OriginValidation ← CORS? ← MCP app
        #
        # We add them innermost first.

        # DNS-rebinding defense: reject browser requests with disallowed Origin.
        # Non-browser tools (curl, mcp-remote, AWS CLI) don't send Origin and
        # are unaffected. Configure via MCP_ALLOWED_ORIGINS (comma-separated).
        # Default: same-origin to MCP_BASE_URL.
        allowed = _parse_allowed_origins()
        app.add_middleware(OriginValidationMiddleware, allowed_origins=allowed)
        if allowed:
            logger.info("Origin allowlist: %s", ", ".join(sorted(allowed)))

        # RFC 6750 §3: append `scope=` to WWW-Authenticate on 401 so clients
        # know exactly which scopes to request from the AS. The scope string
        # is the same one configured on the OAuth proxy.
        auth_scope = _resolve_auth_scope()
        if auth_scope:
            app.add_middleware(WWWAuthenticateScopeMiddleware, scope_param=auth_scope)
            logger.info("WWW-Authenticate scope injection enabled (scope=%s)", auth_scope)

        # MCP spec §4.2 + JSON-RPC 2.0 §4: id MUST NOT be null on requests.
        # The MCP SDK (v1.27+) misclassifies these as notifications and
        # returns 202 instead of 400. We intercept at the HTTP layer and
        # return a proper JSON-RPC error response.
        # See: https://github.com/modelcontextprotocol/python-sdk/issues/2057
        @app.middleware("http")
        async def reject_null_id_requests(request, call_next):
            if request.method == "POST" and request.url.path == "/mcp":
                body = await request.body()
                try:
                    data = json.loads(body)
                    if isinstance(data, dict) and "id" in data and data["id"] is None:
                        return JSONResponse(
                            {
                                "jsonrpc": "2.0",
                                "error": {
                                    "code": -32600,
                                    "message": "Invalid Request: request id must not be null",
                                },
                            },
                            status_code=400,
                        )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            return await call_next(request)

        @app.get("/ping")
        async def ping():
            return JSONResponse({"status": "ok"})

        # Convenience endpoint: email + password → Cognito tokens
        # (USER_PASSWORD_AUTH flow). Useful for curl-based smoke tests, CI, and
        # any scripted client that can't go through a browser OAuth flow.
        # Only enabled when MCP_AUTH_PROVIDER=cognito.
        if CONFIG.auth_provider == "cognito":
            from fastapi import Request

            @app.post("/auth/login")
            async def auth_login(request: Request):
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse(
                        {"error": "invalid_request", "detail": "Body must be JSON."},
                        status_code=400,
                    )
                email = body.get("email")
                password = body.get("password")
                if not email or not password:
                    return JSONResponse(
                        {"error": "invalid_request", "detail": "Both `email` and `password` are required."},
                        status_code=400,
                    )
                try:
                    result = cognito_password_login(email, password)
                except CognitoLoginError as e:
                    return JSONResponse(
                        {"error": e.code, "detail": e.detail},
                        status_code=e.status,
                    )
                return JSONResponse({
                    "access_token": result.get("AccessToken"),
                    "id_token": result.get("IdToken"),
                    "refresh_token": result.get("RefreshToken"),
                    "expires_in": result.get("ExpiresIn"),
                    "token_type": result.get("TokenType", "Bearer"),
                })

            logger.info("Enabled /auth/login (Cognito USER_PASSWORD_AUTH)")

        app.mount("/", mcp_app)

        uvicorn.run(app, host=host, port=port)


def _resolve_auth_scope() -> str | None:
    """Return the scope string to inject into WWW-Authenticate, or None.

    For Cognito we mirror what's configured on OIDCProxy (`openid email
    profile`). Returns None when auth is disabled or for providers we
    don't recognize, so the middleware skips registration entirely.
    """
    provider = (CONFIG.auth_provider or "").lower() if CONFIG.auth_provider else ""
    if provider in ("cognito", "oidc"):
        return "openid email profile"
    return None


def _parse_allowed_origins() -> set[str]:
    """Allowed Origin values for the HTTP transport.

    Sourced from MCP_ALLOWED_ORIGINS (comma-separated). Falls back to the
    server's own MCP_BASE_URL when set, then to the empty set (no
    enforcement). When the set is empty, requests with any Origin pass; when
    non-empty, requests with an Origin not in the set are rejected.
    Requests without an Origin header (non-browser clients) always pass.
    """
    raw = os.environ.get("MCP_ALLOWED_ORIGINS", "")
    parts = {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}
    if parts:
        return parts
    base = os.environ.get("MCP_BASE_URL", "").rstrip("/")
    return {base} if base else set()


class OriginValidationMiddleware:
    """DNS-rebinding defense per MCP Transports §Security Warning #1.

    Pure ASGI middleware (not BaseHTTPMiddleware) — doesn't buffer request
    bodies, so it composes cleanly with the SSE streaming path on /mcp.

    Browsers send the Origin header on cross-origin requests; non-browser
    clients (curl, mcp-remote, AWS CLI) typically omit it. We only enforce
    when the allowlist is non-empty AND the request includes an Origin.

    On rejection, returns 403 with a JSON-RPC error body (no `id`, per
    MCP Transports §2.1 #4 — "the HTTP response body MAY comprise a
    JSON-RPC error response that has no `id`").
    """

    def __init__(self, asgi_app, allowed_origins: set[str]):
        self.app = asgi_app
        self.allowed_origins = allowed_origins

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.allowed_origins:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if origin is None or origin.rstrip("/") in self.allowed_origins:
            await self.app(scope, receive, send)
            return

        logger.warning("Rejected request with disallowed Origin: %s", origin)
        body = json.dumps({
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Forbidden: origin not allowed",
            },
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class WWWAuthenticateScopeMiddleware:
    """Inject `scope=` into the WWW-Authenticate header on 401 responses
    (RFC 6750 §3 SHOULD, surfaced as a separate check by mcpdebugger.dev).

    Tells the client exactly which scopes to request from the AS instead
    of leaving it to guess by reading scopes_supported from PRM. FastMCP's
    RequireAuthMiddleware omits this; we patch the response header here.

    Pure ASGI middleware — only touches response headers, never the body,
    so SSE streams are untouched.
    """

    def __init__(self, asgi_app, scope_param: str):
        self.app = asgi_app
        self.scope_param = scope_param

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_scope(message):
            if (
                message["type"] == "http.response.start"
                and message.get("status") == 401
            ):
                headers = list(message.get("headers", []))
                for i, (name, value) in enumerate(headers):
                    if name.lower() == b"www-authenticate":
                        decoded = value.decode()
                        if "scope=" not in decoded:
                            patched = f'{decoded}, scope="{self.scope_param}"'
                            headers[i] = (name, patched.encode())
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_scope)


def _configure_logging() -> None:
    """Set up logging to stderr with a level controlled by YB_LOG_LEVEL."""
    level_name = os.environ.get("YB_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger("yugabytedb-mcp")
    root.setLevel(level)
    root.addHandler(handler)

    if level > logging.DEBUG:
        for noisy in ("urllib3", "botocore", "boto3", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    """Entry point for the `yugabytedb-mcp` console script and `python -m yugabytedb_mcp_server`."""
    _configure_logging()
    logger.info("yugabytedb-mcp-server starting (pid=%d)", os.getpid())
    global CONFIG
    CONFIG = parse_config()
    server = YugabyteDBMCPServer()
    server.run()


if __name__ == "__main__":
    main()
