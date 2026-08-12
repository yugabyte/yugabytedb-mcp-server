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
from .tools import (
    summarize_database,
    run_read_only_query,
    run_write_query,
    _load_identity_map,
)

logger = logging.getLogger("yugabytedb-mcp.server")


def _pool_reset(conn) -> None:
    """Reset a connection's session state before it returns to the pool.

    Two statements run in sequence:

    1. ``DISCARD ALL`` — Postgres's canonical "scrub session" statement.
       Clears the current role (superset of RESET ROLE — closes DB-22133),
       prepared statements + cached plans (closes the pool-slot poisoning
       half of DB-22202), session GUCs, temp tables, and sequences.

    2. ``SELECT pg_advisory_unlock_all()`` — YugabyteDB's ``DISCARD ALL``
       does NOT release advisory locks, which is a deviation from vanilla
       Postgres 15 semantics. Empirically verified against local YB
       (2025.2.0): a lock taken on one checkout survives ``DISCARD ALL``
       and is visible to other sessions via ``pg_locks``. Explicitly
       releasing here closes the advisory-lock half of DB-22202 (the
       cross-user DoS repro Vishal filed).

    Neither statement can run inside a transaction block. psycopg's
    default is ``autocommit=False``, so a bare ``conn.execute(...)``
    would open an implicit transaction and immediately fail. Flip the
    connection to autocommit for the reset, then restore. Safe here
    because psycopg-pool already rolls back any pending transaction
    before calling the reset callback.

    Failures are logged but not raised — a failing reset is caught by
    ConnectionPool's health check (``check_connection``) on the next
    checkout, so the connection either passes the reset next time or
    gets replaced.
    """
    try:
        conn.set_autocommit(True)
        try:
            conn.execute("DISCARD ALL")
            conn.execute("SELECT pg_advisory_unlock_all()")
        finally:
            conn.set_autocommit(False)
    except Exception as e:
        logger.warning(
            "Pool reset (DISCARD ALL) failed on connection return: %s. "
            "The health check on next checkout will replace the connection "
            "if it's still broken.",
            e,
        )


def _positive_int(s: str) -> int:
    """argparse `type=` helper: parse `s` as int; raise ArgumentTypeError
    on non-int or `< 1`. Used for the DB-22159 resource-limit env vars so
    a typo (`YB_MCP_POOL_MAX_SIZE=abc` or `=0`) fails startup with a
    clear message instead of a raw ValueError traceback."""
    try:
        v = int(s)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {s!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def _tcp_port(s: str) -> int:
    """argparse `type=` helper for MCP_PORT. Requires 1 <= port <= 65535
    so a typo (`MCP_PORT=65536`) fails at parse_config with a clear
    argparse error instead of an uglier uvicorn socket-bind traceback
    at startup."""
    try:
        v = int(s)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"must be an integer 1–65535, got {s!r}")
    if not (1 <= v <= 65535):
        raise argparse.ArgumentTypeError(f"must be 1–65535, got {v}")
    return v


@dataclass
class ServerConfig:
    yugabytedb_url: str
    transport: str
    host: str
    port: int
    stateless_http: bool
    ssl_root_cert_secret_arn: str | None
    ssl_root_cert_key: str | None
    ssl_root_cert_path: str
    ssl_root_cert_secret_region: str
    require_where_on_update: bool
    require_where_on_delete: bool
    auth_provider: str | None
    enable_write_query: bool
    identity_claim: str
    # PR #10 (OIDC v2) identity mapping
    identity_map_path: str | None
    identity_map_name: str
    # DB-22159 resource limits — all defaults documented alongside the
    # argparse definitions in parse_config().
    pool_min_size: int
    pool_max_size: int
    statement_timeout_ms: int
    max_result_rows: int
    max_result_bytes: int
    max_query_len: int


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

    # DB-22159 round-2: validate pool sizing at startup so a misconfig
    # (min > max) fails with a clean error instead of a raw psycopg
    # traceback at pool.open time.
    if CONFIG.pool_min_size > CONFIG.pool_max_size:
        raise ValueError(
            f"YB_MCP_POOL_MIN_SIZE ({CONFIG.pool_min_size}) exceeds "
            f"YB_MCP_POOL_MAX_SIZE ({CONFIG.pool_max_size}). Lower the "
            f"minimum or raise the maximum."
        )

    logger.info("Connecting to database...")
    database_url = CONFIG.yugabytedb_url
    cert_path = write_root_cert()
    if cert_path:
        logger.debug("Wrote TLS root cert to %s", cert_path)
        if "sslrootcert" not in database_url:
            database_url += f" sslrootcert={cert_path}"
            logger.debug("Appended sslrootcert to connection string")

    # DB-22159 round-2: bound the TCP connect attempt too. libpq's default
    # is unlimited; without a cap, a network partition to the DB during
    # pool warm-up hangs startup indefinitely (or the pool acquire path
    # blocks the tool call for hundreds of seconds). Only add when the
    # operator hasn't set one explicitly. libpq accepts both keyword form
    # (`host=… connect_timeout=10`) and URI form
    # (`postgresql://…?connect_timeout=10`) — space-appending to a URI
    # mangles it into `?sslmode=… connect_timeout=10` which psycopg
    # rejects. Detect the form and use the right separator.
    if "connect_timeout" not in database_url.lower():
        if database_url.startswith(("postgres://", "postgresql://")):
            sep = "&" if "?" in database_url else "?"
            database_url = f"{database_url}{sep}connect_timeout=10"
        else:
            database_url = f"{database_url} connect_timeout=10"

    # Connection string can contain a password — log only structural info.
    logger.debug(
        "Opening psycopg ConnectionPool (min_size=%d, max_size=%d, "
        "check=ConnectionPool.check_connection, reset=DISCARD ALL)",
        CONFIG.pool_min_size, CONFIG.pool_max_size,
    )
    pool = ConnectionPool(
        conninfo=database_url,
        min_size=CONFIG.pool_min_size,
        max_size=CONFIG.pool_max_size,
        open=True,
        check=ConnectionPool.check_connection,
        # DB-22133 / DB-22202: reset connections on return to the pool so
        # session-level state from one user's tool call doesn't bleed to
        # the next. DISCARD ALL is a superset of RESET ROLE — it also
        # clears advisory locks (advisory-lock cross-user DoS in DB-22202),
        # prepared-statement plans (pool-slot poisoning in DB-22202),
        # session GUCs, temp tables, and cached plans. Runs after every
        # `with pool.connection() as conn:` block, so any state the tool
        # accidentally leaves behind is scrubbed before the next checkout.
        reset=_pool_reset,
    )
    logger.debug("ConnectionPool opened successfully")

    guardrail_config = GuardrailConfig(
        require_where_on_update=CONFIG.require_where_on_update,
        require_where_on_delete=CONFIG.require_where_on_delete,
    )
    logger.debug(
        "GuardrailConfig: require_where_on_update=%s, "
        "require_where_on_delete=%s",
        guardrail_config.require_where_on_update,
        guardrail_config.require_where_on_delete,
    )
    # Load the identity map file (pg_ident.conf-style) if configured. Parsed
    # once at startup; a malformed file raises ValueError which we let
    # propagate — the server refuses to start rather than silently accepting
    # a typo'd map that could widen access.
    identity_map = None
    if CONFIG.identity_map_path:
        logger.info(
            "Loading identity map from %s (map_name=%s)",
            CONFIG.identity_map_path, CONFIG.identity_map_name,
        )
        identity_map = _load_identity_map(CONFIG.identity_map_path)
        # identity_map is now Dict[map_name, List[MapEntry]]; total = sum of
        # the inner lists, and the count under the configured name is a
        # cheap dict lookup.
        _total = sum(len(v) for v in identity_map.values())
        _under_name = len(identity_map.get(CONFIG.identity_map_name, []))
        logger.info(
            "Identity map loaded: %d entries (%d under map_name=%s)",
            _total, _under_name, CONFIG.identity_map_name,
        )

    logger.info(
        "Resource limits: pool=%d-%d, statement_timeout=%dms, "
        "max_result_rows=%d, max_query_len=%d bytes",
        CONFIG.pool_min_size, CONFIG.pool_max_size,
        CONFIG.statement_timeout_ms, CONFIG.max_result_rows,
        CONFIG.max_query_len,
    )
    if CONFIG.auth_provider:
        logger.info(
            "Per-user SET ROLE enabled (claim=%s, map=%s). "
            "The pool user must have GRANT to the target roles it needs to "
            "SET ROLE to; superuser works but is not required if an identity "
            "map is set.",
            CONFIG.identity_claim,
            CONFIG.identity_map_path or "<none>",
        )
        # Fail-closed at startup for the "list-valued claim + no map" combo.
        # A list claim (e.g. `cognito:groups`, `realm_access.roles`) yields
        # raw role names from the IdP; without a map file to translate them
        # into a controlled PG role set, every group name would be used
        # verbatim as a SET ROLE target — no allowlist. The PR's own
        # framing says "the map file IS the allowlist," so refuse to start
        # in this configuration rather than silently granting whatever the
        # IdP happens to emit.
        _list_claim_paths = ("cognito:groups", "realm_access.roles", "groups")
        looks_list_valued = (
            CONFIG.identity_claim in _list_claim_paths
            or "." in CONFIG.identity_claim   # dotted paths usually target lists
        )
        if identity_map is None and looks_list_valued:
            raise RuntimeError(
                f"YB_MCP_IDENTITY_CLAIM={CONFIG.identity_claim!r} typically "
                f"resolves to a LIST of roles from the IdP, but "
                f"YB_MCP_IDENTITY_MAP is unset. Without a map, every raw role "
                f"name from the token would be a candidate SET ROLE target "
                f"— that removes the allowlist boundary the map is designed "
                f"to enforce. Configure YB_MCP_IDENTITY_MAP to translate the "
                f"IdP's role names to a fixed PG role set, or switch "
                f"YB_MCP_IDENTITY_CLAIM to a scalar claim like `email` or "
                f"`sub`."
            )
        if identity_map is None:
            logger.warning(
                "OIDC identity mapping has no map file configured "
                "(YB_MCP_IDENTITY_MAP unset). The claim value is used as the "
                "DB role name directly, which requires the pool user to be a "
                "superuser or have GRANT on every possible role. Configure "
                "YB_MCP_IDENTITY_MAP to constrain the set of reachable roles."
            )

    try:
        yield {
            "pool": pool,
            "guardrail_config": guardrail_config,
            "identity_claim": CONFIG.identity_claim,
            "identity_map": identity_map,
            "identity_map_name": CONFIG.identity_map_name,
            # DB-22159 resource limits — read by tools.py at each call.
            "statement_timeout_ms": CONFIG.statement_timeout_ms,
            "max_result_rows": CONFIG.max_result_rows,
            "max_result_bytes": CONFIG.max_result_bytes,
            "max_query_len": CONFIG.max_query_len,
        }
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
        "--host",
        default=os.environ.get("MCP_HOST", "127.0.0.1"),
        help="Bind host for HTTP transport. Default 127.0.0.1 (loopback). "
             "Set to 0.0.0.0 to expose on all interfaces; auth is required "
             "in that case (see MCP_AUTH_PROVIDER). "
             "(env: MCP_HOST, default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=_tcp_port,
        default=os.environ.get("MCP_PORT", "8000"),
        help="Bind port for HTTP transport, 1–65535 "
             "(env: MCP_PORT, default: 8000).",
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
    # DB-22131 round 2: YB_MCP_MAX_INSERT_ROWS has been removed. Every
    # write goes through SET LOCAL statement_timeout in run_write_query
    # so a runaway INSERT — VALUES, SELECT, whatever shape — is bounded
    # by the timeout. A static row cap on top is redundant. Warn (not
    # fail) so existing deployments that still set the env are noisy
    # about the removal without blocking startup.
    if os.environ.get("YB_MCP_MAX_INSERT_ROWS") is not None:
        logger.warning(
            "YB_MCP_MAX_INSERT_ROWS has been removed and is ignored. "
            "INSERT statements are bounded by YB_MCP_STATEMENT_TIMEOUT_MS "
            "(SET LOCAL statement_timeout applied to every write)."
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
        "--enable-write-query",
        action="store_true",
        default=os.environ.get("YB_MCP_ENABLE_WRITE_QUERY", "").lower() == "true",
        help="Enable the run_write_query tool (disabled by default) (env: YB_MCP_ENABLE_WRITE_QUERY=true)",
    )
    parser.add_argument(
        "--mcp-auth-provider",
        default=os.environ.get("MCP_AUTH_PROVIDER"),
        help="Auth provider for the MCP server: 'cognito' or 'oidc'. Leave unset to disable auth (env: MCP_AUTH_PROVIDER)",
    )
    # DB-22174: strip_domain has been removed. If an existing deployment
    # still has YB_MCP_IDENTITY_TRANSFORM set (to strip_domain or
    # anything else), fail startup with a clear migration message rather
    # than silently ignoring the config.
    _removed_transform = os.environ.get("YB_MCP_IDENTITY_TRANSFORM")
    if _removed_transform:
        raise SystemExit(
            "YB_MCP_IDENTITY_TRANSFORM has been removed. The only prior "
            "value, `strip_domain`, silently collapsed users from "
            f"different email domains (got {_removed_transform!r}). Migrate "
            "to YB_MCP_IDENTITY_MAP with a pg_ident.conf-style file — see "
            "OIDC.md."
        )

    # DB-22136 (paired with the token_use enforcement flip in auth.py): the
    # default identity_claim switches to ``sub`` in secure mode because
    # ``email`` is not present in Cognito access tokens. Legacy operators
    # who still expect ``email`` (and therefore an id token) opt in via
    # ``YB_MCP_LEGACY_ACCEPT_ID_TOKENS=true``, which also flips the
    # ``require_access_token`` default back to False.
    _legacy_auth = os.environ.get(
        "YB_MCP_LEGACY_ACCEPT_ID_TOKENS", ""
    ).lower() == "true"
    _default_identity_claim = "email" if _legacy_auth else "sub"
    parser.add_argument(
        "--identity-claim",
        default=os.environ.get("YB_MCP_IDENTITY_CLAIM", _default_identity_claim),
        help="JWT claim to use as the DB role identifier "
             "(env: YB_MCP_IDENTITY_CLAIM, default: 'sub' — or 'email' if "
             "YB_MCP_LEGACY_ACCEPT_ID_TOKENS=true).",
    )
    parser.add_argument(
        "--identity-map",
        default=os.environ.get("YB_MCP_IDENTITY_MAP"),
        help="Path to a pg_ident.conf-style identity map file. Each line is "
             "'<map_name> <system_value> <db_role>' — system_value may be a "
             "literal string or /regex/ (leading slash triggers regex; role "
             "may reference capture groups via \\1). When set, replaces the "
             "identity-transform path. (env: YB_MCP_IDENTITY_MAP)",
    )
    parser.add_argument(
        "--identity-map-name",
        default=os.environ.get("YB_MCP_IDENTITY_MAP_NAME", "default"),
        help="Which named map inside the identity map file to apply. "
             "(env: YB_MCP_IDENTITY_MAP_NAME, default: default)",
    )
    # DB-22159 resource limits — bound the blast radius of a slow query
    # or a huge result set. All parsed with _positive_int so a typo in
    # the env fails startup with a clean argparse error.
    parser.add_argument(
        "--pool-min-size",
        type=_positive_int,
        default=os.environ.get("YB_MCP_POOL_MIN_SIZE", "1"),
        help="Minimum connections held by the pool "
             "(env: YB_MCP_POOL_MIN_SIZE, default: 1).",
    )
    parser.add_argument(
        "--pool-max-size",
        type=_positive_int,
        default=os.environ.get("YB_MCP_POOL_MAX_SIZE", "5"),
        help="Maximum connections held by the pool. Raise this if you "
             "expect concurrent tool calls; a low value is the DoS surface "
             "(five long-running queries block everyone else) "
             "(env: YB_MCP_POOL_MAX_SIZE, default: 5).",
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=_positive_int,
        default=os.environ.get("YB_MCP_STATEMENT_TIMEOUT_MS", "30000"),
        help="Per-tool-call statement_timeout in milliseconds. Set "
             "via `SET LOCAL statement_timeout` inside each transaction "
             "so a runaway query (pg_sleep, cartesian, heavy scan) can't "
             "hold a pool connection indefinitely "
             "(env: YB_MCP_STATEMENT_TIMEOUT_MS, default: 30000).",
    )
    parser.add_argument(
        "--max-result-rows",
        type=_positive_int,
        default=os.environ.get("YB_MCP_MAX_RESULT_ROWS", "10000"),
        help="Cap the number of rows returned by run_read_only_query. "
             "Prevents an OOM crash from `SELECT repeat('x', N) FROM "
             "generate_series(1, N)`-style queries. Truncated responses "
             "carry a `truncated: true` marker "
             "(env: YB_MCP_MAX_RESULT_ROWS, default: 10000).",
    )
    parser.add_argument(
        "--max-result-bytes",
        type=_positive_int,
        default=os.environ.get("YB_MCP_MAX_RESULT_BYTES", str(50 * 1024 * 1024)),
        help="Cap the total byte size of a result set. Enforced while "
             "streaming rows so a query like `SELECT repeat('x', 1_000_000) "
             "FROM generate_series(1, 100)` can't OOM the process even with "
             "max_result_rows large. Truncated responses carry a "
             "`truncated: true` marker "
             "(env: YB_MCP_MAX_RESULT_BYTES, default: 50 MiB).",
    )
    parser.add_argument(
        "--max-query-len",
        type=_positive_int,
        default=os.environ.get("YB_MCP_MAX_QUERY_LEN", "100000"),
        help="Reject queries whose text length exceeds this byte count. "
             "Rejection happens before parsing / execution — avoids the "
             "~6s CPU spike Vishal measured when the guardrail parses a "
             "1MB query "
             "(env: YB_MCP_MAX_QUERY_LEN, default: 100000).",
    )

    args = parser.parse_args()
    return ServerConfig(
        yugabytedb_url=args.yugabytedb_url,
        transport=args.transport,
        host=args.host,
        port=args.port,
        stateless_http=args.stateless_http,
        ssl_root_cert_secret_arn=args.yb_aws_ssl_root_cert_secret_arn,
        ssl_root_cert_key=args.yb_aws_ssl_root_cert_key,
        ssl_root_cert_path=args.yb_ssl_root_cert_path,
        ssl_root_cert_secret_region=args.yb_aws_ssl_root_cert_secret_region,
        require_where_on_update=args.require_where_on_update,
        require_where_on_delete=args.require_where_on_delete,
        auth_provider=args.mcp_auth_provider,
        enable_write_query=args.enable_write_query,
        identity_claim=args.identity_claim,
        identity_map_path=args.identity_map,
        identity_map_name=args.identity_map_name,
        pool_min_size=args.pool_min_size,
        pool_max_size=args.pool_max_size,
        statement_timeout_ms=args.statement_timeout_ms,
        max_result_rows=args.max_result_rows,
        max_result_bytes=args.max_result_bytes,
        max_query_len=args.max_query_len,
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
        if CONFIG.enable_write_query:
            self.mcp.tool(
                run_write_query,
                annotations={**_dest, "title": "Run a write SQL query (with guardrails)"},
            )
            logger.info("run_write_query tool enabled")
        else:
            logger.info("run_write_query tool disabled (use --enable-write-query or YB_MCP_ENABLE_WRITE_QUERY=true to enable)")

    def run(self, port: int | None = None):
        if CONFIG.transport == "http":
            # DB-22139: port is now a real config value (MCP_PORT / --port).
            # Fall back to the argument for callers that still pass one.
            self._run_http(CONFIG.host, port if port is not None else CONFIG.port)
        else:
            self.mcp.run(transport="stdio")

    def _run_http(self, host, port):
        # DB-22139: refuse to start when the operator has combined a
        # public bind host with no auth. Pre-fix: server defaulted to
        # `0.0.0.0:8000` and accepted anonymous /mcp requests — an
        # unauthenticated MCP→DB proxy on any IP that could reach the
        # port. Fail-closed check runs BEFORE opening the socket, so a
        # misconfigured deployment surfaces the error at startup, not
        # after the first request lands.
        _check_http_startup(host)


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


_LOOPBACK_NAMES = frozenset({"localhost"})


def _is_loopback(host: str) -> bool:
    """True if `host` is a loopback address / name. Used by the DB-22139
    refuse-to-start guard to decide whether an unauth deployment is
    acceptable (loopback-only = OK; any other bind = require auth).

    Normalizes:
    - strips surrounding whitespace and IPv6 brackets (`[::1]`)
    - lowercases
    - matches the full ``127.0.0.0/8`` block, not just ``127.0.0.1``
    - matches every IPv6 loopback form via ``ipaddress.ip_address`` —
      that catches ``::1``, ``0:0:0:0:0:0:0:1``, and any zero-prefix
      abbreviation libpq / uvicorn would accept
    """
    import ipaddress
    h = host.strip().lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if h in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _env_bool(name: str) -> bool:
    """Read an env var as a boolean. Accepts case-insensitive `true`;
    anything else is False. Matches the parse_config idiom."""
    return os.environ.get(name, "").lower() == "true"


def _check_http_startup(host: str) -> None:
    """DB-22139 fail-closed guard: refuse to start when HTTP mode is
    exposed on a non-loopback host without an auth provider.

    Pre-fix: the server bound `0.0.0.0:8000` by default and accepted
    unauthenticated /mcp requests — a full MCP→DB proxy on any interface
    that could reach the port.

    Escape hatch: `MCP_ALLOW_UNAUTHENTICATED=true` runs the server as
    unauthenticated even on a public host, with a prominent WARNING.
    Documented as dev-only in OIDC.md / README.

    Also warns when running HTTP mode without an Origin allowlist —
    DNS-rebinding attacks from a browser can reach a loopback bind if
    Origin isn't checked (`OriginValidationMiddleware` no-ops on an
    empty allowlist per its docstring).
    """
    if CONFIG.transport != "http":
        return

    on_loopback = _is_loopback(host)
    has_auth = CONFIG.auth_provider is not None
    allow_unauth = _env_bool("MCP_ALLOW_UNAUTHENTICATED")

    if not has_auth and not on_loopback and not allow_unauth:
        logger.critical(
            "HTTP transport on a non-loopback host (%s) requires an auth "
            "provider. Set MCP_AUTH_PROVIDER=cognito|oidc, or set "
            "MCP_ALLOW_UNAUTHENTICATED=true if you're intentionally "
            "running a dev-only unauthenticated instance, or bind to "
            "127.0.0.1 (unset MCP_HOST or set MCP_HOST=127.0.0.1).",
            host,
        )
        sys.exit(1)

    if not has_auth and allow_unauth and not on_loopback:
        # Loud, prominent warning — the operator opted in with the
        # escape hatch; make sure the choice is visible in prod logs.
        logger.warning(
            "=" * 78
        )
        logger.warning(
            "MCP_ALLOW_UNAUTHENTICATED=true — HTTP transport is running "
            "UNAUTHENTICATED on %s. Any client that can reach this port "
            "has full MCP→DB access. This should only be used for "
            "dev/testing.",
            host,
        )
        logger.warning(
            "=" * 78
        )

    # DB-22139 round-2: when auth is OFF, the Origin allowlist is the only
    # thing standing between the loopback bind and a browser DNS-rebinding
    # attack. Fail closed when both are missing rather than emitting a
    # warning and continuing. Escape hatch: the same
    # ``MCP_ALLOW_UNAUTHENTICATED=true`` opt-in the auth guard uses.
    empty_origins = not _parse_allowed_origins()
    if not has_auth and empty_origins and not allow_unauth:
        logger.critical(
            "HTTP transport with no auth provider AND no Origin allowlist. "
            "DNS-rebinding defense is OFF, so a browser page can drive "
            "requests to this loopback bind. Set MCP_ALLOWED_ORIGINS "
            "(comma-separated) or MCP_BASE_URL to enable the allowlist, "
            "configure MCP_AUTH_PROVIDER, or set "
            "MCP_ALLOW_UNAUTHENTICATED=true for dev/testing only."
        )
        sys.exit(1)

    if empty_origins:
        logger.warning(
            "HTTP transport is running with no Origin allowlist "
            "configured — DNS-rebinding defense is OFF. Set "
            "MCP_ALLOWED_ORIGINS (comma-separated) or MCP_BASE_URL to "
            "enable it."
        )


def _parse_allowed_origins() -> set[str]:
    """Allowed Origin values for the HTTP transport.

    Sourced from MCP_ALLOWED_ORIGINS (comma-separated). Falls back to the
    server's own MCP_BASE_URL when set, then to the empty set (no
    enforcement). When the set is empty, requests with any Origin pass; when
    non-empty, requests with an Origin not in the set are rejected.
    Requests without an Origin header (non-browser clients) always pass.

    DB-22176: RFC 6454 declares scheme + host to be case-insensitive.
    Browsers lowercase them before sending the Origin header, so an
    admin who types `MCP_ALLOWED_ORIGINS=https://MyApp.Example.com`
    would silently reject every real browser request (which sends
    `https://myapp.example.com`). Lowercase the allowlist entries here
    and lowercase the incoming Origin at compare time to match spec.
    """
    raw = os.environ.get("MCP_ALLOWED_ORIGINS", "")
    parts = {o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip()}
    if parts:
        return parts
    base = os.environ.get("MCP_BASE_URL", "").rstrip("/").lower()
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
        # DB-22176: RFC 6454 says scheme + host are case-insensitive.
        # `_parse_allowed_origins` lowercases the config; match on the
        # lowercased incoming Origin so `HTTPS://GOOD.EXAMPLE.COM` from
        # a legitimate browser still passes when the allowlist has
        # `https://good.example.com`.
        if origin is None or origin.rstrip("/").lower() in self.allowed_origins:
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
