"""Pytest fixtures for yugabytedb-mcp-server.

Two test tiers:
- Unit tests (test_guardrails.py, test_auth.py): no DB, no fixtures from here.
- Integration tests (test_integration_*.py): require YUGABYTEDB_URL.

Integration tests get:
- `mcp_session`: an initialized MCP ClientSession over stdio (spawns a server subprocess)
- `db_pool`: a direct psycopg ConnectionPool used for setup/teardown side-channel
- `test_schema`: a unique schema name, dropped on teardown
"""
import os
import sys
import uuid
import pathlib
from contextlib import asynccontextmanager

import psycopg
import pytest
import pytest_asyncio

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Make _helpers importable from tests/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
UV_BIN = "uv"


def _yugabytedb_url() -> str | None:
    return os.environ.get("YUGABYTEDB_URL")


# ---------------------------------------------------------------------------
# Direct DB access (side-channel for setup/teardown)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    """A short-lived direct psycopg connection. Bypasses MCP — used by tests
    to seed tables and read-back-verify state after MCP-tool calls."""
    url = _yugabytedb_url()
    if not url:
        pytest.skip("YUGABYTEDB_URL not set")
    conn = psycopg.connect(url, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def test_schema(db_conn):
    """Yield a unique schema name; drop it (cascading) on teardown.

    Tests should fully scope their tables under this schema so they don't
    collide and so cleanup is simple.
    """
    schema = f"mcp_test_{uuid.uuid4().hex[:8]}"
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield schema
    finally:
        with db_conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


# ---------------------------------------------------------------------------
# MCP stdio client session
# ---------------------------------------------------------------------------

async def _safe_aexit(cm):
    """Close an async context manager, suppressing anyio cross-task errors.

    pytest-asyncio runs fixture teardown in a different asyncio task than
    setup. The MCP client / anyio TaskGroups bind their cancel scopes to the
    original task, raising RuntimeError on teardown. The underlying
    subprocesses/streams are still cleaned up; this just swallows the noise.
    """
    try:
        await cm.__aexit__(None, None, None)
    except (RuntimeError, BaseExceptionGroup):
        pass


def _server_params(extra_env: dict | None = None) -> StdioServerParameters:
    url = _yugabytedb_url()
    if not url:
        pytest.skip("YUGABYTEDB_URL not set")
    env = {**os.environ, "YUGABYTEDB_URL": url}
    if extra_env:
        env.update(extra_env)
    return StdioServerParameters(
        command=UV_BIN,
        args=["run", "yugabytedb-mcp"],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


@pytest_asyncio.fixture
async def mcp_session():
    """Initialized MCP ClientSession over stdio for the default config."""
    stdio_cm = stdio_client(_server_params())
    read_stream, write_stream = await stdio_cm.__aenter__()
    session_cm = ClientSession(read_stream, write_stream)
    session = await session_cm.__aenter__()
    await session.initialize()
    try:
        yield session
    finally:
        await _safe_aexit(session_cm)
        await _safe_aexit(stdio_cm)


@pytest_asyncio.fixture
async def mcp_session_strict():
    """MCP session with strict WHERE-clause enforcement and a lower bulk-INSERT limit."""
    stdio_cm = stdio_client(_server_params(extra_env={
        "YB_MCP_REQUIRE_WHERE_ON_UPDATE": "true",
        "YB_MCP_REQUIRE_WHERE_ON_DELETE": "true",
        "YB_MCP_MAX_INSERT_ROWS": "5",
    }))
    read_stream, write_stream = await stdio_cm.__aenter__()
    session_cm = ClientSession(read_stream, write_stream)
    session = await session_cm.__aenter__()
    await session.initialize()
    try:
        yield session
    finally:
        await _safe_aexit(session_cm)
        await _safe_aexit(stdio_cm)


