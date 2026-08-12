"""Unit tests for parse_config resource-limit env vars (DB-22159).

Covers the `_positive_int` argparse helper and verifies the defaults +
env-var → ServerConfig plumbing for the five new resource-limit knobs.
"""
import argparse
import os
import sys
from unittest.mock import patch

import pytest

from yugabytedb_mcp_server.server import _positive_int, parse_config


# ---------------------------------------------------------------------------
# _positive_int helper
# ---------------------------------------------------------------------------

class TestPositiveInt:
    """DB-22159 argparse helper: parse s as int, raise ArgumentTypeError if
    non-int or `< 1`. Same pattern is used for the DB-22162 fix in the
    follow-up release."""

    def test_accepts_positive_int(self):
        assert _positive_int("1") == 1
        assert _positive_int("42") == 42
        assert _positive_int("1000000") == 1_000_000

    def test_rejects_zero(self):
        with pytest.raises(argparse.ArgumentTypeError, match=r"must be >= 1"):
            _positive_int("0")

    def test_rejects_negative(self):
        with pytest.raises(argparse.ArgumentTypeError, match=r"must be >= 1"):
            _positive_int("-5")

    def test_rejects_non_int(self):
        with pytest.raises(argparse.ArgumentTypeError, match=r"positive integer"):
            _positive_int("abc")

    def test_rejects_empty_string(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("")

    def test_rejects_float(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("3.14")

    def test_accepts_whitespace_trimmed_by_int(self):
        # `int("  42  ")` succeeds — argparse defers to int()'s behavior.
        # Documents the behavior; no fix needed.
        assert _positive_int("  42  ") == 42


# ---------------------------------------------------------------------------
# parse_config resource-limit defaults + env parsing
# ---------------------------------------------------------------------------

def _parse_with_env(env: dict) -> object:
    """Run parse_config with a clean argv and the given env overlay."""
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("YB_MCP_") and k not in ("MCP_AUTH_PROVIDER",)
    }
    clean_env["YUGABYTEDB_URL"] = "host=localhost port=5433 dbname=yb user=yb"
    clean_env.update(env)
    with patch.dict(os.environ, clean_env, clear=True), \
         patch.object(sys, "argv", ["yugabytedb-mcp"]):
        return parse_config()


class TestResourceLimitDefaults:
    """When none of the DB-22159 env vars are set, defaults match the
    documented values."""

    def test_pool_min_size_default(self):
        cfg = _parse_with_env({})
        assert cfg.pool_min_size == 1

    def test_pool_max_size_default(self):
        cfg = _parse_with_env({})
        assert cfg.pool_max_size == 5

    def test_statement_timeout_ms_default(self):
        cfg = _parse_with_env({})
        assert cfg.statement_timeout_ms == 30_000

    def test_max_result_rows_default(self):
        cfg = _parse_with_env({})
        assert cfg.max_result_rows == 10_000

    def test_max_query_len_default(self):
        cfg = _parse_with_env({})
        assert cfg.max_query_len == 100_000

    def test_max_result_bytes_default(self):
        """DB-22159 round-2: 50 MiB default byte cap."""
        cfg = _parse_with_env({})
        assert cfg.max_result_bytes == 50 * 1024 * 1024


class TestResourceLimitEnvParsing:
    """Env-var overrides populate ServerConfig correctly."""

    def test_pool_sizes_from_env(self):
        cfg = _parse_with_env({
            "YB_MCP_POOL_MIN_SIZE": "2",
            "YB_MCP_POOL_MAX_SIZE": "20",
        })
        assert cfg.pool_min_size == 2
        assert cfg.pool_max_size == 20

    def test_statement_timeout_from_env(self):
        cfg = _parse_with_env({"YB_MCP_STATEMENT_TIMEOUT_MS": "5000"})
        assert cfg.statement_timeout_ms == 5_000

    def test_max_result_rows_from_env(self):
        cfg = _parse_with_env({"YB_MCP_MAX_RESULT_ROWS": "500"})
        assert cfg.max_result_rows == 500

    def test_max_query_len_from_env(self):
        cfg = _parse_with_env({"YB_MCP_MAX_QUERY_LEN": "1000"})
        assert cfg.max_query_len == 1_000

    def test_max_result_bytes_from_env(self):
        cfg = _parse_with_env({"YB_MCP_MAX_RESULT_BYTES": "1048576"})
        assert cfg.max_result_bytes == 1_048_576


class TestPoolSizingValidation:
    """DB-22159 round-2: pool_min_size <= pool_max_size is enforced at
    app_lifespan startup, not silently at pool.open time."""

    def test_min_larger_than_max_rejected(self):
        """The bad combo is caught inside app_lifespan before the pool is
        opened. parse_config accepts the individual values (both are just
        positive ints); the relational check runs later."""
        from yugabytedb_mcp_server import server as server_module
        cfg = _parse_with_env({
            "YB_MCP_POOL_MIN_SIZE": "10",
            "YB_MCP_POOL_MAX_SIZE": "5",
        })
        assert cfg.pool_min_size == 10
        assert cfg.pool_max_size == 5

        original = getattr(server_module, "CONFIG", None)
        server_module.CONFIG = cfg
        try:
            async def _drive():
                async with server_module.app_lifespan(None) as _:
                    pass
            import asyncio
            with pytest.raises(ValueError, match="POOL_MIN_SIZE"):
                asyncio.run(_drive())
        finally:
            if original is None:
                try:
                    del server_module.CONFIG
                except AttributeError:
                    pass
            else:
                server_module.CONFIG = original


class TestResourceLimitEnvValidation:
    """Bad env values fail startup with a clean argparse error, not a
    traceback."""

    def test_pool_max_size_rejects_non_int(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"YB_MCP_POOL_MAX_SIZE": "abc"})

    def test_pool_max_size_rejects_zero(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"YB_MCP_POOL_MAX_SIZE": "0"})

    def test_pool_max_size_rejects_negative(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"YB_MCP_POOL_MAX_SIZE": "-1"})

    def test_statement_timeout_rejects_zero(self):
        """Zero statement_timeout would mean 'no timeout' in PG — we reject
        it here so the operator can't accidentally disable the DoS guard."""
        with pytest.raises(SystemExit):
            _parse_with_env({"YB_MCP_STATEMENT_TIMEOUT_MS": "0"})

    def test_max_result_rows_rejects_non_int(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"YB_MCP_MAX_RESULT_ROWS": "unlimited"})

    def test_max_query_len_rejects_negative(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"YB_MCP_MAX_QUERY_LEN": "-100"})
