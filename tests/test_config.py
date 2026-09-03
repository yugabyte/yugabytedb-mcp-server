"""Unit tests for parse_config resource-limit env vars.

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
    """ argparse helper: parse s as int, raise ArgumentTypeError if
    non-int or `< 1`. Same pattern is used for the fix in the
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
    """When none of the env vars are set, defaults match the
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
        """50 MiB default byte cap."""
        cfg = _parse_with_env({})
        assert cfg.max_result_bytes == 50 * 1024 * 1024

    def test_port_default(self):
        """MCP_PORT defaults to 8000."""
        cfg = _parse_with_env({})
        assert cfg.port == 8000


class TestHttpBindConfig:
    """ MCP_HOST and MCP_PORT come from env / --host / --port."""

    def test_port_from_env(self):
        cfg = _parse_with_env({"MCP_PORT": "9090"})
        assert cfg.port == 9090

    def test_port_rejects_non_int(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"MCP_PORT": "abc"})

    def test_port_rejects_zero(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"MCP_PORT": "0"})

    def test_port_rejects_above_65535(self):
        """Post-review: port must fit a TCP port. 65536
        used to slip through _positive_int and later crashed uvicorn with
        a raw socket-bind traceback."""
        with pytest.raises(SystemExit):
            _parse_with_env({"MCP_PORT": "65536"})

    def test_port_rejects_negative(self):
        with pytest.raises(SystemExit):
            _parse_with_env({"MCP_PORT": "-1"})

    def test_port_accepts_65535(self):
        cfg = _parse_with_env({"MCP_PORT": "65535"})
        assert cfg.port == 65535


class TestPoolSizingValidation:
    """pool_min_size <= pool_max_size is enforced at
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


class TestConnectTimeoutAppend:
    """Post-review: when ``YUGABYTEDB_URL`` doesn't
    already carry a ``connect_timeout`` we append one. The append has to
    respect the two conninfo formats libpq accepts — keyword form
    (space-separated) and URI form (query-string). Space-appending to a
    URI mangles it (`?sslmode=disable connect_timeout=10` — psycopg
    rejects with "extra key/value separator")."""

    def _append(self, url: str) -> str:
        """Mirror the logic in ``app_lifespan``."""
        if "connect_timeout" in url.lower():
            return url
        if url.startswith(("postgres://", "postgresql://")):
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}connect_timeout=10"
        return f"{url} connect_timeout=10"

    def test_keyword_form_bare(self):
        assert self._append("host=localhost port=5433 dbname=yb user=yb") == (
            "host=localhost port=5433 dbname=yb user=yb connect_timeout=10"
        )

    def test_keyword_form_already_set(self):
        u = "host=localhost connect_timeout=5"
        assert self._append(u) == u

    def test_keyword_form_case_insensitive_already_set(self):
        u = "Connect_Timeout=5 host=localhost"
        assert self._append(u) == u

    def test_uri_form_no_query(self):
        assert self._append("postgresql://yb@localhost:5433/db") == (
            "postgresql://yb@localhost:5433/db?connect_timeout=10"
        )

    def test_uri_form_with_query(self):
        assert self._append(
            "postgresql://yb@localhost:5433/db?sslmode=require"
        ) == "postgresql://yb@localhost:5433/db?sslmode=require&connect_timeout=10"

    def test_uri_form_already_set(self):
        u = "postgresql://yb@localhost:5433/db?connect_timeout=5"
        assert self._append(u) == u

    def test_uri_short_scheme(self):
        assert self._append("postgres://yb@localhost/db") == (
            "postgres://yb@localhost/db?connect_timeout=10"
        )


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


class TestAuthProviderPreflight:
    """DB-22184 follow-up: `_create_cognito` / `_create_oidc` raise
    `ValueError` on missing env; `main()` catches those in the same
    pre-flight window as `parser.error()` so an auth-config mistake
    surfaces as a clean stderr line + exit code 2, not a raw
    ValueError traceback.

    Structurally the check was already pre-ASGI (raised inside
    `YugabyteDBMCPServer.__init__`, which runs before `server.run()`);
    this test pins the presentation as well as the ordering."""

    def test_main_exits_cleanly_on_missing_auth_env(self, capsys):
        from unittest.mock import MagicMock
        from yugabytedb_mcp_server import server as server_module

        with patch.object(server_module, "parse_config") as mock_parse, \
             patch.object(
                 server_module, "YugabyteDBMCPServer",
                 side_effect=ValueError(
                     "Cognito auth is missing required env vars: X, Y"
                 ),
             ):
            mock_parse.return_value = MagicMock()
            with pytest.raises(SystemExit) as excinfo:
                server_module.main()

        assert excinfo.value.code == 2
        stderr = capsys.readouterr().err
        assert "yugabytedb-mcp: error:" in stderr
        assert "Cognito auth is missing required env vars: X, Y" in stderr
        # A regression that lets the ValueError propagate would leave a
        # traceback in stderr — assert we don't see one.
        assert "Traceback" not in stderr

    def test_main_does_not_swallow_non_valueerror(self):
        """Guard against the catch getting overly broad. Only ValueError
        (which is what `_require_env` raises) should be redirected to
        the clean pre-flight exit; anything else must propagate so real
        bugs aren't hidden."""
        from unittest.mock import MagicMock
        from yugabytedb_mcp_server import server as server_module

        with patch.object(server_module, "parse_config") as mock_parse, \
             patch.object(
                 server_module, "YugabyteDBMCPServer",
                 side_effect=RuntimeError("unrelated bug"),
             ):
            mock_parse.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="unrelated bug"):
                server_module.main()


class TestRequiredConfigPreflight:
    """DB-22182: previously the ``YUGABYTEDB_URL`` presence check ran
    inside the async ``app_lifespan`` via ``sys.exit(1)``, which
    surfaced to the operator as a SystemExit traceback + uvicorn's
    "Application startup failed" (exit code 3). Now handled in
    ``parse_config`` via ``parser.error(...)`` — clean argparse error
    with exit code 2 before the ASGI app is constructed."""

    def test_missing_yugabytedb_url_fails_at_parse_config(self):
        # Remove YUGABYTEDB_URL from the clean env baseline.
        clean_env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("YB_MCP_")
            and k not in ("MCP_AUTH_PROVIDER", "YUGABYTEDB_URL")
        }
        with patch.dict(os.environ, clean_env, clear=True), \
             patch.object(sys, "argv", ["yugabytedb-mcp"]):
            with pytest.raises(SystemExit) as excinfo:
                parse_config()
        # argparse's parser.error exits with code 2, not the pre-fix code
        # 1 (from sys.exit inside app_lifespan) or code 3 (uvicorn's ASGI
        # startup failure). Assert the argparse-style code so a
        # regression that resurrects the async-lifespan check is caught.
        assert excinfo.value.code == 2


class TestSslRootCertGuard:
    """DB-22185: the Secrets-Manager cert fetch used to check whether the
    conninfo already carried a ``sslrootcert`` via a naive substring
    match (``"sslrootcert" in conninfo``). Any field value containing
    that literal — e.g. ``password=sslrootcertPW123`` — false-positived,
    silently DROPPING the fetched cert. Fix: match the actual libpq
    keyword form (``\\bsslrootcert=``) or URI query param
    (``?sslrootcert=`` / ``&sslrootcert=``)."""

    def _has(self, conninfo: str) -> bool:
        from yugabytedb_mcp_server.server import _has_sslrootcert
        return _has_sslrootcert(conninfo)

    def test_password_containing_substring_is_not_sslrootcert(self):
        """The exact repro from the ticket: password value contains the
        literal substring but there's no actual sslrootcert parameter."""
        assert not self._has(
            "host=x port=5433 password=sslrootcertPW123 user=y"
        )

    def test_keyword_form_real_sslrootcert_matches(self):
        assert self._has("host=x sslrootcert=/etc/x.crt user=y")

    def test_keyword_form_at_start_of_string_matches(self):
        assert self._has("sslrootcert=/etc/x.crt host=y")

    def test_keyword_form_with_spaces_around_equals(self):
        assert self._has("host=x sslrootcert =/etc/x.crt")

    def test_uri_form_query_param_first_matches(self):
        assert self._has("postgresql://u@h:5433/db?sslrootcert=/x")

    def test_uri_form_query_param_second_matches(self):
        assert self._has(
            "postgresql://u@h:5433/db?sslmode=require&sslrootcert=/x"
        )

    def test_uri_form_password_containing_substring_no_match(self):
        # `hassslrootcertinit` is a fake password with the substring in
        # the middle — no boundary char immediately before `sslrootcert`.
        assert not self._has(
            "postgresql://u:hassslrootcertinit@h:5433/db?sslmode=require"
        )

    def test_case_insensitive(self):
        assert self._has("host=x SSLROOTCERT=/etc/x.crt")

    def test_bare_conninfo_no_match(self):
        assert not self._has("host=x port=5433 dbname=y user=y")


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
