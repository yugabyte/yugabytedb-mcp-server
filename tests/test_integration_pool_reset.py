"""Integration tests for DISCARD ALL pool reset.

Verifies that session state left behind by one tool call is scrubbed
before the next pool checkout. Requires YUGABYTEDB_URL.

Repros driven from the audit's exact wording:

- SET ROLE inside a checkout doesn't leak to the next checkout
  even when `_conn_as_role`'s `RESET ROLE` in `finally` gets skipped
  (belt-and-braces test using a raw pool, not the tool wrapper).
- (advisory locks): SELECT pg_advisory_lock(k) survives the
  read tool's ROLLBACK; with DISCARD ALL, the lock is released on
  return so an independent acquisition of the same key doesn't block.
- (prepared statement cache): DEALLOCATE ALL through the read
  tool desyncs psycopg's prepared-statement cache; with DISCARD ALL,
  the next read query succeeds instead of failing with
  `prepared statement "_pg3_0" does not exist`.
"""
import asyncio
import os
import sys
import pathlib

import psycopg
import pytest
from psycopg_pool import ConnectionPool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import requires_yugabytedb, parse_json_list, raw_text  # noqa: E402

from yugabytedb_mcp_server.server import _pool_reset


pytestmark = requires_yugabytedb


def _url() -> str:
    return os.environ["YUGABYTEDB_URL"]


# ---------------------------------------------------------------------------
# Direct pool-level tests — drive the ConnectionPool with the reset callback
# ourselves, no MCP session in the loop. These test the mechanism in
# isolation, so a failure here points at the pool wiring, not at any tool.
# ---------------------------------------------------------------------------

def test_discard_all_clears_set_role():
    """ SET ROLE on a checkout, return connection, next checkout
    is back to the pool's default role."""
    pool = ConnectionPool(
        conninfo=_url(),
        min_size=1,
        max_size=1,        # single connection so we know the next checkout
                            # gets the same conn back — proves reset ran
        open=True,
        reset=_pool_reset,
    )
    try:
        # Establish baseline
        with pool.connection() as conn:
            baseline = conn.execute("SELECT current_role").fetchone()[0]

        # Create a temp role, use it, return the conn.
        # Postgres CREATE ROLE has no IF NOT EXISTS syntax — swallow the
        # duplicate-object error if a prior run left the role behind.
        with pool.connection() as conn:
            try:
                conn.execute("CREATE ROLE mcp_test_reset")
            except psycopg.errors.DuplicateObject:
                pass
            conn.execute("GRANT mcp_test_reset TO CURRENT_USER")
            conn.execute("SET ROLE mcp_test_reset")
            role_inside = conn.execute("SELECT current_role").fetchone()[0]
            assert role_inside == "mcp_test_reset"

        # Next checkout must NOT still be in mcp_test_reset.
        with pool.connection() as conn:
            role_after = conn.execute("SELECT current_role").fetchone()[0]
            assert role_after == baseline, (
                f" SET ROLE leaked across pool checkouts. "
                f"expected={baseline}, got={role_after}"
            )

        # Cleanup
        with pool.connection() as conn:
            conn.execute("DROP ROLE IF EXISTS mcp_test_reset")
    finally:
        pool.close()


def test_discard_all_clears_advisory_lock():
    """ advisory lock taken on one checkout must not survive to
    the next. Proves the cross-user DoS repro is closed."""
    lock_key = 42421  # arbitrary; test-scoped

    pool = ConnectionPool(
        conninfo=_url(),
        min_size=1,
        max_size=1,
        open=True,
        reset=_pool_reset,
    )
    try:
        # Take the lock and return the connection.
        with pool.connection() as conn:
            got_lock = conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (lock_key,)
            ).fetchone()[0]
            assert got_lock is True

        # After DISCARD ALL, the lock should be gone. Verify from a
        # SEPARATE psycopg connection (not from the pool) so a
        # would-be-reacquisition on the same conn doesn't succeed for
        # unrelated reasons.
        with psycopg.connect(_url(), autocommit=True) as observer:
            still_held = observer.execute(
                "SELECT COUNT(*) FROM pg_locks WHERE locktype = 'advisory' "
                "AND objid = %s",
                (lock_key,),
            ).fetchone()[0]
            assert still_held == 0, (
                f" pg_advisory_lock({lock_key}) survived pool "
                f"return; DISCARD ALL didn't release it. "
                f"still_held rows={still_held}"
            )
    finally:
        pool.close()


def test_discard_all_survives_prepared_statement_desync():
    """ DEALLOCATE ALL inside a tool call desyncs psycopg's
    client-side prepared-statement cache. Without DISCARD ALL on
    connection return, subsequent read queries fail with
    `prepared statement "_pg3_0" does not exist`. With DISCARD ALL,
    the connection is clean for the next checkout."""
    pool = ConnectionPool(
        conninfo=_url(),
        min_size=1,
        max_size=1,
        open=True,
        reset=_pool_reset,
    )
    try:
        # First: cause psycopg to prepare statements (it prepares after
        # a query is executed a few times) and then DEALLOCATE ALL —
        # the client-side cache still thinks the plans exist, but the
        # server has forgotten them.
        with pool.connection() as conn:
            for _ in range(8):
                conn.execute("SELECT 1").fetchone()
            conn.execute("DEALLOCATE ALL")

        # Next checkout — same underlying conn, but DISCARD ALL should
        # have cleared the server-side prepared plans (DISCARD PLANS is
        # part of DISCARD ALL) so any client-side cache mismatch is
        # resolved. This query should succeed.
        with pool.connection() as conn:
            result = conn.execute("SELECT 2").fetchone()[0]
            assert result == 2

        # And a repeat query — this is where the pre-fix behavior would
        # fail with "prepared statement _pg3_0 does not exist" because
        # the second identical query triggers psycopg's prepared path.
        with pool.connection() as conn:
            result = conn.execute("SELECT 2").fetchone()[0]
            assert result == 2
    finally:
        pool.close()


def test_discard_all_clears_session_gucs():
    """Bonus coverage: DISCARD ALL clears session GUC changes so a
    caller can't leave `SET default_transaction_read_only=off` (or any
    other SET) behind for the next user."""
    pool = ConnectionPool(
        conninfo=_url(),
        min_size=1,
        max_size=1,
        open=True,
        reset=_pool_reset,
    )
    try:
        with pool.connection() as conn:
            baseline = conn.execute(
                "SHOW default_transaction_read_only"
            ).fetchone()[0]

        with pool.connection() as conn:
            conn.execute("SET default_transaction_read_only = on")
            during = conn.execute(
                "SHOW default_transaction_read_only"
            ).fetchone()[0]
            assert during == "on"

        with pool.connection() as conn:
            after = conn.execute(
                "SHOW default_transaction_read_only"
            ).fetchone()[0]
            assert after == baseline, (
                f"Session GUC leaked: baseline={baseline}, after checkout={after}"
            )
    finally:
        pool.close()
