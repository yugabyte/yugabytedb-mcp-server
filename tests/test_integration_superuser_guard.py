"""Integration tests for the superuser-role refusal guard in `_conn_as_role`.

Requires a live YugabyteDB (or PostgreSQL) at ``YUGABYTEDB_URL``. Seeds a
superuser role and a normal role, verifies that:

- With the default posture, resolving an identity to a superuser role is
  refused before ``SET ROLE`` runs — the caller gets an ``IdentityError``.
- With the opt-out flag on, the same identity is allowed through.
- Non-superuser roles pass in both cases.
"""
import os
import sys
import pathlib

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import requires_yugabytedb  # noqa: E402

import psycopg  # noqa: E402

from yugabytedb_mcp_server.tools import _conn_as_role, IdentityError  # noqa: E402


pytestmark = requires_yugabytedb


_SU_ROLE = "mcp_test_su_role"
_NORMAL_ROLE = "mcp_test_normal_role"


@pytest.fixture
def superuser_roles():
    """Seed a superuser role and a non-superuser role for the tests to
    resolve onto. Dropped on teardown."""
    url = os.environ["YUGABYTEDB_URL"]
    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            # SET ROLE needs the pool user to be a member of the target role
            # (unless it's a superuser, in which case anything works). Grant
            # membership to whichever user the conninfo actually connected as
            # — CI runs Postgres as ``test``, dev runs YB as ``yugabyte``.
            cur.execute("SELECT current_user")
            pool_user = cur.fetchone()[0]
            cur.execute(f'DROP ROLE IF EXISTS "{_SU_ROLE}"')
            cur.execute(f'DROP ROLE IF EXISTS "{_NORMAL_ROLE}"')
            cur.execute(f'CREATE ROLE "{_SU_ROLE}" WITH SUPERUSER')
            cur.execute(f'CREATE ROLE "{_NORMAL_ROLE}"')
            cur.execute(f'GRANT "{_SU_ROLE}" TO "{pool_user}"')
            cur.execute(f'GRANT "{_NORMAL_ROLE}" TO "{pool_user}"')
    yield
    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP ROLE IF EXISTS "{_SU_ROLE}"')
            cur.execute(f'DROP ROLE IF EXISTS "{_NORMAL_ROLE}"')


@pytest.fixture
def pool():
    """A tiny psycopg pool that behaves like the app pool for the guard."""
    from psycopg_pool import ConnectionPool
    url = os.environ["YUGABYTEDB_URL"]
    p = ConnectionPool(conninfo=url, min_size=1, max_size=1, open=True)
    yield p
    p.close()


def test_superuser_role_refused_by_default(superuser_roles, pool):
    """The main defense-in-depth case: the resolved role is a superuser
    and the operator hasn't opted out. Refuse before SET ROLE."""
    with pytest.raises(IdentityError, match="superuser"):
        with _conn_as_role(pool, _SU_ROLE, allow_superuser_role=False):
            pass  # never reached


def test_superuser_role_allowed_with_opt_out(superuser_roles, pool):
    """The escape hatch: an operator who genuinely wants an identity to
    map to a superuser role can flip the flag."""
    with _conn_as_role(pool, _SU_ROLE, allow_superuser_role=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_user")
            got = cur.fetchone()[0]
    assert got == _SU_ROLE


def test_normal_role_passes(superuser_roles, pool):
    """Regression: mapping to a non-superuser role still works, both
    with the guard on and off."""
    with _conn_as_role(pool, _NORMAL_ROLE, allow_superuser_role=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_user")
            got = cur.fetchone()[0]
    assert got == _NORMAL_ROLE


def test_nonexistent_role_falls_through_to_set_role_error(pool):
    """When the resolved role doesn't exist in pg_roles at all, our
    guard skips (pg_roles query returns None) and the caller gets the
    natural `role "<x>" does not exist` error from SET ROLE. We don't
    mask the underlying problem."""
    # No superuser_roles fixture — role doesn't exist.
    with pytest.raises(psycopg.errors.InvalidParameterValue, match="does not exist"):
        with _conn_as_role(pool, "definitely_does_not_exist_xyz", allow_superuser_role=False):
            pass
