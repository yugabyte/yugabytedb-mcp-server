"""Unit tests for OIDC identity → DB role mapping logic.

No database or MCP session required — these test the pure extraction and
transform functions in tools.py.
"""
from unittest.mock import patch, MagicMock

import pytest

from yugabytedb_mcp_server.tools import _apply_transform, _get_db_role, IdentityError


# ---------------------------------------------------------------------------
# _apply_transform
# ---------------------------------------------------------------------------

class TestApplyTransform:
    def test_none_passthrough(self):
        assert _apply_transform("alice@example.com", "none") == "alice@example.com"

    def test_strip_domain(self):
        assert _apply_transform("alice@example.com", "strip_domain") == "alice"

    def test_strip_domain_no_at_sign(self):
        assert _apply_transform("alice", "strip_domain") == "alice"

    def test_strip_domain_multiple_at_signs(self):
        assert _apply_transform("user@sub@example.com", "strip_domain") == "user"

    def test_unknown_transform_passes_through(self):
        assert _apply_transform("alice@example.com", "unknown") == "alice@example.com"


# ---------------------------------------------------------------------------
# _get_db_role
# ---------------------------------------------------------------------------

def _make_ctx(identity_claim="email", identity_transform="none"):
    """Create a minimal mock Context with lifespan_context."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {
        "identity_claim": identity_claim,
        "identity_transform": identity_transform,
    }
    return ctx


def _make_access_token(claims: dict):
    """Create a mock AccessToken with the given claims."""
    token = MagicMock()
    token.claims = claims
    return token


class TestGetDbRole:
    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_email_claim_no_transform(self, mock_get_token):
        mock_get_token.return_value = _make_access_token({"email": "alice@example.com"})
        ctx = _make_ctx(identity_claim="email", identity_transform="none")
        assert _get_db_role(ctx) == "alice@example.com"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_email_claim_strip_domain(self, mock_get_token):
        mock_get_token.return_value = _make_access_token({"email": "alice@example.com"})
        ctx = _make_ctx(identity_claim="email", identity_transform="strip_domain")
        assert _get_db_role(ctx) == "alice"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_sub_claim(self, mock_get_token):
        mock_get_token.return_value = _make_access_token({"sub": "user-uuid-123"})
        ctx = _make_ctx(identity_claim="sub", identity_transform="none")
        assert _get_db_role(ctx) == "user-uuid-123"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_preferred_username_claim(self, mock_get_token):
        mock_get_token.return_value = _make_access_token({"preferred_username": "bob"})
        ctx = _make_ctx(identity_claim="preferred_username", identity_transform="none")
        assert _get_db_role(ctx) == "bob"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_no_token_returns_none(self, mock_get_token):
        mock_get_token.return_value = None
        ctx = _make_ctx()
        assert _get_db_role(ctx) is None

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_missing_claim_raises_identity_error(self, mock_get_token):
        """When a token is present but the required claim is missing, raise
        IdentityError rather than falling back to pool credentials."""
        mock_get_token.return_value = _make_access_token({"sub": "123"})
        ctx = _make_ctx(identity_claim="email", identity_transform="none")
        with pytest.raises(IdentityError, match="email"):
            _get_db_role(ctx)

    @patch("yugabytedb_mcp_server.tools.get_access_token", side_effect=RuntimeError("no context"))
    def test_runtime_error_returns_none(self, mock_get_token):
        ctx = _make_ctx()
        assert _get_db_role(ctx) is None
