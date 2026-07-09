"""Unit tests for OIDC identity → DB role mapping logic.

No database or MCP session required — these test the pure extraction,
transform, and mapping functions in tools.py.

Layout:
- TestApplyTransform / TestGetDbRole — v1 backward-compat harness (must
  stay green when v2 features are unset).
- TestExtractClaim — v2 dotted-path claim extraction.
- TestMapFileParser — v2 pg_ident.conf-style map file parsing.
- TestApplyMap — v2 map-entry matching (literal + regex + capture).
- TestPickRole — v2 list-claim role selection 
- TestGetDbRoleV2 — end-to-end v2 wiring on `_get_db_role`.
- TestFailClosed — malformed/missing map files at load time.
- TestSecurityInvariants — role-name Identifier quoting, empty-list handling.
"""
import re
from unittest.mock import patch, MagicMock

import pytest

from yugabytedb_mcp_server.tools import (
    IdentityError,
    MapEntry,
    _apply_map,
    _apply_transform,
    _extract_claim,
    _get_db_role,
    _load_identity_map,
    _pick_role,
)


# ---------------------------------------------------------------------------
# _apply_transform — v1 backward-compat helper
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
# Test helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    identity_claim="email",
    identity_transform="none",
    identity_map=None,
    identity_map_name="default",
):
    """Create a minimal mock Context with lifespan_context.

    Passing ``identity_map=None`` (default) selects the v1 backward-compat
    path via ``_apply_transform``; passing a list of ``MapEntry`` selects the
    v2 map-lookup path.
    """
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {
        "identity_claim": identity_claim,
        "identity_transform": identity_transform,
        "identity_map": identity_map,
        "identity_map_name": identity_map_name,
    }
    return ctx


def _make_access_token(claims: dict):
    """Create a mock AccessToken with the given claims."""
    token = MagicMock()
    token.claims = claims
    return token


# ---------------------------------------------------------------------------
# _get_db_role — v1 backward-compat regression harness
# ---------------------------------------------------------------------------

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


# ===========================================================================
# v2 mapping tests below.
# ===========================================================================


# ---------------------------------------------------------------------------
# _extract_claim — dotted-path claim extraction
# ---------------------------------------------------------------------------
# YSQL parity: dots separate nested-dict path segments; colons are literal
# key chars (Cognito's `cognito:groups` is one key, not two).

class TestExtractClaim:
    def test_top_level_string(self):
        claims = {"email": "bob@a.com"}
        assert _extract_claim(claims, "email") == "bob@a.com"

    def test_top_level_list(self):
        claims = {"roles": ["writer", "reader"]}
        assert _extract_claim(claims, "roles") == ["writer", "reader"]

    def test_dotted_path(self):
        """Keycloak-style `realm_access.roles` walks the nested dict."""
        claims = {"realm_access": {"roles": ["writer"]}}
        assert _extract_claim(claims, "realm_access.roles") == ["writer"]

    def test_dotted_path_deep(self):
        claims = {"a": {"b": {"c": "deep"}}}
        assert _extract_claim(claims, "a.b.c") == "deep"

    def test_dotted_with_colon(self):
        """Cognito's `cognito:groups` is a top-level key with a colon —
        NOT a dotted path. Verify the colon doesn't split."""
        claims = {"cognito:groups": ["reader", "writer"]}
        assert _extract_claim(claims, "cognito:groups") == ["reader", "writer"]

    def test_missing_top_level_key(self):
        with pytest.raises(IdentityError, match="email"):
            _extract_claim({"sub": "x"}, "email")

    def test_missing_intermediate_key(self):
        """`a.b.c` where `a` exists but has no `b` — error mentions the path."""
        with pytest.raises(IdentityError, match=r"a\.b\.c"):
            _extract_claim({"a": {}}, "a.b.c")

    def test_missing_final_key(self):
        with pytest.raises(IdentityError, match=r"a\.b\.c"):
            _extract_claim({"a": {"b": {}}}, "a.b.c")

    def test_cannot_descend_into_non_dict(self):
        """`a.b` where `a` is a string — error says can't descend."""
        with pytest.raises(IdentityError, match="cannot descend"):
            _extract_claim({"a": "scalar"}, "a.b")


# ---------------------------------------------------------------------------
# _load_identity_map — pg_ident.conf-style parser
# ---------------------------------------------------------------------------

@pytest.fixture
def map_file_factory(tmp_path):
    """Return a callable that writes given content to a tmp file and returns its path."""
    def _write(content: str, name: str = "yb_ident.conf") -> str:
        p = tmp_path / name
        p.write_text(content)
        return str(p)
    return _write


class TestMapFileParser:
    def test_literal_entry(self, map_file_factory):
        path = map_file_factory("map1  user@yugabyte.com  user\n")
        entries = _load_identity_map(path)
        assert len(entries) == 1
        assert entries[0] == MapEntry(
            name="map1", pattern="user@yugabyte.com", role="user",
            is_regex=False, compiled=None,
        )

    def test_regex_entry(self, map_file_factory):
        """From the YSQL docs, verbatim."""
        path = map_file_factory(
            "map2  /^(.*)@devadmincloudyugabyte\\.onmicrosoft\\.com$  \\1\n"
        )
        entries = _load_identity_map(path)
        assert entries[0].is_regex is True
        assert entries[0].compiled is not None
        assert entries[0].role == r"\1"

    def test_role_to_role_entry(self, map_file_factory):
        """From the YSQL docs, verbatim: OIDC group name → PG role name."""
        path = map_file_factory("map1  OIDC.Test.Read  read_only_user\n")
        entries = _load_identity_map(path)
        assert entries[0].pattern == "OIDC.Test.Read"
        assert entries[0].role == "read_only_user"
        assert entries[0].is_regex is False

    def test_azure_guid_regex(self, map_file_factory):
        """Azure AD group GUID → readable role via regex. Follows the YSQL
        docs convention: leading `/` marks the pattern as regex; NO closing
        `/` (the rest of the field is the regex body verbatim)."""
        path = map_file_factory("azure  /^a12d04b1-.+  reader\n")
        entries = _load_identity_map(path)
        assert entries[0].compiled.fullmatch("a12d04b1-7463-8e23-94d2-8d71f17ab99b")

    def test_comments_and_blank_lines_ignored(self, map_file_factory):
        path = map_file_factory(
            "# leading comment\n"
            "\n"
            "map1  alice  a_role\n"
            "  # indented comment\n"
            "\n"
            "map1  bob    b_role  # trailing comment\n"
        )
        entries = _load_identity_map(path)
        assert len(entries) == 2
        assert entries[0].pattern == "alice"
        assert entries[1].pattern == "bob"

    def test_multiple_named_maps(self, map_file_factory):
        path = map_file_factory(
            "default  alice  a_role\n"
            "azure    bob    b_role\n"
            "default  carol  c_role\n"
        )
        entries = _load_identity_map(path)
        assert [e.name for e in entries] == ["default", "azure", "default"]

    def test_malformed_regex_line_raises(self, map_file_factory):
        path = map_file_factory("map1  /[unterminated  reader\n")
        with pytest.raises(ValueError, match="invalid regex"):
            _load_identity_map(path)

    def test_wrong_field_count_raises(self, map_file_factory):
        path = map_file_factory("map1  too_few\n")
        with pytest.raises(ValueError, match="3 space-separated"):
            _load_identity_map(path)

    def test_extra_fields_collapse_into_role(self, map_file_factory):
        """Fourth+ whitespace-separated fields become part of the role value —
        pg_ident allows this and users occasionally have role names with
        embedded whitespace. Documents current behavior."""
        # split(None, 2) collapses trailing whitespace into field 3
        path = map_file_factory("map1 alice role_with_spaces\n")
        entries = _load_identity_map(path)
        assert entries[0].role == "role_with_spaces"

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _load_identity_map(str(tmp_path / "does-not-exist.conf"))


# ---------------------------------------------------------------------------
# _apply_map — match a value against map entries
# ---------------------------------------------------------------------------

def _literal(name: str, pattern: str, role: str) -> MapEntry:
    return MapEntry(name=name, pattern=pattern, role=role, is_regex=False, compiled=None)


def _regex(name: str, pattern: str, role: str) -> MapEntry:
    return MapEntry(
        name=name, pattern="/" + pattern, role=role, is_regex=True,
        compiled=re.compile(pattern),
    )


class TestApplyMap:
    def test_literal_match(self):
        entries = [_literal("default", "bob@yugabyte.com", "writer")]
        assert _apply_map("bob@yugabyte.com", entries, "default") == "writer"

    def test_regex_capture(self):
        entries = [_regex("default", r"^([a-z]+)@yugabyte\.com$", r"\1")]
        assert _apply_map("carol@yugabyte.com", entries, "default") == "carol"

    def test_role_to_role_literal(self):
        entries = [_literal("default", "OIDC.Test.Read", "read_only_user")]
        assert _apply_map("OIDC.Test.Read", entries, "default") == "read_only_user"

    def test_azure_guid_regex(self):
        entries = [_regex("azure", r"^a12d04b1-.+", "reader")]
        assert _apply_map(
            "a12d04b1-7463-8e23-94d2-8d71f17ab99b", entries, "azure",
        ) == "reader"

    def test_unmapped_returns_none(self):
        entries = [_literal("default", "alice", "a")]
        assert _apply_map("someone-else", entries, "default") is None

    def test_map_name_filter(self):
        """Entries under a different map_name must not match."""
        entries = [
            _literal("default", "alice", "wrong"),
            _literal("azure", "alice", "right"),
        ]
        assert _apply_map("alice", entries, "azure") == "right"

    def test_map_name_isolation_no_leak(self):
        """A `default` entry must not match under `map_name="azure"`."""
        entries = [_literal("default", "alice", "a")]
        assert _apply_map("alice", entries, "azure") is None

    def test_regex_is_anchored(self):
        """Regex uses fullmatch — a pattern without explicit anchors still
        requires the whole value to match, mirroring pg_ident.conf."""
        entries = [_regex("default", r"foo", "matched")]
        assert _apply_map("foo", entries, "default") == "matched"
        assert _apply_map("foobar", entries, "default") is None  # not anchored via search

    def test_first_match_wins(self):
        """When multiple entries under the same map_name could match, the
        first one wins (mirrors pg_ident.conf iteration order)."""
        entries = [
            _literal("default", "alice", "first"),
            _literal("default", "alice", "second"),
        ]
        assert _apply_map("alice", entries, "default") == "first"


# ---------------------------------------------------------------------------
# _pick_role — list-claim role selection
# ---------------------------------------------------------------------------
# User can take any PG role that is in their roles/groups claim". 
# The MCP tool's `requested_role` is the caller
# (agent)'s choice; server clamps against the candidate list.

class TestPickRole:
    def test_single_candidate_auto_pick(self):
        assert _pick_role(["writer"], None) == "writer"

    def test_single_candidate_ignores_requested_role(self):
        """One candidate is always picked, even if requested_role is different
        (we could raise here — but the ergonomic choice is auto-pick since
        the caller had exactly one option)."""
        assert _pick_role(["writer"], "reader") == "writer"

    def test_multi_candidate_requested_role_in_list(self):
        assert _pick_role(["writer", "reader"], "reader") == "reader"

    def test_multi_candidate_requested_role_not_in_list_raises(self):
        with pytest.raises(IdentityError, match="not in the caller's identity-claim"):
            _pick_role(["writer", "reader"], "admin")

    def test_multi_candidate_no_request_defaults_to_first(self, caplog):
        with caplog.at_level("WARNING"):
            assert _pick_role(["writer", "reader"], None) == "writer"
        assert any("multiple roles" in r.message for r in caplog.records)

    def test_empty_candidates_raises(self):
        with pytest.raises(IdentityError, match="None of the identity-claim values"):
            _pick_role([], None)

    def test_empty_candidates_raises_even_with_requested_role(self):
        """No candidates → error regardless of requested_role (can't clamp
        against an empty set)."""
        with pytest.raises(IdentityError, match="None of the identity-claim values"):
            _pick_role([], "anything")


# ---------------------------------------------------------------------------
# _get_db_role — end-to-end v2 wiring
# ---------------------------------------------------------------------------

class TestGetDbRoleV2:
    """Exercise `_get_db_role` with v2 config (list claims, map file,
    requested_role parameter)."""

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_cognito_access_token_groups_flow(self, mock_get_token):
        """Realistic Cognito access-token workflow — DB-22192 unblocked."""
        mock_get_token.return_value = _make_access_token(
            {"sub": "abc", "cognito:groups": ["writer", "reader"]}
        )
        ctx = _make_ctx(identity_claim="cognito:groups", identity_map=None)
        assert _get_db_role(ctx, requested_role="reader") == "reader"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_cognito_groups_no_requested_role_defaults_first(self, mock_get_token):
        mock_get_token.return_value = _make_access_token(
            {"cognito:groups": ["writer", "reader"]}
        )
        ctx = _make_ctx(identity_claim="cognito:groups", identity_map=None)
        assert _get_db_role(ctx) == "writer"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_cognito_groups_requested_not_in_list_raises(self, mock_get_token):
        mock_get_token.return_value = _make_access_token(
            {"cognito:groups": ["writer", "reader"]}
        )
        ctx = _make_ctx(identity_claim="cognito:groups", identity_map=None)
        with pytest.raises(IdentityError):
            _get_db_role(ctx, requested_role="admin")

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_keycloak_realm_roles_flow(self, mock_get_token):
        """Keycloak-style nested claim + literal map — YSQL parity example."""
        mock_get_token.return_value = _make_access_token(
            {"realm_access": {"roles": ["app-writer"]}}
        )
        entries = [_literal("default", "app-writer", "writer")]
        ctx = _make_ctx(
            identity_claim="realm_access.roles",
            identity_map=entries,
            identity_map_name="default",
        )
        assert _get_db_role(ctx) == "writer"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_azure_group_guid_flow(self, mock_get_token):
        """Azure AD GUID → readable role via regex map — closes DB-22174."""
        mock_get_token.return_value = _make_access_token(
            {"groups": ["a12d04b1-7463-8e23-94d2-8d71f17ab99b"]}
        )
        entries = [_regex("azure", r"^a12d04b1-.+", "reader")]
        ctx = _make_ctx(
            identity_claim="groups", identity_map=entries, identity_map_name="azure",
        )
        assert _get_db_role(ctx) == "reader"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_map_filters_unmapped_list_values(self, mock_get_token):
        """List claim with some values mapped, some not — unmapped values are
        dropped, mapped values become the candidate list."""
        mock_get_token.return_value = _make_access_token(
            {"groups": ["known", "unknown-1", "also-known", "unknown-2"]}
        )
        entries = [
            _literal("default", "known", "role_a"),
            _literal("default", "also-known", "role_b"),
        ]
        ctx = _make_ctx(
            identity_claim="groups", identity_map=entries, identity_map_name="default",
        )
        assert _get_db_role(ctx, requested_role="role_b") == "role_b"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_map_all_values_unmapped_raises(self, mock_get_token):
        """DB-22135: if the map has no entry for any of the claim's values,
        raise instead of falling through to the raw claim."""
        mock_get_token.return_value = _make_access_token(
            {"groups": ["unknown-a", "unknown-b"]}
        )
        entries = [_literal("default", "different_value", "some_role")]
        ctx = _make_ctx(
            identity_claim="groups", identity_map=entries, identity_map_name="default",
        )
        with pytest.raises(IdentityError, match="None of the identity-claim values"):
            _get_db_role(ctx)

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_map_with_scalar_claim(self, mock_get_token):
        """Non-list claim (e.g. `email`) + map file — same lookup path."""
        mock_get_token.return_value = _make_access_token(
            {"email": "carol@yugabyte.com"}
        )
        entries = [_regex("default", r"^([a-z]+)@yugabyte\.com$", r"\1")]
        ctx = _make_ctx(identity_claim="email", identity_map=entries)
        assert _get_db_role(ctx) == "carol"


# ---------------------------------------------------------------------------
# Fail-closed on malformed / missing map files
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_map_file_nonexistent_raises_at_load(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _load_identity_map(str(tmp_path / "does-not-exist.conf"))

    def test_map_file_malformed_regex_raises_at_load(self, map_file_factory):
        path = map_file_factory("map1 /[open_bracket_never_closed reader\n")
        with pytest.raises(ValueError):
            _load_identity_map(path)

    def test_map_file_bad_field_count_raises_at_load(self, map_file_factory):
        path = map_file_factory("map1  only_two_fields\n")
        with pytest.raises(ValueError):
            _load_identity_map(path)


# ---------------------------------------------------------------------------
# Security invariants
# ---------------------------------------------------------------------------

class TestSecurityInvariants:
    """Invariants that must hold regardless of config — a violation would be
    a security regression."""

    def test_role_name_with_sql_injection_chars_passes_through(self):
        """`_get_db_role` returns raw role names — SQL quoting happens at
        the SET ROLE site via psycopg.sql.Identifier. Verify the string
        makes it through unchanged so the Identifier layer can quote it."""
        entries = [_literal("default", "evil", "writer'; DROP TABLE t--")]
        with patch("yugabytedb_mcp_server.tools.get_access_token") as mock:
            mock.return_value = _make_access_token({"sub": "evil"})
            ctx = _make_ctx(identity_claim="sub", identity_map=entries)
            # Whatever comes back from the map goes to psycopg's Identifier
            # for quoting — this test asserts we don't accidentally strip or
            # sanitize the value here (that would mask a downstream quoting
            # regression).
            assert _get_db_role(ctx) == "writer'; DROP TABLE t--"

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_empty_list_claim_raises_not_pool_fallback(self, mock_get_token):
        """A token WITH a claim resolving to [] must raise IdentityError, not
        fall back to pool credentials (that would be privilege escalation)."""
        mock_get_token.return_value = _make_access_token({"groups": []})
        ctx = _make_ctx(identity_claim="groups", identity_map=None)
        with pytest.raises(IdentityError):
            _get_db_role(ctx)

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_list_with_only_empty_strings_raises(self, mock_get_token):
        """[""] filters to [] and raises — no ambiguous fallback."""
        mock_get_token.return_value = _make_access_token({"groups": ["", ""]})
        ctx = _make_ctx(identity_claim="groups", identity_map=None)
        with pytest.raises(IdentityError):
            _get_db_role(ctx)

    @patch("yugabytedb_mcp_server.tools.get_access_token")
    def test_requested_role_not_in_mapped_candidates_raises(self, mock_get_token):
        """Even if `requested_role` matches a raw claim value, it must be
        in the *mapped* candidate list — the map is the allowlist."""
        mock_get_token.return_value = _make_access_token(
            {"groups": ["raw_a", "raw_b"]}
        )
        entries = [
            _literal("default", "raw_a", "mapped_a"),
            _literal("default", "raw_b", "mapped_b"),
        ]
        ctx = _make_ctx(identity_claim="groups", identity_map=entries)
        # `requested_role="raw_a"` is the RAW claim value; the candidate list
        # contains the *mapped* values, so this must fail.
        with pytest.raises(IdentityError):
            _get_db_role(ctx, requested_role="raw_a")
        # Sanity: the mapped value works.
        assert _get_db_role(ctx, requested_role="mapped_a") == "mapped_a"
