"""Unit tests for the MCP-spec compliance middlewares in server._run_http().

Mirrors the three middlewares ported from meko-mcp-server PR #97:

1. reject_null_id_requests   – rejects JSON-RPC messages with "id": null (400)
2. OriginValidationMiddleware – rejects requests with disallowed Origin (403)
3. WWWAuthenticateScopeMiddleware – injects scope= into WWW-Authenticate on 401

Plus an AST check that json_response is never passed to http_app() (PR #97
change #1: enables SSE streaming + strict Accept validation).

Each test builds a thin FastAPI app that exercises the middleware in
isolation — no DB, no auth provider, no full MCP server stack.
"""
import json
import os
import sys
import pathlib

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

# Make the yugabytedb_mcp_server package importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the middlewares under test directly from the package so the tests
# exercise the real implementations rather than a duplicated copy.
from yugabytedb_mcp_server.server import (  # noqa: E402
    OriginValidationMiddleware,
    WWWAuthenticateScopeMiddleware,
)


# ===================================================================
# Helpers — tiny apps that wrap each middleware
# ===================================================================


def _app_with_null_id_guard() -> FastAPI:
    """FastAPI app that reproduces the reject_null_id_requests middleware
    from server._run_http(). Kept inline so the test doesn't need to spin up
    the full server."""
    app = FastAPI()

    @app.middleware("http")
    async def reject_null_id_requests(request: Request, call_next):
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

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        return JSONResponse({"ok": True})

    @app.get("/mcp")
    async def mcp_get():
        return JSONResponse({"ok": True})

    @app.post("/other")
    async def other_endpoint(request: Request):
        return JSONResponse({"ok": True})

    return app


def _app_with_origin_validator(allowed_origins: set[str]) -> FastAPI:
    """FastAPI app with the real OriginValidationMiddleware mounted."""
    app = FastAPI()
    app.add_middleware(OriginValidationMiddleware, allowed_origins=allowed_origins)

    @app.post("/mcp")
    async def mcp_endpoint():
        return JSONResponse({"ok": True})

    return app


def _app_with_scope_injection(scope_str: str) -> FastAPI:
    """FastAPI app with WWWAuthenticateScopeMiddleware and an endpoint that
    returns 401 with a bare WWW-Authenticate header."""
    app = FastAPI()
    app.add_middleware(WWWAuthenticateScopeMiddleware, scope_param=scope_str)

    @app.post("/mcp")
    async def mcp_401():
        return JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
        )

    @app.get("/ok")
    async def ok_200():
        return JSONResponse({"ok": True})

    return app


# ===================================================================
# 1. reject_null_id_requests
# ===================================================================


class TestRejectNullId:
    @pytest.fixture()
    def client(self):
        return TestClient(_app_with_null_id_guard())

    def test_null_id_returns_400(self, client):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "initialize", "id": None},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["error"]["code"] == -32600
        assert "null" in body["error"]["message"].lower()

    def test_valid_string_id_passes(self, client):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "initialize", "id": "abc"},
        )
        assert resp.status_code == 200

    def test_valid_integer_id_passes(self, client):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
        )
        assert resp.status_code == 200

    def test_notification_without_id_passes(self, client):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert resp.status_code == 200

    def test_id_zero_passes(self, client):
        """id=0 is a valid integer id, not null."""
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "initialize", "id": 0},
        )
        assert resp.status_code == 200

    def test_get_request_not_affected(self, client):
        resp = client.get("/mcp")
        assert resp.status_code == 200

    def test_malformed_json_passes_through(self, client):
        resp = client.post(
            "/mcp",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200

    def test_batch_with_null_id_element_passes(self, client):
        """Batch requests (JSON arrays) are not intercepted — the SDK
        validates batches itself."""
        resp = client.post(
            "/mcp",
            json=[{"jsonrpc": "2.0", "method": "initialize", "id": None}],
        )
        assert resp.status_code == 200

    def test_non_mcp_post_not_intercepted(self, client):
        """POST to a non-/mcp path should not be parsed or rejected."""
        resp = client.post(
            "/other",
            json={"jsonrpc": "2.0", "method": "initialize", "id": None},
        )
        assert resp.status_code == 200


# ===================================================================
# 2. OriginValidationMiddleware
# ===================================================================


class TestValidateOrigin:
    @pytest.fixture()
    def client(self):
        allowed = {"https://app.example.com", "http://localhost:8000"}
        return TestClient(_app_with_origin_validator(allowed))

    @pytest.fixture()
    def unlocked_client(self):
        """Empty allowlist — middleware should be a no-op."""
        return TestClient(_app_with_origin_validator(set()))

    def test_allowed_origin_passes(self, client):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={"Origin": "https://app.example.com"},
        )
        assert resp.status_code == 200

    def test_disallowed_origin_returns_403(self, client):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status_code == 403
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["error"]["code"] == -32600
        assert "origin" in body["error"]["message"].lower()

    def test_no_origin_header_passes(self, client):
        """Non-browser clients (curl, SDK) don't send Origin."""
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        )
        assert resp.status_code == 200

    def test_localhost_origin_passes(self, client):
        resp = client.post(
            "/mcp",
            json={},
            headers={"Origin": "http://localhost:8000"},
        )
        assert resp.status_code == 200

    def test_trailing_slash_normalized(self, client):
        """An Origin with a trailing slash should match its trimmed form
        in the allowlist."""
        resp = client.post(
            "/mcp",
            json={},
            headers={"Origin": "https://app.example.com/"},
        )
        assert resp.status_code == 200

    def test_response_body_is_jsonrpc_error(self, client):
        resp = client.post(
            "/mcp",
            json={},
            headers={"Origin": "https://dns-rebind.attacker.com"},
        )
        body = resp.json()
        assert "id" not in body, "403 error response should have no id (per MCP spec §2.1)"
        assert body["error"]["code"] == -32600

    def test_empty_allowlist_disables_enforcement(self, unlocked_client):
        """When MCP_ALLOWED_ORIGINS is unset and MCP_BASE_URL is unset,
        no enforcement happens — any Origin passes."""
        resp = unlocked_client.post(
            "/mcp",
            json={},
            headers={"Origin": "https://anywhere.invalid"},
        )
        assert resp.status_code == 200


# ===================================================================
# 3. WWWAuthenticateScopeMiddleware
# ===================================================================


class TestInjectWWWAuthScope:
    @pytest.fixture()
    def client(self):
        return TestClient(_app_with_scope_injection("openid email profile"))

    def test_401_gets_scope_injected(self, client):
        resp = client.post("/mcp", json={})
        assert resp.status_code == 401
        www_auth = resp.headers["www-authenticate"]
        assert 'scope="openid email profile"' in www_auth

    def test_401_preserves_original_challenge(self, client):
        resp = client.post("/mcp", json={})
        www_auth = resp.headers["www-authenticate"]
        assert www_auth.startswith('Bearer realm="mcp"')

    def test_non_401_not_modified(self, client):
        resp = client.get("/ok")
        assert resp.status_code == 200
        assert "www-authenticate" not in resp.headers

    def test_scope_not_duplicated(self):
        """If the header already contains scope=, don't append again."""
        app = FastAPI()
        app.add_middleware(WWWAuthenticateScopeMiddleware, scope_param="openid")

        @app.post("/mcp")
        async def already_scoped():
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp", scope="existing"'},
            )

        client = TestClient(app)
        resp = client.post("/mcp", json={})
        www_auth = resp.headers["www-authenticate"]
        assert www_auth.count("scope=") == 1, "scope should not be duplicated"
        assert 'scope="existing"' in www_auth


# ===================================================================
# 4. json_response=True removal (structural check via AST)
# ===================================================================


class TestJsonResponseDisabled:
    """Verify that the server no longer passes json_response=True to
    mcp.http_app(), which would break SSE streaming and relax Accept
    header validation."""

    def test_http_app_call_has_no_json_response(self):
        import ast

        src_path = PROJECT_ROOT / "src" / "yugabytedb_mcp_server" / "server.py"
        with open(src_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_http_app = (
                isinstance(func, ast.Attribute) and func.attr == "http_app"
            )
            if not is_http_app:
                continue
            for kw in node.keywords:
                assert kw.arg != "json_response", (
                    "mcp.http_app() must not pass json_response=True — "
                    "it disables SSE streaming and relaxes Accept validation "
                    "(see MCP Transports § 2.1 #5)"
                )
