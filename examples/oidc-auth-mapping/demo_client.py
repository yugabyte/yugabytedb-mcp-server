#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "mcp>=1.2",
# ]
# ///
"""One-command MCP client for the identity-mapping tutorial.

Companion to examples/oidc-auth/demo_client.py — same auth-code + PKCE
flow, but drives the v2 identity-mapping path:

  * JWT claim = realm_access.roles  (list-valued, nested)
  * identity map replaces the strip_domain transform
  * for the dual-role user, the tool call passes requested_role so the
    server clamps the ambiguous candidate list to a single pick

Run:

    uv run demo_client.py

Signs you in to Keycloak on 18081 and hits three MCP tools. Try each
user in turn (re-run the script — prompt=login skips cached sessions):

    writer-only@yugabyte.com / Writer123
        realm_access.roles = ["db-writer", "default-roles-yb-mcp-map"]
        map: db-writer → writer (boilerplate dropped) → SET ROLE writer
        INSERT succeeds.

    reader-only@yugabyte.com / Reader123
        realm_access.roles = ["db-reader", "default-roles-yb-mcp-map"]
        map: db-reader → reader → SET ROLE reader
        INSERT is denied at the DB.

    dual-role@yugabyte.com / Dual123
        realm_access.roles = ["db-writer", "db-reader",
                              "default-roles-yb-mcp-map"]
        map: both → ["writer","reader"] — TWO candidates.
        The script picks "writer" via requested_role. INSERT succeeds.
        (Re-run with DEMO_REQUESTED_ROLE=reader to switch to reader.)

Prerequisites: docker compose up in this directory, seed script run
against your YB, MCP server running with YB_MCP_IDENTITY_CLAIM=
realm_access.roles + YB_MCP_IDENTITY_MAP=<path to ident.conf>.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

KEYCLOAK_AUTHORIZE_URL = (
    "http://localhost:18081/realms/yb-mcp-map/protocol/openid-connect/auth"
)
KEYCLOAK_TOKEN_URL = (
    "http://localhost:18081/realms/yb-mcp-map/protocol/openid-connect/token"
)
CLIENT_ID = "yb-mcp-server"
CLIENT_SECRET = "tutorial-secret-not-for-prod"
MCP_URL = "http://localhost:8000/mcp"
CALLBACK_HOST = "127.0.0.1"
# Distinct from oidc-auth/demo_client.py's 9876 so both scripts can run
# concurrently without fighting over the loopback port.
CALLBACK_PORT = 9877
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"

_received: dict[str, str | None] = {"code": None, "state": None, "error": None}


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _received["code"] = (params.get("code") or [None])[0]
        _received["state"] = (params.get("state") or [None])[0]
        _received["error"] = (params.get("error") or [None])[0]
        body = (
            "<html><body style='font-family: sans-serif; max-width: 480px;"
            " margin: 4em auto; color: #202124;'>"
            "<h2>Signed in.</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            "</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args, **kwargs):  # silence default access log
        pass


def authorization_code_flow() -> str:
    """Drive the auth-code-with-PKCE flow and return the access token."""
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    state = secrets.token_urlsafe(16)

    server = http.server.HTTPServer(
        (CALLBACK_HOST, CALLBACK_PORT), _CallbackHandler
    )
    listener = threading.Thread(target=server.handle_request, daemon=True)
    listener.start()

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "openid email profile",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
    }
    authorize_url = f"{KEYCLOAK_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    print("==> Opening browser for Keycloak sign-in...")
    print(f"    (if your browser does not open, visit:\n     {authorize_url}\n    )")
    webbrowser.open(authorize_url)

    listener.join(timeout=300)
    server.server_close()

    if _received["error"]:
        sys.exit(f"Keycloak returned an error: {_received['error']}")
    if not _received["code"]:
        sys.exit("Timed out waiting for the OAuth callback.")
    if _received["state"] != state:
        sys.exit("State mismatch — possible CSRF.")

    with httpx.Client(timeout=10) as http_:
        r = http_.post(
            KEYCLOAK_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": _received["code"],
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code_verifier": code_verifier,
            },
        )
    if r.status_code != 200:
        sys.exit(f"Token exchange failed ({r.status_code}): {r.text}")
    return r.json()["access_token"]


def decode_claims(access_token: str) -> dict:
    """Best-effort decode of the JWT payload without signature check."""
    payload_b64 = access_token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64))


def summarize_identity(claims: dict) -> tuple[str, list[str]]:
    """Return (email, list-of-realm-roles) so the caller can print a preview."""
    email = claims.get("email") or claims.get("preferred_username") or "<unknown>"
    realm_access = claims.get("realm_access") or {}
    roles = list(realm_access.get("roles") or [])
    return email, roles


def render(result) -> None:
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            print(json.dumps(json.loads(text), indent=2))
        except (json.JSONDecodeError, TypeError):
            print(text)


async def run_mcp(token: str, claims: dict) -> None:
    email, realm_roles = summarize_identity(claims)
    print(f"==> Signed in as {email}")
    print(f"    JWT realm_access.roles: {realm_roles}")

    # Figure out which mapped candidates the server will see. This mirrors
    # ident.conf so what the tutorial prints matches what the server does.
    map_from_realm = {"db-writer": "writer", "db-reader": "reader"}
    mapped = [map_from_realm[r] for r in realm_roles if r in map_from_realm]
    print(f"    Server-side mapped candidates: {mapped}")

    ambiguous = len(mapped) > 1
    requested_role: str | None = None
    if ambiguous:
        requested_role = os.environ.get("DEMO_REQUESTED_ROLE", "writer")
        print(
            f"    Two candidates → passing requested_role={requested_role!r}"
            " on each tool call."
        )
    print()

    tool_args_common: dict[str, str] = {}
    if requested_role is not None:
        tool_args_common["requested_role"] = requested_role

    # The new `streamable_http_client` API takes an `httpx.AsyncClient`
    # for header / auth configuration rather than accepting `headers=` as
    # a kwarg (the old `streamablehttp_client` shape was deprecated in
    # mcp SDK v1.19+ in favor of this).
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(headers=headers) as http_client:
        # mcp>=2.0 yields (read_stream, write_stream); v1.x yielded a third
        # `get_session_id` callback. `*_` accepts both so the tutorial works
        # against whichever version uv resolves.
        async with streamable_http_client(MCP_URL, http_client=http_client) as (
            read_stream,
            write_stream,
            *_,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = sorted(t.name for t in tools.tools)
                print(f"==> Connected to MCP server. Tools: {', '.join(tool_names)}\n")

                print("--- 1. Confirm effective database role ---")
                render(
                    await session.call_tool(
                        "run_read_only_query",
                        {
                            "query": (
                                "SELECT current_user, session_user, "
                                "current_setting('role') AS effective_role"
                            ),
                            **tool_args_common,
                        },
                    )
                )

                print("\n--- 2. SELECT from notes ---")
                render(
                    await session.call_tool(
                        "run_read_only_query",
                        {
                            "query": "SELECT id, body FROM notes ORDER BY id",
                            **tool_args_common,
                        },
                    )
                )

                picked = requested_role or (mapped[0] if mapped else "<none>")
                print(f"\n--- 3. INSERT into notes as {picked} ---")
                render(
                    await session.call_tool(
                        "run_write_query",
                        {
                            "query": (
                                "INSERT INTO notes (body) VALUES "
                                f"('hello from {picked} (mapping demo)')"
                            ),
                            **tool_args_common,
                        },
                    )
                )


def main() -> None:
    token = authorization_code_flow()
    claims = decode_claims(token)
    asyncio.run(run_mcp(token, claims))


if __name__ == "__main__":
    main()
