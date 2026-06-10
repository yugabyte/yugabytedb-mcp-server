# OIDC authentication tutorial — with per-user database roles

This tutorial walks you through running `yugabytedb-mcp-server` behind a generic OIDC identity provider AND mapping each OIDC user to a distinct Postgres role, so the database itself enforces what each user can read or write. We use **Keycloak** (open-source, self-hosted, Docker-based) so the whole setup runs on your laptop without any third-party signup. The same configuration pattern works against Auth0, Okta, Azure AD, Google Identity, or any RFC 6749 / OIDC 1.0 compliant provider — only the connection details change.

## What you'll prove by the end

> Two Keycloak users (`writer@yugabyte.com`, `reader@yugabyte.com`) hit the **same MCP server, same connection pool, same `run_write_query` tool**. `writer` can INSERT into the `notes` table. `reader` cannot — the database itself rejects the write. The difference is purely the OIDC bearer token, which drives a different `SET ROLE` per request.

## Architecture

```
┌──────────────────┐   1. browser OAuth   ┌──────────────────┐
│ MCP client       │ ───────────────────► │ Keycloak         │
│ (demo_client.py, │                      │ http://:18080    │
│  Inspector,      │   2. Streamable HTTP │ realm: yb-mcp    │
│  Cursor,         │ ◄───── bearer ──────►│ writer / reader  │
│  Claude...)      │                      │                  │
└──────────────────┘                      └──────────────────┘
         │
         ▼
┌────────────────────────────────────┐         ┌──────────────────┐
│ yugabytedb-mcp-server              │         │ YugabyteDB       │
│ http://:8000  transport=http       │ ──SQL──►│                  │
│ MCP_AUTH_PROVIDER=oidc             │  +SET   │  role: writer    │
│ YB_MCP_IDENTITY_CLAIM=email        │  ROLE   │  role: reader    │
│ YB_MCP_IDENTITY_TRANSFORM=         │         │  GRANTs decide   │
│   strip_domain                     │         │  who can write   │
└────────────────────────────────────┘         └──────────────────┘
```

The MCP server's `OIDCProxy` sits between your MCP client and Keycloak. Clients see a standard OAuth 2.0 surface (`/authorize`, `/token`, `/register`) at the MCP server's URL; the proxy translates those into upstream Keycloak calls. Once a token is validated, the server extracts the email claim, strips the domain, and uses the local-part (`writer` / `reader`) as the Postgres role to `SET ROLE` into for the duration of that tool call.

## Prerequisites

- **Docker** (Engine ≥ 20.x) — to run Keycloak
- **`uv`** — to run the MCP server (`brew install uv`, `pip install uv`, or [docs.astral.sh/uv](https://docs.astral.sh/uv/))
- A reachable **YugabyteDB** instance — local `yugabyted start` works fine. PostgreSQL also works; the tutorial uses `ysqlsh` for the seed step but `psql` is a drop-in substitute.
- 10–15 minutes

## Step 1 — Start Keycloak

From this directory:

```bash
docker compose up -d
```

Wait ~20 seconds for Keycloak to boot and import the realm. Verify:

```bash
curl -fs http://localhost:18080/realms/yb-mcp/.well-known/openid-configuration | head -c 200
```

You should see JSON describing the issuer, authorization endpoint, token endpoint, etc.

What just happened:
- Keycloak 26 booted in dev mode on port `18080`
- The `yb-mcp` realm was imported from `keycloak/realm-export.json`
- A confidential client `yb-mcp-server` was registered with secret `tutorial-secret-not-for-prod`
- **Two** test users were created:
  - `writer@yugabyte.com` / `Writer123`
  - `reader@yugabyte.com` / `Reader123`

### View the realm in the Keycloak admin console (optional)

If you want to inspect the imported realm, browse users, or look at token claims, open the Keycloak admin console:

| Field | Value |
|---|---|
| URL | <http://localhost:18080> |
| Username | `admin` |
| Password | `admin` |

After signing in, use the realm dropdown in the top-left to switch from **master** to **yb-mcp**. The tutorial itself doesn't require the admin console.

## Step 2 — Seed the database

The MCP server side of the role mapping is "extract the claim, `SET ROLE <claim>`". The database side has to actually have those roles and the right GRANTs on them. `postgres-seed.sql` does both. Run `ysqlsh` from your YugabyteDB install and pass the seed script's absolute path:

```bash
./path/to/yugabytedb/installation/bin/ysqlsh \
    -v yb_pool_user=yugabyte \
    -f /path/to/local/yugabytedb-mcp-server/examples/oidc-auth/postgres-seed.sql
```

Substitute the two paths to match your machine:

- `./path/to/yugabytedb/installation/bin/ysqlsh` — the `ysqlsh` binary that ships with your YugabyteDB install. On Postgres-only systems, use `psql` instead — same flags.
- `/path/to/local/yugabytedb-mcp-server/examples/oidc-auth/postgres-seed.sql` — the seed script's absolute path on your machine.

Replace `yugabyte` with whichever username is in your `YUGABYTEDB_URL` if different. `ysqlsh` picks up connection details from the `YUGABYTEDB_URL` / `PG*` environment, or you can pass them inline (`-h`, `-p`, `-U`, `-d`). The script is idempotent; safe to re-run.

What the seed sets up:

| Object | Created with |
|---|---|
| Postgres role `writer` (NOLOGIN) | `CREATE ROLE writer NOLOGIN` |
| Postgres role `reader` (NOLOGIN) | `CREATE ROLE reader NOLOGIN` |
| Table `notes(id, body, created_at)` | `CREATE TABLE IF NOT EXISTS notes …` |
| `writer` GRANTs | `GRANT SELECT, INSERT, UPDATE, DELETE ON notes TO writer` + sequence access |
| `reader` GRANT | `GRANT SELECT ON notes TO reader` |
| Pool user membership | `GRANT writer TO :"yb_pool_user"` + same for reader |

### What's the "pool user" and why does membership matter?

The MCP server keeps a connection pool open to YugabyteDB. The pool authenticates as **the database account inside `YUGABYTEDB_URL`** — in this tutorial that's `yugabyte`. We call that account the "pool user" for short; it's just a label, not a Keycloak or Postgres-specific term. It's the only DB identity the server *natively* has.

When an OIDC user calls a tool, the server runs `SET ROLE <oidc_user>` on a pooled connection. Postgres only allows that switch if the **currently-authenticated** account (the pool user) is a member of the target role:

```text
1. Pool connects as user `yugabyte`.
2. Tool call arrives; OIDC token says reader@yugabyte.com → role `reader`.
3. Server runs:  SET ROLE reader
4. Postgres checks: "Is `yugabyte` a member of role `reader`?"
   YES (because of `GRANT reader TO yugabyte`)  → role switch succeeds
   NO                                            → permission denied to set role "reader"
```

The `GRANT writer TO :"yb_pool_user"` / `GRANT reader TO :"yb_pool_user"` lines at the bottom of `postgres-seed.sql` are what wire that up. Without them, every tool call from a real OIDC user fails — the MCP server keeps running but can't switch roles. **Skip the membership grants and the entire role-mapping demo doesn't work.**

(In some YugabyteDB / Postgres setups the default `yugabyte` user is a superuser, in which case `SET ROLE` works for any role even without an explicit GRANT. Convenient for tutorials but defeats the point in production — you'd want the pool user to be a least-privileged account whose only privileges are membership in the OIDC-mapped roles.)

Confirm the seed by reading its sanity-check output: you should see the two roles listed, the five GRANTs, and two membership rows.

## Step 3 — Configure the MCP server

From the **repo root** (one level up from this folder):

```bash
cp examples/oidc-auth/.env.example .env
# Edit YUGABYTEDB_URL in .env to point at the database you just seeded
```

Notable settings in `.env`:

| Variable | Value | Why |
|---|---|---|
| `MCP_AUTH_PROVIDER` | `oidc` | Generic OIDC factory (not Cognito-specific). |
| `OIDC_CONFIG_URL` | Keycloak's discovery URL | Lets `OIDCProxy` auto-discover authorize / token / jwks endpoints. |
| `YB_MCP_IDENTITY_CLAIM` | `email` | JWT claim used as the role identifier. |
| `YB_MCP_IDENTITY_TRANSFORM` | `strip_domain` | Turns `writer@yugabyte.com` → `writer`. |
| `YB_MCP_ENABLE_WRITE_QUERY` | `true` | Enable `run_write_query` so the demo can show that DB-side GRANTs reject the wrong user. **In production, leave this unset — the write tool is disabled by default for a reason.** |
| `MCP_ALLOWED_ORIGINS` | `http://localhost:8000,http://localhost:6274` | DNS-rebinding defense. The server rejects browser requests whose `Origin` header isn't in this list. `http://localhost:8000` is the server itself; `http://localhost:6274` is where MCP Inspector serves its UI when launched via `npx`. Without the second entry, Inspector's connect attempt in Step 5 returns 403. |

Start the server:

```bash
set -a; source .env; set +a
uv run yugabytedb-mcp --transport http
```

(Or `uvx yugabytedb-mcp-server --transport http` if you've installed the published package — both run the same entry point.)

You should see in the logs:

```
[INFO] yugabytedb-mcp.auth: Creating auth provider: oidc
[INFO] yugabytedb-mcp.auth: OIDC auth provider created (config_url=http://localhost:18080/realms/yb-mcp/.well-known/openid-configuration)
[INFO] yugabytedb-mcp.server: Per-user SET ROLE enabled (claim=email, transform=strip_domain). ...
[INFO] yugabytedb-mcp.server: run_write_query tool enabled
[INFO] yugabytedb-mcp.server: Origin allowlist: http://localhost:8000
[INFO] yugabytedb-mcp.server: WWW-Authenticate scope injection enabled (scope=openid email profile)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Two key proof points:
- `Per-user SET ROLE enabled (claim=email, transform=strip_domain)` confirms the role-mapping plumbing is on.
- `run_write_query tool enabled` confirms the write tool is registered. (Without `YB_MCP_ENABLE_WRITE_QUERY=true`, the log would read `run_write_query tool disabled` and the tool wouldn't appear in `tools/list`.)

Auth is now enforced. Confirm with curl:

```bash
# /ping is unauthenticated → 200
curl -i http://localhost:8000/ping

# /mcp without a token → 401 + WWW-Authenticate header
# (note: /mcp only accepts POST/DELETE — a GET will return 405,
#  so we send a real initialize POST here)
curl -i -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
```

## Step 4 — Run the one-command demo client

The fastest way to see the full demo work is the bundled `demo_client.py`. It walks the full browser-based authorization-code OAuth flow — the same flow Claude Desktop, Cursor, and MCP Inspector use — and then exercises three MCP tool calls.

From this directory:

```bash
uv run demo_client.py
```

The first run takes a few extra seconds while `uv` resolves and caches the script's dependencies (`httpx` and the `mcp` Python SDK, declared inline in the script header). Subsequent runs are instant.

### What happens

1. The script starts a local OAuth callback listener on `http://localhost:9876/callback`.
2. It opens your browser to Keycloak's sign-in page.
3. You enter **either** user's credentials:

   | Username or email | Password | Outcome |
   |---|---|---|
   | `reader@yugabyte.com` | `Reader123` | Reads succeed, writes are rejected by Postgres. |
   | `writer@yugabyte.com` | `Writer123` | Reads and writes both succeed. |

4. Keycloak redirects back to the callback listener with an authorization code.
5. The script exchanges the code for an access token, opens a Streamable HTTP MCP session against `http://localhost:8000/mcp`, and runs three demo calls.

### Expected output — signed in as `reader`

```
==> Opening browser for Keycloak sign-in...
==> Signed in as reader@yugabyte.com

==> Connected to MCP server. Tools: run_read_only_query, run_write_query, summarize_database

--- 1. Confirm effective database role ---
[{"current_user": "reader", "session_user": "yugabyte", "effective_role": "reader"}]

--- 2. SELECT from notes ---
[{"id": 1, "body": "hello, world (seeded)"}, {"id": 2, "body": "reader can see this"}]

--- 3. INSERT into notes as reader ---
{"error": "permission denied for table notes"}
```

### Expected output — signed in as `writer`

```
==> Opening browser for Keycloak sign-in...
==> Signed in as writer@yugabyte.com

==> Connected to MCP server. Tools: run_read_only_query, run_write_query, summarize_database

--- 1. Confirm effective database role ---
[{"current_user": "writer", "session_user": "yugabyte", "effective_role": "writer"}]

--- 2. SELECT from notes ---
[{"id": 1, "body": "hello, world (seeded)"}, {"id": 2, "body": "reader can see this"}]

--- 3. INSERT into notes as writer ---
{"rows_affected": 1}
```

Same MCP server, same connection pool, same tool catalog. Only the OIDC bearer changes, and with it the effective Postgres role — and therefore the GRANT decision.

### Re-run as the other user

Just run the script again. The OAuth request includes `prompt=login`, which tells Keycloak to ignore any cached session and always show the login form — no manual logout step needed.

```bash
uv run demo_client.py
```

## Step 5 — (Optional) Same thing through MCP Inspector

The MCP Inspector is a browser-based MCP client with a tool-execution UI. Step 4's `demo_client.py` already exercises the full role-mapping behaviour end-to-end; Inspector is useful if you want to poke at the tools interactively or experiment with arbitrary queries.

### 5.1 Launch and connect

```bash
npx @modelcontextprotocol/inspector
```

A browser tab opens at `http://localhost:6274`. In the connection panel, set:

| Field | Value |
|---|---|
| Transport | `Streamable HTTP` |
| URL | `http://localhost:8000/mcp` |

Click **Connect**. Inspector opens a Keycloak login tab; sign in as either user:

| Username or email | Password |
|---|---|
| `reader@yugabyte.com` | `Reader123` |
| `writer@yugabyte.com` | `Writer123` |

Then click **Allow Access**. The **Tools** panel will list the three MCP tools (`summarize_database`, `run_read_only_query`, `run_write_query`) and you can call them with arbitrary SQL. The same role-mapping behaviour from Step 4 applies — reads work for both users; writes succeed only for `writer`.

> **Troubleshooting — 403 `invalid_token` instead of the login page.** Inspector serves its UI from `http://localhost:6274`. That origin must be present in `MCP_ALLOWED_ORIGINS` or the MCP server's DNS-rebinding defense will reject Inspector's requests. The shipped `.env.example` includes it; if you're starting from a different config, add `http://localhost:6274` to `MCP_ALLOWED_ORIGINS` and restart the server.

### 5.2 Disconnect properly when done

Inspector's **Disconnect** button drops Inspector's stored token, but it does **not** clear the Keycloak session cookie in your browser. To fully sign out (so the next connect attempt shows the login form again):

1. In Inspector, click **Disconnect**.
2. In the same browser, visit Keycloak's logout URL:

   ```
   http://localhost:18080/realms/yb-mcp/protocol/openid-connect/logout
   ```

   Confirm the logout when prompted.

If you skip step 2, clicking **Connect** again will silently sign you back in as the previous user. Steps 1 and 2 together give you a clean slate so you can sign in as the other user.

## Step 6 — (Optional) The same thing via curl

If you'd rather drive the server from a shell — for CI smoke tests or just to see the wire protocol — you can fetch tokens via Keycloak's **direct-grant (password) OAuth flow** and call tools with `curl`. Unlike the browser-based flow used by `demo_client.py` (Step 4) and Inspector (Step 5), the password grant trades a username/password directly for a token without any redirect. The shipped realm has `directAccessGrantsEnabled: true` on the `yb-mcp-server` client so this works out of the box. **Never enable that flag in production** — real clients should use the browser-based flow.

### Fetch tokens for both users

```bash
get_token() {
  curl -sf -X POST http://localhost:18080/realms/yb-mcp/protocol/openid-connect/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=password&client_id=yb-mcp-server&client_secret=tutorial-secret-not-for-prod&username=$1&password=$2&scope=openid email profile" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])"
}
WRITER_TOKEN=$(get_token writer@yugabyte.com Writer123)
READER_TOKEN=$(get_token reader@yugabyte.com Reader123)
```

### Call tools via curl

The wire format is JSON-RPC over Streamable HTTP. Three examples — the same kind of calls `demo_client.py` makes in Step 4:

```bash
# reader: who am I?
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer $READER_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc":"2.0","id":1,
    "method":"tools/call",
    "params":{
      "name":"run_read_only_query",
      "arguments":{"query":"SELECT current_user, session_user"}
    }
  }'

# writer: INSERT succeeds
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer $WRITER_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc":"2.0","id":2,
    "method":"tools/call",
    "params":{
      "name":"run_write_query",
      "arguments":{"query":"INSERT INTO notes (body) VALUES ('"'"'hello from writer (curl)'"'"')"}
    }
  }'

# reader: same INSERT, permission denied
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer $READER_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc":"2.0","id":3,
    "method":"tools/call",
    "params":{
      "name":"run_write_query",
      "arguments":{"query":"INSERT INTO notes (body) VALUES ('"'"'hello from reader (curl)'"'"')"}
    }
  }'
```

The first two return `rows_affected: 1` (writer) and the SELECT result (reader). The third returns `{"error": "permission denied for table notes"}` — the same database-level rejection you saw in Step 4, just dressed in JSON-RPC framing instead of script output.

## How it works under the hood

1. **MCP client → MCP server** — client tries to call `/mcp`, gets `401 WWW-Authenticate: Bearer ... resource_metadata="..."`.
2. **MCP client → MCP server `/.well-known/oauth-protected-resource/mcp`** — learns the auth server is at `http://localhost:8000/`.
3. **MCP client → MCP server `/.well-known/oauth-authorization-server`** — learns the registration, authorize, token endpoints.
4. **MCP client → MCP server `/register`** — Dynamic Client Registration (RFC 7591); client gets a `client_id` issued by the MCP server.
5. **MCP client → MCP server `/authorize`** — `OIDCProxy` redirects the browser to **Keycloak's** authorize endpoint with the upstream `client_id` (`yb-mcp-server`) and a callback URL pointing back at the MCP server (`/auth/callback`).
6. **Browser → Keycloak** — user logs in as `writer` or `reader`.
7. **Keycloak → MCP server `/auth/callback`** — Keycloak redirects with an authorization code.
8. **MCP server → Keycloak `/token`** — `OIDCProxy` exchanges the code for an upstream access + refresh token.
9. **MCP server → MCP client** — `OIDCProxy` mints its own JWT for the MCP client (so the client never holds the Keycloak token directly), redirects back to the client's callback URL with that token.
10. **MCP client → MCP server `/mcp`** — calls with `Authorization: Bearer <token>`. JWT verification happens at the MCP server; tool dispatched.
11. **MCP server `_get_db_role()` → `_conn_as_role()`** — extracts the `email` claim, strips the domain, and runs `SET ROLE <claim>` on the pooled connection before the tool's SQL. `RESET ROLE` runs in a `finally` block so the connection returns to the pool clean.

The role switch lives in `src/yugabytedb_mcp_server/tools.py`:

```python
@contextmanager
def _conn_as_role(pool, role: str | None):
    """Acquire a connection and optionally SET ROLE for the duration."""
    with pool.connection() as conn:
        if role is not None:
            with conn.cursor() as cur:
                cur.execute(SQL("SET ROLE {}").format(Identifier(role)))
        try:
            yield conn
        finally:
            if role is not None:
                with conn.cursor() as cur:
                    cur.execute("RESET ROLE")
```

Role names are quoted via `psycopg.sql.Identifier`, so a JWT claim of `'; DROP TABLE notes; --` is safely escaped — it would be rejected by Postgres as a role-not-found rather than executed as SQL.

## Adapting to a different OIDC provider

The same MCP server config works with any OIDC IdP — only the env vars need to change:

| Variable | What to set |
|---|---|
| `OIDC_CONFIG_URL` | URL of the provider's `.well-known/openid-configuration` |
| `OIDC_CLIENT_ID` | Confidential client ID registered in the provider |
| `OIDC_CLIENT_SECRET` | That client's secret |
| `OIDC_AUDIENCE` | Optional. Set if the provider issues access tokens with an `aud` claim you want enforced |
| `YB_MCP_IDENTITY_CLAIM` | Which JWT claim identifies the user. Defaults to `email`. Common alternatives: `preferred_username`, `sub`. |
| `YB_MCP_IDENTITY_TRANSFORM` | `strip_domain` for email claims, `none` for everything else |

The client in your provider must:
- Be **confidential** (have a client secret)
- Allow the **authorization_code** grant
- Allow `client_secret_basic` or `client_secret_post` token endpoint auth
- Allow `PKCE S256` (most providers default to this)
- Have `http://<MCP_BASE_URL>/auth/callback` in its allowed redirect URIs (replace `<MCP_BASE_URL>` with your deployment's public URL)

Provider-specific notes:

- **Auth0** — set `OIDC_CONFIG_URL=https://YOUR_TENANT.us.auth0.com/.well-known/openid-configuration`. Use a "Regular Web Application" client type. Add the callback URL under "Allowed Callback URLs" in the dashboard. `email` is in the ID token by default.
- **Okta** — set `OIDC_CONFIG_URL=https://YOUR_DOMAIN.okta.com/.well-known/openid-configuration` (or the per-app variant). Create an OIDC Web app. May need to request the `email` scope explicitly.
- **Google Identity** — set `OIDC_CONFIG_URL=https://accounts.google.com/.well-known/openid-configuration`. Configure an "OAuth 2.0 Client ID" of type "Web application" in Google Cloud Console.
- **Azure AD / Entra ID** — `yugabytedb-mcp-server` ships a separate `MCP_AUTH_PROVIDER=azure` mode tuned for Entra's quirks. Use that instead of the generic `oidc` mode.
- **AWS Cognito** — same story. `MCP_AUTH_PROVIDER=cognito` handles four Cognito-specific quirks (token-endpoint scope handling, no PKCE on upstream, etc.). Use it instead of `oidc`. Cognito's default user identifier is `sub` (UUID), which is rarely what you want as a database role name — consider a custom claim mapping in the user pool.

## Tearing down

```bash
docker compose down -v   # stops Keycloak and removes the volume
```

Kill the MCP server with Ctrl-C in its terminal. Remove `.env` if you don't want the dev-only credentials lying around. The Postgres roles, table, and GRANTs survive the teardown — drop them explicitly if you don't want them:

```sql
DROP TABLE IF EXISTS notes;
REVOKE writer, reader FROM yugabyte;  -- substitute your pool user
DROP ROLE IF EXISTS writer;
DROP ROLE IF EXISTS reader;
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `404 Not Found` from Keycloak `/realms/yb-mcp/.well-known/openid-configuration` | Realm didn't import. Check `docker compose logs keycloak` for an import error. Often a JSON-syntax problem in `realm-export.json` after edits. |
| MCP server fails to start with `httpx.HTTPStatusError: 500` from the OIDC config URL | Keycloak isn't ready yet. Wait ~10 more seconds and try again. |
| 401 from `/mcp` even with a real Keycloak token | Token's `iss` claim doesn't match `OIDC_CONFIG_URL`'s issuer. Check both — they must be byte-for-byte identical. Common case: trailing slash mismatch, or HTTPS vs HTTP. |
| Browser flow fails with `Invalid redirect URI` from Keycloak | The MCP server is sending `redirect_uri=http://...:8000/auth/callback` and the client config in Keycloak doesn't allowlist that exact URL. Edit `realm-export.json`'s `redirectUris`, restart Keycloak. |
| Inspector says "Connected" but tool calls return 401 | Inspector cached an old token. Disconnect and reconnect — it'll re-run the OAuth flow. |
| Inspector's connect attempt fails with `invalid_token` and server logs show `Rejected request with disallowed Origin: http://localhost:6274` | Inspector's UI origin (`http://localhost:6274`) isn't in `MCP_ALLOWED_ORIGINS`. Add it (`MCP_ALLOWED_ORIGINS=http://localhost:8000,http://localhost:6274`) and restart the MCP server. The shipped `.env.example` already does this. |
| `ERROR: role "reader" does not exist` from a tool call | Seed SQL wasn't run, or was run against the wrong database. Re-run `ysqlsh -v yb_pool_user=… -f postgres-seed.sql` against the same DB the MCP server is configured to use. |
| `ERROR: permission denied to set role "reader"` | Pool user wasn't granted membership in the target role. Run `GRANT reader TO <pool_user>;` (or just re-run the seed script). |
| `run_write_query` not in `tools/list` even though I set `YB_MCP_ENABLE_WRITE_QUERY=true` | The env var didn't make it into the server process. Check the startup logs — you should see `run_write_query tool enabled`. If it says `disabled`, you sourced `.env` after starting the server, or the variable name is mistyped. |

## See also

- **Generic OIDC factory**: `src/yugabytedb_mcp_server/auth.py`, function `_create_oidc()`
- **Role switch implementation**: `src/yugabytedb_mcp_server/tools.py`, functions `_get_db_role()` and `_conn_as_role()`
- **Write-tool gate**: `src/yugabytedb_mcp_server/server.py`, in `_register_tools()`
- **FastMCP OIDCProxy docs**: <https://gofastmcp.com/servers/auth/oidc-proxy>
- Top-level repo README for non-auth-related setup (database connection, tools, etc.)
