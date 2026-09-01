# OIDC identity-mapping tutorial — realm-role claim + map file + `requested_role`

An end-to-end tutorial that proves per-user Postgres roles work with
the OIDC identity-mapping features:

- **List-valued claim** — `realm_access.roles` is a JSON array, not a scalar
- **Dotted-path claim extraction** — the server walks `realm_access` then `roles`
- **Identity map file** — an allowlist that translates Keycloak realm-role names to Postgres role names AND drops Keycloak boilerplate roles (`offline_access`, `default-roles-*`, `uma_authorization`)
- **`requested_role` on ambiguity** — a user whose token maps to more than one Postgres role picks explicitly per tool call

## What you'll prove by the end

> Three Keycloak users hit the **same MCP server, same connection pool, same `run_write_query` tool**. Their tokens carry Keycloak REALM ROLES (not emails). The MCP server runs those role names through a map file — Keycloak boilerplate roles are dropped, the two whitelisted realm roles (`db-writer`, `db-reader`) become Postgres role names (`writer`, `reader`), and the resulting `SET ROLE` decides what each user can do at the database. A user granted BOTH realm roles picks one at query time.

## Architecture

```
┌──────────────────┐  1. browser OAuth   ┌──────────────────┐
│ demo_client.py   │ ──────────────────► │ Keycloak         │
│ (or Inspector,   │                     │ http://:18081    │
│  Cursor, Claude) │  2. Streamable HTTP │ realm: yb-mcp-map│
│                  │ ◄──── bearer ─────► │ realm roles:     │
│                  │                     │  db-writer       │
│                  │                     │  db-reader       │
└──────────────────┘                     └──────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐    ┌──────────────────┐
│ yugabytedb-mcp-server                          │    │ YugabyteDB       │
│ http://:8000  transport=http                   │    │                  │
│ MCP_AUTH_PROVIDER=oidc                         │────►│  role: writer   │
│ YB_MCP_IDENTITY_CLAIM=realm_access.roles       │ SQL │  role: reader   │
│ YB_MCP_IDENTITY_MAP=./ident.conf               │+SET │  GRANTs decide  │
│ YB_MCP_IDENTITY_MAP_NAME=default               │ROLE │  who can write  │
└────────────────────────────────────────────────┘    └──────────────────┘
```

## Prerequisites

- **Docker** (Engine ≥ 20.x) — to run Keycloak
- **`uv`** — to run the MCP server (`brew install uv`, `pip install uv`, or [docs.astral.sh/uv](https://docs.astral.sh/uv/))
- A reachable **YugabyteDB** instance — `yugabyted start` locally is fine; PostgreSQL works too
- 15–20 minutes

## Step 1 — Start Keycloak

From this directory:

```bash
docker compose up -d
```

Wait ~20 seconds for Keycloak to boot and import the realm. Verify:

```bash
curl -fs http://localhost:18081/realms/yb-mcp-map/.well-known/openid-configuration | head -c 200
```

What just happened:
- Keycloak 26 booted in dev mode on port `18081` (the sibling tutorial uses `18080` — both can coexist)
- The `yb-mcp-map` realm was imported from `keycloak/realm-export.json`
- A confidential client `yb-mcp-server` was registered with secret `tutorial-secret-not-for-prod`
- Two realm roles were created — `db-writer`, `db-reader`
- **Three** test users were created:

  | Email | Password | Realm roles granted |
  |---|---|---|
  | `writer-only@yugabyte.com` | `Writer123` | `db-writer` |
  | `reader-only@yugabyte.com` | `Reader123` | `db-reader` |
  | `dual-role@yugabyte.com`   | `Dual123`   | `db-writer` AND `db-reader` |

Every user also carries three Keycloak boilerplate roles — `offline_access`, `default-roles-yb-mcp-map`, `uma_authorization`. This is what the map file's allowlist is for — none of those boilerplate roles reach Postgres.

The realm's audience mapper also stamps `aud: yb-mcp-server` on every access token, so the MCP server's audience validation (v2) accepts them.

### View the realm in the admin console (optional)

Open http://localhost:18081, sign in as `admin` / `admin`, and select the realm yb-mcp-map from the drop down menu on the left side. Under **Realm roles**, you'll see `db-writer` / `db-reader`. Under **Users → <username>**, each of the three demo users has role assignments visible on the **Role mapping** tab.

## Step 2 — Seed the database

If you already ran the sibling tutorial's seed, skip this — the roles are the same. Otherwise:

```bash
ysqlsh "$YUGABYTEDB_URL" -v yb_pool_user=yugabyte -f postgres-seed.sql
```

(Or `psql` if you're on stock PostgreSQL. Substitute `yugabyte` with whichever user appears in your `$YUGABYTEDB_URL`.)

The seed creates two `NOLOGIN` Postgres roles (`writer`, `reader`), a demo `notes` table, appropriate GRANTs (writer can SELECT/INSERT/UPDATE/DELETE, reader can only SELECT), and grants the pool user membership in both — necessary for `SET ROLE` to succeed later.

## Step 3 — Configure the MCP server

From the repo root:

```bash
# Point to your database (same pool user you seeded above).
export YUGABYTEDB_URL="host=localhost port=5433 user=yugabyte dbname=yugabyte"

# Generic OIDC provider — Keycloak on 18081.
export MCP_AUTH_PROVIDER=oidc
export MCP_BASE_URL=http://localhost:8000
export OIDC_CONFIG_URL=http://localhost:18081/realms/yb-mcp-map/.well-known/openid-configuration
export OIDC_CLIENT_ID=yb-mcp-server
export OIDC_CLIENT_SECRET=tutorial-secret-not-for-prod
export OIDC_AUDIENCE=yb-mcp-server

# The mapping bits — the whole point of this tutorial.
export YB_MCP_IDENTITY_CLAIM=realm_access.roles
export YB_MCP_IDENTITY_MAP="$(pwd)/examples/oidc-auth-mapping/ident.conf"
export YB_MCP_IDENTITY_MAP_NAME=default

# Write-query surface is off by default. Turn it on so the tutorial's
# INSERT case works.
export YB_MCP_ENABLE_WRITE_QUERY=true

uv run yugabytedb-mcp --transport http
```

You should see the server log lines confirming both the OIDC provider and the identity map were parsed:

```
INFO   yugabytedb-mcp.auth  oidc provider configured  audience=yb-mcp-server
INFO   yugabytedb-mcp.tools identity map loaded  path=/abs/path/ident.conf name=default entries=2
INFO   yugabytedb-mcp.tools server listening       host=0.0.0.0 port=8000
```

If the identity-map line is missing, `YB_MCP_IDENTITY_MAP` didn't resolve — check the path.

### Smoke-test the auth surface without a token

```bash
# /ping is unauthenticated → 200
curl -si http://localhost:8000/ping | head -1

# /mcp without a token → 401 + WWW-Authenticate header
curl -si -X POST http://localhost:8000/mcp \
     -H 'Content-Type: application/json' \
     -H 'Accept: application/json, text/event-stream' \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}' \
     | head -5
```

## Step 4 — Run the one-command demo client

From this directory:

```bash
uv run demo_client.py
```

The browser opens Keycloak's sign-in page. Pick a user:

- `writer-only@yugabyte.com / Writer123` — single realm role, no `requested_role` needed
- `reader-only@yugabyte.com / Reader123` — single realm role, no `requested_role` needed
- `dual-role@yugabyte.com / Dual123` — two realm roles; the script passes `requested_role="writer"` by default

The script prints the decoded token's `realm_access.roles`, what the server-side map will translate them to, whether `requested_role` will be sent, and then runs three tool calls: `current_user`, `SELECT`, `INSERT`.

### Expected output — signed in as `writer-only`

```
==> Signed in as writer-only@yugabyte.com
    JWT realm_access.roles: ['offline_access', 'default-roles-yb-mcp-map',
                             'uma_authorization', 'db-writer']
    Server-side mapped candidates: ['writer']

==> Connected to MCP server. Tools: run_read_only_query, run_write_query, summarize_database

--- 1. Confirm effective database role ---
{
  "columns": ["current_user", "session_user", "effective_role"],
  "rows": [["writer", "yugabyte", "writer"]]
}

--- 2. SELECT from notes ---
{
  "columns": ["id", "body"],
  "rows": [[1, "hello, world (seeded)"], [2, "reader can see this"]]
}

--- 3. INSERT into notes as writer ---
{ "rows_affected": 1 }
```

The three Keycloak boilerplate roles (`offline_access`, `default-roles-yb-mcp-map`, `uma_authorization`) were in the token but never touched the database — the map dropped them.

### Expected output — signed in as `reader-only`

```
==> Signed in as reader-only@yugabyte.com
    JWT realm_access.roles: ['offline_access', 'default-roles-yb-mcp-map',
                             'uma_authorization', 'db-reader']
    Server-side mapped candidates: ['reader']

--- 1. Confirm effective database role ---
{
  "columns": ["current_user", "session_user", "effective_role"],
  "rows": [["reader", "yugabyte", "reader"]]
}

--- 3. INSERT into notes as reader ---
Error: permission denied for table notes
```

Same MCP server, same pool connection, same tool — the database refused because the token drove `SET ROLE reader` for this request.

### Expected output — signed in as `dual-role`

```
==> Signed in as dual-role@yugabyte.com
    JWT realm_access.roles: ['offline_access', 'default-roles-yb-mcp-map',
                             'uma_authorization', 'db-writer', 'db-reader']
    Server-side mapped candidates: ['writer', 'reader']
    Two candidates → passing requested_role='writer' on each tool call.

--- 1. Confirm effective database role ---
{
  "columns": ["current_user", "session_user", "effective_role"],
  "rows": [["writer", "yugabyte", "writer"]]
}

--- 3. INSERT into notes as writer ---
{ "rows_affected": 1 }
```

Re-run with `DEMO_REQUESTED_ROLE=reader uv run demo_client.py` — the same user's token now drives `SET ROLE reader` and the INSERT is denied.

### Try each user

The OAuth request includes `prompt=login`, so Keycloak always shows the sign-in form and ignores cached sessions. Just re-run the script to switch users.

## Step 5 — (Optional) Same thing via curl

Same idea as the sibling tutorial. Fetch tokens directly via the resource-owner-password grant (dev-only), then hit `/mcp` with the bearer.

```bash
# --- Grab tokens ---
# Note: the yb-mcp-audience client scope is a *default* scope on the client,
# so Keycloak applies it automatically. Requesting it explicitly via `scope=`
# returns invalid_scope. Just ask for the standard OIDC scopes.
DUAL=$(curl -sf -X POST http://localhost:18081/realms/yb-mcp-map/protocol/openid-connect/token \
  -d grant_type=password -d client_id=yb-mcp-server \
  -d client_secret=tutorial-secret-not-for-prod \
  -d username=dual-role@yugabyte.com -d password=Dual123 | jq -r .access_token)

# --- Peek at what the server will see ---
echo "$DUAL" | cut -d. -f2 | tr '_-' '/+' | base64 -d 2>/dev/null | jq '.realm_access'
# {
#   "roles": [
#     "offline_access", "default-roles-yb-mcp-map", "uma_authorization",
#     "db-writer", "db-reader"
#   ]
# }

# --- The ambiguous case: two mapped candidates + explicit requested_role ---
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer $DUAL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
       "name":"run_read_only_query",
       "arguments":{
         "query":"SELECT current_setting('\''role'\'') AS effective_role",
         "requested_role":"reader"
       }}}'
```

Change `requested_role` between calls and watch the `effective_role` flip without touching Keycloak or the database.

## How it works under the hood

The mapping pipeline (documented in [`OIDC.md`](../../OIDC.md#the-mapping-flow)):

```
JWT access token
  │  aud=yb-mcp-server (audience validated against OIDC_AUDIENCE)
  │  realm_access.roles = ["offline_access","default-roles-yb-mcp-map",
  │                        "uma_authorization","db-writer","db-reader"]
  │
  ├─ (2) Extract identity claim by dotted path
  │       YB_MCP_IDENTITY_CLAIM=realm_access.roles
  │       → walks claims["realm_access"]["roles"]
  │
  ├─ (3) Normalize to list of raw strings
  │       → ["offline_access", "default-roles-yb-mcp-map",
  │          "uma_authorization", "db-writer", "db-reader"]
  │
  ├─ (4) Apply identity map (ident.conf)
  │       offline_access              → NO MATCH → dropped
  │       default-roles-yb-mcp-map    → NO MATCH → dropped
  │       uma_authorization           → NO MATCH → dropped
  │       db-writer                   → writer
  │       db-reader                   → reader
  │       candidates = ["writer","reader"]
  │
  ├─ (5) Pick one candidate
  │       requested_role="writer" passed by the client
  │       clamped against candidates → allowed
  │
  └─ (6) SET ROLE writer (psycopg.sql.Identifier — always quoted)
```

**Fail-closed guarantees exercised by this setup:**

| Scenario | Outcome |
|---|---|
| Token has no `realm_access.roles` at all | `IdentityError` — no fallback to pool credentials |
| Token has only boilerplate realm roles | `IdentityError` — no candidate remains after the map |
| Ambiguous candidates, `requested_role` omitted | `IdentityError` — server refuses to pick arbitrarily |
| `requested_role` names a role NOT in the mapped candidates | `IdentityError` — clamp fails |
| Map file has an invalid `\N` backreference | Server refuses to start (fail-closed at load) |

The map IS the allowlist — every additional realm role on the token is either mapped explicitly or silently dropped.

## Adapting to a different OIDC provider

Same steps for Auth0, Okta, Azure AD, or any RFC 6749 / OIDC 1.0 provider — swap:

- `OIDC_CONFIG_URL` — provider's discovery document
- `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` — client registered with the provider
- `OIDC_AUDIENCE` — expected `aud` claim (often the client id itself)
- `YB_MCP_IDENTITY_CLAIM` — path to the roles/groups list in the provider's tokens (see [OIDC.md's Worked examples](../../OIDC.md#worked-examples) for provider-specific claim paths)
- `YB_MCP_IDENTITY_MAP` — an ident.conf whose left column matches whatever the IdP emits (raw group names, Azure GUIDs via regex, etc.)

The pipeline downstream of extraction is identical regardless of provider.

## Teardown

```bash
docker compose down -v     # nukes Keycloak state
```

Postgres roles / tables persist — drop them explicitly if you want to reset:

```sql
DROP TABLE IF EXISTS notes;
DROP ROLE  IF EXISTS writer;
DROP ROLE  IF EXISTS reader;
```
