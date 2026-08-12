# OIDC authentication and per-user identity mapping

Everything the yugabytedb-mcp-server does with OIDC lives in this document —
provider setup, environment variables, the JWT → Postgres-role mapping, the
`/auth/login` shortcut, and security guidance. This is the single source of
truth; the top-level `README.md` links here.

Design north star: **parity with YSQL's native OIDC → PG role mapping**
(`ysql_hba_conf_csv` + `ysql_ident_conf_csv` + `matching_claim_key`). If
you've configured YSQL OIDC, the concepts and file format transfer directly.

Reference: YugabyteDB YSQL OIDC docs — [oidc-authentication-aad](https://docs.yugabyte.com/stable/yugabyte-platform/security/authentication/oidc-authentication-aad/).

---

## Table of contents

- [When to use it](#when-to-use-it)
- [Quick start (Cognito + identity map)](#quick-start)
- [Provider setup](#provider-setup)
  - [AWS Cognito](#aws-cognito)
  - [Generic OIDC](#generic-oidc)
- [Environment variables](#environment-variables)
- [Identity → PG role mapping](#identity--pg-role-mapping)
  - [The mapping flow](#the-mapping-flow)
  - [Claim types (scalar / list / dotted-path)](#claim-types)
  - [The identity map file (pg_ident.conf format)](#the-identity-map-file)
  - [List-valued claims and the `requested_role` parameter](#list-valued-claims)
- [Worked examples](#worked-examples)
  - [Example 1 — Cognito with `cognito:groups` (access-token flow)](#example-1)
  - [Example 2 — Keycloak with `realm_access.roles`](#example-2)
  - [Example 3 — Azure AD with GUID-valued groups](#example-3)
  - [Example 4 — Generic OIDC with a custom claim](#example-4)
- [The `/auth/login` endpoint (Cognito password flow)](#the-authlogin-endpoint)
- [Security guidance](#security-guidance)
- [Migrating from v1 to v2](#migrating-from-v1-to-v2)
- [Troubleshooting](#troubleshooting)

---

## When to use it

Turn OIDC on in **HTTP transport** deployments where multiple humans (or
agents authenticating as humans) share one MCP server and you want per-user
authorization enforced at the database layer.

- Stdio transport (`Claude Desktop --> yugabytedb-mcp-server`) has no
  network — auth is unnecessary and the OIDC path is skipped.
- HTTP transport without OIDC = the pool's connection role for every caller.
- HTTP transport with OIDC = `SET ROLE <mapped-role>` per tool call → each
  caller's SQL runs under their own database role. Combines with Postgres
  row-level security, schema grants, and `GRANT`/`REVOKE`.

---

## Quick start

The recommended setup: AWS Cognito + `sub` claim (present in access
tokens) + an explicit identity map that pins each user to a Postgres
role. Each token maps to a Postgres role via a file, not a transform,
so cross-domain collisions can't occur silently.

```bash
# 1. In Postgres — create one role per user and grant the pool user
#    membership (so it can SET ROLE).
psql -c "CREATE ROLE alice;    GRANT SELECT ON tbl TO alice;"
psql -c "CREATE ROLE bob;      GRANT SELECT ON tbl TO bob;"
psql -c "GRANT alice, bob TO yugabyte;"

# 2. Write an identity map — one line per user, mapping the Cognito
#    `sub` (or any claim you pick) to a Postgres role.
cat > /etc/yb-mcp/ident.conf <<'MAP'
# map_name  system_value                            db_role
default     a1b2c3d4-1111-2222-3333-444444444444    alice
default     e5f6g7h8-5555-6666-7777-888888888888    bob
MAP

# 3. Configure the server.
export MCP_AUTH_PROVIDER=cognito
export MCP_BASE_URL=https://mcp.example.com
export COGNITO_USER_POOL_ID=us-west-2_XXXXXXXX
export COGNITO_AWS_REGION=us-west-2
export COGNITO_CLIENT_ID=…
export COGNITO_CLIENT_SECRET=…
export YB_MCP_IDENTITY_CLAIM=sub                 # present in access tokens
export YB_MCP_IDENTITY_MAP=/etc/yb-mcp/ident.conf

# 4. Run.
yugabytedb-mcp --transport http
```

Result: a caller with a Cognito token whose `sub` is one of the
map entries gets `SET ROLE alice` (or `bob`) on every tool call. A
token whose `sub` isn't in the map is rejected. Neither user can
access the other's data (assuming Postgres grants are set up
correctly).

> `YB_MCP_IDENTITY_TRANSFORM=strip_domain` from the v1 release has been
> removed (DB-22174) because it silently collapsed users across email
> domains. Setting the env still at startup fails with a migration
> message. Use the identity map above instead.

For access-token workflows, list-valued claims, and role allowlists via a
map file, jump to the [Worked examples](#worked-examples).

---

## Provider setup

Two providers are supported: `cognito` (tested end-to-end) and `oidc`
(generic OIDC; exercised via Keycloak by the tutorials under
[`examples/oidc-auth-mapping/`](examples/oidc-auth-mapping/)). Any RFC 6749 /
OIDC 1.0 compliant provider should work — only the connection details change.

### AWS Cognito

Required env vars when `MCP_AUTH_PROVIDER=cognito`:

| Env var | Description |
|---|---|
| `COGNITO_USER_POOL_ID` | Cognito user-pool ID (e.g. `us-west-2_XXXXXXXX`) |
| `COGNITO_AWS_REGION` | AWS region of the user pool |
| `COGNITO_CLIENT_ID` | App-client ID within the pool |
| `COGNITO_CLIENT_SECRET` | App-client secret |

The server treats `COGNITO_CLIENT_ID` as the expected token audience — a
token minted for a different app client in the same pool is rejected once
audience validation lands (PR #3 of the audit release).

The Cognito app client should have `openid email profile` scopes enabled.
If you want access-token-friendly claim types (`cognito:groups`, or a custom
`custom:db_role`), configure Cognito to include them either via User Pool
Groups or a Pre-Token-Generation Lambda trigger.

### Generic OIDC

Required env vars when `MCP_AUTH_PROVIDER=oidc`:

| Env var | Description |
|---|---|
| `OIDC_CONFIG_URL` | Full URL of the provider's OIDC discovery document (`…/.well-known/openid-configuration`) |
| `OIDC_CLIENT_ID` | Client ID registered with the provider |
| `OIDC_CLIENT_SECRET` | Client secret |
| `OIDC_AUDIENCE` | *(recommended)* Expected token audience. Currently forwarded to `OIDCProxy` only; verifier-side check lands in PR #3. |

The MCP server delegates OIDC discovery, JWKS retrieval, and token
verification to FastMCP's `JWTVerifier` + `OIDCProxy`. The mapping path
below is provider-agnostic — configure `YB_MCP_IDENTITY_CLAIM` and
optionally an identity map, and the rest of the flow is identical to
Cognito's.

---

## Environment variables

Full reference. Marked with **(v1)** if pre-existed before the v2 mapping
release, **(v2)** if new.

| Env var | CLI flag | Purpose |
|---|---|---|
| `MCP_AUTH_PROVIDER` | `--mcp-auth-provider` | `cognito`, `oidc`, or unset (auth disabled). |
| `MCP_BASE_URL` | — | Public base URL the server is reachable at (used for OAuth redirects). Required when auth is enabled. |
| `YB_MCP_IDENTITY_CLAIM` | `--identity-claim` | **(v1, v2-extended)** JWT claim carrying the user's identity. v2 accepts dotted paths like `realm_access.roles` and top-level names with colons like `cognito:groups`. Default: `sub` (or `email` when `YB_MCP_LEGACY_ACCEPT_ID_TOKENS=true`). |
| `YB_MCP_IDENTITY_MAP` | `--identity-map` | **(v2)** Path to a `pg_ident.conf`-style map file. When set, each claim value is looked up in the map; without a map, the raw claim value is used verbatim as the DB role. See [The identity map file](#the-identity-map-file). |
| `YB_MCP_IDENTITY_MAP_NAME` | `--identity-map-name` | **(v2)** Which named map inside the file to apply. Default: `default`. |
| `YB_MCP_REQUIRE_ACCESS_TOKEN` | — | **(v2, Cognito-only)** When `true`, reject tokens with `token_use != "access"` (rejects ID tokens and refresh tokens presented as bearers). Default: `true`. See [Access-token vs ID-token](#access-token-vs-id-token). |
| `YB_MCP_LEGACY_ACCEPT_ID_TOKENS` | — | **(v2)** Compat flag. When `true`, restores pre-DB-22136 defaults: `identity_claim=email` and `require_access_token=false` (ID tokens accepted). Prefer setting each individually. |

Cognito-specific (only required when `MCP_AUTH_PROVIDER=cognito`) — see
[AWS Cognito](#aws-cognito) above.

Generic-OIDC-specific — see [Generic OIDC](#generic-oidc) above.

---

## Identity → PG role mapping

### The mapping flow

Every tool call goes through this pipeline:

```
JWT access token
  │
  ├─ (1) Verify signature + issuer  ────►  reject on failure (401)
  │
  ├─ (2) Extract the identity claim by name (`YB_MCP_IDENTITY_CLAIM`)
  │      └─ Dotted paths walk nested dicts: `realm_access.roles`
  │      └─ Colons stay literal: `cognito:groups` is one key, not two
  │
  ├─ (3) Normalize to a list of raw string values
  │      └─ Scalar claim: `email = "alice@…"` → `["alice@…"]`
  │      └─ List claim: `groups = ["writer","reader"]` → `["writer","reader"]`
  │
  ├─ (4) Resolve each raw value to a candidate DB role
  │      ├─ If `YB_MCP_IDENTITY_MAP` is set:
  │      │    apply the map — literal or regex lookup per entry.
  │      │    Unmapped values are dropped (not fallen through).
  │      └─ Otherwise:
  │           use the raw claim value verbatim as the DB role name.
  │
  ├─ (5) Pick one candidate
  │      ├─ 0 candidates → IdentityError (fail-closed)
  │      ├─ 1 candidate → auto-pick
  │      └─ ≥2 candidates → agent must pass `requested_role`;
  │           server clamps against the candidate list.
  │
  └─ (6) SET ROLE <picked>   (via psycopg.sql.Identifier — always quoted)
```

Two error modes:

- **`IdentityError`** (returned to the client as a clean error) — the token
  is valid but has no usable claim, no candidate resolves, or
  `requested_role` isn't in the candidate list. Fail-closed: never falls
  through to pool credentials.
- **`SET ROLE` failure** at the DB layer — the mapped role doesn't exist,
  or the pool user doesn't have `GRANT` on it. Currently surfaces as a
  `ToolError`; DB-22175 tracks turning this into a clean `IdentityError`.

### Claim types

Three claim shapes work end-to-end.

**Scalar** (single string value):

```json
{
  "sub": "5ac1e1a0-f8d3-4d5c-9a2b-8c73f4e2b3a1",
  "email": "alice@example.com",
  "preferred_username": "alice"
}
```

Any of these can be the identity claim. Configure via
`YB_MCP_IDENTITY_CLAIM=email` (or `sub`, or `preferred_username`).

**List** (array of strings — usually roles/groups):

```json
{
  "cognito:groups": ["writer", "reader"],
  "realm_access": {
    "roles": ["app-writer", "app-reader"]
  },
  "groups": ["a12d04b1-7463-…", "c22b03b1-2746-…"]
}
```

Configure via `YB_MCP_IDENTITY_CLAIM=cognito:groups` (Cognito),
`realm_access.roles` (Keycloak), or `groups` (Azure AD). The caller (agent)
picks one via `requested_role` — see [List-valued claims](#list-valued-claims).

**Dotted path** — the identity claim can walk nested dict structure:

- `realm_access.roles` → `claims["realm_access"]["roles"]`
- `a.b.c` → `claims["a"]["b"]["c"]`
- `cognito:groups` — NO walk (colon is a literal key char) →
  `claims["cognito:groups"]`

Missing intermediate keys raise `IdentityError` with the exact path that
failed.

### The identity map file

`YB_MCP_IDENTITY_MAP=/etc/yb-mcp/ident.conf` points at a text file with the
same format YSQL uses for `ysql_ident_conf_csv` and PostgreSQL uses for
`pg_ident.conf`.

**Syntax:**

```
# Each non-empty, non-comment line has three space-separated fields:
#   <map_name>  <system_value>  <db_role>
#
# `#` starts a comment (line or trailing).
# Blank lines are ignored.
# `system_value` starting with `/` is a regex — the rest of the field
# (no closing `/`) is the pattern. Match is anchored (fullmatch).
# `\1`, `\2`, ... in `db_role` reference regex capture groups.

# Literal mapping:
default  user@yugabyte.com                                     user
default  bob@yugabyte.com                                      writer
default  reader@yugabyte.com                                   reader

# Regex mapping with capture group:
default  /^(.*)@yugabyte\.com$                                 \1

# Role-name mapping (Keycloak realm-role name → DB role name):
default  app-writer                                            writer
default  app-reader                                            reader

# Azure AD GUID → readable role name (YSQL docs note: GUIDs need regex):
azure    /^a12d04b1-.+                                         reader
azure    /^c22b03b1-.+                                         writer

# Cognito custom-claim OIDC group name → PG role:
cognito  OIDC.Test.Read                                        read_only_user
```

**Multiple named maps** — one file can carry entries for `default`,
`azure`, `cognito`, and any custom name. `YB_MCP_IDENTITY_MAP_NAME`
selects which one to apply at runtime (default: `default`). Entries under
other names are loaded but never consulted — useful for switching between
IdPs by env-var flip.

**Fail-closed** — a syntactically bad line (missing field count, invalid
regex, unreadable file) makes the server refuse to start. A typo doesn't
silently widen access.

**Match order** — entries are iterated in file order; the first match under
the active `map_name` wins. Put more-specific entries above catch-all
regexes.

**No fallback** — if the map has no entry that matches the claim value,
that value is dropped from the candidate list. If no value in a list-claim
resolves, `IdentityError` fires. The map IS the allowlist.

### List-valued claims

Cognito's `cognito:groups`, Keycloak's `realm_access.roles`, and Azure AD's
`groups` all come back as JSON arrays. The mapping semantics match YSQL's
native behavior: *"the user can take any
PG role that is in their roles/groups claim."*

**How the tool interface exposes this:** every DB tool
(`summarize_database`, `run_read_only_query`, `run_write_query`) accepts an
optional `requested_role: str | None = None` parameter.

- If the claim resolves to a single candidate → server ignores
  `requested_role` and uses that candidate.
- If the claim resolves to multiple candidates AND `requested_role` is in
  the candidate list → server uses `requested_role`.
- If the claim resolves to multiple candidates AND `requested_role` is
  passed but NOT in the list → server raises `IdentityError`. The agent
  cannot pick a role that isn't in its JWT's mapped candidates.
- If the claim resolves to multiple candidates AND `requested_role` is
  `None` → server auto-picks the first candidate and logs a `WARNING`.
  Pass `requested_role` explicitly for deterministic behavior.
- If the claim resolves to zero candidates → `IdentityError`.

**How the agent picks:** the LLM invokes `run_read_only_query(..., 
requested_role="reader")` after seeing the JWT's mapped candidate list.
The server clamps — the agent cannot request a role that isn't in the JWT.

---

## Worked examples

### Example 1

**Cognito with `cognito:groups` (access-token workflow; recommended).**

Configure Cognito to add users to User Pool Groups (Console → User pools →
Groups). Access tokens now carry `cognito:groups`:

```json
{
  "sub": "…",
  "token_use": "access",
  "cognito:groups": ["writer", "reader"]
}
```

Server config:

```bash
export YB_MCP_IDENTITY_CLAIM=cognito:groups
# YB_MCP_IDENTITY_MAP optional — group names ARE the DB role names
# unset means each group in the list is used as-is
```

Postgres setup — one role per group name:

```sql
CREATE ROLE writer; GRANT ...ON tbl TO writer; GRANT writer TO yugabyte;
CREATE ROLE reader; GRANT SELECT ON tbl TO reader; GRANT reader TO yugabyte;
```

**Agent invocation** — the LLM sees two candidates and picks one:

```json
{
  "name": "run_read_only_query",
  "arguments": {
    "query": "SELECT * FROM tbl",
    "requested_role": "reader"
  }
}
```

Server-side flow: `_extract_claim` gets `["writer","reader"]` →
no-map path (raw claim used verbatim) → candidates `["writer","reader"]`
→ `_pick_role(candidates, "reader")` returns `"reader"` → `SET ROLE reader`.

### Example 2

**Keycloak with `realm_access.roles` (dotted-path claim + map file).**

Access token payload:

```json
{
  "realm_access": {
    "roles": ["app-writer", "offline_access", "default-roles-realm"]
  }
}
```

The realm has three roles but only one is meaningful for DB access
(`app-writer`); the others are Keycloak boilerplate. Use a map to
whitelist:

```
# /etc/yb-mcp/ident.conf
default  app-writer  writer
default  app-reader  reader
```

Server config:

```bash
export YB_MCP_IDENTITY_CLAIM=realm_access.roles
export YB_MCP_IDENTITY_MAP=/etc/yb-mcp/ident.conf
# YB_MCP_IDENTITY_MAP_NAME=default (implicit)
```

Server-side flow: `_extract_claim` walks
`claims["realm_access"]["roles"]` → `["app-writer", "offline_access",
"default-roles-realm"]` → `_apply_map` per element → `offline_access` and
`default-roles-realm` don't match any map entry (dropped), `app-writer` →
`writer`. Candidates = `["writer"]`, single candidate → auto-pick →
`SET ROLE writer`.

Keycloak boilerplate roles never reach the database. The map IS the
allowlist.

### Example 3

**Azure AD with GUID-valued `groups` (regex map required).**

Azure AD emits group memberships as opaque GUIDs, which aren't valid
Postgres role identifiers. Use a regex map to translate them:

```
# /etc/yb-mcp/ident.conf
azure  /^a12d04b1-.+   readonly_role
azure  /^c22b03b1-.+   writable_role
azure  /^9c8d4f3e-.+   admin_role
```

Access token payload:

```json
{
  "groups": ["a12d04b1-7463-8e23-94d2-8d71f17ab99b"]
}
```

Server config:

```bash
export YB_MCP_IDENTITY_CLAIM=groups
export YB_MCP_IDENTITY_MAP=/etc/yb-mcp/ident.conf
export YB_MCP_IDENTITY_MAP_NAME=azure
```

Server-side flow: `_extract_claim` → `["a12d04b1-..."]` → `_apply_map`
regex hit on `/^a12d04b1-.+` → `readonly_role`. `SET ROLE readonly_role`.

**Regex capture groups** for on-the-fly renaming — e.g. strip a common
prefix from Azure group names:

```
azure  /^APP_YB_MCP_(.+)$   \1
```

`APP_YB_MCP_writer` → `writer`.

### Example 4

**Generic OIDC (e.g. Okta) with a custom claim.**

Configure the IdP to inject a `db_role` claim into access tokens (or use
`preferred_username` — a standard OIDC claim). Server config:

```bash
export MCP_AUTH_PROVIDER=oidc
export OIDC_CONFIG_URL=https://okta.example.com/.well-known/openid-configuration
export OIDC_CLIENT_ID=…
export OIDC_CLIENT_SECRET=…
export OIDC_AUDIENCE=…
export YB_MCP_IDENTITY_CLAIM=db_role
# No map — the custom claim already carries the role name.
```

If the IdP can't inject a custom claim, use `preferred_username` and map
usernames via the map file.

---

## The `/auth/login` endpoint

When `MCP_AUTH_PROVIDER=cognito`, the HTTP transport exposes an unauthenticated
`POST /auth/login` that exchanges Cognito email + password for tokens. This is
for curl-based smoke tests, CI pipelines, and scripted clients that can't run
a browser OAuth flow.

```bash
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email": "alice@example.com", "password": "…"}'
# → { "access_token": "...", "id_token": "...", "refresh_token": "...", ... }

ACCESS_TOKEN=<from above>
curl -H "Authorization: Bearer $ACCESS_TOKEN" http://localhost:8000/mcp
```

Requires the Cognito app client to have `ALLOW_USER_PASSWORD_AUTH` enabled
(Console → User pools → App integration → App client settings). MFA,
password-reset, and other challenge flows are not handled — those require the
browser OAuth flow.

Hardening gaps tracked in DB-22190 (uniform error detail — currently leaks
whether the email exists) and DB-22191 (no app-level rate limiting; returns
refresh token in the response body). Both scheduled for follow-up release.

---

## Security guidance

### Least-privilege pool

The pool DB user (whatever's in `YUGABYTEDB_URL`) must have `GRANT` on
every role the identity mapping can reach — that's how `SET ROLE` works.
Two approaches:

1. **Least-privilege pool + map allowlist (recommended).** Pool user is a
   normal role. Explicitly `GRANT` it membership in the target roles you
   want reachable. Configure `YB_MCP_IDENTITY_MAP` — the map is the
   allowlist; anything not in the map raises `IdentityError` before
   `SET ROLE` runs.

   ```sql
   CREATE ROLE mcp_pool WITH LOGIN PASSWORD '…';
   CREATE ROLE writer;
   CREATE ROLE reader;
   GRANT writer, reader TO mcp_pool;   -- pool can SET ROLE to these two
   ```

2. **Superuser pool (default, less safe).** Pool user is a Postgres
   superuser (`yugabyte` is by default). `SET ROLE` can reach any role,
   including superuser roles. Only safe if `YB_MCP_IDENTITY_MAP` is set —
   the map bounds what roles can be reached even from a superuser pool.

Without a map file AND a superuser pool, any authenticated user whose
claim happens to spell an existing role name gets `SET ROLE` to that role —
including superuser roles named after real users. This is DB-22135.

### Why the map is an allowlist

The map's `<system_value>` field is a whitelist. Values not matching any
entry are dropped from the candidate list. If a list-valued claim's every
value is unmapped, `IdentityError` fires — no fallback to pool credentials.

This is why we recommend running with a map even for simple deployments:
it turns the identity → role edge into an explicit, auditable list.

### Identifier quoting

Role names are always wrapped in `psycopg.sql.Identifier` before reaching
`SET ROLE` — see `tools.py:95`. So even if a claim value or map output
contained SQL-injection-shaped strings, they'd be safely quoted as an
identifier. The map layer doesn't sanitize; the SQL layer does.

### Access-token vs ID-token

Cognito issues two tokens per authentication: an ID token (contains
`email`, `preferred_username`, and other user attributes) and an access
token (contains `sub`, `cognito:groups`, `token_use=access`; NO `email`).
Standard OAuth 2.0 practice is to send access tokens to APIs, ID tokens to
clients.

Today the server doesn't enforce `token_use=access` — it accepts either.
Once PR #3 lands with `YB_MCP_REQUIRE_ACCESS_TOKEN=true`, the ID-token
path closes. To be ready:

- Prefer identity claims present in access tokens (`sub`, `cognito:groups`,
  custom claims via a Cognito Lambda). Don't rely on `email`.
- Example 1 above is the recommended long-term shape.

---

## Migrating from v1 to v2

The v2 mapping is not fully backward-compatible. Two changes require
attention on upgrade:

1. **`YB_MCP_IDENTITY_TRANSFORM` has been removed** (DB-22174). The
   `strip_domain` value silently collapsed users across email domains;
   startup now fails if this env var is set. Migrate to
   `YB_MCP_IDENTITY_MAP` with one entry per user, or key the mapping
   off `preferred_username` / `sub` and use the raw claim as the role
   name.
2. **Cognito access tokens are required by default** (DB-22136). Set
   `YB_MCP_LEGACY_ACCEPT_ID_TOKENS=true` to opt back into ID tokens if
   your deployment still uses `email` from the ID token.

Migration checklist for an existing deployment:

1. **Enumerate your current roles.** `SELECT rolname FROM pg_roles WHERE …`.
2. **Decide the claim source.** For access-token workflows, pick something
   present in access tokens (`sub`, `cognito:groups`, custom claim). For
   ID-token workflows continuing on `email`, no change needed yet.
3. **Write the map file.** Start with literal entries for known users;
   add a regex fallback if needed. Test the file parses locally:

   ```bash
   YB_MCP_IDENTITY_MAP=/path/to/ident.conf yugabytedb-mcp --transport stdio
   # If the map is malformed, startup fails with a clear error.
   ```

4. **Deploy with the map**, keeping the old claim/transform envs as
   fallback. When the map is set, transform is ignored.
5. **Verify with a smoke test.** Use `/auth/login` or your normal auth
   path, invoke a tool with `requested_role` set, confirm the SQL runs as
   the expected role (`SELECT current_role` in a `run_read_only_query`).
6. **(Optional) Switch claim source.** Change `YB_MCP_IDENTITY_CLAIM` to
   an access-token-friendly value (e.g. `cognito:groups`), update the map
   accordingly, re-verify.

---

## Troubleshooting

**`IdentityError: Token present but required claim 'email' is missing or empty.`**
The token is valid but has no value at the configured claim name. Common
cause: an access token was presented where `identity_claim=email` (email is
ID-token-only on Cognito). Fix: switch `identity_claim` to a claim present
in access tokens (`sub`, `cognito:groups`), or accept ID tokens.

**`IdentityError: None of the identity-claim values resolved to a permitted DB role.`**
The claim is present but nothing in the (list of) values matches any entry
in the map under the current `identity_map_name`. Check:
- Is `YB_MCP_IDENTITY_MAP_NAME` the right map? (e.g. `default` vs `azure`).
- Do the map's `<system_value>` fields exactly match what the IdP sends?
  Print the raw JWT and inspect.
- Is a regex missing an anchor? The match is anchored (fullmatch) — a
  pattern like `/foo` will only match the exact string `foo`, not `foobar`.

**`IdentityError: requested_role='admin' is not in the caller's identity-claim candidates [...]`**
The agent passed a `requested_role` that isn't in the JWT's mapped list.
By design — server clamps. Fix: agent picks a value from the candidates
returned to it (or, if the agent is running blind, drop `requested_role`
and let the server auto-pick with a WARNING).

**`role "X" does not exist`** (as a `ToolError`, not `IdentityError`).
The mapping produced a role name that Postgres doesn't have. Either
create the role or fix the map. DB-22175 tracks converting this into a
clean `IdentityError` at the tool layer.

**Server refuses to start with `ValueError: /path/ident.conf:N: …`.**
Fail-closed on a malformed map file. The line number and pattern are in
the error message. Fix the file and restart.

**`SET ROLE` succeeds but the user still can't read/write.**
Postgres role membership vs. object-level `GRANT`s are separate. Confirm:
```sql
SELECT current_role;
GRANT SELECT ON tbl TO reader;
```

---

## YSQL parity reference (for engineers coming from YSQL native OIDC)

The MCP server's mapping is designed to port cleanly from YSQL's native
`ysql_hba_conf_csv` + `ysql_ident_conf_csv` + `matching_claim_key`
approach. Rough equivalence:

| YSQL native OIDC | MCP server v2 |
|---|---|
| `matching_claim_key` (in `ysql_hba_conf_csv`) | `YB_MCP_IDENTITY_CLAIM` env var |
| `map=<name>` (in HBA line) | `YB_MCP_IDENTITY_MAP_NAME` env var |
| `ysql_ident_conf_csv` file | `YB_MCP_IDENTITY_MAP` file (same format) |
| YSQL `HandleValidationResultAndPopulateIdentityClaims` | `tools.py:_extract_claim` |
| YSQL `GetJwtClaimAsStringsList` | `tools.py:_get_db_role` list-normalization |
| YSQL `YBCValidateJWT` | Not called from MCP — FastMCP's `JWTVerifier` handles signature/issuer/audience |

If you have a working `ysql_ident_conf_csv` file for a YSQL-native OIDC
deployment, you can point `YB_MCP_IDENTITY_MAP` at the same file and it
will parse identically.
