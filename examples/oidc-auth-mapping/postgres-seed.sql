-- examples/oidc-auth-mapping/postgres-seed.sql
--
-- Seed for the identity-mapping tutorial. Same two Postgres roles as the
-- companion examples/oidc-auth/ tutorial (writer / reader), but where THIS
-- tutorial differs is HOW the caller lands on them:
--
--   * examples/oidc-auth/ — email → strip_domain → role
--       writer@yugabyte.com  → SET ROLE writer
--       reader@yugabyte.com  → SET ROLE reader
--
--   * this tutorial — realm_access.roles → identity map → role
--       token contains ["db-writer"]           → SET ROLE writer
--       token contains ["db-reader"]           → SET ROLE reader
--       token contains ["db-writer","db-reader"] + requested_role="writer"
--                                              → SET ROLE writer
--
-- Same enforcement, different plumbing. Run alongside the existing tutorial:
-- these are Postgres roles, not Keycloak roles — one seed serves both.
--
-- Usage:
--   ysqlsh "$YUGABYTEDB_URL" -v yb_pool_user=yugabyte -f postgres-seed.sql

\set ON_ERROR_STOP on

-- 1. Roles. NOLOGIN — they exist only to be SET ROLE'd into.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'writer') THEN
        CREATE ROLE writer NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reader') THEN
        CREATE ROLE reader NOLOGIN;
    END IF;
END $$;

-- 2. Demo table (idempotent — same schema as the sibling tutorial).
CREATE TABLE IF NOT EXISTS notes (
    id    SERIAL PRIMARY KEY,
    body  TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO notes (body)
SELECT 'hello, world (seeded)'
WHERE NOT EXISTS (SELECT 1 FROM notes);

INSERT INTO notes (body)
SELECT 'reader can see this'
WHERE (SELECT COUNT(*) FROM notes) < 2;

-- 3. GRANTs.
GRANT SELECT ON notes TO reader;
GRANT SELECT, INSERT, UPDATE, DELETE ON notes TO writer;
GRANT USAGE, SELECT, UPDATE ON SEQUENCE notes_id_seq TO writer;

-- 4. Pool-user membership. `SET ROLE reader` from the pool connection
-- fails with "permission denied to set role" without this.
GRANT writer TO :"yb_pool_user";
GRANT reader TO :"yb_pool_user";

-- 5. Sanity output so the reader can confirm the seed worked.
\echo
\echo '=== Roles ==='
SELECT rolname FROM pg_roles WHERE rolname IN ('writer', 'reader') ORDER BY rolname;

\echo
\echo '=== GRANTs on notes ==='
SELECT grantee, privilege_type
FROM information_schema.table_privileges
WHERE table_name = 'notes' AND grantee IN ('writer', 'reader')
ORDER BY grantee, privilege_type;

\echo
\echo '=== Membership ==='
SELECT r.rolname AS member, g.rolname AS in_group
FROM pg_auth_members m
JOIN pg_roles r ON r.oid = m.member
JOIN pg_roles g ON g.oid = m.roleid
WHERE g.rolname IN ('writer', 'reader')
ORDER BY g.rolname, r.rolname;

\echo
\echo 'Seed complete.'
