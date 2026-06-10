-- examples/oidc-auth/postgres-seed.sql
--
-- Seeds the demo database for the OIDC-to-Postgres role mapping tutorial.
-- Creates two roles (`writer`, `reader`) matching the Keycloak users
-- (writer@yugabyte.com / reader@yugabyte.com) after the strip_domain
-- transform. Creates a `notes` table with row-level GRANTs that enforce
-- the behaviour the tutorial demonstrates:
--
--   writer  -> can SELECT and INSERT/UPDATE/DELETE on notes
--   reader  -> can SELECT only
--
-- IMPORTANT — the pool user
--
-- The MCP server connects with the database user from YUGABYTEDB_URL
-- (call this the "pool user"). The per-request SET ROLE only works if
-- the pool user has been granted membership in the target role. The
-- last block of this script does that automatically using
-- :"yb_pool_user", which ysqlsh/psql substitutes from a -v flag:
--
--   ysqlsh "$YUGABYTEDB_URL" -v yb_pool_user=yugabyte -f postgres-seed.sql
--
-- Substitute `yugabyte` with whichever user appears in your YUGABYTEDB_URL.
-- The script is idempotent; safe to re-run.

\set ON_ERROR_STOP on

-- 1. Roles. Both are NOLOGIN — they exist only to be SET ROLE'd into.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'writer') THEN
        CREATE ROLE writer NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reader') THEN
        CREATE ROLE reader NOLOGIN;
    END IF;
END $$;

-- 2. Demo table. IF NOT EXISTS keeps re-runs cheap.
CREATE TABLE IF NOT EXISTS notes (
    id    SERIAL PRIMARY KEY,
    body  TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A couple of seed rows so SELECT returns something on the first run.
-- (Idempotent via the NOT EXISTS guard.)
INSERT INTO notes (body)
SELECT 'hello, world (seeded)'
WHERE NOT EXISTS (SELECT 1 FROM notes);

INSERT INTO notes (body)
SELECT 'reader can see this'
WHERE (SELECT COUNT(*) FROM notes) < 2;

-- 3. GRANTs.
--
-- We grant on the table itself + on the sequence backing the SERIAL,
-- because writes need both. SELECT is enough for reads.
GRANT SELECT ON notes TO reader;
GRANT SELECT, INSERT, UPDATE, DELETE ON notes TO writer;
GRANT USAGE, SELECT, UPDATE ON SEQUENCE notes_id_seq TO writer;

-- 4. Membership for the pool user.
--
-- Without this, `SET ROLE reader` returns:
--   ERROR: permission denied to set role "reader"
--
-- The pool user must be a member of every role it will SET ROLE into.
GRANT writer TO :"yb_pool_user";
GRANT reader TO :"yb_pool_user";

-- 5. Sanity check — print what we just set up so the tutorial reader can
-- see the seed worked.
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
