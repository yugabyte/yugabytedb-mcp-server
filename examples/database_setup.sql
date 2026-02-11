CREATE SCHEMA app_schema;
CREATE SCHEMA analytics_schema;

ALTER SCHEMA app_schema OWNER TO yugabyte;
ALTER SCHEMA analytics_schema OWNER TO yugabyte;

CREATE TABLE app_schema.customers (
id SERIAL PRIMARY KEY,
     name TEXT
 );

 CREATE TABLE analytics_schema.reports (
     id SERIAL PRIMARY KEY,
          report_name TEXT
 );

 CREATE USER user_full_access WITH PASSWORD 'strong_pw1';
 CREATE USER user_limited WITH PASSWORD 'strong_pw2';

 REVOKE ALL ON SCHEMA app_schema FROM PUBLIC;
 REVOKE ALL ON SCHEMA analytics_schema FROM PUBLIC;

 GRANT USAGE ON SCHEMA app_schema TO user_full_access;
 GRANT USAGE ON SCHEMA analytics_schema TO user_full_access;
 GRANT USAGE ON SCHEMA app_schema TO user_limited;

 GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO user_full_access;
 GRANT SELECT ON ALL TABLES IN SCHEMA analytics_schema TO user_full_access;
 GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO user_limited;