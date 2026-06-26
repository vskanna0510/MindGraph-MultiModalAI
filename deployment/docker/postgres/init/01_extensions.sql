# PostgreSQL initialization
# Runs automatically on first container start via /docker-entrypoint-initdb.d/

\set ON_ERROR_STOP on

SELECT 'Creating MindGraph++ database extensions' AS status;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
