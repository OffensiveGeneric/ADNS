#!/usr/bin/env bash
# Creates a local PostgreSQL database/user for ADNS.
# Usage: ./scripts/setup_postgres_local.sh [db_name] [db_user] [db_password]

set -euo pipefail

DB="${1:-adns}"
USER="${2:-adns}"
PASS="${3:-adns_password}"
HOST="${PGHOST:-127.0.0.1}"
PORT="${PGPORT:-5432}"
SUPERUSER="${PGUSER:-postgres}"

if ! command -v psql >/dev/null 2>&1; then
  echo "psql not found. Install PostgreSQL and ensure it is on PATH (e.g., apt/brew install postgresql)." >&2
  exit 1
fi

run_psql() {
  # Prefer the standard postgres superuser if available; fall back to current user/PG* env.
  if sudo -u postgres psql -Atqc "select 1" >/dev/null 2>&1; then
    sudo -u postgres psql -v ON_ERROR_STOP=1 "$@"
  else
    PGPASSWORD="${PGPASSWORD:-}" psql -h "$HOST" -p "$PORT" -U "$SUPERUSER" -d postgres -v ON_ERROR_STOP=1 "$@"
  fi
}

echo "Ensuring role '$USER' exists..."
run_psql -v user="$USER" -v pass="$PASS" -c \
  "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = :'user') THEN EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L', :'user', :'pass'); END IF; END $$;"

echo "Ensuring database '$DB' exists (owned by '$USER')..."
run_psql -v db="$DB" -v user="$USER" -c \
  "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_database WHERE datname = :'db') THEN EXECUTE format('CREATE DATABASE %I WITH OWNER %I ENCODING ''UTF8''', :'db', :'user'); END IF; END $$;"

echo "Granting privileges on '$DB' to '$USER'..."
run_psql -v db="$DB" -v user="$USER" -c "ALTER DATABASE :\"db\" OWNER TO :\"user\"; GRANT ALL PRIVILEGES ON DATABASE :\"db\" TO :\"user\";"

cat <<EOF

Database ready. Add this to your .env (or export it) and restart services:
SQLALCHEMY_DATABASE_URI=postgresql://$USER:$PASS@$HOST:$PORT/$DB
ADNS_REDIS_URL=redis://127.0.0.1:6379/0
EOF
