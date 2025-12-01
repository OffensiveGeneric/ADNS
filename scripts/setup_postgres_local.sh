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

quote_ident() { sed 's/"/""/g' <<<"$1"; }
quote_literal() { sed "s/'/''/g" <<<"$1"; }

ROLE_IDENT="\"$(quote_ident "$USER")\""
ROLE_LIT="'$(quote_literal "$USER")'"
DB_IDENT="\"$(quote_ident "$DB")\""
DB_LIT="'$(quote_literal "$DB")'"
PASS_LIT="'$(quote_literal "$PASS")'"

echo "Ensuring role '$USER' exists..."
if [[ -z "$(run_psql -tAc "SELECT 1 FROM pg_roles WHERE rolname = $ROLE_LIT" 2>/dev/null || true)" ]]; then
  run_psql -c "CREATE ROLE $ROLE_IDENT LOGIN PASSWORD $PASS_LIT;"
fi

echo "Ensuring database '$DB' exists (owned by '$USER')..."
if [[ -z "$(run_psql -tAc "SELECT 1 FROM pg_database WHERE datname = $DB_LIT" 2>/dev/null || true)" ]]; then
  run_psql -c "CREATE DATABASE $DB_IDENT WITH OWNER $ROLE_IDENT ENCODING 'UTF8';"
fi

echo "Granting privileges on '$DB' to '$USER'..."
run_psql -c "ALTER DATABASE $DB_IDENT OWNER TO $ROLE_IDENT; GRANT ALL PRIVILEGES ON DATABASE $DB_IDENT TO $ROLE_IDENT;"

cat <<EOF

Database ready. Add this to your .env (or export it) and restart services:
SQLALCHEMY_DATABASE_URI=postgresql://$USER:$PASS@$HOST:$PORT/$DB
ADNS_REDIS_URL=redis://127.0.0.1:6379/0
EOF
