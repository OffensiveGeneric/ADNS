#!/usr/bin/env bash
# Bootstrap a local ADNS dev environment (API + agent deps + frontend node_modules).
# Usage: ./scripts/setup_local.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Using repository root: ${ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found in PATH" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm is required to build the frontend. Install Node.js 18+." >&2
  exit 1
fi

# Python virtualenv (shared for API + agent)
if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "==> Creating virtualenv at ${ROOT}/.venv"
  "${PYTHON_BIN}" -m venv "${ROOT}/.venv"
fi

source "${ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip

echo "==> Installing Python dependencies (api + agent)"
python -m pip install -r "${ROOT}/api/requirements.txt" -r "${ROOT}/agent/requirements.txt"

# Frontend dependencies
echo "==> Installing frontend dependencies (npm install)"
pushd "${ROOT}/frontend/adns-frontend" >/dev/null
npm install
popd >/dev/null

if [[ ! -f "${ROOT}/.env" && -f "${ROOT}/.env.example" ]]; then
  echo "==> Copying .env.example to .env (edit as needed)"
  cp "${ROOT}/.env.example" "${ROOT}/.env"
fi

cat <<'EOM'
==> Setup complete.

Next steps:
  1) Ensure tshark is installed and your user can capture on your interface (default eth0). On Debian/Ubuntu:
       sudo apt-get update && sudo apt-get install -y tshark
     Then add your user to the wireshark group or run the agent with sudo.
  2) Edit .env to point SQLALCHEMY_DATABASE_URI where you want (e.g., sqlite:///./adns.db for local runs)
     and set VITE_API_URL if the frontend will reach the API on a different origin.
  3) Run services:
       # Terminal 1 (API)
       source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && cd api && flask run
       # Terminal 2 (worker; optional if relying on inline scoring)
       source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && python api/worker.py
       # Terminal 3 (agent; requires tshark + privileges)
       source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && cd agent && sudo ./capture.py
       # Terminal 4 (frontend)
       cd frontend/adns-frontend && export $(grep -v '^#' ../../.env | xargs) && npm run dev -- --host
EOM
