#!/usr/bin/env bash
set -euo pipefail

# Launch all dev services in a tmux session so they keep running in the background.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVFILE="$ROOT/.env"
SESSION="adns"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install it (e.g., sudo apt-get install -y tmux) and rerun."
  exit 1
fi

if [[ ! -f "$ENVFILE" ]]; then
  echo ".env not found at $ENVFILE. Create it (or copy .env.example) before starting services."
  exit 1
fi

EXPORT_ENV="export $(grep -v '^#' "$ENVFILE" | xargs -r)"

# Reuse existing session if it already exists.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new -d -s "$SESSION" \
  "cd \"$ROOT\" && source .venv/bin/activate && $EXPORT_ENV && cd api && flask run"

tmux new-window -t "$SESSION:1" \
  "cd \"$ROOT\" && source .venv/bin/activate && $EXPORT_ENV && python api/worker.py"

tmux new-window -t "$SESSION:2" \
  "cd \"$ROOT\" && source .venv/bin/activate && $EXPORT_ENV && cd agent && sudo ./capture.py"

tmux new-window -t "$SESSION:3" \
  "cd \"$ROOT/frontend/adns-frontend\" && export \$(grep -v '^#' ../../.env | xargs -r) && npm run dev -- --host"

echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
