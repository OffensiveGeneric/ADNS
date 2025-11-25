# ADNS Agent Brief

Use this page to remind yourself (or other assistants) what lives where in the ADNS stack and how the pieces talk to each other.

## Mission Snapshot
- **Goal**: Demonstrate a modern network anomaly detection loop end to end (capture -> ingest -> score -> visualize) with synthetic attack simulations for workshops.
- **Live topology**: `agent/capture.py` runs `tshark` on `eth0`, POSTs batches to the Flask API at `/api/ingest` (Gunicorn on 127.0.0.1:5000, Nginx proxies `/api/*`). The API persists flows in PostgreSQL, enqueues flow IDs on Redis/RQ, and the scoring worker writes `Prediction` rows. The React/Vite dashboard (`frontend/adns-frontend/dist`) is served by Nginx at `http://159.203.105.167/`.
- **Storage**: PostgreSQL holds `flows` + `predictions` (see `api/app.py`). Retention trims anything older than `ADNS_FLOW_RETENTION_MINUTES` or beyond `ADNS_FLOW_RETENTION_MAX_ROWS`. Future work swaps the remaining in-memory cues for full SQL queries.
- **Models**: `api/model_runner.py` loads `model_artifacts/flow_detector.joblib` and `meta_model_combined.joblib` to drive the DetectionEngine. Training/data prep lives under `ml/`.

## Component Reference
| Piece | Path | Notes |
| --- | --- | --- |
| Capture agent | `agent/capture.py` | Wraps `tshark` (fields defined in `TSHARK_FIELDS`), infers ports/services, batches ~50 flows or ~2 s, retries POSTs with backoff. Configure `API_URL`, `INTERFACE`, `BATCH_SIZE` via env vars. |
| API | `api/app.py` | Flask + SQLAlchemy; exposes `/health`, `/ingest`, `/flows`, `/anomalies`, `/simulate`. Creates tables + ensures `flows.extra` JSON column on bootstrap. Flow inserts enqueue `tasks.score_flow_batch` via `task_queue.py`. |
| Task queue | `api/task_queue.py`, `api/tasks.py` | Redis URL from `ADNS_REDIS_URL`. RQ queue name defaults to `flow_scores`. `score_flow_batch` loads flows, skips already scored IDs, and writes `Prediction` rows within app context. |
| Worker | `api/worker.py` | RQ worker bootstrap; set `ADNS_RQ_QUEUE`, `ADNS_REDIS_URL`, `ADNS_RQ_JOB_TIMEOUT` as needed. Reload when model artifacts change. |
| Detection engine | `api/model_runner.py`, `api/scoring.py` | Combines lightweight flow pipeline and ExtraTrees/XGBoost meta bundle. Builds synthesized features from `Flow.extra` when packet metadata is sparse. |
| Frontend | `frontend/adns-frontend/` | React/Vite dashboard with timeline, severity donut, and attack simulation buttons (calls `/api/simulate`). Build output served from `dist/` via Nginx. Configure `VITE_API_URL` at build time for remote deployments. |
| Deployment + ops | `deployment/`, `worker/`, `assets/` | Systemd unit files (`adns-api.service`, `adns-worker.service`, `adns-agent.service`), docker/nginx snippets, and misc assets for demos. |
| Data + ML lab | `data/`, `outputs/`, `ml/`, `docs/` | Raw datasets (UNSW-NB15, TON_IoT) in `data/`, derived CSVs + models in `outputs/`, preprocessing + training scripts in `ml/`, notebooks + research notes in `docs/`. These dirs are mostly gitignored to keep the repo small. |

## Key Runtime Details
- **Endpoints**:
  - `POST /api/ingest` (list or single flow) -> writes `Flow` rows, enforces retention, enqueues scoring.
  - `GET /api/flows` -> last `MAX_FLOWS` rows ordered oldest-first; falls back to canned demo flows when DB empty.
  - `GET /api/anomalies` -> simple stats derived from current buffer (count, max score, pct > 0.9) or demo stats.
  - `POST /api/simulate` -> generates synthetic flows (botnet flood, data exfiltration, port scan) and scores inline with `DetectionEngine`.
- **Database**: Default DSN `postgresql://adns:adns_password@127.0.0.1/adns`. Tables: `flows` (timestamp/src/dst/proto/bytes/extra JSON) and `predictions` (flow_id, score, label, created_at). Use `init_db()` to set up schema and ensure `extra` exists.
- **Queues**: Redis defaults to `redis://127.0.0.1:6379/0`. Queue names, batch size, and timeouts are configurable via env (`ADNS_RQ_QUEUE`, `ADNS_RQ_BATCH_SIZE`, `ADNS_RQ_JOB_TIMEOUT`).
- **Agent expectations**: Requires `/usr/bin/tshark`, runs with privileges on `eth0`, posts JSON that already includes inferred service + HTTP/DNS metadata so the API can stash it in `flows.extra`.
- **Retention**: Controlled by `ADNS_FLOW_RETENTION_MINUTES` (default 30) and `ADNS_FLOW_RETENTION_MAX_ROWS` (default 5000), purged during ingest/simulate paths.

## Dev Commands & Checks
- **API**:
  ```bash
  cd api
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  export FLASK_APP=app.py
  export SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI:-postgresql://adns:adns_password@127.0.0.1/adns}
  export ADNS_REDIS_URL=${ADNS_REDIS_URL:-redis://127.0.0.1:6379/0}
  flask run  # serves /health on 5000
  ```
- **Worker**: `source api/.venv/bin/activate && python api/worker.py` (honors same env vars; runs the RQ loop).
- **Agent**: `cd agent && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && sudo ./capture.py`. Override `API_URL` when pointing at staging/prod.
- **Frontend**: `cd frontend/adns-frontend && npm install && npm run dev` (hot reload) or `npm run build && npm run preview` for production bundle served via Nginx. `dist/` is deployed to `/root/ADNS/frontend/adns-frontend/dist`.
- **ML**: `cd ml && pip install -r requirements.txt` then run preprocess/train scripts (see README for exact commands). Copy resulting `.joblib` files into `api/model_artifacts/`.
- **Testing**: Use `pytest` inside `api/` and `ml/`; run `npx vitest run` for frontend. Keep coverage near 80% on touched modules and mock external systems (Redis, PostgreSQL, tshark) in unit tests.

## Operational Notes
- Production services run out of `/var/www/adns/api/app.py` via Gunicorn/Nginx; keep this repo in sync when deploying.
- `deployment/` contains systemd units and shell helpers for agent/API/worker. Update these when paths/env vars change.
- Secrets/DSNs live in `.env` (gitignored). Rotate any placeholder passwords before sharing images or demos.
- When changing schema or models, plan for zero-downtime deploys: run migrations (or `init_db`) before restarting Gunicorn/RQ so `/ingest` never sees missing columns.
- When the user signs off on a change set, automatically redeploy the relevant service(s) and sync the repo state (git commit/push or whatever workflow is configured) without waiting for another prompt.
