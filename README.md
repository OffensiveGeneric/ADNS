# ADNS — Anomaly Detection Network System

ADNS is an end-to-end demo of a modern network anomaly detection platform. It ingests live packet captures, stores recent flows in PostgreSQL, pushes scoring jobs over Redis/RQ to a DetectionEngine (meta ensemble → sklearn → heuristics), and visualizes detections on a React dashboard with built-in attack simulations for classroom demos.

## Architecture
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3c972f97-f751-4c92-9d10-fb54f326c4b3" />


| Component | Path | Description |
| --- | --- | --- |
| Packet capture agent | `agent/` | `capture.py` wraps `tshark`, normalizes packet metadata into flow JSON, and POSTs batches to `/api/ingest`. |
| Flask API | `api/` | Persists flows/predictions, exposes `/flows`, `/anomalies`, `/simulate`, and enqueues new flow IDs on Redis/RQ. |
| Redis task queue | `api/task_queue.py`, `api/tasks.py` | RQ helpers that push flow IDs to `flow_scores` and score them inside app context. |
| Scoring worker | `api/worker.py` | RQ worker bootstrap; consumes `flow_scores` jobs and drives the DetectionEngine. |
| Frontend dashboard | `frontend/adns-frontend/` | Vite/React UI with anomaly charts, severity donut, and attack simulation buttons. |
| ML lab | `ml/` | Preprocessing scripts (`preprocess/`), meta-model notebooks, and `train_flow_detector.py` for the live scorer. |
| Model artifacts | `api/model_artifacts/` | `meta_model_combined.joblib` (ExtraTrees+XGBoost) + `flow_detector.joblib` (sklearn pipeline). |
| Ops | `deployment/`, `worker/`, `assets/` | Systemd units, scripts, and misc assets. Research docs live in `docs/`. |

Generated datasets live under `data/`, and derived artifacts (clean CSVs, model outputs) live under `outputs/`; both are gitignored to keep the repo lean.

## Quickstart — Docker first
Prereqs: Docker + Docker Compose, Git.

```bash
git clone https://github.com/OffensiveGeneric/ADNS.git
cd ADNS
docker compose up --build -d          # API:5000, Frontend:8080, Postgres, Redis, worker
```

- Frontend: `http://localhost:8080`
- API health: `curl http://localhost:5000/health`
- Demo traffic: `curl -X POST http://localhost:5000/simulate -H 'Content-Type: application/json' -d '{"type":"botnet_flood","count":50}'`
- Streaming demo traffic (background): `curl -X POST http://localhost:5000/simulate -H 'Content-Type: application/json' -d '{"type":"botnet_flood","duration_seconds":120,"interval_seconds":1}'`
- Live capture (Linux only): `docker compose --profile agent up -d agent` (uses host network + NET_ADMIN; set `INTERFACE`/`API_URL` in `docker-compose.yml` if needed).

### Local dev (bare metal, optional)
If you prefer running services directly:
- macOS/Linux: `./scripts/setup_local.sh` then start API/worker/agent/frontend with the commands in `AGENTS.md`.
- Windows: `pwsh ./scripts/setup_local.ps1` then use the PowerShell commands in `AGENTS.md`.

Databases:
- Postgres default: `SQLALCHEMY_DATABASE_URI=postgresql://adns:adns_password@127.0.0.1/adns`
- SQLite (no install): set `SQLALCHEMY_DATABASE_URI=sqlite:///./adns.db` in `.env`

## Docker Compose (dev stack)
- Build and run API, worker, frontend, Postgres, and Redis: `docker compose up --build` (from repo root). API on `http://localhost:5000`, frontend on `http://localhost:8080`.
- Frontend build arg: override `VITE_API_URL` if you want a different API origin (default `http://localhost:5000`); e.g., `docker compose build --build-arg VITE_API_URL=http://api:5000 frontend`.
- Optional capture agent: `docker compose --profile agent up --build agent` (Linux only, uses `network_mode: host` and `NET_ADMIN` so `tshark` can see host traffic). On macOS/Windows, run the agent on the host instead and point `API_URL` at `http://localhost:5000/ingest`.
- Persistent Postgres data lives in the `pgdata` volume; remove it with `docker volume rm adns_pgdata` if you need a clean slate.
- Redis runs in-memory; queueing can be disabled by stopping the worker container (API will fall back to inline scoring).
- Common fixes:
  - macOS AirPlay can own port 5000; if `curl localhost:5000/health` returns 403 AirTunes, change the API port mapping (e.g., `5100:5000`), restart compose, and point agent/frontend at the new port.
  - If the UI cannot reach the API, rebuild the frontend with the right base: `docker compose build --no-cache --build-arg VITE_API_URL=http://127.0.0.1:5000 frontend && docker compose up -d frontend` (or `VITE_API_URL=""` to use the nginx `/api` proxy). Verify with `curl http://localhost:8080/api/health`.

### Run locally to monitor your own traffic

1) Install system deps: PostgreSQL (or use SQLite via `SQLALCHEMY_DATABASE_URI=sqlite:///./adns.db`), Redis (optional; inline scoring fallback works if Redis is down), `tshark`, Python 3.9+, Node.js 18+.  
2) Bootstrap the repo: `./scripts/setup_local.sh` on macOS/Linux or `pwsh ./scripts/setup_local.ps1` on Windows (creates `.venv`, installs API+agent deps, runs `npm install`, and copies `.env.example` to `.env` if missing).  
3) Edit `.env` as needed:
   - Want Postgres? Install it, run `./scripts/setup_postgres_local.sh` (or `pwsh ./scripts/setup_postgres_local.ps1` on Windows) to create the `adns` database/user, then set `SQLALCHEMY_DATABASE_URI` to the printed URL.
   - `SQLALCHEMY_DATABASE_URI` can be set to `sqlite:///./adns.db` for a zero-install database.
   - `VITE_API_URL` only if the frontend will call the API on a different origin.
   - `API_URL`, `INTERFACE`, `BATCH_SIZE`, etc. to control the capture agent.
   - `ADNS_RDNS_ENABLED` and related knobs to include reverse-DNS resolution as a scoring feature.
4) Run services (separate terminals):
   - API: `source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && cd api && flask run`
   - Worker (optional if relying on inline scoring): `source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && python api/worker.py`
   - Agent (needs tshark + capture privileges): `source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && cd agent && sudo ./capture.py`
   - Frontend: `cd frontend/adns-frontend && export $(grep -v '^#' ../../.env | xargs) && npm run dev -- --host`
   - On Windows/PowerShell: use `.\.venv\Scripts\Activate.ps1` instead of `source ...`, drop `sudo`, and run the agent from an elevated shell so `tshark` can capture.
   - On WSL with Docker Desktop: `sudo apt-get install -y tshark` inside WSL, then run the agent with explicit paths: `API_URL=http://127.0.0.1:5000/ingest TSHARK_BIN=/usr/bin/tshark INTERFACE=eth0 sudo .venv/bin/python agent/capture.py`. If tshark permissions are already set, drop `sudo`.
   - On macOS with Docker: avoid AirPlay port 5000 conflicts by using the mapped API port (e.g., 5001). Run the agent with preserved envs: `sudo env TSHARK_BIN=/usr/local/bin/tshark API_URL=http://127.0.0.1:5001/ingest INTERFACE=en0 .venv/bin/python agent/capture.py`. If you’ve run Wireshark’s ChmodBPF helper, you can omit `sudo`.

### 0. Dependencies

- PostgreSQL (default URL `postgresql://adns:adns_password@127.0.0.1/adns`)
- Redis (default URL `redis://127.0.0.1:6379/0`) for the RQ job queue
- `tshark` on any host that runs the capture agent

The commands below assume those services are already running.

### 1. Backend / API

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=app.py
export SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI:-postgresql://adns:adns_password@127.0.0.1/adns}
export ADNS_REDIS_URL=${ADNS_REDIS_URL:-redis://127.0.0.1:6379/0}
flask run
```

The API exposes:

- `POST /ingest` — ingest flow JSON (single object or list).
- `GET /flows` & `GET /anomalies` — dashboard data feeds.
- `POST /simulate` — synthesize attack traffic (used by the UI buttons).
  - Accepts `count` for one-shot batches.
  - Accepts `duration_seconds` (and optional `interval_seconds`, default 1.0s) to stream batches in the background for the given duration.

On first run `init_db()` creates tables and adds the `flows.extra` JSON column so the agent’s rich metadata can be stored immediately.

### 2. Worker

```bash
source api/.venv/bin/activate
export FLASK_APP=app.py
export SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI:-postgresql://adns:adns_password@127.0.0.1/adns}
export ADNS_REDIS_URL=${ADNS_REDIS_URL:-redis://127.0.0.1:6379/0}
python api/worker.py        # or use systemd unit adns-worker.service
```

This boots an RQ worker that listens on `flow_scores`, loads the DetectionEngine (meta ensemble → sklearn → heuristics), and writes `Prediction` rows for each flow ID it dequeues.

### 3. Packet capture agent

```bash
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # only pulls requests
export API_URL=${API_URL:-http://127.0.0.1:5000/ingest}
sudo ./capture.py                 # needs privileges for the interface
```

The agent wraps `tshark`, infers services, batches ~50 flows or 2 seconds, and POSTs them to the API. Production deployments run it under `systemd` (`adns-agent.service`) so it survives reboots.

### 4. Frontend

```bash
cd frontend/adns-frontend
npm install
npm run dev   # for hot reload
npm run build && npm run preview   # for production bundle
```

Building places static assets under `dist/`. Set `VITE_API_URL` before `npm run build` if the UI is hosted separately; the production droplet serves that folder via Nginx at `http://159.203.105.167/`.

### 5. Training & Data Pipelines

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Preprocess UNSW-NB15 CSVs into features
python preprocess/merge_and_clean.py \
  --data_dir ../data/DataSet/UNSW-NB15/"Training and Testing Sets" \
  --out_dir ../outputs/preprocessed

# Train meta models (ExtraTrees + XGBoost ensemble)
python meta/meta_train.py --raw_train ... --raw_test ... --clean_dir ../outputs/preprocessed --out_dir ../outputs/meta

# Train the lightweight flow detector used in production
python train_flow_detector.py \
  --raw_train ../data/DataSet/UNSW-NB15/"Training and Testing Sets"/UNSW_NB15_training-set.csv \
  --raw_test  ../data/DataSet/UNSW-NB15/"Training and Testing Sets"/UNSW_NB15_testing-set.csv \
  --model_out ../api/model_artifacts/flow_detector.joblib
```

Copy the resulting artifacts (both `flow_detector.joblib` and `meta_model_combined.joblib`) into `/var/www/adns/api/model_artifacts/` (or wherever Gunicorn/RQ runs) and restart `adns-worker` so the DetectionEngine reloads them.

## Demo Tips

- Use the **Attack Simulation Controls** at the top of the dashboard to trigger Botnet Flood, Data Exfiltration, or Port Scan scenarios. They call `/api/simulate`, inject synthetic flows, and immediately refresh the charts/donut.
- The **Threat Timeline** and **Severity Mix** donut help narrate how the model responds as traffic changes.
- `POST /api/simulate` can also be driven via scripts/cURL for automation:

```bash
curl -X POST http://localhost:5000/simulate -H 'Content-Type: application/json' \
     -d '{"type":"botnet_flood","count":80}'
```

## Contributing

- Python code follows PEP 8; React follows the Vite ESLint defaults.
- Add tests near the subsystem you touch (`api/tests`, `ml/tests`, `frontend/.../__tests__`).
- Keep secrets in `.env` (already gitignored), and add any new large/generated directories to `.gitignore`.
- Use short imperative commit messages and include screenshots or metrics when changing UI/ML behavior.

Questions? See `AGENTS.md` for contributor guidelines or open an issue. Happy hunting!
