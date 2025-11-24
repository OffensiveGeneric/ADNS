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

## Quickstart

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
