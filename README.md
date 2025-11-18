# ADNS — Anomaly Detection Network System

ADNS is an end-to-end demo of a modern network anomaly detection platform. It ingests live packet captures, stores recent flows in PostgreSQL, scores them with a trained ML model, and visualizes detections on a React dashboard with built-in attack simulations for classroom demos.

## Architecture

| Component | Path | Description |
| --- | --- | --- |
| Packet capture agent | `agent/` | `capture.py` wraps `tshark` on `eth0`, batches flow metadata, and POSTs to `/api/ingest`. |
| Flask API | `api/` | Persists flows/predictions, exposes `/flows`, `/anomalies`, `/simulate`, and backs Nginx+Gunicorn. |
| Scoring worker | `api/worker.py` | Polls for unscored flows and applies the trained model (`api/model_artifacts/`). |
| Frontend dashboard | `frontend/adns-frontend/` | Vite/React UI with anomaly charts, severity donut, and attack simulation buttons. |
| ML lab | `ml/` | Preprocessing scripts (`preprocess/`), meta-model notebooks, and `train_flow_detector.py` for the live scorer. |
| Ops | `deployment/`, `worker/`, `assets/` | Systemd units, scripts, and misc assets. Research docs live in `docs/`. |

Generated datasets live under `data/`, and derived artifacts (clean CSVs, model outputs) live under `outputs/`; both are gitignored to keep the repo lean.

## Quickstart

### 1. Backend / API

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=app.py
export SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI:-postgresql://adns:adns_password@127.0.0.1/adns}
flask run
```

The API exposes:

- `POST /ingest` — ingest flow JSON (single object or list).
- `GET /flows` & `GET /anomalies` — dashboard data feeds.
- `POST /simulate` — synthesize attack traffic (used by the UI buttons).

### 2. Worker

```bash
source api/.venv/bin/activate
python api/worker.py        # or use systemd unit adns-worker.service
```

The worker loads `api/model_artifacts/flow_detector.joblib`, so keep that directory in sync with the latest training output.

### 3. Frontend

```bash
cd frontend/adns-frontend
npm install
npm run dev   # for hot reload
npm run build && npm run preview   # for production bundle
```

Building places static assets under `dist/`; the production droplet serves that folder via Nginx at `http://159.203.105.167/`.

### 4. Training & Data Pipelines

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

Copy the resulting artifacts into `/var/www/adns/api/model_artifacts/` (or wherever Gunicorn runs) and restart `adns-worker` to pick up the new model.

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
