# Repository Guidelines

## Project Structure & Module Organization
`ml/` houses the training + preprocessing stack (meta pipeline, `preprocess/` helpers, requirements). The Flask API lives in `api/` (`app.py`, `adns/` models) and reaches PostgreSQL through an env-supplied DSN. `frontend/adns-frontend` is the React/Vite dashboard, keeping code in `src/` and static files in `public/`. Research datasets and notebooks stay in `docs/`, while `assets/`, `deployment/`, `agent/`, and `worker/` store media, infra manifests, and automation entry points.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create a virtual env.
- `pip install -r ml/requirements.txt` — ML dependencies (NumPy, pandas, scikit-learn, xgboost, SHAP, dotenv).
- `pip install -r api/requirements.txt && FLASK_APP=app.py flask run --debug` (run inside `api/`) — serve `/health` on port 5000.
- `npm install && npm run dev` (`frontend/adns-frontend`) — Vite dev server; use `npm run build`, `npm run preview`, `npm run lint` for CI parity.
- Run pipelines directly, e.g., `python ml/meta/meta_train.py --raw_train data/UNSW_NB15_training-set.csv --out_dir outputs/meta`.

## Coding Style & Naming Conventions
Use PEP 8 for Python (4 spaces, snake_case, ALL_CAPS constants) and keep modules focused on one responsibility. Secrets such as `SQLALCHEMY_DATABASE_URI` belong in `.env` files loaded with `python-dotenv`. React code follows the Vite ESLint config: PascalCase components, camelCase hooks/utilities, colocated styles, and explicit prop typing. Favor descriptive filenames and docstrings for non-trivial ML steps.

## Testing Guidelines
Adopt `pytest` for API and modeling code, mirroring the package layout (`api/tests/test_health.py`, `ml/tests/test_preprocess.py`). Seed randomness, mock external I/O, and park golden CSV/JSON artifacts in `tests/fixtures/`. Frontend specs live in `frontend/adns-frontend/src/__tests__/` with a `.test.tsx` suffix; run them through `npx vitest run`. Target 80 % coverage on touched modules and note any intentionally skipped flows in the PR.

## Commit & Pull Request Guidelines
History favors short imperative subjects (“Add Flask API, worker, agent directories”), so keep that pattern and scope each commit to a single concern. Reference issue IDs or dataset revisions in the body when applicable. Pull requests need a summary, verification checklist (`pytest`, `npm run lint`, curl `/health`), screenshots for UI changes, and a callout for dependency or schema updates. Tag reviewers for the subsystem you altered.

## Security & Configuration Tips
Populate `.env` with DSNs (`SQLALCHEMY_DATABASE_URI`, upstream API roots) and keep it gitignored. Rotate the placeholder `strongpassword` before any deployment and inspect `deployment/` manifests for exposed ports or credentials. Scrub PII from datasets staged in `docs/` and prefer synthetic packet captures when sharing traces.
