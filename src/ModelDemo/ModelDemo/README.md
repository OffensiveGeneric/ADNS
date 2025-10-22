# ADNS Hybrid Model Demo

This demo contains a minimal hybrid anomaly detection pipeline (classical detectors + simple representation learner + meta-model).
Place your CSVs into `data/` then run `./run_demo.sh`.

See `run_demo.sh` for the default run flow and sample flags.
  
Run This First:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run This Second:
chmod +x run_demo.sh
./run_demo.sh

