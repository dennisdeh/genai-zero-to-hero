## Environment
- Python conda environment to use: `../conda_env_gzh_ml0.yml` (part-level environment `gzh_ml0`, includes `mlflow`),
  e.g. `conda env create -f p05_ml_classic/conda_env_gzh_ml0.yml && conda activate gzh_ml0`
- Docker compose setup: none needed. MLflow runs locally with a SQLite backend store
  (`outputs/mlflow.db`) and a local artifact store (`outputs/mlartifacts/`), both git-ignored.
- Environment variables: none needed for the scripts. For the MLflow CLI (serving) set
  `MLFLOW_TRACKING_URI=sqlite:///<absolute path to the repo>/p05_ml_classic/c03_mlops/outputs/mlflow.db`.

## How to run
Run the scripts in order from the repository root:
```bash
python -m p05_ml_classic.c03_mlops.main_1_experiment_tracking   # log 5 candidate models, compare runs (~1 min)
python -m p05_ml_classic.c03_mlops.main_2_model_registry        # register best model, alias @champion, load + predict
python -m p05_ml_classic.c03_mlops.main_3_drift_monitoring      # drift metrics, retraining trigger, champion/challenger
```
Re-running a script adds new runs/model versions; delete `p05_ml_classic/c03_mlops/outputs/` to start from scratch.

Browse experiments, runs and registered models in the MLflow UI (http://127.0.0.1:5000):
```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/p05_ml_classic/c03_mlops/outputs/mlflow.db --port 5000
```
Serve the champion model as a REST API and query it:
```bash
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/p05_ml_classic/c03_mlops/outputs/mlflow.db
mlflow models serve -m "models:/churn-classifier@champion" -p 5001 --env-manager local
# in another terminal:
curl -X POST http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' \
     -d '{"dataframe_split": {"columns": ["tenure_months", "monthly_charges", "total_charges", "contract", "payment_method", "has_fibre", "support_calls", "age", "num_products", "noise"], "data": [[21.0, 72.28, 1490.68, 0.0, 3.0, 0.0, 1.0, 69.0, 1.0, 0.84]]}}'
```
`main_2_model_registry.py` prints these commands with the paths and an example payload filled in.
