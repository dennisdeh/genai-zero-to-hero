## MLOps with MLflow
Training a good model is only the beginning. In production we need to know *which* model is running, how it was
trained (code, data, hyperparameters), how it compares with alternatives, how to deploy and roll back a version,
and when the model needs retraining because the world has changed. MLOps is the set of practices and tools that
make the machine learning life cycle reproducible and automated:

1. **experiment tracking** - record every training run with its parameters, metrics and artefacts;
2. **model registry** - a catalogue of named, versioned models with lifecycle metadata;
3. **deployment / serving** - package a model with its dependencies and expose it (batch or REST API);
4. **monitoring** - watch the input data and the model's performance, and trigger retraining.

We use **MLflow**, an open-source platform that covers all four steps, is framework agnostic (scikit-learn,
XGBoost, LightGBM, PyTorch, LLMs, ...) and runs fully locally. The chapter continues with the churn example of
c01/c02.

Scripts (run them in order):

- `main_1_experiment_tracking.py`: logs five candidate models (logistic regression, random forest, two
  HistGradientBoosting configurations) and compares them with `mlflow.search_runs`.
- `main_2_model_registry.py`: registers the best model, sets the `@champion` alias, loads it by alias and by version
  and prints serving instructions.
- `main_3_drift_monitoring.py`: simulates weekly production data with increasing drift, computes PSI/KS drift
  metrics, and runs a champion/challenger retraining when an alert fires.

### Tracking server, backend store and artifact store
MLflow separates two kinds of storage:

- the **backend store** for metadata - experiments, runs, parameters, metrics, tags and the model registry. This can be
  a local folder (`./mlruns`) or a database. The model registry requires a **database-backed** store, so we use SQLite:
  `outputs/mlflow.db`;
- the **artifact store** for files - models, plots, data samples - here the folder `outputs/mlartifacts/` (in a
  team setting typically S3, Azure Blob or GCS).

`utils/mlflow_utils.py` sets the tracking URI with `mlflow.set_tracking_uri("sqlite:///.../outputs/mlflow.db")` and
creates the experiment with an explicit `artifact_location`. In a team, one would instead start a shared tracking
server, e.g. `mlflow server --backend-store-uri postgresql://... --artifacts-destination s3://...`, and point
`MLFLOW_TRACKING_URI` at it; the client code stays the same.

### Experiment tracking
An **experiment** groups related **runs**; a run is one execution of training code. Inside
`with mlflow.start_run(run_name=...)` the script logs

- **parameters** (`mlflow.log_params`): hyperparameters and data sizes - immutable inputs of the run;
- **tags** (`mlflow.set_tags`): searchable metadata such as the model family, a data version and the purpose
  (`stage = model-selection`);
- **metrics** (`mlflow.log_metrics`): validation ROC-AUC, log-loss, accuracy, F1 and the fit time. A metric can also
  have a history: the validation loss of every boosting iteration is logged with `step=i` and shown as a curve in the UI;
- **artifacts**: a ROC-curve figure (`mlflow.log_figure`) and a JSON file with the feature list (`mlflow.log_dict`);
- **the model** (`mlflow.sklearn.log_model`) together with
  - a **signature** inferred with `infer_signature(X_train, model.predict_proba(...))`: the names and types of the
    input columns and the shape of the output. It documents the model's interface and is enforced when the model is
    served - a request with a missing column fails loudly instead of producing garbage;
  - an **input example** (the first three training rows), shown in the UI and used to validate serving;
  - `pyfunc_predict_fn="predict_proba"` so that the generic Python-function interface (and the REST API) returns
    probabilities instead of hard labels;
  - `skops_trusted_types`: recent MLflow versions save scikit-learn models with **skops** instead of pickle by
    default. Unpickling can execute arbitrary code, so loading untrusted pickles is a security risk; skops only
    reconstructs types that are explicitly trusted. Plain estimators are trusted by default, the low-level tree
    structures of random forests and HistGradientBoosting are declared explicitly.

MLflow additionally records the git commit, the source file, the user and the start/end time of the run. Many
libraries can also be logged with a single line, `mlflow.sklearn.autolog()` (or `mlflow.xgboost.autolog()`,
`mlflow.lightgbm.autolog()`), at the price of less control over what exactly is recorded.

Runs are queried like a database table: `mlflow.search_runs(experiment_names=[...],
filter_string="tags.stage = 'model-selection'", order_by=["metrics.val_roc_auc DESC"])` returns a pandas DataFrame
with one row per run and columns such as `params.learning_rate`, `metrics.val_roc_auc` and `tags.model_family`.
The script prints a leaderboard; on the churn data the HistGradientBoosting model with learning rate 0.05 and 15
leaves wins (validation ROC-AUC about 0.844), followed by the random forest and the logistic regressions.

### Model registry
The registry gives models a **name** (`churn-classifier`) and sequential **versions**; each version points to the
model artifact of the run that produced it, so the lineage *model version -> run -> parameters, metrics, code, data
version* is never lost. `main_2_model_registry.py`

1. selects the best run with `search_runs`;
2. evaluates that model **once** on the untouched test set and logs `test_roc_auc` to the same run (re-opened with
   `mlflow.start_run(run_id=...)`);
3. registers it: `mlflow.register_model(model_uri, "churn-classifier")` creates a new version; descriptions and tags
   (e.g. `validated = true`, the test metrics) are added with `MlflowClient`;
4. sets the **alias** `champion` on that version: `client.set_registered_model_alias(name, "champion", version)`.
   Aliases are mutable, named pointers to versions (they replace the older, fixed stages *Staging/Production*).
   Deployment code refers to `models:/churn-classifier@champion`; promoting a new model or rolling back is a
   one-line alias change, without touching the consumers;
5. loads the model by alias and by version (`models:/churn-classifier/2`) through the generic
   `mlflow.pyfunc.load_model` interface, which works for any flavour, and checks that the predictions agree.

### Serving
A logged MLflow model is a self-describing folder (an `MLmodel` file listing the flavours, the signature, the
dependencies in `requirements.txt`/`conda.yaml`/`python_env.yaml`, and the serialised model). Therefore it can be
served without writing any code:
```bash
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/p05_ml_classic/c03_mlops/outputs/mlflow.db
mlflow models serve -m "models:/churn-classifier@champion" -p 5001 --env-manager local
```
`--env-manager local` uses the current Python environment; without it MLflow builds an isolated environment from
the logged requirements (virtualenv via uv/pyenv, or conda). The server exposes `/ping`, `/health`, `/version` and
`/invocations`:
```bash
curl -X POST http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' \
     -d '{"dataframe_split": {"columns": ["tenure_months", "monthly_charges", "total_charges",
          "contract", "payment_method", "has_fibre", "support_calls", "age", "num_products", "noise"],
          "data": [[21.0, 72.28, 1490.68, 0.0, 3.0, 0.0, 1.0, 69.0, 1.0, 0.84]]}}'
# -> {"predictions": [[0.572, 0.428]]}
```
For containerised deployments, `mlflow models build-docker -m "models:/churn-classifier@champion" -n churn-model`
builds a Docker image with the same REST API (see part p01 for Docker and Kubernetes), and
`mlflow models predict` scores a file in batch. The runs, metric curves, artefacts and registered models are browsed
in the web UI:
```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/p05_ml_classic/c03_mlops/outputs/mlflow.db --port 5000
```

### Monitoring, data drift and retraining
A model is trained on a snapshot of the world. After deployment the input distribution $P(X)$ can change
(**data drift** or covariate shift - e.g. a marketing campaign brings in many new, short-tenure customers), and the
relation $P(y\mid X)$ can change (**concept drift** - e.g. customers become more price sensitive). Both degrade the
model, but labels (did the customer actually churn?) often arrive weeks later, so we monitor the inputs immediately
and the performance when labels become available.

Two common drift statistics per feature:

- **Population Stability Index (PSI).** Bin the feature (deciles of the reference data for continuous features,
  categories for discrete ones) and compare the fractions $r_b$ (reference) and $c_b$ (current):
  $$\text{PSI} = \sum_b (c_b - r_b)\ln\frac{c_b}{r_b}.$$
  This is a symmetrised Kullback-Leibler divergence. Rule of thumb: below 0.1 stable, 0.1-0.2 moderate shift,
  above 0.2 significant shift.
- **Kolmogorov-Smirnov test**: the maximum distance between the two empirical CDFs, with a p-value. With large
  batches even tiny, irrelevant shifts become "significant", so effect sizes such as PSI are usually more useful for
  alerting than p-values.

`main_3_drift_monitoring.py` loads the `@champion` model and simulates five weekly batches of 4,000 customers with
increasing drift (the generator's `drift` argument shortens tenure, raises prices and the fibre share, and
strengthens price sensitivity). For each week it logs the PSI of every feature, the maximum PSI, the churn rate and the
ROC-AUC (once labels are known) to an MLflow run called `monitoring`, using the week as `step`, so the UI shows the
drift over time. A typical output:

| week | drift | churn rate | ROC-AUC | max PSI (feature) | alert |
|---|---|---|---|---|---|
| 1 | 0.0 | 0.31 | 0.843 | 0.003 (noise) | - |
| 2 | 0.1 | 0.33 | 0.839 | 0.02 (tenure) | - |
| 3 | 0.3 | 0.35 | 0.847 | 0.16 (tenure) | - |
| 4 | 0.6 | 0.45 | 0.826 | 0.46 (tenure) | DRIFT, PERF |

In week 4 the PSI of `tenure_months` exceeds 0.2 and the ROC-AUC drops by more than 0.02 below the reference, so the
**retraining trigger** fires:

1. half of the labelled week-4 batch is combined with the original training data, and a fresh copy of the champion
   pipeline (`sklearn.base.clone`: same model class and hyperparameters) is trained on it, logged as a nested run
   and registered as a new version with the alias `@challenger`;
2. champion and challenger are compared on the other half of the recent batch (data that neither has seen);
3. if the challenger is better (e.g. 0.847 vs 0.844 ROC-AUC), the `@champion` alias is moved to it. The previous
   version stays in the registry, so a rollback is a single alias change.

The demo also shows a subtlety: after retraining, the reference distribution for drift detection must be updated
as well - otherwise the drift alerts keep firing (week 5) although the new model has already adapted. Real systems
add more safeguards: a minimum amount of new labelled data, statistical tests for the champion/challenger comparison,
shadow deployments or A/B tests before promotion, and scheduled (e.g. monthly) retraining in addition to
alert-based triggers. Scheduling such a job with APScheduler or cron is covered in part p02; dedicated
monitoring tools (e.g. Evidently) provide ready-made drift reports.
