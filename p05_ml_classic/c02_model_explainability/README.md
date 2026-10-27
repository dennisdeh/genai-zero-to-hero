## Environment
- Python conda environment to use: `../conda_env_gzh_ml0.yml` (part-level environment `gzh_ml0`, includes `shap`),
  e.g. `conda env create -f p05_ml_classic/conda_env_gzh_ml0.yml && conda activate gzh_ml0`
- Minimum versions (pinned in the yml): lightgbm >= 4.7 (`LGBMClassifier.fit(eval_X=..., eval_y=...)`),
  scikit-learn >= 1.8.
- Docker compose setup: none needed
- Environment variables: none needed. The scripts use the synthetic churn dataset from
  `p05_ml_classic/c01_gradient_boosted_trees/utils/churn_data.py`.

## How to run
Run the scripts from the repository root:
```bash
python -m p05_ml_classic.c02_model_explainability.main_1_shap               # SHAP: bar, beeswarm, dependence, waterfall (~10 s)
python -m p05_ml_classic.c02_model_explainability.main_2_permutation_pdp    # MDI vs permutation importance, PDP/ICE (~20 s)
```
Plots are written to `p05_ml_classic/c02_model_explainability/outputs/` (git-ignored).
