## Environment
- Python conda environment to use: `../conda_env_gzh_ml0.yml` (part-level environment `gzh_ml0`),
  e.g. `conda env create -f p05_ml_classic/conda_env_gzh_ml0.yml && conda activate gzh_ml0`
- Docker compose setup: none needed
- Environment variables: none needed. The synthetic churn dataset is generated on the fly by
  `utils/churn_data.py` (also used by chapters c02 and c03).
- Optional: `optuna` for the Bayesian-optimisation step of `main_3_hyperparameter_search.py` (included in the
  conda environment; the step is skipped if the package is missing).

## How to run
Run the scripts from the repository root:
```bash
python -m p05_ml_classic.c01_gradient_boosted_trees.main_1_boosting_from_scratch    # ~25 s
python -m p05_ml_classic.c01_gradient_boosted_trees.main_2_gbt_comparison           # ~10 s
python -m p05_ml_classic.c01_gradient_boosted_trees.main_3_hyperparameter_search    # ~30 s
python -m p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data                # peek at the dataset
```
Plots are written to `p05_ml_classic/c01_gradient_boosted_trees/outputs/` (git-ignored). The models use one thread
each, which is fastest for this data size; increase `N_JOBS` in `main_2_gbt_comparison.py` for large datasets.
