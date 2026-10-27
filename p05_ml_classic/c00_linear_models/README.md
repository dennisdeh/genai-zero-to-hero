## Environment
- Python conda environment to use: `../conda_env_gzh_ml0.yml` (part-level environment `gzh_ml0`),
  e.g. `conda env create -f p05_ml_classic/conda_env_gzh_ml0.yml && conda activate gzh_ml0`
- Docker compose setup: none needed
- Environment variables: none needed. All datasets (diabetes, breast cancer, two moons) ship with scikit-learn,
  so the scripts run offline.

## How to run
Run the scripts from the repository root (each finishes in well under a minute on a CPU):
```bash
python -m p05_ml_classic.c00_linear_models.main_1_linear_regression    # OLS, Ridge/Lasso/ElasticNet, residuals
python -m p05_ml_classic.c00_linear_models.main_2_logistic_regression  # pipeline, ROC/PR, calibration, threshold
python -m p05_ml_classic.c00_linear_models.main_3_trees_rf_svm         # trees, random forests, SVMs
```
Results are printed to the console; plots are written to `p05_ml_classic/c00_linear_models/outputs/`
(git-ignored). The theory and a walkthrough are in `book_linear_models.md`.
