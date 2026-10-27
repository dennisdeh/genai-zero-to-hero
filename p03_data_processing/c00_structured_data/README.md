# Structured data: cleaning, feature pipelines and data quality

## Environment
- Python conda environment to use: `p03_data_processing/conda_env_gzh_proc0.yml`
  (`conda env create -f p03_data_processing/conda_env_gzh_proc0.yml && conda activate gzh_proc0`)
- No docker setup, environment variables or internet access are needed: the data is generated
  synthetically by `utils/synthetic_data.py`.

## How to run
All scripts are run from the repository root. Outputs (CSV/parquet files, the fitted pipeline, data
quality reports) are written to an `outputs/` folder next to each script.

```bash
# s01: cleaning a messy customers/transactions dataset with pandas
python -m p03_data_processing.c00_structured_data.s01_preprocessing.main_1_cleaning
# s01: feature engineering + leakage-free sklearn pipeline for churn prediction
python -m p03_data_processing.c00_structured_data.s01_preprocessing.main_2_feature_pipeline

# s02: data quality report (pass/fail checks) on raw vs. clean data
python -m p03_data_processing.c00_structured_data.s02_dqa.main_1_quality_report
# s02: drift detection between data batches (PSI, KS test, chi-squared test)
python -m p03_data_processing.c00_structured_data.s02_dqa.main_2_drift_detection
# s02: declarative validation with pandera schemas
python -m p03_data_processing.c00_structured_data.s02_dqa.main_3_pandera_validation
```

Each script finishes within a few seconds on a laptop CPU.

## Structure
- `utils/synthetic_data.py`: generator of the clean and the messy online-shop data
- `s01_preprocessing/utils/cleaning.py`: cleaning helpers, `QuantileClipper` transformer, `load_clean_data()`
- `s01_preprocessing/utils/features.py`: customer-level features and the `ColumnTransformer`
- `s02_dqa/utils/checks.py`: reusable data quality checks and report helpers
- `s02_dqa/utils/drift.py`: PSI and drift report

The clean data (`load_clean_data()`) is also used by the Streamlit dashboard in
`p04_visualisation_and_applications/c01_streamlit_app`.
