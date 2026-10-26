## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml`
- Docker compose setup: none
- Environment variables: none. Downloading from OpenML needs no account or API key (a key is only needed
  for uploads). Internet access to https://www.openml.org is required.

## How to run
Run the scripts from the repository root:
- `python -m p02_data_sourcing.c02_open_ml.main_1_openml_search` - search OpenML for binary classification
  datasets with the `openml` package, download *credit-g* (id 31), inspect its metadata, qualities, features
  and tasks, and store it as Parquet. Another dataset: `--dataset-id 1590` (adult).
- `python -m p02_data_sourcing.c02_open_ml.main_2_fetch_openml_sklearn` - fetch *adult* (version 2) with
  `sklearn.datasets.fetch_openml`, inspect the returned Bunch, check data quality and train a quick baseline.
  Another dataset: `--name credit-g --version 1`.

Downloads and caches are stored in `p02_data_sourcing/c02_open_ml/outputs/` (git-ignored).
If OpenML cannot be reached, the scripts stop with a clear message.
