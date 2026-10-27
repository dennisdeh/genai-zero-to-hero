## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml` (create it with `conda env create -f p02_data_sourcing/conda_env_gzh_data0.yml`)
- Docker compose setup: none
- Environment variables: none. The examples run fully offline with synthetic data.

## How to run
Run the scripts from the repository root:
- `python -m p02_data_sourcing.c00_storage.main_1_file_formats` - compares CSV, CSV.gz, Parquet (snappy/zstd),
  Feather and SQLite: size, speed, type preservation, column projection, predicate push-down and SQL.
  Use `--n-rows 500000` for a larger table and `--repeats 5` for more stable timings (best run is reported).
- `python -m p02_data_sourcing.c00_storage.main_2_local_data_lake` - builds a small local data lake
  (raw JSON -> curated, partitioned Parquet -> gold table) with a SQLite catalog, idempotent ingestion
  and upserts of corrected and late-arriving records into their partitions.
  Run it twice to see that already ingested batches are skipped; `--reset` starts from an empty lake.

All output is written to `p02_data_sourcing/c00_storage/outputs/` (git-ignored).
The synthetic data generator lives in `utils/synthetic_data.py`.
