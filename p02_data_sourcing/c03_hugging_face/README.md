## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml`
- Docker compose setup: none
- Environment variables (optional): `HF_TOKEN` for gated or private datasets (never hard-code it),
  `HF_HUB_OFFLINE=1` to work from the local cache only. Public datasets need no token.

## How to run
Run the scripts from the repository root (internet access to https://huggingface.co is needed unless stated):
- `python -m p02_data_sourcing.c03_hugging_face.main_1_load_and_stream` - load *rotten_tomatoes* in regular
  mode (features, random access, split slicing) and stream *imdb* (take/shuffle/filter/map).
- `python -m p02_data_sourcing.c03_hugging_face.main_2_process_and_save` - map, filter, stratified
  train/validation/test split, pandas statistics, `save_to_disk`/`load_from_disk` and Parquet export.
  Works offline with `--offline` (a tiny built-in dataset with the same schema) and falls back to it
  automatically when the Hub cannot be reached.
- `python -m p02_data_sourcing.c03_hugging_face.main_3_hub_files` - search the Hub, inspect a dataset
  repository, download its dataset card and one data file with `huggingface_hub` and read it with pandas.
  Another repository: `--repo-id stanfordnlp/imdb`.

Caches and outputs are stored in `p02_data_sourcing/c03_hugging_face/outputs/` (git-ignored).
