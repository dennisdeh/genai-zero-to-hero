## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml`
- Docker compose setup: none
- Environment variables / credentials (never hard-code them, never commit them):
  - `KAGGLE_API_TOKEN` (new-style token) or the token file `~/.kaggle/access_token`, or
  - `KAGGLE_USERNAME` and `KAGGLE_KEY`, or the legacy file `~/.kaggle/kaggle.json` (`chmod 600`), or
  - an OAuth login with `kaggle auth login`.

  Create tokens on https://www.kaggle.com under Settings -> API. For competitions, accept the competition rules
  on the website first. Without credentials the scripts stop with an explanation of how to set them up;
  `kagglehub` can usually download public datasets without credentials.

## How to run
Run the scripts from the repository root (internet access to kaggle.com is required):
- `python -m p02_data_sourcing.c01_kaggle.main_1_kaggle_api` - search, inspect and download the dataset
  `uciml/iris` with the official `kaggle` client, unzip it safely and summarise it.
  Competition data: `python -m p02_data_sourcing.c01_kaggle.main_1_kaggle_api --competition titanic --target Survived`
- `python -m p02_data_sourcing.c01_kaggle.main_2_kagglehub` - the same with `kagglehub` in one call.
  Options: `--handle owner/name[/versions/N]`, `--competition titanic --target Survived`.

Downloads are stored in `p02_data_sourcing/c01_kaggle/outputs/` (git-ignored).
