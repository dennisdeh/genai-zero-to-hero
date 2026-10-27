## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml` (needs `requests`, `urllib3`, `pandas`, `pyarrow`)
- Docker compose setup: none
- Environment variables (optional): `GITHUB_TOKEN` raises the GitHub rate limit from 60 to 5,000 requests per
  hour (never hard-code it). Open-Meteo and Frankfurter need no key.

## How to run
Run the scripts from the repository root (internet access to the respective API is required; responses are
cached in `outputs/`, so a repeated run within the cache lifetime works offline):
- `python -m p02_data_sourcing.c06_apis.main_1_open_meteo_weather` - hourly forecasts for four cities from
  https://api.open-meteo.com, `json_normalize` of the metadata, tidy hourly table, daily summary. Option: `--days 7`.
- `python -m p02_data_sourcing.c06_apis.main_2_github_pagination` - issues and pull requests of a repository
  from the GitHub REST API, following `Link` headers, flattening nested JSON, rate-limit headers.
  Options: `--repo scikit-learn/scikit-learn --max-pages 2 --per-page 50`.
- `python -m p02_data_sourcing.c06_apis.main_3_exchange_rates` - ECB euro reference rates from the Frankfurter
  API, nested JSON to a time-series DataFrame, returns and volatility.
  Options: `--start 2024-01-01 --end 2024-12-31 --symbols USD,GBP,DKK --base-url https://api.frankfurter.dev/v1`.

The reusable HTTP client (session, retries, timeouts, error handling, caching) is in `utils/http_client.py`.
Outputs are stored in `p02_data_sourcing/c06_apis/outputs/` (git-ignored).
