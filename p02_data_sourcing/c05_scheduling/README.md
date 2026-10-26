## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml` (needs `apscheduler` 3.x, `pandas`, `pyarrow`, `requests`)
- Docker compose setup: none
- Environment variables: none. The default synthetic source runs offline; `--source open-meteo` needs
  internet access to https://api.open-meteo.com (no API key).

## How to run
Run the scripts from the repository root:
- `python -m p02_data_sourcing.c05_scheduling.main_1_apscheduler_basics` - interval, cron and date triggers,
  `max_instances`, error events and listeners; stops after 20 s (`--duration 30` to change).
- `python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion` - scheduled, idempotent ingestion
  into SQLite with logging, error handling, a run log and a cron-triggered Parquet export; stops after 25 s.
  - more failures: `--failure-rate 0.4`
  - real data: `--source open-meteo --interval 60 --duration 300`
  - single run (for cron): `--once`

### Running with cron
Edit your crontab with `crontab -e` and add, for example (adjust both paths):
```text
*/15 * * * * cd /path/to/genai-zero-to-hero && /path/to/miniconda3/envs/gzh_data0/bin/python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion --once >> /tmp/gzh_ingestion_cron.log 2>&1
```
Use `flock -n /tmp/gzh_ingestion.lock <command>` to prevent overlapping runs.

Outputs (SQLite database, Parquet snapshot, log file) are stored in
`p02_data_sourcing/c05_scheduling/outputs/ingestion/` (git-ignored).
