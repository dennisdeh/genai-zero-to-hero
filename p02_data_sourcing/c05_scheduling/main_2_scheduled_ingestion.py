"""
A scheduled data-ingestion job with APScheduler: fetch -> validate -> store idempotently -> export.

This is the skeleton of many real data-sourcing pipelines: a process wakes up regularly, pulls fresh data
from a source (an API, a web page, a database), stores what is new and periodically publishes a snapshot.
The job must survive failures of the source, must never create duplicates when it is re-run, and must leave
a trail (logs, run table) that tells you what happened.

The script
    1. sets up logging to the console and to outputs/ingestion/ingestion.log,
    2. creates an SQLite store with a natural primary key (see utils/store.py),
    3. schedules an *interval* job that fetches observations every --interval seconds from
       - "synthetic": offline weather-like data (default; can simulate failures with --failure-rate), or
       - "open-meteo": the current weather from the free Open-Meteo API (no API key needed),
    4. schedules a *cron* job that exports the store to Parquet (every 20 seconds in the demo; in production
       this would be e.g. hour=2, minute=0 for a nightly export),
    5. stops after --duration seconds and prints the stored data and the run log.

With --once, a single ingestion run is executed without the scheduler - this is the mode you would call
from cron or any external scheduler (see README.md / book_scheduling.md for an example crontab line).

How to run (from the repository root):
    python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion
    python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion --duration 30 --failure-rate 0.3
    python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion --source open-meteo --interval 60 --duration 180
    python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion --once
"""

import argparse
import logging
import os
import sys
import threading
from datetime import datetime

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from p02_data_sourcing.c05_scheduling.utils.sources import open_meteo_current, synthetic_weather
from p02_data_sourcing.c05_scheduling.utils.store import ObservationStore

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", "ingestion")
PATH_DB = os.path.join(PATH_OUTPUTS, "observations.sqlite")
PATH_SNAPSHOT = os.path.join(PATH_OUTPUTS, "observations_snapshot.parquet")
PATH_LOG = os.path.join(PATH_OUTPUTS, "ingestion.log")

logger = logging.getLogger("ingestion")


def setup_logging() -> None:
    """Log INFO to the console and to a log file (the file is what you inspect when a job ran at 3 am)."""
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(PATH_LOG, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def validate(records: list[dict]) -> list[dict]:
    """Keep only plausible records (basic data-quality gate before anything is stored)."""
    valid = [
        r
        for r in records
        if r.get("observed_at")
        and r.get("temperature_c") is not None
        and -60 <= r["temperature_c"] <= 60
        and (r.get("wind_speed_kmh") is None or 0 <= r["wind_speed_kmh"] <= 400)
    ]
    if len(valid) < len(records):
        logger.warning("dropped %d invalid record(s)", len(records) - len(valid))
    return valid


def ingest(store: ObservationStore, source: str, failure_rate: float = 0.0) -> None:
    """
    One ingestion run. All exceptions are caught and logged: a failing source must not kill the scheduler,
    and the next scheduled run simply tries again.
    """
    run_id = store.start_run(f"ingest:{source}")
    try:
        if source == "open-meteo":
            records = open_meteo_current()
        else:
            records = synthetic_weather(bucket_s=10, failure_rate=failure_rate)
        records = validate(records)
        n_new = store.insert_observations(records)
        store.finish_run(run_id, "success", n_fetched=len(records), n_inserted=n_new)
        logger.info("run %d: fetched %d, inserted %d new (%d duplicates ignored)", run_id, len(records), n_new, len(records) - n_new)
    except Exception as e:  # noqa: BLE001
        store.finish_run(run_id, "failed", error=f"{type(e).__name__}: {e}")
        logger.error("run %d failed: %s: %s", run_id, type(e).__name__, e)


def export_snapshot(store: ObservationStore) -> None:
    """Export the whole store to Parquet (written to a temporary file first, then atomically renamed)."""
    df = store.read("observations")
    tmp = PATH_SNAPSHOT + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, PATH_SNAPSHOT)  # readers never see a half-written file
    logger.info("snapshot exported: %d rows -> %s", len(df), os.path.basename(PATH_SNAPSHOT))


def report(store: ObservationStore) -> None:
    """Print a summary of the stored data and of the run log."""
    obs = store.read("observations")
    runs = store.read("runs")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
        print(f"\nLatest observations ({len(obs)} rows in total):")
        print(obs.sort_values("observed_at").tail(6).to_string(index=False))
        print(f"\nRun log (last 8 of {len(runs)}):")
        print(runs.tail(8).to_string(index=False))
    print(f"\nRuns by status: {runs['status'].value_counts().to_dict()}")
    print(f"Files: {PATH_DB}, {PATH_SNAPSHOT}, {PATH_LOG}")


def main(source: str = "synthetic", interval_s: int = 3, duration_s: int = 25, failure_rate: float = 0.2, once: bool = False) -> None:
    """Run the ingestion once or on a schedule."""
    # 1. Logging and store
    setup_logging()
    store = ObservationStore(PATH_DB)

    # 2. Single run (for cron / external schedulers)
    if once:
        ingest(store, source, failure_rate=0.0)
        export_snapshot(store)
        return

    # 3. Scheduled runs
    scheduler = BlockingScheduler(job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 30})
    scheduler.add_job(
        ingest, "interval", seconds=interval_s, args=[store, source, failure_rate], id="ingest",
        next_run_time=datetime.now(),  # also run immediately at start-up
    )
    scheduler.add_job(export_snapshot, "cron", second="*/20", args=[store], id="export")  # production: hour=2, minute=0
    # stop after duration_s: a timer thread calls shutdown(), which makes the blocking start() return
    # (shutting down from *inside* a job would race with the scheduler's own bookkeeping)
    stopper = threading.Timer(duration_s, scheduler.shutdown, kwargs={"wait": False})
    stopper.daemon = True
    logger.info("starting scheduler: source=%s, every %ds, for %ds (Ctrl+C to stop)", source, interval_s, duration_s)
    try:
        stopper.start()
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        stopper.cancel()
    export_snapshot(store)  # final export so the snapshot is complete
    logger.info("scheduler stopped")

    # 4. What happened?
    report(store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "open-meteo"], default="synthetic", help="data source")
    parser.add_argument("--interval", type=int, default=3, help="seconds between ingestion runs")
    parser.add_argument("--duration", type=int, default=25, help="seconds to run before shutting down")
    parser.add_argument("--failure-rate", type=float, default=0.2, help="simulated failure probability (synthetic)")
    parser.add_argument("--once", action="store_true", help="run a single ingestion and exit (for cron)")
    args = parser.parse_args()
    main(source=args.source, interval_s=args.interval, duration_s=args.duration, failure_rate=args.failure_rate, once=args.once)
