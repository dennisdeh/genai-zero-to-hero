"""
A small SQLite-backed store for scheduled ingestion jobs.

Design choices that make a scheduled job safe to re-run (idempotent):
    - a natural primary key (source, location, observed_at): the same observation can never be stored twice,
    - INSERT ... ON CONFLICT DO NOTHING: duplicates are silently ignored instead of raising,
    - every run is recorded in a 'runs' table (start, end, status, counts, error) for monitoring,
    - each write happens in one transaction: either the whole batch is stored or nothing.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    source          TEXT NOT NULL,
    location        TEXT NOT NULL,
    observed_at     TEXT NOT NULL,
    temperature_c   REAL,
    wind_speed_kmh  REAL,
    ingested_at     TEXT NOT NULL,
    PRIMARY KEY (source, location, observed_at)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job         TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    n_fetched   INTEGER DEFAULT 0,
    n_inserted  INTEGER DEFAULT 0,
    error       TEXT
);
"""


def utc_now() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ObservationStore:
    """Thin wrapper around an SQLite database file (safe to use from APScheduler's worker threads)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """
        Open a connection, commit on success / roll back on error, and always close it.
        A new connection per call keeps things thread-safe; WAL mode allows readers while a job writes.
        """
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:  # transaction: commit or rollback
                yield conn
        finally:
            conn.close()

    def start_run(self, job: str) -> int:
        """Record the start of a run and return its id."""
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO runs (job, started_at, status) VALUES (?, ?, 'running')", (job, utc_now()))
            return cur.lastrowid

    def finish_run(self, run_id: int, status: str, n_fetched: int = 0, n_inserted: int = 0, error: str | None = None) -> None:
        """Record the outcome of a run."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at=?, status=?, n_fetched=?, n_inserted=?, error=? WHERE run_id=?",
                (utc_now(), status, n_fetched, n_inserted, error, run_id),
            )

    def insert_observations(self, records: list[dict]) -> int:
        """Insert records idempotently in one transaction; return the number of new rows."""
        rows = [
            (r["source"], r["location"], r["observed_at"], r["temperature_c"], r["wind_speed_kmh"], utc_now())
            for r in records
        ]
        with self._connect() as conn:  # one transaction: all rows or none
            before = conn.total_changes
            conn.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (source, location, observed_at) DO NOTHING",
                rows,
            )
            return conn.total_changes - before

    def read(self, table: str = "observations") -> pd.DataFrame:
        """Read a whole table into a DataFrame."""
        if table not in ("observations", "runs"):
            raise ValueError(f"Unknown table {table}")
        with self._connect() as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)
