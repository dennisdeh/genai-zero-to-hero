"""
A small local "data lake": landing raw data, curating it into partitioned Parquet and keeping a catalog.

Real data platforms (S3/ADLS/GCS + Spark/Databricks/BigQuery, ...) follow the same basic layout that we
build here on the local file system:

    outputs/lake/
        raw/transactions/ingest_date=YYYY-MM-DD/batch_<hash>.json.gz   <- immutable, exactly as received
        curated/transactions/date=YYYY-MM-DD/part-0.parquet             <- cleaned, typed, partitioned
        gold/daily_revenue.parquet                                      <- aggregated, ready for reporting
        catalog.sqlite                                                  <- which batch was ingested when

The script demonstrates:
    1. landing raw batches (JSON records, as they would arrive from an API) unchanged in a "raw" zone,
    2. idempotent ingestion: a batch that was already ingested (same content hash) is skipped,
    3. curation: parsing, typing and de-duplicating records and writing them as Hive-partitioned Parquet,
       where re-processing a day overwrites that day's partition instead of appending duplicates,
    4. querying the lake with partition filters (pandas/pyarrow) and SQL (SQLite via SQLAlchemy).

How to run (from the repository root):
    python -m p02_data_sourcing.c00_storage.main_2_local_data_lake
    python -m p02_data_sourcing.c00_storage.main_2_local_data_lake --reset   # start with an empty lake

Run it twice without --reset to see that all batches are recognised as already ingested.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from p02_data_sourcing.c00_storage.utils.synthetic_data import make_daily_batch

PATH_LAKE = os.path.join(os.path.dirname(__file__), "outputs", "lake")
PATH_RAW = os.path.join(PATH_LAKE, "raw", "transactions")
PATH_CURATED = os.path.join(PATH_LAKE, "curated", "transactions")
PATH_GOLD = os.path.join(PATH_LAKE, "gold")
PATH_CATALOG = os.path.join(PATH_LAKE, "catalog.sqlite")


def get_catalog(path: str = PATH_CATALOG) -> Engine:
    """Create (if needed) and return the SQLite catalog that records every ingested batch."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    batch_id    TEXT PRIMARY KEY,   -- sha256 of the raw payload
                    source      TEXT NOT NULL,
                    ingest_date TEXT NOT NULL,
                    raw_path    TEXT NOT NULL,
                    n_records   INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                )
                """
            )
        )
    return engine


def batch_hash(records: list[dict]) -> str:
    """Content hash of a batch: identical payloads always get the same id (basis for idempotency)."""
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def land_raw(records: list[dict], ingest_date: str, batch_id: str) -> str:
    """Write a batch unchanged (gzip-compressed JSON) into the raw zone and return its path."""
    path_dir = os.path.join(PATH_RAW, f"ingest_date={ingest_date}")
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, f"batch_{batch_id[:12]}.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    return path


def curate(records: list[dict]) -> pd.DataFrame:
    """
    Turn raw JSON records into a clean, typed DataFrame.

    Parameters:
    records: list[dict]
        Raw records as received.

    Return:
        DataFrame with proper dtypes, a 'date' partition column and no duplicate transaction ids.
    """
    df = pd.DataFrame.from_records(records)
    expected = {"transaction_id", "ts", "customer_id", "category", "amount", "country", "is_refund", "note"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Schema check failed, missing columns: {sorted(missing)}")
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%dT%H:%M:%S")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_refund"] = df["is_refund"].astype(bool)
    df = df.dropna(subset=["amount"]).drop_duplicates(subset="transaction_id", keep="last")
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    return df


def write_curated(df: pd.DataFrame) -> None:
    """
    Write curated data as a Hive-partitioned Parquet dataset (one folder per day).
    'delete_matching' replaces the partitions that are being written, so re-processing a day
    overwrites it instead of creating duplicates - this keeps the curated zone idempotent.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=PATH_CURATED,
        format="parquet",
        partitioning=["date"],
        partitioning_flavor="hive",
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
    )


def ingest_batch(engine: Engine, records: list[dict], ingest_date: str, source: str = "shop_api") -> bool:
    """
    Ingest one batch end-to-end (raw -> curated -> catalog), skipping it if it was seen before.

    Return:
        True if the batch was ingested, False if it was skipped as a duplicate.
    """
    batch_id = batch_hash(records)
    with engine.connect() as conn:
        seen = conn.execute(text("SELECT 1 FROM ingestion_log WHERE batch_id = :b"), {"b": batch_id}).first()
    if seen:
        print(f"  [skip]   batch {batch_id[:12]} ({ingest_date}) already ingested")
        return False

    raw_path = land_raw(records, ingest_date, batch_id)
    df = curate(records)
    write_curated(df)
    with engine.begin() as conn:  # the catalog entry is written only after the data is safely stored
        conn.execute(
            text(
                "INSERT INTO ingestion_log VALUES (:batch_id, :source, :ingest_date, :raw_path, :n, :at)"
            ),
            {
                "batch_id": batch_id,
                "source": source,
                "ingest_date": ingest_date,
                "raw_path": os.path.relpath(raw_path, PATH_LAKE),
                "n": len(records),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
    print(f"  [ingest] batch {batch_id[:12]} ({ingest_date}): {len(df):,} rows -> partitions {sorted(df['date'].unique())}")
    return True


def build_gold(engine: Engine) -> pd.DataFrame:
    """Aggregate the curated zone into a daily revenue table (Parquet + SQLite table for SQL users)."""
    df = pd.read_parquet(PATH_CURATED, columns=["date", "country", "amount", "is_refund"])
    df["signed_amount"] = df["amount"].where(~df["is_refund"], -df["amount"])
    gold = (
        df.groupby(["date", "country"], observed=True)
        .agg(n_transactions=("amount", "size"), revenue=("signed_amount", "sum"))
        .round(2)
        .reset_index()
    )
    gold["date"] = gold["date"].astype(str)
    os.makedirs(PATH_GOLD, exist_ok=True)
    gold.to_parquet(os.path.join(PATH_GOLD, "daily_revenue.parquet"), index=False)
    gold.to_sql("daily_revenue", engine, index=False, if_exists="replace")
    return gold


def print_tree(path: str, max_files: int = 30) -> None:
    """Print the folder structure of the lake (like the Unix 'tree' command, abbreviated)."""
    n = 0
    for root, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        level = os.path.relpath(root, path).count(os.sep) + (0 if root == path else 1)
        print("    " * level + os.path.basename(root) + "/")
        for f in sorted(files):
            print("    " * (level + 1) + f)
            n += 1
            if n >= max_files:
                print("    ...")
                return


def main(reset: bool = False) -> None:
    """Run the data lake demonstration."""
    # 0. Initialisation
    if reset and os.path.exists(PATH_LAKE):
        shutil.rmtree(PATH_LAKE)
        print(f"Removed existing lake at {PATH_LAKE}")
    engine = get_catalog()

    # 1. Simulate batches arriving over three days (as if pulled from an API once per day)
    print("\n *** 1. Ingesting daily batches *** ")
    days = ["2025-03-01", "2025-03-02", "2025-03-03"]
    for day in days:
        ingest_batch(engine, make_daily_batch(day), ingest_date=day)

    # 2. The source re-delivers the batch of day 2 (happens all the time with retries) -> skipped
    print("\n *** 2. Re-delivery of an identical batch (idempotency) *** ")
    ingest_batch(engine, make_daily_batch("2025-03-02"), ingest_date="2025-03-04")

    # 3. A corrected batch for day 1 arrives (different content) -> its partition is overwritten
    print("\n *** 3. Corrected batch for an earlier day (partition overwrite) *** ")
    corrected = make_daily_batch("2025-03-01", n_rows=1_500, seed=7)
    ingest_batch(engine, corrected, ingest_date="2025-03-04")

    # 4. Query the curated zone: partition pruning means only the matching folders are read
    print("\n *** 4. Querying the curated zone *** ")
    df_all = pd.read_parquet(PATH_CURATED)
    print(f"Curated rows per day (day 1 now holds the corrected batch):\n{df_all.groupby('date').size().to_string()}")
    df_day = pd.read_parquet(PATH_CURATED, filters=[("date", "=", "2025-03-03")], columns=["category", "amount"])
    print(f"\nRead only partition date=2025-03-03: {len(df_day):,} rows")
    print(df_day.groupby("category", observed=True)["amount"].describe().round(2))

    # 5. Gold layer + SQL access through the catalog database
    print("\n *** 5. Gold layer and SQL *** ")
    build_gold(engine)
    with engine.connect() as conn:
        top = pd.read_sql(
            text("SELECT date, SUM(revenue) AS revenue, SUM(n_transactions) AS n FROM daily_revenue GROUP BY date"),
            conn,
        )
        log = pd.read_sql(text("SELECT substr(batch_id, 1, 12) AS batch, ingest_date, n_records, raw_path FROM ingestion_log"), conn)
    engine.dispose()
    print(f"Daily revenue (SQL on the gold table):\n{top.round(2).to_string(index=False)}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(f"\nIngestion log (catalog):\n{log.to_string(index=False)}")

    # 6. Show the layout
    print("\n *** 6. Lake layout *** ")
    print_tree(PATH_LAKE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the lake before running")
    args = parser.parse_args()
    main(reset=args.reset)
