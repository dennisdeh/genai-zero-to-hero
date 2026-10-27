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
    3. curation: parsing, typing and de-duplicating records and upserting them into Hive-partitioned
       Parquet: late-arriving records are added to their (earlier) day, corrected records replace the
       old version with the same transaction id, nothing else in the partition is lost,
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
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from p02_data_sourcing.c00_storage.utils.synthetic_data import make_daily_batch

PATH_LAKE = os.path.join(os.path.dirname(__file__), "outputs", "lake")
PATH_RAW = os.path.join(PATH_LAKE, "raw", "transactions")
PATH_CURATED = os.path.join(PATH_LAKE, "curated", "transactions")
PATH_GOLD = os.path.join(PATH_LAKE, "gold")
PATH_CATALOG = os.path.join(PATH_LAKE, "catalog.sqlite")

# Explicit schema of the curated zone (schema-on-write): every partition file gets exactly these types,
# no matter what pandas inferred for a particular batch. The partition column 'date' lives in the folder name.
CURATED_SCHEMA = pa.schema(
    [
        ("transaction_id", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("customer_id", pa.int64()),
        ("category", pa.string()),
        ("amount", pa.float64()),
        ("country", pa.string()),
        ("is_refund", pa.bool_()),
        ("note", pa.string()),
    ]
)


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


def write_curated(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """
    Upsert curated rows into the Hive-partitioned Parquet dataset (one folder per event day).

    A batch can contain records for several days: late-arriving events for an earlier day, or corrections
    of records that were already stored. For every partition the batch touches, the existing rows are read,
    the new rows are appended and duplicates on the record id are dropped (keep="last": the newest version
    wins). The result replaces that partition's file; partitions the batch does not touch are neither read
    nor written. Applying the same batch twice gives the same result (idempotent), and - unlike a plain
    partition overwrite - rows that are not part of the batch are never lost.

    Parameters:
    df: pd.DataFrame
        Output of curate(), including the 'date' partition column.

    Return:
        Dictionary {date: (rows before, rows after)} for every touched partition.
    """
    stats = {}
    for date, df_new in df.groupby("date", sort=True):
        path_dir = os.path.join(PATH_CURATED, f"date={date}")
        path = os.path.join(path_dir, "part-0.parquet")
        df_part = df_new[CURATED_SCHEMA.names]
        n_before = 0
        if os.path.exists(path):
            df_old = pq.read_table(path).to_pandas()
            n_before = len(df_old)
            df_part = pd.concat([df_old, df_part], ignore_index=True)
        df_part = df_part.drop_duplicates(subset="transaction_id", keep="last").sort_values("ts", kind="stable")
        # write to a hidden temporary file first (readers ignore names starting with '.' or '_'),
        # then rename: os.replace is atomic, so readers see either the old or the new partition
        os.makedirs(path_dir, exist_ok=True)
        path_tmp = os.path.join(path_dir, ".part-0.parquet.tmp")
        pq.write_table(pa.Table.from_pandas(df_part, schema=CURATED_SCHEMA, preserve_index=False), path_tmp)
        os.replace(path_tmp, path)
        stats[str(date)] = (n_before, len(df_part))
    return stats


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
    stats = write_curated(df)
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
    changes = ", ".join(f"date={d}: {a:,} -> {b:,}" for d, (a, b) in stats.items())
    print(f"  [ingest] batch {batch_id[:12]} ({ingest_date}): {len(df):,} rows | partition rows {changes}")
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

    # 3. A corrected batch for day 1 arrives: 1,500 of its 2,000 records are re-sent with new values
    #    -> upsert: those 1,500 transaction ids are updated, the other 500 rows of day 1 stay untouched
    print("\n *** 3. Corrections for an earlier day (upsert by transaction_id) *** ")
    corrected = make_daily_batch("2025-03-01", n_rows=1_500, seed=7)
    ingest_batch(engine, corrected, ingest_date="2025-03-04")

    # 4. The batch of day 4 also carries 5 late-arriving events that happened on day 2
    #    -> they are added to the partition of day 2, which keeps its 2,000 existing rows
    print("\n *** 4. Late-arriving records (a batch that spans several partitions) *** ")
    late = make_daily_batch("2025-03-02", n_rows=5, seed=99)
    for record in late:
        record["transaction_id"] += 90_000  # ids that do not exist yet: new events, not corrections
    ingest_batch(engine, make_daily_batch("2025-03-04") + late, ingest_date="2025-03-04")

    # 5. Query the curated zone: partition pruning means only the matching folders are read
    print("\n *** 5. Querying the curated zone *** ")
    df_all = pd.read_parquet(PATH_CURATED)
    print(f"Curated rows per day (day 2 includes the 5 late events):\n{df_all.groupby('date', observed=True).size().to_string()}")
    n_dup = df_all["transaction_id"].duplicated().sum()
    print(f"Duplicate transaction ids in the curated zone: {n_dup}")
    df_day = pd.read_parquet(PATH_CURATED, filters=[("date", "=", "2025-03-03")], columns=["category", "amount"])
    print(f"\nRead only partition date=2025-03-03: {len(df_day):,} rows")
    print(df_day.groupby("category", observed=True)["amount"].describe().round(2))

    # 6. Gold layer + SQL access through the catalog database
    print("\n *** 6. Gold layer and SQL *** ")
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

    # 7. Show the layout
    print("\n *** 7. Lake layout *** ")
    print_tree(PATH_LAKE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the lake before running")
    args = parser.parse_args()
    main(reset=args.reset)
