"""
Storing tabular data: CSV vs. Parquet vs. SQLite (and friends).

Once data has been sourced it has to be stored somewhere. This script writes the same synthetic
transactions table to several formats and compares them on what matters in practice:
    - file size on disk,
    - write and read time,
    - whether the data types survive a round trip (timestamps, categoricals, booleans, missing values),
    - selective reads: reading only some columns (column projection) and only some rows
      (predicate push-down / SQL WHERE).

Formats compared: CSV, gzip-compressed CSV, Parquet (snappy and zstd compression), Feather (Arrow IPC)
and an SQLite database accessed through SQLAlchemy.

How to run (from the repository root):
    python -m p02_data_sourcing.c00_storage.main_1_file_formats
    python -m p02_data_sourcing.c00_storage.main_1_file_formats --n-rows 500000

All files are written to p02_data_sourcing/c00_storage/outputs/formats/.
"""

import argparse
import os
import time
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

from p02_data_sourcing.c00_storage.utils.synthetic_data import make_transactions

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", "formats")


def timed(fn: Callable, *args, **kwargs) -> tuple[object, float]:
    """Run fn(*args, **kwargs) and return its result together with the elapsed wall time in seconds."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def size_mb(path: str) -> float:
    """Return the size of a file in megabytes."""
    return os.path.getsize(path) / 1024**2


def dtypes_preserved(original: pd.DataFrame, restored: pd.DataFrame) -> str:
    """
    Compare the dtypes of two DataFrames and return a short human-readable verdict,
    listing the columns whose dtype changed during the round trip.
    """
    changed = [
        f"{c}: {original[c].dtype}->{restored[c].dtype}"
        for c in original.columns
        if c in restored.columns and str(original[c].dtype) != str(restored[c].dtype)
    ]
    return "all preserved" if not changed else "; ".join(changed)


def benchmark_formats(df: pd.DataFrame, path_dir: str) -> pd.DataFrame:
    """
    Write df to every format, read it back and collect size, timings and a dtype verdict.

    Parameters:
    df: pd.DataFrame
        The table to store.
    path_dir: str
        Directory in which the files are written.

    Return:
        DataFrame with one row per format.
    """
    os.makedirs(path_dir, exist_ok=True)
    p_csv = os.path.join(path_dir, "transactions.csv")
    p_csv_gz = os.path.join(path_dir, "transactions.csv.gz")
    p_pq_snappy = os.path.join(path_dir, "transactions_snappy.parquet")
    p_pq_zstd = os.path.join(path_dir, "transactions_zstd.parquet")
    p_feather = os.path.join(path_dir, "transactions.feather")
    p_sqlite = os.path.join(path_dir, "transactions.sqlite")
    if os.path.exists(p_sqlite):
        os.remove(p_sqlite)
    engine = create_engine(f"sqlite:///{p_sqlite}")

    # each entry: name -> (path, write function, read function)
    formats = {
        "csv": (
            p_csv,
            lambda: df.to_csv(p_csv, index=False),
            lambda: pd.read_csv(p_csv),
        ),
        "csv.gz": (
            p_csv_gz,
            lambda: df.to_csv(p_csv_gz, index=False, compression="gzip"),
            lambda: pd.read_csv(p_csv_gz),
        ),
        "parquet (snappy)": (
            p_pq_snappy,
            lambda: df.to_parquet(p_pq_snappy, compression="snappy", index=False),
            lambda: pd.read_parquet(p_pq_snappy),
        ),
        "parquet (zstd)": (
            p_pq_zstd,
            # several row groups so that min/max statistics can be used to skip data (see step 3)
            lambda: df.to_parquet(p_pq_zstd, compression="zstd", index=False, row_group_size=25_000),
            lambda: pd.read_parquet(p_pq_zstd),
        ),
        "feather": (
            p_feather,
            lambda: df.to_feather(p_feather),
            lambda: pd.read_feather(p_feather),
        ),
        "sqlite": (
            p_sqlite,
            lambda: df.to_sql("transactions", engine, index=False, if_exists="replace", chunksize=50_000),
            lambda: pd.read_sql_table("transactions", engine),
        ),
    }

    rows = []
    for name, (path, write_fn, read_fn) in formats.items():
        _, t_write = timed(write_fn)
        restored, t_read = timed(read_fn)
        rows.append(
            {
                "format": name,
                "size_mb": round(size_mb(path), 2),
                "write_s": round(t_write, 3),
                "read_s": round(t_read, 3),
                "dtypes": dtypes_preserved(df, restored),
            }
        )
    engine.dispose()
    return pd.DataFrame(rows)


def selective_reads(path_dir: str) -> None:
    """
    Demonstrate reading only a subset of the data, which is where columnar formats and
    databases shine compared to CSV (which always has to be parsed in full).
    """
    p_csv = os.path.join(path_dir, "transactions.csv")
    p_pq = os.path.join(path_dir, "transactions_zstd.parquet")
    p_sqlite = os.path.join(path_dir, "transactions.sqlite")

    # a) Column projection: only two of eight columns are needed
    _, t_csv = timed(pd.read_csv, p_csv, usecols=["category", "amount"])
    _, t_pq = timed(pd.read_parquet, p_pq, columns=["category", "amount"])
    print(f"Read 2 columns  | CSV: {t_csv:.3f}s (still parses every line) | Parquet: {t_pq:.3f}s")

    # b) Predicate push-down: Parquet stores min/max statistics per row group, so whole row groups
    #    can be skipped. The data is sorted by time, which makes time filters very effective.
    flt = [("ts", ">=", pd.Timestamp("2025-03-15")), ("country", "==", "DK")]
    df_pq, t_pq = timed(pd.read_parquet, p_pq, filters=flt)
    print(f"Filter rows     | Parquet with filters (ts >= 2025-03-15, country == DK): {len(df_pq):,} rows in {t_pq:.3f}s")
    pf = pq.ParquetFile(p_pq)
    idx_ts = pf.schema_arrow.get_field_index("ts")
    for i in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(i).column(idx_ts).statistics
        print(f"    row group {i}: ts min={stats.min}  max={stats.max}")
    print("    -> row groups whose max(ts) < 2025-03-15 are skipped without being decompressed.")

    # c) The same question in SQL: the database does the filtering and aggregation for us
    engine = create_engine(f"sqlite:///{p_sqlite}")
    query = text(
        "SELECT category, COUNT(*) AS n, ROUND(SUM(amount), 2) AS revenue "
        "FROM transactions WHERE country = :country AND is_refund = 0 "
        "GROUP BY category ORDER BY revenue DESC"
    )
    with engine.connect() as conn:
        df_sql, t_sql = timed(pd.read_sql, query, conn, params={"country": "DK"})
    engine.dispose()
    print(f"SQL aggregation | SQLite GROUP BY in {t_sql:.3f}s:\n{df_sql.to_string(index=False)}")

    # d) Parquet files are self-describing: the schema and statistics live in the footer
    meta = pq.ParquetFile(p_pq).metadata
    print(
        f"\nParquet metadata: {meta.num_rows:,} rows, {meta.num_row_groups} row group(s), "
        f"{meta.num_columns} columns, created by '{meta.created_by}'"
    )
    print(pq.read_schema(p_pq))


def main(n_rows: int = 200_000) -> pd.DataFrame:
    """Run the comparison and print the results."""
    # 1. Generate the data
    print(f"\n *** 1. Generating {n_rows:,} synthetic transactions *** ")
    df = make_transactions(n_rows=n_rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.head())
    print(f"\nIn-memory dtypes:\n{df.dtypes.to_string()}")
    print(f"In-memory size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # 2. Write and read every format
    print("\n *** 2. Writing and reading each format *** ")
    results = benchmark_formats(df, PATH_OUTPUTS)
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(results.to_string(index=False))

    # 3. Selective reads
    print("\n *** 3. Selective reads (projection, filters, SQL) *** ")
    selective_reads(PATH_OUTPUTS)

    # 4. Take-aways
    print(
        "\n *** 4. Take-aways *** \n"
        "- CSV is universal and human readable, but large, slow and loses types (timestamps and\n"
        "  categoricals come back as strings unless you re-specify them with dtype=/parse_dates=).\n"
        "- Parquet is columnar, compressed and typed: the default choice for analytical data.\n"
        "- Feather/Arrow IPC is very fast for short-lived intermediate files.\n"
        "- SQLite gives you SQL, indexes and transactions in a single file: great for small\n"
        "  operational stores (e.g. appending records from a scheduled job), but its type system is\n"
        "  loose (categoricals are lost; dates are restored only thanks to SQLAlchemy reflection).\n"
        f"Files written to: {PATH_OUTPUTS}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rows", type=int, default=200_000, help="number of rows to generate")
    args = parser.parse_args()
    main(n_rows=args.n_rows)
