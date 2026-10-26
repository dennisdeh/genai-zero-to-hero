"""
Small helpers that generate realistic-looking synthetic data for the storage examples.

The data mimics card transactions of an online shop: a timestamp, a customer, a product category,
an amount, a country, a refund flag and a free-text note that is sometimes missing. It deliberately
contains the data types that tend to cause trouble when data is stored and read back again:
timestamps, categoricals, booleans, missing values and free text.
"""

import numpy as np
import pandas as pd

CATEGORIES = ["books", "electronics", "garden", "groceries", "toys", "clothing"]
COUNTRIES = ["DK", "DE", "SE", "NO", "NL", "FR", "GB"]
NOTES = ["gift", "express delivery", "pick-up in store", "discount code used", None]


def make_transactions(
    n_rows: int = 100_000,
    start: str = "2025-01-01",
    days: int = 90,
    seed: int = 42,
    id_offset: int = 0,
) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic shop transactions.

    Parameters:
    n_rows: int
        Number of rows (transactions) to generate.
    start: str
        First calendar day (ISO format) of the generated period.
    days: int
        Length of the period in days; timestamps are spread uniformly over it.
    seed: int
        Seed for the random number generator (makes the data reproducible).
    id_offset: int
        Offset added to the transaction ids, useful when generating several batches.

    Return:
        DataFrame with the columns transaction_id, ts, customer_id, category, amount,
        country, is_refund and note.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    seconds = rng.integers(0, days * 24 * 3600, size=n_rows)
    ts = (t0 + pd.to_timedelta(np.sort(seconds), unit="s")).astype("datetime64[us]")

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(n_rows, dtype="int64") + id_offset,
            "ts": ts,
            "customer_id": rng.integers(1, 5_000, size=n_rows),
            "category": pd.Categorical(rng.choice(CATEGORIES, size=n_rows), categories=CATEGORIES),
            # log-normal amounts look like real prices: many small, few large purchases
            "amount": np.round(rng.lognormal(mean=3.5, sigma=0.8, size=n_rows), 2),
            "country": rng.choice(COUNTRIES, size=n_rows, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]),
            "is_refund": rng.random(n_rows) < 0.03,
            "note": pd.array(rng.choice(np.array(NOTES, dtype=object), size=n_rows), dtype="string"),
        }
    )
    return df


def make_daily_batch(day: str, n_rows: int = 2_000, seed: int | None = None) -> list[dict]:
    """
    Generate one day's worth of transactions as a list of JSON-like records, i.e. the shape
    in which data typically arrives from an API or a message queue.

    Parameters:
    day: str
        The calendar day (ISO format) of the batch.
    n_rows: int
        Number of records in the batch.
    seed: int, None
        Seed for reproducibility; if None it is derived from the day so that the same day
        always produces the same batch (handy to demonstrate idempotent ingestion).

    Return:
        List of dictionaries, one per transaction, with ISO formatted timestamps.
    """
    if seed is None:
        seed = int(pd.Timestamp(day).strftime("%Y%m%d"))
    id_offset = int(pd.Timestamp(day).strftime("%Y%m%d")) * 100_000
    df = make_transactions(n_rows=n_rows, start=day, days=1, seed=seed, id_offset=id_offset)
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["category"] = df["category"].astype(str)
    df["note"] = df["note"].astype(object).where(df["note"].notna(), None)
    return df.to_dict(orient="records")
