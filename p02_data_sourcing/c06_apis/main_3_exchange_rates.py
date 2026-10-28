"""
Time series from a REST API: euro foreign exchange reference rates via the Frankfurter API (v2).

Frankfurter (https://frankfurter.dev) is a free, open-source API without key. Version 2 serves exchange rates
from many central banks and official sources; by default it blends them, and with `providers=ecb` it returns
only the euro reference rates published by the European Central Bank (ECB), which is what this example uses.
One endpoint is used, in two ways:
    /v2/rates?quotes=...&providers=ecb                    -> the most recent rates
    /v2/rates?from=...&to=...&quotes=...&providers=ecb    -> a time series (business days only; the ECB
                                                             publishes around 16:00 CET)

v2 answers with a flat list of records, one per date and currency pair ("long" format):
    [{"date": "2025-01-02", "base": "EUR", "quote": "USD", "rate": 1.0321}, ...]
(v1, which is frozen upstream, returned a dictionary keyed by date instead; see the book chapter.)

The script
    1. calls the endpoint twice (with retries, timeouts and a file cache),
    2. turns the list of records into a DataFrame and pivots it to one column per currency,
    3. computes daily returns, monthly averages and the volatility per currency,
    4. stores the series as Parquet.

How to run (from the repository root):
    python -m p02_data_sourcing.c06_apis.main_3_exchange_rates
    python -m p02_data_sourcing.c06_apis.main_3_exchange_rates --start 2024-01-01 --end 2024-12-31 --symbols USD,GBP,DKK
    python -m p02_data_sourcing.c06_apis.main_3_exchange_rates --providers ""     # blended rates of all providers

Limits of the API: a daily range filtered by provider may span at most 5 years and at most 5 quote currencies
(use the `group=week` / `group=month` parameter of the API for longer histories).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from p02_data_sourcing.c06_apis.utils.http_client import ApiError, build_session, get_json

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "cache_frankfurter")
BASE_URL = "https://api.frankfurter.dev/v2"


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """[{'date', 'base', 'quote', 'rate'}, ...] -> long DataFrame with a datetime 'date' column."""
    if not records:
        raise ValueError("the API returned no rates for this query")
    df = pd.DataFrame.from_records(records, columns=["date", "base", "quote", "rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def rates_to_frame(records: list[dict], start: str | None = None) -> pd.DataFrame:
    """
    Long records -> wide DataFrame (index = date, one column per quote currency).

    If the range starts on a day without ECB rates (weekend, holiday), the API carries the previous
    business day's rate forward and returns it with its own, earlier date. `start` drops such rows so the
    series covers exactly the requested period.
    """
    long = records_to_frame(records)
    if start is not None:
        long = long[long["date"] >= pd.Timestamp(start)]
    wide = long.pivot(index="date", columns="quote", values="rate").sort_index()
    wide.columns.name = None
    return wide


def main(start: str = "2025-01-01", end: str = "2025-06-30", symbols: str = "USD,GBP,DKK,SEK,JPY",
         providers: str = "ecb", base_url: str = BASE_URL) -> int:
    """Run the example. Returns a process exit code (0 = success)."""
    session = build_session()
    params = {"quotes": symbols}
    if providers:
        params["providers"] = providers
    try:
        # 1. Latest rates and a time series
        print("\n *** 1. Calling the API *** ")
        latest = get_json(session, f"{base_url}/rates", params=params, cache_dir=PATH_CACHE, ttl_s=3600)
        latest_df = records_to_frame(latest.data)
        print(f"Latest rates (base {latest_df['base'].iloc[0]}, source: {providers or 'blend of all providers'}):")
        print(latest_df.to_string(index=False))
        series = get_json(session, f"{base_url}/rates", params={**params, "from": start, "to": end},
                          cache_dir=PATH_CACHE, ttl_s=24 * 3600)
    except (ApiError, ValueError) as e:
        print(f"API call failed: {e}")
        return 1

    # 2. List of records -> DataFrame
    print("\n *** 2. JSON records to DataFrame *** ")
    long = records_to_frame(series.data)
    print(f"The API returns one record per date and currency ({len(long)} records), i.e. a tidy 'long' table:")
    print(long.head())
    # a flat list of records is exactly what pd.DataFrame / json_normalize expect; for analysis we pivot it
    # to one column per currency (the 'wide' format)
    df = rates_to_frame(series.data, start=start)
    print(f"\nPivoted to one column per currency:\n{df.head()}")
    print(f"... {len(df)} business days from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")

    # 3. Analysis
    print("\n *** 3. Analysis *** ")
    returns = np.log(df).diff().dropna()
    summary = pd.DataFrame(
        {
            "first": df.iloc[0],
            "last": df.iloc[-1],
            "change_%": ((df.iloc[-1] / df.iloc[0] - 1) * 100).round(2),
            "ann_volatility_%": (returns.std() * np.sqrt(252) * 100).round(2),
        }
    )
    print(summary)
    print(f"\nMonthly averages:\n{df.resample('MS').mean().round(4)}")

    # 4. Store
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path = os.path.join(PATH_OUTPUTS, f"eur_fx_{start}_{end}.parquet")
    df.to_parquet(path)
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2025-01-01", help="start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-06-30", help="end date YYYY-MM-DD")
    parser.add_argument("--symbols", default="USD,GBP,DKK,SEK,JPY",
                        help="comma-separated quote currencies (at most 5 for a daily range filtered by provider)")
    parser.add_argument("--providers", default="ecb",
                        help='rate provider(s), e.g. "ecb" (default) or "ecb,bis"; "" for the blend of all providers')
    parser.add_argument("--base-url", default=BASE_URL, help="API base URL (Frankfurter v2)")
    args = parser.parse_args()
    sys.exit(main(start=args.start, end=args.end, symbols=args.symbols, providers=args.providers,
                  base_url=args.base_url))
