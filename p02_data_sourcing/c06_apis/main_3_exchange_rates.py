"""
Time series from a REST API: euro foreign exchange reference rates via the Frankfurter API.

Frankfurter (https://frankfurter.dev) is a free, open-source API without key that serves the daily euro
foreign exchange reference rates published by the European Central Bank (ECB). Two endpoints are used:
    /latest                         -> the most recent rates
    /<start>..<end>?symbols=...     -> a time series (business days only; the ECB publishes around 16:00 CET)

The time-series response is a nested dictionary keyed by date:
    {"base": "EUR", "start_date": "...", "end_date": "...", "rates": {"2024-01-02": {"USD": 1.0956, ...}, ...}}

The script
    1. calls both endpoints (with retries, timeouts and a file cache),
    2. converts the nested dictionary into a DataFrame indexed by date (and shows the json_normalize way),
    3. computes daily returns, monthly averages and the volatility per currency,
    4. stores the series as Parquet.

How to run (from the repository root):
    python -m p02_data_sourcing.c06_apis.main_3_exchange_rates
    python -m p02_data_sourcing.c06_apis.main_3_exchange_rates --start 2024-01-01 --end 2024-12-31 --symbols USD,GBP,DKK

API version: the script uses Frankfurter's v1 endpoints (https://api.frankfurter.dev/v1). Upstream, v1 is frozen
(kept running for existing clients, but deprecated) and v2 is the current version for new projects. Do not assume
that v2 is a drop-in replacement: check its documentation for the endpoint paths and the response format, and
adapt the URLs built in main() and rates_to_frame() before pointing --base-url at it. If the v1 host moves, pass another base URL with
--base-url (older documentation uses https://api.frankfurter.app).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from p02_data_sourcing.c06_apis.utils.http_client import ApiError, build_session, get_json

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "cache_frankfurter")


def rates_to_frame(payload: dict) -> pd.DataFrame:
    """{'rates': {date: {currency: rate}}} -> DataFrame (index = date, one column per currency)."""
    df = pd.DataFrame.from_dict(payload["rates"], orient="index")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


# Frankfurter v1: frozen/deprecated upstream but still served; v2 is current (see the module docstring)
def main(start: str = "2025-01-01", end: str = "2025-06-30", symbols: str = "USD,GBP,DKK,SEK,JPY",
         base_url: str = "https://api.frankfurter.dev/v1") -> int:
    """Run the example. Returns a process exit code (0 = success)."""
    session = build_session()
    try:
        # 1. Latest rates and a time series
        print("\n *** 1. Calling the API *** ")
        latest = get_json(session, f"{base_url}/latest", params={"symbols": symbols}, cache_dir=PATH_CACHE, ttl_s=3600)
        print(f"Latest ({latest.data['date']}, base {latest.data['base']}): {latest.data['rates']}")
        series = get_json(session, f"{base_url}/{start}..{end}", params={"symbols": symbols},
                          cache_dir=PATH_CACHE, ttl_s=24 * 3600)
    except ApiError as e:
        print(f"API call failed: {e}")
        return 1

    # 2. Nested JSON -> DataFrame
    print("\n *** 2. Nested JSON to DataFrame *** ")
    df = rates_to_frame(series.data)
    print(df.head())
    print(f"... {len(df)} business days from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    # the same with json_normalize: every date becomes a column 'rates.<date>.<currency>' - for this
    # shape the dictionary approach above is clearly the better tool
    wide = pd.json_normalize(series.data)
    print(f"json_normalize alternative gives 1 row x {wide.shape[1]} columns, e.g. {list(wide.columns[4:6])}")

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
    parser.add_argument("--symbols", default="USD,GBP,DKK,SEK,JPY", help="comma-separated currency codes")
    parser.add_argument("--base-url", default="https://api.frankfurter.dev/v1",
                        help="API base URL (v1 endpoints; v1 is frozen upstream, check the v2 docs before switching)")
    args = parser.parse_args()
    sys.exit(main(start=args.start, end=args.end, symbols=args.symbols, base_url=args.base_url))
