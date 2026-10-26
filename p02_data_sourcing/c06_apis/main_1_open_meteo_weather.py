"""
Consuming a public REST API without a key: weather forecasts from Open-Meteo (https://open-meteo.com).

Open-Meteo offers free weather forecasts and historical weather data for non-commercial use, without
registration. A forecast request is a simple GET with query parameters:

    https://api.open-meteo.com/v1/forecast?latitude=55.68&longitude=12.57&hourly=temperature_2m&forecast_days=3

The response is JSON with metadata (coordinates, elevation, timezone, units) and "column-oriented" data:
    {"hourly": {"time": [...], "temperature_2m": [...], ...}, "hourly_units": {...}, ...}

The script
    1. builds a session with retries and calls the API once per city (with a file cache in outputs/),
    2. flattens the metadata of all responses with pd.json_normalize,
    3. turns the column-oriented hourly data into a long ("tidy") DataFrame with proper datetimes,
    4. aggregates it to daily statistics per city and stores the data as Parquet.

How to run (from the repository root):
    python -m p02_data_sourcing.c06_apis.main_1_open_meteo_weather
    python -m p02_data_sourcing.c06_apis.main_1_open_meteo_weather --days 7

Please respect the API's fair-use terms: cache results and do not poll more often than needed.
"""

import argparse
import os
import sys

import pandas as pd

from p02_data_sourcing.c06_apis.utils.http_client import ApiError, build_session, get_json

URL = "https://api.open-meteo.com/v1/forecast"
PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "cache_open_meteo")
CITIES = {
    "Copenhagen": (55.6761, 12.5683),
    "Stockholm": (59.3293, 18.0686),
    "Berlin": (52.52, 13.405),
    "Madrid": (40.4168, -3.7038),
}
VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]


def fetch_forecasts(days: int) -> dict[str, dict]:
    """Request the hourly forecast for every city; responses are cached for 30 minutes."""
    session = build_session()
    responses = {}
    for city, (lat, lon) in CITIES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(VARIABLES),
            "forecast_days": days,
            "timezone": "auto",  # timestamps in the local time of the location
        }
        resp = get_json(session, URL, params=params, cache_dir=PATH_CACHE, ttl_s=1800)
        print(f"  {city:10s} -> {len(resp.data['hourly']['time'])} hourly values{' (cache)' if resp.from_cache else ''}")
        responses[city] = resp.data
    return responses


def to_long_frame(responses: dict[str, dict]) -> pd.DataFrame:
    """Convert column-oriented hourly arrays into one tidy DataFrame (one row per city and hour)."""
    frames = []
    for city, data in responses.items():
        df = pd.DataFrame(data["hourly"])  # dict of equally long lists -> columns
        df["time"] = pd.to_datetime(df["time"])  # local time of the location (see 'timezone' metadata)
        df.insert(0, "city", city)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(days: int = 3) -> int:
    """Run the example. Returns a process exit code (0 = success)."""
    # 1. Fetch
    print(f"\n *** 1. Fetching {days}-day hourly forecasts *** ")
    try:
        responses = fetch_forecasts(days)
    except ApiError as e:
        print(f"API call failed: {e}")
        return 1

    # 2. Metadata: json_normalize flattens nested dicts into columns such as 'hourly_units.temperature_2m'
    print("\n *** 2. Response metadata (pd.json_normalize) *** ")
    meta = pd.json_normalize([{k: v for k, v in d.items() if k != "hourly"} for d in responses.values()])
    meta.insert(0, "city", list(responses))
    cols = ["city", "latitude", "longitude", "elevation", "timezone", "generationtime_ms", "hourly_units.temperature_2m",
            "hourly_units.precipitation"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(meta[[c for c in cols if c in meta.columns]].to_string(index=False))

    # 3. Hourly data as a tidy DataFrame
    print("\n *** 3. Hourly data *** ")
    df = to_long_frame(responses)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.head(), f"\n... {len(df):,} rows")

    # 4. Daily aggregation and storage
    print("\n *** 4. Daily summary per city *** ")
    daily = (
        df.assign(date=df["time"].dt.date)
        .groupby(["city", "date"])
        .agg(t_min=("temperature_2m", "min"), t_max=("temperature_2m", "max"),
             precip_mm=("precipitation", "sum"), wind_max=("wind_speed_10m", "max"))
        .round(1)
    )
    print(daily.to_string())
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path = os.path.join(PATH_OUTPUTS, "open_meteo_hourly.parquet")
    df.to_parquet(path, index=False)
    print(f"\nSaved hourly data to {path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=3, help="number of forecast days (1-16)")
    args = parser.parse_args()
    sys.exit(main(days=args.days))
