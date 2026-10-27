"""
Data sources for the scheduled ingestion example. Each source is a function returning a list of
observation records with the keys: source, location, observed_at, temperature_c, wind_speed_kmh.

    - synthetic_weather : offline, deterministic per time bucket (re-running within the same bucket returns
                          the same observations, which lets us demonstrate idempotent ingestion), and it
                          can simulate failures.
    - open_meteo_current: the current weather from the free Open-Meteo API (https://open-meteo.com, no key).
                          The API reports 15-minute values, so polling more often yields duplicates -
                          again handled by the idempotent store.
"""

import math
import random
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOCATIONS = {
    "Copenhagen": (55.6761, 12.5683),
    "Berlin": (52.52, 13.405),
    "London": (51.5072, -0.1276),
}


def synthetic_weather(bucket_s: int = 10, failure_rate: float = 0.0) -> list[dict]:
    """
    Generate one observation per location for the current time bucket.

    Parameters:
    bucket_s: int
        Width of a time bucket in seconds; all calls within one bucket return identical records.
    failure_rate: float
        Probability of raising a ConnectionError, to simulate an unreliable upstream system.
    """
    if random.random() < failure_rate:
        raise ConnectionError("simulated upstream failure")
    now = datetime.now(timezone.utc)
    bucket = int(now.timestamp()) // bucket_s * bucket_s
    observed_at = datetime.fromtimestamp(bucket, tz=timezone.utc).isoformat(timespec="seconds")
    records = []
    for i, location in enumerate(LOCATIONS):
        rng = random.Random(f"{location}-{bucket}")  # deterministic per (location, bucket)
        daily_cycle = 5 * math.sin(2 * math.pi * (bucket % 86_400) / 86_400)
        records.append(
            {
                "source": "synthetic",
                "location": location,
                "observed_at": observed_at,
                "temperature_c": round(10 + 2 * i + daily_cycle + rng.gauss(0, 0.5), 2),
                "wind_speed_kmh": round(abs(rng.gauss(15, 5)), 1),
            }
        )
    return records


def make_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """A requests session with retries and exponential back-off for transient errors."""
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "gzh-scheduled-ingestion/1.0 (educational example)"
    return session


def open_meteo_current(session: requests.Session | None = None, timeout_s: float = 10.0) -> list[dict]:
    """Fetch the current temperature and wind speed for all LOCATIONS from Open-Meteo."""
    session = session or make_session()
    records = []
    for location, (lat, lon) in LOCATIONS.items():
        resp = session.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m", "timezone": "UTC"},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        current = resp.json()["current"]
        records.append(
            {
                "source": "open-meteo",
                "location": location,
                "observed_at": current["time"],
                "temperature_c": current["temperature_2m"],
                "wind_speed_kmh": current["wind_speed_10m"],
            }
        )
    return records
