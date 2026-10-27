"""
A client for the iris model-serving API (main_api.py) using the requests library.

The script calls every endpoint of the API and prints the responses:
1. liveness and readiness (/health, /ready)
2. model metadata (/v1/model/metadata)
3. a single prediction (/v1/predict)
4. a batch prediction (/v1/predict/batch)
5. an invalid request, to show the automatic input validation (HTTP 422)

How to run (from the repository root, with the API running, e.g. via
`uvicorn p00_fundamentals.c02_fastapi.main_api:app --port 5000`):
    python -m p00_fundamentals.c02_fastapi.main_client

The base URL can be changed with the API_URL environment variable, e.g. to call the containerised or
cloud-deployed service (see part p01): API_URL=https://my-service.example.com python -m ...
"""

import json
import os
import sys

import requests

from p00_fundamentals.c02_fastapi.utils.schemas import EXAMPLE_FLOWER

API_URL = os.getenv("API_URL", "http://localhost:5000").rstrip("/")
TIMEOUT_S = 10


def show(title: str, resp: requests.Response) -> None:
    """Pretty-print a response."""
    print(f"\n=== {title}: {resp.request.method} {resp.url} -> {resp.status_code}")
    if "X-Process-Time-Ms" in resp.headers:
        print(f"(server processing time: {resp.headers['X-Process-Time-Ms']} ms)")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)


def wait_until_ready(session: requests.Session) -> bool:
    """Check that the API is reachable and has a model loaded."""
    try:
        show("Liveness", session.get(f"{API_URL}/health", timeout=TIMEOUT_S))
        resp = session.get(f"{API_URL}/ready", timeout=TIMEOUT_S)
    except requests.ConnectionError:
        print(f"Cannot reach the API at {API_URL}. Is it running?")
        return False
    show("Readiness", resp)
    return resp.status_code == 200


def main() -> int:
    """Call all endpoints and return a process exit code."""
    with requests.Session() as session:
        # 1. Liveness and readiness
        if not wait_until_ready(session):
            return 1

        # 2. Model metadata
        show("Model metadata", session.get(f"{API_URL}/v1/model/metadata", timeout=TIMEOUT_S))

        # 3. Single prediction
        resp = session.post(f"{API_URL}/v1/predict", json=EXAMPLE_FLOWER, timeout=TIMEOUT_S)
        show("Single prediction", resp)
        resp.raise_for_status()

        # 4. Batch prediction (one flower of each species)
        batch = {
            "instances": [
                EXAMPLE_FLOWER,
                {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.2, "petal_width": 1.3},
                {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.7, "petal_width": 2.3},
            ]
        }
        resp = session.post(f"{API_URL}/v1/predict/batch", json=batch, timeout=TIMEOUT_S)
        show("Batch prediction", resp)
        resp.raise_for_status()

        # 5. Invalid request: negative value, a typo in a field name and a missing field
        invalid = {"sepal_length": -1.0, "sepal_widht": 3.5, "petal_length": 1.4}
        resp = session.post(f"{API_URL}/v1/predict", json=invalid, timeout=TIMEOUT_S)
        show("Invalid request (expected 422)", resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
