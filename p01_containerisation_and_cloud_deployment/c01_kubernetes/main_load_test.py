"""
A simple load generator for the iris model-serving API.

It sends batch prediction requests from several concurrent threads for a given duration and reports the
throughput, the error rate and latency percentiles. Use it to:
- benchmark the API locally (uvicorn) or in a container;
- create enough CPU load in Kubernetes to watch the HorizontalPodAutoscaler add pods
  (`kubectl get hpa -n gzh --watch` in a second terminal).

How to run (from the repository root, with the API reachable at API_URL, e.g. via
`kubectl port-forward -n gzh service/iris-api 5000:80`):
    python -m p01_containerisation_and_cloud_deployment.c01_kubernetes.main_load_test
    python -m p01_containerisation_and_cloud_deployment.c01_kubernetes.main_load_test --duration 300 --concurrency 16
"""

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_URL = os.getenv("API_URL", "http://localhost:5000").rstrip("/")


def random_flower(rng: random.Random) -> dict[str, float]:
    """Generate a plausible random iris flower (measurements in cm)."""
    return {
        "sepal_length": round(rng.uniform(4.3, 7.9), 1),
        "sepal_width": round(rng.uniform(2.0, 4.4), 1),
        "petal_length": round(rng.uniform(1.0, 6.9), 1),
        "petal_width": round(rng.uniform(0.1, 2.5), 1),
    }


def worker(
    worker_id: int, deadline: float, batch_size: int, latencies: list[float], errors: list[int]
) -> None:
    """Send requests until the deadline; record latencies (s) of successful calls and error counts."""
    rng = random.Random(worker_id)
    with requests.Session() as session:
        while time.perf_counter() < deadline:
            payload = {"instances": [random_flower(rng) for _ in range(batch_size)]}
            start = time.perf_counter()
            try:
                resp = session.post(f"{API_URL}/v1/predict/batch", json=payload, timeout=10)
                ok = resp.status_code == 200
            except requests.RequestException:
                ok = False
            elapsed = time.perf_counter() - start
            # list.append is atomic in CPython, so the threads can share the lists without a lock
            if ok:
                latencies.append(elapsed)
            else:
                errors.append(1)


def run_load_test(duration_s: float, concurrency: int, batch_size: int) -> dict[str, float]:
    """Run the load test and return summary statistics."""
    # 1. Check that the service is ready before generating load
    requests.get(f"{API_URL}/ready", timeout=5).raise_for_status()

    # 2. Start the workers and wait until they have finished
    latencies: list[float] = []
    errors: list[int] = []
    deadline = time.perf_counter() + duration_s
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(worker, i, deadline, batch_size, latencies, errors) for i in range(concurrency)
        ]
        for f in futures:
            f.result()  # re-raises unexpected exceptions of the workers

    # 3. Summarise
    n_ok = len(latencies)
    stats = {"requests_ok": n_ok, "requests_failed": len(errors), "throughput_rps": n_ok / duration_s}
    if n_ok >= 2:
        q = statistics.quantiles(latencies, n=100)
        stats.update(
            {"latency_p50_ms": q[49] * 1000, "latency_p95_ms": q[94] * 1000, "latency_p99_ms": q[98] * 1000}
        )
    return stats


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Load test of the iris API.")
    parser.add_argument("--duration", type=float, default=20, help="seconds (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=8, help="threads (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=50, help="flowers/request (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        f"Load test of {API_URL}: {args.concurrency} threads, batches of {args.batch_size}, "
        f"{args.duration:.0f} s ..."
    )
    results = run_load_test(args.duration, args.concurrency, args.batch_size)
    for key, value in results.items():
        print(f"{key:>16}: {value:,.1f}" if isinstance(value, float) else f"{key:>16}: {value}")
