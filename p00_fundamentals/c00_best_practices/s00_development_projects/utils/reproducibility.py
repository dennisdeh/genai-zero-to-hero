"""
Helpers for reproducible and observable runs: seeding, timing and run metadata.

- `set_seed` seeds Python's `random` module and NumPy's legacy global generator, and returns a
  dedicated `numpy.random.Generator`. New code should pass such a generator around explicitly
  instead of relying on global state; the global seeds only protect third-party code that still uses it.
- `timer` is a context manager that logs how long a block of code took.
- `run_metadata` collects the information needed to reproduce a run (versions, git commit, platform).
"""

import logging
import platform
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> np.random.Generator:
    """Seed the global random number generators and return a new NumPy generator.

    Deep-learning code would additionally call `torch.manual_seed(seed)` (and, for full determinism on a
    GPU, `torch.use_deterministic_algorithms(True)`); see part p06.

    Args:
        seed: a non-negative integer.

    Returns:
        A `numpy.random.Generator` seeded with `seed`.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)  # legacy global state, used by some libraries
    logger.debug("Seeded random and numpy with %d", seed)
    return np.random.default_rng(seed)


@contextmanager
def timer(name: str, log: logging.Logger = logger) -> Iterator[dict[str, float]]:
    """Log the wall-clock duration of a block of code.

    The yielded dictionary is filled with the duration (key "seconds") when the block exits,
    so the caller can also store it::

        with timer("training") as t:
            train()
        print(t["seconds"])
    """
    result: dict[str, float] = {}
    start = time.perf_counter()  # monotonic, high resolution: use this rather than time.time() for durations
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        log.info("%s took %.3f s", name, result["seconds"])


def git_commit(short: bool = True) -> str:
    """Return the current git commit hash, or "unknown" outside a git repository."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_metadata() -> dict[str, str]:
    """Collect metadata that makes a run traceable: time, versions, platform and git commit."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
