"""
Tests of the configuration and reproducibility helpers, and a smoke test of the main script.

Patterns demonstrated:
- `monkeypatch.setenv` / `monkeypatch.delenv` to control environment variables in a test; pytest undoes
  the changes afterwards, so tests cannot leak configuration into each other;
- an autouse fixture that gives every test a clean environment;
- parametrised tests for parsing;
- a fast end-to-end smoke test of the main script, writing to `tmp_path`.

How to run (from the repository root):
    python -m pytest p00_fundamentals/c00_best_practices -v
"""

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from p00_fundamentals.c00_best_practices.s00_development_projects import main_best_practices
from p00_fundamentals.c00_best_practices.s00_development_projects.utils.config import (
    DEFAULT_OUTPUT_DIR,
    ENV_PREFIX,
    Settings,
    parse_bool,
)
from p00_fundamentals.c00_best_practices.s00_development_projects.utils.reproducibility import set_seed, timer

ENV_NAMES = ["SEED", "N_SAMPLES", "TEST_FRACTION", "LOG_LEVEL", "SAVE_OUTPUTS", "OUTPUT_DIR"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all GZH_* variables, so that the developer's shell cannot influence the tests."""
    for name in ENV_NAMES:
        monkeypatch.delenv(ENV_PREFIX + name, raising=False)


# 1. Configuration from environment variables
def test_defaults_without_env() -> None:
    settings = Settings.from_env()
    assert settings == Settings()
    assert settings.output_dir == DEFAULT_OUTPUT_DIR


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GZH_SEED", "7")
    monkeypatch.setenv("GZH_N_SAMPLES", "123")
    monkeypatch.setenv("GZH_TEST_FRACTION", "0.5")
    monkeypatch.setenv("GZH_LOG_LEVEL", "debug")
    monkeypatch.setenv("GZH_SAVE_OUTPUTS", "no")
    monkeypatch.setenv("GZH_OUTPUT_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert settings.seed == 7
    assert settings.n_samples == 123
    assert settings.test_fraction == pytest.approx(0.5)
    assert settings.log_level == "DEBUG"
    assert settings.save_outputs is False
    assert settings.output_dir == str(tmp_path)


@pytest.mark.parametrize(
    "name, value",
    [("N_SAMPLES", "5"), ("TEST_FRACTION", "1.5"), ("LOG_LEVEL", "LOUD"), ("SEED", "abc"), ("SAVE_OUTPUTS", "maybe")],
)
def test_invalid_env_values_fail_early(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(ENV_PREFIX + name, value)
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize(
    "text, expected", [("1", True), ("TRUE", True), (" yes ", True), ("off", False), ("0", False), ("False", False)]
)
def test_parse_bool(text: str, expected: bool) -> None:
    assert parse_bool(text) is expected


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(AttributeError):
        settings.seed = 1  # type: ignore[misc]


# 2. Reproducibility helpers
def test_set_seed_makes_random_numbers_reproducible() -> None:
    rng1 = set_seed(123)
    a = (random.random(), np.random.rand(), rng1.normal())
    rng2 = set_seed(123)
    b = (random.random(), np.random.rand(), rng2.normal())
    assert a == b


def test_set_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_timer_measures_duration() -> None:
    with timer("noop") as t:
        sum(range(1000))
    assert t["seconds"] >= 0.0


# 3. Smoke test of the whole script with a small configuration
@pytest.fixture
def restore_logging() -> Iterator[None]:
    """main() reconfigures the root logger (incl. a log file); restore the original handlers afterwards."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers, root.level = handlers, level


@pytest.mark.usefixtures("restore_logging")
def test_main_runs_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GZH_N_SAMPLES", "200")
    monkeypatch.setenv("GZH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("GZH_LOG_LEVEL", "WARNING")
    summary = main_best_practices.main()
    results = summary["results"]
    assert isinstance(results, dict)
    # standardisation lets gradient descent reach the exact least-squares solution
    assert results["standardised"]["test_rmse_gd"] == pytest.approx(
        results["standardised"]["test_rmse_exact"], rel=1e-3
    )
    assert results["standardised"]["condition_number"] < results["raw"]["condition_number"]
    saved = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert saved["settings"]["n_samples"] == 200
    assert (tmp_path / "scaler_params.json").exists()
