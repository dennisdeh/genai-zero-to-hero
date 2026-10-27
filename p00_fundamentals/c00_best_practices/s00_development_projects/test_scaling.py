"""
Unit tests of utils/scaling.py, showing common pytest patterns for (ML) code.

Patterns demonstrated:
- fixtures (`rng`, `data`) that provide small, deterministic test data;
- `@pytest.mark.parametrize` to run one test over many inputs;
- `pytest.approx` / `np.testing.assert_allclose` to compare floating-point numbers with a tolerance;
- `pytest.raises` to check that invalid input fails loudly with a helpful message;
- `tmp_path` for file input/output without leaving files behind;
- `caplog` to check that a warning is logged;
- property-style tests: instead of hard-coding expected numbers, check properties that must hold for
  ANY input (zero mean/unit variance, invertibility, invariance to affine rescaling).

How to run (from the repository root):
    python -m pytest p00_fundamentals/c00_best_practices -v
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from p00_fundamentals.c00_best_practices.s00_development_projects.utils.scaling import (
    ScalerParams,
    fit_standard_scaler,
    inverse_standardise,
    load_params,
    save_params,
    standardise,
    train_test_split_indices,
)


# 1. Fixtures: small, deterministic data
@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator: tests must be deterministic, never flaky."""
    return np.random.default_rng(0)


@pytest.fixture
def data(rng: np.random.Generator) -> np.ndarray:
    """A small matrix with features on very different scales."""
    return rng.normal(loc=[0.0, 100.0, -5.0], scale=[1.0, 50.0, 0.01], size=(50, 3))


# 2. Example-based tests: known inputs, known outputs
def test_fit_known_values() -> None:
    x = np.array([[1.0, 10.0], [3.0, 10.0]])
    params = fit_standard_scaler(x)
    assert params.mean.tolist() == [2.0, 10.0]
    # the second feature is constant -> scale 1 instead of a division by zero
    assert params.scale == pytest.approx([1.0, 1.0])
    assert standardise(x, params) == pytest.approx(np.array([[-1.0, 0.0], [1.0, 0.0]]))


def test_constant_feature_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    x = np.column_stack([np.arange(5.0), np.full(5, 7.0)])
    with caplog.at_level(logging.WARNING):
        fit_standard_scaler(x)
    assert "constant" in caplog.text


@pytest.mark.parametrize(
    "bad_input, message",
    [
        (np.array([1.0, 2.0, 3.0]), "2-dimensional"),
        (np.empty((0, 3)), "at least one sample"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), "NaN"),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), "infinite"),
    ],
    ids=["1d", "empty", "nan", "inf"],
)
def test_invalid_input_raises(bad_input: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_standard_scaler(bad_input)


def test_feature_count_mismatch_raises(data: np.ndarray) -> None:
    params = fit_standard_scaler(data)
    with pytest.raises(ValueError, match="fitted on 3"):
        standardise(data[:, :2], params)


# 3. Property-style tests: run over many random shapes/seeds and check invariants
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(2, 1), (10, 3), (200, 7)])
def test_standardised_data_has_zero_mean_and_unit_std(seed: int, shape: tuple[int, int]) -> None:
    x = np.random.default_rng(seed).normal(loc=3.0, scale=10.0, size=shape)
    z = standardise(x, fit_standard_scaler(x))
    assert z.shape == x.shape  # shape test: cheap and catches many bugs
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_inverse_is_exact_round_trip(seed: int) -> None:
    x = np.random.default_rng(seed).uniform(-1e3, 1e3, size=(20, 4))
    params = fit_standard_scaler(x)
    np.testing.assert_allclose(inverse_standardise(standardise(x, params), params), x, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("a, b", [(2.0, 0.0), (1e-3, 5.0), (1e4, -3.0)])
def test_invariant_to_positive_affine_rescaling(data: np.ndarray, a: float, b: float) -> None:
    """Changing units (metres -> km) or shifting a feature must not change the z-scores."""
    z = standardise(data, fit_standard_scaler(data))
    x_rescaled = a * data + b
    z_rescaled = standardise(x_rescaled, fit_standard_scaler(x_rescaled))
    np.testing.assert_allclose(z_rescaled, z, atol=1e-8)


def test_test_data_uses_training_statistics(data: np.ndarray) -> None:
    """No leakage: transforming new data must not re-estimate mean and std."""
    params = fit_standard_scaler(data)
    z_shifted = standardise(data + 100.0, params)
    # a shift of the new data shows up in the z-scores instead of being "normalised away"
    np.testing.assert_allclose(z_shifted, standardise(data, params) + 100.0 / params.scale)


# 4. File I/O with tmp_path: a fresh temporary folder per test
def test_save_and_load_round_trip(tmp_path: Path, data: np.ndarray) -> None:
    params = fit_standard_scaler(data)
    path = tmp_path / "nested" / "scaler.json"
    save_params(params, str(path))
    loaded = load_params(str(path))
    assert path.exists()
    assert isinstance(loaded, ScalerParams)
    np.testing.assert_array_equal(loaded.mean, params.mean)
    np.testing.assert_array_equal(loaded.scale, params.scale)


# 5. Reproducible splitting
@pytest.mark.parametrize("n, fraction, n_test", [(10, 0.2, 2), (101, 0.25, 25), (4, 0.5, 2)])
def test_split_sizes_and_disjointness(n: int, fraction: float, n_test: int) -> None:
    train, test = train_test_split_indices(n, fraction, np.random.default_rng(1))
    assert len(test) == n_test
    assert set(train).isdisjoint(test)
    assert sorted(set(train) | set(test)) == list(range(n))


def test_split_is_reproducible_with_same_seed() -> None:
    first = train_test_split_indices(100, 0.3, np.random.default_rng(42))
    second = train_test_split_indices(100, 0.3, np.random.default_rng(42))
    third = train_test_split_indices(100, 0.3, np.random.default_rng(43))
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[1], third[1])


@pytest.mark.parametrize("fraction", [0.0, 1.0, 0.01])
def test_split_rejects_degenerate_fractions(fraction: float) -> None:
    with pytest.raises(ValueError):
        train_test_split_indices(10, fraction, np.random.default_rng(0))
