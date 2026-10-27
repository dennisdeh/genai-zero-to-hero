"""
A small, well-tested feature-standardisation module (z-scores), written to the standards of this chapter.

Standardisation maps every feature (column) j to z = (x - mean_j) / std_j, where mean and standard deviation
are estimated on the TRAINING data only and then reused for validation/test/production data. The
parameters are kept in a small dataclass that can be saved to and loaded from JSON, so that exactly the
same transformation is applied at inference time.

The module illustrates:
- type hints and docstrings on every public function;
- input validation with clear error messages (fail early and loudly);
- a module-level logger instead of print statements;
- pure functions without hidden global state, which makes them easy to test.

scikit-learn's `StandardScaler` does the same in production code; the point here is the engineering, not
the algorithm.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalerParams:
    """Parameters of a fitted standard scaler.

    Attributes:
        mean: per-feature mean, shape (n_features,).
        scale: per-feature standard deviation, shape (n_features,); constant features get scale 1.
    """

    mean: np.ndarray
    scale: np.ndarray

    @property
    def n_features(self) -> int:
        """Number of features the scaler was fitted on."""
        return int(self.mean.shape[0])

    def to_dict(self) -> dict[str, list[float]]:
        """Convert to a JSON-serialisable dictionary."""
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "ScalerParams":
        """Create the parameters from a dictionary produced by `to_dict`."""
        mean = np.asarray(data["mean"], dtype=float)
        scale = np.asarray(data["scale"], dtype=float)
        if mean.shape != scale.shape or mean.ndim != 1:
            raise ValueError("mean and scale must be 1-d arrays of equal length")
        return cls(mean=mean, scale=scale)


def _check_2d(x: np.ndarray, name: str = "x") -> np.ndarray:
    """Convert to a float array and check that it is a finite 2-d matrix with at least one row."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional (n_samples, n_features), got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values; impute or drop them first")
    return arr


def fit_standard_scaler(x: np.ndarray, eps: float = 1e-12) -> ScalerParams:
    """Estimate the per-feature mean and standard deviation of a data matrix.

    Args:
        x: training data of shape (n_samples, n_features).
        eps: features whose standard deviation is below `eps` are treated as constant and get scale 1
            (otherwise we would divide by zero).

    Returns:
        The fitted `ScalerParams`.
    """
    arr = _check_2d(x)
    mean = arr.mean(axis=0)
    scale = arr.std(axis=0)  # population std (ddof=0), as in scikit-learn
    constant = scale < eps
    if np.any(constant):
        logger.warning("Features %s are constant; their scale is set to 1", np.flatnonzero(constant).tolist())
        scale = np.where(constant, 1.0, scale)
    logger.debug("Fitted scaler on %d samples and %d features", arr.shape[0], arr.shape[1])
    return ScalerParams(mean=mean, scale=scale)


def standardise(x: np.ndarray, params: ScalerParams) -> np.ndarray:
    """Apply a fitted scaler: z = (x - mean) / scale."""
    arr = _check_2d(x)
    if arr.shape[1] != params.n_features:
        raise ValueError(f"x has {arr.shape[1]} features, but the scaler was fitted on {params.n_features}")
    z: np.ndarray = (arr - params.mean) / params.scale
    return z


def inverse_standardise(z: np.ndarray, params: ScalerParams) -> np.ndarray:
    """Undo `standardise`: x = z * scale + mean."""
    arr = _check_2d(z, name="z")
    if arr.shape[1] != params.n_features:
        raise ValueError(f"z has {arr.shape[1]} features, but the scaler was fitted on {params.n_features}")
    x: np.ndarray = arr * params.scale + params.mean
    return x


def save_params(params: ScalerParams, path: str) -> None:
    """Save the scaler parameters as JSON (human readable, safe to load - unlike pickle)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params.to_dict(), f, indent=2)
    logger.info("Saved scaler parameters to %s", path)


def load_params(path: str) -> ScalerParams:
    """Load scaler parameters saved with `save_params`."""
    with open(path, encoding="utf-8") as f:
        return ScalerParams.from_dict(json.load(f))


def train_test_split_indices(
    n_samples: int, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split the indices 0..n_samples-1 into disjoint train and test sets.

    Args:
        n_samples: number of samples.
        test_fraction: fraction of samples in the test set, in (0, 1).
        rng: random number generator (passed explicitly to make the split reproducible).

    Returns:
        (train_indices, test_indices), each sorted.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    n_test = int(round(n_samples * test_fraction))
    if n_test == 0 or n_test == n_samples:
        raise ValueError(f"test_fraction={test_fraction} leaves an empty train or test set for n={n_samples}")
    permutation = rng.permutation(n_samples)
    return np.sort(permutation[n_test:]), np.sort(permutation[:n_test])
