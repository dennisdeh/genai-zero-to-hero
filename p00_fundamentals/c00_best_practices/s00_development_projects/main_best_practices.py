"""
Best practices in a small ML script: configuration, logging, seeding, timing and traceable outputs.

The script trains a linear regression with gradient descent on synthetic "housing" data whose features
live on very different scales, once on the raw features and once on standardised features. Along the way
it demonstrates the engineering habits described in book_software_development.md:

1. configuration from environment variables, collected in one validated dataclass (utils/config.py);
2. logging configured once at the entry point (console + log file) instead of print statements;
3. seeding and an explicit random number generator, so that the run is reproducible;
4. timing of the individual steps;
5. fitting preprocessing on the training data only (no test-set leakage), using a tested module
   (utils/scaling.py, see test_scaling.py);
6. saving artefacts together with the configuration and metadata (versions, git commit) of the run.

How to run (from the repository root):
    python -m p00_fundamentals.c00_best_practices.s00_development_projects.main_best_practices
    # change the configuration through environment variables, e.g.
    export GZH_SEED=7 GZH_LOG_LEVEL=DEBUG
    python -m p00_fundamentals.c00_best_practices.s00_development_projects.main_best_practices
Outputs (log file, scaler parameters, run summary) are written to the outputs/ folder next to this script.
"""

import json
import logging
import os

import numpy as np

from p00_fundamentals.c00_best_practices.s00_development_projects.utils.config import SECTION_DIR, Settings
from p00_fundamentals.c00_best_practices.s00_development_projects.utils.reproducibility import (
    run_metadata,
    set_seed,
    timer,
)
from p00_fundamentals.c00_best_practices.s00_development_projects.utils.scaling import (
    fit_standard_scaler,
    save_params,
    standardise,
    train_test_split_indices,
)

# a named logger for this script; when run with `python -m`, __name__ is "__main__"
logger = logging.getLogger("best_practices")

FEATURE_NAMES = ["area_m2", "n_rooms", "distance_to_centre_m", "age_years"]


def configure_logging(level: str, log_file: str | None = None) -> None:
    """Configure the root logger once, at the entry point of the program.

    Library modules only create loggers (`logging.getLogger(__name__)`) and never configure handlers;
    the application decides where messages go and how verbose they are.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,  # replace handlers configured earlier (e.g. by an imported library)
    )


def load_dotenv_if_available() -> None:
    """Load a .env file next to this script into os.environ, if python-dotenv is installed.

    Existing environment variables take precedence over the file. The .env file is gitignored and is
    the place for local settings and secrets; it is never committed and its values are never logged.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(SECTION_DIR, ".env"), override=False)


def make_housing_data(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset whose features have very different scales.

    Returns:
        x of shape (n_samples, 4) and the target y (a price in thousands) of shape (n_samples,).
    """
    area = rng.uniform(30, 250, n_samples)  # mean ~ 140
    rooms = rng.integers(1, 8, n_samples).astype(float)  # mean ~ 4
    distance = rng.uniform(100, 30_000, n_samples)  # mean ~ 15 000
    age = rng.uniform(0, 120, n_samples)  # mean ~ 60
    x = np.column_stack([area, rooms, distance, age])
    true_coef = np.array([3.0, 15.0, -0.01, -0.8])
    y = 100.0 + x @ true_coef + rng.normal(0.0, 20.0, n_samples)
    return x, y


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for the intercept."""
    return np.column_stack([np.ones(x.shape[0]), x])


def gradient_descent(x: np.ndarray, y: np.ndarray, n_iter: int) -> np.ndarray:
    """Least-squares linear regression with full-batch gradient descent.

    The step size 1/L uses the Lipschitz constant L of the gradient (largest eigenvalue of 2/n X^T X),
    which guarantees convergence; the speed of convergence is governed by the condition number of X^T X.
    """
    n = x.shape[0]
    hessian = 2.0 / n * x.T @ x
    step = 1.0 / np.linalg.eigvalsh(hessian).max()
    beta = np.zeros(x.shape[1])
    for _ in range(n_iter):
        beta -= step * (2.0 / n * x.T @ (x @ beta - y))
    return beta


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> dict[str, object]:
    """Run the example and return the run summary."""
    # 1. Configuration: .env (optional) -> environment variables -> validated, immutable settings
    load_dotenv_if_available()
    settings = Settings.from_env()

    # 2. Logging: configured once, here at the entry point
    log_file = os.path.join(settings.output_dir, "run.log") if settings.save_outputs else None
    configure_logging(settings.log_level, log_file)
    logger.info("Settings: %s", settings.to_dict())

    # 3. Seeding: one explicit generator for everything random in this run
    rng = set_seed(settings.seed)
    metadata = run_metadata()
    logger.info("Run metadata: %s", metadata)

    # 4. Data, and a quick check that the seed really makes the data reproducible
    timings: dict[str, float] = {}
    with timer("data generation", logger) as t:
        x, y = make_housing_data(settings.n_samples, rng)
    timings["data_generation_s"] = t["seconds"]
    x_again, _ = make_housing_data(settings.n_samples, np.random.default_rng(settings.seed))
    logger.info("Same seed gives identical data: %s", bool(np.array_equal(x, x_again)))
    for name, col in zip(FEATURE_NAMES, x.T, strict=True):
        logger.info("  %-22s mean=%10.2f  std=%10.2f", name, col.mean(), col.std())

    # 5. Split first, then fit the scaler on the TRAINING data only (no leakage from the test set)
    train_idx, test_idx = train_test_split_indices(settings.n_samples, settings.test_fraction, rng)
    x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]
    scaler = fit_standard_scaler(x_train)
    z_train, z_test = standardise(x_train, scaler), standardise(x_test, scaler)
    logger.info("Train/test sizes: %d/%d", len(train_idx), len(test_idx))

    # 6. Gradient descent on raw vs standardised features (same budget of iterations)
    n_iter = 500
    results: dict[str, dict[str, float]] = {}
    for label, a_train, a_test in [("raw", x_train, x_test), ("standardised", z_train, z_test)]:
        a_train1, a_test1 = add_intercept(a_train), add_intercept(a_test)
        cond = float(np.linalg.cond(a_train1.T @ a_train1))
        with timer(f"gradient descent ({label})", logger) as t:
            beta = gradient_descent(a_train1, y_train, n_iter=n_iter)
        beta_exact, *_ = np.linalg.lstsq(a_train1, y_train, rcond=None)
        results[label] = {
            "condition_number": cond,
            "test_rmse_gd": rmse(y_test, a_test1 @ beta),
            "test_rmse_exact": rmse(y_test, a_test1 @ beta_exact),
            "seconds": t["seconds"],
        }
        logger.info(
            "%-12s cond(X^T X)=%9.2e | test RMSE after %d GD steps: %8.2f (exact least squares: %.2f)",
            label,
            cond,
            n_iter,
            results[label]["test_rmse_gd"],
            results[label]["test_rmse_exact"],
        )
    if results["raw"]["test_rmse_gd"] > 1.5 * results["raw"]["test_rmse_exact"]:
        logger.warning("Gradient descent on raw features has not converged - standardise your features!")

    # 7. Save artefacts together with everything needed to reproduce them
    summary: dict[str, object] = {
        "settings": settings.to_dict(),
        "metadata": metadata,
        "timings": timings,
        "results": results,
    }
    if settings.save_outputs:
        save_params(scaler, os.path.join(settings.output_dir, "scaler_params.json"))
        summary_path = os.path.join(settings.output_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved run summary to %s and the log to %s", summary_path, log_file)
    return summary


if __name__ == "__main__":
    main()
