"""
Train the iris classifier that is served by the FastAPI application (main_api.py).

The script trains a scikit-learn pipeline (StandardScaler + LogisticRegression) on the iris dataset,
which ships with scikit-learn (no download required), evaluates it, and saves the model package:
- model.joblib:  the fitted pipeline
- metadata.json: version, training time, library versions, features, classes and metrics

How to run (from the repository root):
    python -m p00_fundamentals.c02_fastapi.main_train_model
    python -m p00_fundamentals.c02_fastapi.main_train_model --version 1.1.0 --output-dir /tmp/model

By default, the artefacts are written to p00_fundamentals/c02_fastapi/outputs/model (gitignored).
"""

import argparse
import json

from p00_fundamentals.c02_fastapi.utils.config import DEFAULT_MODEL_DIR
from p00_fundamentals.c02_fastapi.utils.training import train_and_save


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and save the iris classifier.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_MODEL_DIR,
        help="folder for model.joblib and metadata.json (default: %(default)s)",
    )
    parser.add_argument(
        "--version", default="1.0.0", help="model version exposed by the API (default: %(default)s)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 1. Read the command-line arguments
    args = parse_args()

    # 2. Train, evaluate and save the model package
    print(f"Training model version {args.version} ...")
    metadata = train_and_save(model_dir=args.output_dir, version=args.version)

    # 3. Report the result
    print("Metrics:", json.dumps(metadata.metrics, indent=2))
    print(f"Model package saved to: {args.output_dir}")
    print("Start the API with: uvicorn p00_fundamentals.c02_fastapi.main_api:app --port 5000")
