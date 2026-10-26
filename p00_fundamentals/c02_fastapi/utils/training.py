"""
Training of the iris classifier that is served by the API.

The model is a scikit-learn Pipeline (standardisation + multinomial logistic regression). Bundling the
preprocessing with the estimator is essential for serving: the API receives raw measurements and the
exact same transformations as during training are applied automatically.
"""

import platform
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p00_fundamentals.c02_fastapi.utils.model_io import save_model_bundle
from p00_fundamentals.c02_fastapi.utils.schemas import ModelMetadata

MODEL_NAME = "iris-classifier"

# map the sklearn column names, e.g. "sepal length (cm)", to API-friendly snake_case names
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the iris dataset (shipped with scikit-learn, so no download is needed)."""
    iris = load_iris(as_frame=True)
    x = iris.data.copy()
    x.columns = FEATURE_NAMES
    y = iris.target
    class_names = [str(c) for c in iris.target_names]
    return x, y, class_names


def build_pipeline() -> Pipeline:
    """Create the (untrained) preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_save(model_dir: str, version: str = "1.0.0", verbose: bool = True) -> ModelMetadata:
    """
    Train the pipeline, evaluate it on a hold-out set and save model + metadata to model_dir.

    :param model_dir: output folder of the model artefacts
    :param version: semantic version of the model, exposed by the API
    :param verbose: print the evaluation results
    :return: the metadata saved next to the model
    """
    # 1. Load data and create a stratified train/test split
    x, y, class_names = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Cross-validate on the training set to get a robust performance estimate
    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="accuracy")

    # 3. Fit on the full training set and evaluate on the untouched test set
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    metrics = {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }
    if verbose:
        print(classification_report(y_test, y_pred, target_names=class_names))

    # 4. Collect metadata: what was trained, when, with which library versions and how well it does
    metadata = ModelMetadata(
        model_name=MODEL_NAME,
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        framework="scikit-learn",
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        dataset="iris (sklearn.datasets.load_iris)",
        features=FEATURE_NAMES,
        classes=class_names,
        metrics=metrics,
        n_train=len(x_train),
        n_test=len(x_test),
    )

    # 5. Persist the model package
    save_model_bundle(pipeline, metadata, model_dir)
    return metadata
