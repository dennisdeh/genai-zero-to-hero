"""
Fetching OpenML datasets directly with scikit-learn's `fetch_openml`.

If all you need is the data (and not the search or task functionality of the `openml` package),
`sklearn.datasets.fetch_openml` is the quickest way: one call returns a Bunch with a pandas DataFrame,
the target, the feature names and the dataset metadata. Downloads are cached in `data_home`.

The script
    1. fetches a dataset by name and version (default: "adult" v2, the census income data),
    2. inspects the returned Bunch: metadata (details), description, feature names and dtypes,
    3. performs a few data-quality checks (missing values, class balance, duplicates),
    4. stores the frame as Parquet in outputs/,
    5. trains a quick baseline model to confirm the data is ready for modelling.

How to run (from the repository root):
    python -m p02_data_sourcing.c02_open_ml.main_2_fetch_openml_sklearn
    python -m p02_data_sourcing.c02_open_ml.main_2_fetch_openml_sklearn --name credit-g --version 1
"""

import argparse
import os
import sys
import textwrap

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_SKLEARN_CACHE = os.path.join(PATH_OUTPUTS, "sklearn_openml_cache")


def quality_report(frame: pd.DataFrame, target: str) -> None:
    """Print a few basic data-quality indicators."""
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    print(f"Missing values per column:\n{missing.to_string() if not missing.empty else '    none'}")
    print(f"Duplicate rows: {frame.duplicated().sum():,}")
    print(f"Target '{target}' balance:\n{frame[target].value_counts(normalize=True).round(3).to_string()}")


def baseline_model(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> None:
    """
    Fit a majority-class baseline and a gradient-boosting model and print their test accuracy.
    HistGradientBoosting handles missing values and pandas 'category' columns natively, so the only
    preparation needed is to turn remaining text columns into categoricals.
    """
    X = X.copy()
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=200, random_state=seed)
    model.fit(X_train, y_train)
    print(f"Majority-class baseline accuracy : {dummy.score(X_test, y_test):.3f}")
    print(f"HistGradientBoosting accuracy     : {model.score(X_test, y_test):.3f}")


def main(name: str = "adult", version: int = 2) -> int:
    """Run the example. Returns a process exit code (0 = success)."""
    # 1. Fetch (cached in data_home after the first call)
    print(f"\n *** 1. Fetching '{name}' (version {version}) from OpenML *** ")
    try:
        bunch = fetch_openml(name=name, version=version, as_frame=True, data_home=PATH_SKLEARN_CACHE, parser="auto")
    except Exception as e:  # noqa: BLE001
        print(f"Could not fetch the dataset: {type(e).__name__}: {e}")
        print("Check your internet connection / proxy, and the dataset name and version on https://www.openml.org")
        return 1

    # 2. Inspect the Bunch
    print("\n *** 2. What fetch_openml returns *** ")
    print(f"Keys of the Bunch: {list(bunch.keys())}")
    details = bunch.details
    for k in ["id", "name", "version", "format", "licence", "default_target_attribute", "upload_date", "md5_checksum"]:
        print(f"    {k:26s} {details.get(k)}")
    print(f"URL: {bunch.url}")
    print("Description:\n" + textwrap.indent(textwrap.shorten(bunch.DESCR, 500), "    "))
    frame, target = bunch.frame, bunch.target.name
    print(f"\nFrame: {frame.shape}; dtypes:\n{frame.dtypes.astype(str).value_counts().to_string()}")

    # 3. Data quality
    print("\n *** 3. Data quality *** ")
    quality_report(frame, target)

    # 4. Store
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path_parquet = os.path.join(PATH_OUTPUTS, f"sklearn_{name}_v{version}.parquet")
    frame.to_parquet(path_parquet, index=False)
    print(f"\nSaved to {path_parquet}")

    # 5. Baseline model
    print("\n *** 5. Quick baseline model *** ")
    baseline_model(bunch.data, bunch.target)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="adult", help="OpenML dataset name")
    parser.add_argument("--version", type=int, default=2, help="dataset version (adult: 2, credit-g: 1)")
    args = parser.parse_args()
    sys.exit(main(name=args.name, version=args.version))
