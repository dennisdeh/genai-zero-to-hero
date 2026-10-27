"""
Searching and downloading datasets with the `openml` Python package.

OpenML (https://www.openml.org) is an open platform for sharing datasets, machine learning tasks
(dataset + target + evaluation procedure) and experiment results. Every dataset comes with rich,
machine-readable metadata: feature types, missing values, the default target and pre-computed
"qualities" (number of instances, class imbalance, ...). That makes it ideal to search for suitable
datasets programmatically.

The script
    1. points the OpenML cache to outputs/ (downloads are cached and re-used),
    2. lists datasets using server-side filters and refines the search with pandas,
    3. downloads one dataset (default: credit-g, id 31) with its metadata,
    4. inspects description, licence, target, qualities and feature metadata,
    5. extracts X and y as pandas objects and stores them as Parquet,
    6. lists the OpenML tasks (standardised evaluation set-ups) defined on the dataset.

No account or API key is needed to download data (a key is only required for uploading).

How to run (from the repository root):
    python -m p02_data_sourcing.c02_open_ml.main_1_openml_search
    python -m p02_data_sourcing.c02_open_ml.main_1_openml_search --dataset-id 1590   # adult
"""

import argparse
import os
import sys
import textwrap

import pandas as pd

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "openml_cache")


def configure_openml(cache_dir: str = PATH_CACHE):
    """Import openml, set the cache directory and use a short retry policy suited for interactive use."""
    import openml

    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(openml.config, "set_root_cache_directory"):
        openml.config.set_root_cache_directory(cache_dir)
    else:  # older versions of the package
        openml.config.cache_directory = cache_dir
    openml.config.set_retry_policy("human", n_retries=2)
    return openml


def search_datasets(openml, n_min: int = 1_000, n_max: int = 10_000) -> pd.DataFrame:
    """
    Search for binary classification datasets of moderate size without missing values.

    Server-side filters (number_instances, number_classes) reduce the amount of data sent over the
    network; the remaining filtering and sorting is done locally with pandas.
    """
    df = openml.datasets.list_datasets(
        output_format="dataframe",
        number_instances=f"{n_min}..{n_max}",
        number_classes="2",
        status="active",
    )
    cols = ["did", "name", "version", "NumberOfInstances", "NumberOfFeatures", "NumberOfMissingValues",
            "MinorityClassSize", "MajorityClassSize"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if {"MinorityClassSize", "MajorityClassSize"} <= set(df.columns):
        df["imbalance"] = (df["MajorityClassSize"] / df["MinorityClassSize"]).round(2)
    if "NumberOfMissingValues" in df.columns:
        df = df[df["NumberOfMissingValues"] == 0]
    # keep only the newest version of every dataset name
    df = df.sort_values("version").drop_duplicates("name", keep="last")
    return df.sort_values("NumberOfInstances", ascending=False)


def describe_dataset(ds) -> pd.DataFrame:
    """Print the most useful metadata of an OpenMLDataset and return its feature table."""
    print(f"Name / version : {ds.name} / v{ds.version}  (id {ds.dataset_id})")
    print(f"Default target : {ds.default_target_attribute}")
    print(f"Licence        : {ds.licence}")
    print(f"OpenML page    : https://www.openml.org/d/{ds.dataset_id}")
    print(f"Original source: {ds.original_data_url}")
    print(f"Citation       : {ds.citation}")
    print("Description    :\n" + textwrap.indent(textwrap.shorten(ds.description or "", 600), "    "))

    q = ds.qualities or {}
    keys = ["NumberOfInstances", "NumberOfFeatures", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures",
            "NumberOfMissingValues", "NumberOfClasses", "MajorityClassPercentage", "MinorityClassPercentage"]
    print("\nSelected qualities:")
    for k in keys:
        if k in q:
            print(f"    {k:28s} {q[k]:g}")

    features = pd.DataFrame(
        [
            {
                "index": f.index,
                "name": f.name,
                "data_type": f.data_type,
                "n_missing": f.number_missing_values,
                "n_categories": len(f.nominal_values) if f.nominal_values else None,
            }
            for f in ds.features.values()
        ]
    )
    features["n_categories"] = features["n_categories"].astype("Int64")
    return features


def main(dataset_id: int = 31) -> int:
    """Run the OpenML example. Returns a process exit code (0 = success)."""
    try:
        openml = configure_openml()
    except ImportError:
        print("The 'openml' package is not installed: pip install openml")
        return 1

    try:
        # 1. Search
        print("\n *** 1. Searching OpenML for binary classification datasets *** ")
        hits = search_datasets(openml)
        print(f"{len(hits)} matching datasets, the 15 largest:")
        print(hits.head(15).to_string(index=False))

        # 2. Download one dataset with all metadata
        print(f"\n *** 2. Downloading dataset {dataset_id} *** ")
        ds = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach OpenML: {type(e).__name__}: {e}")
        print("Check your internet connection / proxy (the API lives at https://www.openml.org/api/v1/).")
        return 1

    # 3. Inspect metadata
    print("\n *** 3. Metadata *** ")
    features = describe_dataset(ds)
    print(f"\nFeature metadata:\n{features.to_string(index=False)}")

    # 4. Extract the data
    print("\n *** 4. Extracting X and y *** ")
    X, y, categorical_indicator, attribute_names = ds.get_data(target=ds.default_target_attribute)
    print(f"X: {X.shape}, y: {y.shape}; categorical features: "
          f"{[n for n, c in zip(attribute_names, categorical_indicator) if c]}")
    print(f"Class balance:\n{y.value_counts(normalize=True).round(3).to_string()}")
    df = X.copy()
    df[ds.default_target_attribute] = y
    path_parquet = os.path.join(PATH_OUTPUTS, f"openml_{dataset_id}_{ds.name}.parquet")
    df.to_parquet(path_parquet, index=False)
    print(f"Saved to {path_parquet}")

    # 5. Tasks defined on the dataset: they fix the target and the evaluation procedure (e.g. 10-fold CV),
    #    so that results from different people are directly comparable
    print("\n *** 5. OpenML tasks on this dataset *** ")
    try:
        tasks = openml.tasks.list_tasks(data_id=dataset_id, output_format="dataframe")
        cols = [c for c in ["tid", "task_type", "estimation_procedure", "evaluation_measures", "target_feature"]
                if c in tasks.columns]
        print(tasks[cols].head(10).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print(f"Could not list tasks: {e}")
    print(f"\nOpenML cache (re-used on the next run): {PATH_CACHE}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=31, help="OpenML dataset id (31 = credit-g, 1590 = adult)")
    args = parser.parse_args()
    sys.exit(main(dataset_id=args.dataset_id))
