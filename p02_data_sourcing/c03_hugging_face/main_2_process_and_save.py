"""
Processing and saving a Hugging Face dataset: map, filter, train/validation/test split and export.

After loading a dataset you typically clean it, derive new columns, create the splits you need and
store the result so that later steps (training, evaluation) start from a fixed, versionable artefact.
The `datasets` library does all of this efficiently on top of Apache Arrow, with results cached on disk.

The script
    1. loads "cornell-movie-review-data/rotten_tomatoes" (or, with --offline or when the Hub cannot be
       reached, a tiny built-in dataset with the same schema),
    2. cleans the text and adds features with a batched `map`,
    3. removes unwanted rows with `filter`,
    4. creates a stratified validation split with `train_test_split`,
    5. looks at the data with pandas,
    6. saves the result with `save_to_disk` (Arrow) and `to_parquet`, and reloads both - including loading
       local Parquet files through the same `load_dataset` API.

How to run (from the repository root):
    python -m p02_data_sourcing.c03_hugging_face.main_2_process_and_save
    python -m p02_data_sourcing.c03_hugging_face.main_2_process_and_save --offline
"""

import argparse
import os
import re

from datasets import DatasetDict, load_dataset, load_from_disk

from p02_data_sourcing.c03_hugging_face.utils.offline_data import offline_rotten_tomatoes

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "hf_cache")
PATH_SAVED = os.path.join(PATH_OUTPUTS, "rotten_tomatoes_processed")


def load_data(offline: bool) -> tuple[DatasetDict, str]:
    """Load the dataset from the Hub, falling back to the built-in offline dataset if necessary."""
    if not offline:
        try:
            return load_dataset("cornell-movie-review-data/rotten_tomatoes", cache_dir=PATH_CACHE), "hub"
        except Exception as e:  # noqa: BLE001
            print(f"Hub not reachable ({type(e).__name__}); falling back to the built-in offline dataset.")
    return offline_rotten_tomatoes(), "offline"


def clean_batch(batch: dict) -> dict:
    """
    Batched map function: receives a dict of lists (one list per column) and returns new/updated columns.
    Batched processing is much faster than example-by-example processing for simple Python functions.
    """
    texts = [re.sub(r"\s+([,.!?;:'])", r"\1", t).strip() for t in batch["text"]]  # "word ," -> "word,"
    return {
        "text": texts,
        "n_words": [len(t.split()) for t in texts],
        "n_chars": [len(t) for t in texts],
    }


def main(offline: bool = False, min_words: int = 3) -> DatasetDict:
    """Run the processing pipeline and return the final DatasetDict."""
    # 1. Load
    print("\n *** 1. Loading *** ")
    dsd, origin = load_data(offline)
    print(f"Source: {origin}\n{dsd}")

    # 2. Map: clean text, add length features (num_proc > 1 would parallelise large datasets)
    print("\n *** 2. map (batched) *** ")
    dsd = dsd.map(clean_batch, batched=True, batch_size=1_000, desc="cleaning")
    print(f"Example after cleaning: {dsd['train'][0]}")

    # 3. Filter: drop very short reviews
    print("\n *** 3. filter *** ")
    before = {k: v.num_rows for k, v in dsd.items()}
    dsd = dsd.filter(lambda ex: ex["n_words"] >= min_words, desc="filtering")
    print(f"Rows before: {before} -> after (n_words >= {min_words}): { {k: v.num_rows for k, v in dsd.items()} }")

    # 4. Split: carve a stratified validation set out of train (the label is a ClassLabel, so
    #    stratification keeps the class ratio identical in both parts)
    print("\n *** 4. train/validation/test *** ")
    split = dsd["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
    dsd = DatasetDict({"train": split["train"], "validation": split["test"], "test": dsd["test"]})
    print(dsd)

    # 5. Inspect with pandas
    print("\n *** 5. Statistics with pandas *** ")
    names = dsd["train"].features["label"].names
    for name, ds in dsd.items():
        df = ds.to_pandas()
        balance = df["label"].map(dict(enumerate(names))).value_counts(normalize=True).round(3).to_dict()
        print(f"{name:10s} rows={len(df):5d}  mean words={df['n_words'].mean():5.1f}  balance={balance}")

    # 6. Save and reload
    print("\n *** 6. Saving and reloading *** ")
    dsd.save_to_disk(PATH_SAVED)  # Arrow files + metadata, reload with load_from_disk
    reloaded = load_from_disk(PATH_SAVED)
    print(f"save_to_disk -> {PATH_SAVED}\nload_from_disk: { {k: v.num_rows for k, v in reloaded.items()} }")

    path_parquet = os.path.join(PATH_OUTPUTS, "rotten_tomatoes_parquet")
    os.makedirs(path_parquet, exist_ok=True)
    data_files = {}
    for name, ds in dsd.items():
        data_files[name] = os.path.join(path_parquet, f"{name}.parquet")
        ds.to_parquet(data_files[name])  # portable: readable by pandas, Spark, DuckDB, ...
    from_parquet = load_dataset("parquet", data_files=data_files, cache_dir=PATH_CACHE)
    print(f"to_parquet -> {path_parquet}\nload_dataset('parquet', ...): { {k: v.num_rows for k, v in from_parquet.items()} }")
    print(
        f"Reloaded label feature: {from_parquet['train'].features['label']} - `datasets` stores its feature "
        "definitions in the Parquet schema metadata; other tools (e.g. pandas) simply see integers."
    )
    return dsd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use the built-in dataset instead of the Hub")
    parser.add_argument("--min-words", type=int, default=3, help="minimum number of words to keep a review")
    args = parser.parse_args()
    main(offline=args.offline, min_words=args.min_words)
