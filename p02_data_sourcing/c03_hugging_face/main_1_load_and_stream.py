"""
Loading datasets from the Hugging Face Hub with the `datasets` library: regular vs. streaming mode.

The Hub hosts hundreds of thousands of datasets that can all be loaded with one function,
`datasets.load_dataset`. There are two fundamentally different ways of doing so:

    - regular (map-style) mode: the data files are downloaded once, converted to Apache Arrow and
      memory-mapped from a local cache. You get random access (ds[123]), len(ds), fast slicing, and the
      cache is re-used next time. Suitable when the data fits on your disk.
    - streaming (iterable) mode (streaming=True): nothing is downloaded up front; examples are read lazily
      while you iterate. Ideal for very large datasets, for a quick look at a dataset, or when disk space is
      limited - but there is no len() or random access, and shuffling uses a buffer.

The script
    1. loads the small "cornell-movie-review-data/rotten_tomatoes" dataset in regular mode, inspects its
       splits and features and shows split slicing ("train[:10%]"),
    2. streams the larger "stanfordnlp/imdb" dataset and shows take/shuffle/filter/map on the stream,
    3. compares how long it took to get the first examples in both modes.

How to run (from the repository root):
    python -m p02_data_sourcing.c03_hugging_face.main_1_load_and_stream

Downloads are cached in p02_data_sourcing/c03_hugging_face/outputs/hf_cache. Gated or private datasets
require a Hugging Face token in the environment variable HF_TOKEN (or `hf auth login`).
"""

import os
import sys
import time

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "hf_cache")


def regular_mode(repo_id: str = "cornell-movie-review-data/rotten_tomatoes") -> float:
    """Load a dataset in regular (map-style) mode, inspect it and return the time it took."""
    from datasets import load_dataset

    t0 = time.perf_counter()
    dsd = load_dataset(repo_id, cache_dir=PATH_CACHE)
    elapsed = time.perf_counter() - t0
    print(dsd)  # a DatasetDict: one Dataset per split
    train = dsd["train"]
    print(f"\nFeatures: {train.features}")
    print(f"Label names: {train.features['label'].names}")
    print(f"Random access, train[0]: {train[0]}")
    print(f"Column access, first 3 labels: {train['label'][:3]}")
    print(f"Arrow cache files: {[os.path.relpath(f['filename'], PATH_OUTPUTS) for f in train.cache_files][:2]}")

    # Split slicing: select (parts of) splits directly in load_dataset - re-uses the cached files
    small = load_dataset(repo_id, split="train[:10%]+test[:10%]", cache_dir=PATH_CACHE)
    print(f"\nSplit 'train[:10%]+test[:10%]' -> {small.num_rows} rows")
    return elapsed


def streaming_mode(repo_id: str = "stanfordnlp/imdb", n: int = 5) -> float:
    """Stream a dataset, apply lazy transformations and return the time to the first examples."""
    from datasets import load_dataset

    t0 = time.perf_counter()
    stream = load_dataset(repo_id, split="train", streaming=True)
    first = list(stream.take(n))
    elapsed = time.perf_counter() - t0
    print(f"Type: {type(stream).__name__} (no len(), no random access)")
    for ex in first[:3]:
        print(f"    label={ex['label']}  text={ex['text'][:90]!r}...")

    # Transformations on a stream are lazy: they run while iterating
    processed = (
        stream.shuffle(buffer_size=1_000, seed=42)  # approximate shuffling with a buffer
        .filter(lambda ex: len(ex["text"]) < 600)  # keep short reviews only
        .map(lambda ex: {"n_words": len(ex["text"].split())})
    )
    sample = list(processed.take(n))
    print(f"\nAfter shuffle/filter/map, {len(sample)} examples, n_words: {[ex['n_words'] for ex in sample]}")
    return elapsed


def main() -> int:
    """Run both loading modes. Returns a process exit code (0 = success)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("The 'datasets' package is not installed: pip install datasets")
        return 1

    try:
        # 1. Regular mode
        print("\n *** 1. Regular (map-style) mode *** ")
        t_regular = regular_mode()

        # 2. Streaming mode
        print("\n *** 2. Streaming (iterable) mode *** ")
        t_stream = streaming_mode()
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not load from the Hugging Face Hub: {type(e).__name__}: {e}")
        print(
            "Check your internet connection / proxy (https://huggingface.co must be reachable). "
            "To work with already cached data only, set HF_HUB_OFFLINE=1. The processing example "
            "main_2_process_and_save.py has an --offline mode that works without the Hub."
        )
        return 1

    # 3. Comparison
    print("\n *** 3. Comparison *** ")
    print(f"Regular mode, full download + Arrow conversion (or cache hit): {t_regular:.2f}s")
    print(f"Streaming mode, time to the first examples:                    {t_stream:.2f}s")
    print(f"Cache directory: {PATH_CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
