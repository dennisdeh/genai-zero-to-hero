"""
Working with the Hugging Face Hub directly through `huggingface_hub`: search, inspect and download files.

`datasets.load_dataset` is convenient, but sometimes you only need a single file (a dataset card, one
Parquet shard, a CSV) or want to search the Hub programmatically. The `huggingface_hub` library offers:
    - HfApi().list_datasets(...)  : search the Hub (by text, author, task, sort order),
    - HfApi().dataset_info(...)   : metadata of a dataset repository (tags, licence, files, last commit),
    - list_repo_files(...)        : all files in a repository,
    - hf_hub_download(...)        : download one file (cached, resumable, pinned to a revision if wanted),
    - snapshot_download(...)      : download (parts of) a whole repository with allow_patterns.

The script searches for sentiment datasets, inspects one repository, downloads its dataset card and one
data file into outputs/ and reads the data file with pandas.

How to run (from the repository root):
    python -m p02_data_sourcing.c03_hugging_face.main_3_hub_files
    python -m p02_data_sourcing.c03_hugging_face.main_3_hub_files --repo-id stanfordnlp/imdb

Gated/private repositories need a token: set the environment variable HF_TOKEN (never hard-code it).
"""

import argparse
import os
import sys

import pandas as pd

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", "hub_files")


def search_hub(query: str = "sentiment", limit: int = 10) -> pd.DataFrame:
    """Search the Hub for datasets matching a query, most downloaded first."""
    from huggingface_hub import HfApi

    hits = HfApi().list_datasets(search=query, sort="downloads", limit=limit)
    return pd.DataFrame(
        [{"id": d.id, "downloads": getattr(d, "downloads", None), "likes": getattr(d, "likes", None)} for d in hits]
    )


def inspect_repo(repo_id: str) -> list[str]:
    """Print metadata of a dataset repository and return its file list."""
    from huggingface_hub import HfApi, list_repo_files

    info = HfApi().dataset_info(repo_id)
    card = info.card_data.to_dict() if info.card_data else {}
    print(f"Repository   : https://huggingface.co/datasets/{info.id}")
    print(f"Last commit  : {info.sha} ({info.last_modified})")
    print(f"Licence      : {card.get('license')}")
    print(f"Task         : {card.get('task_categories')}")
    print(f"Tags (first) : {(info.tags or [])[:8]}")
    files = list_repo_files(repo_id, repo_type="dataset")
    print(f"Files ({len(files)}): {files[:15]}")
    return files


def download_files(repo_id: str, files: list[str]) -> str | None:
    """Download the README (dataset card) and the first tabular data file; return the data file path."""
    from huggingface_hub import hf_hub_download

    local_dir = os.path.join(PATH_OUTPUTS, repo_id.replace("/", "__"))
    if "README.md" in files:
        path_card = hf_hub_download(repo_id, "README.md", repo_type="dataset", local_dir=local_dir)
        with open(path_card, encoding="utf-8") as f:
            head = f.read(400)
        print(f"Dataset card saved to {path_card}; first lines:\n{head}\n...")

    data_files = [f for f in files if f.endswith((".parquet", ".csv", ".jsonl"))]
    if not data_files:
        print("No Parquet/CSV/JSONL files in the repository (it may use a loading script or another format).")
        return None
    # pin the download to a revision (branch, tag or commit sha) for reproducibility; "main" = latest
    return hf_hub_download(repo_id, data_files[0], repo_type="dataset", revision="main", local_dir=local_dir)


def read_tabular(path: str) -> pd.DataFrame:
    """Read a Parquet, CSV or JSON-lines file with pandas."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def main(repo_id: str = "cornell-movie-review-data/rotten_tomatoes", query: str = "sentiment") -> int:
    """Run the Hub example. Returns a process exit code (0 = success)."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("The 'huggingface_hub' package is not installed: pip install huggingface_hub")
        return 1

    try:
        # 1. Search
        print(f"\n *** 1. Searching the Hub for '{query}' datasets *** ")
        print(search_hub(query).to_string(index=False))

        # 2. Inspect a repository
        print(f"\n *** 2. Inspecting {repo_id} *** ")
        files = inspect_repo(repo_id)

        # 3. Download single files
        print("\n *** 3. Downloading single files *** ")
        path_data = download_files(repo_id, files)
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the Hugging Face Hub: {type(e).__name__}: {e}")
        print("Check your internet connection / proxy (https://huggingface.co must be reachable).")
        return 1

    # 4. Read with pandas
    if path_data:
        print("\n *** 4. Reading the downloaded file with pandas *** ")
        df = read_tabular(path_data)
        print(f"{path_data}: {df.shape}")
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(df.head())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="cornell-movie-review-data/rotten_tomatoes", help="dataset repository")
    parser.add_argument("--query", default="sentiment", help="search term for the Hub search")
    args = parser.parse_args()
    sys.exit(main(repo_id=args.repo_id, query=args.query))
