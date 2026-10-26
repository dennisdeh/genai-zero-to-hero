"""
Downloading Kaggle data with `kagglehub`, the lightweight alternative to the `kaggle` API client.

`kagglehub` is the library Kaggle recommends for *consuming* datasets, models and competition data
from Python code. Compared to the `kaggle` client it
    - needs a single function call (kagglehub.dataset_download("owner/name")),
    - caches downloads (by default under ~/.cache/kagglehub) and re-uses them on the next call,
    - supports versioned handles, e.g. "uciml/iris/versions/2", which makes pipelines reproducible,
    - can download many *public* datasets without any credentials (competitions and private
      data still require you to authenticate, e.g. with KAGGLE_API_TOKEN or kagglehub.login()).

The script downloads a dataset into outputs/ (bypassing the global cache with output_dir) and
gives a first summary of the tables with pandas.

How to run (from the repository root):
    python -m p02_data_sourcing.c01_kaggle.main_2_kagglehub
    python -m p02_data_sourcing.c01_kaggle.main_2_kagglehub --handle uciml/iris --target Species
    python -m p02_data_sourcing.c01_kaggle.main_2_kagglehub --competition titanic --target Survived
"""

import argparse
import os
import sys

from p02_data_sourcing.c01_kaggle.utils.credentials import HOW_TO_AUTHENTICATE, credential_sources
from p02_data_sourcing.c01_kaggle.utils.files import load_and_summarise

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", "kagglehub")


def download_with_kagglehub(handle: str | None, competition: str | None, path_dir: str) -> str:
    """
    Download a dataset or competition with kagglehub into path_dir.

    Return:
        Local directory containing the (already extracted) files.
    """
    import kagglehub

    if competition:
        target_dir = os.path.join(path_dir, f"competition__{competition}")
        return kagglehub.competition_download(competition, output_dir=target_dir)
    target_dir = os.path.join(path_dir, handle.replace("/", "__"))
    return kagglehub.dataset_download(handle, output_dir=target_dir)


def main(handle: str = "uciml/iris", competition: str | None = None, target: str | None = "Species") -> int:
    """Run the kagglehub example. Returns a process exit code (0 = success)."""
    # 1. Credentials are optional for public datasets, but we report what is configured
    print("\n *** 1. Credentials *** ")
    sources = credential_sources()
    print("Credential sources found:", ", ".join(sources) if sources else "none (public datasets may still work)")

    # 2. Download
    print("\n *** 2. Downloading with kagglehub *** ")
    try:
        data_dir = download_with_kagglehub(handle, competition, PATH_OUTPUTS)
    except ImportError:
        print("The 'kagglehub' package is not installed: pip install kagglehub")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {type(e).__name__}: {e}")
        print("Check your internet connection / proxy and the handle (owner/dataset-name).")
        if not sources:
            print("Competitions and private datasets also need credentials:")
            print(HOW_TO_AUTHENTICATE)
        return 1
    print(f"Files available in {data_dir}: {sorted(os.listdir(data_dir))}")

    # 3. Load and summarise
    print("\n *** 3. First look at the data *** ")
    load_and_summarise(data_dir, target=target)
    print(
        "\nTip: kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, handle, path) loads a single file "
        "straight into a DataFrame (requires `pip install kagglehub[pandas-datasets]`)."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handle", default="uciml/iris", help="dataset handle owner/name[/versions/N]")
    parser.add_argument("--competition", default=None, help="competition slug, e.g. titanic (overrides --handle)")
    parser.add_argument("--target", default="Species", help="name of the target column to summarise")
    args = parser.parse_args()
    sys.exit(main(handle=args.handle, competition=args.competition, target=args.target))
