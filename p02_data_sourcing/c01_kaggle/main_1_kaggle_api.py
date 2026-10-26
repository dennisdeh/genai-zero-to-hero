"""
Downloading data from Kaggle with the official `kaggle` API client.

The script
    1. checks which Kaggle credentials are configured (without ever printing them),
    2. authenticates against the Kaggle API,
    3. searches Kaggle datasets for a keyword and lists the files of the chosen dataset,
    4. downloads the dataset (or, optionally, a competition) as a zip archive into outputs/,
    5. unzips the archive safely and gives a first summary of the tables with pandas.

Default dataset: "uciml/iris" (the classic Iris data from the UCI repository, a few kB).
Optionally a competition can be downloaded, e.g. "titanic" - this requires that you have accepted
the competition rules on https://www.kaggle.com/competitions/titanic once.

Credentials: see utils/credentials.py and book_kaggle.md. Typical options are the environment
variable KAGGLE_API_TOKEN, a token file ~/.kaggle/access_token, or the legacy ~/.kaggle/kaggle.json
(or KAGGLE_USERNAME + KAGGLE_KEY). Never hard-code credentials in scripts.

How to run (from the repository root):
    python -m p02_data_sourcing.c01_kaggle.main_1_kaggle_api
    python -m p02_data_sourcing.c01_kaggle.main_1_kaggle_api --dataset uciml/iris --search iris
    python -m p02_data_sourcing.c01_kaggle.main_1_kaggle_api --competition titanic --target Survived
"""

import argparse
import glob
import os
import sys

import pandas as pd

from p02_data_sourcing.c01_kaggle.utils.credentials import HOW_TO_AUTHENTICATE, credential_sources
from p02_data_sourcing.c01_kaggle.utils.files import load_and_summarise, safe_unzip

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs", "kaggle_api")


def get_authenticated_api():
    """
    Create and authenticate a KaggleApi instance.

    Return:
        The authenticated KaggleApi, or None if authentication was not possible.
    """
    # 1. Which credential sources are present? (names only, never the secret values)
    sources = credential_sources()
    print("Credential sources found:", ", ".join(sources) if sources else "none")
    if not sources:
        print(HOW_TO_AUTHENTICATE)
        return None

    # 2. Import and authenticate. Older versions of the client authenticated at import time and raised
    #    an exception (or called sys.exit) without credentials, so both the import and authenticate()
    #    are guarded.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()  # calls exit(1) if no credentials can be found
        return api
    except ImportError:
        print("The 'kaggle' package is not installed: pip install kaggle")
    except (SystemExit, Exception) as e:  # noqa: BLE001 - we want a friendly message for any failure
        print(f"Kaggle authentication failed ({type(e).__name__}: {e}).")
        print(HOW_TO_AUTHENTICATE)
    return None


def search_datasets(api, search: str, n: int = 10) -> pd.DataFrame:
    """Search Kaggle datasets for a keyword and return the most voted hits as a DataFrame."""
    hits = api.dataset_list(search=search, sort_by="votes") or []
    rows = [
        {
            "ref": getattr(d, "ref", None),
            "title": getattr(d, "title", None),
            "size_kb": round((getattr(d, "total_bytes", 0) or 0) / 1024, 1),
            "downloads": getattr(d, "download_count", None),
            "votes": getattr(d, "vote_count", None),
            "usability": getattr(d, "usability_rating", None),
        }
        for d in hits[:n]
    ]
    return pd.DataFrame(rows)


def list_dataset_files(api, dataset: str) -> pd.DataFrame:
    """List the files contained in a Kaggle dataset (name and size)."""
    response = api.dataset_list_files(dataset)
    files = getattr(response, "files", None) or getattr(response, "dataset_files", None) or []
    return pd.DataFrame(
        [{"name": f.name, "size_kb": round((getattr(f, "total_bytes", 0) or 0) / 1024, 1)} for f in files]
    )


def download(api, dataset: str | None, competition: str | None, path_dir: str) -> str:
    """
    Download a dataset or a competition as zip archive(s) and extract them safely.

    Return:
        The directory that contains the extracted files.
    """
    os.makedirs(path_dir, exist_ok=True)
    if competition:
        target_dir = os.path.join(path_dir, f"competition__{competition}")
        os.makedirs(target_dir, exist_ok=True)
        api.competition_download_files(competition, path=target_dir, quiet=False)
    else:
        target_dir = os.path.join(path_dir, dataset.replace("/", "__"))
        os.makedirs(target_dir, exist_ok=True)
        # unzip=True would also work; we unzip ourselves to show how to do it safely
        api.dataset_download_files(dataset, path=target_dir, quiet=False, unzip=False)

    for path_zip in glob.glob(os.path.join(target_dir, "*.zip")):
        extracted = safe_unzip(path_zip, target_dir)
        print(f"Extracted {len(extracted)} file(s) from {os.path.basename(path_zip)}")
        os.remove(path_zip)  # keep only the extracted data
    return target_dir


def main(dataset: str = "uciml/iris", search: str = "iris", competition: str | None = None, target: str | None = "Species") -> int:
    """Run the Kaggle API example. Returns a process exit code (0 = success)."""
    # 1. Authenticate
    print("\n *** 1. Authenticating with the Kaggle API *** ")
    api = get_authenticated_api()
    if api is None:
        return 1

    try:
        # 2. Search datasets and inspect the chosen one
        if not competition:
            print(f"\n *** 2. Searching datasets for '{search}' *** ")
            with pd.option_context("display.width", 200, "display.max_colwidth", 50):
                print(search_datasets(api, search).to_string(index=False))
            print(f"\nFiles in '{dataset}':")
            print(list_dataset_files(api, dataset).to_string(index=False))

        # 3. Download and unzip
        print("\n *** 3. Downloading *** ")
        data_dir = download(api, dataset, competition, PATH_OUTPUTS)
        print(f"Data stored in {data_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"\nThe Kaggle request failed: {type(e).__name__}: {e}")
        print(
            "Common causes: no internet connection / blocked proxy, a typo in the dataset reference "
            "(format owner/dataset-name), or competition rules not yet accepted on the website."
        )
        return 1

    # 4. Load and summarise with pandas
    print("\n *** 4. First look at the data *** ")
    load_and_summarise(data_dir, target=target)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="uciml/iris", help="dataset reference owner/name")
    parser.add_argument("--search", default="iris", help="keyword for the dataset search")
    parser.add_argument("--competition", default=None, help="competition slug, e.g. titanic (overrides --dataset)")
    parser.add_argument("--target", default="Species", help="name of the target column to summarise")
    args = parser.parse_args()
    sys.exit(main(dataset=args.dataset, search=args.search, competition=args.competition, target=args.target))
