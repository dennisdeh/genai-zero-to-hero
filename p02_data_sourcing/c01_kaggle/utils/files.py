"""
File helpers shared by the Kaggle examples: safe unzipping and a quick summary of downloaded tables.
"""

import os
import sqlite3
import zipfile

import pandas as pd


def safe_unzip(path_zip: str, path_dir: str) -> list[str]:
    """
    Extract a zip archive, refusing members that would be written outside path_dir.

    Downloaded archives are untrusted input: a malicious archive can contain member names such as
    '../../.bashrc' ("zip slip"). We therefore resolve every target path and check that it stays
    inside the destination directory before extracting.

    Return:
        List of extracted file paths.
    """
    os.makedirs(path_dir, exist_ok=True)
    root = os.path.realpath(path_dir)
    extracted = []
    with zipfile.ZipFile(path_zip) as zf:
        for member in zf.infolist():
            target = os.path.realpath(os.path.join(root, member.filename))
            if os.path.commonpath([root, target]) != root:
                raise ValueError(f"Unsafe path in archive, refusing to extract: {member.filename}")
            zf.extract(member, root)
            if not member.is_dir():
                extracted.append(target)
    return extracted


def summarise_table(df: pd.DataFrame, name: str, target: str | None = None) -> None:
    """Print a compact first look at a table: shape, dtypes, missing values, statistics and target balance."""
    print(f"\n--- {name}: {df.shape[0]:,} rows x {df.shape[1]} columns ---")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.head())
        overview = pd.DataFrame(
            {"dtype": df.dtypes.astype(str), "n_missing": df.isna().sum(), "n_unique": df.nunique()}
        )
        print(f"\nColumn overview:\n{overview}")
        num = df.select_dtypes("number")
        if not num.empty:
            print(f"\nNumeric summary:\n{num.describe().T.round(2)}")
    if target and target in df.columns:
        print(f"\nTarget '{target}' distribution:\n{df[target].value_counts(normalize=True).round(3)}")


def load_and_summarise(path_dir: str, target: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load every CSV file (and every table of every SQLite database) found in path_dir and summarise it.

    Return:
        Dictionary mapping a table name to its DataFrame.
    """
    tables = {}
    for root, _, files in os.walk(path_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            if f.lower().endswith(".csv"):
                tables[f] = pd.read_csv(path)
            elif f.lower().endswith((".sqlite", ".db")):
                # Kaggle datasets sometimes ship the same data as an SQLite database
                with sqlite3.connect(path) as conn:
                    names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"]
                    for t in names:
                        tables[f"{f}:{t}"] = pd.read_sql(f'SELECT * FROM "{t}"', conn)
    if not tables:
        print(f"No CSV or SQLite files found in {path_dir}")
    for name, df in tables.items():
        summarise_table(df, name, target=target)
    return tables
