"""
Paginated REST APIs and nested JSON: collecting issues from the GitHub REST API.

Most APIs never return a full collection in one response; they split it into pages. Common schemes are
    - page numbers   (?page=2&per_page=50),
    - offset/limit   (?offset=100&limit=50),
    - cursors/tokens (?cursor=abc123, the next cursor is part of the response),
    - Link headers   (RFC 8288: Link: <https://...&page=3>; rel="next") - used by GitHub.
Following the "next" link until there is none is the most robust approach, because the server decides.

The script
    1. reads metadata of a repository (one call),
    2. collects issues and pull requests page by page by following the Link header (max --max-pages),
    3. flattens the nested JSON (user.login, labels[].name, ...) with pd.json_normalize,
    4. analyses the result and stores it as Parquet,
    5. reports the rate-limit headers (unauthenticated: 60 requests per hour per IP address).

Authentication (optional): export GITHUB_TOKEN=<token> raises the rate limit to 5,000 requests/hour.
The token is read from the environment, never hard-coded.

How to run (from the repository root):
    python -m p02_data_sourcing.c06_apis.main_2_github_pagination
    python -m p02_data_sourcing.c06_apis.main_2_github_pagination --repo scikit-learn/scikit-learn --max-pages 2
"""

import argparse
import os
import sys

import pandas as pd

from p02_data_sourcing.c06_apis.utils.http_client import ApiError, build_session, get_json

API = "https://api.github.com"
PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
PATH_CACHE = os.path.join(PATH_OUTPUTS, "cache_github")


def make_github_session():
    """Session with GitHub's recommended headers and an optional token from the environment."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return build_session(headers=headers), bool(token)


def fetch_all_pages(session, url: str, params: dict, max_pages: int) -> tuple[list[dict], dict]:
    """
    Follow the 'next' links of a paginated endpoint.

    Return:
        All items of up to max_pages pages, and the headers of the last response (for the rate limit).
    """
    items, headers, page = [], {}, 0
    while url and page < max_pages:
        resp = get_json(session, url, params=params, cache_dir=PATH_CACHE, ttl_s=900)
        page += 1
        items.extend(resp.data)
        headers = resp.headers or headers
        print(f"  page {page}: {len(resp.data)} items{' (from cache)' if resp.from_cache else ''}")
        url = resp.links.get("next", {}).get("url")
        params = None  # the 'next' URL already contains all query parameters
    return items, headers


def flatten_issues(items: list[dict]) -> pd.DataFrame:
    """Flatten nested issue JSON into a tidy DataFrame."""
    df = pd.json_normalize(items, sep=".")  # nested dicts -> 'user.login', 'reactions.total_count', ...
    df["is_pull_request"] = df["pull_request.url"].notna() if "pull_request.url" in df else False
    df["label_names"] = [[lab["name"] for lab in labels] for labels in df["labels"]]  # list of dicts -> names
    for col in ["created_at", "updated_at", "closed_at"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    df["days_open"] = ((df["closed_at"].fillna(pd.Timestamp.now(tz="UTC")) - df["created_at"]).dt.total_seconds() / 86_400).round(1)
    cols = ["number", "title", "state", "is_pull_request", "user.login", "comments", "reactions.total_count",
            "label_names", "created_at", "closed_at", "days_open", "html_url"]
    return df[[c for c in cols if c in df.columns]]


def main(repo: str = "pandas-dev/pandas", max_pages: int = 3, per_page: int = 50) -> int:
    """Run the example. Returns a process exit code (0 = success)."""
    session, authenticated = make_github_session()
    print(f"Authenticated: {authenticated} (set GITHUB_TOKEN for a higher rate limit)")
    try:
        # 1. A single resource
        print(f"\n *** 1. Repository metadata: {repo} *** ")
        meta = get_json(session, f"{API}/repos/{repo}", cache_dir=PATH_CACHE, ttl_s=900).data
        for k in ["full_name", "description", "stargazers_count", "forks_count", "open_issues_count", "license", "default_branch"]:
            v = meta.get(k)
            print(f"    {k:18s} {v.get('spdx_id') if isinstance(v, dict) else v}")

        # 2. A paginated collection
        print(f"\n *** 2. Issues and pull requests (up to {max_pages} pages of {per_page}) *** ")
        params = {"state": "all", "per_page": per_page, "sort": "created", "direction": "desc"}
        items, headers = fetch_all_pages(session, f"{API}/repos/{repo}/issues", params, max_pages)
    except ApiError as e:
        print(f"\nAPI call failed: {e}")
        return 1

    if not items:
        print(f"No issues or pull requests found in {repo}.")
        return 0

    # 3. Flatten nested JSON
    print("\n *** 3. Flattening with pd.json_normalize *** ")
    df = flatten_issues(items)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 40):
        print(df.drop(columns=["html_url"]).head(8).to_string(index=False))

    # 4. Analysis and storage
    print("\n *** 4. Analysis *** ")
    print(df.groupby(["is_pull_request", "state"]).size().rename("count").to_string())
    print(f"\nMost active authors:\n{df['user.login'].value_counts().head(5).to_string()}")
    print(f"\nMost frequent labels:\n{df['label_names'].explode().value_counts().head(8).to_string()}")
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path = os.path.join(PATH_OUTPUTS, f"github_issues_{repo.replace('/', '__')}.parquet")
    df.to_parquet(path, index=False)
    print(f"\nSaved {len(df)} rows to {path}")

    # 5. Rate limit
    if headers:
        print(
            f"Rate limit: {headers.get('x-ratelimit-remaining', '?')} of {headers.get('x-ratelimit-limit', '?')} "
            "requests left in the current window (cached responses do not count)"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="pandas-dev/pandas", help="owner/repository")
    parser.add_argument("--max-pages", type=int, default=3, help="maximum number of pages to fetch")
    parser.add_argument("--per-page", type=int, default=50, help="items per page (max 100)")
    args = parser.parse_args()
    sys.exit(main(repo=args.repo, max_pages=args.max_pages, per_page=args.per_page))
