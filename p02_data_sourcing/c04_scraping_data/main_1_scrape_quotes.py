"""
Scraping a paginated web site politely with requests + BeautifulSoup: https://quotes.toscrape.com

quotes.toscrape.com is a sandbox site made for learning web scraping. The script
    1. selects the target: the real site if reachable, otherwise a local sandbox with the same HTML
       structure (see utils/sandbox_site.py), so the example always runs,
    2. reads robots.txt and shows which URLs we may fetch,
    3. crawls the paginated list by following the "Next" link (with rate limiting and retries),
    4. parses every quote (text, author, author link, tags) with CSS selectors,
    5. builds a tidy pandas DataFrame, does a small analysis and stores CSV + Parquet in outputs/.

How to run (from the repository root):
    python -m p02_data_sourcing.c04_scraping_data.main_1_scrape_quotes
    python -m p02_data_sourcing.c04_scraping_data.main_1_scrape_quotes --site local --max-pages 3
    python -m p02_data_sourcing.c04_scraping_data.main_1_scrape_quotes --site remote --max-pages 2 --delay 1

Please keep --delay at >= 1 second for real sites: be a good citizen of the web.
"""

import argparse
import os
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from p02_data_sourcing.c04_scraping_data.utils.polite_scraper import PoliteScraper
from p02_data_sourcing.c04_scraping_data.utils.sandbox_site import target_site

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")


def parse_quotes(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """
    Extract all quotes of one page.

    The HTML of one quote looks like:
        <div class="quote">
          <span class="text">"..."</span>
          <span>by <small class="author">Name</small> <a href="/author/Name">(about)</a></span>
          <div class="tags"> <a class="tag" href="...">tag</a> ... </div>
        </div>
    """
    records = []
    for div in soup.select("div.quote"):
        text = div.select_one("span.text").get_text(strip=True).strip("“”\"")
        author = div.select_one("small.author").get_text(strip=True)
        about = div.select_one("a[href*='/author/']")
        records.append(
            {
                "text": text,
                "author": author,
                "author_url": urljoin(page_url, about["href"]) if about else None,
                "tags": [a.get_text(strip=True) for a in div.select("div.tags a.tag")],
                "source_page": page_url,
            }
        )
    return records


def crawl(scraper: PoliteScraper, start_url: str = "/", max_pages: int = 3) -> list[dict]:
    """Follow the 'Next' links from start_url and collect the quotes of up to max_pages pages."""
    url, records, page = urljoin(scraper.base_url, start_url), [], 0
    while url and page < max_pages:
        soup = scraper.get_soup(url)
        page += 1
        found = parse_quotes(soup, url)
        print(f"  page {page}: {url} -> {len(found)} quotes")
        records.extend(found)
        nxt = soup.select_one("li.next > a")
        url = urljoin(url, nxt["href"]) if nxt else None  # relative -> absolute URL
    return records


def main(site_mode: str = "auto", max_pages: int = 3, delay_s: float = 1.0) -> pd.DataFrame:
    """Run the scraper and return the DataFrame of quotes."""
    with target_site("quotes", mode=site_mode) as base_url:
        # 1. Polite client and robots.txt
        print("\n *** 1. robots.txt *** ")
        scraper = PoliteScraper(base_url, delay_s=delay_s)
        print(f"Effective delay between requests: {scraper.delay_s}s (Crawl-delay from robots.txt honoured)")
        for path in ["/", "/page/2/", "/private/secret.html"]:
            print(f"  may fetch {path:22s}: {scraper.can_fetch(urljoin(base_url, path))}")

        # 2. Crawl
        print(f"\n *** 2. Crawling up to {max_pages} pages *** ")
        records = crawl(scraper, "/", max_pages=max_pages)
        print(f"{len(records)} quotes from {scraper.n_requests} HTTP requests (incl. robots.txt, excl. retries)")

    if not records:
        print("Nothing scraped.")
        return pd.DataFrame()

    # 3. Tidy DataFrame + analysis
    print("\n *** 3. Analysis *** ")
    df = pd.DataFrame(records)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60, "display.max_columns", 10):
        print(df[["text", "author", "tags"]].head())
    print(f"\nQuotes per author:\n{df['author'].value_counts().head(10).to_string()}")
    tags = df.explode("tags")["tags"].value_counts()
    print(f"\nMost frequent tags:\n{tags.head(10).to_string()}")

    # 4. Store: Parquet keeps the list column; CSV needs a flat string
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    df.to_parquet(os.path.join(PATH_OUTPUTS, "quotes.parquet"), index=False)
    df.assign(tags=df["tags"].str.join("|")).to_csv(os.path.join(PATH_OUTPUTS, "quotes.csv"), index=False)
    print(f"\nSaved quotes.parquet and quotes.csv to {PATH_OUTPUTS}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", choices=["auto", "remote", "local"], default="auto", help="which site to scrape")
    parser.add_argument("--max-pages", type=int, default=3, help="maximum number of pages to crawl")
    parser.add_argument("--delay", type=float, default=1.0, help="minimum seconds between requests")
    args = parser.parse_args()
    main(site_mode=args.site, max_pages=args.max_pages, delay_s=args.delay)
