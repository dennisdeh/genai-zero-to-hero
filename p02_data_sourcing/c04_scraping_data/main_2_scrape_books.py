"""
Scraping a product catalogue: listing pages, detail pages and HTML tables - https://books.toscrape.com

A typical e-commerce scraping job has two levels: listing pages (many products, a few fields each, with
pagination) and detail pages (one product, many fields, often in an HTML table). The script
    1. selects the target: the real books.toscrape.com if reachable, otherwise a local sandbox with the
       same HTML structure,
    2. crawls the catalogue listing pages and extracts title, price, rating, availability and link,
    3. visits the first N detail pages and turns their "product information" table into columns,
    4. cleans the values (currency strings -> float, "Three" -> 3, "In stock (5 available)" -> 5),
    5. joins both levels into one DataFrame and stores it in outputs/.

How to run (from the repository root):
    python -m p02_data_sourcing.c04_scraping_data.main_2_scrape_books
    python -m p02_data_sourcing.c04_scraping_data.main_2_scrape_books --site local --max-pages 2 --n-details 6
"""

import argparse
import os
import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from p02_data_sourcing.c04_scraping_data.utils.polite_scraper import PoliteScraper
from p02_data_sourcing.c04_scraping_data.utils.sandbox_site import target_site

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(text: str) -> float | None:
    """'£51.77' -> 51.77 (robust against currency symbols and stray characters)."""
    m = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def parse_listing(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Extract the products of one catalogue listing page (one <article class="product_pod"> each)."""
    records = []
    for pod in soup.select("article.product_pod"):
        link = pod.select_one("h3 > a")
        rating_classes = pod.select_one("p.star-rating")["class"]  # e.g. ["star-rating", "Three"]
        records.append(
            {
                "title": link["title"],  # the visible text is truncated, the title attribute is complete
                "price_gbp": parse_price(pod.select_one("p.price_color").get_text()),
                "rating": next((RATING_WORDS[c] for c in rating_classes if c in RATING_WORDS), None),
                "in_stock": "in stock" in pod.select_one("p.availability").get_text(strip=True).lower(),
                "url": urljoin(page_url, link["href"]),
            }
        )
    return records


def parse_detail_table(soup: BeautifulSoup) -> dict:
    """
    Turn the product information table (<tr><th>key</th><td>value</td></tr>) into a dict.
    For well-formed tables pandas.read_html(...) is an alternative (it needs lxml or html5lib).
    """
    table = soup.select_one("table.table-striped") or soup.select_one("table")
    data = {tr.th.get_text(strip=True): tr.td.get_text(strip=True) for tr in table.select("tr") if tr.th and tr.td}
    crumbs = [li.get_text(strip=True) for li in soup.select("ul.breadcrumb li")]
    data["category"] = crumbs[-2] if len(crumbs) >= 2 else None  # Home > Books > <category> > title
    return data


def crawl_catalogue(scraper: PoliteScraper, start: str, max_pages: int) -> list[dict]:
    """Crawl listing pages by following the 'next' link."""
    url, records, page = urljoin(scraper.base_url, start), [], 0
    while url and page < max_pages:
        soup = scraper.get_soup(url)
        page += 1
        found = parse_listing(soup, url)
        print(f"  listing page {page}: {len(found)} books")
        records.extend(found)
        nxt = soup.select_one("li.next > a")
        url = urljoin(url, nxt["href"]) if nxt else None
    return records


def clean_details(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw strings of the detail tables into proper numeric columns."""
    out = pd.DataFrame(index=df.index)
    out["upc"] = df.get("UPC")
    out["category"] = df.get("category")
    out["price_incl_tax"] = df.get("Price (incl. tax)").map(parse_price)
    out["n_available"] = df.get("Availability").str.extract(r"(\d+)\s+available", expand=False).astype("Int64")
    out["n_reviews"] = pd.to_numeric(df.get("Number of reviews"), errors="coerce").astype("Int64")
    return out


def main(site_mode: str = "auto", max_pages: int = 2, n_details: int = 6, delay_s: float = 1.0) -> pd.DataFrame:
    """Run the two-level scraper and return the combined DataFrame."""
    with target_site("books", mode=site_mode) as base_url:
        scraper = PoliteScraper(base_url, delay_s=delay_s)

        # 1. Listing pages
        print(f"\n *** 1. Crawling up to {max_pages} listing pages *** ")
        books = pd.DataFrame(crawl_catalogue(scraper, "catalogue/page-1.html", max_pages))

        # 2. Detail pages (only the first n_details to keep the number of requests small)
        print(f"\n *** 2. Visiting {min(n_details, len(books))} detail pages *** ")
        details = []
        for url in books["url"].head(n_details):
            try:
                details.append({"url": url, **parse_detail_table(scraper.get_soup(url))})
            except Exception as e:  # noqa: BLE001 - one broken page must not stop the whole job
                print(f"  skipped {url}: {type(e).__name__}: {e}")
        print(f"{scraper.n_requests} HTTP requests in total")

    # 3. Clean and join both levels
    print("\n *** 3. Cleaning and joining *** ")
    df_details = pd.DataFrame(details)
    if not df_details.empty:
        df_details = pd.concat([df_details[["url"]], clean_details(df_details)], axis=1)
        books = books.merge(df_details, on="url", how="left")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 30):
        print(books.drop(columns="url").to_string(index=False))
    print(f"\nMean price by rating:\n{books.groupby('rating')['price_gbp'].mean().round(2).to_string()}")

    # 4. Store
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path = os.path.join(PATH_OUTPUTS, "books.parquet")
    books.to_parquet(path, index=False)
    print(f"\nSaved {len(books)} books to {path}")
    return books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", choices=["auto", "remote", "local"], default="auto", help="which site to scrape")
    parser.add_argument("--max-pages", type=int, default=2, help="maximum number of listing pages")
    parser.add_argument("--n-details", type=int, default=6, help="number of detail pages to visit")
    parser.add_argument("--delay", type=float, default=1.0, help="minimum seconds between requests")
    args = parser.parse_args()
    main(site_mode=args.site, max_pages=args.max_pages, n_details=args.n_details, delay_s=args.delay)
