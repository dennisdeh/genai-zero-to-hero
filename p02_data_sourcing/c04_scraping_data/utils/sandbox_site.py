"""
A local "sandbox" web site for practising scraping without touching the internet.

It imitates the HTML structure of the scraping sandboxes https://quotes.toscrape.com and
https://books.toscrape.com (run by Zyte for exactly this purpose), so the same parsing code works on both.
The content is our own. The local site also has a robots.txt with a disallowed path and a Crawl-delay,
and it simulates one temporary outage (HTTP 503 with Retry-After) to exercise the retry logic.

Usage:
    with target_site("quotes", mode="auto") as base_url:
        ...  # base_url is the real site if reachable, otherwise the local sandbox
"""

import html
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import requests

REMOTE_URLS = {"quotes": "https://quotes.toscrape.com", "books": "https://books.toscrape.com"}

ROBOTS_TXT = "User-agent: *\nDisallow: /private/\nCrawl-delay: 0.2\n"

QUOTES = [
    ("The only true wisdom is in knowing you know nothing.", "Socrates", ["wisdom", "knowledge"]),
    ("Knowing yourself is the beginning of all wisdom.", "Aristotle", ["wisdom", "self"]),
    ("We are what we repeatedly do.", "Aristotle", ["habits", "excellence"]),
    ("The unexamined life is not worth living.", "Socrates", ["life", "philosophy"]),
    ("Whereof one cannot speak, thereof one must be silent.", "Ludwig Wittgenstein", ["philosophy", "language"]),
    ("The limits of my language mean the limits of my world.", "Ludwig Wittgenstein", ["language", "world"]),
    ("I think, therefore I am.", "Rene Descartes", ["philosophy", "self"]),
    ("Happiness depends upon ourselves.", "Aristotle", ["happiness", "self"]),
    ("Nothing in life is to be feared, it is only to be understood.", "Marie Curie", ["life", "knowledge", "fear"]),
    ("Be less curious about people and more curious about ideas.", "Marie Curie", ["curiosity", "ideas"]),
    ("If I have seen further it is by standing on the shoulders of giants.", "Isaac Newton", ["knowledge", "humility"]),
    ("Imagination is more important than knowledge.", "Albert Einstein", ["imagination", "knowledge"]),
    ("Life is like riding a bicycle. To keep your balance you must keep moving.", "Albert Einstein", ["life", "balance"]),
    ("The important thing is not to stop questioning.", "Albert Einstein", ["curiosity", "questions"]),
    ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci", ["simplicity", "design"]),
]
QUOTES_PER_PAGE = 5

RATINGS = ["One", "Two", "Three", "Four", "Five"]
BOOKS = [
    # title, price (GBP), rating index, stock count, category
    ("The Quiet Algorithm", 23.50, 3, 12, "Science"),
    ("Gradient Descent at Dawn", 41.99, 4, 3, "Poetry"),
    ("A Field Guide to Outliers", 17.25, 2, 0, "Science"),
    ("Tensors for the Weary", 35.00, 1, 7, "Science"),
    ("The Overfitted Garden", 12.99, 0, 22, "Fiction"),
    ("Midnight in the Data Lake", 28.40, 4, 5, "Mystery"),
    ("Recipes from a Random Forest", 19.95, 3, 9, "Food and Drink"),
    ("The Bayesian Detective", 32.10, 2, 1, "Mystery"),
    ("Notes on Attention", 44.00, 4, 15, "Science"),
    ("Kernel Tricks and Other Stories", 9.99, 1, 0, "Fiction"),
    ("The Long Tail of Summer", 15.75, 3, 6, "Fiction"),
    ("Cross-Validation for Beginners", 26.60, 2, 4, "Science"),
]
BOOKS_PER_PAGE = 6


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def _page(title: str, body: str) -> str:
    return f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{html.escape(title)}</title></head><body>{body}</body></html>'


def quotes_page(page: int) -> str | None:
    """Render quotes page number `page` (1-based) like quotes.toscrape.com, or None if it does not exist."""
    n_pages = -(-len(QUOTES) // QUOTES_PER_PAGE)
    if not 1 <= page <= n_pages:
        return None
    items = []
    for text, author, tags in QUOTES[(page - 1) * QUOTES_PER_PAGE : page * QUOTES_PER_PAGE]:
        tag_links = " ".join(f'<a class="tag" href="/tag/{t}/page/1/">{t}</a>' for t in tags)
        items.append(
            '<div class="quote" itemscope itemtype="http://schema.org/CreativeWork">'
            f'<span class="text" itemprop="text">“{html.escape(text)}”</span>'
            f'<span>by <small class="author" itemprop="author">{html.escape(author)}</small> '
            f'<a href="/author/{_slug(author)}">(about)</a></span>'
            f'<div class="tags">Tags: <meta class="keywords" itemprop="keywords" content="{",".join(tags)}"> {tag_links}</div>'
            "</div>"
        )
    pager = '<nav><ul class="pager">'
    if page > 1:
        pager += f'<li class="previous"><a href="/page/{page - 1}/"><span aria-hidden="true">&larr;</span> Previous</a></li>'
    if page < n_pages:
        pager += f'<li class="next"><a href="/page/{page + 1}/">Next <span aria-hidden="true">&rarr;</span></a></li>'
    pager += "</ul></nav>"
    return _page("Quotes to Scrape (local sandbox)", '<div class="container"><h1>Quotes</h1>' + "".join(items) + pager + "</div>")


def books_listing(page: int) -> str | None:
    """Render a catalogue listing page like books.toscrape.com/catalogue/page-N.html."""
    n_pages = -(-len(BOOKS) // BOOKS_PER_PAGE)
    if not 1 <= page <= n_pages:
        return None
    pods = []
    for i, (title, price, rating, stock, _) in enumerate(BOOKS[(page - 1) * BOOKS_PER_PAGE : page * BOOKS_PER_PAGE]):
        book_id = (page - 1) * BOOKS_PER_PAGE + i + 1
        href = f"{_slug(title)}_{book_id}/index.html"
        availability = "In stock" if stock > 0 else "Out of stock"
        short = title if len(title) < 20 else title[:17] + "..."
        pods.append(
            '<li class="col-xs-6"><article class="product_pod">'
            f'<p class="star-rating {RATINGS[rating]}"></p>'
            f'<h3><a href="{href}" title="{html.escape(title)}">{html.escape(short)}</a></h3>'
            f'<div class="product_price"><p class="price_color">£{price:.2f}</p>'
            f'<p class="instock availability">{availability}</p></div>'
            "</article></li>"
        )
    pager = '<ul class="pager">'
    if page < n_pages:
        pager += f'<li class="next"><a href="page-{page + 1}.html">next</a></li>'
    pager += "</ul>"
    return _page("Books to Scrape (local sandbox)", '<section><ol class="row">' + "".join(pods) + "</ol>" + pager + "</section>")


def books_detail(book_id: int) -> str | None:
    """Render a product page with a product information table like books.toscrape.com."""
    if not 1 <= book_id <= len(BOOKS):
        return None
    title, price, rating, stock, category = BOOKS[book_id - 1]
    availability = f"In stock ({stock} available)" if stock > 0 else "Out of stock (0 available)"
    rows = {
        "UPC": f"{book_id:016x}",
        "Product Type": "Books",
        "Price (excl. tax)": f"£{price:.2f}",
        "Price (incl. tax)": f"£{price:.2f}",
        "Tax": "£0.00",
        "Availability": availability,
        "Number of reviews": str(book_id % 4),
    }
    table = "".join(f"<tr><th>{k}</th><td>{html.escape(v)}</td></tr>" for k, v in rows.items())
    body = (
        '<ul class="breadcrumb"><li><a href="../../index.html">Home</a></li><li><a href="#">Books</a></li>'
        f'<li><a href="#">{category}</a></li><li class="active">{html.escape(title)}</li></ul>'
        f'<div class="product_main"><h1>{html.escape(title)}</h1><p class="price_color">£{price:.2f}</p></div>'
        f'<table class="table table-striped">{table}</table>'
    )
    return _page(title, body)


class _Handler(BaseHTTPRequestHandler):
    """Request handler that routes the paths of both sandbox sites."""

    site = "quotes"
    outage_paths: set[str] = set()  # paths that answer 503 exactly once

    def log_message(self, fmt, *args):  # keep the console clean
        pass

    def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8", headers=None):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (name required by BaseHTTPRequestHandler)
        path = self.path.split("?")[0]
        if path == "/robots.txt":
            return self._send(200, ROBOTS_TXT, "text/plain; charset=utf-8")
        if path in self.outage_paths:
            self.outage_paths.discard(path)
            print(f"    [sandbox] simulated temporary outage: 503 for {path} (Retry-After: 1)")
            return self._send(503, "Service temporarily unavailable", headers={"Retry-After": "1"})
        if path.startswith("/private/"):
            return self._send(200, _page("Private", "<p>Scrapers should never see this.</p>"))

        body = None
        if self.site == "quotes":
            if path == "/":
                body = quotes_page(1)
            elif path.startswith("/page/"):
                body = quotes_page(int(path.strip("/").split("/")[1]))
        else:
            if path in ("/", "/index.html"):
                body = books_listing(1)
            elif path.startswith("/catalogue/page-"):
                body = books_listing(int(path.rsplit("-", 1)[1].split(".")[0]))
            elif path.startswith("/catalogue/") and path.endswith("/index.html"):
                body = books_detail(int(path.split("/")[2].rsplit("_", 1)[1]))
        if body is None:
            return self._send(404, _page("Not found", "<p>404</p>"))
        return self._send(200, body)


@contextmanager
def local_sandbox(site: str = "quotes") -> Iterator[str]:
    """Start the local sandbox for `site` ("quotes" or "books") on a free port and yield its base URL."""
    outage = {"quotes": {"/page/2/"}, "books": {"/catalogue/page-2.html"}}[site]
    handler = type("Handler", (_Handler,), {"site": site, "outage_paths": set(outage)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def remote_reachable(url: str, timeout_s: float = 5.0) -> bool:
    """Return True if the remote site answers (any HTTP status) within the timeout."""
    try:
        requests.head(url, timeout=timeout_s)
        return True
    except requests.RequestException:
        return False


@contextmanager
def target_site(site: str, mode: str = "auto") -> Iterator[str]:
    """
    Yield the base URL to scrape.

    Parameters:
    site: str
        "quotes" or "books".
    mode: str
        "remote" = always the real sandbox site, "local" = always the local sandbox,
        "auto" = the real site if it can be reached, otherwise the local sandbox.
    """
    remote = REMOTE_URLS[site]
    if mode == "remote" or (mode == "auto" and remote_reachable(remote)):
        print(f"Target: {remote}")
        yield remote
        return
    if mode == "auto":
        print(f"{remote} is not reachable (offline or blocked) -> using the local sandbox instead.")
    with local_sandbox(site) as base_url:
        print(f"Target: local sandbox at {base_url}")
        yield base_url
