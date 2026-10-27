"""
A small, reusable "polite" HTTP client for web scraping.

Good scraping practice built into this class:
    - identify yourself with a descriptive User-Agent (including a way to contact you),
    - read robots.txt and never fetch disallowed URLs; honour a Crawl-delay if one is given,
    - rate limit: wait a minimum delay between two requests to the same site,
    - retry transient failures (429 Too Many Requests, 5xx) with exponential back-off and respect the
      server's Retry-After header,
    - always use timeouts, and re-use one connection pool (requests.Session),
    - decode the HTML with the right character encoding.
"""

import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "gzh-learning-scraper/1.0 (+https://github.com/dennisdeh/genai-zero-to-hero; educational use)"


class PoliteScraper:
    """
    HTTP client that respects robots.txt, rate limits itself and retries transient errors.

    Parameters:
    base_url: str
        Root URL of the site, e.g. "https://quotes.toscrape.com". robots.txt is read from there.
    user_agent: str
        The User-Agent header sent with every request (and used for the robots.txt rules).
    delay_s: float
        Minimum number of seconds between two requests (increased if robots.txt asks for more).
    max_retries: int
        Number of retries for connection errors and the status codes 429, 500, 502, 503, 504.
    backoff_factor: float
        Exponential back-off: sleep backoff_factor * 2**(retry-1) seconds between retries.
    timeout_s: float
        Timeout (connect and read) for each request.
    """

    def __init__(
        self,
        base_url: str,
        user_agent: str = DEFAULT_USER_AGENT,
        delay_s: float = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout_s: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.n_requests = 0
        self._last_request = 0.0

        # 1. One session for all requests: connection pooling, shared headers and the retry policy
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            respect_retry_after_header=True,
            raise_on_status=False,  # return the last response instead of raising, we check it ourselves
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # 2. robots.txt
        self.robots = self._load_robots()
        crawl_delay = self.robots.crawl_delay(user_agent)
        self.delay_s = max(delay_s, float(crawl_delay or 0))

    def _load_robots(self) -> urllib.robotparser.RobotFileParser:
        """
        Download and parse robots.txt with our own session (so that the User-Agent, timeouts and proxies
        apply). Semantics follow the standard library: a missing robots.txt (4xx) allows everything,
        401/403 disallow everything.
        """
        rp = urllib.robotparser.RobotFileParser()
        url = urljoin(self.base_url, "/robots.txt")
        rp.set_url(url)
        resp = self.session.get(url, timeout=self.timeout_s)
        self.n_requests += 1
        self._last_request = time.monotonic()
        if resp.status_code in (401, 403):
            rp.disallow_all = True
        elif 400 <= resp.status_code < 500:
            rp.allow_all = True
        else:
            resp.raise_for_status()
            rp.parse(resp.text.splitlines())
        rp.modified()  # marks the rules as loaded (otherwise can_fetch() returns False)
        return rp

    def can_fetch(self, url: str) -> bool:
        """Return True if robots.txt allows our User-Agent to fetch url."""
        return self.robots.can_fetch(self.user_agent, url)

    def _wait(self) -> None:
        """Sleep until at least delay_s seconds have passed since the previous request."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)

    def get(self, url: str) -> requests.Response:
        """
        GET a URL politely: robots.txt check, rate limiting, retries, timeout and status check.

        Raises:
            PermissionError if robots.txt disallows the URL, requests.HTTPError for 4xx/5xx responses.
        """
        url = urljoin(self.base_url, url)
        if urlparse(url).netloc != urlparse(self.base_url).netloc:
            raise ValueError(f"Refusing to leave the site {self.base_url}: {url}")
        if not self.can_fetch(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        self._wait()
        resp = self.session.get(url, timeout=self.timeout_s)
        self._last_request = time.monotonic()
        self.n_requests += 1
        resp.raise_for_status()
        # requests falls back to ISO-8859-1 for text/html without a charset ("Â£" instead of "£");
        # in that case we let it detect the encoding from the content
        if "charset" not in resp.headers.get("Content-Type", "").lower():
            resp.encoding = resp.apparent_encoding
        return resp

    def get_soup(self, url: str) -> BeautifulSoup:
        """GET a URL and parse it with BeautifulSoup (the built-in 'html.parser'; 'lxml' is faster if installed)."""
        return BeautifulSoup(self.get(url).text, "html.parser")
