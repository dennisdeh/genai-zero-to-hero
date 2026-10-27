## Web Scraping

When data is only available as web pages - product catalogues, public registers, news, documentation -
*web scraping* turns the HTML into structured data. Scraping is powerful but fragile and comes with legal
and ethical responsibilities. Before scraping, always check whether the data is available in a better way:
an official API (chapter c06), a bulk download or an open-data portal are more reliable and usually preferred
by the site owner.

### How a scraper works
1. **Fetch** a page with an HTTP GET request (`requests`).
2. **Parse** the HTML into a tree (BeautifulSoup with the built-in `html.parser`, or the faster `lxml`).
3. **Select** the elements of interest with CSS selectors (`soup.select("div.quote span.text")`) and extract
   text and attributes (`get_text(strip=True)`, `tag["href"]`).
4. **Navigate**: follow pagination links ("Next") and links to detail pages; convert relative links to
   absolute URLs with `urllib.parse.urljoin`.
5. **Clean and store**: convert strings to numbers and dates, build a tidy DataFrame, save as Parquet/CSV.

Use the browser's developer tools (right click -> *Inspect*) to find stable selectors: prefer semantic class
names and attributes (`article.product_pod`, `h3 > a[title]`) over positions in the tree. Pages that build
their content with JavaScript cannot be scraped this way; check the browser's network tab - often the page
loads its data from a JSON API that can be called directly - or use a browser automation tool such as
Playwright or Selenium.

### Scraping politely
The class `PoliteScraper` in `utils/polite_scraper.py` bundles the practices every scraper should follow:

- **robots.txt** (Robots Exclusion Protocol, RFC 9309) tells crawlers which paths they may visit. It is read
  with `urllib.robotparser.RobotFileParser` and checked before every request; a `Crawl-delay` is honoured.
  A missing robots.txt (404) allows everything, a 401/403 is treated as "disallow all".
- **Identify yourself** with a descriptive `User-Agent` including a URL or e-mail address, so that site owners
  can contact you instead of blocking you.
- **Rate limiting**: wait at least `delay_s` (default 1 second) between requests. A scraper can easily send
  hundreds of requests per second - enough to degrade a small site.
- **Retries with exponential back-off** for transient failures (HTTP 429 *Too Many Requests* and 5xx),
  respecting the `Retry-After` header - configured once on a `requests.Session` via urllib3's `Retry`.
- **Timeouts** on every request (requests waits forever by default).
- **Stay on the site**: the client refuses to follow links to other domains.
- **Encoding**: if the server does not declare a charset, requests assumes ISO-8859-1 for HTML; the client then
  uses the detected encoding, which avoids mojibake such as "Â£" instead of "£".

Further good practice: cache pages during development so you do not download them again for every parser
change; scrape at off-peak hours; download only what you need; stop when you receive many errors.

### The examples
Both examples use sandbox sites built for practising scraping: https://quotes.toscrape.com and
https://books.toscrape.com. With `--site auto` (default) the scripts use the real site if it is reachable and
otherwise start a **local sandbox** (`utils/sandbox_site.py`): a small HTTP server in a background thread that
serves pages with the same HTML structure (our own content), a robots.txt with a disallowed path and a
Crawl-delay, and a simulated temporary outage (one HTTP 503 with `Retry-After`) that the retry logic absorbs.
This makes the examples reproducible and runnable offline.

`main_1_scrape_quotes.py` reads robots.txt and shows which URLs may be fetched, crawls the paginated list of
quotes by following the "Next" link, extracts text, author, author link and tags of every quote and analyses
the result (quotes per author, most frequent tags after `explode`). The list-valued `tags` column is kept as a
list in Parquet and joined into a string for CSV.

`main_2_scrape_books.py` is a two-level scraper as used for product catalogues: listing pages give title
(from the `title` attribute, because the visible text is truncated), price, star rating (encoded in a CSS
class: `star-rating Three`) and availability; detail pages contain a product-information table
(`<tr><th>UPC</th><td>...</td></tr>`) that is turned into columns. Values are cleaned with regular expressions
(`"£51.77"` -> `51.77`, `"In stock (22 available)"` -> `22`) and both levels are joined on the URL. A failing
detail page is logged and skipped instead of aborting the whole job. For well-formed tables,
`pandas.read_html` is a convenient alternative.

### Legal and ethical considerations
This section is general information, not legal advice; the legal situation depends on the country, the
data and the purpose. Points to consider before scraping:

- **Terms of service.** Many sites forbid automated access in their terms. Violating them can have
  contractual consequences, in particular when you have to log in (and thereby accept the terms) to see
  the data. Never circumvent logins, paywalls, CAPTCHAs or other technical protection measures.
- **robots.txt** is not a law, but it expresses the owner's wishes. Respecting it is standard practice and
  relevant for the legal assessment in several jurisdictions.
- **Copyright and database rights.** Facts are generally not protected, but texts, images and the
  selection/arrangement of data can be. The EU additionally protects databases (Database Directive 96/9/EC,
  *sui generis* right). The EU Copyright Directive 2019/790 permits text and data mining under conditions
  (Article 3 for research organisations; Article 4 for everyone else unless the rights holder has opted out in a
  machine-readable way, e.g. in robots.txt or the site's terms). In the US, "fair use" and court decisions
  such as *hiQ Labs v. LinkedIn* shape the picture.
- **Personal data.** Names, e-mail addresses, photos or profiles are personal data under the GDPR even when
  publicly visible. Processing them needs a legal basis, a clear purpose, data minimisation and respect for
  the data subjects' rights. Avoid collecting personal data unless it is essential.
- **Load and harm.** Scraping must not impair the service (rate limiting!). Overloading a site can be
  unlawful in itself.
- **Transparency.** Identify your scraper, document what you collected, when and from where (lineage), and
  honour requests to stop.

A good rule of thumb: if you would be uncomfortable explaining your scraper to the site owner, do not run it.
