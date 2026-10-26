## Consuming REST APIs

An API (application programming interface) is the preferred way to obtain data from another system: it
returns structured data (usually JSON), it is documented, versioned and meant to be used by programs - unlike
web pages, which are meant for humans (chapter c04). Most public data APIs follow the REST style over HTTP.

### HTTP and REST in a nutshell
- A **resource** is identified by a URL, e.g. `https://api.github.com/repos/pandas-dev/pandas`.
- The **method** expresses the action: `GET` reads, `POST` creates, `PUT`/`PATCH` update, `DELETE` removes.
  Data sourcing is almost entirely `GET`, which is *safe* and *idempotent* - and therefore safe to retry.
- **Query parameters** filter and shape the result: `?latitude=55.68&hourly=temperature_2m`. Pass them as a
  dictionary (`params=`) and let `requests` encode them; never assemble query strings by hand.
- **Headers** carry metadata: `Accept`, `User-Agent`, `Authorization` (API keys or tokens), and in responses
  e.g. rate-limit information and pagination links.
- The **status code** tells what happened: 2xx success; 4xx client error (400 bad request, 401 not
  authenticated, 403 forbidden or rate limit exceeded, 404 not found, 429 too many requests) - fix the request,
  retrying will not help (except for 429); 5xx server error - often transient, retry with back-off.

### A robust client
`utils/http_client.py` implements the building blocks used by all examples:

- **`requests.Session`** re-uses TCP and TLS connections (much faster for many calls) and shares default
  headers such as `User-Agent` and `Accept`.
- **Retries with exponential back-off** via urllib3's `Retry` mounted on the session: connection errors and
  the status codes 429, 500, 502, 503 and 504 are retried up to three times, waiting 0.5 s, 1 s, 2 s, and
  honouring the server's `Retry-After` header. Only `GET` is retried, because only idempotent requests can be
  repeated safely.
- **Timeouts** on every call; without them a hanging server blocks the program forever.
- **Error translation**: status codes and exceptions are turned into an `ApiError` with an explanation
  (rate limit exhausted with reset time, not authorised, not found, bad parameters, server error, invalid JSON).
- **Caching**: responses are stored as JSON files keyed by a hash of URL and parameters, with a time-to-live.
  During development this avoids hitting the API (and the rate limit) again for every code change, and it
  makes the scripts work offline once the data has been fetched. HTTP itself offers conditional requests
  (`ETag`/`If-None-Match`) for the same purpose; libraries such as `requests-cache` implement caching
  transparently.

API keys and tokens belong in environment variables or a secret store, never in code; send them in a header
(`Authorization: Bearer <token>`) rather than in the URL, where they end up in logs.

### Example 1: weather forecasts (Open-Meteo)
`main_1_open_meteo_weather.py` requests hourly forecasts for four cities from Open-Meteo, a free API without
registration. The response is *column oriented*:

```json
{"latitude": 55.68, "longitude": 12.56, "timezone": "Europe/Copenhagen", "elevation": 9.0,
 "hourly_units": {"temperature_2m": "°C", ...},
 "hourly": {"time": ["2025-03-01T00:00", ...], "temperature_2m": [3.1, ...], ...}}
```

`pd.json_normalize` flattens the nested metadata of all responses into one table (columns such as
`hourly_units.temperature_2m`), while `pd.DataFrame(response["hourly"])` turns the dictionary of equally long
arrays directly into columns. The per-city frames are concatenated into one *tidy* (long) table - one row per
city and hour - which is aggregated into daily minimum/maximum temperature, precipitation and maximum wind.

### Example 2: pagination and nested JSON (GitHub)
Collections are returned in pages. Common schemes are page numbers (`?page=2&per_page=50`), offset/limit,
cursors (an opaque token for the next page in the response body) and **Link headers** (RFC 8288):

```text
Link: <https://api.github.com/repositories/<id>/issues?page=2&per_page=50>; rel="next",
      <https://api.github.com/repositories/<id>/issues?page=40&per_page=50>; rel="last"
```

`requests` parses this header into `response.links`. `main_2_github_pagination.py` follows the `next` link
until there is none (or `--max-pages` is reached), which is more robust than computing page numbers yourself.
GitHub's issue objects are deeply nested (`user`, `labels`, `reactions`, `pull_request`); `pd.json_normalize`
flattens nested dictionaries into dotted columns (`user.login`, `reactions.total_count`), while lists of
dictionaries such as `labels` are converted explicitly (or with `record_path=` for one-to-many tables).
The script also reads the **rate-limit headers** (`X-RateLimit-Limit`, `X-RateLimit-Remaining`,
`X-RateLimit-Reset`): unauthenticated clients get 60 requests per hour, a token in `GITHUB_TOKEN` raises the
limit to 5,000.

### Example 3: time series (ECB exchange rates via Frankfurter)
`main_3_exchange_rates.py` queries the Frankfurter API, which serves the euro reference rates published by the
European Central Bank. The time-series endpoint returns a dictionary keyed by date
(`{"rates": {"2025-01-02": {"USD": 1.03, ...}, ...}}`). Here `pd.DataFrame.from_dict(..., orient="index")` is the
right tool - `json_normalize` would produce one row with one column per date and currency, which the script
prints for comparison. The series is then analysed (log returns, annualised volatility, monthly averages) and
stored. Note that rates exist only for TARGET business days.

### Checklist for consuming an API
1. Read the documentation: authentication, rate limits, pagination, terms of use and licence of the data.
2. Use a session with timeouts, retries with back-off and a meaningful `User-Agent`.
3. Keep credentials in the environment; never log them.
4. Handle errors explicitly and distinguish "fix the request" (4xx) from "try again later" (429/5xx).
5. Respect rate limits; cache responses; fetch incrementally (e.g. `since=` parameters) instead of everything.
6. Store the raw responses (chapter c00) before normalising them, and record when they were fetched.
