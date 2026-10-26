"""
A small, robust JSON-over-HTTP client for consuming REST APIs.

Building blocks shown here (used by all examples in this chapter):
    - requests.Session: re-uses TCP/TLS connections and shares headers (User-Agent, auth) across calls,
    - urllib3 Retry: automatic retries with exponential back-off for 429/5xx and connection errors,
      respecting the server's Retry-After header,
    - timeouts on every request (requests waits forever by default!),
    - clear error messages for the typical failure classes,
    - a simple file cache: identical requests within a time-to-live are answered from outputs/ instead
      of hitting the API again (saves rate limit, speeds up development, works offline once cached).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "gzh-api-client/1.0 (+https://github.com/dennisdeh/genai-zero-to-hero; educational use)"


class ApiError(RuntimeError):
    """Raised when an API call fails, with a human-readable explanation."""


@dataclass
class ApiResponse:
    """The parts of an HTTP response we care about (serialisable for the cache)."""

    data: object
    status: int
    headers: dict = field(default_factory=dict)  # selected headers, lower-case names
    links: dict = field(default_factory=dict)  # parsed 'Link' header, used for pagination
    from_cache: bool = False


def build_session(
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
    backoff_factor: float = 0.5,
    headers: dict | None = None,
) -> requests.Session:
    """
    Create a requests.Session with a retry policy and default headers.

    Parameters:
    user_agent: str
        Identify your client; many APIs reject requests without a User-Agent.
    retries: int
        Maximum number of retries for connection errors and the status codes 429, 500, 502, 503, 504.
    backoff_factor: float
        Sleep backoff_factor * 2**(n-1) seconds before retry n (0.5 -> 0.5s, 1s, 2s, ...).
    headers: dict, None
        Extra default headers (e.g. Accept or Authorization).
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    session.headers.update(headers or {})
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],  # only retry idempotent requests
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _cache_path(cache_dir: str, url: str, params: dict | None) -> str:
    key = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
    return os.path.join(cache_dir, hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def get_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    timeout_s: float = 15.0,
    cache_dir: str | None = None,
    ttl_s: float = 3600.0,
) -> ApiResponse:
    """
    GET a JSON resource with timeout, retries (via the session), error handling and optional caching.

    Parameters:
    session: requests.Session
        Session created with build_session().
    url: str
        Full URL of the endpoint.
    params: dict, None
        Query parameters (requests URL-encodes them for you - never build query strings by hand).
    timeout_s: float
        Timeout in seconds for connecting and for reading the response.
    cache_dir: str, None
        If given, responses are cached as JSON files in this directory.
    ttl_s: float
        Time-to-live of cached responses in seconds.

    Return:
        ApiResponse with the decoded JSON in .data.

    Raises:
        ApiError with an explanation of what went wrong.
    """
    # 1. Cache lookup
    if cache_dir:
        path = _cache_path(cache_dir, url, params)
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl_s:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            return ApiResponse(**cached, from_cache=True)

    # 2. Request
    try:
        resp = session.get(url, params=params, timeout=timeout_s)
    except requests.exceptions.Timeout as e:
        raise ApiError(f"Timeout after {timeout_s}s calling {url}") from e
    except requests.exceptions.ConnectionError as e:
        raise ApiError(f"Could not connect to {url} (offline, DNS, proxy or firewall?): {e}") from e

    # 3. Status handling
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        raise ApiError(f"Rate limit exhausted; it resets at {time.strftime('%H:%M:%S', time.localtime(reset))}")
    if resp.status_code in (401, 403):
        raise ApiError(f"{resp.status_code} - not authorised for {resp.url}: check your token / permissions")
    if resp.status_code == 404:
        raise ApiError(f"404 - resource not found: {resp.url}")
    if 400 <= resp.status_code < 500:
        raise ApiError(f"{resp.status_code} - the request was rejected (check the parameters): {resp.text[:300]}")
    if resp.status_code >= 500:
        raise ApiError(f"{resp.status_code} - server error at {resp.url} (still failing after retries)")
    try:
        data = resp.json()
    except ValueError as e:
        raise ApiError(f"Response from {resp.url} is not valid JSON: {resp.text[:200]!r}") from e

    # keep a few useful headers, with lower-case names (HTTP header names are case-insensitive)
    keep = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith(("x-ratelimit", "content-type", "date"))}
    result = ApiResponse(data=data, status=resp.status_code, headers=keep, links=resp.links)

    # 4. Cache store
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"data": data, "status": result.status, "headers": keep, "links": resp.links}, f)
    return result
