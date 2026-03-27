"""Internal arXiv API service helpers for query labels and search execution."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime

import httpx

from arxiv_browser.http_retry import retry_with_backoff
from arxiv_browser.models import ArxivSearchRequest, Paper
from arxiv_browser.parsing import build_arxiv_search_query, parse_arxiv_api_feed, parse_arxiv_date

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_API_TIMEOUT = 30
ARXIV_API_USER_AGENT = "arxiv-subscription-viewer/1.0"
ARXIV_API_MIN_INTERVAL_SECONDS = 3.0


def _search_params(
    request: ArxivSearchRequest, *, start: int, max_results: int
) -> dict[str, str | int]:
    """Build the arXiv query-string parameter set shared by sync and async callers."""
    search_query = build_arxiv_search_query(request.query, request.field, request.category)
    return {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": max(0, start),
        "max_results": max_results,
    }


def format_query_label(request: ArxivSearchRequest) -> str:
    """Build a human-readable query label for API mode UI."""
    try:
        return build_arxiv_search_query(request.query, request.field, request.category)
    except ValueError:
        return request.query or f"cat:{request.category}"


async def enforce_rate_limit(
    *,
    last_request_at: float,
    min_interval_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> tuple[float, float]:
    """Wait as needed to respect API rate limits.

    Returns:
        Tuple of (new_last_request_at, waited_seconds).
    """
    current = now()
    waited_seconds = 0.0
    elapsed = current - last_request_at
    if last_request_at > 0 and elapsed < min_interval_seconds:
        waited_seconds = min_interval_seconds - elapsed
        await sleep(waited_seconds)
    return now(), waited_seconds


async def fetch_page(
    *,
    client: httpx.AsyncClient,
    request: ArxivSearchRequest,
    start: int,
    max_results: int,
    timeout_seconds: int,
    user_agent: str,
) -> list[Paper]:
    """Fetch a single page of arXiv API results and parse to papers."""
    params = _search_params(request, start=start, max_results=max_results)
    headers = {"User-Agent": user_agent}

    async def _do_request() -> httpx.Response:
        resp = await client.get(
            ARXIV_API_URL,
            params=params,
            headers=headers,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        return resp

    response = await retry_with_backoff(_do_request, operation="arXiv API search")
    return parse_arxiv_api_feed(response.text)


def fetch_page_sync(
    *,
    request: ArxivSearchRequest,
    start: int,
    max_results: int,
    timeout_seconds: int = ARXIV_API_TIMEOUT,
    user_agent: str = ARXIV_API_USER_AGENT,
) -> list[Paper]:
    """Fetch a single page of arXiv API results for synchronous CLI startup."""
    response = httpx.get(
        ARXIV_API_URL,
        params=_search_params(request, start=start, max_results=max_results),
        headers={"User-Agent": user_agent},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return parse_arxiv_api_feed(response.text)


def fetch_latest_day_digest(
    *,
    request: ArxivSearchRequest,
    max_results: int,
    fetch_page_fn: Callable[..., list[Paper]] = fetch_page_sync,
    sleep_fn: Callable[[float], None] = time.sleep,
    min_interval_seconds: float = ARXIV_API_MIN_INTERVAL_SECONDS,
) -> list[Paper]:
    """Fetch all papers from the newest matching arXiv day."""
    start = 0
    target_day: date | None = None
    papers: list[Paper] = []
    seen_ids: set[str] = set()

    while True:
        page = fetch_page_fn(
            request=request,
            start=start,
            max_results=max_results,
            timeout_seconds=ARXIV_API_TIMEOUT,
            user_agent=ARXIV_API_USER_AGENT,
        )
        if not page:
            break

        reached_older_day = False
        for paper in page:
            parsed_date = parse_arxiv_date(paper.date)
            if parsed_date == datetime.min:
                continue
            day = parsed_date.date()
            if target_day is None:
                target_day = day
            if day != target_day:
                reached_older_day = True
                break
            if paper.arxiv_id not in seen_ids:
                papers.append(paper)
                seen_ids.add(paper.arxiv_id)

        if reached_older_day or len(page) < max_results:
            break

        start += max_results
        sleep_fn(min_interval_seconds)

    return papers


__all__ = [
    "ARXIV_API_MIN_INTERVAL_SECONDS",
    "ARXIV_API_TIMEOUT",
    "ARXIV_API_URL",
    "ARXIV_API_USER_AGENT",
    "enforce_rate_limit",
    "fetch_latest_day_digest",
    "fetch_page",
    "fetch_page_sync",
    "format_query_label",
]
