"""Internal arXiv API service helpers for query labels, rate limits, and page fetches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx

from arxiv_browser.models import ArxivSearchRequest, Paper
from arxiv_browser.parsing import build_arxiv_search_query, parse_arxiv_api_feed

ARXIV_API_URL = "https://export.arxiv.org/api/query"


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
    client: httpx.AsyncClient | None,
    request: ArxivSearchRequest,
    start: int,
    max_results: int,
    timeout_seconds: int,
    user_agent: str,
) -> list[Paper]:
    """Fetch a single page of arXiv API results and parse to papers."""
    search_query = build_arxiv_search_query(request.query, request.field, request.category)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    }
    headers = {"User-Agent": user_agent}

    if client is not None:
        response = await client.get(
            ARXIV_API_URL,
            params=params,
            headers=headers,
            timeout=timeout_seconds,
        )
    else:
        async with httpx.AsyncClient() as tmp_client:
            response = await tmp_client.get(
                ARXIV_API_URL,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )

    response.raise_for_status()
    return parse_arxiv_api_feed(response.text)


__all__ = [
    "enforce_rate_limit",
    "fetch_page",
    "format_query_label",
]
