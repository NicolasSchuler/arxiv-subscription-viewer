"""Semantic Scholar API client, data models, and SQLite cache.

Provides async functions for fetching paper metadata and recommendations
from the Semantic Scholar Graph API, with persistent SQLite caching.

All API functions accept an httpx.AsyncClient and never raise — callers
get None / empty list on failure for graceful degradation.
"""

from __future__ import annotations

__all__ = [
    "S2_CITATIONS_PAGE_SIZE",
    "S2_CITATIONS_SCAN_CAP",
    "S2_CITATION_GRAPH_CACHE_TTL_DAYS",
    "S2_DEFAULT_CACHE_TTL_DAYS",
    "S2_REC_CACHE_TTL_DAYS",
    "CitationEntry",
    "S2PaperCacheSnapshot",
    "S2RecommendationsCacheSnapshot",
    "S2Request",
    "SemanticScholarPaper",
    "fetch_s2_citations",
    "fetch_s2_paper",
    "fetch_s2_recommendations",
    "fetch_s2_recommendations_with_status",
    "fetch_s2_references",
    "get_s2_db_path",
    "has_s2_citation_graph_cache",
    "init_s2_db",
    "load_s2_citation_graph",
    "load_s2_paper",
    "load_s2_paper_snapshot",
    "load_s2_recommendations",
    "load_s2_recommendations_snapshot",
    "parse_citation_entry",
    "parse_s2_paper_response",
    "save_s2_citation_graph",
    "save_s2_paper",
    "save_s2_paper_not_found",
    "save_s2_recommendations",
]

import logging
from typing import Any, Literal, overload

import httpx

import arxiv_browser.semantic_scholar_cache as _s2_cache
from arxiv_browser.http_retry import retry_with_backoff
from arxiv_browser.semantic_scholar_cache import (
    get_s2_db_path,
    has_s2_citation_graph_cache,
    init_s2_db,
    load_s2_citation_graph,
    load_s2_paper,
    load_s2_paper_snapshot,
    load_s2_recommendations,
    load_s2_recommendations_snapshot,
    save_s2_citation_graph,
    save_s2_paper,
    save_s2_paper_not_found,
    save_s2_recommendations,
)
from arxiv_browser.semantic_scholar_models import (
    S2_API_BASE,
    S2_CITATION_FIELDS,
    S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    S2_CITATIONS_PAGE_SIZE,
    S2_CITATIONS_SCAN_CAP,
    S2_DEFAULT_CACHE_TTL_DAYS,
    S2_INITIAL_BACKOFF,
    S2_MAX_CITATIONS,
    S2_MAX_REFERENCES,
    S2_MAX_RETRIES,
    S2_PAPER_FIELDS,
    S2_REC_BASE,
    S2_REC_CACHE_TTL_DAYS,
    S2_REC_FIELDS,
    S2_REQUEST_TIMEOUT,
    CitationEntry,
    S2PaperCacheSnapshot,
    S2RecommendationsCacheSnapshot,
    S2Request,
    SemanticScholarPaper,
    parse_citation_entry,
    parse_s2_paper_response,
)

logger = logging.getLogger(__name__)
sqlite3 = _s2_cache.sqlite3
_citation_entry_to_json = _s2_cache._citation_entry_to_json
_is_fresh = _s2_cache._is_fresh
_json_to_citation_entry = _s2_cache._json_to_citation_entry
_json_to_paper = _s2_cache._json_to_paper
_load_s2_paper_fetch_state = _s2_cache._load_s2_paper_fetch_state
_load_s2_recommendation_fetch_state = _s2_cache._load_s2_recommendation_fetch_state
_paper_to_json = _s2_cache._paper_to_json


# ============================================================================
# API Functions (async, accept httpx.AsyncClient)
# ============================================================================


def _build_s2_headers(request: S2Request) -> dict[str, str]:
    """Build request headers for one S2 API call."""
    headers: dict[str, str] = {}
    if request.api_key:
        headers["x-api-key"] = request.api_key
    return headers


async def _s2_get_with_retry(
    client: httpx.AsyncClient,
    request: S2Request,
) -> httpx.Response | None:
    """Send a GET request with retries and return only the response object."""
    response, _ = await _s2_get_with_retry_status(client, request)
    return response


async def _s2_get_with_retry_status(
    client: httpx.AsyncClient,
    request: S2Request,
) -> tuple[httpx.Response | None, bool]:
    """Send a GET request with exponential backoff retry on 429/5xx.

    Returns the response on 200, or None on terminal failure (404, other 4xx,
    exhausted retries, timeout, or network error). Never raises. The boolean
    indicates whether the request path completed cleanly enough for callers to
    cache an empty result (404 => complete, malformed/transport error => not).
    """
    headers = _build_s2_headers(request)

    async def _do_request() -> httpx.Response:
        response = await client.get(
            request.url,
            params=request.params,
            headers=headers,
            timeout=request.timeout,
        )
        response.raise_for_status()
        return response

    try:
        response = await retry_with_backoff(
            _do_request,
            max_retries=S2_MAX_RETRIES - 1,
            backoff_base=S2_INITIAL_BACKOFF,
            operation=request.label,
        )
        return response, True
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info("%s not found", request.label)
            return None, True
        else:
            logger.warning("%s returned %d", request.label, exc.response.status_code)
        return None, False
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
        logger.warning("%s timeout/connection error after retries", request.label)
        return None, False
    except httpx.HTTPError:
        logger.warning("%s HTTP error", request.label, exc_info=True)
        return None, False


def _parse_json_object(response: httpx.Response, label: str) -> dict[str, Any] | None:
    """Parse a response body as JSON object, returning None on malformed payloads."""
    try:
        payload = response.json()
    except ValueError:
        logger.warning("%s returned invalid JSON", label, exc_info=True)
        return None
    if not isinstance(payload, dict):
        logger.warning("%s returned non-object JSON payload", label)
        return None
    return payload


def _s2_list_result[T](
    items: list[T], complete: bool, include_status: bool
) -> list[T] | tuple[list[T], bool]:
    return (items, complete) if include_status else items


def _payload_list(payload: dict[str, Any], key: str, label: str) -> list[Any] | None:
    data = payload.get(key)
    if not isinstance(data, list):
        logger.warning("%s returned non-list %s", label, key)
        return None
    return data


def _parse_recommended_papers(
    payload: dict[str, Any], arxiv_id: str
) -> list[SemanticScholarPaper] | None:
    papers_data = _payload_list(payload, "recommendedPapers", f"S2 recs arXiv:{arxiv_id}")
    if papers_data is None:
        return None
    papers: list[SemanticScholarPaper] = []
    for item in papers_data:
        if not isinstance(item, dict):
            continue
        parsed = parse_s2_paper_response(item)
        if parsed is not None:
            papers.append(parsed)
    return papers


def _parse_citation_entries(
    payload: dict[str, Any], label: str, paper_key: str
) -> list[CitationEntry] | None:
    data = _payload_list(payload, "data", label)
    if data is None:
        return None
    entries: list[CitationEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        entry = parse_citation_entry(item.get(paper_key) or {})
        if entry:
            entries.append(entry)
    return entries


def _sort_citation_entries(entries: list[CitationEntry]) -> list[CitationEntry]:
    return sorted(entries, key=lambda entry: entry.citation_count, reverse=True)


@overload
async def fetch_s2_paper(
    arxiv_id: str,
    client: httpx.AsyncClient,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> SemanticScholarPaper | None:
    """Overload — returns the paper only when include_status is False."""
    ...


@overload
async def fetch_s2_paper(
    arxiv_id: str,
    client: httpx.AsyncClient,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[SemanticScholarPaper | None, bool]:
    """Overload — returns a (paper, complete) tuple when include_status is True."""
    ...


async def fetch_s2_paper(
    arxiv_id: str,
    client: httpx.AsyncClient,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> SemanticScholarPaper | None | tuple[SemanticScholarPaper | None, bool]:
    """Fetch paper metadata from S2 Graph API.

    When include_status=True, returns (paper, complete) where complete=False
    means the request or parse path failed. A 404 counts as complete and yields
    (None, True).
    """
    response, complete = await _s2_get_with_retry_status(
        client,
        S2Request(
            url=f"{S2_API_BASE}/paper/ARXIV:{arxiv_id}",
            params={"fields": S2_PAPER_FIELDS},
            api_key=api_key,
            timeout=timeout,
            label=f"S2 paper arXiv:{arxiv_id}",
        ),
    )
    if response is None:
        return (None, complete) if include_status else None
    payload = _parse_json_object(response, f"S2 paper arXiv:{arxiv_id}")
    if payload is None:
        return (None, False) if include_status else None
    paper = parse_s2_paper_response(payload, arxiv_id=arxiv_id)
    if include_status:
        return paper, paper is not None
    return paper


async def fetch_s2_recommendations(
    arxiv_id: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
) -> list[SemanticScholarPaper]:
    """Fetch recommended papers from S2 Recommendations API."""
    result = await fetch_s2_recommendations_with_status(
        arxiv_id, client, limit=limit, api_key=api_key, timeout=timeout
    )
    return result if isinstance(result, list) else result[0]


@overload
async def fetch_s2_recommendations_with_status(
    arxiv_id: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: Literal[False] = ...,
) -> list[SemanticScholarPaper]:
    """Overload — returns recommended papers only when include_status is False."""
    ...


@overload
async def fetch_s2_recommendations_with_status(
    arxiv_id: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: Literal[True] = ...,
) -> tuple[list[SemanticScholarPaper], bool]:
    """Overload — returns a (papers, complete) tuple when include_status is True."""
    ...


async def fetch_s2_recommendations_with_status(
    arxiv_id: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[SemanticScholarPaper] | tuple[list[SemanticScholarPaper], bool]:
    """Fetch S2 recommendations while preserving request completion status."""
    response, complete = await _s2_get_with_retry_status(
        client,
        S2Request(
            url=f"{S2_REC_BASE}/papers/forpaper/ARXIV:{arxiv_id}",
            params={"fields": S2_REC_FIELDS, "limit": str(limit)},
            api_key=api_key,
            timeout=timeout,
            label=f"S2 recs arXiv:{arxiv_id}",
        ),
    )
    if response is None:
        return _s2_list_result([], complete, include_status)
    payload = _parse_json_object(response, f"S2 recs arXiv:{arxiv_id}")
    if payload is None:
        return _s2_list_result([], False, include_status)
    papers = _parse_recommended_papers(payload, arxiv_id)
    if papers is None:
        return _s2_list_result([], False, include_status)
    return _s2_list_result(papers, True, include_status)


@overload
async def fetch_s2_references(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> list[CitationEntry]:
    """Overload — returns reference entries only when include_status is False."""
    ...


@overload
async def fetch_s2_references(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[list[CitationEntry], bool]:
    """Overload — returns a (entries, complete) tuple when include_status is True."""
    ...


async def fetch_s2_references(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = S2_MAX_REFERENCES,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[CitationEntry] | tuple[list[CitationEntry], bool]:
    """Fetch papers cited by the given paper. Sorted by citation_count desc.

    When include_status=True, returns (entries, complete) where complete=False
    means the request/parse path failed and callers should avoid caching.
    """
    response = await _s2_get_with_retry(
        client,
        S2Request(
            url=f"{S2_API_BASE}/paper/{paper_id}/references",
            params={"fields": S2_CITATION_FIELDS, "limit": str(limit)},
            api_key=api_key,
            timeout=timeout,
            label=f"S2 refs {paper_id}",
        ),
    )
    if response is None:
        return _s2_list_result([], False, include_status)
    payload = _parse_json_object(response, f"S2 refs {paper_id}")
    if payload is None:
        return _s2_list_result([], False, include_status)
    entries = _parse_citation_entries(payload, f"S2 refs {paper_id}", "citedPaper")
    if entries is None:
        return _s2_list_result([], False, include_status)
    return _s2_list_result(_sort_citation_entries(entries), True, include_status)


@overload
async def fetch_s2_citations(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> list[CitationEntry]:
    """Overload — returns citation entries only when include_status is False."""
    ...


@overload
async def fetch_s2_citations(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[list[CitationEntry], bool]:
    """Overload — returns a (entries, complete) tuple when include_status is True."""
    ...


async def fetch_s2_citations(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = S2_MAX_CITATIONS,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[CitationEntry] | tuple[list[CitationEntry], bool]:
    """Fetch papers citing the given paper. Top N by citation_count.

    Uses an adaptive scan cap based on limit to avoid unnecessary API calls.
    When include_status=True, returns (entries, complete) where complete=False
    means at least one page fetch/parse failed.
    """
    if limit <= 0:
        empty: list[CitationEntry] = []
        return (empty, True) if include_status else empty

    entries: list[CitationEntry] = []
    offset = 0
    scan_cap = _citation_scan_cap(limit)
    complete = True
    while offset < scan_cap:
        page_entries, page_complete, page_size, page_limit = await _fetch_s2_citation_page(
            client, paper_id, offset, scan_cap, api_key, timeout
        )
        if not page_complete:
            complete = False
            break
        if not page_entries:
            break
        entries.extend(page_entries)
        if page_size < page_limit:
            break
        offset += page_limit

    trimmed = _sort_citation_entries(entries)[:limit]
    return _s2_list_result(trimmed, complete, include_status)


def _citation_scan_cap(limit: int) -> int:
    return min(S2_CITATIONS_SCAN_CAP, max(S2_CITATIONS_PAGE_SIZE * 2, limit * 4))


async def _fetch_s2_citation_page(
    client: httpx.AsyncClient,
    paper_id: str,
    offset: int,
    scan_cap: int,
    api_key: str,
    timeout: int,
) -> tuple[list[CitationEntry], bool, int, int]:
    page_limit = min(S2_CITATIONS_PAGE_SIZE, scan_cap - offset)
    label = f"S2 cites {paper_id}"
    response = await _s2_get_with_retry(
        client,
        S2Request(
            url=f"{S2_API_BASE}/paper/{paper_id}/citations",
            params={
                "fields": S2_CITATION_FIELDS,
                "limit": str(page_limit),
                "offset": str(offset),
            },
            api_key=api_key,
            timeout=timeout,
            label=label,
        ),
    )
    if response is None:
        return [], False, 0, page_limit
    payload = _parse_json_object(response, label)
    if payload is None:
        return [], False, 0, page_limit
    data = _payload_list(payload, "data", label)
    if data is None:
        return [], False, 0, page_limit
    entries = _parse_citation_entries(payload, label, "citingPaper") or []
    return entries, True, len(data), page_limit
