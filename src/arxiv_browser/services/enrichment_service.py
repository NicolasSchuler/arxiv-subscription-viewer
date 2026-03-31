"""Internal enrichment services (Semantic Scholar + HuggingFace cache orchestration)."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import httpx

from arxiv_browser.huggingface import (
    HFDailyCacheSnapshot,
    HuggingFacePaper,
    fetch_hf_daily_papers,
    load_hf_daily_cache_snapshot,
    save_hf_daily_cache,
)
from arxiv_browser.semantic_scholar import (
    SemanticScholarPaper,
    fetch_s2_paper,
    fetch_s2_recommendations_with_status,
    load_s2_paper_snapshot,
    load_s2_recommendations_snapshot,
    save_s2_paper,
    save_s2_paper_not_found,
    save_s2_recommendations,
)

logger = logging.getLogger(__name__)


def _normalize_db_path(db_path: Path | str) -> Path:
    """Accept string paths from legacy callers while using ``Path`` internally."""
    return Path(db_path)


async def _best_effort_cache_write(write_fn, *args) -> None:
    """Persist cache state when possible without dropping successful fetches."""
    try:
        await asyncio.to_thread(write_fn, *args)
    except (OSError, sqlite3.Error) as exc:
        write_name = getattr(write_fn, "__name__", type(write_fn).__name__)
        logger.warning("Failed to persist enrichment cache via %s: %s", write_name, exc)


@dataclass(slots=True, frozen=True)
class S2PaperFetchResult:
    """Resolved S2 paper fetch state after cache lookup and optional remote fetch."""

    state: Literal["found", "not_found", "unavailable"]
    paper: SemanticScholarPaper | None
    complete: bool
    from_cache: bool


@dataclass(slots=True, frozen=True)
class HFDailyFetchResult:
    """Resolved HF daily fetch state after cache lookup and optional remote fetch."""

    state: Literal["found", "empty", "unavailable"]
    papers: list[HuggingFacePaper]
    complete: bool
    from_cache: bool


@dataclass(slots=True, frozen=True)
class S2RecommendationsFetchResult:
    """Resolved S2 recommendations fetch state after cache lookup and optional remote fetch."""

    state: Literal["found", "empty", "unavailable"]
    papers: list[SemanticScholarPaper]
    complete: bool
    from_cache: bool


async def load_or_fetch_s2_paper_result(
    *,
    arxiv_id: str,
    db_path: Path | str,
    cache_ttl_days: int,
    client: httpx.AsyncClient,
    api_key: str,
) -> S2PaperFetchResult:
    """Load S2 paper data from cache, or fetch and persist on cache miss."""
    db_path = _normalize_db_path(db_path)
    cached = await asyncio.to_thread(load_s2_paper_snapshot, db_path, arxiv_id, cache_ttl_days)
    if cached.status == "found":
        return S2PaperFetchResult(state="found", paper=cached.paper, complete=True, from_cache=True)
    if cached.status == "not_found":
        return S2PaperFetchResult(
            state="not_found",
            paper=None,
            complete=True,
            from_cache=True,
        )

    result, complete = await fetch_s2_paper(
        arxiv_id,
        client,
        api_key=api_key,
        include_status=True,
    )
    if not complete:
        return S2PaperFetchResult(
            state="unavailable",
            paper=None,
            complete=False,
            from_cache=False,
        )
    if result is not None:
        await _best_effort_cache_write(save_s2_paper, db_path, result)
        return S2PaperFetchResult(
            state="found",
            paper=result,
            complete=True,
            from_cache=False,
        )
    await _best_effort_cache_write(save_s2_paper_not_found, db_path, arxiv_id)
    return S2PaperFetchResult(
        state="not_found",
        paper=None,
        complete=True,
        from_cache=False,
    )


async def load_or_fetch_hf_daily_result(
    *,
    db_path: Path | str,
    cache_ttl_hours: int,
    client: httpx.AsyncClient,
) -> HFDailyFetchResult:
    """Load HF daily papers from cache, or fetch and persist on cache miss."""
    db_path = _normalize_db_path(db_path)
    cached: HFDailyCacheSnapshot = await asyncio.to_thread(
        load_hf_daily_cache_snapshot, db_path, cache_ttl_hours
    )
    if cached.status == "found":
        return HFDailyFetchResult(
            state="found",
            papers=list(cached.papers.values()),
            complete=True,
            from_cache=True,
        )
    if cached.status == "empty":
        return HFDailyFetchResult(state="empty", papers=[], complete=True, from_cache=True)

    papers, complete = await fetch_hf_daily_papers(client, include_status=True)
    if not complete:
        return HFDailyFetchResult(
            state="unavailable",
            papers=[],
            complete=False,
            from_cache=False,
        )
    await _best_effort_cache_write(save_hf_daily_cache, db_path, papers)
    if papers:
        return HFDailyFetchResult(
            state="found",
            papers=papers,
            complete=True,
            from_cache=False,
        )
    return HFDailyFetchResult(state="empty", papers=[], complete=True, from_cache=False)


async def load_or_fetch_s2_recommendations_result(
    *,
    arxiv_id: str,
    db_path: Path | str,
    cache_ttl_days: int,
    client: httpx.AsyncClient,
    api_key: str,
) -> S2RecommendationsFetchResult:
    """Load S2 recommendations from cache, or fetch and persist on cache miss."""
    db_path = _normalize_db_path(db_path)
    cached = await asyncio.to_thread(
        load_s2_recommendations_snapshot,
        db_path,
        arxiv_id,
        cache_ttl_days,
    )
    if cached.status == "found":
        return S2RecommendationsFetchResult(
            state="found",
            papers=cached.papers,
            complete=True,
            from_cache=True,
        )
    if cached.status == "empty":
        return S2RecommendationsFetchResult(
            state="empty",
            papers=[],
            complete=True,
            from_cache=True,
        )

    papers, complete = await fetch_s2_recommendations_with_status(
        arxiv_id,
        client,
        api_key=api_key,
        include_status=True,
    )
    if not complete:
        return S2RecommendationsFetchResult(
            state="unavailable",
            papers=[],
            complete=False,
            from_cache=False,
        )
    await _best_effort_cache_write(save_s2_recommendations, db_path, arxiv_id, papers)
    if papers:
        return S2RecommendationsFetchResult(
            state="found",
            papers=papers,
            complete=True,
            from_cache=False,
        )
    return S2RecommendationsFetchResult(
        state="empty",
        papers=[],
        complete=True,
        from_cache=False,
    )


@overload
async def load_or_fetch_s2_paper_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: Literal[False] = ...,
) -> SemanticScholarPaper | None:
    """Overload — returns the paper only when include_status is False."""
    ...


@overload
async def load_or_fetch_s2_paper_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: Literal[True],
) -> tuple[SemanticScholarPaper | None, bool]:
    """Overload — returns a (paper, complete) tuple when include_status is True."""
    ...


async def load_or_fetch_s2_paper_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: bool = False,
) -> SemanticScholarPaper | None | tuple[SemanticScholarPaper | None, bool]:
    """Load S2 paper data from cache, or fetch and persist on cache miss.

    Args:
        arxiv_id: Bare arXiv identifier (e.g. ``"2401.12345"``).
        db_path: Path to the Semantic Scholar SQLite cache database.
        cache_ttl_days: Maximum age in days for a cached result to be
            considered fresh.
        client: An active ``httpx.AsyncClient`` for remote fetching.  When
            ``None``, the remote fetch step is skipped and the function
            returns ``(None, True)`` (or ``None`` without status) — treating
            "no client" as a normal no-result, not an error.
        api_key: Semantic Scholar API key (may be empty string for
            unauthenticated access).
        include_status: When ``True``, return a ``(paper, complete)`` tuple
            instead of just the paper.

    Returns:
        When ``include_status=False``: the ``SemanticScholarPaper`` if found
        (cache or remote), or ``None`` otherwise.

        When ``include_status=True``: a ``(paper, complete)`` tuple where
        ``complete=False`` means the remote fetch path was tried but failed
        (e.g. network error, parse error).  ``complete=True`` means either a
        cache hit was returned, no client was provided, or a successful fetch
        was completed.
    """
    cached = await asyncio.to_thread(load_s2_paper_snapshot, db_path, arxiv_id, cache_ttl_days)
    if cached.status == "found":
        return (cached.paper, True) if include_status else cached.paper
    if cached.status == "not_found":
        return (None, True) if include_status else None
    if client is None:
        return (None, True) if include_status else None

    result = await load_or_fetch_s2_paper_result(
        arxiv_id=arxiv_id,
        db_path=db_path,
        cache_ttl_days=cache_ttl_days,
        client=client,
        api_key=api_key,
    )
    if include_status:
        return result.paper, result.complete
    return result.paper


@overload
async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
    include_status: Literal[False] = ...,
) -> list[HuggingFacePaper]:
    """Overload — returns the papers list only when include_status is False."""
    ...


@overload
async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
    include_status: Literal[True],
) -> tuple[list[HuggingFacePaper], bool]:
    """Overload — returns a (papers, complete) tuple when include_status is True."""
    ...


async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
    include_status: bool = False,
) -> list[HuggingFacePaper] | tuple[list[HuggingFacePaper], bool]:
    """Load HF daily papers from cache, or fetch and persist on cache miss.

    Args:
        db_path: Path to the HuggingFace daily-papers SQLite cache database.
        cache_ttl_hours: Maximum age in hours for a cached result to be
            considered fresh.
        client: An active ``httpx.AsyncClient`` for remote fetching.  When
            ``None``, the remote fetch step is skipped and the function
            returns ``([], True)`` (or ``[]`` without status) — treating
            "no client" as a normal empty result, not an error.
        include_status: When ``True``, return a ``(papers, complete)`` tuple
            instead of just the paper list.

    Returns:
        When ``include_status=False``: a list of ``HuggingFacePaper`` objects
        (may be empty if the cache is cold and no client was provided).

        When ``include_status=True``: a ``(papers, complete)`` tuple where
        ``complete=False`` means the remote fetch path was tried but failed.
        ``complete=True`` means either a cache hit was returned, no client was
        provided, or a successful fetch was completed.
    """
    cached = await asyncio.to_thread(load_hf_daily_cache_snapshot, db_path, cache_ttl_hours)
    if cached.status == "found":
        papers = list(cached.papers.values())
        return (papers, True) if include_status else papers
    if cached.status == "empty":
        empty: list[HuggingFacePaper] = []
        return (empty, True) if include_status else empty
    if client is None:
        empty = []
        return (empty, True) if include_status else empty

    result = await load_or_fetch_hf_daily_result(
        db_path=db_path,
        cache_ttl_hours=cache_ttl_hours,
        client=client,
    )
    if include_status:
        return result.papers, result.complete
    return result.papers


@overload
async def load_or_fetch_s2_recommendations_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: Literal[False] = ...,
) -> list[SemanticScholarPaper]:
    """Overload — returns recommended papers only when include_status is False."""
    ...


@overload
async def load_or_fetch_s2_recommendations_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: Literal[True],
) -> tuple[list[SemanticScholarPaper], bool]:
    """Overload — returns a (papers, complete) tuple when include_status is True."""
    ...


async def load_or_fetch_s2_recommendations_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: bool = False,
) -> list[SemanticScholarPaper] | tuple[list[SemanticScholarPaper], bool]:
    """Load S2 recommendations from cache, or fetch and persist on cache miss."""
    cached = await asyncio.to_thread(
        load_s2_recommendations_snapshot,
        db_path,
        arxiv_id,
        cache_ttl_days,
    )
    if cached.status == "found":
        return (cached.papers, True) if include_status else cached.papers
    if cached.status == "empty":
        empty: list[SemanticScholarPaper] = []
        return (empty, True) if include_status else empty
    if client is None:
        empty = []
        return (empty, True) if include_status else empty

    result = await load_or_fetch_s2_recommendations_result(
        arxiv_id=arxiv_id,
        db_path=db_path,
        cache_ttl_days=cache_ttl_days,
        client=client,
        api_key=api_key,
    )
    if include_status:
        return result.papers, result.complete
    return result.papers


__all__ = [
    "HFDailyFetchResult",
    "S2PaperFetchResult",
    "S2RecommendationsFetchResult",
    "load_or_fetch_hf_daily_cached",
    "load_or_fetch_hf_daily_result",
    "load_or_fetch_s2_paper_cached",
    "load_or_fetch_s2_paper_result",
    "load_or_fetch_s2_recommendations_cached",
    "load_or_fetch_s2_recommendations_result",
]
