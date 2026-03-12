"""Internal enrichment services (Semantic Scholar + HuggingFace cache orchestration)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, overload

import httpx

from arxiv_browser.huggingface import (
    HuggingFacePaper,
    fetch_hf_daily_papers,
    load_hf_daily_cache,
    save_hf_daily_cache,
)
from arxiv_browser.semantic_scholar import (
    SemanticScholarPaper,
    fetch_s2_paper,
    load_s2_paper,
    save_s2_paper,
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
) -> SemanticScholarPaper | None: ...


@overload
async def load_or_fetch_s2_paper_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
    include_status: Literal[True],
) -> tuple[SemanticScholarPaper | None, bool]: ...


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

    When include_status=True, returns (paper, complete) where complete=False
    means the remote fetch/parse path failed.
    """
    cached = await asyncio.to_thread(load_s2_paper, db_path, arxiv_id, cache_ttl_days)
    if cached is not None:
        return (cached, True) if include_status else cached
    if client is None:
        return (None, True) if include_status else None

    if include_status:
        result, complete = await fetch_s2_paper(
            arxiv_id,
            client,
            api_key=api_key,
            include_status=True,
        )
    else:
        result = await fetch_s2_paper(arxiv_id, client, api_key=api_key)
        complete = True
    if result is not None:
        await asyncio.to_thread(save_s2_paper, db_path, result)
    if include_status:
        return result, complete
    return result


@overload
async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
    include_status: Literal[False] = ...,
) -> list[HuggingFacePaper]: ...


@overload
async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
    include_status: Literal[True],
) -> tuple[list[HuggingFacePaper], bool]: ...


async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
    include_status: bool = False,
) -> list[HuggingFacePaper] | tuple[list[HuggingFacePaper], bool]:
    """Load HF daily papers from cache, or fetch and persist on cache miss.

    When include_status=True, returns (papers, complete) where complete=False
    means the remote fetch/parse path failed.
    """
    cached = await asyncio.to_thread(load_hf_daily_cache, db_path, cache_ttl_hours)
    if cached is not None:
        papers = list(cached.values())
        return (papers, True) if include_status else papers
    if client is None:
        empty: list[HuggingFacePaper] = []
        return (empty, True) if include_status else empty

    if include_status:
        papers, complete = await fetch_hf_daily_papers(client, include_status=True)
    else:
        papers = await fetch_hf_daily_papers(client)
        complete = True
    if papers:
        await asyncio.to_thread(save_hf_daily_cache, db_path, papers)
    if include_status:
        return papers, complete
    return papers


__all__ = [
    "load_or_fetch_hf_daily_cached",
    "load_or_fetch_s2_paper_cached",
]
