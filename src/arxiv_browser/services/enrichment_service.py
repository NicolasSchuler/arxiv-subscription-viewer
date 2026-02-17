"""Internal enrichment services (Semantic Scholar + HuggingFace cache orchestration)."""

from __future__ import annotations

import asyncio
from pathlib import Path

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


async def load_or_fetch_s2_paper_cached(
    *,
    arxiv_id: str,
    db_path: Path,
    cache_ttl_days: int,
    client: httpx.AsyncClient | None,
    api_key: str,
) -> SemanticScholarPaper | None:
    """Load S2 paper data from cache, or fetch and persist on cache miss."""
    cached = await asyncio.to_thread(load_s2_paper, db_path, arxiv_id, cache_ttl_days)
    if cached is not None:
        return cached
    if client is None:
        return None

    result = await fetch_s2_paper(arxiv_id, client, api_key=api_key)
    if result is not None:
        await asyncio.to_thread(save_s2_paper, db_path, result)
    return result


async def load_or_fetch_hf_daily_cached(
    *,
    db_path: Path,
    cache_ttl_hours: int,
    client: httpx.AsyncClient | None,
) -> list[HuggingFacePaper]:
    """Load HF daily papers from cache, or fetch and persist on cache miss."""
    cached = await asyncio.to_thread(load_hf_daily_cache, db_path, cache_ttl_hours)
    if cached is not None:
        return list(cached.values())
    if client is None:
        return []

    papers = await fetch_hf_daily_papers(client)
    if papers:
        await asyncio.to_thread(save_hf_daily_cache, db_path, papers)
    return papers


__all__ = [
    "load_or_fetch_hf_daily_cached",
    "load_or_fetch_s2_paper_cached",
]
