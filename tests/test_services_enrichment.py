"""Tests for enrichment service helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arxiv_browser.huggingface import HFDailyCacheSnapshot, HuggingFacePaper
from arxiv_browser.semantic_scholar import (
    S2PaperCacheSnapshot,
    SemanticScholarPaper,
)
from arxiv_browser.services.enrichment_service import (
    load_or_fetch_hf_daily_cached,
    load_or_fetch_s2_paper_cached,
)


@pytest.mark.asyncio
async def test_load_or_fetch_s2_cache_hit(tmp_path) -> None:
    cached = SemanticScholarPaper(
        arxiv_id="2401.10001",
        s2_paper_id="s2:1",
        citation_count=1,
        influential_citation_count=0,
        tldr="",
        fields_of_study=(),
        year=2024,
        url="https://example.org",
    )

    with patch(
        "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
        return_value=S2PaperCacheSnapshot(status="found", paper=cached),
    ):
        result = await load_or_fetch_s2_paper_cached(
            arxiv_id="2401.10001",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=None,
            api_key="",
        )

    assert result is cached


@pytest.mark.asyncio
async def test_load_or_fetch_s2_fetch_and_save_on_cache_miss(tmp_path) -> None:
    fetched = SemanticScholarPaper(
        arxiv_id="2401.10002",
        s2_paper_id="s2:2",
        citation_count=2,
        influential_citation_count=1,
        tldr="",
        fields_of_study=(),
        year=2024,
        url="https://example.org",
    )

    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
            return_value=S2PaperCacheSnapshot(status="miss", paper=None),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_s2_paper",
            new=AsyncMock(return_value=(fetched, True)),
        ),
        patch("arxiv_browser.services.enrichment_service.save_s2_paper") as save_mock,
    ):
        result = await load_or_fetch_s2_paper_cached(
            arxiv_id="2401.10002",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=object(),
            api_key="k",
        )

    assert result is fetched
    save_mock.assert_called_once()


@pytest.mark.asyncio
async def test_load_or_fetch_s2_include_status_reports_incomplete_fetch(tmp_path) -> None:
    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
            return_value=S2PaperCacheSnapshot(status="miss", paper=None),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_s2_paper",
            new=AsyncMock(return_value=(None, False)),
        ),
        patch("arxiv_browser.services.enrichment_service.save_s2_paper") as save_mock,
    ):
        result = await load_or_fetch_s2_paper_cached(
            arxiv_id="2401.10003",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=object(),
            api_key="k",
            include_status=True,
        )

    assert result == (None, False)
    save_mock.assert_not_called()


@pytest.mark.asyncio
async def test_load_or_fetch_hf_cache_hit(tmp_path) -> None:
    cached_paper = HuggingFacePaper(
        arxiv_id="2401.20001",
        title="title",
        upvotes=1,
        num_comments=0,
        ai_summary="",
        ai_keywords=(),
        github_repo="",
        github_stars=0,
    )
    cached = {"2401.20001": cached_paper}

    with patch(
        "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
        return_value=HFDailyCacheSnapshot(status="found", papers=cached),
    ):
        result = await load_or_fetch_hf_daily_cached(
            db_path=tmp_path / "hf.db",
            cache_ttl_hours=6,
            client=None,
        )

    assert result == [cached_paper]


@pytest.mark.asyncio
async def test_load_or_fetch_hf_fetch_and_save_on_cache_miss(tmp_path) -> None:
    fetched = [
        HuggingFacePaper(
            arxiv_id="2401.20002",
            title="title",
            upvotes=2,
            num_comments=1,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
    ]

    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
            return_value=HFDailyCacheSnapshot(status="miss", papers={}),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_hf_daily_papers",
            new=AsyncMock(return_value=(fetched, True)),
        ),
        patch("arxiv_browser.services.enrichment_service.save_hf_daily_cache") as save_mock,
    ):
        result = await load_or_fetch_hf_daily_cached(
            db_path=tmp_path / "hf.db",
            cache_ttl_hours=6,
            client=object(),
        )

    assert result == fetched
    save_mock.assert_called_once()


@pytest.mark.asyncio
async def test_load_or_fetch_hf_include_status_reports_incomplete_fetch(tmp_path) -> None:
    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
            return_value=HFDailyCacheSnapshot(status="miss", papers={}),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_hf_daily_papers",
            new=AsyncMock(return_value=([], False)),
        ),
        patch("arxiv_browser.services.enrichment_service.save_hf_daily_cache") as save_mock,
    ):
        result = await load_or_fetch_hf_daily_cached(
            db_path=tmp_path / "hf.db",
            cache_ttl_hours=6,
            client=object(),
            include_status=True,
        )

    assert result == ([], False)
    save_mock.assert_not_called()
