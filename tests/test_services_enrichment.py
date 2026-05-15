"""Tests for enrichment service helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from arxiv_browser.conference_deadlines import (
    ConferenceDeadline,
    ConferenceDeadlineCacheSnapshot,
)
from arxiv_browser.huggingface import HFDailyCacheSnapshot, HuggingFacePaper
from arxiv_browser.semantic_scholar import (
    S2PaperCacheSnapshot,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)
from arxiv_browser.services import enrichment_service as enrich
from arxiv_browser.services.enrichment_service import (
    load_or_fetch_hf_daily_cached,
    load_or_fetch_s2_paper_cached,
    load_or_fetch_s2_paper_result,
)


def _s2_paper(arxiv_id: str = "2401.10001") -> SemanticScholarPaper:
    return SemanticScholarPaper(
        arxiv_id=arxiv_id,
        s2_paper_id=f"s2:{arxiv_id}",
        citation_count=1,
        influential_citation_count=0,
        tldr="",
        fields_of_study=(),
        year=2024,
        url="https://example.org",
    )


def _deadline(conference_id: str = "iclr2027") -> ConferenceDeadline:
    return ConferenceDeadline(
        conference_id=conference_id,
        title="ICLR",
        year=2027,
        subjects=("ML",),
        deadline_at=datetime(2026, 9, 1, 23, 59, tzinfo=UTC),
        timezone_name="UTC",
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


@pytest.mark.asyncio
async def test_enrichment_remote_success_survives_cache_write_failure(tmp_path, caplog) -> None:
    fetched = SemanticScholarPaper(
        arxiv_id="2401.77777",
        s2_paper_id="s2:77777",
        citation_count=3,
        influential_citation_count=1,
        tldr="",
        fields_of_study=(),
        year=2026,
        url="https://example.org/s2",
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
        patch(
            "arxiv_browser.services.enrichment_service.save_s2_paper",
            side_effect=OSError("readonly"),
        ),
    ):
        caplog.set_level("WARNING", logger="arxiv_browser.services.enrichment_service")
        result = await load_or_fetch_s2_paper_result(
            arxiv_id="2401.77777",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=object(),
            api_key="",
        )

    assert result.state == "found"
    assert result.paper is fetched
    assert result.complete is True
    assert result.from_cache is False
    assert "Failed to persist enrichment cache" in caplog.text


@pytest.mark.asyncio
async def test_load_or_fetch_s2_recommendations_cache_hit_skips_remote(tmp_path) -> None:
    cached = [_s2_paper("2401.30001")]

    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=S2RecommendationsCacheSnapshot(status="found", papers=cached),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
            new=AsyncMock(),
        ) as fetch_mock,
    ):
        result = await enrich.load_or_fetch_s2_recommendations_result(
            arxiv_id="2401.30001",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=object(),
            api_key="",
        )

    assert result.state == "found"
    assert result.papers == cached
    assert result.complete is True
    assert result.from_cache is True
    fetch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_or_fetch_s2_recommendations_empty_fetch_is_cached(tmp_path) -> None:
    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=S2RecommendationsCacheSnapshot(status="miss", papers=[]),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
            new=AsyncMock(return_value=([], True)),
        ),
        patch("arxiv_browser.services.enrichment_service.save_s2_recommendations") as save_mock,
    ):
        result = await enrich.load_or_fetch_s2_recommendations_result(
            arxiv_id="2401.30002",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=object(),
            api_key="",
        )

    assert result.state == "empty"
    assert result.papers == []
    assert result.complete is True
    assert result.from_cache is False
    save_mock.assert_called_once()


@pytest.mark.asyncio
async def test_load_or_fetch_s2_recommendations_incomplete_fetch_not_cached(tmp_path) -> None:
    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=S2RecommendationsCacheSnapshot(status="miss", papers=[]),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
            new=AsyncMock(return_value=([], False)),
        ),
        patch("arxiv_browser.services.enrichment_service.save_s2_recommendations") as save_mock,
    ):
        result = await enrich.load_or_fetch_s2_recommendations_result(
            arxiv_id="2401.30003",
            db_path=tmp_path / "s2.db",
            cache_ttl_days=7,
            client=object(),
            api_key="",
        )

    assert result.state == "unavailable"
    assert result.papers == []
    assert result.complete is False
    assert result.from_cache is False
    save_mock.assert_not_called()


@pytest.mark.asyncio
async def test_load_or_fetch_conference_deadlines_cache_hit_and_empty_skip_remote(
    tmp_path,
) -> None:
    deadline = _deadline()

    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_conference_deadlines_cache_snapshot",
            side_effect=[
                ConferenceDeadlineCacheSnapshot(status="found", deadlines=[deadline]),
                ConferenceDeadlineCacheSnapshot(status="empty", deadlines=[]),
            ],
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_conference_deadlines",
            new=AsyncMock(),
        ) as fetch_mock,
    ):
        found = await enrich.load_or_fetch_conference_deadlines_result(
            db_path=tmp_path / "deadlines.db",
            cache_ttl_hours=24,
            client=object(),
            source_url="https://example.test/deadlines.yml",
        )
        empty = await enrich.load_or_fetch_conference_deadlines_result(
            db_path=tmp_path / "deadlines.db",
            cache_ttl_hours=24,
            client=object(),
            source_url="https://example.test/deadlines.yml",
        )

    assert found.state == "found"
    assert found.deadlines == [deadline]
    assert found.from_cache is True
    assert empty.state == "empty"
    assert empty.deadlines == []
    assert empty.from_cache is True
    fetch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_or_fetch_conference_deadlines_survives_cache_write_failure(
    tmp_path,
    caplog,
) -> None:
    deadline = _deadline()

    with (
        patch(
            "arxiv_browser.services.enrichment_service.load_conference_deadlines_cache_snapshot",
            return_value=ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[]),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.fetch_conference_deadlines",
            new=AsyncMock(return_value=([deadline], True)),
        ),
        patch(
            "arxiv_browser.services.enrichment_service.save_conference_deadlines_cache",
            side_effect=OSError("readonly"),
        ),
    ):
        caplog.set_level("WARNING", logger="arxiv_browser.services.enrichment_service")
        result = await enrich.load_or_fetch_conference_deadlines_result(
            db_path=tmp_path / "deadlines.db",
            cache_ttl_hours=24,
            client=object(),
            source_url="https://example.test/deadlines.yml",
        )

    assert result.state == "found"
    assert result.deadlines == [deadline]
    assert result.complete is True
    assert result.from_cache is False
    assert "Failed to persist enrichment cache" in caplog.text
