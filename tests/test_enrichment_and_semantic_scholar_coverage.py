"""Additional branch coverage for smaller modules."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import deque
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import arxiv_browser.app as app_mod
import arxiv_browser.cli as cli
import arxiv_browser.llm_providers as llm_providers
import arxiv_browser.semantic_scholar as s2
from arxiv_browser.actions import external_io_actions as io_actions
from arxiv_browser.actions import llm_actions as llm_actions
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionsModal,
    CollectionViewModal,
)
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, UserConfig
from arxiv_browser.services import enrichment_service as enrich
from tests.support.app_stubs import (
    _DummyInput,
    _DummyLabel,
    _DummyListView,
    _DummyTimer,
    _make_app_config,
    _new_app_stub,
    _paper,
)


class TestEnrichmentServiceCoverage:
    @pytest.mark.asyncio
    async def test_best_effort_cache_write_and_result_states(
        self, tmp_path, make_paper, caplog
    ) -> None:
        await enrich._best_effort_cache_write(lambda *_args: None)
        with patch(
            "arxiv_browser.services.enrichment_service.asyncio.to_thread",
            side_effect=OSError("boom"),
        ):
            await enrich._best_effort_cache_write(lambda *_args: None)

        s2_paper = s2.SemanticScholarPaper(
            arxiv_id="2401.1",
            s2_paper_id="s2:1",
            citation_count=1,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        hf_paper = enrich.HuggingFacePaper(
            arxiv_id="2401.2",
            title="hf",
            upvotes=1,
            num_comments=0,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )

        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
            return_value=s2.S2PaperCacheSnapshot(status="found", paper=s2_paper),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.paper is s2_paper
        assert result.from_cache is True

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="not_found", paper=None),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.state == "not_found"

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_paper",
                new=AsyncMock(return_value=(None, False)),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.state == "unavailable"

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                return_value=enrich.HFDailyCacheSnapshot(
                    status="found", papers={"2401.2": hf_paper}
                ),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_hf_daily_papers",
                new=AsyncMock(return_value=([hf_paper], True)),
            ),
        ):
            result = await enrich.load_or_fetch_hf_daily_result(
                db_path=tmp_path / "hf.db",
                cache_ttl_hours=6,
                client=object(),
            )
        assert result.papers == [hf_paper]
        assert result.from_cache is True

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=s2.S2RecommendationsCacheSnapshot(status="empty", papers=[]),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
                new=AsyncMock(return_value=([s2_paper], True)),
            ),
        ):
            result = await enrich.load_or_fetch_s2_recommendations_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=object(),
                api_key="",
            )
        assert result.from_cache is True

    @pytest.mark.asyncio
    async def test_cached_wrappers_cover_no_client_and_status_paths(
        self, tmp_path, make_paper
    ) -> None:
        paper = s2.SemanticScholarPaper(
            arxiv_id="2401.10",
            s2_paper_id="s2:10",
            citation_count=5,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
            return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
        ):
            assert (
                await enrich.load_or_fetch_s2_paper_cached(
                    arxiv_id="2401.10",
                    db_path=tmp_path / "s2.db",
                    cache_ttl_days=7,
                    client=None,
                    api_key="",
                )
                is None
            )

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.load_or_fetch_s2_paper_result",
                new=AsyncMock(
                    return_value=enrich.S2PaperFetchResult(
                        state="found",
                        paper=paper,
                        complete=True,
                        from_cache=False,
                    )
                ),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_cached(
                arxiv_id="2401.10",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
                include_status=True,
            )
        assert result == (paper, True)

        with patch(
            "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
            return_value=enrich.HFDailyCacheSnapshot(status="miss", papers={}),
        ):
            assert (
                await enrich.load_or_fetch_hf_daily_cached(
                    db_path=tmp_path / "hf.db",
                    cache_ttl_hours=6,
                    client=None,
                )
                == []
            )

        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=s2.S2RecommendationsCacheSnapshot(status="miss", papers=[]),
        ):
            assert (
                await enrich.load_or_fetch_s2_recommendations_cached(
                    arxiv_id="2401.10",
                    db_path=tmp_path / "recs.db",
                    cache_ttl_days=3,
                    client=None,
                    api_key="",
                )
                == []
            )


class TestSemanticScholarCoverage:
    def test_citation_graph_cache_round_trip_and_helpers(self, tmp_path, make_paper) -> None:
        db_path = tmp_path / "s2.db"
        entry = s2.CitationEntry(
            s2_paper_id="paper-1",
            arxiv_id="2401.99999",
            title="Cited",
            authors="A. Author",
            year=2024,
            citation_count=9,
            url="https://arxiv.org/abs/2401.99999",
        )
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1") is False
        s2.save_s2_citation_graph(db_path, "paper-1", "references", [entry])
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1") is True
        assert s2.load_s2_citation_graph(db_path, "paper-1", "references") == [entry]
        assert s2.load_s2_citation_graph(db_path, "paper-1", "citations") == []
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1", ttl_days=0) is False

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_citation_graph (paper_id, direction, rank, payload_json, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("paper-2", "references", 0, '{"bad": "json"}', datetime.now(UTC).isoformat()),
            )
        assert s2.load_s2_citation_graph(db_path, "paper-2", "references") == []

    def test_s2_snapshot_cache_states(self, tmp_path, make_paper) -> None:
        db_path = tmp_path / "s2.db"
        paper = s2.SemanticScholarPaper(
            arxiv_id="2401.1",
            s2_paper_id="s2:1",
            citation_count=1,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )

        assert s2.load_s2_paper_snapshot(db_path, "2401.1").status == "miss"
        s2.save_s2_paper_not_found(db_path, "2401.1")
        assert s2.load_s2_paper_snapshot(db_path, "2401.1").status == "not_found"
        s2.save_s2_paper(db_path, paper)
        assert s2.load_s2_paper_snapshot(db_path, "2401.1").status == "found"

        rec = s2.SemanticScholarPaper(
            arxiv_id="2401.2",
            s2_paper_id="s2:2",
            citation_count=2,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        s2.save_s2_recommendations(db_path, "paper-x", [rec])
        assert s2.load_s2_recommendations_snapshot(db_path, "paper-x").status == "found"
        assert s2.load_s2_recommendations_snapshot(db_path, "missing").status == "miss"

    @pytest.mark.asyncio
    async def test_fetch_s2_citations_and_references_edge_cases(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await s2.fetch_s2_citations("paper", client, limit=0) == []

        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"data": []}
        client.get.return_value = response
        assert await s2.fetch_s2_references("paper", client, include_status=True) == ([], True)

        response.json.return_value = {"recommendedPapers": "oops"}
        assert await s2.fetch_s2_recommendations("paper", client) == []
