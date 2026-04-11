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

    @staticmethod
    async def _run_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    @staticmethod
    def _sample_s2_paper() -> s2.SemanticScholarPaper:
        return s2.SemanticScholarPaper(
            arxiv_id="2401.20",
            s2_paper_id="s2:20",
            citation_count=2,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com/s2",
        )

    @staticmethod
    def _sample_hf_paper() -> enrich.HuggingFacePaper:
        return enrich.HuggingFacePaper(
            arxiv_id="2401.21",
            title="hf",
            upvotes=3,
            num_comments=1,
            ai_summary="summary",
            ai_keywords=("ai",),
            github_repo="",
            github_stars=0,
        )

    @staticmethod
    def _sample_rec_paper() -> s2.SemanticScholarPaper:
        return s2.SemanticScholarPaper(
            arxiv_id="2401.22",
            s2_paper_id="s2:22",
            citation_count=4,
            influential_citation_count=2,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com/recs",
        )

    @pytest.mark.asyncio
    async def test_s2_paper_result_matrix_cache_and_remote_paths(self, tmp_path) -> None:
        s2_paper = self._sample_s2_paper()
        with patch(
            "arxiv_browser.services.enrichment_service.asyncio.to_thread",
            new=AsyncMock(side_effect=self._run_to_thread),
        ):
            with patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="found", paper=s2_paper),
            ):
                result = await enrich.load_or_fetch_s2_paper_result(
                    arxiv_id="2401.20",
                    db_path=tmp_path / "s2.db",
                    cache_ttl_days=7,
                    client=object(),
                    api_key="",
                )
            assert result.state == "found" and result.from_cache is True

            with patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="not_found", paper=None),
            ):
                result = await enrich.load_or_fetch_s2_paper_result(
                    arxiv_id="2401.20",
                    db_path=tmp_path / "s2.db",
                    cache_ttl_days=7,
                    client=object(),
                    api_key="",
                )
            assert result.state == "not_found" and result.from_cache is True

            with (
                patch(
                    "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                    return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
                ),
                patch(
                    "arxiv_browser.services.enrichment_service.fetch_s2_paper",
                    new=AsyncMock(return_value=(s2_paper, True)),
                ),
            ):
                result = await enrich.load_or_fetch_s2_paper_result(
                    arxiv_id="2401.20",
                    db_path=tmp_path / "s2.db",
                    cache_ttl_days=7,
                    client=object(),
                    api_key="",
                )
            assert result.state == "found" and result.from_cache is False

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
                    arxiv_id="2401.20",
                    db_path=tmp_path / "s2.db",
                    cache_ttl_days=7,
                    client=object(),
                    api_key="",
                )
            assert result.state == "unavailable"

    @pytest.mark.asyncio
    async def test_hf_daily_result_matrix_cache_and_remote_paths(self, tmp_path) -> None:
        hf_paper = self._sample_hf_paper()
        with patch(
            "arxiv_browser.services.enrichment_service.asyncio.to_thread",
            new=AsyncMock(side_effect=self._run_to_thread),
        ):
            with patch(
                "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                return_value=enrich.HFDailyCacheSnapshot(
                    status="found", papers={hf_paper.arxiv_id: hf_paper}
                ),
            ):
                result = await enrich.load_or_fetch_hf_daily_result(
                    db_path=tmp_path / "hf.db",
                    cache_ttl_hours=6,
                    client=object(),
                )
            assert result.state == "found" and result.from_cache is True

            with patch(
                "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                return_value=enrich.HFDailyCacheSnapshot(status="empty", papers={}),
            ):
                result = await enrich.load_or_fetch_hf_daily_result(
                    db_path=tmp_path / "hf.db",
                    cache_ttl_hours=6,
                    client=object(),
                )
            assert result.state == "empty" and result.from_cache is True

            with (
                patch(
                    "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                    return_value=enrich.HFDailyCacheSnapshot(status="miss", papers={}),
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
            assert result.state == "found" and result.from_cache is False

    @pytest.mark.asyncio
    async def test_s2_recommendations_result_matrix_cache_and_remote_paths(self, tmp_path) -> None:
        rec_paper = self._sample_rec_paper()
        with patch(
            "arxiv_browser.services.enrichment_service.asyncio.to_thread",
            new=AsyncMock(side_effect=self._run_to_thread),
        ):
            with patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=s2.S2RecommendationsCacheSnapshot(status="found", papers=[rec_paper]),
            ):
                result = await enrich.load_or_fetch_s2_recommendations_result(
                    arxiv_id="2401.22",
                    db_path=tmp_path / "recs.db",
                    cache_ttl_days=3,
                    client=object(),
                    api_key="",
                )
            assert result.state == "found" and result.from_cache is True

            with patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=s2.S2RecommendationsCacheSnapshot(status="empty", papers=[]),
            ):
                result = await enrich.load_or_fetch_s2_recommendations_result(
                    arxiv_id="2401.22",
                    db_path=tmp_path / "recs.db",
                    cache_ttl_days=3,
                    client=object(),
                    api_key="",
                )
            assert result.state == "empty" and result.from_cache is True

            with (
                patch(
                    "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                    return_value=s2.S2RecommendationsCacheSnapshot(status="miss", papers=[]),
                ),
                patch(
                    "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
                    new=AsyncMock(return_value=([rec_paper], True)),
                ),
            ):
                result = await enrich.load_or_fetch_s2_recommendations_result(
                    arxiv_id="2401.22",
                    db_path=tmp_path / "recs.db",
                    cache_ttl_days=3,
                    client=object(),
                    api_key="",
                )
            assert result.state == "found" and result.from_cache is False

    @pytest.mark.asyncio
    async def test_s2_recommendations_cached_miss_without_client_returns_empty(
        self, tmp_path
    ) -> None:
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

    @pytest.mark.asyncio
    async def test_remote_not_found_and_empty_write_paths(self, tmp_path) -> None:
        _s2_paper = s2.SemanticScholarPaper(
            arxiv_id="2401.30",
            s2_paper_id="s2:30",
            citation_count=1,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com/s2",
        )
        _hf_paper = enrich.HuggingFacePaper(
            arxiv_id="2401.31",
            title="hf",
            upvotes=1,
            num_comments=0,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
        _rec_paper = s2.SemanticScholarPaper(
            arxiv_id="2401.32",
            s2_paper_id="s2:32",
            citation_count=2,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com/recs",
        )

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_paper",
                new=AsyncMock(return_value=(None, True)),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.30",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.state == "not_found"
        assert result.from_cache is False

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                return_value=enrich.HFDailyCacheSnapshot(status="miss", papers={}),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_hf_daily_papers",
                new=AsyncMock(return_value=([], True)),
            ),
        ):
            result = await enrich.load_or_fetch_hf_daily_result(
                db_path=tmp_path / "hf.db",
                cache_ttl_hours=6,
                client=object(),
            )
        assert result.state == "empty"
        assert result.from_cache is False

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=s2.S2RecommendationsCacheSnapshot(status="miss", papers=[]),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
                new=AsyncMock(return_value=([], True)),
            ),
        ):
            result = await enrich.load_or_fetch_s2_recommendations_result(
                arxiv_id="2401.32",
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=object(),
                api_key="",
            )
        assert result.state == "empty"
        assert result.from_cache is False

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                return_value=enrich.HFDailyCacheSnapshot(status="miss", papers={}),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=s2.S2RecommendationsCacheSnapshot(status="miss", papers=[]),
            ),
        ):
            assert await enrich.load_or_fetch_s2_paper_cached(
                arxiv_id="2401.30",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=None,
                api_key="",
                include_status=True,
            ) == (None, True)
            assert await enrich.load_or_fetch_hf_daily_cached(
                db_path=tmp_path / "hf.db",
                cache_ttl_hours=6,
                client=None,
                include_status=True,
            ) == ([], True)
            assert await enrich.load_or_fetch_s2_recommendations_cached(
                arxiv_id="2401.32",
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=None,
                api_key="",
                include_status=True,
            ) == ([], True)


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

        with patch("arxiv_browser.semantic_scholar._s2_get_with_retry", return_value=None):
            assert await s2.fetch_s2_recommendations("paper", client) == []
            assert await s2.fetch_s2_citations("paper", client) == []

    @pytest.mark.asyncio
    async def test_malformed_snapshot_and_fetch_error_branches(self, tmp_path) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        request = httpx.Request("GET", "https://example.com")
        db_path = tmp_path / "s2.db"
        s2.init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_paper_fetch_state (arxiv_id, status, fetched_at) "
                "VALUES (?, ?, ?)",
                ("paper", "weird", now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_recommendation_fetch_state "
                "(source_arxiv_id, status, fetched_at) VALUES (?, ?, ?)",
                ("paper", "weird", now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_papers (arxiv_id, payload_json, fetched_at) "
                "VALUES (?, ?, ?)",
                ("paper", '{"bad": true}', now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_recommendations "
                "(source_arxiv_id, rank, payload_json, fetched_at) VALUES (?, ?, ?, ?)",
                ("paper", 0, '{"bad": true}', now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_citation_graph_fetches (paper_id, fetched_at) "
                "VALUES (?, ?)",
                ("paper", now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_citation_graph "
                "(paper_id, direction, rank, payload_json, fetched_at) VALUES (?, ?, ?, ?, ?)",
                ("paper", "references", 0, '{"bad": true}', now),
            )

        assert s2.load_s2_paper_snapshot(db_path, "paper").status == "miss"
        assert s2.load_s2_recommendations_snapshot(db_path, "paper").status == "miss"
        assert s2.load_s2_citation_graph(db_path, "paper", "references") == []
        assert s2.has_s2_citation_graph_cache(db_path, "paper") is True

        ref_payload = {
            "data": [
                {
                    "citedPaper": {
                        "paperId": "s2:1",
                        "externalIds": {"ArXiv": "2401.00001"},
                        "title": "Cited",
                        "authors": [{"name": "Alice"}],
                        "year": 2024,
                        "citationCount": 3,
                        "url": "https://example.com/cited",
                    }
                },
                "skip",
            ]
        }
        cite_payload = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "s2:2",
                        "externalIds": {"ArXiv": "2401.00002"},
                        "title": "Citing",
                        "authors": [{"name": "Bob"}],
                        "year": 2024,
                        "citationCount": 5,
                        "url": "https://example.com/citing",
                    }
                },
                "skip",
            ]
        }
        with patch("arxiv_browser.semantic_scholar._s2_get_with_retry", return_value=None):
            assert await s2.fetch_s2_references("paper", client, include_status=True) == ([], False)
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry",
            return_value=MagicMock(
                json=MagicMock(return_value={"data": "oops"}), raise_for_status=MagicMock()
            ),
        ):
            assert await s2.fetch_s2_citations("paper", client, include_status=True) == ([], False)
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry",
            return_value=MagicMock(
                json=MagicMock(return_value=ref_payload),
                raise_for_status=MagicMock(),
                status_code=200,
                request=request,
            ),
        ):
            refs, complete = await s2.fetch_s2_references("paper", client, include_status=True)
        assert complete is True and len(refs) == 1
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry",
            side_effect=[
                MagicMock(json=MagicMock(return_value=cite_payload), raise_for_status=MagicMock()),
                MagicMock(json=MagicMock(return_value={}), raise_for_status=MagicMock()),
            ],
        ):
            cites, complete = await s2.fetch_s2_citations("paper", client, include_status=True)
        assert complete is True and len(cites) == 1
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry_status",
            return_value=(None, True),
        ):
            assert await s2.fetch_s2_recommendations_with_status(
                "paper", client, include_status=True
            ) == (
                [],
                True,
            )
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry_status",
            return_value=(
                MagicMock(
                    json=MagicMock(return_value={"recommendedPapers": "oops"}),
                    raise_for_status=MagicMock(),
                    request=request,
                ),
                True,
            ),
        ):
            assert await s2.fetch_s2_recommendations_with_status(
                "paper",
                client,
                include_status=True,
            ) == ([], False)

        with patch(
            "arxiv_browser.semantic_scholar.sqlite3.connect", side_effect=sqlite3.Error("boom")
        ):
            assert s2.load_s2_paper_snapshot(db_path, "paper").status == "miss"
            assert s2.load_s2_recommendations_snapshot(db_path, "paper").status == "miss"
            assert s2.load_s2_citation_graph(db_path, "paper", "references") == []
            assert s2.has_s2_citation_graph_cache(db_path, "paper") is False
            s2.save_s2_paper(
                db_path,
                s2.SemanticScholarPaper(
                    arxiv_id="2401.9",
                    s2_paper_id="s2:9",
                    citation_count=1,
                    influential_citation_count=0,
                    tldr="",
                    fields_of_study=(),
                    year=2024,
                    url="https://example.com",
                ),
            )
            s2.save_s2_paper_not_found(db_path, "paper")
            s2.save_s2_recommendations(db_path, "paper", [])
            s2.save_s2_citation_graph(db_path, "paper", "references", [])

    def test_malformed_fetch_state_rows_are_rejected(self, tmp_path) -> None:
        db_path = tmp_path / "s2.db"
        s2.init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_paper_fetch_state (arxiv_id, status, fetched_at) "
                "VALUES (?, ?, ?)",
                ("paper", 1, now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_recommendation_fetch_state "
                "(source_arxiv_id, status, fetched_at) VALUES (?, ?, ?)",
                ("paper", 1, now),
            )
            assert s2._load_s2_paper_fetch_state(conn, "paper", 7) == (None, None)
            assert s2._load_s2_recommendation_fetch_state(conn, "paper", 7) == (None, None)

    @pytest.mark.asyncio
    async def test_retry_parse_and_fetch_branch_matrix(self) -> None:
        async def _retry_passthrough(fn, **_kwargs):
            return await fn()

        request = httpx.Request("GET", "https://example.com")
        client = AsyncMock(spec=httpx.AsyncClient)
        headers = s2._build_s2_headers(s2.S2Request(url="https://example.com", params={}))
        assert headers == {}
        assert s2._build_s2_headers(
            s2.S2Request(url="https://example.com", params={}, api_key="k")
        ) == {"x-api-key": "k"}

        paper_payload = {
            "paperId": "s2:1",
            "externalIds": {"ArXiv": "2401.00001"},
            "tldr": {"text": "summary"},
            "fieldsOfStudy": ["cs.AI", 1],
            "citationCount": 3,
            "influentialCitationCount": 1,
            "year": 2024,
            "url": "https://example.com/paper",
            "title": "Paper",
            "abstract": "Abstract",
        }
        citation_payload = {
            "paperId": "s2:2",
            "externalIds": {"ArXiv": "2401.00002"},
            "title": "Cited",
            "authors": [{"name": "Alice"}, {"name": "Bob"}, {"foo": "skip"}],
            "year": 2024,
            "citationCount": 5,
            "url": "https://example.com/cited",
        }
        assert s2.parse_s2_paper_response({}) is None
        parsed_paper = s2.parse_s2_paper_response(paper_payload)
        assert parsed_paper is not None
        assert parsed_paper.arxiv_id == "2401.00001"
        assert parsed_paper.fields_of_study == ("cs.AI",)
        assert s2.parse_citation_entry({}) is None
        parsed_citation = s2.parse_citation_entry(citation_payload)
        assert parsed_citation is not None
        assert parsed_citation.arxiv_id == "2401.00002"
        assert parsed_citation.authors == "Alice, Bob"

        with patch(
            "arxiv_browser.semantic_scholar.retry_with_backoff",
            new=AsyncMock(side_effect=_retry_passthrough),
        ):
            client.get = AsyncMock(return_value=httpx.Response(404, request=request))
            assert await s2._s2_get_with_retry_status(
                client,
                s2.S2Request(url="https://example.com", params={}, label="paper"),
            ) == (None, True)
            client.get = AsyncMock(return_value=httpx.Response(503, request=request))
            assert await s2._s2_get_with_retry_status(
                client,
                s2.S2Request(url="https://example.com", params={}, label="paper"),
            ) == (None, False)
            client.get = AsyncMock(side_effect=httpx.ConnectError("boom", request=request))
            assert await s2._s2_get_with_retry_status(
                client,
                s2.S2Request(url="https://example.com", params={}, label="paper"),
            ) == (None, False)
            client.get = AsyncMock(side_effect=httpx.HTTPError("boom"))
            assert await s2._s2_get_with_retry_status(
                client,
                s2.S2Request(url="https://example.com", params={}, label="paper"),
            ) == (None, False)

        invalid_json = httpx.Response(200, content=b"{bad", request=request)
        assert s2._parse_json_object(invalid_json, "label") is None
        list_json = httpx.Response(200, json=[1, 2], request=request)
        assert s2._parse_json_object(list_json, "label") is None
        object_json = httpx.Response(200, json={"ok": True}, request=request)
        assert s2._parse_json_object(object_json, "label") == {"ok": True}

        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry_status", return_value=(None, True)
        ):
            assert await s2.fetch_s2_paper("2401.1", client, include_status=True) == (None, True)

        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry_status",
            return_value=(httpx.Response(200, json=paper_payload, request=request), True),
        ):
            paper, complete = await s2.fetch_s2_paper("2401.1", client, include_status=True)
        assert complete is True
        assert paper is not None and paper.arxiv_id == "2401.1"

        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry_status",
            return_value=(None, False),
        ):
            assert await s2.fetch_s2_recommendations_with_status(
                "2401.1", client, include_status=True
            ) == (
                [],
                False,
            )

        recs_payload = {"recommendedPapers": [paper_payload, "skip"]}
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry_status",
            return_value=(httpx.Response(200, json=recs_payload, request=request), True),
        ):
            recs = await s2.fetch_s2_recommendations("2401.1", client, limit=5)
        assert len(recs) == 1
        assert recs[0].arxiv_id == "2401.00001"

        refs_payload = {"data": [{"citedPaper": citation_payload}, {"citedPaper": {}}, "skip"]}
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry",
            return_value=httpx.Response(200, json=refs_payload, request=request),
        ):
            refs, complete = await s2.fetch_s2_references("paper-1", client, include_status=True)
        assert complete is True
        assert len(refs) == 1
        assert refs[0].arxiv_id == "2401.00002"

        citation_page = {"data": [{"citingPaper": citation_payload} for _ in range(100)]}
        with patch(
            "arxiv_browser.semantic_scholar._s2_get_with_retry",
            side_effect=[
                httpx.Response(200, json=citation_page, request=request),
                httpx.Response(200, json={"data": []}, request=request),
            ],
        ):
            cites, complete = await s2.fetch_s2_citations(
                "paper-1", client, limit=120, include_status=True
            )
        assert complete is True
        assert len(cites) == 100

        assert await s2.fetch_s2_citations("paper-1", client, limit=0, include_status=True) == (
            [],
            True,
        )

    def test_cache_freshness_and_snapshot_branches(self, tmp_path) -> None:
        db_path = tmp_path / "semantic_scholar.db"
        paper = s2.SemanticScholarPaper(
            arxiv_id="2401.00001",
            s2_paper_id="s2:1",
            citation_count=1,
            influential_citation_count=0,
            tldr="summary",
            fields_of_study=("cs.AI",),
            year=2024,
            url="https://example.com",
        )
        rec = s2.SemanticScholarPaper(
            arxiv_id="2401.00002",
            s2_paper_id="s2:2",
            citation_count=2,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        entry = s2.CitationEntry(
            s2_paper_id="s2:3",
            arxiv_id="2401.00003",
            title="Cited",
            authors="Alice",
            year=2024,
            citation_count=5,
            url="https://example.com/cited",
        )

        assert s2._is_fresh("not-a-date", 7) is False
        assert s2._is_fresh("2000-01-01T00:00:00", 1_000_000) is True

        s2.init_s2_db(db_path)
        assert s2.load_s2_paper_snapshot(db_path, "missing").status == "miss"
        s2.save_s2_paper_not_found(db_path, paper.arxiv_id)
        assert s2.load_s2_paper_snapshot(db_path, paper.arxiv_id).status == "not_found"
        s2.save_s2_paper(db_path, paper)
        assert s2.load_s2_paper_snapshot(db_path, paper.arxiv_id).status == "found"

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "UPDATE s2_papers SET payload_json = ?, fetched_at = ? WHERE arxiv_id = ?",
                ('{"bad": true}', "2026-01-01T00:00:00+00:00", paper.arxiv_id),
            )
        assert s2.load_s2_paper_snapshot(db_path, paper.arxiv_id).status == "miss"

        s2.save_s2_recommendations(db_path, "src", [rec])
        assert s2.load_s2_recommendations_snapshot(db_path, "src").status == "found"
        assert s2.load_s2_recommendations_snapshot(db_path, "missing").status == "miss"
        s2.save_s2_recommendations(db_path, "empty-src", [])
        assert s2.load_s2_recommendations_snapshot(db_path, "empty-src").status == "empty"

        assert s2.has_s2_citation_graph_cache(db_path, "paper-1") is False
        s2.save_s2_citation_graph(db_path, "paper-1", "references", [entry])
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1") is True
        assert s2.load_s2_citation_graph(db_path, "paper-1", "references") == [entry]
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "UPDATE s2_citation_graph_fetches SET fetched_at = ? WHERE paper_id = ?",
                ("2026-01-01T00:00:00+00:00", "paper-1"),
            )
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1", ttl_days=0) is False
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_citation_graph "
                "(paper_id, direction, rank, payload_json, fetched_at) VALUES (?, ?, ?, ?, ?)",
                ("paper-1", "references", 0, '{"bad": true}', "2026-01-01T00:00:00+00:00"),
            )
        assert s2.load_s2_citation_graph(db_path, "paper-1", "references") == []
