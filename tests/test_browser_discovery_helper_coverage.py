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
from textual.app import ScreenStackError
from textual.css.query import NoMatches
from textual.widgets.option_list import OptionDoesNotExist

import arxiv_browser.actions.ui_actions as ui_actions
import arxiv_browser.browser.core as browser_core
import arxiv_browser.browser.discovery as discovery
import arxiv_browser.cli as cli
import arxiv_browser.llm_providers as llm_providers
import arxiv_browser.semantic_scholar as s2
from arxiv_browser.actions import external_io_actions as io_actions
from arxiv_browser.actions import llm_actions as llm_actions
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, PaperMetadata, UserConfig
from arxiv_browser.services import enrichment_service as enrich
from tests.support.app_stubs import (
    _DummyInput,
    _DummyLabel,
    _DummyListView,
    _DummyTimer,
    _make_app_config,
    _new_app_stub,
    _OptionListStub,
    _paper,
)
from tests.support.patch_helpers import patch_save_config


class TestDiscoveryMixinCoverage:
    @pytest.mark.asyncio
    async def test_version_check_and_page_navigation_branches(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.70001")
        app._config = _make_app_config(
            paper_metadata={
                paper.arxiv_id: PaperMetadata(
                    arxiv_id=paper.arxiv_id,
                    starred=True,
                    last_checked_version=1,
                )
            }
        )
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._apply_arxiv_rate_limit = AsyncMock()
        app._update_footer = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._mark_badges_dirty = MagicMock()
        refresh_detail = MagicMock()
        app._get_ui_refresh_coordinator = MagicMock(
            return_value=SimpleNamespace(refresh_detail_pane=refresh_detail)
        )
        app._update_status_bar = MagicMock()
        app.notify = MagicMock()
        app._arxiv_search_state = None
        app._arxiv_api_fetch_inflight = False
        app._check_versions_async = discovery.DiscoveryMixin._check_versions_async.__get__(
            app, discovery.DiscoveryMixin
        )
        app._change_arxiv_page = discovery.DiscoveryMixin._change_arxiv_page.__get__(
            app, discovery.DiscoveryMixin
        )
        valid_feed = (
            '<atom:feed xmlns:atom="http://www.w3.org/2005/Atom">'
            "<atom:entry><atom:id>http://arxiv.org/abs/2401.70001v2</atom:id></atom:entry>"
            "<atom:entry><atom:id>http://arxiv.org/abs/2401.70002v3</atom:id></atom:entry>"
            "</atom:feed>"
        )
        response = SimpleNamespace(text=valid_feed, raise_for_status=MagicMock())

        app._http_client = None
        await app._check_versions_async({paper.arxiv_id})

        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        await app._check_versions_async({paper.arxiv_id})
        assert app._http_client.get.await_count == 0

        ids = {f"2401.{i:05d}" for i in range(41)}
        app._http_client = SimpleNamespace(get=AsyncMock(side_effect=[response, response]))
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        await app._check_versions_async(ids)
        assert app._http_client.get.await_count == 2

        epoch_checks = {"count": 0}

        def _epoch_check(_epoch: int) -> bool:
            epoch_checks["count"] += 1
            return epoch_checks["count"] == 1

        app._is_current_dataset_epoch = MagicMock(side_effect=_epoch_check)
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        await app._check_versions_async({paper.arxiv_id})

        app._is_current_dataset_epoch = MagicMock(return_value=True)
        assert app._version_updates[paper.arxiv_id] == (1, 2)
        assert refresh_detail.called
        assert "new versions" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        await app._check_versions_async({paper.arxiv_id})
        assert "up to date" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch(
            "arxiv_browser.browser.discovery.parse_arxiv_version_map",
            side_effect=ValueError("bad xml"),
        ):
            await app._check_versions_async({paper.arxiv_id})
        assert "up to date" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch(
            "arxiv_browser.browser.discovery.apply_version_updates",
            side_effect=RuntimeError("boom"),
        ):
            await app._check_versions_async({paper.arxiv_id})
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        with (
            patch(
                "arxiv_browser.browser.discovery.apply_version_updates",
                side_effect=Exception("boom"),
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await app._check_versions_async({paper.arxiv_id})

        app._in_arxiv_api_mode = False
        await app._change_arxiv_page(1)

        app._in_arxiv_api_mode = True
        app._arxiv_api_fetch_inflight = True
        app._arxiv_search_state = SimpleNamespace(
            start=0,
            max_results=10,
            request=SimpleNamespace(query="graph"),
        )
        await app._change_arxiv_page(1)
        assert "Search already in progress" in app.notify.call_args.args[0]

        app._arxiv_api_fetch_inflight = False
        await app._change_arxiv_page(-1)
        assert "first API page" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._arxiv_search_state = SimpleNamespace(
            start=10,
            max_results=10,
            request=SimpleNamespace(query="graph"),
        )
        app._run_arxiv_search = AsyncMock()
        await app._change_arxiv_page(1)
        app._run_arxiv_search.assert_awaited_once_with(app._arxiv_search_state.request, start=20)

    @pytest.mark.asyncio
    async def test_similarity_and_citation_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(
            arxiv_id="2401.80001",
            title="Graph transformers for molecular reasoning",
            authors="A. Author and B. Author",
            categories="cs.AI cs.LG",
            abstract="Graph transformers for molecular reasoning and representation learning.",
        )
        other = make_paper(
            arxiv_id="2401.80002",
            title="Graph transformers for molecular reasoning",
            authors="A. Author and B. Author",
            categories="cs.AI cs.LG",
            abstract="Graph transformers for molecular reasoning and representation learning.",
        )
        s2_paper = s2.SemanticScholarPaper(
            arxiv_id="2401.80003",
            s2_paper_id="s2:80003",
            citation_count=10,
            influential_citation_count=2,
            tldr="summary",
            fields_of_study=(),
            year=2024,
            url="https://example.com/s2",
            title="S2 title",
            abstract="S2 abstract",
        )
        citation_entry = s2.CitationEntry(
            s2_paper_id="s2:80004",
            arxiv_id="2401.80004",
            title="Citation",
            authors="A. Author",
            year=2024,
            citation_count=6,
            url="https://example.com/cite",
        )
        app.all_papers = [paper, other]
        app.filtered_papers = [paper, other]
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._config = _make_app_config(paper_metadata={})
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_list_widget = MagicMock(return_value=SimpleNamespace(highlighted=0))
        app._track_dataset_task = MagicMock(
            side_effect=lambda coro: (
                coro.close(),
                SimpleNamespace(done=MagicMock(return_value=True)),
            )[1]
        )

        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._tfidf_build_task = SimpleNamespace(done=MagicMock(return_value=False))
        app._pending_similarity_paper_id = None
        app.notify.reset_mock()
        app._show_local_recommendations(paper)
        assert "Similarity indexing in progress" in app.notify.call_args.args[0]

        app._tfidf_build_task = SimpleNamespace(done=MagicMock(return_value=True))
        app._track_dataset_task = MagicMock(
            side_effect=lambda coro: (
                coro.close(),
                SimpleNamespace(done=MagicMock(return_value=True)),
            )[1]
        )
        app.notify.reset_mock()
        app._show_local_recommendations(paper)
        assert "Indexing papers for similarity" in app.notify.call_args.args[0]
        assert app._track_dataset_task.called
        assert app._pending_similarity_paper_id == paper.arxiv_id

        app._tfidf_index = discovery.DiscoveryMixin._build_tfidf_index_for_similarity(
            [paper, other]
        )
        app._tfidf_corpus_key = discovery.build_similarity_corpus_key(app.all_papers)
        app._tfidf_build_task = SimpleNamespace(done=MagicMock(return_value=True))
        app._pending_similarity_paper_id = None

        app.all_papers = [paper]
        app._tfidf_index = discovery.DiscoveryMixin._build_tfidf_index_for_similarity(
            app.all_papers
        )
        app._tfidf_corpus_key = discovery.build_similarity_corpus_key(app.all_papers)
        app.notify.reset_mock()
        app._show_local_recommendations(paper)
        assert "No similar papers" in app.notify.call_args.args[0]

        app.all_papers = [paper, other]
        app.filtered_papers = [paper, other]
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._tfidf_index = discovery.DiscoveryMixin._build_tfidf_index_for_similarity(
            app.all_papers
        )
        app._tfidf_corpus_key = discovery.build_similarity_corpus_key(app.all_papers)
        app.notify.reset_mock()
        with patch("arxiv_browser.browser.discovery.RecommendationsScreen", return_value="screen"):
            app._show_local_recommendations(paper)
        assert app.push_screen.call_args.args[0] == "screen"

        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._pending_similarity_paper_id = None
        app._get_current_paper = MagicMock(return_value=None)
        app._show_local_recommendations = MagicMock()
        app._tfidf_build_task = None
        await app._build_tfidf_index_async(discovery.build_similarity_corpus_key(app.all_papers))
        assert app._tfidf_index is not None
        assert discovery.build_similarity_corpus_key(app.all_papers) == app._tfidf_corpus_key
        assert app._tfidf_build_task is None
        assert app.notify.call_args.kwargs["title"] == "Similar"

        app.notify.reset_mock()
        app._pending_similarity_paper_id = paper.arxiv_id
        app._get_current_paper = MagicMock(return_value=paper)
        await app._build_tfidf_index_async(discovery.build_similarity_corpus_key(app.all_papers))
        assert app._show_local_recommendations.called

        app.notify.reset_mock()
        previous_index = app._tfidf_index
        await app._build_tfidf_index_async("old")
        assert app._tfidf_index is previous_index
        assert app._tfidf_build_task is None

        app.notify.reset_mock()
        with patch(
            "arxiv_browser.browser.discovery.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await app._build_tfidf_index_async(
                discovery.build_similarity_corpus_key(app.all_papers)
            )
        assert app.notify.call_args.kwargs["severity"] == "error"

        app._is_current_dataset_epoch = MagicMock(return_value=False)
        with patch(
            "arxiv_browser.browser.discovery.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await app._build_tfidf_index_async(
                discovery.build_similarity_corpus_key(app.all_papers)
            )

        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app.notify.reset_mock()

        app.notify.reset_mock()
        app._s2_db_path = tmp_path / "s2.db"
        app._http_client = None
        assert await ui_actions._fetch_s2_recommendations_async(app, paper.arxiv_id) == []

        app._http_client = object()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_recommendations=AsyncMock(
                        return_value=SimpleNamespace(papers=[s2_paper])
                    )
                )
            )
        )
        assert await ui_actions._fetch_s2_recommendations_async(app, paper.arxiv_id) == [s2_paper]

        app._fetch_s2_recommendations_async = AsyncMock(return_value=[])
        await app._show_s2_recommendations(paper)
        assert "No Semantic Scholar recommendations" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(return_value=[s2_paper])
        with patch("arxiv_browser.browser.discovery.RecommendationsScreen", return_value="screen"):
            await app._show_s2_recommendations(paper)
        assert app.push_screen.call_args.args[0] == "screen"

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(side_effect=RuntimeError("boom"))
        await app._show_s2_recommendations(paper)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app._fetch_s2_recommendations_async = AsyncMock(return_value=[s2_paper])
        await app._show_s2_recommendations(paper)

        app._fetch_citation_graph = AsyncMock(return_value=([], []))
        await app._show_citation_graph(paper.arxiv_id, paper.title)
        assert "No citation graph data" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(return_value=([citation_entry], [citation_entry]))
        with patch(
            "arxiv_browser.browser.discovery.CitationGraphScreen", return_value="graph-screen"
        ):
            await app._show_citation_graph(paper.arxiv_id, paper.title)
        assert app.push_screen.call_args.args[0] == "graph-screen"

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(side_effect=ValueError("boom"))
        await app._show_citation_graph(paper.arxiv_id, paper.title)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app._s2_db_path = tmp_path / "s2.db"
        s2.init_s2_db(app._s2_db_path)
        s2.save_s2_citation_graph(app._s2_db_path, paper.arxiv_id, "references", [citation_entry])
        s2.save_s2_citation_graph(app._s2_db_path, paper.arxiv_id, "citations", [citation_entry])
        app._http_client = object()
        assert await ui_actions._fetch_citation_graph(app, paper.arxiv_id) == (
            [citation_entry],
            [citation_entry],
        )

        refs = [citation_entry]
        cites = [citation_entry]
        app._http_client = object()
        with (
            patch(
                "arxiv_browser.actions.ui_actions.fetch_s2_references",
                new=AsyncMock(return_value=(refs, False)),
            ),
            patch(
                "arxiv_browser.actions.ui_actions.fetch_s2_citations",
                new=AsyncMock(return_value=(cites, True)),
            ),
        ):
            assert await ui_actions._fetch_citation_graph(app, f"ARXIV:{paper.arxiv_id}") == (
                refs,
                cites,
            )

    @pytest.mark.asyncio
    async def test_similarity_and_recommendation_edge_branches(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(
            arxiv_id="2401.80021",
            title="Graph transformers for molecular reasoning",
            authors="A. Author and B. Author",
            categories="cs.AI cs.LG",
            abstract=None,
            abstract_raw="Graph \\alpha and more text.",
        )
        other = make_paper(
            arxiv_id="2401.80022",
            title="Neighbor paper",
            authors="C. Author",
            categories="cs.AI",
            abstract=None,
            abstract_raw="",
        )
        s2_paper = s2.SemanticScholarPaper(
            arxiv_id="2401.80023",
            s2_paper_id="s2:80023",
            citation_count=0,
            influential_citation_count=0,
            tldr="summary",
            fields_of_study=(),
            year=2024,
            url="https://example.com/s2",
            title="S2 title",
            abstract="S2 abstract",
        )
        citation_entry = s2.CitationEntry(
            s2_paper_id="s2:80024",
            arxiv_id="2401.80024",
            title="Citation",
            authors="A. Author",
            year=2024,
            citation_count=0,
            url="https://example.com/cite",
        )
        app.all_papers = [paper, other]
        app.filtered_papers = [paper, other]
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._config = _make_app_config(paper_metadata={})
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_list_widget = MagicMock(return_value=SimpleNamespace(highlighted=0))
        app._track_dataset_task = MagicMock(side_effect=lambda coro: (coro.close(), None)[1])
        app._resolve_visible_index = MagicMock(return_value=None)

        index = discovery.DiscoveryMixin._build_tfidf_index_for_similarity([paper, other])
        assert index is not None

        app._http_client = SimpleNamespace(get=AsyncMock(side_effect=asyncio.CancelledError()))
        app._apply_arxiv_rate_limit = AsyncMock()
        with pytest.raises(asyncio.CancelledError):
            await app._check_versions_async({paper.arxiv_id})

        app._http_client = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(text="", raise_for_status=MagicMock()))
        )
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        await app._check_versions_async({paper.arxiv_id})

        app._fetch_s2_recommendations_async = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await app._show_s2_recommendations(paper)

        app._fetch_s2_recommendations_async = AsyncMock(side_effect=httpx.HTTPError("boom"))
        await app._show_s2_recommendations(paper)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(return_value=[s2_paper])
        with patch("arxiv_browser.browser.discovery.RecommendationsScreen", return_value="screen"):
            await app._show_s2_recommendations(paper)
        assert app.push_screen.call_args.args[0] == "screen"

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(side_effect=httpx.HTTPError("boom"))
        await app._show_citation_graph(paper.arxiv_id, paper.title)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(return_value=([], [citation_entry]))
        with patch(
            "arxiv_browser.browser.discovery.CitationGraphScreen", return_value="graph-screen"
        ):
            await app._show_citation_graph(paper.arxiv_id, paper.title)
        assert app.push_screen.call_args.args[0] == "graph-screen"

        app.notify.reset_mock()
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._fetch_s2_recommendations_async = AsyncMock(return_value=[s2_paper])
        await app._show_s2_recommendations(paper)

        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._fetch_citation_graph = AsyncMock(return_value=([citation_entry], [citation_entry]))
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        await app._show_citation_graph(paper.arxiv_id, paper.title)

        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._on_recommendation_selected(None)
        app._llm_provider = object()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    suggest_tags_once=AsyncMock(side_effect=RuntimeError("boom")),
                )
            )
        )
        assert await discovery.DiscoveryMixin._call_auto_tag_llm(app, paper, ["existing"]) is None
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    suggest_tags_once=AsyncMock(side_effect=Exception("boom")),
                )
            )
        )
        with pytest.raises(Exception, match="boom"):
            await discovery.DiscoveryMixin._call_auto_tag_llm(app, paper, ["existing"])

        app._resolve_visible_index = MagicMock(return_value=None)
        app.notify.reset_mock()
        app._on_recommendation_selected("missing")
        assert "not in the current filtered view" in app.notify.call_args[0][0]

        recs = [
            s2.SemanticScholarPaper(
                arxiv_id="2401.80025",
                s2_paper_id="s2:80025",
                citation_count=0,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=2024,
                url="https://example.com/zero",
            )
        ]
        tuples = discovery.DiscoveryMixin._s2_recs_to_paper_tuples(recs)
        assert tuples[0][1] == 0

    @pytest.mark.asyncio
    async def test_discovery_stale_and_unexpected_exception_edges(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.80031", abstract="Paper abstract")
        other = make_paper(arxiv_id="2401.80032", abstract="Other abstract")
        app.all_papers = [paper, other]
        app.filtered_papers = [paper, other]
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._config = _make_app_config(paper_metadata={})
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._pending_similarity_paper_id = paper.arxiv_id
        app._get_current_paper = MagicMock(return_value=other)
        app._show_local_recommendations = MagicMock()
        app._resolve_visible_index = MagicMock(return_value=None)

        await app._build_tfidf_index_async(discovery.build_similarity_corpus_key(app.all_papers))
        assert "Similarity index ready" in app.notify.call_args.args[0]
        app._show_local_recommendations.assert_not_called()

        app.notify.reset_mock()
        with (
            patch(
                "arxiv_browser.browser.discovery.asyncio.to_thread",
                new=AsyncMock(side_effect=Exception("boom")),
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await app._build_tfidf_index_async(
                discovery.build_similarity_corpus_key(app.all_papers)
            )

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            await app._show_s2_recommendations(paper)

        app.notify.reset_mock()
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._fetch_s2_recommendations_async = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            await app._show_s2_recommendations(paper)
        assert app.notify.call_count == 1
        assert app.notify.call_args.args[0] == "Fetching S2 recommendations..."

        app.notify.reset_mock()
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._fetch_citation_graph = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            await app._show_citation_graph(paper.arxiv_id, paper.title)

        app.notify.reset_mock()
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._fetch_citation_graph = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            await app._show_citation_graph(paper.arxiv_id, paper.title)
        app.notify.assert_not_called()
