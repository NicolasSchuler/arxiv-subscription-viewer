#!/usr/bin/env python3
"""High-impact coverage tests for action-heavy paths in app.py."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.cli import (
    _resolve_legacy_fallback,
    _resolve_papers,
)
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import (
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
)
from arxiv_browser.semantic_scholar import (
    CitationEntry,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)
from tests.support.app_stubs import (
    _DummyOptionList,
    _make_hf_paper,
    _make_s2_paper,
    _new_app,
)


class TestStatusCommandPaletteAndChatCoverage:
    def test_update_status_bar_renders_mode_badges(self):
        class DummyLabel:
            def __init__(self):
                self.content = ""

            def update(self, text):
                self.content = text

        label = DummyLabel()
        app = _new_app()
        app.query_one = MagicMock(return_value=label)
        app._update_footer = MagicMock()
        app._get_active_query = MagicMock(return_value="graph transformers")
        app.all_papers = [object(), object(), object()]
        app.filtered_papers = [object()]
        app.selected_ids = {"a", "b"}
        app._sort_index = 0
        app._watch_filter_active = False
        app._in_arxiv_api_mode = False
        app._arxiv_search_state = None
        app._arxiv_api_loading = False
        app._show_abstract_preview = True
        app._s2_active = True
        app._s2_loading = {"a"}
        app._s2_cache = {"a": object()}
        app._s2_api_error = False
        app._hf_active = True
        app._hf_loading = False
        app._hf_cache = {"a": _make_hf_paper("a")}
        app._hf_api_error = False
        app._papers_by_id = {"a": object()}
        app._version_checking = True
        app._version_updates = {"a": (1, 2)}

        app._update_status_bar()

        content = str(label.content)
        assert "S2 loading" in content
        assert "HF:1" in content
        assert "Checking versions" in content
        assert "Preview" in content
        app._update_footer.assert_called_once()

    def test_api_error_flags_default_false(self):
        from arxiv_browser.models import Paper

        fresh = ArxivBrowser(papers=[])
        assert fresh._s2_api_error is False
        assert fresh._hf_api_error is False

    def test_status_bar_shows_s2_err_on_api_error(self):
        class DummyLabel:
            def __init__(self):
                self.content = ""

            def update(self, text):
                self.content = text

        label = DummyLabel()
        app = _new_app()
        app.query_one = MagicMock(return_value=label)
        app._update_footer = MagicMock()
        app._get_active_query = MagicMock(return_value="")
        app.all_papers = []
        app.filtered_papers = []
        app.selected_ids = set()
        app._sort_index = 0
        app._watch_filter_active = False
        app._in_arxiv_api_mode = False
        app._arxiv_search_state = None
        app._arxiv_api_loading = False
        app._show_abstract_preview = False
        app._s2_active = True
        app._s2_loading = set()
        app._s2_cache = {}
        app._s2_api_error = True
        app._hf_active = False
        app._hf_loading = False
        app._hf_cache = {}
        app._hf_api_error = False
        app._papers_by_id = {}
        app._version_checking = False
        app._version_updates = {}

        app._update_status_bar()

        content = str(label.content)
        assert "S2:err" in content

    def test_status_bar_shows_hf_err_on_api_error(self):
        class DummyLabel:
            def __init__(self):
                self.content = ""

            def update(self, text):
                self.content = text

        label = DummyLabel()
        app = _new_app()
        app.query_one = MagicMock(return_value=label)
        app._update_footer = MagicMock()
        app._get_active_query = MagicMock(return_value="")
        app.all_papers = []
        app.filtered_papers = []
        app.selected_ids = set()
        app._sort_index = 0
        app._watch_filter_active = False
        app._in_arxiv_api_mode = False
        app._arxiv_search_state = None
        app._arxiv_api_loading = False
        app._show_abstract_preview = False
        app._s2_active = False
        app._s2_loading = set()
        app._s2_cache = {}
        app._s2_api_error = False
        app._hf_active = True
        app._hf_loading = False
        app._hf_cache = {}
        app._hf_api_error = True
        app._papers_by_id = {}
        app._version_checking = False
        app._version_updates = {}

        app._update_status_bar()

        content = str(label.content)
        assert "HF:err" in content

        # Test that action_command_palette opens OmniInput in command mode
        app = _new_app()
        omni_mock = SimpleNamespace(
            set_commands=MagicMock(),
            open=MagicMock(),
        )
        app._get_search_container_widget = MagicMock(return_value=omni_mock)
        app._build_command_palette_commands = MagicMock(return_value=[("Demo", "demo")])
        app.action_command_palette()
        omni_mock.set_commands.assert_called_once_with([("Demo", "demo")])
        omni_mock.open.assert_called_once_with(">")

        # Test on_omni_command_selected dispatches actions correctly
        from arxiv_browser.widgets.omni_input import OmniInput

        app2 = _new_app()
        omni_mock2 = SimpleNamespace(close=MagicMock())
        app2._get_search_container_widget = MagicMock(return_value=omni_mock2)
        app2._update_footer = MagicMock()

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app2._track_task = MagicMock(side_effect=track_task)

        async def fake_async():
            return None

        def action_demo():
            return fake_async()

        def action_boom():
            raise RuntimeError("boom")

        app2.action_demo = action_demo
        app2.action_boom = action_boom

        handler = ArxivBrowser.on_omni_command_selected.__get__(app2, ArxivBrowser)
        with patch("arxiv_browser.browser.core.logger") as logger_mock:
            # Successful async action
            event = OmniInput.CommandSelected(action="demo")
            handler(event)
            app2._track_task.assert_called_once()

            # Failing action
            event = OmniInput.CommandSelected(action="boom")
            handler(event)
            boom_warnings = [
                call
                for call in logger_mock.warning.call_args_list
                if call.args and "OmniInput command failed" in call.args[0]
            ]
            assert boom_warnings

            # Missing action — no crash
            event = OmniInput.CommandSelected(action="missing")
            handler(event)

    @pytest.mark.asyncio
    async def test_chat_and_summary_action_paths(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        app.notify = MagicMock()
        app._summary_loading = set()
        app._paper_summaries = {}
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._summary_db_path = MagicMock()
        app._update_abstract_display = MagicMock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())

        paper = make_paper(arxiv_id="2401.60001")
        app._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app._get_current_paper = MagicMock(return_value=paper)
        app._llm_provider = MagicMock()
        app.push_screen = MagicMock()

        app.action_generate_summary()
        app.push_screen.assert_called_once()

        with patch(
            "arxiv_browser.actions.llm_actions._load_summary", return_value="cached summary"
        ):
            app._on_summary_mode_selected("default", paper, "cmd {prompt}")
        assert app._paper_summaries[paper.arxiv_id] == "cached summary"

        app.action_chat_with_paper()
        app._track_task.assert_called()

        with patch(
            "arxiv_browser.browser.core.ArxivBrowser._fetch_paper_content_async",
            return_value="full text",
        ):
            await app._open_chat_screen(paper, app._llm_provider)
        assert app.push_screen.call_count >= 2


class TestArxivApiAndSimilarityCoverage:
    @pytest.mark.asyncio
    async def test_arxiv_api_service_injection_controls_label_rate_limit_and_page(self, make_paper):
        from arxiv_browser.models import ArxivSearchRequest

        request = ArxivSearchRequest(query="transformers", field="all", category="")
        app = _new_app()
        app.notify = MagicMock()
        app._last_arxiv_api_request_at = 1.0
        app._http_client = object()
        app._services = SimpleNamespace(
            arxiv_api=SimpleNamespace(
                format_query_label=MagicMock(return_value="custom-label"),
                enforce_rate_limit=AsyncMock(return_value=(4.5, 1.2)),
                fetch_page=AsyncMock(return_value=[make_paper(arxiv_id="2401.70000")]),
            )
        )

        label = app._format_arxiv_search_label(request)
        assert label == "custom-label"

        await app._apply_arxiv_rate_limit()
        assert app._last_arxiv_api_request_at == 4.5
        assert "Waiting 1.2s for arXiv API rate limit" in app.notify.call_args[0][0]

        papers = await app._fetch_arxiv_api_page(request, start=0, max_results=5)
        assert [paper.arxiv_id for paper in papers] == ["2401.70000"]

    @pytest.mark.asyncio
    async def test_fetch_arxiv_api_page_with_and_without_shared_client(self, make_paper):
        request = SimpleNamespace(query="transformers", field="all", category="")

        app = _new_app()
        app._apply_arxiv_rate_limit = AsyncMock()
        app._http_client = object()
        app._services = SimpleNamespace(
            arxiv_api=SimpleNamespace(fetch_page=AsyncMock(return_value=[make_paper()]))
        )

        papers = await app._fetch_arxiv_api_page(request, start=0, max_results=5)
        assert len(papers) == 1

        app2 = _new_app()
        app2._apply_arxiv_rate_limit = AsyncMock()
        app2._http_client = None
        app2._services = SimpleNamespace(
            arxiv_api=SimpleNamespace(fetch_page=AsyncMock(return_value=[make_paper()]))
        )
        papers2 = await app2._fetch_arxiv_api_page(request, start=0, max_results=5)
        assert len(papers2) == 1

    def test_show_similar_actions_dispatch_and_local_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._show_recommendations = MagicMock()
        app._get_current_paper = MagicMock(return_value=None)
        app.action_show_similar()
        assert "No paper is selected." in app.notify.call_args[0][0]

        paper = make_paper(arxiv_id="2401.70001")
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_active = True
        app.action_show_similar()
        app._show_recommendations.assert_called_with(paper, "local", s2_available=True)

        app._show_recommendations.reset_mock()
        app._s2_active = False
        app.action_show_similar()
        app._show_recommendations.assert_called_with(paper, "local", s2_available=False)

    def test_show_recommendations_dispatch(self, make_paper):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.70002")
        app._show_local_recommendations = MagicMock()
        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)

        app._show_recommendations(paper, None)
        app._show_recommendations(paper, "local")
        app._show_recommendations(paper, "s2")

        app._show_local_recommendations.assert_called_once_with(paper, s2_available=False)
        app._track_task.assert_called_once()

    def test_show_local_recommendations_empty_and_success(self, make_paper):
        from arxiv_browser.similarity import build_similarity_corpus_key

        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app._tfidf_index = object()
        app._get_abstract_text = MagicMock(return_value="abstract")
        paper = make_paper(arxiv_id="2401.70003")
        app.all_papers = [paper, make_paper(arxiv_id="2401.70004")]
        app._tfidf_corpus_key = build_similarity_corpus_key(app.all_papers)

        with patch("arxiv_browser.browser.discovery.find_similar_papers", return_value=[]):
            app._show_local_recommendations(paper)
        assert "No similar papers were found." in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with patch(
            "arxiv_browser.browser.discovery.find_similar_papers",
            return_value=[(make_paper(arxiv_id="2401.70005"), 0.9)],
        ):
            app._show_local_recommendations(paper)
        app.push_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_s2_recommendations_and_citation_graph_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        paper = make_paper(arxiv_id="2401.70006")

        app._fetch_s2_recommendations_async = AsyncMock(return_value=[])
        await app._show_s2_recommendations(paper)
        assert "No Semantic Scholar recommendations were found." in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(return_value=[_make_s2_paper("2401.70007")])
        app._s2_recs_to_paper_tuples = MagicMock(
            return_value=[(make_paper(arxiv_id="2401.70007"), 0.8)]
        )
        await app._show_s2_recommendations(paper)
        app.push_screen.assert_called()

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(side_effect=RuntimeError("s2 error"))
        await app._show_s2_recommendations(paper)
        assert "Could not fetch Semantic Scholar recommendations." in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = False
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_cache = {}
        app._track_task = MagicMock()
        app.action_citation_graph()
        assert "Semantic Scholar is disabled" in app.notify.call_args[0][0]

        app._s2_active = True
        app.notify.reset_mock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app.action_citation_graph()
        app._track_task.assert_called_once()

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(return_value=([], []))
        await app._show_citation_graph("ARXIV:2401.70006", "Test")
        assert "No citation graph data was found." in app.notify.call_args[0][0]

        refs = [
            CitationEntry(
                s2_paper_id="s2p",
                arxiv_id="2401.70008",
                title="Cited",
                authors="A",
                year=2024,
                citation_count=1,
                url="",
            )
        ]
        app._fetch_citation_graph = AsyncMock(return_value=(refs, []))
        app._papers_by_id = {"2401.70008": make_paper(arxiv_id="2401.70008")}
        await app._show_citation_graph("ARXIV:2401.70006", "Test")
        assert app.push_screen.call_count >= 2

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(side_effect=RuntimeError("boom"))
        await app._show_citation_graph("ARXIV:2401.70006", "Test")
        assert "Could not load the citation graph." in app.notify.call_args[0][0]
