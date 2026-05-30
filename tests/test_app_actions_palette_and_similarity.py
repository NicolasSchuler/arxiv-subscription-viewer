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
        app.push_screen.reset_mock()
        with (
            patch("arxiv_browser.browser.discovery.find_similar_papers", return_value=[]),
            patch("arxiv_browser.browser.discovery.RecommendationsScreen", return_value="screen"),
        ):
            app._show_local_recommendations(paper, s2_available=True)
        app.notify.assert_not_called()
        app.push_screen.assert_called_once()
        assert app.push_screen.call_args.args[0] == "screen"

        app.notify.reset_mock()
        app.push_screen.reset_mock()
        with patch(
            "arxiv_browser.browser.discovery.find_similar_papers",
            return_value=[(make_paper(arxiv_id="2401.70005"), 0.9)],
        ):
            app._show_local_recommendations(paper)
        app.push_screen.assert_called_once()

    def test_serendipity_palette_entry_resolves_to_action(self):
        from arxiv_browser.browser.contracts import COMMAND_PALETTE_COMMANDS

        entry = next(command for command in COMMAND_PALETTE_COMMANDS if command[0] == "Surprise Me")
        assert entry[3] == "serendipity"
        assert hasattr(ArxivBrowser, "action_serendipity")

    def test_review_palette_entries_and_availability(self):
        from arxiv_browser.browser.contracts import COMMAND_PALETTE_COMMANDS, _PaletteAppState

        entries = {command[0]: command[3] for command in COMMAND_PALETTE_COMMANDS}
        assert entries["Schedule Review"] == "schedule_review"
        assert entries["Mark Reviewed"] == "mark_reviewed"
        assert entries["Clear Review"] == "clear_review"
        assert entries["Show Due Reviews"] == "show_due_reviews"
        assert hasattr(ArxivBrowser, "action_schedule_review")
        assert hasattr(ArxivBrowser, "action_show_due_reviews")

        app = _new_app()
        state = _PaletteAppState(
            in_arxiv_api_mode=False,
            hf_active=False,
            watch_filter_active=False,
            show_abstract_preview=False,
            compact_list=False,
            detail_mode="scan",
            active_query="",
            has_history_files=False,
            has_history_navigation=False,
            watch_list=[],
            has_marks=False,
            has_starred=False,
            llm_configured=False,
            has_visible_papers=False,
            has_selection=False,
            selected_count=0,
            has_current_paper=False,
            has_target_papers=False,
            s2_active=False,
            s2_data_loaded=False,
        )
        assert app._palette_basic_blocked_reason("schedule_review", state) == "selection"
        assert app._palette_basic_blocked_reason("mark_reviewed", state) == "selection"
        assert app._palette_basic_blocked_reason("clear_review", state) == "selection"
        assert app._palette_basic_blocked_reason("show_due_reviews", state) == "visible papers"

    def test_serendipity_action_jumps_to_ranked_visible_paper(self, make_paper):
        from arxiv_browser.similarity import build_similarity_corpus_key

        starred = make_paper(
            arxiv_id="2401.71001",
            title="Transformer Attention",
            categories="cs.CL",
            abstract="transformer attention language token",
            abstract_raw="transformer attention language token",
        )
        near = make_paper(
            arxiv_id="2401.71002",
            title="Language Attention",
            categories="cs.CL",
            abstract="language attention transformer token",
            abstract_raw="language attention transformer token",
        )
        far = make_paper(
            arxiv_id="2401.71003",
            title="Quantum Qubits",
            categories="quant-ph",
            abstract="quantum qubit entanglement superconducting",
            abstract_raw="quantum qubit entanglement superconducting",
        )
        papers = [starred, near, far]
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = papers
        app.filtered_papers = papers
        app._config = UserConfig(
            paper_metadata={starred.arxiv_id: PaperMetadata(starred.arxiv_id, starred=True)}
        )
        app._tfidf_index = app._build_tfidf_index_for_similarity(papers)
        app._tfidf_corpus_key = build_similarity_corpus_key(papers)
        app._get_current_paper = MagicMock(return_value=starred)
        app._resolve_visible_index = MagicMock(return_value=2)
        option_list = _DummyOptionList()
        app._get_paper_list_widget = MagicMock(return_value=option_list)

        app.action_serendipity()

        app._resolve_visible_index.assert_called_once_with(far.arxiv_id)
        assert option_list.highlighted == 2
        assert "new category quant-ph" in app.notify.call_args.args[0]

    def test_serendipity_action_starts_lazy_index_build(self, make_paper):
        paper = make_paper(arxiv_id="2401.71101")
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = [paper]
        app.filtered_papers = [paper]
        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._tfidf_build_task = None

        tracked = []

        def track_dataset_task(coro):
            tracked.append(coro)
            coro.close()
            return SimpleNamespace(done=lambda: False)

        async def build_index(_corpus_key):
            return None

        app._track_dataset_task = MagicMock(side_effect=track_dataset_task)
        app._build_tfidf_index_async = build_index

        app.action_serendipity()

        assert app._pending_serendipity is True
        assert len(tracked) == 1
        assert "Indexing papers for serendipity" in app.notify.call_args.args[0]

    def test_serendipity_action_reports_index_build_in_progress(self, make_paper):
        paper = make_paper(arxiv_id="2401.71102")
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = [paper]
        app.filtered_papers = [paper]
        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._tfidf_build_task = SimpleNamespace(done=lambda: False)
        app._track_dataset_task = MagicMock()

        app.action_serendipity()

        assert app._pending_serendipity is True
        assert "Similarity indexing in progress" in app.notify.call_args.args[0]
        app._track_dataset_task.assert_not_called()

    def test_serendipity_action_warns_when_ranker_finds_no_candidates(self, make_paper):
        from arxiv_browser.similarity import build_similarity_corpus_key

        paper = make_paper(arxiv_id="2401.71103")
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = [paper]
        app.filtered_papers = [paper]
        app._tfidf_index = object()
        app._tfidf_corpus_key = build_similarity_corpus_key(app.all_papers)
        app._get_current_paper = MagicMock(return_value=paper)
        app._resolve_visible_index = MagicMock()

        with patch("arxiv_browser.actions.ui_actions.find_serendipitous_papers", return_value=[]):
            app.action_serendipity()

        assert "No serendipitous paper could be selected" in app.notify.call_args.args[0]
        app._resolve_visible_index.assert_not_called()

    def test_serendipity_action_warns_when_candidate_is_not_visible(self, make_paper):
        from arxiv_browser.similarity import (
            SerendipityCandidate,
            build_similarity_corpus_key,
        )

        paper = make_paper(arxiv_id="2401.71104")
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = [paper]
        app.filtered_papers = [paper]
        app._tfidf_index = object()
        app._tfidf_corpus_key = build_similarity_corpus_key(app.all_papers)
        app._get_current_paper = MagicMock(return_value=paper)
        app._resolve_visible_index = MagicMock(return_value=None)
        app._get_paper_list_widget = MagicMock()

        with patch(
            "arxiv_browser.actions.ui_actions.find_serendipitous_papers",
            return_value=[SerendipityCandidate(paper=paper, score=1.0, reason="surprise")],
        ):
            app.action_serendipity()

        assert "not in the current filtered view" in app.notify.call_args.args[0]
        app._get_paper_list_widget.assert_not_called()

    @pytest.mark.asyncio
    async def test_tfidf_build_resumes_pending_serendipity(self, make_paper):
        from arxiv_browser.similarity import build_similarity_corpus_key

        papers = [
            make_paper(arxiv_id="2401.71201", abstract_raw="alpha beta gamma"),
            make_paper(arxiv_id="2401.71202", abstract_raw="delta epsilon zeta"),
        ]
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = papers
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._pending_similarity_paper_id = None
        app._pending_similarity_s2_available = False
        app._pending_serendipity = True
        app._tfidf_build_task = None
        app.action_serendipity = MagicMock()

        await app._build_tfidf_index_async(build_similarity_corpus_key(papers))

        app.action_serendipity.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_tfidf_build_resumes_pending_serendipity_after_stale_similarity(self, make_paper):
        from arxiv_browser.similarity import build_similarity_corpus_key

        current = make_paper(arxiv_id="2401.71203", abstract_raw="alpha beta gamma")
        pending = make_paper(arxiv_id="2401.71204", abstract_raw="delta epsilon zeta")
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = [current, pending]
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._pending_similarity_paper_id = pending.arxiv_id
        app._pending_similarity_s2_available = False
        app._pending_serendipity = True
        app._tfidf_build_task = None
        app._get_current_paper = MagicMock(return_value=current)
        app.action_serendipity = MagicMock()
        app._show_local_recommendations = MagicMock()

        await app._build_tfidf_index_async(build_similarity_corpus_key(app.all_papers))

        app.action_serendipity.assert_called_once_with()
        app._show_local_recommendations.assert_not_called()

    def test_serendipity_action_empty_pool_warns_without_jump(self):
        app = _new_app()
        app.notify = MagicMock()
        app.all_papers = []
        app.filtered_papers = []
        app._get_paper_list_widget = MagicMock()

        app.action_serendipity()

        assert "No visible papers are available" in app.notify.call_args.args[0]
        app._get_paper_list_widget.assert_not_called()

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
