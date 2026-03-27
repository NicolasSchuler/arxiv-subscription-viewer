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

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.semantic_scholar import (
    CitationEntry,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)
from tests.support import canonical_exports as app_mod
from tests.support.app_stubs import (
    _DummyOptionList,
    _make_hf_paper,
    _make_s2_paper,
    _new_app,
)
from tests.support.canonical_exports import (
    ArxivBrowser,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
    _resolve_legacy_fallback,
    _resolve_papers,
)
from tests.support.patch_helpers import patch_save_config


class TestBookmarkRelevanceAndCopyCoverage:
    @pytest.mark.asyncio
    async def test_bookmark_add_goto_and_remove_paths(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.bookmarks = []
        app._active_bookmark_index = -1
        app._update_bookmark_bar = AsyncMock()
        app._apply_filter = MagicMock()
        app.notify = MagicMock()

        search_input = SimpleNamespace(value="")
        app.query_one = MagicMock(return_value=search_input)
        await app.action_add_bookmark()
        assert "Bookmark requires an active search query." in app.notify.call_args[0][0]

        search_input.value = "graph transformers"
        with patch_save_config(return_value=True) as save_mock:
            await app.action_add_bookmark()
            assert len(app._config.bookmarks) == 1
            assert app._active_bookmark_index == 0

            await app.action_goto_bookmark(99)
            await app.action_goto_bookmark(0)
            app._apply_filter.assert_called_with("graph transformers")

            app._active_bookmark_index = -1
            await app.action_remove_bookmark()
            assert "No active bookmark is selected." in app.notify.call_args[0][0]

            app._active_bookmark_index = 0
            await app.action_remove_bookmark()
            assert len(app._config.bookmarks) == 0
            assert save_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_bookmark_add_rolls_back_on_save_failure(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.bookmarks = []
        original_bookmarks = app._config.bookmarks
        app._active_bookmark_index = -1
        app._update_bookmark_bar = AsyncMock()
        app.notify = MagicMock()
        app.query_one = MagicMock(return_value=SimpleNamespace(value="graph transformers"))

        with patch_save_config(return_value=False):
            await app.action_add_bookmark()

        assert app._config.bookmarks == []
        assert app._config.bookmarks is original_bookmarks
        assert app._active_bookmark_index == -1
        app._update_bookmark_bar.assert_not_called()
        app.notify.assert_called_once_with(
            "Failed to save bookmarks",
            title="Bookmark",
            severity="error",
        )

    @pytest.mark.asyncio
    async def test_bookmark_remove_rolls_back_on_save_failure(self):
        app = _new_app()
        app._config = UserConfig()
        bookmark = SearchBookmark(name="graph transforme", query="graph transformers")
        app._config.bookmarks = [bookmark]
        original_bookmarks = app._config.bookmarks
        app._active_bookmark_index = 0
        app._update_bookmark_bar = AsyncMock()
        app.notify = MagicMock()

        with patch_save_config(return_value=False):
            await app.action_remove_bookmark()

        assert app._config.bookmarks == [bookmark]
        assert app._config.bookmarks is original_bookmarks
        assert app._active_bookmark_index == 0
        app._update_bookmark_bar.assert_not_called()
        app.notify.assert_called_once_with(
            "Failed to save bookmarks",
            title="Bookmark",
            severity="error",
        )

    @pytest.mark.asyncio
    async def test_bookmark_limit_guard(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.bookmarks = [SearchBookmark(name=f"b{i}", query=f"q{i}") for i in range(9)]
        app.notify = MagicMock()
        app._update_bookmark_bar = AsyncMock()
        app.query_one = MagicMock(return_value=SimpleNamespace(value="new query"))
        await app.action_add_bookmark()
        assert "Maximum 9 bookmarks allowed" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_action_check_versions_and_relevance_branches(self):
        app = _new_app()
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app._version_checking = True
        app._config = UserConfig()
        app._config.paper_metadata = {}
        await app.action_check_versions()
        assert "already in progress" in app.notify.call_args[0][0]

        app._version_checking = False
        app.notify.reset_mock()
        await app.action_check_versions()
        assert "No starred papers" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.paper_metadata = {
            "2401.80001": PaperMetadata(arxiv_id="2401.80001", starred=True)
        }
        await app.action_check_versions()
        app._track_task.assert_called_once()

        app2 = _new_app()
        app2.notify = MagicMock()
        app2.push_screen = MagicMock()
        app2._start_relevance_scoring = MagicMock()
        app2._require_llm_command = MagicMock(return_value=None)
        app2._ensure_llm_command_trusted = MagicMock(return_value=True)
        app2._relevance_scoring_active = False
        app2._config = UserConfig()
        app2.action_score_relevance()
        app2.push_screen.assert_not_called()

        app2._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app2._relevance_scoring_active = True
        app2.action_score_relevance()
        assert "already in progress" in app2.notify.call_args[0][0]

        app2._relevance_scoring_active = False
        app2._config.research_interests = ""
        app2.action_score_relevance()
        app2.push_screen.assert_called_once()

        app2._config.research_interests = "nlp"
        app2.action_score_relevance()
        app2._start_relevance_scoring.assert_called_with("cmd {prompt}", "nlp")

    def test_on_interests_edited_and_copy_selected_paths(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._config.research_interests = "old"
        app._relevance_scores = {"x": (8, "fit")}
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app.notify = MagicMock()

        with patch_save_config(return_value=True) as save:
            app._on_interests_edited("new")
            app._on_interests_edited("new")

        save.assert_called_once()
        assert app._relevance_scores == {}
        assert "updated" in app.notify.call_args[0][0]

        app2 = _new_app()
        app2.notify = MagicMock()
        app2._get_target_papers = MagicMock(return_value=[])
        app2.action_copy_selected()
        assert "No papers to copy" in app2.notify.call_args[0][0]

        p1 = make_paper(arxiv_id="2401.80002")
        p2 = make_paper(arxiv_id="2401.80003")
        app2._get_target_papers = MagicMock(return_value=[p1, p2])
        app2._format_paper_for_clipboard = MagicMock(side_effect=["A", "B"])
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.action_copy_selected()
        assert "Copied 2 papers" in app2.notify.call_args[0][0]

        app2._copy_to_clipboard = MagicMock(return_value=False)
        app2._format_paper_for_clipboard = MagicMock(return_value="X")
        app2.action_copy_selected()
        assert "Failed to copy to clipboard" in app2.notify.call_args[0][0]


class TestAutoTagAndPdfOpenCoverage:
    @pytest.mark.asyncio
    async def test_call_auto_tag_llm_and_acceptance_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app._llm_provider = object()
        paper = make_paper(arxiv_id="2401.90001")
        suggest_tags = AsyncMock(side_effect=app_mod._LLMExecutionError("bad command"))
        app._services = SimpleNamespace(llm=SimpleNamespace(suggest_tags_once=suggest_tags))

        result = await app._call_auto_tag_llm(paper, ["topic:ml"])
        assert result is None
        app.notify.assert_not_called()

        suggest_tags.side_effect = None
        suggest_tags.return_value = None
        result = await app._call_auto_tag_llm(paper, ["topic:ml"])
        assert result is None
        assert "Could not parse LLM response" in app.notify.call_args[0][0]

        suggest_tags.return_value = ["topic:ml"]
        result = await app._call_auto_tag_llm(paper, ["topic:ml"])
        assert result == ["topic:ml"]

        app._config = UserConfig()
        app._config.paper_metadata = {}
        app._update_option_for_paper = MagicMock()
        app._refresh_detail_pane = MagicMock()

        def get_meta(arxiv_id: str) -> PaperMetadata:
            return app._config.paper_metadata.setdefault(arxiv_id, PaperMetadata(arxiv_id=arxiv_id))

        app._get_or_create_metadata = get_meta
        with patch_save_config(return_value=True):
            app._on_auto_tag_accepted(None, paper.arxiv_id)
            app._on_auto_tag_accepted(["topic:ml"], paper.arxiv_id)
        assert app._config.paper_metadata[paper.arxiv_id].tags == ["topic:ml"]
        assert "Tags updated" in app.notify.call_args[0][0]

    def test_open_pdf_and_copy_bibtex_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._do_open_pdfs = MagicMock()
        app._get_target_papers = MagicMock(return_value=[])
        app.action_open_pdf()
        app._do_open_pdfs.assert_not_called()

        app._get_target_papers = MagicMock(return_value=[make_paper(arxiv_id="2401.90002")])
        app.action_open_pdf()
        app._do_open_pdfs.assert_called_once()

        app._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id=f"2401.{i:05d}") for i in range(20)]
        )
        app.action_open_pdf()
        app.push_screen.assert_called_once()

        app2 = _new_app()
        app2.notify = MagicMock()
        app2._get_target_papers = MagicMock(return_value=[])
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.action_copy_bibtex()
        assert "No paper selected" in app2.notify.call_args[0][0]

        app2._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id="2401.90003"), make_paper(arxiv_id="2401.90004")]
        )
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.action_copy_bibtex()
        assert "Copied 2 BibTeX entries" in app2.notify.call_args[0][0]


class TestCliResolutionCoverage:
    def test_resolve_legacy_fallback_branches(self, make_paper, tmp_path):
        missing = _resolve_legacy_fallback(tmp_path)
        assert missing == 1

        arxiv_file = tmp_path / "arxiv.txt"
        arxiv_file.write_text("placeholder", encoding="utf-8")

        with patch("arxiv_browser.cli.os.access", return_value=False):
            unreadable = _resolve_legacy_fallback(tmp_path)
        assert unreadable == 1

        with (
            patch("arxiv_browser.cli.os.access", return_value=True),
            patch("arxiv_browser.cli.parse_arxiv_file", side_effect=OSError("read error")),
        ):
            read_error = _resolve_legacy_fallback(tmp_path)
        assert read_error == 1

        with (
            patch("arxiv_browser.cli.os.access", return_value=True),
            patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]),
        ):
            ok = _resolve_legacy_fallback(tmp_path)
        assert isinstance(ok, list) and len(ok) == 1

    def test_resolve_papers_uses_explicit_date_argument(self, make_paper, tmp_path):
        args = argparse.Namespace(input=None, date="2026-01-22", no_restore=False)
        config = UserConfig()
        history_files = [
            (date(2026, 1, 23), tmp_path / "2026-01-23.txt"),
            (date(2026, 1, 22), tmp_path / "2026-01-22.txt"),
        ]
        with patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]):
            result = _resolve_papers(args, tmp_path, config, history_files)
        assert isinstance(result, tuple)
        assert result[2] == 1

    def test_resolve_papers_handles_invalid_saved_date(self, make_paper, tmp_path):
        args = argparse.Namespace(input=None, date=None, no_restore=False)
        config = UserConfig()
        config.session.current_date = "invalid-date"
        history_files = [
            (date(2026, 1, 23), tmp_path / "2026-01-23.txt"),
            (date(2026, 1, 22), tmp_path / "2026-01-22.txt"),
        ]
        with patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]):
            result = _resolve_papers(args, tmp_path, config, history_files)
        assert isinstance(result, tuple)
        assert result[2] == 0

    def test_resolve_papers_uses_saved_current_date_when_valid(self, make_paper, tmp_path):
        args = argparse.Namespace(input=None, date=None, no_restore=False)
        config = UserConfig()
        config.session.current_date = "2026-01-23"
        history_files = [
            (date(2026, 1, 23), tmp_path / "2026-01-23.txt"),
            (date(2026, 1, 22), tmp_path / "2026-01-22.txt"),
        ]
        with patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]):
            result = _resolve_papers(args, tmp_path, config, history_files)
        assert isinstance(result, tuple)
        assert result[2] == 0


class TestSaveConfigOrWarn:
    """Fix 7: _save_config_or_warn helper notifies on failure."""

    def test_returns_true_on_success(self):
        app = _new_app()
        app._config = UserConfig()
        app.notify = MagicMock()
        with patch_save_config(return_value=True):
            result = app._save_config_or_warn("test context")
        assert result is True
        app.notify.assert_not_called()

    def test_returns_false_and_notifies_on_failure(self):
        app = _new_app()
        app._config = UserConfig()
        app.notify = MagicMock()
        with patch_save_config(return_value=False):
            result = app._save_config_or_warn("theme preference")
        assert result is False
        app.notify.assert_called_once()
        call_args = app.notify.call_args
        assert "Failed to save theme preference" in call_args[0][0]
        assert call_args[1]["severity"] == "warning"


class TestLlmProviderGuards:
    """Fix 8: LLM provider None guards return gracefully instead of crashing."""

    @pytest.mark.asyncio
    async def test_generate_summary_returns_on_none_provider(self, make_paper):
        """_generate_summary_async should return early if _llm_provider is None."""
        app = _new_app()
        app._config = UserConfig()
        app._llm_provider = None
        app._summary_cache = {}
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._summary_loading = set()
        app._paper_summaries = {}
        app._update_abstract_display = MagicMock()
        app.notify = MagicMock()
        paper = make_paper()
        # Should not raise — just return
        await app._generate_summary_async(paper, "prompt_template", "hash")

    @pytest.mark.asyncio
    async def test_start_chat_returns_on_none_provider(self, make_paper):
        """_start_chat_with_paper should return early if _llm_provider is None."""
        app = _new_app()
        app._config = UserConfig()
        app._llm_provider = None
        app._papers_by_id = {"2401.00001": make_paper()}
        app._track_task = MagicMock()
        app.notify = MagicMock()
        app._get_current_paper = MagicMock(return_value=make_paper())
        # Should not raise
        app._start_chat_with_paper()
        app._track_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_score_relevance_returns_on_none_provider(self, make_paper):
        """_score_relevance_batch_async should return early if _llm_provider is None."""
        app = _new_app()
        app._config = UserConfig()
        app._llm_provider = None
        app._relevance_scores = {}
        app._scoring_in_progress = True
        app._scoring_progress = None
        app.notify = MagicMock()
        app._update_footer = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_status_bar = MagicMock()
        # Should not raise
        await app._score_relevance_batch_async([make_paper()], "cmd", "interests")

    @pytest.mark.asyncio
    async def test_call_auto_tag_returns_none_on_none_provider(self, make_paper):
        """_call_auto_tag_llm should return None if _llm_provider is None."""
        app = _new_app()
        app._llm_provider = None
        result = await app._call_auto_tag_llm(make_paper(), ["topic:ml"])
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_summary_uses_injected_llm_service(self, make_paper, tmp_path):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.90999")
        app._llm_provider = object()
        app._summary_loading = {paper.arxiv_id}
        app._paper_summaries = {}
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._summary_db_path = tmp_path / "summaries.db"
        app._update_abstract_display = MagicMock()
        app.notify = MagicMock()
        app._http_client = None
        app._config = UserConfig()
        generate_summary = AsyncMock(return_value=("service summary", None))
        app._services = SimpleNamespace(llm=SimpleNamespace(generate_summary=generate_summary))

        with patch(
            "arxiv_browser.actions.llm_actions.asyncio.to_thread", new=AsyncMock(return_value=None)
        ):
            await app._generate_summary_async(
                paper,
                "prompt_template",
                "hash123",
                mode_label="Q",
                use_full_paper_content=False,
            )

        assert app._paper_summaries[paper.arxiv_id] == "service summary"
        assert app._summary_mode_label[paper.arxiv_id] == "Q"
        assert app._summary_command_hash[paper.arxiv_id] == "hash123"
        assert paper.arxiv_id not in app._summary_loading
        generate_summary.assert_awaited_once()


class TestBestEffortS2CacheWrite:
    """Fix 9: S2 recommendation cache write failure doesn't lose API data."""

    @pytest.mark.asyncio
    async def test_recs_returned_despite_cache_write_failure(self):
        """Recommendations should be returned even if cache write fails."""
        app = _new_app()
        app._config = UserConfig()
        app._s2_db_path = "/tmp/nonexistent/s2.db"
        rec = _make_s2_paper("2401.99999")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        app._http_client = mock_client

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=S2RecommendationsCacheSnapshot(status="miss", papers=[]),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
                new=AsyncMock(return_value=([rec], True)),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.save_s2_recommendations",
                side_effect=OSError("disk full"),
            ),
        ):
            result = await app._fetch_s2_recommendations_async("2401.00001")

        assert len(result) == 1
        assert result[0].arxiv_id == "2401.99999"


class TestSearchActionCoverage:
    def test_toggle_cancel_and_date_navigation_edges(self):
        class _Container:
            def __init__(self) -> None:
                self.classes: set[str] = set()

            def add_class(self, class_name: str) -> None:
                self.classes.add(class_name)

            def remove_class(self, class_name: str) -> None:
                self.classes.discard(class_name)

        app = _new_app()
        container = _Container()
        search_input = SimpleNamespace(value="graph transformers", focused=False)

        def _focus() -> None:
            search_input.focused = True

        search_input.focus = _focus
        app._get_search_container_widget = MagicMock(return_value=container)
        app._get_search_input_widget = MagicMock(return_value=search_input)
        app._update_footer = MagicMock()
        app._apply_filter = MagicMock()
        app.notify = MagicMock()
        app._in_arxiv_api_mode = True
        app.action_exit_arxiv_search_mode = MagicMock()
        app._relevance_scoring_active = True
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app._change_arxiv_page = AsyncMock(return_value=None)

        app.action_toggle_search()
        assert "visible" in container.classes
        assert search_input.focused is True
        assert "Search mode" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app.action_toggle_search()
        assert "visible" not in container.classes
        app.notify.assert_not_called()

        container.add_class("visible")
        app.action_cancel_search()
        assert search_input.value == ""
        app._apply_filter.assert_called_once_with("")
        app.action_exit_arxiv_search_mode.assert_called_once()
        assert app._cancel_batch_requested is True
        assert "Cancelling batch operation" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._in_arxiv_api_mode = True
        app.action_prev_date()
        app.action_next_date()
        assert app._track_task.call_count == 2

        app._in_arxiv_api_mode = False
        app._is_history_mode = MagicMock(return_value=False)
        app.notify.reset_mock()
        app.action_prev_date()
        assert "Not in history mode" in app.notify.call_args.args[0]
        app.notify.reset_mock()
        app.action_next_date()
        assert "Not in history mode" in app.notify.call_args.args[0]

        app._is_history_mode = MagicMock(return_value=True)
        app._history_files = [object(), object()]
        app._current_date_index = 1
        app.notify.reset_mock()
        app.action_prev_date()
        assert "Already at oldest" in app.notify.call_args.args[0]

        app._current_date_index = 0
        app.notify.reset_mock()
        app.action_next_date()
        assert "Already at newest" in app.notify.call_args.args[0]

        app._set_history_index = MagicMock(return_value=True)
        app._get_current_date = MagicMock(return_value=date(2026, 2, 13))
        app._current_date_index = 0
        app.notify.reset_mock()
        app.action_prev_date()
        assert "Loaded 2026-02-13" in app.notify.call_args.args[0]

        app._current_date_index = 1
        app.notify.reset_mock()
        app.action_next_date()
        assert "Loaded 2026-02-13" in app.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_run_arxiv_search_status_and_stale_response_matrix(self, make_paper):
        def _status_error(status_code: int) -> httpx.HTTPStatusError:
            request = httpx.Request("GET", "https://arxiv.org/api")
            response = httpx.Response(status_code, request=request)
            return httpx.HTTPStatusError("boom", request=request, response=response)

        app = _new_app()
        app._config = UserConfig()
        app._arxiv_api_request_token = 0
        app._arxiv_api_fetch_inflight = False
        app._arxiv_api_loading = False
        app._update_status_bar = MagicMock()
        app._apply_arxiv_search_results = MagicMock()
        app.notify = MagicMock()
        request = app_mod.ArxivSearchRequest(query="graph")

        app._fetch_arxiv_api_page = AsyncMock(side_effect=_status_error(429))
        await app._run_arxiv_search(request, start=0)
        assert "HTTP 429" in app.notify.call_args.args[0]
        assert app._arxiv_api_fetch_inflight is False
        assert app._arxiv_api_loading is False

        app.notify.reset_mock()
        app._fetch_arxiv_api_page = AsyncMock(side_effect=_status_error(503))
        await app._run_arxiv_search(request, start=0)
        assert "unavailable right now" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._fetch_arxiv_api_page = AsyncMock(side_effect=_status_error(404))
        await app._run_arxiv_search(request, start=0)
        assert "rejected the request" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._fetch_arxiv_api_page = AsyncMock(side_effect=OSError("boom"))
        await app._run_arxiv_search(request, start=0)
        assert "network or I/O error" in app.notify.call_args.args[0]

        async def _stale_fetch(*_args, **_kwargs):
            app._arxiv_api_request_token += 1
            return [make_paper(arxiv_id="2401.92001")]

        app.notify.reset_mock()
        app._apply_arxiv_search_results.reset_mock()
        app._arxiv_api_fetch_inflight = False
        app._arxiv_api_loading = False
        app._fetch_arxiv_api_page = AsyncMock(side_effect=_stale_fetch)
        await app._run_arxiv_search(request, start=0)
        app.notify.assert_not_called()
        app._apply_arxiv_search_results.assert_not_called()
        assert app._arxiv_api_fetch_inflight is True
        assert app._arxiv_api_loading is True
