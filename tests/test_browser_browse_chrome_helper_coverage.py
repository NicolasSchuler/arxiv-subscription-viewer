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
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.modals.editing import PaperEditResult
from arxiv_browser.models import (
    MAX_COLLECTIONS,
    SORT_OPTIONS,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
)
from arxiv_browser.services import enrichment_service as enrich
from arxiv_browser.services.download_service import DownloadFailure, DownloadResult
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


class TestBrowserHelperCoverage:
    @pytest.mark.asyncio
    async def test_snapshot_filter_index_and_watch_helpers(self, make_paper) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper1 = make_paper(
            arxiv_id="2401.10001",
            title="Graph Learning for Papers",
            authors="A. Author",
            abstract_raw="Graph learning in practice",
        )
        paper2 = make_paper(
            arxiv_id="2401.10002",
            title="Other Paper",
            authors="B. Author",
            abstract_raw="Other abstract",
        )
        app._config = _make_app_config(
            watch_list=[
                SimpleNamespace(pattern="graph", match_type="keyword", case_sensitive=False)
            ],
            paper_metadata={
                paper1.arxiv_id: PaperMetadata(
                    arxiv_id=paper1.arxiv_id, tags=["shared"], starred=True
                ),
                paper2.arxiv_id: PaperMetadata(
                    arxiv_id=paper2.arxiv_id, tags=["shared"], starred=False
                ),
            },
            bookmarks=[SearchBookmark(name="Saved", query="graph")],
        )
        app.all_papers = [paper1, paper2]
        app.filtered_papers = [paper1, paper2]
        app._papers_by_id = {paper.arxiv_id: paper for paper in app.all_papers}
        app.selected_ids = {paper1.arxiv_id, "ghost"}
        app._watched_paper_ids = set()
        app._highlight_terms = {
            "title": ["graph"],
            "author": ["author"],
            "abstract": ["learning"],
        }
        app._match_scores = {paper1.arxiv_id: 99}
        app._pending_query = "graph"
        app._applied_query = "graph"
        app._watch_filter_active = True
        app._active_bookmark_index = 1
        app._sort_index = SORT_OPTIONS.index("title")

        app._get_search_input_widget = MagicMock(return_value=SimpleNamespace(value="  graph  "))
        option_list = _OptionListStub(highlighted=1, option_count=2)
        app._get_paper_list_widget = MagicMock(return_value=option_list)

        snapshot = app._capture_local_browse_snapshot()
        assert snapshot is not None
        assert snapshot.search_query == "graph"
        assert snapshot.list_index == 1
        assert snapshot.highlight_terms["title"] == ["graph"]

        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        assert app._capture_local_browse_snapshot() is None

        app._local_browse_snapshot = snapshot
        app._advance_dataset_epoch = MagicMock()
        app._reset_dataset_view_state = MagicMock()
        app._compute_watched_papers = MagicMock()
        app._apply_filter = MagicMock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app._get_search_input_widget = MagicMock(return_value=SimpleNamespace(value="before"))
        option_list = _OptionListStub(highlighted=0, option_count=2)
        app._get_paper_list_widget = MagicMock(return_value=option_list)

        app._restore_local_browse_snapshot()
        assert app._advance_dataset_epoch.called
        assert app.all_papers == snapshot.all_papers
        assert app._papers_by_id == snapshot.papers_by_id
        assert app.filtered_papers == snapshot.all_papers
        assert app._pending_query == snapshot.pending_query
        assert app._applied_query == snapshot.applied_query
        assert app._watch_filter_active == snapshot.watch_filter_active
        assert option_list.highlighted == 1
        assert option_list.focused is True
        app._apply_filter.assert_called_with("graph")

        app._get_search_input_widget = MagicMock(side_effect=NoMatches())
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        app._restore_local_browse_snapshot()
        assert app._apply_filter.call_count == 2

        app._search_timer = object()
        app._shutting_down = True
        app._debounced_filter()
        assert app._search_timer is None

        app._shutting_down = False
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        app._debounced_filter()
        assert app._apply_filter.call_count == 2

        app._get_paper_list_widget = MagicMock(return_value=_OptionListStub())
        app._apply_filter = MagicMock()
        app._pending_query = "  cogsci  "
        app._debounced_filter()
        app._apply_filter.assert_called_once_with("  cogsci  ")

        app._get_search_input_widget = MagicMock(return_value=SimpleNamespace(value="  live  "))
        assert app._get_live_query() == "live"
        app._get_search_input_widget = MagicMock(side_effect=AttributeError())
        app._pending_query = "  pending  "
        assert app._get_live_query() == "pending"

        app._get_or_create_metadata("2401.20001")
        app._get_or_create_metadata("2401.20001")
        assert "2401.20001" in app._config.paper_metadata

        app.filtered_papers = [paper1, paper2]
        app._get_paper_list_widget = MagicMock(return_value=_OptionListStub(highlighted=0))
        assert app._get_current_paper() == paper1
        assert app._get_current_index() == 0

        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        assert app._get_current_paper() is None
        assert app._get_current_index() is None

        app.filtered_papers = [paper1, paper2]
        app._rebuild_visible_index()
        assert app._visible_index_by_id == {paper1.arxiv_id: 0, paper2.arxiv_id: 1}
        app._visible_index_by_id = {paper1.arxiv_id: 1, paper2.arxiv_id: 0}
        assert app._get_visible_index(paper1.arxiv_id) is None
        assert app._resolve_visible_index(paper1.arxiv_id) == 0
        assert app._visible_index_by_id[paper1.arxiv_id] == 0
        assert app._resolve_visible_index("missing") is None
        assert app._visible_index_by_id.get("missing") is None

        app.filtered_papers = [paper1, paper2]
        app._get_paper_list_widget = MagicMock(return_value=_OptionListStub())
        app._update_option_at_index(-1)
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        app._update_option_at_index(0)

        calls: list[str] = []
        app.filtered_papers = [paper1, paper2]
        app.selected_ids = {paper1.arxiv_id, "ghost"}
        app._update_option_at_index = MagicMock()
        app._apply_to_selected(lambda aid: calls.append(aid))
        assert calls == [paper1.arxiv_id, "ghost"]
        assert app._update_option_at_index.called

        calls.clear()
        app._apply_to_selected(
            lambda aid: calls.append(f"target:{aid}"), target_ids={paper2.arxiv_id}
        )
        assert calls == [f"target:{paper2.arxiv_id}"]

        app._apply_to_selected = ArxivBrowser._apply_to_selected.__get__(app, ArxivBrowser)
        app._update_option_at_index = MagicMock()
        app.selected_ids = {paper1.arxiv_id, paper2.arxiv_id}
        app._config.paper_metadata[paper1.arxiv_id].is_read = False
        app._config.paper_metadata[paper2.arxiv_id].is_read = True
        app.notify = MagicMock()
        app._bulk_toggle_bool("is_read", "read", "unread", "Read")
        assert app._config.paper_metadata[paper1.arxiv_id].is_read is True
        assert app._config.paper_metadata[paper2.arxiv_id].is_read is True
        assert "2 papers read" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        app._bulk_toggle_bool("is_read", "read", "unread", "Read")
        assert app._config.paper_metadata[paper1.arxiv_id].is_read is False
        assert app._config.paper_metadata[paper2.arxiv_id].is_read is False
        assert "2 papers unread" in app.notify.call_args[0][0]

        app._watched_paper_ids.clear()
        app._config.watch_list = []
        app._compute_watched_papers = ArxivBrowser._compute_watched_papers.__get__(
            app, ArxivBrowser
        )
        app._compute_watched_papers()
        assert app._watched_paper_ids == set()
        app._config.watch_list = [
            SimpleNamespace(pattern="graph", match_type="keyword", case_sensitive=False)
        ]
        app._compute_watched_papers()
        assert app._watched_paper_ids == {paper1.arxiv_id}
        app.notify.reset_mock()
        app._notify_watch_list_matches()
        assert "1 paper match your watch list" in app.notify.call_args[0][0]
        app._watched_paper_ids.add(paper2.arxiv_id)
        app.notify.reset_mock()
        app._notify_watch_list_matches()
        assert "papers match your watch list" in app.notify.call_args[0][0]

        app.all_papers = []
        app.notify.reset_mock()
        app._show_daily_digest()
        app.all_papers = [paper1, paper2]
        app._watched_paper_ids = {paper1.arxiv_id}
        with patch("arxiv_browser.browser.browse.build_daily_digest", return_value="Digest"):
            app._show_daily_digest()
        assert app.notify.call_args.kwargs["title"] == "Daily Digest"

        bookmark_bar = SimpleNamespace(update_bookmarks=AsyncMock(return_value=None))
        app._get_bookmark_bar_widget = MagicMock(return_value=bookmark_bar)
        await app._update_bookmark_bar()
        assert bookmark_bar.update_bookmarks.await_count == 1

    @pytest.mark.asyncio
    async def test_history_reset_download_and_browser_open_helpers(
        self, make_paper, tmp_path
    ) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.20001")

        class _TaskStub:
            def __init__(self) -> None:
                self.cancelled = False
                self.cancel = MagicMock()

            def done(self) -> bool:
                return False

        app._config = _make_app_config()
        app._paper_summaries = {"2401.20001": "summary"}
        app._summary_loading = {"2401.20001"}
        app._summary_mode_label = {"2401.20001": "tldr"}
        app._summary_command_hash = {"x": "y"}
        app._badge_timer = _DummyTimer()
        app._sort_refresh_timer = _DummyTimer()
        app._abstract_cache = {"2401.20001": "abstract"}
        app._abstract_loading = {"2401.20001"}
        app._abstract_queue.append(paper)
        app._abstract_pending_ids = {"2401.20001"}
        details = SimpleNamespace(clear_cache=MagicMock())
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._s2_cache = {"2401.20001": object()}
        app._s2_loading = {"2401.20001"}
        app._s2_api_error = True
        app._hf_cache = {"2401.20001": object()}
        app._hf_loading = True
        app._hf_api_error = True
        app._version_updates = {"2401.20001": (2, 1)}
        app._version_checking = True
        app._version_progress = (1, 2)
        app._relevance_scores = {"2401.20001": (7, "relevant")}
        app._relevance_scoring_active = True
        app._scoring_progress = (1, 2)
        app._auto_tag_active = True
        app._auto_tag_progress = (1, 2)
        app._cancel_batch_requested = True
        app._tfidf_index = object()
        app._tfidf_corpus_key = "old"
        app._pending_similarity_paper_id = "2401.20001"
        app._tfidf_build_task = _TaskStub()

        app._reset_dataset_view_state()
        assert details.clear_cache.called
        assert app._abstract_cache == {}
        assert app._abstract_loading == set()
        assert app._abstract_queue == deque()
        assert app._abstract_pending_ids == set()
        assert app._s2_cache == {}
        assert app._hf_cache == {}
        assert app._version_updates == {}
        assert app._relevance_scores == {}
        assert app._auto_tag_active is False
        assert app._cancel_batch_requested is False
        assert app._tfidf_index is None
        assert app._tfidf_build_task is None

        assert app._is_history_mode() is False
        app._history_files = [(date(2026, 3, 22), tmp_path / "2026-03-22.txt")]
        assert app._is_history_mode() is True
        assert app._get_current_date() == date(2026, 3, 22)

        missing = tmp_path / "2026-03-22.txt"
        app._history_files = [(date(2026, 3, 22), missing)]
        with patch("arxiv_browser.browser.browse.parse_arxiv_file", side_effect=OSError("boom")):
            assert app._load_current_date() is False
        assert "Failed to load" in app.notify.call_args[0][0]

        history_file = tmp_path / "2026-03-22.txt"
        history_file.write_text("paper", encoding="utf-8")
        app._history_files = [(date(2026, 3, 22), history_file)]
        app._current_date_index = 0
        app._advance_dataset_epoch = MagicMock()
        app._reset_dataset_view_state = MagicMock()
        app._compute_watched_papers = MagicMock()
        app._notify_watch_list_matches = MagicMock()
        app._show_daily_digest = MagicMock()
        app._get_live_query = MagicMock(return_value="query")
        app._apply_filter = MagicMock()
        app._update_subtitle = MagicMock()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._hf_active = True
        app._fetch_hf_daily = AsyncMock(return_value=None)
        app.call_after_refresh = MagicMock()
        app.selected_ids = {paper.arxiv_id}
        with patch("arxiv_browser.browser.browse.parse_arxiv_file", return_value=[paper]):
            assert app._load_current_date() is True
        assert app._advance_dataset_epoch.called
        assert app.all_papers == [paper]
        assert app._papers_by_id == {paper.arxiv_id: paper}
        assert app.selected_ids == set()
        app._apply_filter.assert_called_with("query")
        app.call_after_refresh.assert_called_once()

        app._history_files = [
            (date(2026, 3, 22), history_file),
            (date(2026, 3, 23), history_file),
        ]
        app._current_date_index = 0
        app._load_current_date = MagicMock(return_value=True)
        assert app._set_history_index(1) is True
        assert app._current_date_index == 1
        app._load_current_date = MagicMock(return_value=False)
        assert app._set_history_index(0) is False
        assert app._current_date_index == 1
        assert app._set_history_index(-1) is False
        assert app._set_history_index(1) is True

        assert await app._download_pdf_async(paper, client=None) is False
        download_service = SimpleNamespace(
            download_pdf=AsyncMock(return_value=DownloadResult(success=True)),
        )
        app._get_services = MagicMock(return_value=SimpleNamespace(download=download_service))
        assert await app._download_pdf_async(paper, client=object()) is True
        download_service.download_pdf = AsyncMock(
            return_value=DownloadResult(success=False, failure=DownloadFailure.NETWORK)
        )
        assert await app._download_pdf_async(paper, client=object()) is False

        status_bar = SimpleNamespace(update=MagicMock())
        app._get_status_bar_widget = MagicMock(return_value=status_bar)
        app._update_footer = MagicMock()
        app._update_download_progress(2, 5)
        assert "Downloading: 2/5 complete" in status_bar.update.call_args[0][0]
        app._get_status_bar_widget = MagicMock(side_effect=NoMatches())
        app._update_download_progress(3, 5)
        assert app._update_footer.call_count == 2

        with patch("arxiv_browser.browser.browse.webbrowser.open", return_value=True):
            assert app._safe_browser_open("https://example.com") is True
        with patch("arxiv_browser.browser.browse.webbrowser.open", side_effect=OSError("boom")):
            app.notify.reset_mock()
            assert app._safe_browser_open("https://example.com") is False
        assert "open your browser" in app.notify.call_args[0][0]

    async def test_session_save_refresh_mark_and_toggle_helpers(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.20011")
        other = make_paper(arxiv_id="2401.20012")
        app._config = _make_app_config(
            paper_metadata={
                paper.arxiv_id: PaperMetadata(
                    arxiv_id=paper.arxiv_id,
                    tags=["alpha"],
                ),
                other.arxiv_id: PaperMetadata(
                    arxiv_id=other.arxiv_id,
                    tags=["alpha", "beta"],
                ),
            },
            marks={},
            bookmarks=[SearchBookmark(name="Saved", query="graph")],
        )
        app.all_papers = [paper, other]
        app.filtered_papers = [paper, other]
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_current_date = MagicMock(return_value=date(2026, 3, 22))
        app._get_active_query = MagicMock(return_value="graph")
        app._cancel_pending_detail_update = MagicMock()
        app._render_option = MagicMock(side_effect=lambda p: f"row:{p.arxiv_id}")
        app._get_paper_list_widget = MagicMock(
            return_value=_OptionListStub(highlighted=1, option_count=2)
        )
        details = SimpleNamespace(update_state=MagicMock())
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._is_history_mode = MagicMock(return_value=False)
        app._save_config_or_warn = MagicMock(return_value=True)
        app._refresh_list_view = ArxivBrowser._refresh_list_view.__get__(app, ArxivBrowser)
        app._resolve_visible_index = MagicMock(return_value=1)
        app._goto_mark = ArxivBrowser._goto_mark.__get__(app, ArxivBrowser)
        app._bulk_edit_tags = ArxivBrowser._bulk_edit_tags.__get__(app, ArxivBrowser)
        app.action_toggle_preview = ArxivBrowser.action_toggle_preview.__get__(app, ArxivBrowser)
        app.action_toggle_detail_mode = ArxivBrowser.action_toggle_detail_mode.__get__(
            app, ArxivBrowser
        )
        app.action_start_mark = ArxivBrowser.action_start_mark.__get__(app, ArxivBrowser)
        app.action_start_goto_mark = ArxivBrowser.action_start_goto_mark.__get__(app, ArxivBrowser)
        app._update_details_header = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_status_bar = MagicMock()
        app._update_option_at_index = MagicMock()
        app._collect_all_tags = MagicMock(return_value=["alpha", "beta"])
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        app._in_arxiv_api_mode = True
        app._local_browse_snapshot = SimpleNamespace(
            list_index=1,
            applied_query=" graph ",
            sort_index=0,
            selected_ids={paper.arxiv_id},
        )
        with patch_save_config(return_value=True):
            app._save_session_state()
        assert app._config.session.scroll_index == 1
        assert app._config.session.current_filter == "graph"
        assert app._config.session.selected_ids == [paper.arxiv_id]

        app._in_arxiv_api_mode = False
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        with patch_save_config(return_value=False):
            app._save_session_state()
        assert app._config.session.scroll_index == 0
        assert "Failed to save session" in app.notify.call_args[0][0]

        app._get_paper_list_widget = MagicMock(
            return_value=_OptionListStub(highlighted=1, option_count=2)
        )
        app._refresh_list_view()
        assert app._get_paper_list_widget.return_value.option_count == 2
        assert app._get_paper_list_widget.return_value.highlighted == 1

        app.filtered_papers = []
        app._watch_filter_active = False
        app._get_active_query = MagicMock(return_value="graph")
        app._is_history_mode = MagicMock(return_value=False)
        app._refresh_list_view()
        assert details.update_state.call_args.args[0] is None

        app._config.marks = {}
        app._goto_mark("a")
        assert "not set" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        app._config.marks = {"a": "missing"}
        app._get_paper_by_id = MagicMock(return_value=None)
        app._goto_mark("a")
        assert "not found" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        app._get_paper_by_id = MagicMock(return_value=paper)
        app.filtered_papers = [paper, other]
        list_view = _OptionListStub(highlighted=0, option_count=2)
        app._get_paper_list_widget = MagicMock(return_value=list_view)
        app._resolve_visible_index = MagicMock(return_value=1)
        app._goto_mark("a")
        assert list_view.highlighted == 1
        app.notify.reset_mock()
        app._resolve_visible_index = MagicMock(return_value=None)
        app._goto_mark("a")
        assert "current view" in app.notify.call_args[0][0].lower()

        app._config.paper_metadata[paper.arxiv_id].tags = ["alpha"]
        app._config.paper_metadata[other.arxiv_id].tags = ["alpha", "beta"]
        app.filtered_papers = [paper, other]
        app.selected_ids = {paper.arxiv_id, other.arxiv_id}
        app._update_option_at_index = MagicMock()
        app._bulk_edit_tags()
        callback = app.push_screen.call_args.args[1]
        callback(None)
        callback(PaperEditResult(notes="", tags=["alpha", "gamma"], active_tab="tags"))
        assert "gamma" in app._config.paper_metadata[paper.arxiv_id].tags
        assert "gamma" in app._config.paper_metadata[other.arxiv_id].tags
        assert "Added gamma" in app.notify.call_args[0][0]

        app._show_abstract_preview = False
        app.action_toggle_preview()
        assert app._show_abstract_preview is True
        app.action_toggle_preview()
        assert app._show_abstract_preview is False

        app._detail_mode = "scan"
        app.action_toggle_detail_mode()
        assert app._detail_mode == "full"
        app.action_toggle_detail_mode()
        assert app._detail_mode == "scan"

        app.action_start_mark()
        assert app._pending_mark_action == "set"
        app.action_start_goto_mark()
        assert app._pending_mark_action == "goto"

    async def test_sort_sensitive_refresh_and_badge_branches(self, make_paper) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.30011")
        other = make_paper(arxiv_id="2401.30012")
        app.filtered_papers = [paper, other]
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._s2_cache = {paper.arxiv_id: object()}
        app._hf_cache = {other.arxiv_id: object()}
        app._version_updates = {paper.arxiv_id: (2, 1)}
        app._relevance_scores = {other.arxiv_id: (8, "fit")}
        app._sort_refresh_dirty = set()
        app._sort_refresh_timer = None
        app._sort_papers = MagicMock()
        app._refresh_list_view = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_list_widget = MagicMock(
            return_value=_OptionListStub(highlighted=0, option_count=2)
        )
        app._resolve_visible_index = MagicMock(return_value=1)
        app.set_timer = MagicMock(return_value=SimpleNamespace(stop=MagicMock()))

        app._sort_index = SORT_OPTIONS.index("date")
        assert app._badge_refresh_indices(set()) == [0, 1]
        assert app._badge_refresh_indices({"unknown"}) == [0, 1]
        assert app._badge_refresh_indices({"s2"}) == [0, 1]

        app._s2_active = True
        app._hf_active = True
        assert app._badge_refresh_indices({"s2"}) == [0]
        assert app._badge_refresh_indices({"hf"}) == [1]
        assert app._badge_refresh_indices({"version"}) == [0]
        assert app._badge_refresh_indices({"relevance"}) == [1]
        app._sort_refresh_dirty = set()
        app._sort_index = SORT_OPTIONS.index("citations")
        app._schedule_sort_sensitive_refresh("s2")
        assert app._sort_refresh_dirty == {"s2"}
        assert app.set_timer.called

        app._sort_refresh_dirty = set()
        app._sort_refresh_timer = SimpleNamespace(stop=MagicMock())
        app._schedule_sort_sensitive_refresh("s2", immediate=True)
        assert app._sort_refresh_timer is None

        app._sort_refresh_dirty = {"s2"}
        app._sort_refresh_timer = None
        app._sort_papers = MagicMock()
        app._refresh_list_view = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._get_current_paper = MagicMock(return_value=paper)
        app._resolve_visible_index = MagicMock(return_value=1)
        app._get_paper_list_widget = MagicMock(
            return_value=_OptionListStub(highlighted=0, option_count=2)
        )
        app._flush_sort_sensitive_refresh()
        assert app._sort_papers.called
        assert app._refresh_list_view.called
        assert app._refresh_detail_pane.called

        app._sort_index = SORT_OPTIONS.index("trending")
        app._sort_refresh_dirty = set()
        app._schedule_sort_sensitive_refresh("hf")
        assert app._sort_refresh_dirty == {"hf"}

    @staticmethod
    def _build_chrome_state_app(make_paper):
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper1 = make_paper(arxiv_id="2401.30001")
        paper2 = make_paper(arxiv_id="2401.30002")
        theme_runtime = SimpleNamespace(
            colors={"accent": "#fff"},
            category_colors={"cs.AI": "#f00"},
            tag_namespace_colors={"tag": "#0f0"},
        )
        app._config = _make_app_config(
            theme_name="monokai",
            theme={"accent": "#fff"},
            category_colors={"cs.AI": "#f00"},
            collapsed_sections=["tags", "summary"],
            paper_metadata={
                paper1.arxiv_id: PaperMetadata(
                    arxiv_id=paper1.arxiv_id, tags=["alpha"], starred=True
                ),
                paper2.arxiv_id: PaperMetadata(arxiv_id=paper2.arxiv_id, tags=[], starred=False),
            },
            watch_list=[
                SimpleNamespace(pattern="graph", match_type="keyword", case_sensitive=False)
            ],
            marks={"a": paper1.arxiv_id},
            bookmarks=[SearchBookmark(name="Saved", query="graph")],
            show_abstract_preview=True,
            llm_command="echo {prompt}",
        )
        app.all_papers = [paper1, paper2]
        app.filtered_papers = [paper1]
        app.selected_ids = {paper1.arxiv_id}
        app._papers_by_id = {paper1.arxiv_id: paper1, paper2.arxiv_id: paper2}
        app._watched_paper_ids = {paper1.arxiv_id}
        app._highlight_terms = {"title": ["graph"], "author": ["author"], "abstract": ["learning"]}
        app._paper_summaries = {paper1.arxiv_id: "summary"}
        app._summary_loading = {paper1.arxiv_id}
        app._summary_mode_label = {paper1.arxiv_id: "tldr"}
        app._s2_cache = {paper1.arxiv_id: object()}
        app._s2_loading = {paper1.arxiv_id}
        app._s2_active = True
        app._hf_cache = {paper1.arxiv_id: object()}
        app._hf_active = True
        app._version_updates = {paper1.arxiv_id: (2, 1)}
        app._relevance_scores = {paper1.arxiv_id: (7, "relevant")}
        app._size = SimpleNamespace(width=100)
        app._in_arxiv_api_mode = True
        app._arxiv_api_loading = True
        app._show_abstract_preview = True
        app._watch_filter_active = True
        app._hf_loading = True
        app._hf_api_error = True
        app._s2_api_error = True
        app._version_checking = True
        app._version_progress = (1, 2)
        app._scoring_progress = None
        app._relevance_scoring_active = False
        app._auto_tag_progress = None
        app._auto_tag_active = False
        app._download_queue = deque([paper1])
        app._downloading = {paper1.arxiv_id}
        app._download_results = {paper1.arxiv_id: True}
        app._download_total = 1
        app._sort_index = SORT_OPTIONS.index("title")
        app._resolved_theme_runtime = MagicMock(return_value=theme_runtime)
        app._s2_state_for = MagicMock(return_value=("s2data", True))
        app._hf_state_for = MagicMock(return_value="hfdata")
        app._version_update_for = MagicMock(return_value=(2, 1))
        app._get_abstract_text = MagicMock(return_value="abstract")
        app._get_current_paper = MagicMock(return_value=paper1)
        app._get_active_query = MagicMock(return_value="graph")
        app._get_current_date = MagicMock(return_value=date(2026, 3, 22))
        app._format_arxiv_search_label = MagicMock(return_value="graph search")
        app._arxiv_search_state = SimpleNamespace(
            start=20, max_results=10, request=SimpleNamespace(query="graph")
        )
        app._get_details_header_widget = MagicMock(return_value=SimpleNamespace(update=MagicMock()))
        app._get_list_header_widget = MagicMock(return_value=SimpleNamespace(update=MagicMock()))
        app._get_status_bar_widget = MagicMock(return_value=SimpleNamespace(update=MagicMock()))
        app._get_footer_widget = MagicMock(
            return_value=SimpleNamespace(render_bindings=MagicMock())
        )
        app._get_search_container_widget = MagicMock(return_value=SimpleNamespace(is_open=True))
        app._get_paper_details_widget = MagicMock(
            return_value=SimpleNamespace(paper=paper1, update_state=MagicMock())
        )
        app._update_option_for_paper = MagicMock()
        app.notify = MagicMock()
        return app, paper1, paper2, theme_runtime

    def test_chrome_theme_and_state_builders(self, make_paper) -> None:
        app, paper1, _paper2, theme_runtime = self._build_chrome_state_app(make_paper)
        with (
            patch("arxiv_browser.browser.chrome.build_theme_runtime", return_value=theme_runtime),
            patch("arxiv_browser.browser.chrome.format_categories.cache_clear") as cache_clear,
        ):
            app._apply_category_overrides()
        assert app._theme_runtime is theme_runtime
        cache_clear.assert_called_once()
        app._config.theme = {}
        with patch("arxiv_browser.browser.chrome.build_theme_runtime", return_value=theme_runtime):
            app._apply_theme_overrides()
        app._config.theme = {"accent": "#fff"}
        app.register_theme = MagicMock(side_effect=Exception("boom"))
        with patch("arxiv_browser.browser.chrome.build_theme_runtime", return_value=theme_runtime):
            app._apply_theme_overrides()
        detail_state = app._build_detail_state(paper1.arxiv_id, paper1)
        assert detail_state.summary == "summary" and detail_state.tags == ("alpha",)
        assert detail_state.s2_data == "s2data" and detail_state.hf_data == "hfdata"
        row_state = app._build_paper_row_state(paper1)
        assert row_state.selected is True and row_state.watched is True
        status_state = app._build_status_bar_state()
        assert status_state.total == 2 and status_state.filtered == 1 and status_state.api_page == 3

    def test_chrome_subtitle_header_and_save_config_paths(self, make_paper) -> None:
        app, paper1, _paper2, _theme_runtime = self._build_chrome_state_app(make_paper)
        with patch("arxiv_browser._ascii.is_ascii_mode", return_value=True):
            assert app._format_details_header_text() == " Paper Details - scan"
            assert "Search - graph search - page 3" in app._build_subtitle_text()
        app._in_arxiv_api_mode = False
        app._get_active_query = MagicMock(return_value="graph")
        with patch("arxiv_browser._ascii.is_ascii_mode", return_value=True):
            assert app._build_subtitle_text() == "Filtered - 1/2 papers"
        app._get_active_query = MagicMock(return_value="")
        with patch("arxiv_browser._ascii.is_ascii_mode", return_value=True):
            assert app._build_subtitle_text() == "Browse - 2 papers - 2026-03-22"
        app._update_details_header()
        app._get_details_header_widget = MagicMock(side_effect=NoMatches())
        app._update_details_header()
        app._show_abstract_preview = True
        app._update_abstract_display(paper1.arxiv_id)
        app._update_option_for_paper.assert_called_with(paper1.arxiv_id)
        app._save_config_or_warn = ArxivBrowser._save_config_or_warn.__get__(app, ArxivBrowser)
        with patch_save_config(return_value=False):
            assert app._save_config_or_warn("theme preference") is False
        with patch_save_config(return_value=True):
            assert app._save_config_or_warn("theme preference") is True

    def test_chrome_command_palette_blocked_and_active_names(self, make_paper) -> None:
        app, paper1, _paper2, _theme_runtime = self._build_chrome_state_app(make_paper)
        assert app._command_palette_state().has_selection is True
        app._config.watch_list = []
        app._config.marks = {}
        app._config.paper_metadata = {}
        app.filtered_papers = []
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=None)
        app._in_arxiv_api_mode = False
        app._hf_active = False
        app._watch_filter_active = False
        app._show_abstract_preview = False
        app._detail_mode = "scan"
        app._history_files = []
        app._s2_cache = {}
        with patch("arxiv_browser.browser.chrome._resolve_llm_command", return_value=""):
            commands = app._build_command_palette_commands()
        assert (
            next(cmd for cmd in commands if cmd.action == "fetch_s2").blocked_reason == "selection"
        )
        assert (
            next(cmd for cmd in commands if cmd.action == "check_versions").blocked_reason
            == "starred papers"
        )
        app._config.watch_list = [
            SimpleNamespace(pattern="graph", match_type="keyword", case_sensitive=False)
        ]
        app._config.marks = {"a": paper1.arxiv_id}
        app._config.paper_metadata = {
            paper1.arxiv_id: PaperMetadata(arxiv_id=paper1.arxiv_id, starred=True)
        }
        app.filtered_papers = [paper1]
        app.selected_ids = {paper1.arxiv_id}
        app._get_current_paper = MagicMock(return_value=paper1)
        app._in_arxiv_api_mode = True
        app._hf_active = True
        app._watch_filter_active = True
        app._show_abstract_preview = True
        app._detail_mode = "full"
        app._history_files = [
            (date(2026, 3, 22), Path("2026-03-22.txt")),
            (date(2026, 3, 23), Path("2026-03-23.txt")),
        ]
        app._s2_cache = {paper1.arxiv_id: object()}
        with patch(
            "arxiv_browser.browser.chrome._resolve_llm_command", return_value="llm {prompt}"
        ):
            commands = app._build_command_palette_commands()
        assert (
            next(cmd for cmd in commands if cmd.action == "ctrl_e_dispatch").name
            == "Exit Search Results"
        )
        assert (
            next(cmd for cmd in commands if cmd.action == "toggle_hf").name
            == "Disable HuggingFace Trending"
        )
        assert (
            next(cmd for cmd in commands if cmd.action == "toggle_watch_filter").name
            == "Show All Papers"
        )

    def test_chrome_help_and_footer_progress_bindings(self, make_paper) -> None:
        app, paper1, _paper2, _theme_runtime = self._build_chrome_state_app(make_paper)
        app._download_queue = deque()
        app._downloading = set()
        sections = app._build_help_sections(search_first=True)
        assert sections and isinstance(sections[0][0], str)
        app._scoring_progress = (1, 2)
        assert app._get_footer_bindings()[0][1].startswith("Scoring")
        app._scoring_progress = None
        app._relevance_scoring_active = True
        assert app._get_footer_bindings()[0][1].startswith("Scoring papers")
        app._relevance_scoring_active = False
        app._version_progress = (1, 2)
        assert app._get_footer_bindings()[0][1].startswith("Versions")
        app._version_progress = None
        app._version_checking = True
        assert app._get_footer_bindings()[0][1].startswith("Checking versions")
        app._version_checking = False
        app._download_results = {paper1.arxiv_id: True}
        app._download_total = 1
        assert app._get_footer_bindings()[0][1].startswith("Downloading")
        app._download_results = {}
        app._download_total = 0
        app._auto_tag_progress = (1, 2)
        assert app._get_footer_bindings()[0][1].startswith("Auto-tagging")

    def test_chrome_widget_footer_and_status_update_hooks(self, make_paper) -> None:
        app, paper1, _paper2, _theme_runtime = self._build_chrome_state_app(make_paper)
        app._scoring_progress = None
        app._relevance_scoring_active = False
        app._version_progress = None
        app._version_checking = False
        app._download_queue = deque()
        app._downloading = set()
        app._download_results = {}
        app._download_total = 0
        app._auto_tag_progress = None
        app._auto_tag_active = False
        with (
            patch(
                "arxiv_browser.browser.chrome._widget_chrome.build_search_footer_bindings",
                return_value=[("", "search")],
            ),
            patch(
                "arxiv_browser.browser.chrome._widget_chrome.build_api_footer_bindings",
                return_value=[("", "api")],
            ),
            patch(
                "arxiv_browser.browser.chrome._widget_chrome.build_selection_footer_bindings",
                return_value=[("", "selection")],
            ),
            patch(
                "arxiv_browser.browser.chrome._widget_chrome.build_browse_footer_bindings",
                return_value=[("", "browse")],
            ),
            patch(
                "arxiv_browser.browser.chrome._widget_chrome.build_footer_mode_badge",
                return_value="badge",
            ),
            patch(
                "arxiv_browser.browser.chrome._widget_chrome.build_status_bar_text",
                return_value="status",
            ),
        ):
            app._update_footer = ArxivBrowser._update_footer.__get__(app, ArxivBrowser)
            app._update_status_bar = ArxivBrowser._update_status_bar.__get__(app, ArxivBrowser)
            assert app._get_footer_bindings() == [("", "search")]
            app._get_search_container_widget = MagicMock(
                return_value=SimpleNamespace(is_open=False)
            )
            app._in_arxiv_api_mode = True
            assert app._get_footer_bindings() == [("", "api")]
            app._in_arxiv_api_mode = False
            app.selected_ids = {paper1.arxiv_id}
            assert app._get_footer_bindings() == [("", "selection")]
            app.selected_ids = set()
            assert app._get_footer_bindings() == [("", "browse")]
            assert app._get_footer_mode_badge() == "badge"
            footer = SimpleNamespace(render_bindings=MagicMock())
            app._get_footer_widget = MagicMock(return_value=footer)
            app._update_footer()
            footer.render_bindings.assert_called_once()
            app._get_footer_widget = MagicMock(side_effect=NoMatches)
            app._update_footer()
            status = SimpleNamespace(update=MagicMock())
            app._get_status_bar_widget = MagicMock(return_value=status)
            app._update_status_bar()
            app._get_status_bar_widget = MagicMock(side_effect=NoMatches)
            app._update_status_bar()

    @pytest.mark.asyncio
    async def test_chrome_queue_refresh_and_detail_edge_branches(self, make_paper) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.30021", abstract_raw="Graph \\alpha")
        other = make_paper(arxiv_id="2401.30022", abstract_raw="Other")

        app._abstract_loading = {paper.arxiv_id}
        app._abstract_cache = {other.arxiv_id: "cached"}
        app._abstract_queue = deque([paper, other])
        app._abstract_pending_ids = {paper.arxiv_id, other.arxiv_id}
        app._track_dataset_task = MagicMock()
        app._drain_abstract_queue()
        assert app._track_dataset_task.call_count == 0
        assert app._abstract_pending_ids == set()

        app._abstract_loading = set()
        app._abstract_cache = {}
        app._abstract_queue = deque([paper])
        app._abstract_pending_ids = {paper.arxiv_id}
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._drain_abstract_queue()
        assert app._track_dataset_task.called

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._update_abstract_display = MagicMock()
        with patch(
            "arxiv_browser.browser.chrome.asyncio.to_thread",
            new=AsyncMock(return_value="cleaned"),
        ):
            await app._load_abstract_async(paper)
        assert paper.arxiv_id not in app._abstract_cache

        app._is_current_dataset_epoch = MagicMock(return_value=True)
        paper.abstract = None
        with patch(
            "arxiv_browser.browser.chrome.asyncio.to_thread",
            new=AsyncMock(return_value="cleaned"),
        ):
            await app._load_abstract_async(paper)
        assert app._abstract_cache[paper.arxiv_id] == "cleaned"
        assert paper.abstract == "cleaned"

        with patch(
            "arxiv_browser.browser.chrome.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await app._load_abstract_async(paper)

        app._in_arxiv_api_mode = False
        app._get_current_date = MagicMock(return_value=date(2026, 3, 22))
        app._get_active_query = MagicMock(return_value="graph")
        app._sort_index = 0
        app._get_paper_list_widget = MagicMock(side_effect=ScreenStackError("boom"))
        app._config = _make_app_config()
        with patch_save_config(return_value=True):
            app._save_session_state()
        assert app._config.session.scroll_index == 0

        app._pending_detail_paper = paper
        app._pending_detail_started_at = None
        app._get_current_paper = MagicMock(return_value=None)
        app._debounced_detail_update()

        app._pending_detail_paper = paper
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_details_widget = MagicMock(side_effect=NoMatches())
        app._debounced_detail_update()

        app._pending_detail_paper = paper
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_details_widget = MagicMock(
            return_value=SimpleNamespace(update_state=MagicMock())
        )
        app._get_abstract_text = MagicMock(return_value="abstract")
        app._debounced_detail_update()

        app.filtered_papers = []
        assert app._badge_refresh_indices({"s2"}) == []
        app.filtered_papers = [paper, other]
        app._s2_active = False
        assert app._badge_refresh_indices({"s2"}) == [0, 1]
        app._s2_active = True
        app._s2_cache = {paper.arxiv_id: object()}
        app._hf_active = True
        app._hf_cache = {other.arxiv_id: object()}
        app._version_updates = {paper.arxiv_id: (2, 1)}
        app._relevance_scores = {other.arxiv_id: (7, "fit")}
        assert app._badge_refresh_indices({"s2", "hf", "version", "relevance"}) == [0, 1]

        app._sort_index = -1
        assert app._sort_sensitive_badge_kind("s2") is False

        app._get_current_paper = MagicMock(return_value=None)
        app._refresh_detail_pane()
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_details_widget = MagicMock(side_effect=NoMatches())
        app._refresh_detail_pane()

        app._get_current_index = MagicMock(return_value=None)
        app._update_option_at_index = MagicMock()
        app._refresh_current_list_item()
        assert not app._update_option_at_index.called
