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
import arxiv_browser.browser.content as browser_content
import arxiv_browser.browser.core as browser_core
import arxiv_browser.browser.discovery as discovery
import arxiv_browser.cli as cli
import arxiv_browser.llm_providers as llm_providers
import arxiv_browser.semantic_scholar as s2
from arxiv_browser.actions import external_io_actions as io_actions
from arxiv_browser.actions import llm_actions as llm_actions
from arxiv_browser.browser.content import (
    MAX_PAPER_CONTENT_LENGTH,
    _fetch_paper_content_async,
)
from arxiv_browser.browser.core import (
    MAX_ABSTRACT_LOADS,
    ArxivBrowser,
    build_list_empty_message,
)
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionsModal,
    CollectionViewModal,
)
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, SessionState, UserConfig
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


class TestAppHelperCoverage:
    def test_list_message_and_paper_content_branches(self) -> None:
        assert "No papers match your search" in build_list_empty_message(
            query="x",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No API results on this page" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=True,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No watched papers found" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=True,
            history_mode=False,
        )
        assert "No papers available for this date" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=True,
        )
        assert "No papers available." in build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )

        class _Response:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

        class _Client:
            def __init__(self, response: _Response) -> None:
                self.response = response

            async def get(self, *_args, **_kwargs):
                return self.response

        class _TempClient:
            def __init__(self, response: _Response) -> None:
                self.response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *_args, **_kwargs):
                return self.response

        paper = _paper(arxiv_id="2401.00001", abstract="Fallback abstract.")
        empty_paper = _paper(arxiv_id="2401.00002", abstract=None, abstract_raw=None)
        long_text = "x" * (MAX_PAPER_CONTENT_LENGTH + 10)

        with patch("arxiv_browser.browser.content.extract_text_from_html", return_value=long_text):
            text = asyncio.run(
                _fetch_paper_content_async(paper, _Client(_Response(200, "<p>x</p>")))
            )
        assert len(text) == MAX_PAPER_CONTENT_LENGTH

        with patch("arxiv_browser.browser.content.extract_text_from_html", return_value=""):
            text = asyncio.run(_fetch_paper_content_async(paper, _Client(_Response(404, ""))))
        assert text == "Abstract:\nFallback abstract."

        with (
            patch("arxiv_browser.browser.content.extract_text_from_html", return_value=""),
            patch(
                "arxiv_browser.browser.content.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
        ):
            text = asyncio.run(_fetch_paper_content_async(empty_paper, None))
        assert text == ""

    @pytest.mark.asyncio
    async def test_core_module_helper_branches(self, make_paper, tmp_path) -> None:
        base_options = browser_core.ArxivBrowserOptions(
            config=UserConfig(),
            restore_session=False,
            history_files=[(date(2026, 3, 22), tmp_path / "2026-03-22.txt")],
            current_date_index=2,
            ascii_icons=True,
            services=SimpleNamespace(marker="services"),
        )
        cloned = browser_core._coerce_browser_options(base_options, (), {})
        assert cloned is not base_options
        assert cloned.history_files == base_options.history_files
        assert cloned.history_files is not base_options.history_files

        legacy = browser_core._coerce_browser_options(UserConfig(), (), {})
        assert isinstance(legacy, browser_core.ArxivBrowserOptions)
        assert legacy.config is not None

        with pytest.raises(TypeError):
            browser_core._coerce_browser_options(base_options, (object(),), {})
        with pytest.raises(TypeError):
            browser_core._coerce_browser_options(
                None,
                (UserConfig(),),
                {"config": UserConfig()},
            )
        with pytest.raises(TypeError):
            browser_core._coerce_browser_options(None, tuple(range(7)), {})

        class _Response:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

        class _Client:
            def __init__(self, response: _Response) -> None:
                self.response = response

            async def get(self, *_args, **_kwargs):
                return self.response

        class _TempClient:
            def __init__(self, response: _Response) -> None:
                self.response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *_args, **_kwargs):
                return self.response

        paper = make_paper(arxiv_id="2401.10051", abstract="Fallback abstract.")
        empty_paper = make_paper(arxiv_id="2401.10052", abstract=None, abstract_raw=None)
        long_text = "x" * (browser_content.MAX_PAPER_CONTENT_LENGTH + 10)

        with patch(
            "arxiv_browser.browser.content.asyncio.to_thread",
            new=AsyncMock(return_value=long_text),
        ):
            text = await browser_core._fetch_paper_content_async(
                paper,
                _Client(_Response(200, "<p>x</p>")),
            )
        assert len(text) == browser_content.MAX_PAPER_CONTENT_LENGTH

        with patch(
            "arxiv_browser.browser.content.asyncio.to_thread",
            new=AsyncMock(return_value=""),
        ):
            text = await browser_core._fetch_paper_content_async(
                paper,
                _Client(_Response(404, "")),
            )
        assert text == "Abstract:\nFallback abstract."

        with (
            patch(
                "arxiv_browser.browser.content.asyncio.to_thread",
                new=AsyncMock(return_value=""),
            ),
            patch(
                "arxiv_browser.browser.content.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
        ):
            text = await browser_core._fetch_paper_content_async(empty_paper, None)
        assert text == ""

        with (
            patch(
                "arxiv_browser.browser.content.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
            patch(
                "arxiv_browser.browser.content.asyncio.to_thread",
                new=AsyncMock(side_effect=httpx.HTTPError("boom")),
            ),
        ):
            text = await browser_core._fetch_paper_content_async(paper, None)
        assert text == "Abstract:\nFallback abstract."

        app = _new_app_stub()
        app._ui_refs = None
        assert (
            browser_core.ArxivBrowser._get_cached_widget(app, "search_input", lambda: "resolved")
            == "resolved"
        )

        class _LiveWidget:
            def __init__(self, *, value: str = "", focus_raises: bool = False) -> None:
                self.is_attached = True
                self.value = value
                self.highlighted = None
                self.option_count = 2
                self.focused = False
                self._focus_raises = focus_raises

            def focus(self) -> None:
                if self._focus_raises:
                    raise NoMatches("missing")
                self.focused = True

            def remove_class(self, *_args, **_kwargs) -> None:
                return None

            def add_class(self, *_args, **_kwargs) -> None:
                return None

        live_widget = _LiveWidget(value="seed")
        app._ui_refs = SimpleNamespace(search_input=live_widget)
        resolver = MagicMock(return_value="fresh")
        assert (
            browser_core.ArxivBrowser._get_cached_widget(app, "search_input", resolver)
            is live_widget
        )
        app._ui_refs.search_input = SimpleNamespace(is_attached=False)
        assert (
            browser_core.ArxivBrowser._get_cached_widget(app, "search_input", resolver) == "fresh"
        )
        assert app._ui_refs.search_input == "fresh"

        assert browser_core.ArxivBrowser._is_live_widget(SimpleNamespace(is_attached=True))
        assert not browser_core.ArxivBrowser._is_live_widget(SimpleNamespace(is_attached=False))

        app._refresh_list_view = MagicMock()
        app._update_list_header = MagicMock()
        app._update_status_bar = MagicMock()
        app._update_filter_pills = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._refresh_current_list_item = MagicMock()
        coordinator = browser_core.ArxivBrowser._get_ui_refresh_coordinator(app)
        assert coordinator is app._ui_refresh
        assert browser_core.ArxivBrowser._get_ui_refresh_coordinator(app) is coordinator

        app._history_files = [
            (date(2026, 3, 22), tmp_path / "2026-03-22.txt"),
            (date(2026, 3, 23), tmp_path / "2026-03-23.txt"),
        ]
        app._current_date_index = 1
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app._get_date_navigator_widget = MagicMock(
            return_value=SimpleNamespace(update_dates=AsyncMock(return_value=None))
        )
        browser_core.ArxivBrowser._refresh_date_navigator(app)
        app._get_date_navigator_widget = MagicMock(side_effect=NoMatches("missing"))
        browser_core.ArxivBrowser._refresh_date_navigator(app)

    @pytest.mark.asyncio
    async def test_core_mount_unmount_and_event_branches(self, make_paper) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.10061")

        class _Widget:
            def __init__(self, *, value: str = "", focus_raises: bool = False) -> None:
                self.is_attached = True
                self.value = value
                self.highlighted = 1
                self.option_count = 2
                self.focused = False
                self._focus_raises = focus_raises
                self.updated: list[object] = []

            def focus(self) -> None:
                if self._focus_raises:
                    raise NoMatches("missing")
                self.focused = True

            def remove_class(self, *_args, **_kwargs) -> None:
                return None

            def add_class(self, *_args, **_kwargs) -> None:
                return None

            def update(self, value: object) -> None:
                self.updated.append(value)

        first_refs = SimpleNamespace(
            search_input=_Widget(value=""),
            search_container=_Widget(),
            paper_list=_Widget(),
            list_header=_Widget(),
            details_header=_Widget(),
            status_bar=_Widget(),
            footer=_Widget(),
            date_navigator=_Widget(),
            filter_pill_bar=_Widget(),
            bookmark_bar=_Widget(),
            paper_details=_Widget(),
        )
        app._ui_refs = first_refs
        app._config = _make_app_config(
            config_defaulted=True,
            s2_enabled=True,
            hf_enabled=True,
            session=SessionState(
                scroll_index=3,
                current_filter="graph",
                selected_ids=[paper.arxiv_id],
            ),
        )
        app._restore_session = True
        app._history_files = [
            (date(2026, 3, 22), Path("2026-03-22.txt")),
            (date(2026, 3, 23), Path("2026-03-23.txt")),
        ]
        app._is_history_mode = MagicMock(return_value=True)
        app._fetch_hf_daily = AsyncMock(return_value=None)
        app._update_bookmark_bar = AsyncMock(return_value=None)
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app._apply_filter = MagicMock()
        app._refresh_list_view = MagicMock()
        app._update_header = MagicMock()
        app._update_subtitle = MagicMock()
        app._update_details_header = MagicMock()
        app._notify_watch_list_matches = MagicMock()
        app.call_after_refresh = MagicMock()
        app._get_footer_widget = MagicMock(side_effect=NoMatches("missing"))

        first_client = SimpleNamespace(aclose=AsyncMock(side_effect=Exception("boom")))
        second_client = SimpleNamespace(aclose=AsyncMock(side_effect=Exception("boom")))
        with patch(
            "arxiv_browser.browser.core.httpx.AsyncClient",
            side_effect=[first_client, second_client],
        ):
            browser_core.ArxivBrowser.on_mount(app)

        assert app.call_after_refresh.called
        assert app._apply_filter.call_args.args == ("graph",)
        assert first_refs.search_input.value == "graph"
        assert first_refs.paper_list.highlighted == 1
        assert first_refs.paper_list.focused is True

        second_refs = SimpleNamespace(
            search_input=_Widget(value=""),
            search_container=_Widget(),
            paper_list=_Widget(focus_raises=True),
            list_header=_Widget(),
            details_header=_Widget(),
            status_bar=_Widget(),
            footer=_Widget(),
            date_navigator=_Widget(),
            filter_pill_bar=_Widget(),
            bookmark_bar=_Widget(),
            paper_details=_Widget(),
        )
        app._ui_refs = second_refs
        app._config.session = SessionState(
            scroll_index=0,
            current_filter="",
            selected_ids=[],
        )
        app._refresh_list_view = MagicMock()
        app._apply_filter = MagicMock()
        with patch("arxiv_browser.browser.core.httpx.AsyncClient", return_value=second_client):
            browser_core.ArxivBrowser.on_mount(app)
        assert second_refs.paper_list.highlighted == 0

        search_timer = _DummyTimer()
        detail_timer = _DummyTimer()
        badge_timer = _DummyTimer()
        sort_timer = _DummyTimer()
        app._search_timer = search_timer
        app._detail_timer = detail_timer
        app._badge_timer = badge_timer
        app._sort_refresh_timer = sort_timer

        class _TaskStub:
            def __init__(self) -> None:
                self.cancel = MagicMock()

            def done(self) -> bool:
                return False

        pending_task = _TaskStub()
        app._background_tasks = {pending_task}
        app._tfidf_build_task = object()
        app._save_session_state = MagicMock()
        app._cancel_dataset_tasks = MagicMock()
        app._ui_refs = SimpleNamespace(reset=MagicMock())
        with patch(
            "arxiv_browser.browser.core.asyncio.wait",
            new=AsyncMock(return_value=(set(), {pending_task})),
        ):
            await browser_core.ArxivBrowser.on_unmount(app)
        assert search_timer.stopped is True
        assert detail_timer.stopped is True
        assert badge_timer.stopped is True
        assert sort_timer.stopped is True
        pending_task.cancel.assert_called_once()
        assert second_client.aclose.await_count == 1
        app._ui_refs.reset.assert_called_once()

        app._get_search_container_widget = MagicMock(
            return_value=SimpleNamespace(remove_class=MagicMock())
        )
        app._get_paper_list_widget = MagicMock(return_value=SimpleNamespace(focus=MagicMock()))
        app._apply_filter = MagicMock()
        browser_core.ArxivBrowser.on_search_submitted(app, SimpleNamespace(value="graph"))
        app._apply_filter.assert_called_with("graph")

        app._search_timer = _DummyTimer()
        app.set_timer = MagicMock(return_value=_DummyTimer())
        browser_core.ArxivBrowser.on_search_changed(app, SimpleNamespace(value="query"))
        assert app._search_timer is not None

        app._history_files = [Path("one"), Path("two")]
        app._set_history_index = MagicMock()
        browser_core.ArxivBrowser.on_date_jump(app, SimpleNamespace(index=1))
        app._set_history_index.assert_called_with(1)

        app.action_prev_date = MagicMock()
        app.action_next_date = MagicMock()
        browser_core.ArxivBrowser.on_date_navigate(app, SimpleNamespace(direction=1))
        browser_core.ArxivBrowser.on_date_navigate(app, SimpleNamespace(direction=-1))
        app.action_prev_date.assert_called_once()
        app.action_next_date.assert_called_once()

        app._pending_query = "seed"
        app._apply_filter = MagicMock()
        browser_core.ArxivBrowser.on_remove_watch_filter(app, SimpleNamespace())
        app._apply_filter.assert_called_with("seed")

        app._pending_mark_action = "set"
        app._set_mark = MagicMock()
        key_event = SimpleNamespace(
            key="a",
            prevent_default=MagicMock(),
            stop=MagicMock(),
        )
        browser_core.ArxivBrowser.on_key(app, key_event)
        app._set_mark.assert_called_with("a")
        assert key_event.prevent_default.called
        assert key_event.stop.called

        app._pending_mark_action = "goto"
        app._goto_mark = MagicMock()
        browser_core.ArxivBrowser.on_key(app, key_event)
        app._goto_mark.assert_called_with("a")
        app._pending_mark_action = None
        browser_core.ArxivBrowser.on_key(app, key_event)

        app._build_help_sections = MagicMock(return_value=[("shortcuts", [])])
        app.push_screen = MagicMock()
        browser_core.ArxivBrowser.action_show_search_syntax(app)
        assert app.push_screen.called

    @pytest.mark.asyncio
    async def test_track_and_cancel_helpers_and_task_done(self) -> None:
        async def noop():
            return None

        app = _new_app_stub()
        app._track_task = browser_core.ArxivBrowser._track_task.__get__(
            app, browser_core.ArxivBrowser
        )
        app._track_dataset_task = browser_core.ArxivBrowser._track_dataset_task.__get__(
            app, browser_core.ArxivBrowser
        )
        task = app._track_task(noop())
        assert task in app._background_tasks
        await task
        assert task not in app._background_tasks

        task2 = app._track_task(noop(), dataset_bound=True)
        assert task2 in app._background_tasks
        assert task2 in app._dataset_tasks
        await task2
        assert task2 not in app._background_tasks
        assert task2 not in app._dataset_tasks

        app._track_task = MagicMock(return_value=asyncio.create_task(noop()))
        pending_coro = noop()
        tracked = app._track_dataset_task(pending_coro)
        pending_coro.close()
        assert tracked in app._dataset_tasks
        await tracked
        assert tracked not in app._dataset_tasks

        pending = asyncio.create_task(asyncio.sleep(10))
        done = asyncio.create_task(noop())
        await asyncio.sleep(0)
        app._dataset_tasks = {pending, done}
        app._cancel_dataset_tasks()
        await asyncio.sleep(0)
        assert app._dataset_tasks == set()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        app.notify = MagicMock()
        exc_task = MagicMock()
        exc_task.cancelled.return_value = False
        exc_task.exception.return_value = RuntimeError("boom")
        app._on_task_done(exc_task)
        assert app.notify.called

        app.notify.reset_mock()
        exc_task.cancelled.return_value = True
        app._on_task_done(exc_task)
        app.notify.assert_not_called()

        app.notify = MagicMock(side_effect=RuntimeError("boom"))
        exc_task.cancelled.return_value = False
        app._on_task_done(exc_task)

        app.notify = MagicMock()
        app._shutting_down = True
        exc_task.cancelled.return_value = False
        app._on_task_done(exc_task)
        app._shutting_down = False
        app.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_abstract_and_detail_helpers(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.10001", abstract=None, abstract_raw="x^2")
        other = make_paper(arxiv_id="2401.10002", abstract=None, abstract_raw="y^2")
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())

        app._schedule_abstract_load(paper)
        assert paper.arxiv_id in app._abstract_loading
        app._schedule_abstract_load(paper)
        assert len(app._abstract_loading) == 1

        app._abstract_loading = {str(i) for i in range(MAX_ABSTRACT_LOADS)}
        app._schedule_abstract_load(other)
        assert other.arxiv_id in app._abstract_pending_ids
        assert next(iter(app._abstract_queue)) == other

        app._abstract_loading = set()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._drain_abstract_queue()
        assert other.arxiv_id in app._abstract_loading

        app._abstract_cache = {paper.arxiv_id: "cached"}
        assert app._get_abstract_text(paper, allow_async=False) == "cached"

        fresh = make_paper(arxiv_id="2401.10003", abstract="already cleaned", abstract_raw="raw")
        assert app._get_abstract_text(fresh, allow_async=False) == "already cleaned"

        blank = make_paper(arxiv_id="2401.10004", abstract=None, abstract_raw=None)
        assert app._get_abstract_text(blank, allow_async=False) == ""

        latex = make_paper(arxiv_id="2401.10005", abstract=None, abstract_raw="\\alpha")
        queued = make_paper(arxiv_id="2401.10006", abstract=None, abstract_raw="\\beta")
        app._schedule_abstract_load = MagicMock()
        assert app._get_abstract_text(queued, allow_async=True) is None
        app._schedule_abstract_load.assert_called_once_with(queued)
        with patch("arxiv_browser.browser.chrome.clean_latex", return_value="alpha"):
            assert app._get_abstract_text(latex, allow_async=False) == "alpha"
        assert app._get_abstract_text(latex, allow_async=True) == "alpha"

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._update_abstract_display = MagicMock()
        with patch(
            "arxiv_browser.browser.chrome.asyncio.to_thread", new=AsyncMock(return_value="beta")
        ):
            await app._load_abstract_async(latex)
        assert app._abstract_cache[latex.arxiv_id] == "beta"
        app._update_abstract_display.assert_called_with(latex.arxiv_id)

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        with (
            patch(
                "arxiv_browser.browser.chrome.asyncio.to_thread",
                new=AsyncMock(side_effect=asyncio.CancelledError()),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await app._load_abstract_async(queued)

        app._update_abstract_display.reset_mock()
        with patch(
            "arxiv_browser.browser.chrome.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await app._load_abstract_async(other)
        assert other.arxiv_id not in app._abstract_loading

        details = SimpleNamespace(
            paper=SimpleNamespace(arxiv_id=latex.arxiv_id),
            update_state=MagicMock(),
        )
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._build_detail_state = MagicMock(return_value=object())
        app._update_abstract_display = ArxivBrowser._update_abstract_display.__get__(
            app, ArxivBrowser
        )
        app._show_abstract_preview = False
        app._abstract_cache[latex.arxiv_id] = "beta"
        app._update_abstract_display(latex.arxiv_id)
        details.update_state.assert_called_once()

        app._show_abstract_preview = True
        app._update_option_for_paper = MagicMock()
        app._update_abstract_display(latex.arxiv_id)
        app._update_option_for_paper.assert_called_with(latex.arxiv_id)

        app.filtered_papers = [latex]
        app._get_paper_details_widget = MagicMock(return_value=details)
        details.update_state.reset_mock()
        app.on_paper_selected(SimpleNamespace(option_index=0))
        assert details.update_state.called

        app._pending_detail_paper = latex
        app._pending_detail_started_at = None
        app._detail_timer = _DummyTimer()
        app._get_current_paper = MagicMock(return_value=latex)
        app._get_abstract_text = MagicMock(return_value="abstract")
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._debounced_detail_update()
        assert app._pending_detail_paper is None
