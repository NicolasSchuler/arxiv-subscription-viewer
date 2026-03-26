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
import arxiv_browser.app as app_mod
import arxiv_browser.browser.core as browser_core
import arxiv_browser.browser.discovery as discovery
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


class _OptionListStub:
    def __init__(self, highlighted: int | None = None, option_count: int = 0) -> None:
        self.highlighted = highlighted
        self.highlighted_child = None
        self.option_count = option_count
        self.focused = False
        self.options: list[object] = []
        self.replaced: list[tuple[int, str]] = []
        self.classes: set[str] = set()

    def clear_options(self) -> None:
        self.options.clear()
        self.option_count = 0

    def add_options(self, options: list[object]) -> None:
        self.options.extend(options)
        self.option_count = len(self.options)

    def add_option(self, option: object) -> None:
        self.options.append(option)
        self.option_count = len(self.options)

    def replace_option_prompt_at_index(self, index: int, markup: str) -> None:
        self.replaced.append((index, markup))

    def focus(self) -> None:
        self.focused = True

    def remove_class(self, class_name: str) -> None:
        self.classes.discard(class_name)

    def add_class(self, class_name: str) -> None:
        self.classes.add(class_name)

    def has_class(self, class_name: str) -> bool:
        return class_name in self.classes


class TestAppActionBranches:
    @pytest.mark.asyncio
    async def test_ui_action_branches_cover_app_wrappers(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50001")
        app._config = _make_app_config(
            s2_enabled=False,
            hf_enabled=False,
            s2_cache_ttl_days=7,
            hf_cache_ttl_hours=6,
            theme_name="not-a-theme",
            collections=[PaperCollection(name="Reading", paper_ids=["2401.50001"])],
        )
        app._http_client = object()
        app._s2_db_path = tmp_path / "s2.db"
        app._hf_db_path = tmp_path / "hf.db"
        app._s2_active = False
        app._hf_active = False
        app._s2_loading = set()
        app._hf_loading = False
        app._s2_cache = {}
        app._hf_cache = {}
        app._papers_by_id = {paper.arxiv_id: paper}
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_target_papers = MagicMock(return_value=[paper])
        app._get_ui_refresh_coordinator.return_value.refresh_detail_pane = MagicMock()
        app._get_paper_details_widget = MagicMock(
            return_value=SimpleNamespace(clear_cache=MagicMock())
        )
        app._apply_theme_overrides = MagicMock()
        app._apply_category_overrides = MagicMock()
        app._show_recommendations = MagicMock()
        app._show_citation_graph = AsyncMock(return_value=None)
        app.push_screen = MagicMock()

        with patch("arxiv_browser.app.save_config", return_value=False):
            app.action_toggle_s2()
        assert app._config.s2_enabled is False
        assert "Failed to save Semantic Scholar setting" in app.notify.call_args[0][0]

        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_toggle_s2()
        assert app._s2_active is True
        assert app._config.s2_enabled is True

        app._get_current_paper = MagicMock(return_value=None)
        await app.action_fetch_s2()
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_loading = {paper.arxiv_id}
        await app.action_fetch_s2()
        app._s2_loading = set()
        app._s2_cache = {paper.arxiv_id: object()}
        await app.action_fetch_s2()
        app._s2_cache = {}
        app._s2_active = True

        def _track_raises(coro):
            coro.close()
            raise OSError("boom")

        app._track_dataset_task = MagicMock(side_effect=_track_raises)
        with pytest.raises(OSError):
            await app.action_fetch_s2()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())

        app._http_client = None
        await app._fetch_s2_paper_async(paper.arxiv_id)
        app._http_client = object()
        app._services = SimpleNamespace(
            enrichment=SimpleNamespace(
                load_or_fetch_s2_paper=AsyncMock(
                    return_value=SimpleNamespace(
                        state="not_found",
                        paper=None,
                        complete=True,
                        from_cache=False,
                    )
                )
            )
        )
        await app._fetch_s2_paper_async(paper.arxiv_id)
        app._services.enrichment.load_or_fetch_s2_paper = AsyncMock(
            return_value=SimpleNamespace(
                state="found",
                paper=SimpleNamespace(arxiv_id=paper.arxiv_id, s2_paper_id="s2:1"),
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_s2_paper_async(paper.arxiv_id)
        app._services.enrichment.load_or_fetch_s2_paper = AsyncMock(
            return_value=SimpleNamespace(
                state="unavailable",
                paper=None,
                complete=False,
                from_cache=False,
            )
        )
        await app._fetch_s2_paper_async(paper.arxiv_id)

        with patch("arxiv_browser.app.save_config", return_value=False):
            await app.action_toggle_hf()
        assert app._config.hf_enabled is False

        app._config.hf_enabled = True
        app._hf_active = False
        app._hf_cache = {}
        with patch("arxiv_browser.app.save_config", return_value=True):
            await app.action_toggle_hf()
        assert app._hf_active is True
        app._hf_cache = {paper.arxiv_id: object()}
        await app.action_toggle_hf()

        app._hf_loading = False
        await app._fetch_hf_daily()
        app._hf_loading = True
        await app._fetch_hf_daily()
        app._hf_loading = True
        app._http_client = None
        await app._fetch_hf_daily_async()
        app._http_client = object()
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="empty",
                papers=[],
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="found",
                papers=[SimpleNamespace(arxiv_id=paper.arxiv_id)],
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="unavailable",
                papers=[],
                complete=False,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()

        app._get_current_paper = MagicMock(return_value=None)
        app.action_show_similar()
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_active = False
        app.action_show_similar()
        app._s2_active = True
        app.action_show_similar()
        app._s2_cache = {}
        app.action_citation_graph()
        app._s2_cache = {paper.arxiv_id: SimpleNamespace(s2_paper_id="s2:1")}
        app.action_citation_graph()

        captured = {}
        app.push_screen = lambda modal, cb: captured.update(modal=modal, callback=cb)
        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_cycle_theme()
            app.action_toggle_sections()
            captured["callback"](None)
            app.action_toggle_sections()
            captured["callback"](["abstract"])

            app._config.collections = [PaperCollection(name="Reading", paper_ids=["2401.50001"])]
            app._get_target_papers = MagicMock(return_value=[paper])
            app.action_collections()
            captured["callback"]("save")

        captured_add = {}
        app.push_screen = lambda modal, cb: captured_add.update(modal=modal, callback=cb)
        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_add_to_collection()
            captured_add["callback"]("Reading")


class TestAppHelperCoverage:
    def test_list_message_and_paper_content_branches(self) -> None:
        assert "No papers match your search" in app_mod.build_list_empty_message(
            query="x",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No API results on this page" in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=True,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No watched papers found" in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=True,
            history_mode=False,
        )
        assert "No papers available for this date" in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=True,
        )
        assert "No papers available." in app_mod.build_list_empty_message(
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
        long_text = "x" * (app_mod.MAX_PAPER_CONTENT_LENGTH + 10)

        with patch("arxiv_browser.app.extract_text_from_html", return_value=long_text):
            text = asyncio.run(
                app_mod._fetch_paper_content_async(paper, _Client(_Response(200, "<p>x</p>")))
            )
        assert len(text) == app_mod.MAX_PAPER_CONTENT_LENGTH

        with patch("arxiv_browser.app.extract_text_from_html", return_value=""):
            text = asyncio.run(
                app_mod._fetch_paper_content_async(paper, _Client(_Response(404, "")))
            )
        assert text == "Abstract:\nFallback abstract."

        with (
            patch("arxiv_browser.app.extract_text_from_html", return_value=""),
            patch(
                "arxiv_browser.app.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
        ):
            text = asyncio.run(app_mod._fetch_paper_content_async(empty_paper, None))
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
        long_text = "x" * (browser_core.MAX_PAPER_CONTENT_LENGTH + 10)

        with patch(
            "arxiv_browser.browser.core.asyncio.to_thread",
            new=AsyncMock(return_value=long_text),
        ):
            text = await browser_core._fetch_paper_content_async(
                paper,
                _Client(_Response(200, "<p>x</p>")),
            )
        assert len(text) == browser_core.MAX_PAPER_CONTENT_LENGTH

        with patch(
            "arxiv_browser.browser.core.asyncio.to_thread",
            new=AsyncMock(return_value=""),
        ):
            text = await browser_core._fetch_paper_content_async(
                paper,
                _Client(_Response(404, "")),
            )
        assert text == "Abstract:\nFallback abstract."

        with (
            patch(
                "arxiv_browser.browser.core.asyncio.to_thread",
                new=AsyncMock(return_value=""),
            ),
            patch(
                "arxiv_browser.browser.core.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
        ):
            text = await browser_core._fetch_paper_content_async(empty_paper, None)
        assert text == ""

        with (
            patch(
                "arxiv_browser.browser.core.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
            patch(
                "arxiv_browser.browser.core.asyncio.to_thread",
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
            session=app_mod.SessionState(
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
        app._config.session = app_mod.SessionState(
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

        app._abstract_loading = {str(i) for i in range(app_mod.MAX_ABSTRACT_LOADS)}
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
        with patch("arxiv_browser.app.clean_latex", return_value="alpha"):
            assert app._get_abstract_text(latex, allow_async=False) == "alpha"
        assert app._get_abstract_text(latex, allow_async=True) == "alpha"

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._update_abstract_display = MagicMock()
        with patch("arxiv_browser.app.asyncio.to_thread", new=AsyncMock(return_value="beta")):
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
            "arxiv_browser.app.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            await app._load_abstract_async(other)
        assert other.arxiv_id not in app._abstract_loading

        details = SimpleNamespace(
            paper=SimpleNamespace(arxiv_id=latex.arxiv_id),
            update_state=MagicMock(),
        )
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._build_detail_state = MagicMock(return_value=object())
        app._update_abstract_display = app_mod.ArxivBrowser._update_abstract_display.__get__(
            app, app_mod.ArxivBrowser
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
                paper1.arxiv_id: app_mod.PaperMetadata(
                    arxiv_id=paper1.arxiv_id, tags=["shared"], starred=True
                ),
                paper2.arxiv_id: app_mod.PaperMetadata(
                    arxiv_id=paper2.arxiv_id, tags=["shared"], starred=False
                ),
            },
            bookmarks=[app_mod.SearchBookmark(name="Saved", query="graph")],
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
        app._sort_index = app_mod.SORT_OPTIONS.index("title")

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

        app._apply_to_selected = app_mod.ArxivBrowser._apply_to_selected.__get__(
            app, app_mod.ArxivBrowser
        )
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
        app._compute_watched_papers = app_mod.ArxivBrowser._compute_watched_papers.__get__(
            app, app_mod.ArxivBrowser
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
        with patch("arxiv_browser.app.build_daily_digest", return_value="Digest"):
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
        with patch("arxiv_browser.app.parse_arxiv_file", side_effect=OSError("boom")):
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
        with patch("arxiv_browser.app.parse_arxiv_file", return_value=[paper]):
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
            download_pdf=AsyncMock(return_value=True),
        )
        app._get_services = MagicMock(return_value=SimpleNamespace(download=download_service))
        assert await app._download_pdf_async(paper, client=object()) is True
        download_service.download_pdf = AsyncMock(return_value=False)
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
                paper.arxiv_id: app_mod.PaperMetadata(
                    arxiv_id=paper.arxiv_id,
                    tags=["alpha"],
                ),
                other.arxiv_id: app_mod.PaperMetadata(
                    arxiv_id=other.arxiv_id,
                    tags=["alpha", "beta"],
                ),
            },
            marks={},
            bookmarks=[app_mod.SearchBookmark(name="Saved", query="graph")],
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
        app._refresh_list_view = app_mod.ArxivBrowser._refresh_list_view.__get__(
            app, app_mod.ArxivBrowser
        )
        app._resolve_visible_index = MagicMock(return_value=1)
        app._goto_mark = app_mod.ArxivBrowser._goto_mark.__get__(app, app_mod.ArxivBrowser)
        app._bulk_edit_tags = app_mod.ArxivBrowser._bulk_edit_tags.__get__(
            app, app_mod.ArxivBrowser
        )
        app.action_toggle_preview = app_mod.ArxivBrowser.action_toggle_preview.__get__(
            app, app_mod.ArxivBrowser
        )
        app.action_toggle_detail_mode = app_mod.ArxivBrowser.action_toggle_detail_mode.__get__(
            app, app_mod.ArxivBrowser
        )
        app.action_start_mark = app_mod.ArxivBrowser.action_start_mark.__get__(
            app, app_mod.ArxivBrowser
        )
        app.action_start_goto_mark = app_mod.ArxivBrowser.action_start_goto_mark.__get__(
            app, app_mod.ArxivBrowser
        )
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
        with patch("arxiv_browser.app.save_config", return_value=True):
            app._save_session_state()
        assert app._config.session.scroll_index == 1
        assert app._config.session.current_filter == "graph"
        assert app._config.session.selected_ids == [paper.arxiv_id]

        app._in_arxiv_api_mode = False
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        with patch("arxiv_browser.app.save_config", return_value=False):
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
        callback(["alpha", "gamma"])
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

        app._sort_index = app_mod.SORT_OPTIONS.index("date")
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
        app._sort_index = app_mod.SORT_OPTIONS.index("citations")
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

        app._sort_index = app_mod.SORT_OPTIONS.index("trending")
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
                paper1.arxiv_id: app_mod.PaperMetadata(
                    arxiv_id=paper1.arxiv_id, tags=["alpha"], starred=True
                ),
                paper2.arxiv_id: app_mod.PaperMetadata(
                    arxiv_id=paper2.arxiv_id, tags=[], starred=False
                ),
            },
            watch_list=[
                SimpleNamespace(pattern="graph", match_type="keyword", case_sensitive=False)
            ],
            marks={"a": paper1.arxiv_id},
            bookmarks=[app_mod.SearchBookmark(name="Saved", query="graph")],
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
        app._sort_index = app_mod.SORT_OPTIONS.index("title")
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
        app._get_search_container_widget = MagicMock(
            return_value=SimpleNamespace(has_class=MagicMock(return_value=True))
        )
        app._get_paper_details_widget = MagicMock(
            return_value=SimpleNamespace(paper=paper1, update_state=MagicMock())
        )
        app._update_option_for_paper = MagicMock()
        app.notify = MagicMock()
        return app, paper1, paper2, theme_runtime

    def test_chrome_theme_and_state_builders(self, make_paper) -> None:
        app, paper1, _paper2, theme_runtime = self._build_chrome_state_app(make_paper)
        with (
            patch("arxiv_browser.app.build_theme_runtime", return_value=theme_runtime),
            patch("arxiv_browser.app.format_categories.cache_clear") as cache_clear,
        ):
            app._apply_category_overrides()
        assert app._theme_runtime is theme_runtime
        cache_clear.assert_called_once()
        app._config.theme = {}
        with patch("arxiv_browser.app.build_theme_runtime", return_value=theme_runtime):
            app._apply_theme_overrides()
        app._config.theme = {"accent": "#fff"}
        app.register_theme = MagicMock(side_effect=Exception("boom"))
        with patch("arxiv_browser.app.build_theme_runtime", return_value=theme_runtime):
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
        app._save_config_or_warn = app_mod.ArxivBrowser._save_config_or_warn.__get__(
            app, app_mod.ArxivBrowser
        )
        with patch("arxiv_browser.app.save_config", return_value=False):
            assert app._save_config_or_warn("theme preference") is False
        with patch("arxiv_browser.app.save_config", return_value=True):
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
        with patch("arxiv_browser.app._resolve_llm_command", return_value=""):
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
            paper1.arxiv_id: app_mod.PaperMetadata(arxiv_id=paper1.arxiv_id, starred=True)
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
        with patch("arxiv_browser.app._resolve_llm_command", return_value="llm {prompt}"):
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
            app._update_footer = app_mod.ArxivBrowser._update_footer.__get__(
                app, app_mod.ArxivBrowser
            )
            app._update_status_bar = app_mod.ArxivBrowser._update_status_bar.__get__(
                app, app_mod.ArxivBrowser
            )
            assert app._get_footer_bindings() == [("", "search")]
            app._get_search_container_widget = MagicMock(
                return_value=SimpleNamespace(has_class=MagicMock(return_value=False))
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
        with patch("arxiv_browser.app.save_config", return_value=True):
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


class TestDiscoveryMixinCoverage:
    @pytest.mark.asyncio
    async def test_version_check_and_page_navigation_branches(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.70001")
        app._config = _make_app_config(
            paper_metadata={
                paper.arxiv_id: app_mod.PaperMetadata(
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
        with patch.object(app_mod, "parse_arxiv_version_map", side_effect=ValueError("bad xml")):
            await app._check_versions_async({paper.arxiv_id})
        assert "up to date" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(app_mod, "apply_version_updates", side_effect=RuntimeError("boom")):
            await app._check_versions_async({paper.arxiv_id})
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(app_mod, "apply_version_updates", side_effect=Exception("boom")):
            await app._check_versions_async({paper.arxiv_id})
        assert app.notify.call_args.kwargs["severity"] == "error"

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
        with patch.object(app_mod, "RecommendationsScreen", return_value="screen"):
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
        with patch.object(app_mod, "RecommendationsScreen", return_value="screen"):
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
        with patch.object(app_mod, "CitationGraphScreen", return_value="graph-screen"):
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
            patch.object(app_mod, "fetch_s2_references", new=AsyncMock(return_value=(refs, False))),
            patch.object(app_mod, "fetch_s2_citations", new=AsyncMock(return_value=(cites, True))),
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
        with patch.object(app_mod, "RecommendationsScreen", return_value="screen"):
            await app._show_s2_recommendations(paper)
        assert app.push_screen.call_args.args[0] == "screen"

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(side_effect=httpx.HTTPError("boom"))
        await app._show_citation_graph(paper.arxiv_id, paper.title)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(return_value=([], [citation_entry]))
        with patch.object(app_mod, "CitationGraphScreen", return_value="graph-screen"):
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
        assert await discovery.DiscoveryMixin._call_auto_tag_llm(app, paper, ["existing"]) is None

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
