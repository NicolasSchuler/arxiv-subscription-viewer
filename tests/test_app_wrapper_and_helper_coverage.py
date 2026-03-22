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
    async def test_track_and_cancel_helpers_and_task_done(self) -> None:
        async def noop():
            return None

        app = _new_app_stub()
        app._track_task = app_mod.ArxivBrowser._track_task.__get__(app, app_mod.ArxivBrowser)
        app._track_dataset_task = app_mod.ArxivBrowser._track_dataset_task.__get__(
            app, app_mod.ArxivBrowser
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
        app._schedule_abstract_load = MagicMock()
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
