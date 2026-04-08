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

import arxiv_browser.actions.ui_actions as ui_actions
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
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, PaperMetadata, UserConfig
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
from tests.support.patch_helpers import patch_save_config


class TestUiActionCoverage:
    @staticmethod
    def _build_ui_action_app(make_paper, tmp_path):
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.60001")
        other = make_paper(arxiv_id="2401.60002")
        s2_paper = SimpleNamespace(
            arxiv_id=paper.arxiv_id,
            s2_paper_id="s2:60001",
            citation_count=5,
            influential_citation_count=1,
            tldr="summary",
            fields_of_study=(),
            year=2024,
            url="https://example.com/s2",
            title="S2 paper",
            abstract="S2 abstract",
        )
        app._config = _make_app_config(
            s2_enabled=False,
            hf_enabled=False,
            s2_cache_ttl_days=7,
            hf_cache_ttl_hours=6,
        )
        app._s2_db_path = tmp_path / "s2.db"
        app._hf_db_path = tmp_path / "hf.db"
        app._http_client = object()
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._s2_active = False
        app._hf_active = False
        app._s2_loading = set()
        app._hf_loading = False
        app._s2_cache = {}
        app._hf_cache = {}
        app._s2_api_error = False
        app._hf_api_error = False
        app._version_checking = False
        app._version_progress = None
        app._version_updates = {}
        app._update_status_bar = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._get_ui_refresh_coordinator = MagicMock(
            return_value=SimpleNamespace(refresh_detail_pane=MagicMock())
        )
        app._get_current_paper = MagicMock(return_value=paper)
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._show_recommendations = MagicMock()
        app._show_citation_graph = AsyncMock(return_value=None)
        app._fetch_hf_daily = AsyncMock(return_value=None)
        app.action_exit_arxiv_search_mode = MagicMock()
        app.action_toggle_s2 = MagicMock()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(),
                    load_or_fetch_hf_daily=AsyncMock(),
                )
            )
        )
        return app, paper, other, s2_paper

    @pytest.mark.asyncio
    async def test_ctrl_e_dispatch_and_toggle_s2_paths(self, make_paper, tmp_path) -> None:
        app, _selected, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)
        app._in_arxiv_api_mode = True
        ui_actions.action_ctrl_e_dispatch(app)
        app.action_exit_arxiv_search_mode.assert_called_once()
        app._in_arxiv_api_mode = False
        ui_actions.action_ctrl_e_dispatch(app)
        app.action_toggle_s2.assert_called_once()

        app.notify.reset_mock()
        with patch_save_config(return_value=True):
            ui_actions.action_toggle_s2(app)
        assert app._s2_active is True
        assert app._config.s2_enabled is True
        assert "Semantic Scholar enabled" in app.notify.call_args[0][0]
        assert app._mark_badges_dirty.call_args.args == ("s2",)
        assert app._mark_badges_dirty.call_args.kwargs["immediate"] is True

        app.notify.reset_mock()
        with patch_save_config(return_value=False):
            ui_actions.action_toggle_s2(app)
        assert app._s2_active is True
        assert app._config.s2_enabled is True
        assert "Failed to save Semantic Scholar setting" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = True
        with patch_save_config(return_value=True):
            ui_actions.action_toggle_s2(app)
        assert app._s2_active is False
        assert "Semantic Scholar disabled" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_s2_guard_and_queue_paths(self, make_paper, tmp_path) -> None:
        app, paper, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app.notify.reset_mock()
        app._s2_active = False
        await ui_actions.action_fetch_s2(app)
        assert "Semantic Scholar is disabled" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = True
        app._get_current_paper = MagicMock(return_value=None)
        await ui_actions.action_fetch_s2(app)
        assert not app.notify.called

        app._get_current_paper = MagicMock(return_value=paper)
        scheduled_calls = app._track_dataset_task.call_count
        app._s2_loading = {paper.arxiv_id}
        await ui_actions.action_fetch_s2(app)
        assert app._track_dataset_task.call_count == scheduled_calls

        app._s2_loading = set()
        app._s2_cache = {paper.arxiv_id: object()}
        await ui_actions.action_fetch_s2(app)
        assert "already loaded" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_cache = {}
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        await ui_actions.action_fetch_s2(app)
        assert paper.arxiv_id in app._s2_loading
        assert app._get_ui_refresh_coordinator.return_value.refresh_detail_pane.called

    @pytest.mark.asyncio
    async def test_fetch_s2_track_failure_cleans_loading_set(self, make_paper, tmp_path) -> None:
        app, paper, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)
        app._s2_active = True

        def _raise_oserror(coro):
            coro.close()
            raise OSError("boom")

        app._s2_loading = set()
        app._track_dataset_task = MagicMock(side_effect=_raise_oserror)
        with pytest.raises(OSError):
            await ui_actions.action_fetch_s2(app)
        assert paper.arxiv_id not in app._s2_loading

    @pytest.mark.asyncio
    async def test_fetch_s2_async_result_and_error_paths(self, make_paper, tmp_path) -> None:
        app, paper, _other, s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app._s2_loading = {paper.arxiv_id}
        app._http_client = None
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert app._s2_api_error is True
        assert paper.arxiv_id not in app._s2_loading

        app._s2_loading = {paper.arxiv_id}
        app._http_client = object()
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=False,
                            state="unavailable",
                            paper=None,
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._s2_loading = {paper.arxiv_id}
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="not_found",
                            paper=None,
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert "No Semantic Scholar data" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_loading = {paper.arxiv_id}
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="found",
                            paper=s2_paper,
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert app._s2_cache[paper.arxiv_id] is s2_paper
        assert app._mark_badges_dirty.called

        app.notify.reset_mock()
        app._s2_loading = {paper.arxiv_id}
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(side_effect=httpx.HTTPError("boom"))
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert app.notify.call_args.kwargs["severity"] == "error"

        app.notify.reset_mock()
        app._s2_loading = {paper.arxiv_id}
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(side_effect=Exception("boom"))
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert app.notify.call_args.kwargs["severity"] == "error"

    @pytest.mark.asyncio
    async def test_toggle_hf_enable_disable_and_save_failure_paths(
        self, make_paper, tmp_path
    ) -> None:
        app, _selected, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app.notify.reset_mock()
        with patch_save_config(return_value=True):
            await ui_actions.action_toggle_hf(app)
        assert app._hf_active is True
        assert app._fetch_hf_daily.await_count == 1
        assert app._mark_badges_dirty.call_args.args[0] == "hf"

        app.notify.reset_mock()
        app._hf_active = True
        with patch_save_config(return_value=False):
            await ui_actions.action_toggle_hf(app)
        assert app._hf_active is True
        assert "Failed to save HuggingFace setting" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._hf_active = True
        with patch_save_config(return_value=True):
            await ui_actions.action_toggle_hf(app)
        assert app._hf_active is False
        assert "HuggingFace trending disabled" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_guard_and_schedule_paths(self, make_paper, tmp_path) -> None:
        app, _selected, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app._hf_loading = True
        scheduled = app._track_dataset_task.call_count
        await ui_actions._fetch_hf_daily(app)
        assert app._track_dataset_task.call_count == scheduled

        app._hf_loading = False
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        await ui_actions._fetch_hf_daily(app)
        assert app._hf_loading is True
        app._track_dataset_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_schedule_failure_sets_error(self, make_paper, tmp_path) -> None:
        app, _selected, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        def _raise_hf(coro):
            coro.close()
            raise OSError("boom")

        app._hf_loading = False
        app._track_dataset_task = MagicMock(side_effect=_raise_hf)
        await ui_actions._fetch_hf_daily(app)
        assert app._hf_loading is False
        assert "fetch HuggingFace trending data" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_async_no_client_and_unavailable_paths(
        self, make_paper, tmp_path
    ) -> None:
        app, _selected, _other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app.notify.reset_mock()
        app._hf_loading = True
        app._http_client = None
        await ui_actions._fetch_hf_daily_async(app)
        assert app._hf_api_error is True

        app.notify.reset_mock()
        app._hf_loading = True
        app._http_client = object()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_hf_daily=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=False,
                            state="unavailable",
                            papers=[],
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_hf_daily_async(app)
        assert app.notify.call_args.kwargs["severity"] == "error"

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_async_empty_found_and_http_error_paths(
        self, make_paper, tmp_path
    ) -> None:
        app, paper, other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app.notify.reset_mock()
        app._hf_loading = True
        app._http_client = object()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_hf_daily=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="empty",
                            papers=[],
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_hf_daily_async(app)
        assert "No HuggingFace trending data" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._hf_loading = True
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_hf_daily=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="found",
                            papers=[paper, other],
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_hf_daily_async(app)
        assert paper.arxiv_id in app._hf_cache
        assert "paper" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._hf_loading = True
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_hf_daily=AsyncMock(side_effect=httpx.HTTPError("boom"))
                )
            )
        )
        await ui_actions._fetch_hf_daily_async(app)
        assert app.notify.call_args.kwargs["severity"] == "error"

    @pytest.mark.asyncio
    async def test_recommendation_and_version_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.60011")
        other = make_paper(arxiv_id="2401.60012")
        app._config = _make_app_config(
            paper_metadata={
                paper.arxiv_id: PaperMetadata(arxiv_id=paper.arxiv_id, starred=True),
                other.arxiv_id: PaperMetadata(arxiv_id=other.arxiv_id, starred=False),
            }
        )
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._s2_cache = {}
        app._s2_active = False
        app._version_checking = False
        app._version_updates = {}
        app._version_progress = None
        app._in_arxiv_api_mode = True
        app._arxiv_api_fetch_inflight = False
        app._arxiv_search_state = SimpleNamespace(
            start=0,
            max_results=10,
            request=SimpleNamespace(query="graph"),
        )
        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._tfidf_build_task = None
        app._pending_similarity_paper_id = None
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_paper_list_widget = MagicMock(return_value=SimpleNamespace(highlighted=0))
        app._resolve_visible_index = MagicMock(return_value=1)
        app._show_local_recommendations = MagicMock()
        app._show_s2_recommendations = AsyncMock(return_value=None)
        app._show_citation_graph = AsyncMock(return_value=None)
        app._fetch_s2_recommendations_async = AsyncMock(return_value=[])
        app._fetch_citation_graph = AsyncMock(return_value=([], []))
        app._run_arxiv_search = AsyncMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._update_footer = MagicMock()
        app._update_status_bar = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._get_ui_refresh_coordinator = MagicMock(
            return_value=SimpleNamespace(refresh_detail_pane=MagicMock())
        )
        app.notify = MagicMock()
        app.push_screen = MagicMock()

        ui_actions.action_show_similar(app)
        app._show_local_recommendations.assert_called_once_with(paper)

        app._show_local_recommendations.reset_mock()
        app._s2_active = True
        with patch(
            "arxiv_browser.actions.ui_actions.RecommendationSourceModal",
            return_value="source-modal",
        ):
            ui_actions.action_show_similar(app)
        callback = app.push_screen.call_args.kwargs["callback"]
        callback("s2")
        callback("local")
        assert app._show_local_recommendations.called

        app.notify.reset_mock()
        app._get_current_paper = MagicMock(return_value=None)
        ui_actions.action_show_similar(app)
        assert "No paper is selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = False
        app._get_current_paper = MagicMock(return_value=paper)
        ui_actions.action_citation_graph(app)
        assert "Semantic Scholar is disabled" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = True
        app._get_current_paper = MagicMock(return_value=None)
        ui_actions.action_citation_graph(app)
        assert not app.notify.called

        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_cache = {paper.arxiv_id: SimpleNamespace(s2_paper_id="s2:root")}
        ui_actions.action_citation_graph(app)
        assert app._show_citation_graph.call_args.args[0] == "s2:root"

        app._s2_cache = {}
        ui_actions.action_citation_graph(app)
        assert app._show_citation_graph.call_args.args[0] == f"ARXIV:{paper.arxiv_id}"

        await ui_actions.action_check_versions(app)
        assert app._version_checking is True
        assert app._track_dataset_task.called

        app.notify.reset_mock()
        app._version_checking = True
        await ui_actions.action_check_versions(app)
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._version_checking = False
        app._config.paper_metadata = {}
        await ui_actions.action_check_versions(app)
        assert "No starred papers" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.paper_metadata = {
            paper.arxiv_id: PaperMetadata(arxiv_id=paper.arxiv_id, starred=True)
        }
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        await ui_actions.action_check_versions(app)
        assert "Checking 1 starred papers" in app.notify.call_args[0][0]

    def test_palette_help_collection_and_theme_branches(self, make_paper) -> None:
        app = _new_app_stub()
        object.__setattr__(app, "_id", "stub")
        paper = make_paper(arxiv_id="2401.60021")
        other = make_paper(arxiv_id="2401.60022")
        app._config = _make_app_config(
            theme_name="not-a-theme",
            collapsed_sections=["tags"],
            collections=[PaperCollection(name="Reading", paper_ids=[paper.arxiv_id])],
        )
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._build_help_sections = MagicMock(return_value=[("Core", [("Ctrl+x", "X")])])
        app._apply_theme_overrides = MagicMock()
        app._apply_category_overrides = MagicMock()
        app._refresh_list_view = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_status_bar = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._get_paper_details_widget = MagicMock(
            return_value=SimpleNamespace(clear_cache=MagicMock())
        )
        app._get_target_papers = MagicMock(return_value=[paper, other])
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        with patch(
            "arxiv_browser.actions.ui_actions.SectionToggleModal", return_value="sections-modal"
        ):
            ui_actions.action_toggle_sections(app)
        callback = app.push_screen.call_args.args[1]
        callback(None)
        callback(["summary", "tags"])
        assert app._config.collapsed_sections == ["summary", "tags"]
        assert app._save_config_or_warn.called
        assert app._refresh_detail_pane.called

        app.push_screen = MagicMock()
        ui_actions.action_show_help(app)
        assert app.push_screen.called

        app._build_command_palette_commands = MagicMock(
            return_value=[
                SimpleNamespace(action="sample", name="Sample"),
                SimpleNamespace(action="broken", name="Broken"),
                SimpleNamespace(action="explode", name="Explode"),
            ]
        )
        app.action_sample = AsyncMock(return_value=None)
        app.action_broken = MagicMock(side_effect=OSError("boom"))
        app.action_explode = MagicMock(side_effect=Exception("boom"))
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app.push_screen = MagicMock()
        with patch(
            "arxiv_browser.actions.ui_actions.CommandPaletteModal", return_value="palette-modal"
        ):
            ui_actions.action_command_palette(app)
        callback = app.push_screen.call_args.args[1]
        callback(None)
        callback("sample")
        callback("broken")
        callback("explode")
        callback("missing")
        assert app._track_task.called
        assert "failed" in app.notify.call_args[0][0].lower()

        app._config.collections = []
        ui_actions.action_add_to_collection(app)
        assert "No collections" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.collections = [PaperCollection(name="Reading", paper_ids=[paper.arxiv_id])]
        with patch(
            "arxiv_browser.actions.ui_actions.AddToCollectionModal", return_value="add-modal"
        ):
            ui_actions.action_add_to_collection(app)
        callback = app.push_screen.call_args.args[1]
        callback(None)
        callback("Reading")
        assert other.arxiv_id in app._config.collections[0].paper_ids
        assert "Added 1 paper" in app.notify.call_args[0][0]

        app._config.theme_name = "not-a-theme"
        ui_actions.action_cycle_theme(app)
        assert app._config.theme_name in ui_actions.THEME_NAMES
        assert app._save_config_or_warn.call_args.args[0] == "theme preference"

    @pytest.mark.asyncio
    async def test_cache_and_stale_epoch_suppression_edges(self, make_paper, tmp_path) -> None:
        app, paper, other, s2_paper = self._build_ui_action_app(make_paper, tmp_path)

        app.notify.reset_mock()
        app._s2_loading = {paper.arxiv_id}
        app._http_client = object()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="not_found",
                            paper=None,
                            from_cache=True,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        app.notify.assert_not_called()
        assert app._s2_api_error is False

        app.notify.reset_mock()
        app._mark_badges_dirty.reset_mock()
        app._s2_cache = {}
        app._s2_loading = {paper.arxiv_id}
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_s2_paper=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="found",
                            paper=s2_paper,
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_s2_paper_async(app, paper.arxiv_id)
        assert app._s2_cache == {}
        assert paper.arxiv_id in app._s2_loading
        app.notify.assert_not_called()
        app._mark_badges_dirty.assert_not_called()

        app, paper, other, _s2_paper = self._build_ui_action_app(make_paper, tmp_path)
        app.notify.reset_mock()
        app._hf_loading = True
        app._http_client = object()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_hf_daily=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="empty",
                            papers=[],
                            from_cache=True,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_hf_daily_async(app)
        app.notify.assert_not_called()
        assert app._hf_api_error is False

        app.notify.reset_mock()
        app._mark_badges_dirty.reset_mock()
        app._hf_cache = {}
        app._hf_loading = True
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                enrichment=SimpleNamespace(
                    load_or_fetch_hf_daily=AsyncMock(
                        return_value=SimpleNamespace(
                            complete=True,
                            state="found",
                            papers=[paper, other],
                            from_cache=False,
                        )
                    )
                )
            )
        )
        await ui_actions._fetch_hf_daily_async(app)
        assert app._hf_cache == {}
        app.notify.assert_not_called()
        app._mark_badges_dirty.assert_not_called()

    def test_collection_cancel_and_theme_no_matches_edges(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.60031")
        app._config = _make_app_config(
            theme_name="not-a-theme",
            collections=[PaperCollection(name="Reading", paper_ids=[paper.arxiv_id])],
        )
        app._papers_by_id = {paper.arxiv_id: paper}
        app._apply_theme_overrides = MagicMock()
        app._apply_category_overrides = MagicMock()
        app._refresh_list_view = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_status_bar = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._get_paper_details_widget = MagicMock(side_effect=ui_actions.NoMatches("missing"))
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        ui_actions.action_collections(app)
        callback = app.push_screen.call_args.args[1]
        callback(None)
        app._save_config_or_warn.assert_not_called()
        app.notify.assert_not_called()

        ui_actions.action_cycle_theme(app)
        assert app._config.theme_name in ui_actions.THEME_NAMES
        app._refresh_list_view.assert_called_once()
        app._refresh_detail_pane.assert_called_once()
        app._save_config_or_warn.assert_called_once_with("theme preference")
