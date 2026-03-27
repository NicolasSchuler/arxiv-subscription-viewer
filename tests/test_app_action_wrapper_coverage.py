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
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionsModal,
    CollectionViewModal,
)
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, UserConfig
from arxiv_browser.services import enrichment_service as enrich
from tests.support import canonical_exports as app_mod
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

        with patch_save_config(return_value=False):
            app.action_toggle_s2()
        assert app._config.s2_enabled is False
        assert "Failed to save Semantic Scholar setting" in app.notify.call_args[0][0]

        with patch_save_config(return_value=True):
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

        with patch_save_config(return_value=False):
            await app.action_toggle_hf()
        assert app._config.hf_enabled is False

        app._config.hf_enabled = True
        app._hf_active = False
        app._hf_cache = {}
        with patch_save_config(return_value=True):
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
        with patch_save_config(return_value=True):
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
        with patch_save_config(return_value=True):
            app.action_add_to_collection()
            captured_add["callback"]("Reading")
