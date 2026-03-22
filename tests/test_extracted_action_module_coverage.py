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


class TestExternalIoCoverage:
    def test_export_and_import_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.30001")
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app.notify = MagicMock()

        app._get_target_papers = MagicMock(return_value=[])
        io_actions.action_copy_bibtex(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._get_target_papers = MagicMock(return_value=[paper])
        app._copy_to_clipboard = MagicMock(return_value=True)
        io_actions.action_copy_bibtex(app)
        assert "Copied 1 BibTeX entry" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._copy_to_clipboard = MagicMock(return_value=False)
        app._format_paper_as_markdown = MagicMock(return_value="## title")
        io_actions.action_export_markdown(app)
        assert "Failed to copy to clipboard" in app.notify.call_args[0][0]

        app._copy_to_clipboard = MagicMock(return_value=True)
        app._format_paper_for_clipboard = MagicMock(side_effect=["A", "B"])
        app._get_target_papers = MagicMock(return_value=[paper, make_paper(arxiv_id="2401.30002")])
        io_actions.action_copy_selected(app)
        assert "Copied 2 papers to clipboard" in app.notify.call_args[0][0]

        app._export_file_csv = MagicMock()
        io_actions._do_export(app, "file-csv", [paper])
        app._export_file_csv.assert_called_once_with([paper])
        app._export_clipboard_ris = MagicMock()
        io_actions._do_export(app, "clipboard-ris", [paper])
        app._export_clipboard_ris.assert_called_once_with([paper])

        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        json1 = export_dir / "arxiv-2026-01-01_010101.json"
        json1.write_text("{}", encoding="utf-8")
        json2 = export_dir / "arxiv-2026-01-02_010101.json"
        json2.write_text("{}", encoding="utf-8")

        app._get_export_dir = MagicMock(return_value=export_dir)
        app.notify.reset_mock()
        with patch(
            "arxiv_browser.actions.external_io_actions._list_metadata_snapshots", return_value=[]
        ):
            io_actions.action_import_metadata(app)
        assert "No metadata snapshots found" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with (
            patch(
                "arxiv_browser.actions.external_io_actions._list_metadata_snapshots",
                return_value=[json1],
            ),
            patch("arxiv_browser.actions.external_io_actions._import_metadata_file") as import_mock,
        ):
            io_actions.action_import_metadata(app)
        import_mock.assert_called_once_with(app, json1)

        app.push_screen = MagicMock()
        with patch(
            "arxiv_browser.actions.external_io_actions._list_metadata_snapshots",
            return_value=[json1, json2],
        ):
            io_actions.action_import_metadata(app)
        modal = app.push_screen.call_args.args[0]
        callback = app.push_screen.call_args.kwargs["callback"]
        assert modal is not None
        callback(None)
        callback(json2)

        bad_json = export_dir / "arxiv-2026-01-03_010101.json"
        bad_json.write_text("not json", encoding="utf-8")
        app._compute_watched_papers = MagicMock()
        app._refresh_list_view = MagicMock()
        with (
            patch(
                "arxiv_browser.actions.external_io_actions.import_metadata",
                return_value=(1, 0, 0, 0),
            ),
            patch("arxiv_browser.actions.external_io_actions.save_config", return_value=False),
        ):
            io_actions._import_metadata_file(app, bad_json)
        assert (
            app.notify.call_args[0][0].startswith("Import failed")
            or "Imported" in app.notify.call_args[0][0]
        )

    def test_export_and_download_failure_edges(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.30003")
        paper2 = make_paper(arxiv_id="2401.30004")
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app._get_target_papers = MagicMock(return_value=[])
        app._get_export_dir = MagicMock(return_value=tmp_path / "exports")
        app._copy_to_clipboard = MagicMock(return_value=False)
        app._get_abstract_text = MagicMock(return_value="abstract")
        app._http_client = object()
        app._start_downloads = MagicMock()
        app._finish_download_batch = MagicMock()
        app._update_download_progress = MagicMock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app.notify = MagicMock()

        io_actions.action_export_bibtex_file(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app._get_target_papers = MagicMock(return_value=[paper])
        app.notify.reset_mock()
        app._export_to_file = MagicMock()
        io_actions.action_export_metadata(app)
        app._export_to_file.assert_called_once()

        with patch(
            "arxiv_browser.app.write_timestamped_export_file",
            side_effect=OSError("boom"),
        ):
            io_actions._export_to_file(app, "content", "json", "Metadata")
        assert "Failed to export Metadata" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        export_file = tmp_path / "exports" / "metadata.json"
        with patch(
            "arxiv_browser.app.write_timestamped_export_file",
            return_value=export_file,
        ):
            io_actions._export_to_file(app, "content", "json", "Metadata")
        assert str(export_file.resolve()) in app.notify.call_args[0][0]

        app.notify.reset_mock()
        io_actions._export_clipboard_ris(app, [paper])
        assert "Failed to copy to clipboard" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        io_actions._export_clipboard_csv(app, [paper])
        assert "Failed to copy to clipboard" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        io_actions._export_clipboard_mdtable(app, [paper])
        assert "Failed to copy to clipboard" in app.notify.call_args[0][0]

        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        old = export_dir / "arxiv-2026-01-01_010101.json"
        old.write_text("{}", encoding="utf-8")
        newer = export_dir / "arxiv-2026-01-02_010101.json"
        newer.write_text("{}", encoding="utf-8")
        with patch("pathlib.Path.stat", side_effect=OSError("stat")):
            snapshots = io_actions._list_metadata_snapshots(export_dir)
        assert {path.name for path in snapshots} == {old.name, newer.name}

        app._get_target_papers = MagicMock(return_value=[paper, paper2])
        app._download_queue = deque([paper, paper2])
        app._downloading = {paper.arxiv_id}
        io_actions._start_downloads(app)
        assert app._track_task.called
        assert paper2.arxiv_id in app._downloading

        app._download_total = 0
        io_actions._finish_download_batch(app)
        app._download_total = 2
        app._download_results = {paper.arxiv_id: True, paper2.arxiv_id: True}
        app._config.pdf_download_dir = None
        io_actions._finish_download_batch(app)
        assert app._download_total == 0

        app._download_results = {paper.arxiv_id: True, paper2.arxiv_id: False}
        app._download_total = 2
        io_actions._finish_download_batch(app)
        assert app._download_total == 0

        app._download_total = 1
        app._download_results = {}
        app._download_pdf_async = AsyncMock(side_effect=asyncio.CancelledError())
        app._downloading = {paper.arxiv_id}
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(io_actions._process_single_download(app, paper))

        app._download_results = {}
        app._download_total = 1
        app._download_pdf_async = AsyncMock(side_effect=RuntimeError("boom"))
        app._downloading = {paper.arxiv_id}
        asyncio.run(io_actions._process_single_download(app, paper))
        assert app._download_results[paper.arxiv_id] is False

        app._download_results = {}
        app._download_total = 1
        app._download_pdf_async = AsyncMock(side_effect=Exception("boom"))
        app._downloading = {paper.arxiv_id}
        asyncio.run(io_actions._process_single_download(app, paper))
        assert app._download_results[paper.arxiv_id] is False

    def test_open_and_download_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.40001")
        paper2 = make_paper(arxiv_id="2401.40002")
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._safe_browser_open = MagicMock(return_value=True)
        app._ensure_pdf_viewer_trusted = MagicMock(return_value=False)
        app._open_with_viewer = MagicMock(return_value=True)
        app._is_download_batch_active = MagicMock(return_value=False)
        app._start_downloads = MagicMock()
        app._do_start_downloads = MagicMock()
        app._download_pdf_async = AsyncMock(return_value=True)
        app._update_download_progress = MagicMock()
        app._finish_download_batch = MagicMock()

        app._get_target_papers = MagicMock(return_value=[])
        io_actions.action_open_url(app)
        io_actions.action_open_pdf(app)
        io_actions.action_download_pdf(app)

        app._get_target_papers = MagicMock(return_value=[paper])
        io_actions.action_open_url(app)
        io_actions.action_open_pdf(app)
        io_actions.action_download_pdf(app)
        assert app._safe_browser_open.called or app._open_with_viewer.called

        app._config = UserConfig()
        app._config.pdf_viewer = "viewer {url}"
        io_actions._do_open_pdfs(app, [paper, paper2])
        app._ensure_pdf_viewer_trusted.assert_called()

        app._ensure_pdf_viewer_trusted = MagicMock(return_value=True)
        io_actions._do_open_pdfs(app, [paper])
        app._open_with_viewer.assert_called()

        with patch("arxiv_browser.actions.external_io_actions.subprocess.Popen", return_value=None):
            assert io_actions._open_with_viewer(app, "viewer {url}", "https://arxiv.org/abs/x")

        with patch(
            "arxiv_browser.actions.external_io_actions.subprocess.Popen",
            side_effect=OSError("bad"),
        ):
            assert (
                io_actions._open_with_viewer(app, "viewer {url}", "https://arxiv.org/abs/x")
                is False
            )

        app._download_queue = deque([paper, paper2])
        app._downloading = set()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        io_actions._start_downloads(app)
        assert app._track_task.called

        app._download_results = {}
        app._download_total = 2
        app._download_pdf_async = AsyncMock(return_value=True)
        app._downloading = {paper.arxiv_id}
        asyncio.run(io_actions._process_single_download(app, paper))
        assert paper.arxiv_id not in app._downloading

        app._download_pdf_async = AsyncMock(side_effect=RuntimeError("boom"))
        app._downloading = {paper2.arxiv_id}
        asyncio.run(io_actions._process_single_download(app, paper2))
        assert paper2.arxiv_id not in app._downloading

        app._download_results = {"a": True, "b": False}
        app._download_total = 2
        app._update_status_bar = MagicMock()
        io_actions._finish_download_batch(app)
        assert app._download_total == 0


class TestUiActionCoverage:
    @pytest.mark.asyncio
    async def test_toggle_and_fetch_state_branches(self, make_paper, tmp_path) -> None:
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

        app._in_arxiv_api_mode = True
        ui_actions.action_ctrl_e_dispatch(app)
        app.action_exit_arxiv_search_mode.assert_called_once()
        app._in_arxiv_api_mode = False
        ui_actions.action_ctrl_e_dispatch(app)
        app.action_toggle_s2.assert_called_once()

        app.notify.reset_mock()
        with patch("arxiv_browser.app.save_config", return_value=True):
            ui_actions.action_toggle_s2(app)
        assert app._s2_active is True
        assert app._config.s2_enabled is True
        assert "Semantic Scholar enabled" in app.notify.call_args[0][0]
        assert app._mark_badges_dirty.call_args.args == ("s2",)
        assert app._mark_badges_dirty.call_args.kwargs["immediate"] is True

        app.notify.reset_mock()
        with patch("arxiv_browser.app.save_config", return_value=False):
            ui_actions.action_toggle_s2(app)
        assert app._s2_active is True
        assert app._config.s2_enabled is True
        assert "Failed to save Semantic Scholar setting" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = True
        with patch("arxiv_browser.app.save_config", return_value=True):
            ui_actions.action_toggle_s2(app)
        assert app._s2_active is False
        assert "Semantic Scholar disabled" in app.notify.call_args[0][0]

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

        def _raise_oserror(coro):
            coro.close()
            raise OSError("boom")

        app._s2_loading = set()
        app._track_dataset_task = MagicMock(side_effect=_raise_oserror)
        with pytest.raises(OSError):
            await ui_actions.action_fetch_s2(app)
        assert paper.arxiv_id not in app._s2_loading

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

        app.notify.reset_mock()
        with patch("arxiv_browser.app.save_config", return_value=True):
            await ui_actions.action_toggle_hf(app)
        assert app._hf_active is True
        assert app._fetch_hf_daily.await_count == 1
        assert app._mark_badges_dirty.call_args.args[0] == "hf"

        app.notify.reset_mock()
        app._hf_active = True
        with patch("arxiv_browser.app.save_config", return_value=False):
            await ui_actions.action_toggle_hf(app)
        assert app._hf_active is True
        assert "Failed to save HuggingFace setting" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._hf_active = True
        with patch("arxiv_browser.app.save_config", return_value=True):
            await ui_actions.action_toggle_hf(app)
        assert app._hf_active is False
        assert "HuggingFace trending disabled" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._hf_loading = True
        await ui_actions._fetch_hf_daily(app)
        assert app._track_dataset_task.call_count >= 1

        app._hf_loading = False
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        await ui_actions._fetch_hf_daily(app)
        assert app._hf_loading is True

        def _raise_hf(coro):
            coro.close()
            raise OSError("boom")

        app._hf_loading = False
        app._track_dataset_task = MagicMock(side_effect=_raise_hf)
        await ui_actions._fetch_hf_daily(app)
        assert app._hf_loading is False
        assert "fetch HuggingFace trending data" in app.notify.call_args[0][0]

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

        app.notify.reset_mock()
        app._hf_loading = True
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
                paper.arxiv_id: app_mod.PaperMetadata(arxiv_id=paper.arxiv_id, starred=True),
                other.arxiv_id: app_mod.PaperMetadata(arxiv_id=other.arxiv_id, starred=False),
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
        with patch("arxiv_browser.app.RecommendationSourceModal", return_value="source-modal"):
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
            paper.arxiv_id: app_mod.PaperMetadata(arxiv_id=paper.arxiv_id, starred=True)
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

        with patch("arxiv_browser.app.SectionToggleModal", return_value="sections-modal"):
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
        with patch("arxiv_browser.app.CommandPaletteModal", return_value="palette-modal"):
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
        with patch("arxiv_browser.app.AddToCollectionModal", return_value="add-modal"):
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


class TestLlmActionCoverage:
    def test_trust_and_command_resolution_branches(self, tmp_path) -> None:
        app = _new_app_stub()
        app._trust_hash = llm_actions._trust_hash
        app._config = UserConfig()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._config.trusted_llm_command_hashes = []
        app._config.trusted_pdf_viewer_hashes = []

        app._config.llm_command = ""
        app._config.llm_preset = ""
        assert app._is_llm_command_trusted("echo {prompt}") is True

        preset_name = next(iter(llm_actions.LLM_PRESETS))
        app._config.llm_preset = preset_name
        assert app._is_llm_command_trusted(llm_actions.LLM_PRESETS[preset_name]) is True

        app._config.llm_preset = ""
        app._config.llm_command = "echo {prompt}"
        trusted_hash = llm_actions._trust_hash("echo {prompt}")
        app._config.trusted_llm_command_hashes = [trusted_hash]
        assert app._is_llm_command_trusted("echo {prompt}") is True
        app._config.trusted_pdf_viewer_hashes = [llm_actions._trust_hash("viewer {url}")]
        assert app._is_pdf_viewer_trusted("viewer {url}") is True

        app.notify.reset_mock()
        app._config.trusted_llm_command_hashes = []
        with patch("arxiv_browser.app.save_config", return_value=False):
            assert (
                app._remember_trusted_hash(
                    "echo {prompt}", app._config.trusted_llm_command_hashes, "LLM"
                )
                is True
            )
        assert "session only" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        on_trusted = MagicMock()
        request = llm_actions.CommandTrustRequest(
            command_template="viewer {url}",
            title="PDF",
            prompt_heading="Run untrusted custom PDF viewer command?",
            trust_button_label="Open",
            cancel_message="PDF open cancelled",
            trusted_hashes=app._config.trusted_pdf_viewer_hashes,
            on_trusted=on_trusted,
        )
        with patch("arxiv_browser.app.save_config", return_value=True):
            assert app._ensure_command_trusted(request) is False
        callback = app.push_screen.call_args.args[1]
        callback(None)
        assert "PDF open cancelled" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        callback(True)
        on_trusted.assert_called_once()

        app.push_screen = MagicMock(side_effect=ScreenStackError("boom"))
        with patch("arxiv_browser.app.save_config", return_value=True):
            assert app._ensure_command_trusted(request) is False
        assert "could not confirm pdf command trust" in app.notify.call_args[0][0].lower()

        app._is_llm_command_trusted = MagicMock(return_value=True)
        app._ensure_command_trusted = MagicMock(return_value=False)
        assert app._ensure_llm_command_trusted("cmd {prompt}", MagicMock()) is True
        app._is_llm_command_trusted.return_value = False
        assert app._ensure_llm_command_trusted("cmd {prompt}", MagicMock()) is False
        app._is_pdf_viewer_trusted = MagicMock(return_value=True)
        assert app._ensure_pdf_viewer_trusted("viewer {url}", MagicMock()) is True
        app._is_pdf_viewer_trusted.return_value = False
        assert app._ensure_pdf_viewer_trusted("viewer {url}", MagicMock()) is False

        with patch("arxiv_browser.app.get_config_path", return_value=tmp_path / "config.json"):
            app._config.llm_command = ""
            app._config.llm_preset = ""
            assert app._require_llm_command() is None
        assert "Set llm_command or llm_preset" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.llm_preset = "missing"
        assert app._require_llm_command() is None
        assert "Unknown preset" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.llm_command = "echo {prompt} | cat"
        app._config.llm_preset = ""
        app._config.allow_llm_shell_fallback = False
        with patch("arxiv_browser.app.llm_command_requires_shell", return_value=True):
            assert app._require_llm_command() is None
        assert "allow_llm_shell_fallback is disabled" in app.notify.call_args[0][0]

        app._config.llm_command = "echo {prompt}"
        app._config.allow_llm_shell_fallback = True
        with patch("arxiv_browser.app.resolve_provider", return_value="provider"):
            assert app._require_llm_command() == "echo {prompt}"
        assert app._llm_provider == "provider"

    @pytest.mark.asyncio
    async def test_summary_and_chat_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50001")
        app._config = UserConfig(
            llm_command="cmd {prompt}",
            llm_timeout=5,
            llm_prompt_template="",
            research_interests="",
        )
        app._llm_provider = object()
        app._summary_db_path = tmp_path / "summary.db"
        app._summary_loading = set()
        app._paper_summaries = {}
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._relevance_scoring_active = False
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._get_current_paper = MagicMock(return_value=paper)
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._update_abstract_display = MagicMock()
        app._update_footer = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()

        app._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        app._start_summary_flow = MagicMock()
        llm_actions.action_generate_summary(app)
        app._start_summary_flow.assert_called_once_with("cmd {prompt}")

        app._start_summary_flow.reset_mock()
        app._ensure_llm_command_trusted = MagicMock(return_value=False)
        llm_actions.action_generate_summary(app)
        app._start_summary_flow.assert_not_called()

        app._get_current_paper = MagicMock(return_value=None)
        llm_actions._start_summary_flow(app, "cmd {prompt}")
        assert "No paper selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._get_current_paper = MagicMock(return_value=paper)
        app._summary_loading = {paper.arxiv_id}
        llm_actions._start_summary_flow(app, "cmd {prompt}")
        assert "Summary already generating" in app.notify.call_args[0][0]

        app._summary_loading = set()
        app.push_screen = MagicMock()
        llm_actions._start_summary_flow(app, "cmd {prompt}")
        app.notify.reset_mock()
        with patch("arxiv_browser.app._load_summary", return_value="cached"):
            llm_actions._on_summary_mode_selected(app, "default", paper, "cmd {prompt}")
        assert "Summary loaded from cache" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with patch("arxiv_browser.app._load_summary", return_value=None):
            llm_actions._on_summary_mode_selected(app, "quick", paper, "cmd {prompt}")
        assert paper.arxiv_id in app._summary_loading
        assert app._track_dataset_task.called

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._services = SimpleNamespace(
            llm=SimpleNamespace(generate_summary=AsyncMock(return_value=(None, "bad summary")))
        )
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "bad summary" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(return_value=("summary", None))
        with patch(
            "arxiv_browser.actions.llm_actions.asyncio.to_thread", new=AsyncMock(return_value=None)
        ):
            await llm_actions._generate_summary_async(
                app,
                paper,
                "prompt",
                "cmd-hash",
                mode_label="Q",
                use_full_paper_content=False,
            )
        assert app._paper_summaries[paper.arxiv_id] == "summary"
        assert "Summary generated" in app.notify.call_args[0][0]

        app._start_chat_with_paper = MagicMock()
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        llm_actions.action_chat_with_paper(app)
        app._start_chat_with_paper.assert_called_once()

        app._start_chat_with_paper = llm_actions._start_chat_with_paper.__get__(
            app, app_mod.ArxivBrowser
        )
        app._get_current_paper = MagicMock(return_value=None)
        llm_actions._start_chat_with_paper(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._get_current_paper = MagicMock(return_value=paper)
        app._llm_provider = None
        llm_actions._start_chat_with_paper(app)

        app._llm_provider = object()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        llm_actions._start_chat_with_paper(app)
        assert app._track_dataset_task.called

        app._fetch_paper_content_async = AsyncMock(return_value="content")
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app.push_screen = MagicMock()
        await llm_actions._open_chat_screen(app, paper, object())
        assert app.push_screen.called

        app.push_screen.reset_mock()
        app._capture_dataset_epoch = MagicMock(return_value=2)
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        await llm_actions._open_chat_screen(app, paper, object())
        assert not app.push_screen.called

        app._summary_loading = set()
        app._summary_command_hash = {paper.arxiv_id: "stale"}
        app._paper_summaries[paper.arxiv_id] = "stale"
        app._config.llm_prompt_template = "custom prompt"
        with patch("arxiv_browser.app._load_summary", return_value=None):
            llm_actions._on_summary_mode_selected(app, "default", paper, "cmd {prompt}")
        assert paper.arxiv_id in app._summary_loading
        assert app._paper_summaries.get(paper.arxiv_id) != "stale"
        assert app._summary_mode_label[paper.arxiv_id] == ""

        app.push_screen.reset_mock()
        llm_actions._on_summary_mode_selected(app, None, paper, "cmd {prompt}")
        assert not app.push_screen.called

        app.notify.reset_mock()
        llm_actions._on_summary_mode_selected(app, "bogus", paper, "cmd {prompt}")
        assert "Unknown summary mode" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._llm_provider = None
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert not app.notify.called

        app._llm_provider = object()
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._services.llm.generate_summary = AsyncMock(return_value=(None, None))
        app.notify.reset_mock()
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "LLM command failed" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(side_effect=ValueError("bad prompt"))
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "bad prompt" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(side_effect=RuntimeError("boom"))
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "Summary failed" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(side_effect=Exception("boom"))
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "Summary failed" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_relevance_and_auto_tag_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50011")
        other = make_paper(arxiv_id="2401.50012")
        app._config = UserConfig(
            llm_command="echo {prompt}",
            llm_timeout=8,
            llm_prompt_template="",
            research_interests="",
        )
        app._llm_provider = object()
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._paper_summaries = {}
        app._summary_loading = set()
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._summary_db_path = tmp_path / "summary.db"
        app.all_papers = [paper, other]
        app.selected_ids = {paper.arxiv_id, other.arxiv_id}
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._get_current_paper = MagicMock(return_value=paper)
        app._collect_all_tags = MagicMock(return_value=["existing"])
        app._tags_for = MagicMock(return_value=["existing"])
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    generate_summary=AsyncMock(return_value=("summary", None)),
                    score_relevance_once=AsyncMock(return_value=(7, "reason")),
                    suggest_tags_once=AsyncMock(return_value=["topic:new"]),
                )
            )
        )
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._update_footer = MagicMock()
        app._update_abstract_display = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)

        app._require_llm_command = MagicMock(return_value="echo {prompt}")
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        app._start_score_relevance_flow = MagicMock()
        llm_actions.action_score_relevance(app)
        app._start_score_relevance_flow.assert_called_once()

        app._start_auto_tag_flow = MagicMock()
        llm_actions.action_auto_tag(app)
        app._start_auto_tag_flow.assert_called_once()

        app._relevance_scoring_active = True
        llm_actions._start_score_relevance_flow(app, "echo {prompt}")
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._relevance_scoring_active = False
        app._config.research_interests = ""
        app.push_screen = MagicMock()
        llm_actions._start_score_relevance_flow(app, "echo {prompt}")
        assert app.push_screen.called
        callback = app.push_screen.call_args.args[1]
        callback(None)
        callback("new interests")

        app.notify.reset_mock()
        llm_actions._on_interests_saved_then_score(app, None, "echo {prompt}")
        app._relevance_scoring_active = True
        llm_actions._on_interests_saved_then_score(app, "saved", "echo {prompt}")
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._relevance_scoring_active = False
        app._start_relevance_scoring = MagicMock()
        llm_actions._on_interests_saved_then_score(app, "saved", "echo {prompt}")
        app._start_relevance_scoring.assert_called_once()

        app._config.research_interests = "saved"
        llm_actions._on_interests_edited(app, "saved")
        app._relevance_scores = {"x": (1, "reason")}
        llm_actions._on_interests_edited(app, "new interests")
        assert not app._relevance_scores
        assert app._mark_badges_dirty.called

        async def _run_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch(
                "arxiv_browser.app._load_all_relevance_scores",
                return_value={
                    paper.arxiv_id: (5, "cached"),
                    other.arxiv_id: (4, "cached"),
                },
            ),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper, other],
                "echo {prompt}",
                "interest",
            )
        assert "already scored" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._llm_provider = None
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.app._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )

        app._llm_provider = object()
        app._cancel_batch_requested = True
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.app._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )
        assert "cancelled" in app.notify.call_args[0][0].lower()

        app._cancel_batch_requested = False
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    score_relevance_once=AsyncMock(side_effect=RuntimeError("boom"))
                )
            )
        )
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.app._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )
        assert "failed" in app.notify.call_args[0][0].lower()

        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(score_relevance_once=AsyncMock(side_effect=Exception("boom")))
            )
        )
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.app._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )

        app._get_or_create_metadata = MagicMock(
            return_value=SimpleNamespace(tags=["existing", "topic:new"])
        )
        taxonomy = ["existing"]
        llm_actions._apply_auto_tag_batch_result(
            app,
            paper=paper,
            suggested=["topic:new", "topic:extra"],
            taxonomy=taxonomy,
        )
        assert "topic:extra" in taxonomy

        app._cancel_batch_requested = False
        assert llm_actions._maybe_cancel_auto_tag_batch(app, index=1, total=2, tagged=0) is False
        app._cancel_batch_requested = True
        assert llm_actions._maybe_cancel_auto_tag_batch(app, index=2, total=3, tagged=1) is True
        assert app._save_config_or_warn.called

        assert llm_actions._auto_tag_failure_message(0) == "Auto-tagging failed"
        assert (
            llm_actions._auto_tag_failure_message(2)
            == "Auto-tagging failed (2 tagged before error)"
        )

        app._auto_tag_active = True
        llm_actions._start_auto_tag_flow(app)
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._auto_tag_active = False
        app.selected_ids = {paper.arxiv_id}
        app.all_papers = [paper]
        llm_actions._start_auto_tag_flow(app)
        assert app._track_dataset_task.called

        app.notify.reset_mock()
        app._auto_tag_active = False
        app.selected_ids = {"missing"}
        app.all_papers = [paper]
        llm_actions._start_auto_tag_flow(app)
        assert "No selected papers found" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=None)
        llm_actions._start_auto_tag_flow(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app._get_current_paper = MagicMock(return_value=paper)
        app._tags_for = MagicMock(return_value=["existing"])
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        llm_actions._start_auto_tag_flow(app)
        assert app._track_dataset_task.called

        app.notify.reset_mock()
        app._call_auto_tag_llm = AsyncMock(return_value=None)
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])
        assert "Auto-tagging failed" in app.notify.call_args[0][0]

        app._call_auto_tag_llm = AsyncMock(return_value=["topic:new"])
        app.push_screen = MagicMock()
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])
        assert app.push_screen.called
        callback = app.push_screen.call_args.args[1]
        callback(["topic:accepted"])

        app._call_auto_tag_llm = AsyncMock(side_effect=RuntimeError("boom"))
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])
        app._call_auto_tag_llm = AsyncMock(side_effect=Exception("boom"))
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])

        app._cancel_batch_requested = False
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:a"], None])
        app._save_config_or_warn = MagicMock()
        await llm_actions._auto_tag_batch_async(app, [paper, other], ["existing"])
        assert "failed" in app.notify.call_args[0][0].lower()

        app._llm_provider = None
        assert await app_mod.ArxivBrowser._call_auto_tag_llm(app, paper, ["existing"]) is None

        app._llm_provider = object()
        app.notify.reset_mock()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    suggest_tags_once=AsyncMock(return_value=None),
                )
            )
        )
        assert await app_mod.ArxivBrowser._call_auto_tag_llm(app, paper, ["existing"]) is None
        assert "Could not parse LLM response" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_relevance_and_auto_tag_success_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        papers = [
            make_paper(arxiv_id="2401.50101"),
            make_paper(arxiv_id="2401.50102"),
            make_paper(arxiv_id="2401.50103"),
            make_paper(arxiv_id="2401.50104"),
            make_paper(arxiv_id="2401.50105"),
            make_paper(arxiv_id="2401.50106"),
        ]
        app._config = UserConfig(
            llm_command="echo {prompt}",
            llm_timeout=8,
            llm_prompt_template="",
            research_interests="interest",
        )
        app._llm_provider = object()
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._papers_by_id = {paper.arxiv_id: paper for paper in papers}
        app.all_papers = papers
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_footer = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._save_config_or_warn = MagicMock()
        app.notify = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    score_relevance_once=AsyncMock(
                        side_effect=[
                            (10, "a"),
                            None,
                            (8, "b"),
                            (7, "c"),
                            (6, "d"),
                        ]
                    )
                )
            )
        )

        async def _run_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch(
                "arxiv_browser.app._load_all_relevance_scores",
                return_value={papers[0].arxiv_id: (5, "cached")},
            ),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                papers,
                "echo {prompt}",
                "interest",
            )

        assert any("Scoring relevance 5/" in call.args[0] for call in app.notify.call_args_list)
        assert any("cached" in call.args[0].lower() for call in app.notify.call_args_list)
        assert app._update_relevance_badge.called

        app.notify.reset_mock()
        app._config.paper_metadata = {}
        app._get_or_create_metadata = MagicMock(
            side_effect=lambda aid: app._config.paper_metadata.setdefault(
                aid, app_mod.PaperMetadata(arxiv_id=aid, tags=["existing"])
            )
        )
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:new"], None, ["topic:extra"]])
        taxonomy = ["existing"]
        await llm_actions._auto_tag_batch_async(app, papers[:3], taxonomy)
        assert "Auto-tagged 2 papers (1 failed)" in app.notify.call_args[0][0]
        assert "topic:new" in app._config.paper_metadata[papers[0].arxiv_id].tags
        assert "topic:extra" in taxonomy
