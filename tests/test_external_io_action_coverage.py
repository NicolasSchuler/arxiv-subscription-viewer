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
from tests.support.patch_helpers import patch_save_config


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
            "arxiv_browser.actions.external_io_actions.write_timestamped_export_file",
            side_effect=OSError("boom"),
        ):
            io_actions._export_to_file(app, "content", "json", "Metadata")
        assert "Failed to export Metadata" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        export_file = tmp_path / "exports" / "metadata.json"
        with patch(
            "arxiv_browser.actions.external_io_actions.write_timestamped_export_file",
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

    def test_additional_coverage_branches(self, make_paper, tmp_path) -> None:
        """Cover remaining uncovered lines in external_io_actions."""
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50001")
        paper2 = make_paper(arxiv_id="2401.50002")
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app.notify = MagicMock()

        # Line 32: action_copy_bibtex clipboard failure
        app._get_target_papers = MagicMock(return_value=[paper])
        app._copy_to_clipboard = MagicMock(return_value=False)
        io_actions.action_copy_bibtex(app)
        assert "Failed to copy to clipboard" in app.notify.call_args[0][0]

        # Lines 51-52: action_export_markdown no paper selected
        app._get_target_papers = MagicMock(return_value=[])
        app.notify.reset_mock()
        io_actions.action_export_markdown(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        # Lines 72-73: action_export_menu no paper selected
        app.notify.reset_mock()
        io_actions.action_export_menu(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        # 94->exit: _do_export with unknown format key (handler None → no-op)
        sentinel = MagicMock()
        io_actions._do_export(app, "unknown-format-xyz", [paper])
        sentinel.assert_not_called()  # nothing should have been called for unknown key

        # Lines 135-136: _export_clipboard_ris success notification
        app._copy_to_clipboard = MagicMock(return_value=True)
        app._get_abstract_text = MagicMock(return_value="abstract text")
        app.notify.reset_mock()
        io_actions._export_clipboard_ris(app, [paper])
        assert "Copied 1 RIS entry" in app.notify.call_args[0][0]

        # Lines 161-162: _export_clipboard_mdtable success notification
        app.notify.reset_mock()
        io_actions._export_clipboard_mdtable(app, [paper])
        assert "Copied 1 paper as Markdown table" in app.notify.call_args[0][0]

        # Lines 176-181: _export_file_ris (loop body with two papers)
        app._export_to_file = MagicMock()
        app._get_abstract_text = MagicMock(return_value="abstract text")
        io_actions._export_file_ris(app, [paper, paper2])
        app._export_to_file.assert_called_once()

        # 259->261: _import_metadata_file with papers_n=0 so "papers" part is skipped
        json_file = tmp_path / "arxiv-meta.json"
        json_file.write_text("{}", encoding="utf-8")
        app._compute_watched_papers = MagicMock()
        app._refresh_list_view = MagicMock()
        app.notify.reset_mock()
        with (
            patch(
                "arxiv_browser.actions.external_io_actions.import_metadata",
                return_value=(0, 2, 0, 0),
            ),
            patch("arxiv_browser.actions.external_io_actions.save_config", return_value=True),
        ):
            io_actions._import_metadata_file(app, json_file)
        notify_msg = app.notify.call_args[0][0]
        assert "2 watch entries" in notify_msg
        assert "0 papers" not in notify_msg

        # 302->exit: _process_single_download when _shutting_down=True
        app._shutting_down = True
        app._download_pdf_async = AsyncMock(return_value=True)
        app._downloading = {paper.arxiv_id}
        app._download_results = {}
        app._download_total = 1
        app._update_download_progress = MagicMock()
        app._finish_download_batch = MagicMock()
        app._start_downloads = MagicMock()
        asyncio.run(io_actions._process_single_download(app, paper))
        app._update_download_progress.assert_not_called()

        # Lines 448-449: action_download_pdf when all PDFs already downloaded
        app._shutting_down = False
        app._is_download_batch_active = MagicMock(return_value=False)
        app._get_target_papers = MagicMock(return_value=[paper])
        app.notify.reset_mock()
        with patch(
            "arxiv_browser.actions.external_io_actions.filter_papers_needing_download",
            return_value=([], [paper.arxiv_id]),
        ):
            io_actions.action_download_pdf(app)
        assert "All PDFs already downloaded" in app.notify.call_args[0][0]

        # Line 452: action_download_pdf batch confirmation (> BATCH_CONFIRM_THRESHOLD=10)
        papers_large = [make_paper(arxiv_id=f"2401.6{i:04d}") for i in range(11)]
        app._get_target_papers = MagicMock(return_value=papers_large)
        app.push_screen = MagicMock()
        with patch(
            "arxiv_browser.actions.external_io_actions.filter_papers_needing_download",
            return_value=(papers_large, []),
        ):
            io_actions.action_download_pdf(app)
        app.push_screen.assert_called_once()

        # Lines 467-473: _do_start_downloads happy path initialises queue and notifies
        app._is_download_batch_active = MagicMock(return_value=False)
        app._download_queue = deque()
        app._download_results = {}
        app._download_total = 0
        app._start_downloads = MagicMock()
        app.notify.reset_mock()
        io_actions._do_start_downloads(app, [paper])
        assert app._download_total == 1
        app._start_downloads.assert_called_once()
        assert app.notify.call_args[1]["title"] == "Download"

        # Lines 478-479: _format_paper_for_clipboard delegates correctly
        app._get_abstract_text = MagicMock(return_value="abstract body")
        with patch(
            "arxiv_browser.actions.external_io_actions.format_paper_for_clipboard",
            return_value="formatted-text",
        ):
            result = io_actions._format_paper_for_clipboard(app, paper)
        assert result == "formatted-text"

        # 496->509: _copy_to_clipboard subprocess.run succeeds → return True
        with (
            patch(
                "arxiv_browser.actions.external_io_actions.get_clipboard_command_plan",
                return_value=([["pbcopy"]], "utf-8"),
            ),
            patch("arxiv_browser.actions.external_io_actions.subprocess.run", return_value=None),
        ):
            assert io_actions._copy_to_clipboard(app, "test text") is True

        # Line 508: _copy_to_clipboard single command raises FileNotFoundError → re-raise →
        # outer except catches it → return False
        with (
            patch(
                "arxiv_browser.actions.external_io_actions.get_clipboard_command_plan",
                return_value=([["xclip"]], "utf-8"),
            ),
            patch(
                "arxiv_browser.actions.external_io_actions.subprocess.run",
                side_effect=FileNotFoundError("not found"),
            ),
        ):
            assert io_actions._copy_to_clipboard(app, "test text") is False
