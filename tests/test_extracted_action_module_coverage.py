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


class TestLlmActionCoverage:
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
