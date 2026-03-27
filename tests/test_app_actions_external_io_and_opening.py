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


class TestS2AndHfCoverage:
    @pytest.mark.asyncio
    async def test_action_fetch_s2_disabled_and_cache_paths(self, make_paper, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app._config.s2_cache_ttl_days = 7
        app._config.s2_api_key = ""
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._refresh_current_list_item = MagicMock()
        app._track_task = MagicMock()
        app._s2_db_path = tmp_path / "s2.db"
        app._s2_loading = set()
        app._s2_cache = {}

        paper = make_paper(arxiv_id="2401.30001")
        app._get_current_paper = MagicMock(return_value=paper)

        app._s2_active = False
        await app.action_fetch_s2()
        assert "Semantic Scholar is disabled" in app.notify.call_args[0][0]
        assert "Next step:" in app.notify.call_args[0][0]

        app._s2_active = True
        app._s2_cache[paper.arxiv_id] = _make_s2_paper(paper.arxiv_id)
        await app.action_fetch_s2()

        assert "already loaded" in app.notify.call_args[0][0]
        app._track_task.assert_not_called()
        assert paper.arxiv_id not in app._s2_loading

    @pytest.mark.asyncio
    async def test_action_fetch_s2_handles_cache_errors_and_schedules_fetch(
        self, make_paper, tmp_path
    ):
        app = _new_app()
        app._config = UserConfig()
        app._config.s2_cache_ttl_days = 7
        app._config.s2_api_key = ""
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._refresh_current_list_item = MagicMock()
        app._s2_db_path = tmp_path / "s2.db"
        app._s2_loading = set()
        app._s2_cache = {}
        paper = make_paper(arxiv_id="2401.30002")
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_active = True

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_dataset_task = MagicMock(side_effect=track_task)
        app._update_status_bar.reset_mock()
        await app.action_fetch_s2()
        app._track_dataset_task.assert_called_once()
        assert paper.arxiv_id in app._s2_loading
        app._update_status_bar.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_s2_paper_async_success_and_no_data(self, tmp_path):
        app = _new_app()
        app._http_client = object()
        app._config = UserConfig()
        app._config.s2_api_key = ""
        app._s2_db_path = tmp_path / "s2.db"
        app._s2_cache = {}
        app._s2_loading = {"2401.30003"}
        app._s2_api_error = False
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._get_ui_refresh_coordinator = MagicMock(
            return_value=SimpleNamespace(refresh_detail_pane=app._refresh_detail_pane)
        )
        app._services = SimpleNamespace(
            enrichment=SimpleNamespace(
                load_or_fetch_s2_paper=AsyncMock(
                    return_value=SimpleNamespace(
                        state="found",
                        paper=_make_s2_paper("2401.30003"),
                        complete=True,
                        from_cache=False,
                    )
                )
            )
        )

        await app._fetch_s2_paper_async("2401.30003")

        assert "2401.30003" in app._s2_cache
        assert "2401.30003" not in app._s2_loading
        assert app._s2_api_error is False
        app._update_status_bar.assert_called_once()
        app._mark_badges_dirty.assert_called_once_with("s2")

        app._s2_loading.add("2401.30004")
        app._s2_api_error = True
        app._update_status_bar.reset_mock()
        app._services.enrichment.load_or_fetch_s2_paper = AsyncMock(
            return_value=SimpleNamespace(
                state="not_found",
                paper=None,
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_s2_paper_async("2401.30004")
        assert "No Semantic Scholar data was found for this paper." in app.notify.call_args[0][0]
        assert "2401.30004" not in app._s2_loading
        assert app._s2_api_error is False
        app._update_status_bar.assert_called_once()

        app._s2_loading.add("2401.30005")
        app._update_status_bar.reset_mock()
        app._services.enrichment.load_or_fetch_s2_paper = AsyncMock(
            return_value=SimpleNamespace(
                state="unavailable",
                paper=None,
                complete=False,
                from_cache=False,
            )
        )
        await app._fetch_s2_paper_async("2401.30005")
        assert "Could not fetch Semantic Scholar data." in app.notify.call_args[0][0]
        assert "2401.30005" not in app._s2_loading
        assert app._s2_api_error is True
        app._update_status_bar.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_cache_hit_error_and_schedule(self, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app._config.hf_cache_ttl_hours = 6
        app._hf_cache = {}
        app._hf_loading = False
        app._hf_db_path = tmp_path / "hf.db"
        app._papers_by_id = {"2401.40001": object()}
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._track_dataset_task = MagicMock()

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_dataset_task = MagicMock(side_effect=track_task)
        app._hf_loading = False
        await app._fetch_hf_daily()
        app._track_dataset_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_async_success_and_empty(self, tmp_path):
        app = _new_app()
        app._http_client = object()
        app._config = UserConfig()
        app._hf_db_path = tmp_path / "hf.db"
        app._hf_cache = {}
        app._hf_loading = True
        app._hf_api_error = False
        app._papers_by_id = {"2401.40002": object()}
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._get_ui_refresh_coordinator = MagicMock(
            return_value=SimpleNamespace(refresh_detail_pane=app._refresh_detail_pane)
        )
        app._services = SimpleNamespace(
            enrichment=SimpleNamespace(
                load_or_fetch_hf_daily=AsyncMock(
                    return_value=SimpleNamespace(
                        state="found",
                        papers=[_make_hf_paper("2401.40002")],
                        complete=True,
                        from_cache=False,
                    )
                )
            )
        )

        hf = _make_hf_paper("2401.40002")
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="found",
                papers=[hf],
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        assert app._hf_cache["2401.40002"] is hf
        assert app._hf_loading is False
        assert app._hf_api_error is False

        app._hf_loading = True
        app._hf_api_error = True
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="empty",
                papers=[],
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        assert "No HuggingFace trending data was returned." in app.notify.call_args[0][0]
        assert app._hf_loading is False
        assert app._hf_api_error is False

        app._hf_loading = True
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="unavailable",
                papers=[],
                complete=False,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        assert "Could not fetch HuggingFace trending data." in app.notify.call_args[0][0]
        assert app._hf_loading is False
        assert app._hf_api_error is True


class TestDownloadClipboardAndOpenCoverage:
    @pytest.mark.asyncio
    async def test_download_pdf_async_delegates_to_service(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        paper = make_paper(arxiv_id="2401.50001")
        client = object()
        service_mock = AsyncMock(return_value=True)
        app._services = SimpleNamespace(download=SimpleNamespace(download_pdf=service_mock))

        ok = await app._download_pdf_async(paper, client)
        assert ok is True
        service_mock.assert_awaited_once()

        service_mock.reset_mock()
        service_mock.return_value = False
        ok = await app._download_pdf_async(paper, client)
        assert ok is False
        service_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_single_download_updates_state_and_finishes(self, make_paper):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.50002")
        app._download_pdf_async = AsyncMock(return_value=False)
        app._download_results = {}
        app._download_total = 1
        app._downloading = {paper.arxiv_id}
        app._update_download_progress = MagicMock()
        app._start_downloads = MagicMock()
        app._finish_download_batch = MagicMock()

        await app._process_single_download(paper)

        assert app._download_results[paper.arxiv_id] is False
        assert paper.arxiv_id not in app._downloading
        app._update_download_progress.assert_called_once_with(1, 1)
        app._start_downloads.assert_called_once()
        app._finish_download_batch.assert_called_once()

    def test_start_downloads_tracks_non_dataset_tasks(self, make_paper):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.50009")
        app._download_queue = deque([paper])
        app._downloading = set()

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)
        app._track_dataset_task = MagicMock()

        app._start_downloads()

        app._track_task.assert_called_once()
        app._track_dataset_task.assert_not_called()
        assert tracked
        assert paper.arxiv_id in app._downloading

    @pytest.mark.asyncio
    async def test_process_single_download_handles_download_exception(self, make_paper):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.50003")
        app._download_pdf_async = AsyncMock(side_effect=RuntimeError("boom"))
        app._download_results = {}
        app._download_total = 1
        app._downloading = {paper.arxiv_id}
        app._update_download_progress = MagicMock()
        app._start_downloads = MagicMock()
        app._finish_download_batch = MagicMock()

        await app._process_single_download(paper)

        assert app._download_results[paper.arxiv_id] is False
        assert paper.arxiv_id not in app._downloading
        app._update_download_progress.assert_called_once_with(1, 1)
        app._start_downloads.assert_called_once()
        app._finish_download_batch.assert_called_once()

    def test_finish_download_batch_notifies_and_resets_state(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.pdf_download_dir = None
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._download_results = {"a": True, "b": False}
        app._download_total = 2

        app._finish_download_batch()

        assert app._download_total == 0
        assert app._download_results == {}
        app.notify.assert_called_once()
        assert "Downloaded 1/2 PDFs (1 failed)" in app.notify.call_args[0][0]

    def test_action_download_pdf_filters_existing_and_starts(self, make_paper, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._do_start_downloads = MagicMock()
        app._is_download_batch_active = MagicMock(return_value=False)

        paper_existing = make_paper(arxiv_id="2401.50003")
        paper_new = make_paper(arxiv_id="2401.50004")
        app._get_target_papers = MagicMock(return_value=[paper_existing, paper_new])

        existing = tmp_path / "existing.pdf"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"ready")
        missing = tmp_path / "missing.pdf"

        with patch(
            "arxiv_browser.actions.external_io_actions.get_pdf_download_path",
            side_effect=[existing, missing],
        ):
            app.action_download_pdf()

        app._do_start_downloads.assert_called_once_with([paper_new])

    def test_action_download_pdf_handles_no_target(self):
        app = _new_app()
        app.notify = MagicMock()
        app._is_download_batch_active = MagicMock(return_value=False)
        app._get_target_papers = MagicMock(return_value=[])
        app.action_download_pdf()
        assert "No papers to download" in app.notify.call_args[0][0]

    def test_copy_to_clipboard_platform_paths(self):
        app = _new_app()

        with (
            patch(
                "arxiv_browser.actions.external_io_actions.platform.system", return_value="Darwin"
            ),
            patch(
                "arxiv_browser.actions.external_io_actions.subprocess.run", return_value=None
            ) as run,
        ):
            assert app._copy_to_clipboard("abc") is True
            assert run.call_count == 1

        with (
            patch(
                "arxiv_browser.actions.external_io_actions.platform.system", return_value="Linux"
            ),
            patch(
                "arxiv_browser.actions.external_io_actions.subprocess.run",
                side_effect=[FileNotFoundError(), None],
            ) as run,
        ):
            assert app._copy_to_clipboard("abc") is True
            assert run.call_count == 2

        with (
            patch(
                "arxiv_browser.actions.external_io_actions.platform.system", return_value="Plan9"
            ),
            patch("arxiv_browser.actions.external_io_actions.subprocess.run", return_value=None),
        ):
            assert app._copy_to_clipboard("abc") is False

    def test_copy_to_clipboard_handles_subprocess_failure(self):
        import subprocess

        app = _new_app()
        with (
            patch(
                "arxiv_browser.actions.external_io_actions.platform.system", return_value="Darwin"
            ),
            patch(
                "arxiv_browser.actions.external_io_actions.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="pbcopy", timeout=1),
            ),
        ):
            assert app._copy_to_clipboard("abc") is False

    def test_open_with_viewer_and_open_pdf_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app._safe_browser_open = MagicMock(return_value=True)
        app._open_with_viewer = MagicMock(return_value=True)
        app._ensure_pdf_viewer_trusted = MagicMock(return_value=True)
        app._config = UserConfig()
        app._config.pdf_viewer = "viewer {url}"

        paper = make_paper(arxiv_id="2401.50005")
        app._do_open_pdfs([paper])
        app._open_with_viewer.assert_called_once()
        app._safe_browser_open.assert_not_called()

    def test_open_with_viewer_handles_errors(self):
        app = _new_app()
        app.notify = MagicMock()

        with patch(
            "arxiv_browser.actions.external_io_actions.subprocess.Popen",
            side_effect=OSError("bad viewer"),
        ):
            ok = app._open_with_viewer("broken-viewer {url}", "https://arxiv.org/pdf/1")

        assert ok is False
        message = app.notify.call_args[0][0]
        assert "Could not open the configured PDF viewer." in message
        assert "Why:" in message
        assert "Next step:" in message

    def test_open_with_viewer_handles_invalid_command_template(self):
        app = _new_app()
        app.notify = MagicMock()

        ok = app._open_with_viewer("   ", "https://arxiv.org/pdf/1")

        assert ok is False
        message = app.notify.call_args[0][0]
        assert "Could not open the configured PDF viewer." in message
        assert "Why:" in message
        assert "Next step:" in message

    def test_safe_browser_open_handles_errors(self):
        app = _new_app()
        app.notify = MagicMock()

        with patch(
            "arxiv_browser.actions.external_io_actions.webbrowser.open",
            side_effect=OSError("no browser"),
        ):
            ok = app._safe_browser_open("https://arxiv.org/abs/2602.12345")

        assert ok is False
        message = app.notify.call_args[0][0]
        assert "Could not open your browser." in message
        assert "Why:" in message
        assert "Next step:" in message

    def test_open_with_viewer_uses_subprocess_args_not_shell(self):
        app = _new_app()
        app.notify = MagicMock()

        with patch("arxiv_browser.actions.external_io_actions.subprocess.Popen") as popen:
            ok = app._open_with_viewer("open -a Skim {path}", "/tmp/my paper.pdf")

        assert ok is True
        popen.assert_called_once()
        call_args = popen.call_args
        assert call_args.args[0] == ["open", "-a", "Skim", "/tmp/my paper.pdf"]
        assert "shell" not in call_args.kwargs

    def test_action_open_url_uses_confirmation_for_large_batches(self, make_paper):
        app = _new_app()
        app._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id=f"2401.{i:05d}") for i in range(20)]
        )
        app.push_screen = MagicMock()
        app.action_open_url()
        app.push_screen.assert_called_once()

    def test_export_dispatch_and_markdown_export(self, make_paper):
        paper = make_paper(arxiv_id="2401.50006")
        app = _new_app()
        app.action_copy_selected = MagicMock()
        app.action_copy_bibtex = MagicMock()
        app.action_export_markdown = MagicMock()
        app._export_clipboard_ris = MagicMock()
        app._export_clipboard_csv = MagicMock()
        app._export_clipboard_mdtable = MagicMock()
        app.action_export_bibtex_file = MagicMock()
        app._export_file_ris = MagicMock()
        app._export_file_csv = MagicMock()
        app._do_export("clipboard-bibtex", [paper])
        app.action_copy_bibtex.assert_called_once()

        app._do_export("file-csv", [paper])
        app._export_file_csv.assert_called_once_with([paper])

        app2 = _new_app()
        app2._get_target_papers = MagicMock(return_value=[paper])
        app2._format_paper_as_markdown = MagicMock(return_value="## Test")
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.notify = MagicMock()
        app2.action_export_markdown()
        app2._copy_to_clipboard.assert_called_once()
        assert "as Markdown" in app2.notify.call_args[0][0]
