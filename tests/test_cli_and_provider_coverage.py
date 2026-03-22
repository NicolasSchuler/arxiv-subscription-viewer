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


class TestLlmProvidersCoverage:
    def test_llm_command_requires_shell_and_windows_quote_handling(self, monkeypatch) -> None:
        assert llm_providers.llm_command_requires_shell("echo {prompt} | cat") is True
        assert llm_providers.llm_command_requires_shell("echo {prompt}") is False

        monkeypatch.setattr(llm_providers.os, "name", "nt", raising=False)
        assert llm_providers.llm_command_requires_shell('"quoted {prompt}"') is False
        assert llm_providers._strip_wrapping_quotes_windows(['"C:\\App\\tool.exe"', "x"]) == [
            "C:\\App\\tool.exe",
            "x",
        ]

    def test_build_invocation_plan_error_paths(self) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            llm_providers._build_invocation_plan("echo hello", "prompt")

        with pytest.raises(ValueError, match="disabled"):
            llm_providers._build_invocation_plan("echo {prompt} | cat", "prompt", allow_shell=False)

    @pytest.mark.asyncio
    async def test_cli_provider_retry_and_no_retry_paths(self) -> None:
        provider = llm_providers.CLIProvider("echo {prompt}", max_retries=1)
        with (
            patch.object(
                llm_providers.CLIProvider,
                "_execute_once",
                new=AsyncMock(
                    side_effect=[
                        llm_providers.LLMResult(output="", success=False, error="Timed out"),
                        llm_providers.LLMResult(output="ok", success=True),
                    ]
                ),
            ) as execute_once,
            patch("arxiv_browser.llm_providers.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await provider.execute("hello", timeout=5)

        assert result.success is True
        assert result.output == "ok"
        assert execute_once.await_count == 2
        sleep.assert_awaited_once()


class TestCliCoverage:
    def test_resolve_input_file_and_history_date_branches(
        self, tmp_path, make_paper, capsys
    ) -> None:
        missing = cli._resolve_input_file(tmp_path / "missing.txt")
        assert missing == 1

        unreadable = tmp_path / "paper.txt"
        unreadable.write_text("paper", encoding="utf-8")
        with patch("arxiv_browser.cli.os.access", return_value=False):
            assert cli._resolve_input_file(unreadable) == 1

        with patch("arxiv_browser.cli.parse_arxiv_file", side_effect=OSError("boom")):
            assert cli._resolve_input_file(unreadable) == 1

        with patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]):
            assert len(cli._resolve_input_file(unreadable)) == 1

        history_files = [(date(2026, 1, 23), tmp_path / "2026-01-23.txt")]
        assert cli._resolve_history_date(history_files, "2026-13-01") is None
        assert cli._resolve_history_date(history_files, "2026-01-22") is None
        assert cli._resolve_history_date(history_files, "2026-01-23") == 0

    def test_resolve_legacy_fallback_branches(self, tmp_path, make_paper) -> None:
        assert cli._resolve_legacy_fallback(tmp_path) == 1

        arxiv_txt = tmp_path / "arxiv.txt"
        arxiv_txt.write_text("placeholder", encoding="utf-8")
        with patch("arxiv_browser.cli.os.access", return_value=False):
            assert cli._resolve_legacy_fallback(tmp_path) == 1
        with (
            patch("arxiv_browser.cli.os.access", return_value=True),
            patch("arxiv_browser.cli.parse_arxiv_file", side_effect=OSError("boom")),
        ):
            assert cli._resolve_legacy_fallback(tmp_path) == 1
        with (
            patch("arxiv_browser.cli.os.access", return_value=True),
            patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]),
        ):
            assert len(cli._resolve_legacy_fallback(tmp_path)) == 1

    def test_resolve_arxiv_api_mode_success_and_errors(self, make_paper, capsys) -> None:
        args = argparse.Namespace(
            command="search",
            query="graph",
            category="cs.AI",
            field="title",
            mode="page",
            max_results=5,
        )
        config = UserConfig(arxiv_api_max_results=7)
        with patch("arxiv_browser.cli._fetch_arxiv_api_papers", return_value=[make_paper()]):
            result = cli._resolve_arxiv_api_mode(args, config)
        assert isinstance(result, tuple)
        assert result[0][0].arxiv_id == "2401.12345"

        args.mode = "latest"
        with patch("arxiv_browser.cli._fetch_latest_arxiv_digest", return_value=[make_paper()]):
            result = cli._resolve_arxiv_api_mode(args, config)
        assert isinstance(result, tuple)

        with patch(
            "arxiv_browser.cli._fetch_arxiv_api_papers", side_effect=ValueError("bad query")
        ):
            assert cli._resolve_arxiv_api_mode(args, config) == 1

        response_429 = MagicMock(spec=httpx.Response)
        response_429.status_code = 429
        exc_429 = httpx.HTTPStatusError(
            "429",
            request=httpx.Request("GET", "https://example.com"),
            response=response_429,
        )
        with patch("arxiv_browser.cli._fetch_latest_arxiv_digest", side_effect=exc_429):
            assert cli._resolve_arxiv_api_mode(args, config) == 1

        response_503 = MagicMock(spec=httpx.Response)
        response_503.status_code = 503
        exc_503 = httpx.HTTPStatusError(
            "503",
            request=httpx.Request("GET", "https://example.com"),
            response=response_503,
        )
        with patch("arxiv_browser.cli._fetch_latest_arxiv_digest", side_effect=exc_503):
            assert cli._resolve_arxiv_api_mode(args, config) == 1

        assert cli._resolve_arxiv_api_mode(argparse.Namespace(command="browse"), config) is None

    def test_cli_helpers_and_doctor_paths(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(cli.os, "name", "posix", raising=False)
        assert cli._extract_command_binary("OPENAI_API_KEY=1 llm {prompt}") == "llm"
        assert cli._extract_command_binary('bad "quote') is None

        cli._configure_color_mode("never")
        assert cli.os.environ["NO_COLOR"] == "1"
        cli._configure_color_mode("always")
        assert cli.os.environ["FORCE_COLOR"] == "1"
        cli._configure_color_mode("auto")

        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
        assert cli._validate_interactive_tty() is True

        config_path = tmp_path / "config.json"
        assert cli._doctor_config_issue_count(config_path, ok_marker="OK", info_marker="INFO") == 0
        config_path.write_text("{}", encoding="utf-8")
        assert cli._doctor_config_issue_count(config_path, ok_marker="OK", info_marker="INFO") == 0

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        assert (
            cli._doctor_history_issue_count(
                [], ok_marker="OK", warn_marker="WARN", info_marker="INFO"
            )
            == 1
        )
        (history_dir / "2026-03-19.txt").write_text("paper", encoding="utf-8")
        assert (
            cli._doctor_history_issue_count(
                [(date(2026, 3, 19), history_dir / "2026-03-19.txt")],
                ok_marker="OK",
                warn_marker="WARN",
                info_marker="INFO",
            )
            == 0
        )

        cfg = UserConfig()
        cfg.llm_command = ""
        cfg.llm_preset = ""
        assert (
            cli._doctor_llm_issue_count(cfg, ok_marker="OK", warn_marker="WARN", info_marker="INFO")
            == 0
        )
