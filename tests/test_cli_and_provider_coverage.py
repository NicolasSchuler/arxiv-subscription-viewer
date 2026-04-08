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

import arxiv_browser.cli as cli
import arxiv_browser.llm as llm_mod
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

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


class _FakeProcess:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False
        self.wait_called = False

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self):
        self.wait_called = True


class TestLlmProvidersCoverage:
    def test_llm_command_requires_shell_and_windows_quote_handling(self, monkeypatch) -> None:
        assert llm_providers.llm_command_requires_shell("echo {prompt} | cat") is True
        assert llm_providers.llm_command_requires_shell("echo {prompt}") is False
        assert llm_providers.llm_command_requires_shell("llm") is False

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

        with patch("arxiv_browser.llm_providers.shlex.split", side_effect=ValueError("bad")):
            plan = llm_providers._build_invocation_plan("echo {prompt}", "hi")
        assert plan.use_shell is True
        assert "hi" in plan.shell_command

    def test_resolve_provider_none_and_non_none(self) -> None:
        assert llm_providers.resolve_provider(UserConfig()) is None
        provider = llm_providers.resolve_provider(UserConfig(llm_command="echo {prompt}"))
        assert provider is not None
        assert provider.command_template == "echo {prompt}"

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

    @pytest.mark.asyncio
    async def test_cli_provider_execute_branch_matrix(self) -> None:
        provider = llm_providers.CLIProvider("echo {prompt}")
        shell_provider = llm_providers.CLIProvider("echo {prompt} | cat")

        success_proc = _FakeProcess(0, stdout=b"ok\n")
        exit_proc = _FakeProcess(7, stderr=b"bad")
        empty_proc = _FakeProcess(0, stdout=b"")
        timeout_proc = _FakeProcess(0)
        shell_proc = _FakeProcess(0, stdout=b"shell\n")

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=success_proc),
        ):
            result = await provider.execute("hello", timeout=5)
        assert result.success is True
        assert result.output == "ok"

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=exit_proc),
        ):
            result = await provider.execute("hello", timeout=5)
        assert result.success is False
        assert result.error.startswith("Exit 7")

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=empty_proc),
        ):
            result = await provider.execute("hello", timeout=5)
        assert result.error == "Empty output"

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
            side_effect=OSError("boom"),
        ):
            result = await provider.execute("hello", timeout=5)
        assert "boom" in result.error

        async def _raise_timeout(coro, *_args, **_kwargs):
            coro.close()
            raise TimeoutError()

        with (
            patch(
                "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=timeout_proc),
            ),
            patch("arxiv_browser.llm_providers.asyncio.wait_for", side_effect=_raise_timeout),
        ):
            result = await provider.execute("hello", timeout=1)
        assert result.error.startswith("Timed out after 1s")
        assert timeout_proc.killed is True

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_shell",
            new=AsyncMock(return_value=shell_proc),
        ):
            result = await shell_provider.execute("hello", timeout=5)
        assert result.success is True
        assert result.output == "shell"

    def test_build_invocation_plan_more_branches(self, monkeypatch) -> None:
        with (
            patch("arxiv_browser.llm_providers.shlex.split", return_value=[]),
            pytest.raises(ValueError, match="empty"),
        ):
            llm_providers._build_invocation_plan("echo {prompt}", "hi")

        plan = llm_providers._build_invocation_plan("echo {prompt} | cat", "hi")
        assert plan.use_shell is True

        monkeypatch.setattr(llm_providers.os, "name", "nt", raising=False)
        assert llm_providers._strip_wrapping_quotes_windows(['"C:\\Tool\\llm.exe"', "x"]) == [
            "C:\\Tool\\llm.exe",
            "x",
        ]


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
        assert cli._extract_command_binary("") is None

        monkeypatch.setattr(cli.os, "name", "nt", raising=False)
        assert cli._extract_command_binary('"C:\\Tool\\llm.exe" --prompt') == "C:\\Tool\\llm.exe"
        monkeypatch.setattr(cli.os, "name", "posix", raising=False)

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

    def test_fetch_latest_digest_and_config_logging_branches(
        self, make_paper, tmp_path, monkeypatch
    ) -> None:
        same_day = make_paper(arxiv_id="2401.90001", date="Mon, 15 Jan 2024")
        dup_same_day = make_paper(arxiv_id="2401.90001", date="Mon, 15 Jan 2024")
        malformed_date = make_paper(arxiv_id="2401.90002", date="")
        older_day = make_paper(arxiv_id="2401.90003", date="Tue, 16 Jan 2024")

        with (
            patch(
                "arxiv_browser.cli._fetch_arxiv_api_papers",
                side_effect=[[malformed_date, same_day, dup_same_day], []],
            ),
            patch("arxiv_browser.cli.time.sleep") as sleep,
        ):
            digest = cli._fetch_latest_arxiv_digest(
                query="graph",
                field="all",
                category="",
                max_results=3,
            )
        assert [paper.arxiv_id for paper in digest] == [same_day.arxiv_id]
        sleep.assert_called_once_with(cli.ARXIV_API_MIN_INTERVAL_SECONDS)

        with (
            patch(
                "arxiv_browser.cli._fetch_arxiv_api_papers",
                side_effect=[[same_day, older_day]],
            ),
            patch("arxiv_browser.cli.time.sleep") as sleep,
        ):
            digest = cli._fetch_latest_arxiv_digest(
                query="graph",
                field="all",
                category="",
                max_results=2,
            )
        assert [paper.arxiv_id for paper in digest] == [same_day.arxiv_id]
        sleep.assert_not_called()

        disable = MagicMock()
        monkeypatch.setattr(cli.logging, "disable", disable)
        cli._configure_logging(False)
        disable.assert_called_once_with(cli.logging.CRITICAL)

        handler = SimpleNamespace(setFormatter=MagicMock())
        add_handler = MagicMock()
        set_level = MagicMock()
        monkeypatch.setattr(cli.logging.root, "addHandler", add_handler, raising=False)
        monkeypatch.setattr(cli.logging.root, "setLevel", set_level, raising=False)
        with (
            patch("arxiv_browser.cli.user_config_dir", return_value=str(tmp_path)),
            patch(
                "arxiv_browser.cli.logging.handlers.RotatingFileHandler",
                return_value=handler,
            ),
        ):
            cli._configure_logging(True)
        add_handler.assert_called_once_with(handler)
        set_level.assert_called_once_with(cli.logging.DEBUG)

    def test_doctor_helpers_and_main_entrypoints(
        self, make_paper, tmp_path, monkeypatch, capsys
    ) -> None:
        cfg = UserConfig()
        cfg.llm_command = ""
        cfg.llm_preset = ""
        cfg.s2_enabled = True
        cfg.s2_api_key = "key"
        cfg.hf_enabled = False
        cfg.bibtex_export_dir = None
        cfg.pdf_download_dir = None

        cli._doctor_feature_summary(cfg, ok_marker="OK", info_marker="INFO")
        out = capsys.readouterr().out
        assert "Semantic Scholar: enabled (API key set)" in out
        assert "HuggingFace trending: disabled" in out

        cfg.s2_enabled = False
        cfg.hf_enabled = True
        cfg.bibtex_export_dir = str(tmp_path / "exports")
        cfg.pdf_download_dir = str(tmp_path / "pdfs")
        (tmp_path / "exports").mkdir()
        cli._doctor_feature_summary(cfg, ok_marker="OK", info_marker="INFO")
        cli._doctor_export_dirs(cfg, ok_marker="OK", info_marker="INFO")
        out = capsys.readouterr().out
        assert "Semantic Scholar: disabled" in out
        assert "HuggingFace trending: enabled" in out
        assert "Export dir" in out
        assert "PDF dir" in out

        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
        cli._doctor_terminal_summary(ok_marker="OK", info_marker="INFO")
        out = capsys.readouterr().out
        assert "not an interactive TTY" in out

        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
        cli._doctor_terminal_summary(ok_marker="OK", info_marker="INFO")
        out = capsys.readouterr().out
        assert "interactive TTY" in out

        cfg = UserConfig()
        cfg.llm_command = ""
        cfg.llm_preset = "missing"
        assert (
            cli._doctor_llm_issue_count(cfg, ok_marker="OK", warn_marker="WARN", info_marker="INFO")
            == 1
        )

        preset_name = next(iter(llm_mod.LLM_PRESETS))
        cfg.llm_preset = preset_name
        cfg.allow_llm_shell_fallback = True
        with (
            patch("arxiv_browser.llm._resolve_llm_command", return_value="echo {prompt}"),
            patch("arxiv_browser.cli.shutil.which", return_value="/usr/bin/echo"),
        ):
            assert (
                cli._doctor_llm_issue_count(
                    cfg, ok_marker="OK", warn_marker="WARN", info_marker="INFO"
                )
                == 0
            )

        def _deps(
            *,
            resolve_result,
            history_files,
            tty_ok,
            configure_color_mode_fn=None,
            app_factory=None,
            factory_supports_options=None,
        ) -> cli.CliDependencies:
            return cli.CliDependencies(
                load_config_fn=lambda: UserConfig(),
                discover_history_files_fn=lambda _base_dir: history_files,
                resolve_papers_fn=lambda *_args: resolve_result,
                configure_logging_fn=MagicMock(),
                configure_color_mode_fn=configure_color_mode_fn or MagicMock(),
                validate_interactive_tty_fn=lambda: tty_ok,
                app_factory=app_factory,
                app_factory_supports_options=factory_supports_options,
            )

        with patch("arxiv_browser.completions.get_completion_script", return_value="script"):
            assert (
                cli.main(
                    ["completions", "bash"],
                    deps=_deps(resolve_result=1, history_files=[], tty_ok=False),
                )
                == 0
            )

        with patch("arxiv_browser.cli._print_config_path", return_value=0):
            assert (
                cli.main(
                    ["config-path"],
                    deps=_deps(resolve_result=1, history_files=[], tty_ok=False),
                )
                == 0
            )

        assert (
            cli.main(["dates"], deps=_deps(resolve_result=1, history_files=[], tty_ok=False)) == 1
        )
        assert (
            cli.main(
                ["dates"],
                deps=_deps(
                    resolve_result=1,
                    history_files=[(date(2026, 3, 19), tmp_path / "2026-03-19.txt")],
                    tty_ok=False,
                ),
            )
            == 0
        )

        no_papers_color = MagicMock()
        assert (
            cli.main(
                ["--no-color", "browse"],
                deps=_deps(
                    resolve_result=([], [], 0),
                    history_files=[],
                    tty_ok=True,
                    configure_color_mode_fn=no_papers_color,
                ),
            )
            == 1
        )
        no_papers_color.assert_called_with("never")

        explicit_color = MagicMock()
        assert (
            cli.main(
                ["--color", "always", "browse"],
                deps=_deps(
                    resolve_result=([], [], 0),
                    history_files=[],
                    tty_ok=True,
                    configure_color_mode_fn=explicit_color,
                ),
            )
            == 1
        )
        explicit_color.assert_called_with("always")

        monkeypatch.setenv("NO_COLOR", "1")
        env_color = MagicMock()
        assert (
            cli.main(
                ["browse"],
                deps=_deps(
                    resolve_result=([], [], 0),
                    history_files=[],
                    tty_ok=True,
                    configure_color_mode_fn=env_color,
                ),
            )
            == 1
        )
        env_color.assert_called_with("never")
        monkeypatch.delenv("NO_COLOR", raising=False)

        tty_false_deps = _deps(
            resolve_result=([make_paper()], [], 0),
            history_files=[],
            tty_ok=False,
        )
        assert cli.main(["browse"], deps=tty_false_deps) == 2

        run_mock = MagicMock()
        patched_factory = MagicMock(return_value=SimpleNamespace(run=run_mock))
        search_deps = _deps(
            resolve_result=(
                [make_paper(arxiv_id="2401.90010")],
                [(date(2026, 3, 19), tmp_path / "2026-03-19.txt")],
                0,
            ),
            history_files=[(date(2026, 3, 19), tmp_path / "2026-03-19.txt")],
            tty_ok=True,
            app_factory=None,
            factory_supports_options=True,
        )
        search_deps.app_factory = patched_factory
        assert cli.main(["search"], deps=search_deps) == 0
        patched_factory.assert_called_once()
        assert "options" in patched_factory.call_args.kwargs
        run_mock.assert_called_once()

        legacy_run = MagicMock()
        legacy_factory = MagicMock(return_value=SimpleNamespace(run=legacy_run))
        legacy_deps = _deps(
            resolve_result=(
                [make_paper(arxiv_id="2401.90011")],
                [(date(2026, 3, 19), tmp_path / "2026-03-19.txt")],
                0,
            ),
            history_files=[(date(2026, 3, 19), tmp_path / "2026-03-19.txt")],
            tty_ok=True,
            app_factory=None,
            factory_supports_options=False,
        )
        legacy_deps.app_factory = legacy_factory
        assert cli.main(["browse", "--no-restore"], deps=legacy_deps) == 0
        legacy_factory.assert_called_once()
        assert "config" in legacy_factory.call_args.kwargs
        legacy_run.assert_called_once()
