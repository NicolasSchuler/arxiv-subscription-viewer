"""Tests for CLI ergonomics fixes: typo handling, global-flag placement,

bounded --max-results, --theme validation, and the doctor network probe.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arxiv_browser import cli as cli_module
from arxiv_browser.cli import CliDependencies, _build_cli_parser, main
from arxiv_browser.models import UserConfig


def _deps(config: UserConfig | None = None) -> CliDependencies:
    return CliDependencies(
        load_config_fn=lambda: config or UserConfig(),
        discover_history_files_fn=lambda _base: [],
        resolve_papers_fn=lambda *_a: ([], [], 0),
        configure_logging_fn=MagicMock(),
        configure_color_mode_fn=MagicMock(),
        validate_interactive_tty_fn=lambda: False,
    )


class TestTypoSubcommand:
    def test_unknown_bare_word_is_invalid_choice(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["serch"], deps=_deps())
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice" in err
        assert "serch" in err

    def test_bare_word_is_not_rewritten_to_browse(self) -> None:
        assert cli_module._normalize_cli_argv(["serch"]) == ["serch"]

    def test_known_command_still_dispatches(self) -> None:
        assert cli_module._normalize_cli_argv(["browse"]) == ["browse"]
        assert cli_module._normalize_cli_argv([]) == ["browse"]


class TestGlobalFlagsAfterSubcommand:
    @pytest.mark.parametrize(
        "argv",
        [
            ["dates", "--ascii"],
            ["--ascii", "dates"],
            ["dates", "--theme", "nord"],
            ["dates", "--color", "never"],
        ],
    )
    def test_global_flag_parses_in_either_position(self, argv: list[str]) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(argv)  # should not raise
        assert args.command == "dates"

    def test_flag_before_command_survives_subparser_defaults(self) -> None:
        parser = _build_cli_parser()
        assert parser.parse_args(["--ascii", "dates"]).ascii is True
        assert parser.parse_args(["dates", "--ascii"]).ascii is True
        assert parser.parse_args(["dates"]).ascii is False

    def test_color_value_resolves_in_subcommand_position(self) -> None:
        parser = _build_cli_parser()
        assert parser.parse_args(["--color", "always", "dates"]).color == "always"
        assert parser.parse_args(["dates", "--color", "always"]).color == "always"


class TestMaxResultsBounds:
    @pytest.mark.parametrize("value", ["0", "999", "-3"])
    def test_out_of_range_is_usage_error(self, value: str, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["search", "--query", "x", "--max-results", value], deps=_deps())
        assert exc.value.code == 2
        assert "1 to 200" in capsys.readouterr().err

    def test_non_integer_is_usage_error(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["digest", "--max-results", "abc"], deps=_deps())
        assert exc.value.code == 2
        assert "expected an integer" in capsys.readouterr().err

    def test_in_range_value_accepted(self) -> None:
        parser = _build_cli_parser()
        assert parser.parse_args(["search", "--max-results", "200"]).max_results == 200


class TestThemeValidation:
    def test_unknown_theme_is_rejected(self, capsys) -> None:
        assert main(["browse", "--theme", "bogus"], deps=_deps()) == 2
        err = capsys.readouterr().err
        assert "not a known theme" in err
        assert "nord" in err

    def test_known_theme_passes_validation(self) -> None:
        assert cli_module.validate_theme_override("nord", UserConfig()) is None

    def test_custom_theme_name_passes_validation(self) -> None:
        config = UserConfig(custom_themes={"paper-night": {"primary": "#fff"}})
        assert cli_module.validate_theme_override("paper-night", config) is None

    def test_no_theme_is_noop(self) -> None:
        assert cli_module.validate_theme_override(None, UserConfig()) is None


class TestDoctorNetworkProbe:
    def test_probe_reports_ok_when_reachable(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(
            "arxiv_browser.cli_doctor._doctor_network_reachable", lambda *_a, **_k: True
        )
        cli_module._run_doctor(UserConfig(), [], probe_network=True)
        assert "Network: arXiv API reachable" in capsys.readouterr().out

    def test_probe_warns_without_changing_exit_code_when_unreachable(
        self, capsys, monkeypatch
    ) -> None:
        # The probe is non-counting: reachable vs unreachable yield the same exit
        # code, so doctor stays usable offline.
        monkeypatch.setattr(
            "arxiv_browser.cli_doctor._doctor_network_reachable", lambda *_a, **_k: True
        )
        reachable_code = cli_module._run_doctor(UserConfig(), [], probe_network=True)
        capsys.readouterr()
        monkeypatch.setattr(
            "arxiv_browser.cli_doctor._doctor_network_reachable", lambda *_a, **_k: False
        )
        unreachable_code = cli_module._run_doctor(UserConfig(), [], probe_network=True)
        out = capsys.readouterr().out
        assert "WARN" in out
        assert "unreachable" in out
        assert reachable_code == unreachable_code

    def test_probe_skipped_by_default(self, capsys) -> None:
        cli_module._run_doctor(UserConfig(), [])
        assert "Network:" not in capsys.readouterr().out

    def test_reachable_helper_handles_transport_errors(self, monkeypatch) -> None:
        import httpx

        def _boom(*_a, **_k):
            raise httpx.ConnectError("down")

        monkeypatch.setattr("arxiv_browser.cli_doctor.httpx.head", _boom)
        from arxiv_browser.cli_doctor import _doctor_network_reachable

        assert _doctor_network_reachable("https://example.invalid", timeout=1.0) is False
