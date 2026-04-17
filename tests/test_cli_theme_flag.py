"""Tests for the ``--theme`` CLI override flag."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arxiv_browser.cli import CliDependencies, _build_cli_parser, _normalize_cli_argv, main
from arxiv_browser.models import UserConfig
from arxiv_browser.themes import THEME_NAMES


def test_theme_flag_accepts_each_known_theme() -> None:
    parser = _build_cli_parser()
    for name in THEME_NAMES:
        args = parser.parse_args(["--theme", name])
        assert args.theme == name


def test_theme_flag_rejects_unknown_theme() -> None:
    parser = _build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--theme", "does-not-exist"])


def test_theme_flag_defaults_to_none() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])
    assert args.theme is None


def test_theme_flag_normalization_keeps_root_option_before_browse() -> None:
    assert _normalize_cli_argv(["--theme", "high-contrast"]) == [
        "--theme",
        "high-contrast",
        "browse",
    ]
    assert _normalize_cli_argv(["--theme=high-contrast"]) == [
        "--theme=high-contrast",
        "browse",
    ]


def test_theme_flag_normalization_leaves_missing_value_for_argparse() -> None:
    assert _normalize_cli_argv(["--theme"]) == ["--theme"]


def test_theme_flag_is_passed_as_runtime_option_without_mutating_config(make_paper) -> None:
    config = UserConfig(theme_name="monokai")
    captured = {}

    def _app_factory(papers, *, options):
        captured["papers"] = papers
        captured["options"] = options
        return SimpleNamespace(run=MagicMock())

    deps = CliDependencies(
        load_config_fn=lambda: config,
        discover_history_files_fn=lambda _base_dir: [],
        resolve_papers_fn=lambda *_args: ([make_paper()], [], 0),
        configure_logging_fn=MagicMock(),
        configure_color_mode_fn=MagicMock(),
        validate_interactive_tty_fn=lambda: True,
        app_factory=_app_factory,
        app_factory_supports_options=True,
    )

    assert main(["--theme", "high-contrast"], deps=deps) == 0
    assert config.theme_name == "monokai"
    assert captured["options"].config is config
    assert captured["options"].theme_override == "high-contrast"
