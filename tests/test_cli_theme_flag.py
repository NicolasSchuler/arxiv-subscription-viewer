"""Tests for the ``--theme`` CLI override flag."""

from __future__ import annotations

import pytest

from arxiv_browser.cli import _build_cli_parser
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
