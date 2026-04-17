"""Tests for the optional ``user.tcss`` stylesheet override."""

from __future__ import annotations

from pathlib import Path

import pytest

from arxiv_browser import config as config_module
from arxiv_browser.browser import core as browser_core
from arxiv_browser.config import USER_TCSS_FILENAME, get_user_tcss_path


def test_user_tcss_path_lives_next_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_config = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "get_config_path", lambda: fake_config)
    assert get_user_tcss_path() == tmp_path / USER_TCSS_FILENAME


def test_resolve_user_css_path_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        browser_core, "get_user_tcss_path", lambda: tmp_path / "does-not-exist.tcss"
    )
    assert browser_core._resolve_user_css_path() is None


def test_resolve_user_css_path_returns_path_when_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "user.tcss"
    override.write_text("Screen { background: #000; }\n", encoding="utf-8")
    monkeypatch.setattr(browser_core, "get_user_tcss_path", lambda: override)
    assert browser_core._resolve_user_css_path() == override


def test_resolve_user_css_path_ignores_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_a_file = tmp_path / "user.tcss"
    not_a_file.mkdir()
    monkeypatch.setattr(browser_core, "get_user_tcss_path", lambda: not_a_file)
    assert browser_core._resolve_user_css_path() is None


def test_resolve_user_css_path_handles_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Path:
        raise RuntimeError("config dir unavailable")

    monkeypatch.setattr(browser_core, "get_user_tcss_path", _boom)
    assert browser_core._resolve_user_css_path() is None
