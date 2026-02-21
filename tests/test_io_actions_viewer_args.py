"""Targeted tests for platform-aware viewer command parsing."""

from __future__ import annotations

import pytest

from arxiv_browser.app import build_viewer_args


def test_build_viewer_args_windows_quoted_executable(monkeypatch) -> None:
    monkeypatch.setattr("arxiv_browser.io_actions.os.name", "nt", raising=False)

    args = build_viewer_args(
        '"C:\\Program Files\\SumatraPDF\\SumatraPDF.exe" {path}',
        "C:\\Users\\alice\\paper.pdf",
    )

    assert args == [
        "C:\\Program Files\\SumatraPDF\\SumatraPDF.exe",
        "C:\\Users\\alice\\paper.pdf",
    ]


def test_build_viewer_args_windows_appends_target_when_no_placeholder(monkeypatch) -> None:
    monkeypatch.setattr("arxiv_browser.io_actions.os.name", "nt", raising=False)

    args = build_viewer_args(
        '"C:\\Program Files\\MyViewer\\viewer.exe"',
        "https://arxiv.org/pdf/2401.12345.pdf",
    )

    assert args == [
        "C:\\Program Files\\MyViewer\\viewer.exe",
        "https://arxiv.org/pdf/2401.12345.pdf",
    ]


def test_build_viewer_args_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_viewer_args("   ", "https://arxiv.org/pdf/2401.12345.pdf")
