"""Targeted tests for export path security and session parsing hardening."""

from __future__ import annotations

from pathlib import Path

import pytest

from arxiv_browser.app import (
    ArxivBrowser,
    Paper,
    UserConfig,
    _dict_to_config,
    get_pdf_download_path,
)


def _make_paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        date="",
        title="Test",
        authors="Author",
        categories="cs.AI",
        comments=None,
        abstract=None,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def test_get_pdf_download_path_allows_normal_id(tmp_path) -> None:
    base_dir = tmp_path / "pdfs"
    config = UserConfig(pdf_download_dir=str(base_dir))

    path = get_pdf_download_path(_make_paper("2401.12345"), config)

    assert path == (base_dir / "2401.12345.pdf").resolve()


def test_get_pdf_download_path_expands_user_home() -> None:
    config = UserConfig(pdf_download_dir="~/arxiv-pdfs-test")

    path = get_pdf_download_path(_make_paper("2401.12345"), config)

    assert path == (Path.home() / "arxiv-pdfs-test" / "2401.12345.pdf").resolve()


def test_get_export_dir_expands_user_home() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._config = UserConfig(bibtex_export_dir="~/arxiv-exports-test")

    export_dir = app._get_export_dir()

    assert export_dir == (Path.home() / "arxiv-exports-test")


@pytest.mark.parametrize(
    "arxiv_id",
    [
        "../../etc/passwd",
        "../arxiv-evil/escape",
        "/tmp/absolute-escape",
    ],
)
def test_get_pdf_download_path_rejects_traversal_and_prefix_bypass(arxiv_id, tmp_path) -> None:
    base_dir = tmp_path / "arxiv"
    config = UserConfig(pdf_download_dir=str(base_dir))

    with pytest.raises(ValueError, match="Invalid arXiv ID"):
        get_pdf_download_path(_make_paper(arxiv_id), config)


def test_dict_to_config_filters_non_string_selected_ids() -> None:
    data = {
        "session": {
            "selected_ids": [
                "2401.00001",
                {"bad": "value"},
                ["also-bad"],
                123,
                True,
                "2401.00002",
            ]
        }
    }

    config = _dict_to_config(data)

    assert config.session.selected_ids == ["2401.00001", "2401.00002"]
