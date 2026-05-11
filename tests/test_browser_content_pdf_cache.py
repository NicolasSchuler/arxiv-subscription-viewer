"""Tests for cache-aware paper content retrieval and PDF fallback."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_browser.browser.content import (
    PaperContentFetchRequest,
    _extract_text_from_pdf_bytes,
    _load_cached_paper_content,
    _save_paper_content,
    fetch_paper_content,
)
from arxiv_browser.export import get_pdf_download_path
from arxiv_browser.models import UserConfig


@dataclass(slots=True)
class _Response:
    status_code: int
    text: str = ""
    content: bytes = b""


class _SequenceClient:
    def __init__(self, *responses: _Response) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []

    async def get(self, url: str, **_kwargs) -> _Response:
        self.urls.append(url)
        if not self._responses:
            raise AssertionError(f"Unexpected fetch: {url}")
        return self._responses.pop(0)


def test_paper_content_cache_save_load_and_ttl(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"

    _save_paper_content(db_path, "2401.12345", "html", "Fresh content")

    cached = _load_cached_paper_content(db_path, "2401.12345", ttl_days=7)
    assert cached is not None
    assert cached.content == "Fresh content"
    assert cached.source == "html"
    assert cached.cached is True

    old_time = (datetime.now(UTC) - timedelta(days=9)).isoformat()
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "UPDATE paper_content SET fetched_at = ? WHERE arxiv_id = ?",
            (old_time, "2401.12345"),
        )

    assert _load_cached_paper_content(db_path, "2401.12345", ttl_days=7) is None
    assert _load_cached_paper_content(db_path, "missing", ttl_days=7) is None

    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "UPDATE paper_content SET fetched_at = ? WHERE arxiv_id = ?",
            ("not-a-date", "2401.12345"),
        )
    assert _load_cached_paper_content(db_path, "2401.12345", ttl_days=7) is None


def test_pdf_text_extraction_bad_bytes_returns_empty() -> None:
    assert _extract_text_from_pdf_bytes(b"not a pdf") == ""


@pytest.mark.asyncio
async def test_html_content_is_cached_and_stale_cache_refetches(make_paper, tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    paper = make_paper(arxiv_id="2401.20001", abstract="Fallback")
    config = UserConfig(paper_content_cache_ttl_days=7)

    with patch("arxiv_browser.browser.content.extract_text_from_html", return_value="HTML content"):
        first = await fetch_paper_content(
            PaperContentFetchRequest(
                paper=paper,
                client=_SequenceClient(_Response(200, "<p>paper</p>")),
                db_path=db_path,
                config=config,
            )
        )
    assert first.source == "html"
    assert first.cached is False

    no_fetch_client = _SequenceClient()
    cached = await fetch_paper_content(
        PaperContentFetchRequest(
            paper=paper,
            client=no_fetch_client,
            db_path=db_path,
            config=config,
        )
    )
    assert cached.content == "HTML content"
    assert cached.cached is True
    assert no_fetch_client.urls == []

    old_time = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "UPDATE paper_content SET fetched_at = ? WHERE arxiv_id = ?",
            (old_time, paper.arxiv_id),
        )

    with patch("arxiv_browser.browser.content.extract_text_from_html", return_value="Fresh HTML"):
        refreshed = await fetch_paper_content(
            PaperContentFetchRequest(
                paper=paper,
                client=_SequenceClient(_Response(200, "<p>fresh</p>")),
                db_path=db_path,
                config=config,
            )
        )
    assert refreshed.content == "Fresh HTML"
    assert refreshed.cached is False


@pytest.mark.asyncio
async def test_pdf_fallback_uses_existing_downloaded_pdf(make_paper, tmp_path: Path) -> None:
    paper = make_paper(arxiv_id="2401.20002", abstract="Fallback")
    config = UserConfig(pdf_download_dir=str(tmp_path), paper_content_pdf_fallback=True)
    pdf_path = get_pdf_download_path(paper, config)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF fake existing bytes")
    client = _SequenceClient(_Response(404, ""))

    with patch(
        "arxiv_browser.browser.content._extract_text_from_pdf_bytes", return_value="PDF text"
    ):
        result = await fetch_paper_content(
            PaperContentFetchRequest(paper=paper, client=client, config=config)
        )

    assert result.source == "pdf"
    assert result.content == "PDF text"
    assert client.urls == [f"https://arxiv.org/html/{paper.arxiv_id}"]


@pytest.mark.asyncio
async def test_pdf_fallback_download_success(make_paper, tmp_path: Path) -> None:
    paper = make_paper(arxiv_id="2401.20003", abstract="Fallback")
    config = UserConfig(pdf_download_dir=str(tmp_path), paper_content_pdf_fallback=True)
    client = _SequenceClient(
        _Response(404, ""),
        _Response(200, content=b"%PDF downloaded bytes"),
    )

    with patch(
        "arxiv_browser.browser.content._extract_text_from_pdf_bytes", return_value="PDF text"
    ):
        result = await fetch_paper_content(
            PaperContentFetchRequest(paper=paper, client=client, config=config)
        )

    assert result.source == "pdf"
    assert result.content == "PDF text"
    assert client.urls[-1] == f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"


@pytest.mark.asyncio
async def test_corrupt_or_missing_pdf_falls_back_to_abstract(make_paper, tmp_path: Path) -> None:
    paper = make_paper(arxiv_id="2401.20004", abstract="Fallback abstract.")
    config = UserConfig(pdf_download_dir=str(tmp_path), paper_content_pdf_fallback=True)
    corrupt_client = _SequenceClient(_Response(404, ""), _Response(200, content=b"not a pdf"))

    with patch("arxiv_browser.browser.content._extract_text_from_pdf_bytes", return_value=""):
        corrupt = await fetch_paper_content(
            PaperContentFetchRequest(paper=paper, client=corrupt_client, config=config)
        )

    missing = await fetch_paper_content(
        PaperContentFetchRequest(
            paper=paper,
            client=_SequenceClient(_Response(404, ""), _Response(404, "")),
            config=config,
        )
    )

    assert corrupt.source == "abstract"
    assert corrupt.content == "Abstract:\nFallback abstract."
    assert missing.source == "abstract"
    assert missing.content == "Abstract:\nFallback abstract."
