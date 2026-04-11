"""Tests for download service helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arxiv_browser.models import UserConfig
from arxiv_browser.services.download_service import DownloadFailure, DownloadResult, download_pdf


class _FakeResponse:
    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _StreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *_args) -> bool:
        return False


@pytest.mark.asyncio
async def test_download_pdf_success_and_failure(make_paper, tmp_path, caplog) -> None:
    paper = make_paper(arxiv_id="2401.50001")
    target = tmp_path / "pdfs" / "2401.50001.pdf"

    response = _FakeResponse([b"%PDF-1.4", b"", b" body"])
    client = SimpleNamespace(stream=MagicMock(return_value=_StreamContext(response)))

    with (
        patch(
            "arxiv_browser.services.download_service.get_pdf_url",
            return_value="https://example/pdf",
        ),
        patch("arxiv_browser.services.download_service.get_pdf_download_path", return_value=target),
    ):
        ok = await download_pdf(
            paper=paper,
            config=UserConfig(),
            client=client,
            timeout_seconds=30,
        )

    assert ok == DownloadResult(success=True)
    assert target.exists()
    assert target.read_bytes() == b"%PDF-1.4 body"
    assert list(target.parent.glob(".*.tmp")) == []

    with (
        patch(
            "arxiv_browser.services.download_service.get_pdf_url",
            return_value="https://example/pdf",
        ),
        patch("arxiv_browser.services.download_service.get_pdf_download_path", return_value=target),
    ):
        caplog.clear()
        caplog.set_level("WARNING", logger="arxiv_browser.services.download_service")
        client.stream = MagicMock(side_effect=OSError("network"))
        ok = await download_pdf(
            paper=paper,
            config=UserConfig(),
            client=client,
            timeout_seconds=30,
        )

    assert ok.success is False
    assert ok.failure == DownloadFailure.DISK
    assert list(target.parent.glob(".*.tmp")) == []
    assert "PDF download failed for 2401.50001" in caplog.text


@pytest.mark.asyncio
async def test_download_pdf_replace_failure_cleans_temp_file(make_paper, tmp_path, caplog) -> None:
    paper = make_paper(arxiv_id="2401.50002")
    target = tmp_path / "pdfs" / "2401.50002.pdf"
    response = _FakeResponse([b"%PDF-1.4", b" body"])
    client = SimpleNamespace(stream=MagicMock(return_value=_StreamContext(response)))

    with (
        patch(
            "arxiv_browser.services.download_service.get_pdf_url",
            return_value="https://example/pdf",
        ),
        patch("arxiv_browser.services.download_service.get_pdf_download_path", return_value=target),
        patch(
            "arxiv_browser.services.download_service.os.replace", side_effect=OSError("disk full")
        ),
    ):
        caplog.clear()
        caplog.set_level("WARNING", logger="arxiv_browser.services.download_service")
        ok = await download_pdf(
            paper=paper,
            config=UserConfig(),
            client=client,
            timeout_seconds=30,
        )

    assert ok.success is False
    assert ok.failure == DownloadFailure.DISK
    assert target.exists() is False
    assert list(target.parent.glob(".*.tmp")) == []
    assert "PDF download failed for 2401.50002" in caplog.text


@pytest.mark.asyncio
async def test_download_pdf_logs_cleanup_failure_after_stream_error(
    make_paper, tmp_path, caplog
) -> None:
    paper = make_paper(arxiv_id="2401.50003")
    target = tmp_path / "pdfs" / "2401.50003.pdf"
    response = _FakeResponse([], error=OSError("network"))
    client = SimpleNamespace(stream=MagicMock(return_value=_StreamContext(response)))

    with (
        patch(
            "arxiv_browser.services.download_service.get_pdf_url",
            return_value="https://example/pdf",
        ),
        patch("arxiv_browser.services.download_service.get_pdf_download_path", return_value=target),
        patch("arxiv_browser.services.download_service.os.unlink", side_effect=OSError("cleanup")),
    ):
        caplog.clear()
        caplog.set_level("WARNING", logger="arxiv_browser.services.download_service")
        ok = await download_pdf(
            paper=paper,
            config=UserConfig(),
            client=client,
            timeout_seconds=30,
        )

    assert ok.success is False
    assert ok.failure == DownloadFailure.DISK
    assert "PDF download failed for 2401.50003" in caplog.text
    assert "Failed to clean temp PDF file" in caplog.text


@pytest.mark.asyncio
async def test_download_pdf_returns_false_when_temp_file_is_never_created(
    make_paper, tmp_path, caplog
) -> None:
    paper = make_paper(arxiv_id="2401.50004")
    target = tmp_path / "pdfs" / "2401.50004.pdf"
    client = SimpleNamespace(stream=MagicMock())

    with (
        patch(
            "arxiv_browser.services.download_service.get_pdf_url",
            return_value="https://example/pdf",
        ),
        patch("arxiv_browser.services.download_service.get_pdf_download_path", return_value=target),
        patch("pathlib.Path.mkdir", side_effect=OSError("mkdir failed")),
    ):
        caplog.clear()
        caplog.set_level("WARNING", logger="arxiv_browser.services.download_service")
        ok = await download_pdf(
            paper=paper,
            config=UserConfig(),
            client=client,
            timeout_seconds=30,
        )

    assert ok.success is False
    assert ok.failure == DownloadFailure.DISK
    assert "PDF download failed for 2401.50004" in caplog.text
