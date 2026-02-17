"""Tests for download service helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arxiv_browser.models import UserConfig
from arxiv_browser.services.download_service import download_pdf


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
async def test_download_pdf_success_and_failure(make_paper, tmp_path) -> None:
    paper = make_paper(arxiv_id="2401.50001")
    target = tmp_path / "pdfs" / "2401.50001.pdf"

    response = _FakeResponse([b"%PDF-1.4", b" body"])
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

    assert ok is True
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
        client.stream = MagicMock(side_effect=OSError("network"))
        ok = await download_pdf(
            paper=paper,
            config=UserConfig(),
            client=client,
            timeout_seconds=30,
        )

    assert ok is False
    assert list(target.parent.glob(".*.tmp")) == []
