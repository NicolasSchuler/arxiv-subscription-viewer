"""Internal PDF download service helpers."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum

import httpx

from arxiv_browser.export import get_pdf_download_path, get_pdf_url
from arxiv_browser.models import Paper, UserConfig

logger = logging.getLogger(__name__)


class DownloadFailure(StrEnum):
    """Categorised download failure reasons for user-facing messages."""

    NETWORK = "network"
    HTTP_ERROR = "http_error"
    INVALID_CONTENT = "invalid_content"
    DISK = "disk"
    NOT_FOUND = "not_found"


@dataclass(slots=True, frozen=True)
class DownloadResult:
    """Structured outcome of a single PDF download attempt."""

    success: bool
    failure: DownloadFailure | None = None
    detail: str = ""


class DownloadContentError(RuntimeError):
    """Raised when a successful response does not contain PDF bytes."""


PDF_SIGNATURE = b"%PDF-"


def _classify_failure(exc: Exception) -> DownloadFailure:
    """Map an exception to a user-facing failure category."""
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 404:
            return DownloadFailure.NOT_FOUND
        return DownloadFailure.HTTP_ERROR
    if isinstance(exc, DownloadContentError):
        return DownloadFailure.INVALID_CONTENT
    if isinstance(exc, OSError):
        return DownloadFailure.DISK
    return DownloadFailure.NETWORK


def _validate_pdf_response_start(response: httpx.Response, prefix: bytes) -> None:
    """Reject non-PDF responses before they can replace a cached PDF."""
    if prefix.lstrip().startswith(PDF_SIGNATURE):
        return
    content_type = response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip()
    detail = f"content-type {content_type or 'unknown'}"
    raise DownloadContentError(f"downloaded response is not a PDF ({detail})")


async def download_pdf(
    *,
    paper: Paper,
    config: UserConfig,
    client: httpx.AsyncClient,
    timeout_seconds: int,
) -> DownloadResult:
    """Download a single PDF using atomic temp-file replacement."""
    url = get_pdf_url(paper)
    path = get_pdf_download_path(paper, config)
    tmp_path: str | None = None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
        )

        async def _stream_to_tmp(active_client: httpx.AsyncClient) -> None:
            """Stream the PDF response body into the temporary file."""
            with os.fdopen(fd, "wb") as tmp_file:
                async with active_client.stream(
                    "GET",
                    url,
                    timeout=timeout_seconds,
                    follow_redirects=True,
                ) as response:
                    response.raise_for_status()
                    prefix = b""
                    validated = False
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        if not validated:
                            prefix += bytes(chunk)
                            if len(prefix) < len(PDF_SIGNATURE):
                                continue
                            _validate_pdf_response_start(response, prefix)
                            tmp_file.write(prefix)
                            prefix = b""
                            validated = True
                            continue
                        tmp_file.write(chunk)
                    if not validated:
                        _validate_pdf_response_start(response, prefix)

        await _stream_to_tmp(client)

        os.replace(tmp_path, path)
        return DownloadResult(success=True)
    except (httpx.HTTPError, DownloadContentError, OSError) as exc:
        logger.warning(
            "PDF download failed for %s from %s to %s: %s",
            paper.arxiv_id,
            url,
            path,
            exc,
            exc_info=True,
        )
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "Failed to clean temp PDF file %s after error for %s: %s",
                    tmp_path,
                    paper.arxiv_id,
                    cleanup_exc,
                    exc_info=True,
                )
        failure = _classify_failure(exc)
        return DownloadResult(success=False, failure=failure, detail=str(exc)[:200])


__all__ = [
    "DownloadContentError",
    "DownloadFailure",
    "DownloadResult",
    "download_pdf",
]
