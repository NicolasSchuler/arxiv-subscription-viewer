"""Internal PDF download service helpers."""

from __future__ import annotations

import os
import tempfile

import httpx

from arxiv_browser.export import get_pdf_download_path, get_pdf_url
from arxiv_browser.models import Paper, UserConfig


async def download_pdf(
    *,
    paper: Paper,
    config: UserConfig,
    client: httpx.AsyncClient | None,
    timeout_seconds: int,
) -> bool:
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
            with os.fdopen(fd, "wb") as tmp_file:
                async with active_client.stream(
                    "GET",
                    url,
                    timeout=timeout_seconds,
                    follow_redirects=True,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            tmp_file.write(chunk)

        if client is not None:
            await _stream_to_tmp(client)
        else:
            async with httpx.AsyncClient() as tmp_client:
                await _stream_to_tmp(tmp_client)

        os.replace(tmp_path, path)
        return True
    except (httpx.HTTPError, OSError):
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return False


__all__ = [
    "download_pdf",
]
