"""Paper-content helpers shared by browser and LLM flows."""

from __future__ import annotations

import asyncio
import logging

import httpx

from arxiv_browser.models import Paper
from arxiv_browser.parsing import extract_text_from_html

logger = logging.getLogger("arxiv_browser.browser")

MAX_PAPER_CONTENT_LENGTH = 60_000  # ~15k tokens; truncate fetched paper content
ARXIV_HTML_TIMEOUT = 30  # Seconds to wait for arXiv HTML fetch


async def _fetch_paper_content_async(
    paper: Paper,
    client: httpx.AsyncClient | None = None,
    timeout: int = ARXIV_HTML_TIMEOUT,
) -> str:
    """Fetch the full paper content from arXiv HTML, falling back to the abstract."""
    html_url = f"https://arxiv.org/html/{paper.arxiv_id}"
    try:
        if client is not None:
            response = await client.get(html_url, timeout=timeout, follow_redirects=True)
        else:
            async with httpx.AsyncClient() as tmp_client:
                response = await tmp_client.get(html_url, timeout=timeout, follow_redirects=True)
        if response.status_code == 200:
            text = await asyncio.to_thread(extract_text_from_html, response.text)
            if text:
                return text[:MAX_PAPER_CONTENT_LENGTH]
        else:
            logger.warning(
                "arXiv HTML fetch returned %d for %s",
                response.status_code,
                paper.arxiv_id,
            )
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to fetch HTML for %s", paper.arxiv_id, exc_info=True)

    abstract = paper.abstract or paper.abstract_raw or ""
    return f"Abstract:\n{abstract}" if abstract else ""


__all__ = ["ARXIV_HTML_TIMEOUT", "MAX_PAPER_CONTENT_LENGTH", "_fetch_paper_content_async"]
