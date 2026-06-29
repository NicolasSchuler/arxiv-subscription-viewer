"""Paper-content helpers shared by browser and LLM flows."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import httpx

from arxiv_browser.export import get_pdf_download_path, get_pdf_url
from arxiv_browser.models import Paper, UserConfig
from arxiv_browser.parsing import extract_text_from_html
from arxiv_browser.sources import is_arxiv_paper

logger = logging.getLogger("arxiv_browser.browser")

MAX_PAPER_CONTENT_LENGTH = 60_000  # ~15k tokens; truncate fetched paper content
ARXIV_HTML_TIMEOUT = 30  # Seconds to wait for arXiv HTML fetch
SUMMARY_HTML_TIMEOUT = 10  # Faster timeout for summary generation path
PAPER_CONTENT_CACHE_TTL_DAYS = 7
PAPER_CONTENT_PDF_FALLBACK = True
PDF_FALLBACK_TIMEOUT = 60


@dataclass(slots=True, frozen=True)
class PaperContentFetchRequest:
    """Inputs for cache-aware full-paper content retrieval."""

    paper: Paper
    client: httpx.AsyncClient | None = None
    db_path: Path | None = None
    config: UserConfig | None = None
    timeout: int = ARXIV_HTML_TIMEOUT


@dataclass(slots=True, frozen=True)
class PaperContentFetchResult:
    """Resolved paper content and the source that produced it."""

    content: str
    source: str
    cached: bool = False


def _abstract_fallback(paper: Paper) -> str:
    abstract = paper.abstract or paper.abstract_raw or ""
    return f"Abstract:\n{abstract}" if abstract else ""


def _init_paper_content_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_content ("
            "  arxiv_id TEXT PRIMARY KEY,"
            "  source TEXT NOT NULL,"
            "  content_hash TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL,"
            "  content TEXT NOT NULL"
            ")"
        )


def _load_cached_paper_content(
    db_path: Path | None,
    arxiv_id: str,
    ttl_days: int,
) -> PaperContentFetchResult | None:
    if db_path is None or not db_path.exists():
        return None
    try:
        _init_paper_content_table(db_path)
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            row = conn.execute(
                "SELECT source, fetched_at, content FROM paper_content WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()
        if not row:
            return None
        source, fetched_at, content = row
        fetched = datetime.fromisoformat(str(fetched_at))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=UTC)
        if datetime.now(UTC) - fetched > timedelta(days=max(1, ttl_days)):
            return None
        return PaperContentFetchResult(content=str(content), source=str(source), cached=True)
    except (sqlite3.Error, ValueError, OSError):
        logger.warning("Failed to load cached paper content for %s", arxiv_id, exc_info=True)
        return None


def _save_paper_content(
    db_path: Path | None,
    arxiv_id: str,
    source: str,
    content: str,
) -> None:
    if db_path is None or not content:
        return
    try:
        _init_paper_content_table(db_path)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO paper_content "
                "(arxiv_id, source, content_hash, fetched_at, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (arxiv_id, source, content_hash, datetime.now(UTC).isoformat(), content),
            )
    except (sqlite3.Error, OSError):
        logger.warning("Failed to save paper content for %s", arxiv_id, exc_info=True)


async def _get_with_temp_client(
    client: httpx.AsyncClient | None,
    url: str,
    *,
    timeout: int,
) -> httpx.Response:
    if client is not None:
        return await client.get(url, timeout=timeout, follow_redirects=True)
    async with httpx.AsyncClient() as tmp_client:
        return await tmp_client.get(url, timeout=timeout, follow_redirects=True)


async def _fetch_html_content(request: PaperContentFetchRequest) -> str | None:
    if not is_arxiv_paper(request.paper):
        return None
    html_url = f"https://arxiv.org/html/{request.paper.arxiv_id}"
    try:
        response = await _get_with_temp_client(
            request.client,
            html_url,
            timeout=request.timeout,
        )
        if response.status_code == 200:
            text = await asyncio.to_thread(extract_text_from_html, response.text)
            if text:
                return text[:MAX_PAPER_CONTENT_LENGTH]
        else:
            logger.warning(
                "arXiv HTML fetch returned %d for %s",
                response.status_code,
                request.paper.arxiv_id,
            )
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to fetch HTML for %s", request.paper.arxiv_id, exc_info=True)
    return None


def _read_existing_pdf(paper: Paper, config: UserConfig) -> bytes | None:
    if not is_arxiv_paper(paper):
        return None
    path = get_pdf_download_path(paper, config)
    try:
        if path.is_file():
            return path.read_bytes()
    except OSError:
        logger.warning("Failed to read cached PDF for %s", paper.arxiv_id, exc_info=True)
    return None


async def _download_pdf_bytes(
    paper: Paper,
    client: httpx.AsyncClient | None,
    timeout: int,
) -> bytes | None:
    if not is_arxiv_paper(paper):
        return None
    try:
        response = await _get_with_temp_client(
            client,
            get_pdf_url(paper),
            timeout=max(timeout, PDF_FALLBACK_TIMEOUT),
        )
        if response.status_code != 200:
            logger.warning(
                "arXiv PDF fetch returned %d for %s",
                response.status_code,
                paper.arxiv_id,
            )
            return None
        content = getattr(response, "content", b"")
        return bytes(content) if isinstance(content, bytes | bytearray) else None
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to fetch PDF for %s", paper.arxiv_id, exc_info=True)
        return None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        logger.warning("pypdf is not installed; skipping PDF text fallback")
        return ""

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_text = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        logger.warning("Failed to extract PDF text", exc_info=True)
        return ""
    text = "\n\n".join(part.strip() for part in page_text if part.strip())
    return text[:MAX_PAPER_CONTENT_LENGTH]


async def _fetch_pdf_content(request: PaperContentFetchRequest) -> str | None:
    if not is_arxiv_paper(request.paper):
        return None
    config = request.config or UserConfig()
    try:
        pdf_bytes = await asyncio.to_thread(_read_existing_pdf, request.paper, config)
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to inspect cached PDF for %s", request.paper.arxiv_id, exc_info=True)
        pdf_bytes = None
    if pdf_bytes is None:
        pdf_bytes = await _download_pdf_bytes(request.paper, request.client, request.timeout)
    if not pdf_bytes:
        return None
    try:
        text = await asyncio.to_thread(_extract_text_from_pdf_bytes, pdf_bytes)
    except (httpx.HTTPError, OSError):
        logger.warning(
            "Failed to extract cached PDF text for %s", request.paper.arxiv_id, exc_info=True
        )
        return None
    return text or None


async def fetch_paper_content(
    request: PaperContentFetchRequest,
) -> PaperContentFetchResult:
    """Fetch full paper content from cache, arXiv HTML, PDF text, or abstract."""
    config = request.config or UserConfig()
    cached = None
    if request.db_path is not None:
        cached = await asyncio.to_thread(
            _load_cached_paper_content,
            request.db_path,
            request.paper.arxiv_id,
            config.paper_content_cache_ttl_days,
        )
    if cached is not None:
        return cached

    html_text = await _fetch_html_content(request)
    if html_text:
        await asyncio.to_thread(
            _save_paper_content,
            request.db_path,
            request.paper.arxiv_id,
            "html",
            html_text,
        )
        return PaperContentFetchResult(content=html_text, source="html")

    if config.paper_content_pdf_fallback:
        pdf_text = await _fetch_pdf_content(request)
        if pdf_text:
            await asyncio.to_thread(
                _save_paper_content,
                request.db_path,
                request.paper.arxiv_id,
                "pdf",
                pdf_text,
            )
            return PaperContentFetchResult(content=pdf_text, source="pdf")

    return PaperContentFetchResult(content=_abstract_fallback(request.paper), source="abstract")


async def _fetch_paper_content_async(
    paper: Paper,
    client: httpx.AsyncClient | None = None,
    timeout: int = ARXIV_HTML_TIMEOUT,
    *,
    db_path: Path | None = None,
    config: UserConfig | None = None,
) -> str:
    """Fetch canonical paper content for LLM workflows."""
    result = await fetch_paper_content(
        PaperContentFetchRequest(
            paper=paper,
            client=client,
            db_path=db_path,
            config=config,
            timeout=timeout,
        )
    )
    return result.content


async def fetch_browser_paper_content(app, paper: Paper) -> str:
    """Fetch paper content with browser-owned network, config, and cache state."""
    return await _fetch_paper_content_async(
        paper,
        app._http_client,
        SUMMARY_HTML_TIMEOUT,
        db_path=app._cache_db_path,
        config=app._config,
    )


__all__ = [
    "ARXIV_HTML_TIMEOUT",
    "MAX_PAPER_CONTENT_LENGTH",
    "PAPER_CONTENT_CACHE_TTL_DAYS",
    "PAPER_CONTENT_PDF_FALLBACK",
    "SUMMARY_HTML_TIMEOUT",
    "PaperContentFetchRequest",
    "PaperContentFetchResult",
    "_extract_text_from_pdf_bytes",
    "_fetch_paper_content_async",
    "_load_cached_paper_content",
    "_save_paper_content",
    "fetch_browser_paper_content",
    "fetch_paper_content",
]
