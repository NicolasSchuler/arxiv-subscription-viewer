"""Semantic Scholar API client, data models, and SQLite cache.

Provides async functions for fetching paper metadata and recommendations
from the Semantic Scholar Graph API, with persistent SQLite caching.

All API functions accept an httpx.AsyncClient and never raise — callers
get None / empty list on failure for graceful degradation.
"""

from __future__ import annotations

__all__ = [
    "S2_CITATIONS_PAGE_SIZE",
    "S2_CITATIONS_SCAN_CAP",
    "S2_CITATION_GRAPH_CACHE_TTL_DAYS",
    "S2_DEFAULT_CACHE_TTL_DAYS",
    "S2_REC_CACHE_TTL_DAYS",
    "CitationEntry",
    "SemanticScholarPaper",
    "fetch_s2_citations",
    "fetch_s2_paper",
    "fetch_s2_recommendations",
    "fetch_s2_references",
    "get_s2_db_path",
    "has_s2_citation_graph_cache",
    "init_s2_db",
    "load_s2_citation_graph",
    "load_s2_paper",
    "load_s2_recommendations",
    "parse_citation_entry",
    "parse_s2_paper_response",
    "save_s2_citation_graph",
    "save_s2_paper",
    "save_s2_recommendations",
]

import asyncio
import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, overload

import httpx
from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

CONFIG_APP_NAME = "arxiv-browser"

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
S2_PAPER_FIELDS = "paperId,citationCount,influentialCitationCount,tldr,fieldsOfStudy,year,url"
S2_REC_FIELDS = (
    "paperId,externalIds,title,citationCount,influentialCitationCount,tldr,year,url,abstract"
)
S2_DB_FILENAME = "semantic_scholar.db"
S2_DEFAULT_CACHE_TTL_DAYS = 7
S2_REC_CACHE_TTL_DAYS = 3
S2_CITATION_FIELDS = "paperId,externalIds,title,authors,year,citationCount,url"
S2_CITATION_GRAPH_CACHE_TTL_DAYS = 3
S2_MAX_REFERENCES = 100
S2_MAX_CITATIONS = 50
S2_CITATIONS_PAGE_SIZE = 100
S2_CITATIONS_SCAN_CAP = 1000
S2_REQUEST_TIMEOUT = 20  # seconds
S2_MAX_RETRIES = 3
S2_INITIAL_BACKOFF = 1.0  # seconds, doubles each retry

# ============================================================================
# Data Model
# ============================================================================


@dataclass(slots=True)
class SemanticScholarPaper:
    """Semantic Scholar paper metadata."""

    arxiv_id: str
    s2_paper_id: str
    citation_count: int
    influential_citation_count: int
    tldr: str  # Empty string if unavailable
    fields_of_study: tuple[str, ...]  # Immutable for caching
    year: int | None
    url: str
    # Optional fields for recommendations
    title: str = ""
    abstract: str = ""


@dataclass(slots=True)
class CitationEntry:
    """A paper in a citation graph (reference or citation)."""

    s2_paper_id: str
    arxiv_id: str  # Empty string if no arXiv ID
    title: str
    authors: str  # Comma-joined author names
    year: int | None
    citation_count: int
    url: str  # arXiv URL if arxiv_id present, else S2 URL


# ============================================================================
# Response Parsing
# ============================================================================


def parse_s2_paper_response(data: dict, arxiv_id: str = "") -> SemanticScholarPaper | None:
    """Parse S2 API JSON into dataclass. Returns None if essential fields missing."""
    paper_id = data.get("paperId")
    if not paper_id:
        return None

    # Extract arXiv ID from externalIds if not provided
    if not arxiv_id:
        external_ids = data.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")

    # Parse TLDR — it's an object with a "text" field
    tldr_obj = data.get("tldr")
    tldr = ""
    if isinstance(tldr_obj, dict):
        tldr = tldr_obj.get("text", "")

    # Parse fields of study
    fos_raw = data.get("fieldsOfStudy") or []
    fields_of_study = tuple(f for f in fos_raw if isinstance(f, str))

    return SemanticScholarPaper(
        arxiv_id=arxiv_id,
        s2_paper_id=paper_id,
        citation_count=data.get("citationCount") or 0,
        influential_citation_count=data.get("influentialCitationCount") or 0,
        tldr=tldr,
        fields_of_study=fields_of_study,
        year=data.get("year"),
        url=data.get("url") or "",
        title=data.get("title") or "",
        abstract=data.get("abstract") or "",
    )


def parse_citation_entry(data: dict) -> CitationEntry | None:
    """Parse an S2 citation/reference wrapper into CitationEntry.

    The ``data`` dict is the inner paper object (e.g. ``citedPaper`` or
    ``citingPaper``) from the S2 references/citations endpoint.
    """
    paper_id = data.get("paperId")
    if not paper_id:
        return None
    external_ids = data.get("externalIds")
    if not isinstance(external_ids, dict):
        external_ids = {}
    arxiv_id = external_ids.get("ArXiv", "")
    authors_raw = data.get("authors") or []
    authors = ", ".join(
        a.get("name", "")
        for a in authors_raw
        if isinstance(a, dict) and isinstance(a.get("name"), str) and a.get("name")
    )
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else data.get("url") or ""
    return CitationEntry(
        s2_paper_id=paper_id,
        arxiv_id=arxiv_id,
        title=data.get("title") or "Unknown Title",
        authors=authors,
        year=data.get("year"),
        citation_count=data.get("citationCount") or 0,
        url=url,
    )


# ============================================================================
# API Functions (async, accept httpx.AsyncClient)
# ============================================================================


async def _s2_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str],
    api_key: str,
    timeout: int,
    label: str,
) -> httpx.Response | None:
    """Send a GET request with exponential backoff retry on 429/5xx.

    Returns the response on 200, or None on terminal failure (404, other 4xx,
    exhausted retries, timeout, or network error). Never raises.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    backoff = S2_INITIAL_BACKOFF
    for attempt in range(S2_MAX_RETRIES):
        try:
            response = await client.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                logger.info("%s not found", label)
                return None
            if response.status_code in (429, 500, 502, 503, 504) and attempt < S2_MAX_RETRIES - 1:
                jitter = random.uniform(0, backoff * 0.5)
                logger.info(
                    "%s %d, retrying in %.1fs (attempt %d/%d)",
                    label,
                    response.status_code,
                    backoff + jitter,
                    attempt + 1,
                    S2_MAX_RETRIES,
                )
                await asyncio.sleep(backoff + jitter)
                backoff *= 2
                continue
            logger.warning("%s returned %d", label, response.status_code)
            return None
        except httpx.TimeoutException:
            if attempt < S2_MAX_RETRIES - 1:
                logger.info(
                    "%s timeout, retrying (attempt %d/%d)",
                    label,
                    attempt + 1,
                    S2_MAX_RETRIES,
                )
                jitter = random.uniform(0, backoff * 0.5)
                await asyncio.sleep(backoff + jitter)
                backoff *= 2
                continue
            logger.warning("%s timeout after %d retries", label, S2_MAX_RETRIES)
            return None
        except httpx.HTTPError:
            logger.warning("%s HTTP error", label, exc_info=True)
            return None

    return None


def _parse_json_object(response: httpx.Response, label: str) -> dict[str, Any] | None:
    """Parse a response body as JSON object, returning None on malformed payloads."""
    try:
        payload = response.json()
    except ValueError:
        logger.warning("%s returned invalid JSON", label, exc_info=True)
        return None
    if not isinstance(payload, dict):
        logger.warning("%s returned non-object JSON payload", label)
        return None
    return payload


async def fetch_s2_paper(
    arxiv_id: str,
    client: httpx.AsyncClient,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
) -> SemanticScholarPaper | None:
    """Fetch paper metadata from S2 Graph API. Returns None on failure."""
    response = await _s2_get_with_retry(
        client,
        url=f"{S2_API_BASE}/paper/ARXIV:{arxiv_id}",
        params={"fields": S2_PAPER_FIELDS},
        api_key=api_key,
        timeout=timeout,
        label=f"S2 paper arXiv:{arxiv_id}",
    )
    if response is None:
        return None
    payload = _parse_json_object(response, f"S2 paper arXiv:{arxiv_id}")
    if payload is None:
        return None
    return parse_s2_paper_response(payload, arxiv_id=arxiv_id)


async def fetch_s2_recommendations(
    arxiv_id: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
) -> list[SemanticScholarPaper]:
    """Fetch recommended papers from S2 Recommendations API."""
    response = await _s2_get_with_retry(
        client,
        url=f"{S2_REC_BASE}/papers/forpaper/ARXIV:{arxiv_id}",
        params={"fields": S2_REC_FIELDS, "limit": str(limit)},
        api_key=api_key,
        timeout=timeout,
        label=f"S2 recs arXiv:{arxiv_id}",
    )
    if response is None:
        return []
    payload = _parse_json_object(response, f"S2 recs arXiv:{arxiv_id}")
    if payload is None:
        return []
    papers_data = payload.get("recommendedPapers")
    if not isinstance(papers_data, list):
        logger.warning("S2 recs arXiv:%s returned non-list recommendedPapers", arxiv_id)
        return []
    papers: list[SemanticScholarPaper] = []
    for item in papers_data:
        if not isinstance(item, dict):
            continue
        parsed = parse_s2_paper_response(item)
        if parsed is not None:
            papers.append(parsed)
    return papers


@overload
async def fetch_s2_references(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> list[CitationEntry]: ...


@overload
async def fetch_s2_references(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[list[CitationEntry], bool]: ...


async def fetch_s2_references(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = S2_MAX_REFERENCES,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[CitationEntry] | tuple[list[CitationEntry], bool]:
    """Fetch papers cited by the given paper. Sorted by citation_count desc.

    When include_status=True, returns (entries, complete) where complete=False
    means the request/parse path failed and callers should avoid caching.
    """
    response = await _s2_get_with_retry(
        client,
        url=f"{S2_API_BASE}/paper/{paper_id}/references",
        params={"fields": S2_CITATION_FIELDS, "limit": str(limit)},
        api_key=api_key,
        timeout=timeout,
        label=f"S2 refs {paper_id}",
    )
    if response is None:
        return ([], False) if include_status else []
    payload = _parse_json_object(response, f"S2 refs {paper_id}")
    if payload is None:
        return ([], False) if include_status else []
    data = payload.get("data")
    if not isinstance(data, list):
        logger.warning("S2 refs %s returned non-list data", paper_id)
        return ([], False) if include_status else []
    entries: list[CitationEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        entry = parse_citation_entry(item.get("citedPaper") or {})
        if entry:
            entries.append(entry)
    entries.sort(key=lambda e: e.citation_count, reverse=True)
    if include_status:
        return entries, True
    return entries


@overload
async def fetch_s2_citations(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> list[CitationEntry]: ...


@overload
async def fetch_s2_citations(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = ...,
    api_key: str = ...,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[list[CitationEntry], bool]: ...


async def fetch_s2_citations(
    paper_id: str,
    client: httpx.AsyncClient,
    limit: int = S2_MAX_CITATIONS,
    api_key: str = "",
    timeout: int = S2_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[CitationEntry] | tuple[list[CitationEntry], bool]:
    """Fetch papers citing the given paper. Top N by citation_count.

    Uses an adaptive scan cap based on limit to avoid unnecessary API calls.
    When include_status=True, returns (entries, complete) where complete=False
    means at least one page fetch/parse failed.
    """
    if limit <= 0:
        empty: list[CitationEntry] = []
        return (empty, True) if include_status else empty

    entries: list[CitationEntry] = []
    offset = 0
    scan_cap = min(
        S2_CITATIONS_SCAN_CAP,
        max(S2_CITATIONS_PAGE_SIZE * 2, limit * 4),
    )
    complete = True
    while offset < scan_cap:
        remaining = scan_cap - offset
        page_limit = min(S2_CITATIONS_PAGE_SIZE, remaining)
        label = f"S2 cites {paper_id}"
        response = await _s2_get_with_retry(
            client,
            url=f"{S2_API_BASE}/paper/{paper_id}/citations",
            params={
                "fields": S2_CITATION_FIELDS,
                "limit": str(page_limit),
                "offset": str(offset),
            },
            api_key=api_key,
            timeout=timeout,
            label=label,
        )
        if response is None:
            complete = False
            break

        payload = _parse_json_object(response, label)
        if payload is None:
            complete = False
            break

        data = payload.get("data")
        if not isinstance(data, list):
            logger.warning("S2 cites %s returned non-list data", paper_id)
            complete = False
            break
        if not data:
            break

        for item in data:
            if not isinstance(item, dict):
                continue
            entry = parse_citation_entry(item.get("citingPaper") or {})
            if entry:
                entries.append(entry)

        if len(data) < page_limit:
            break
        offset += page_limit

    entries.sort(key=lambda e: e.citation_count, reverse=True)
    trimmed = entries[:limit]
    if include_status:
        return trimmed, complete
    return trimmed


# ============================================================================
# SQLite Cache
# ============================================================================


def get_s2_db_path() -> Path:
    """Get the path to the Semantic Scholar SQLite cache."""
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / S2_DB_FILENAME


def init_s2_db(db_path: Path) -> None:
    """Create S2 cache tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_papers ("
            "  arxiv_id TEXT PRIMARY KEY,"
            "  payload_json TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_recommendations ("
            "  source_arxiv_id TEXT NOT NULL,"
            "  rank INTEGER NOT NULL,"
            "  payload_json TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL,"
            "  PRIMARY KEY (source_arxiv_id, rank)"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_citation_graph ("
            "  paper_id TEXT NOT NULL,"
            "  direction TEXT NOT NULL,"
            "  rank INTEGER NOT NULL,"
            "  payload_json TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL,"
            "  PRIMARY KEY (paper_id, direction, rank)"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_citation_graph_fetches ("
            "  paper_id TEXT PRIMARY KEY,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )


def _paper_to_json(paper: SemanticScholarPaper) -> str:
    """Serialize a SemanticScholarPaper to JSON string."""
    return json.dumps(
        {
            "arxiv_id": paper.arxiv_id,
            "s2_paper_id": paper.s2_paper_id,
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "tldr": paper.tldr,
            "fields_of_study": list(paper.fields_of_study),
            "year": paper.year,
            "url": paper.url,
            "title": paper.title,
            "abstract": paper.abstract,
        },
        ensure_ascii=False,
    )


def _json_to_paper(payload: str) -> SemanticScholarPaper | None:
    """Deserialize a JSON string to SemanticScholarPaper."""
    try:
        d = json.loads(payload)
        return SemanticScholarPaper(
            arxiv_id=d["arxiv_id"],
            s2_paper_id=d["s2_paper_id"],
            citation_count=d.get("citation_count", 0),
            influential_citation_count=d.get("influential_citation_count", 0),
            tldr=d.get("tldr", ""),
            fields_of_study=tuple(d.get("fields_of_study", ())),
            year=d.get("year"),
            url=d.get("url", ""),
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        logger.warning("Failed to deserialize S2 paper from cache", exc_info=True)
        return None


def _is_fresh(fetched_at_str: str, ttl_days: int) -> bool:
    """Check if a cached entry is still within its TTL."""
    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
        # Ensure timezone-aware comparison
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        age_days = (now - fetched_at).total_seconds() / 86400
        return age_days < ttl_days
    except (ValueError, TypeError):
        return False


def load_s2_paper(
    db_path: Path, arxiv_id: str, ttl_days: int = S2_DEFAULT_CACHE_TTL_DAYS
) -> SemanticScholarPaper | None:
    """Load a cached S2 paper if it exists and is fresh."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT payload_json, fetched_at FROM s2_papers WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()
            if row is None:
                return None
            payload, fetched_at = row
            if not _is_fresh(fetched_at, ttl_days):
                return None
            return _json_to_paper(payload)
    except sqlite3.Error:
        logger.warning("Failed to load S2 cache for %s", arxiv_id, exc_info=True)
        return None


def save_s2_paper(db_path: Path, paper: SemanticScholarPaper) -> None:
    """Persist S2 paper data to the SQLite cache."""
    try:
        init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        payload = _paper_to_json(paper)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_papers (arxiv_id, payload_json, fetched_at) "
                "VALUES (?, ?, ?)",
                (paper.arxiv_id, payload, now),
            )
    except sqlite3.Error:
        logger.warning("Failed to save S2 cache for %s", paper.arxiv_id, exc_info=True)


def load_s2_recommendations(
    db_path: Path,
    arxiv_id: str,
    ttl_days: int = S2_REC_CACHE_TTL_DAYS,
) -> list[SemanticScholarPaper]:
    """Load cached S2 recommendations for a paper."""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT payload_json, fetched_at FROM s2_recommendations "
                "WHERE source_arxiv_id = ? ORDER BY rank",
                (arxiv_id,),
            ).fetchall()
            if not rows:
                return []
            # Check freshness of the first entry (all saved at same time)
            _, fetched_at = rows[0]
            if not _is_fresh(fetched_at, ttl_days):
                return []
            results = []
            for payload, _ in rows:
                paper = _json_to_paper(payload)
                if paper is not None:
                    results.append(paper)
            return results
    except sqlite3.Error:
        logger.warning("Failed to load S2 recommendations for %s", arxiv_id, exc_info=True)
        return []


def save_s2_recommendations(
    db_path: Path,
    source_arxiv_id: str,
    papers: list[SemanticScholarPaper],
) -> None:
    """Persist S2 recommendations to the SQLite cache."""
    try:
        init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(str(db_path)) as conn:
            # Clear old recommendations for this source
            conn.execute(
                "DELETE FROM s2_recommendations WHERE source_arxiv_id = ?",
                (source_arxiv_id,),
            )
            for rank, paper in enumerate(papers):
                payload = _paper_to_json(paper)
                conn.execute(
                    "INSERT INTO s2_recommendations "
                    "(source_arxiv_id, rank, payload_json, fetched_at) "
                    "VALUES (?, ?, ?, ?)",
                    (source_arxiv_id, rank, payload, now),
                )
    except sqlite3.Error:
        logger.warning(
            "Failed to save S2 recommendations for %s",
            source_arxiv_id,
            exc_info=True,
        )


# ============================================================================
# Citation Graph Serialization & Cache
# ============================================================================


def _citation_entry_to_json(entry: CitationEntry) -> str:
    """Serialize a CitationEntry to JSON string."""
    return json.dumps(
        {
            "s2_paper_id": entry.s2_paper_id,
            "arxiv_id": entry.arxiv_id,
            "title": entry.title,
            "authors": entry.authors,
            "year": entry.year,
            "citation_count": entry.citation_count,
            "url": entry.url,
        },
        ensure_ascii=False,
    )


def _json_to_citation_entry(payload: str) -> CitationEntry | None:
    """Deserialize a JSON string to CitationEntry."""
    try:
        d = json.loads(payload)
        return CitationEntry(
            s2_paper_id=d["s2_paper_id"],
            arxiv_id=d.get("arxiv_id", ""),
            title=d.get("title", "Unknown Title"),
            authors=d.get("authors", ""),
            year=d.get("year"),
            citation_count=d.get("citation_count", 0),
            url=d.get("url", ""),
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        logger.warning("Failed to deserialize citation entry from cache", exc_info=True)
        return None


def has_s2_citation_graph_cache(
    db_path: Path, paper_id: str, ttl_days: int = S2_CITATION_GRAPH_CACHE_TTL_DAYS
) -> bool:
    """Check whether citation graph data was fetched recently for this paper."""
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT fetched_at FROM s2_citation_graph_fetches WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if row is None:
                return False
            return _is_fresh(row[0], ttl_days)
    except sqlite3.Error:
        logger.warning(
            "Failed to check citation graph cache marker for %s", paper_id, exc_info=True
        )
        return False


def load_s2_citation_graph(
    db_path: Path,
    paper_id: str,
    direction: str,
    ttl_days: int = S2_CITATION_GRAPH_CACHE_TTL_DAYS,
) -> list[CitationEntry]:
    """Load cached citation graph entries for a paper + direction."""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT payload_json, fetched_at FROM s2_citation_graph "
                "WHERE paper_id = ? AND direction = ? ORDER BY rank",
                (paper_id, direction),
            ).fetchall()
            if not rows:
                return []
            # Check freshness of the first entry (all saved at same time)
            _, fetched_at = rows[0]
            if not _is_fresh(fetched_at, ttl_days):
                return []
            results: list[CitationEntry] = []
            for payload, _ in rows:
                entry = _json_to_citation_entry(payload)
                if entry is not None:
                    results.append(entry)
            return results
    except sqlite3.Error:
        logger.warning(
            "Failed to load citation graph for %s/%s",
            paper_id,
            direction,
            exc_info=True,
        )
        return []


def save_s2_citation_graph(
    db_path: Path,
    paper_id: str,
    direction: str,
    entries: list[CitationEntry],
) -> None:
    """Persist citation graph entries to the SQLite cache."""
    try:
        init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "DELETE FROM s2_citation_graph WHERE paper_id = ? AND direction = ?",
                (paper_id, direction),
            )
            for rank, entry in enumerate(entries):
                payload = _citation_entry_to_json(entry)
                conn.execute(
                    "INSERT INTO s2_citation_graph "
                    "(paper_id, direction, rank, payload_json, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (paper_id, direction, rank, payload, now),
                )
            conn.execute(
                "INSERT OR REPLACE INTO s2_citation_graph_fetches (paper_id, fetched_at) "
                "VALUES (?, ?)",
                (paper_id, now),
            )
    except sqlite3.Error:
        logger.warning(
            "Failed to save citation graph for %s/%s",
            paper_id,
            direction,
            exc_info=True,
        )
