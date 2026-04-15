"""HuggingFace Daily Papers API client, data models, and SQLite cache.

Provides async functions for fetching trending paper data from the
HuggingFace Daily Papers API, with persistent SQLite caching.

All API functions accept an httpx.AsyncClient and never raise — callers
get empty list on failure for graceful degradation.
"""

from __future__ import annotations

__all__ = [
    "HF_DEFAULT_CACHE_TTL_HOURS",
    "HFDailyCacheSnapshot",
    "HuggingFacePaper",
    "fetch_hf_daily_papers",
    "get_hf_db_path",
    "init_hf_db",
    "load_hf_daily_cache",
    "load_hf_daily_cache_snapshot",
    "parse_hf_paper_response",
    "save_hf_daily_cache",
]

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, overload

import httpx

from arxiv_browser.config import _safe_get
from arxiv_browser.http_retry import retry_with_backoff

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

HF_API_BASE = "https://huggingface.co/api/daily_papers"
HF_DB_FILENAME = "huggingface.db"
HF_DEFAULT_CACHE_TTL_HOURS = 6  # Trending data changes frequently
HF_REQUEST_TIMEOUT = 15  # seconds
HF_MAX_RETRIES = 3
HF_INITIAL_BACKOFF = 1.0  # seconds, doubles each retry

# ============================================================================
# Data Model
# ============================================================================


@dataclass(slots=True)
class HuggingFacePaper:
    """HuggingFace Daily Papers metadata for a single paper."""

    arxiv_id: str
    title: str
    upvotes: int
    num_comments: int
    ai_summary: str  # Empty string if unavailable
    ai_keywords: tuple[str, ...]  # Immutable for caching
    github_repo: str  # Empty string if no repo
    github_stars: int  # 0 if no repo


@dataclass(slots=True, frozen=True)
class HFDailyCacheSnapshot:
    """Resolved cache state for the daily HuggingFace snapshot."""

    status: Literal["miss", "empty", "found"]
    papers: dict[str, HuggingFacePaper]


# ============================================================================
# Response Parsing
# ============================================================================


def parse_hf_paper_response(item: dict) -> HuggingFacePaper | None:
    """Parse a single HF daily papers API item. Returns None if missing essential fields."""
    if not isinstance(item, dict):
        return None
    paper = item.get("paper") or {}
    if not isinstance(paper, dict):
        return None
    arxiv_id = paper.get("id", "")
    if not arxiv_id:
        return None

    title = paper.get("title") or ""

    # Upvotes are on the paper sub-object
    upvotes = paper.get("upvotes") or 0
    if not isinstance(upvotes, int):
        upvotes = 0

    # Comments count is a top-level field
    num_comments = item.get("numComments") or 0
    if not isinstance(num_comments, int):
        num_comments = 0

    # AI summary — may be absent or null
    ai_summary = paper.get("ai_summary") or ""
    if not isinstance(ai_summary, str):
        ai_summary = ""

    # AI keywords — array of strings
    raw_keywords = paper.get("ai_keywords") or []
    if not isinstance(raw_keywords, list):
        raw_keywords = []
    ai_keywords = tuple(kw for kw in raw_keywords if isinstance(kw, str))

    # GitHub repo info (optional)
    github_repo = paper.get("githubRepo") or ""
    if not isinstance(github_repo, str):
        github_repo = ""
    github_stars = paper.get("githubStars") or 0
    if not isinstance(github_stars, int):
        github_stars = 0

    return HuggingFacePaper(
        arxiv_id=arxiv_id,
        title=title,
        upvotes=upvotes,
        num_comments=num_comments,
        ai_summary=ai_summary,
        ai_keywords=ai_keywords,
        github_repo=github_repo,
        github_stars=github_stars,
    )


# ============================================================================
# API Function (async, accepts httpx.AsyncClient)
# ============================================================================


@overload
async def fetch_hf_daily_papers(
    client: httpx.AsyncClient,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> list[HuggingFacePaper]: ...


@overload
async def fetch_hf_daily_papers(
    client: httpx.AsyncClient,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[list[HuggingFacePaper], bool]: ...


async def fetch_hf_daily_papers(
    client: httpx.AsyncClient,
    timeout: int = HF_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[HuggingFacePaper] | tuple[list[HuggingFacePaper], bool]:
    """Fetch today's trending papers from HuggingFace Daily Papers.

    Returns empty list on failure. Never raises.

    When include_status=True, returns (papers, complete) where complete=False
    means the request or parse path failed.
    """

    async def _do_request() -> httpx.Response:
        response = await client.get(HF_API_BASE, timeout=timeout)
        response.raise_for_status()
        return response

    try:
        response = await retry_with_backoff(
            _do_request,
            max_retries=HF_MAX_RETRIES - 1,
            backoff_base=HF_INITIAL_BACKOFF,
            operation="HF daily papers",
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info("HF daily papers endpoint not found")
        else:
            logger.warning("HF API returned %d", exc.response.status_code)
        return ([], False) if include_status else []
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
        logger.warning("HF API timeout/connection error after retries")
        return ([], False) if include_status else []
    except httpx.HTTPError:
        logger.warning("HF API HTTP error", exc_info=True)
        return ([], False) if include_status else []

    try:
        data = response.json()
    except ValueError:
        logger.warning("HF API returned invalid JSON", exc_info=True)
        return ([], False) if include_status else []
    if not isinstance(data, list):
        logger.warning("HF API returned non-list response")
        return ([], False) if include_status else []
    papers: list[HuggingFacePaper] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        parsed = parse_hf_paper_response(item)
        if parsed is not None:
            papers.append(parsed)
    if include_status:
        return papers, True
    return papers


def _coerce_int(value: Any, default: int = 0) -> int:
    return _safe_get({"v": value}, "v", default, int)


def _coerce_str(value: Any, default: str = "") -> str:
    return _safe_get({"v": value}, "v", default, str)


# ============================================================================
# SQLite Cache
# ============================================================================


def get_hf_db_path() -> Path:
    """Get the path to the HuggingFace SQLite cache."""
    from arxiv_browser.database import resolve_db_path

    return resolve_db_path(HF_DB_FILENAME)


def init_hf_db(db_path: Path) -> None:
    """Create HF cache table if it doesn't exist.

    Raises sqlite3.OperationalError if the parent directory cannot be created.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise sqlite3.OperationalError(f"Cannot create DB directory: {e}") from e
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hf_daily_papers ("
            "  arxiv_id TEXT PRIMARY KEY,"
            "  payload_json TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hf_daily_fetch_state ("
            "  cache_key TEXT PRIMARY KEY,"
            "  status TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )


def _hf_paper_to_json(paper: HuggingFacePaper) -> str:
    """Serialize a HuggingFacePaper to JSON string."""
    return json.dumps(
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "upvotes": paper.upvotes,
            "num_comments": paper.num_comments,
            "ai_summary": paper.ai_summary,
            "ai_keywords": list(paper.ai_keywords),
            "github_repo": paper.github_repo,
            "github_stars": paper.github_stars,
        },
        ensure_ascii=False,
    )


def _json_to_hf_paper(payload: str) -> HuggingFacePaper | None:
    """Deserialize a JSON string to HuggingFacePaper."""
    try:
        d = json.loads(payload)
        if not isinstance(d, dict):
            return None
        arxiv_id = _coerce_str(d.get("arxiv_id"))
        if not arxiv_id:
            return None
        raw_keywords = d.get("ai_keywords", ())
        if not isinstance(raw_keywords, (list, tuple)):
            raw_keywords = ()
        return HuggingFacePaper(
            arxiv_id=arxiv_id,
            title=_coerce_str(d.get("title", "")),
            upvotes=_coerce_int(d.get("upvotes", 0)),
            num_comments=_coerce_int(d.get("num_comments", 0)),
            ai_summary=_coerce_str(d.get("ai_summary", "")),
            ai_keywords=tuple(kw for kw in raw_keywords if isinstance(kw, str)),
            github_repo=_coerce_str(d.get("github_repo", "")),
            github_stars=_coerce_int(d.get("github_stars", 0)),
        )
    except (TypeError, json.JSONDecodeError):
        logger.warning("Failed to deserialize HF paper from cache", exc_info=True)
        return None


def _is_fresh(fetched_at_str: str, ttl_hours: int) -> bool:
    """Check if a cached entry is still within its TTL (hours-based)."""
    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        age_hours = (now - fetched_at).total_seconds() / 3600
        return age_hours < ttl_hours
    except (ValueError, TypeError):
        return False


_HF_DAILY_CACHE_KEY = "daily"


def _load_hf_fetch_state(
    conn: sqlite3.Connection,
    ttl_hours: int,
) -> tuple[Literal["empty", "found"] | None, str | None]:
    """Load fresh daily snapshot state from metadata, if present."""
    row = conn.execute(
        "SELECT status, fetched_at FROM hf_daily_fetch_state WHERE cache_key = ?",
        (_HF_DAILY_CACHE_KEY,),
    ).fetchone()
    if row is None:
        return None, None
    status, fetched_at = row
    if not isinstance(status, str) or not isinstance(fetched_at, str):
        return None, None
    if not _is_fresh(fetched_at, ttl_hours):
        return None, None
    if status not in {"empty", "found"}:
        return None, None
    if status == "empty":
        return "empty", fetched_at
    return "found", fetched_at


def load_hf_daily_cache_snapshot(
    db_path: Path,
    ttl_hours: int = HF_DEFAULT_CACHE_TTL_HOURS,
) -> HFDailyCacheSnapshot:
    """Load cached HF daily papers and preserve explicit empty-state metadata."""
    if not db_path.exists():
        return HFDailyCacheSnapshot(status="miss", papers={})
    try:
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            status, _ = _load_hf_fetch_state(conn, ttl_hours)
            if status == "empty":
                return HFDailyCacheSnapshot(status="empty", papers={})

            rows = conn.execute("SELECT payload_json, fetched_at FROM hf_daily_papers").fetchall()
            if not rows:
                return HFDailyCacheSnapshot(status="miss", papers={})

            # Check freshness of the first entry (all saved at same time)
            _, fetched_at = rows[0]
            if not _is_fresh(fetched_at, ttl_hours):
                return HFDailyCacheSnapshot(status="miss", papers={})

            result: dict[str, HuggingFacePaper] = {}
            for payload, _ in rows:
                paper = _json_to_hf_paper(payload)
                if paper is not None:
                    result[paper.arxiv_id] = paper
            if result:
                return HFDailyCacheSnapshot(status="found", papers=result)
            if status == "found":
                return HFDailyCacheSnapshot(status="miss", papers={})
            return HFDailyCacheSnapshot(status="miss", papers={})
    except sqlite3.Error:
        logger.warning("Failed to load HF cache", exc_info=True)
        return HFDailyCacheSnapshot(status="miss", papers={})


def load_hf_daily_cache(
    db_path: Path, ttl_hours: int = HF_DEFAULT_CACHE_TTL_HOURS
) -> dict[str, HuggingFacePaper] | None:
    """Load cached HF daily papers if they exist and are fresh.

    Returns the entire cached dict (keyed by arxiv_id), or None if
    stale, empty, or not yet cached. This reflects the bulk-fetch
    nature of the API — all rows share the same fetch time.
    """
    snapshot = load_hf_daily_cache_snapshot(db_path, ttl_hours)
    if snapshot.status == "found":
        return snapshot.papers
    return None


def save_hf_daily_cache(db_path: Path, papers: list[HuggingFacePaper]) -> None:
    """Persist HF daily papers to SQLite cache.

    Replaces all existing rows since this is a daily snapshot.
    """
    try:
        init_hf_db(db_path)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("DELETE FROM hf_daily_papers")
            status = "empty"
            for paper in papers:
                payload = _hf_paper_to_json(paper)
                conn.execute(
                    "INSERT INTO hf_daily_papers (arxiv_id, payload_json, fetched_at) "
                    "VALUES (?, ?, ?)",
                    (paper.arxiv_id, payload, now),
                )
                status = "found"
            conn.execute(
                "INSERT OR REPLACE INTO hf_daily_fetch_state (cache_key, status, fetched_at) "
                "VALUES (?, ?, ?)",
                (_HF_DAILY_CACHE_KEY, status, now),
            )
    except sqlite3.Error:
        logger.warning("Failed to save HF cache", exc_info=True)
