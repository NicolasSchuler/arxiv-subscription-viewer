"""Persistent SQLite cache layer for Semantic Scholar lookups."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from arxiv_browser.semantic_scholar_models import (
    S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    S2_DB_FILENAME,
    S2_DEFAULT_CACHE_TTL_DAYS,
    S2_REC_CACHE_TTL_DAYS,
    CitationEntry,
    S2PaperCacheSnapshot,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)

logger = logging.getLogger(__name__)


def get_s2_db_path() -> Path:
    """Get the path to the Semantic Scholar SQLite cache."""
    from arxiv_browser.database import resolve_db_path

    return resolve_db_path(S2_DB_FILENAME)


def init_s2_db(db_path: Path) -> None:
    """Create S2 cache tables if they don't exist."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise sqlite3.OperationalError(f"Cannot create DB directory: {e}") from e
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_papers ("
            "  arxiv_id TEXT PRIMARY KEY,"
            "  payload_json TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_paper_fetch_state ("
            "  arxiv_id TEXT PRIMARY KEY,"
            "  status TEXT NOT NULL,"
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
            "CREATE TABLE IF NOT EXISTS s2_recommendation_fetch_state ("
            "  source_arxiv_id TEXT PRIMARY KEY,"
            "  status TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
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


def _load_s2_paper_fetch_state(
    conn: sqlite3.Connection,
    arxiv_id: str,
    ttl_days: int,
) -> tuple[Literal["found", "not_found"] | None, str | None]:
    """Load fresh paper fetch metadata, if present."""
    row = conn.execute(
        "SELECT status, fetched_at FROM s2_paper_fetch_state WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None, None
    status, fetched_at = row
    if not isinstance(status, str) or not isinstance(fetched_at, str):
        return None, None
    if not _is_fresh(fetched_at, ttl_days):
        return None, None
    if status not in {"found", "not_found"}:
        return None, None
    if status == "not_found":
        return "not_found", fetched_at
    return "found", fetched_at


def _load_s2_recommendation_fetch_state(
    conn: sqlite3.Connection,
    arxiv_id: str,
    ttl_days: int,
) -> tuple[Literal["found", "empty"] | None, str | None]:
    """Load fresh recommendation fetch metadata, if present."""
    row = conn.execute(
        "SELECT status, fetched_at FROM s2_recommendation_fetch_state WHERE source_arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None, None
    status, fetched_at = row
    if not isinstance(status, str) or not isinstance(fetched_at, str):
        return None, None
    if not _is_fresh(fetched_at, ttl_days):
        return None, None
    if status not in {"found", "empty"}:
        return None, None
    if status == "empty":
        return "empty", fetched_at
    return "found", fetched_at


def load_s2_paper_snapshot(
    db_path: Path,
    arxiv_id: str,
    ttl_days: int = S2_DEFAULT_CACHE_TTL_DAYS,
) -> S2PaperCacheSnapshot:
    """Load a cached S2 paper and preserve explicit not-found metadata."""
    if not db_path.exists():
        return S2PaperCacheSnapshot(status="miss", paper=None)
    try:
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            status, _ = _load_s2_paper_fetch_state(conn, arxiv_id, ttl_days)
            if status == "not_found":
                return S2PaperCacheSnapshot(status="not_found", paper=None)
            row = conn.execute(
                "SELECT payload_json, fetched_at FROM s2_papers WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()
            if row is None:
                return S2PaperCacheSnapshot(status="miss", paper=None)
            payload, fetched_at = row
            if not _is_fresh(fetched_at, ttl_days):
                return S2PaperCacheSnapshot(status="miss", paper=None)
            paper = _json_to_paper(payload)
            if paper is None:
                return S2PaperCacheSnapshot(status="miss", paper=None)
            return S2PaperCacheSnapshot(status="found", paper=paper)
    except sqlite3.Error:
        logger.warning("Failed to load S2 cache for %s", arxiv_id, exc_info=True)
        return S2PaperCacheSnapshot(status="miss", paper=None)


def load_s2_paper(
    db_path: Path,
    arxiv_id: str,
    ttl_days: int = S2_DEFAULT_CACHE_TTL_DAYS,
) -> SemanticScholarPaper | None:
    """Load a cached S2 paper if it exists and is fresh."""
    snapshot = load_s2_paper_snapshot(db_path, arxiv_id, ttl_days)
    if snapshot.status == "found":
        return snapshot.paper
    return None


def save_s2_paper(db_path: Path, paper: SemanticScholarPaper) -> None:
    """Persist S2 paper data to the SQLite cache."""
    try:
        init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        payload = _paper_to_json(paper)
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_papers (arxiv_id, payload_json, fetched_at) "
                "VALUES (?, ?, ?)",
                (paper.arxiv_id, payload, now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO s2_paper_fetch_state (arxiv_id, status, fetched_at) "
                "VALUES (?, ?, ?)",
                (paper.arxiv_id, "found", now),
            )
    except sqlite3.Error:
        logger.warning("Failed to save S2 cache for %s", paper.arxiv_id, exc_info=True)


def save_s2_paper_not_found(db_path: Path, arxiv_id: str) -> None:
    """Persist a fresh 'not found' marker for an S2 paper lookup."""
    try:
        init_s2_db(db_path)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("DELETE FROM s2_papers WHERE arxiv_id = ?", (arxiv_id,))
            conn.execute(
                "INSERT OR REPLACE INTO s2_paper_fetch_state (arxiv_id, status, fetched_at) "
                "VALUES (?, ?, ?)",
                (arxiv_id, "not_found", now),
            )
    except sqlite3.Error:
        logger.warning("Failed to save S2 not-found cache for %s", arxiv_id, exc_info=True)


def load_s2_recommendations_snapshot(
    db_path: Path,
    arxiv_id: str,
    ttl_days: int = S2_REC_CACHE_TTL_DAYS,
) -> S2RecommendationsCacheSnapshot:
    """Load cached S2 recommendations and preserve explicit empty-state metadata."""
    if not db_path.exists():
        return S2RecommendationsCacheSnapshot(status="miss", papers=[])
    try:
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            status, _ = _load_s2_recommendation_fetch_state(conn, arxiv_id, ttl_days)
            if status == "empty":
                return S2RecommendationsCacheSnapshot(status="empty", papers=[])
            rows = conn.execute(
                "SELECT payload_json, fetched_at FROM s2_recommendations "
                "WHERE source_arxiv_id = ? ORDER BY rank",
                (arxiv_id,),
            ).fetchall()
            if not rows:
                return S2RecommendationsCacheSnapshot(status="miss", papers=[])
            # Check freshness of the first entry (all saved at same time)
            _, fetched_at = rows[0]
            if not _is_fresh(fetched_at, ttl_days):
                return S2RecommendationsCacheSnapshot(status="miss", papers=[])
            results: list[SemanticScholarPaper] = []
            for payload, _ in rows:
                paper = _json_to_paper(payload)
                if paper is not None:
                    results.append(paper)
            if results:
                return S2RecommendationsCacheSnapshot(status="found", papers=results)
            return S2RecommendationsCacheSnapshot(status="miss", papers=[])
    except sqlite3.Error:
        logger.warning("Failed to load S2 recommendations for %s", arxiv_id, exc_info=True)
        return S2RecommendationsCacheSnapshot(status="miss", papers=[])


def load_s2_recommendations(
    db_path: Path,
    arxiv_id: str,
    ttl_days: int = S2_REC_CACHE_TTL_DAYS,
) -> list[SemanticScholarPaper]:
    """Load cached S2 recommendations for a paper."""
    snapshot = load_s2_recommendations_snapshot(db_path, arxiv_id, ttl_days)
    if snapshot.status == "found":
        return snapshot.papers
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
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            # Clear old recommendations for this source
            conn.execute(
                "DELETE FROM s2_recommendations WHERE source_arxiv_id = ?",
                (source_arxiv_id,),
            )
            status = "empty"
            for rank, paper in enumerate(papers):
                payload = _paper_to_json(paper)
                conn.execute(
                    "INSERT INTO s2_recommendations "
                    "(source_arxiv_id, rank, payload_json, fetched_at) "
                    "VALUES (?, ?, ?, ?)",
                    (source_arxiv_id, rank, payload, now),
                )
                status = "found"
            conn.execute(
                "INSERT OR REPLACE INTO s2_recommendation_fetch_state "
                "(source_arxiv_id, status, fetched_at) VALUES (?, ?, ?)",
                (source_arxiv_id, status, now),
            )
    except sqlite3.Error:
        logger.warning(
            "Failed to save S2 recommendations for %s",
            source_arxiv_id,
            exc_info=True,
        )


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
    db_path: Path,
    paper_id: str,
    ttl_days: int = S2_CITATION_GRAPH_CACHE_TTL_DAYS,
) -> bool:
    """Check whether citation graph data was fetched recently for this paper."""
    if not db_path.exists():
        return False
    try:
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
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
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
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
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
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
