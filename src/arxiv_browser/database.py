"""Unified cache database — single SQLite file for all cached data.

New installs get a single ``cache.db`` file.  Existing installs that already
have legacy per-module database files (``summaries.db``, ``relevance.db``,
``semantic_scholar.db``, ``huggingface.db``) continue using those files
transparently via :func:`resolve_db_path`.  No data migration is performed.

This module is a dependency-DAG leaf (zero internal imports) so it can be
imported by any other module without introducing cycles.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from platformdirs import user_config_dir

from arxiv_browser.models import CONFIG_APP_NAME

logger = logging.getLogger(__name__)

CACHE_DB_FILENAME = "cache.db"

# All table-creation DDL statements, collected from the individual modules.
_CREATE_TABLE_SQL: tuple[str, ...] = (
    # llm.py — summaries
    "CREATE TABLE IF NOT EXISTS summaries ("
    "  arxiv_id TEXT NOT NULL,"
    "  command_hash TEXT NOT NULL,"
    "  summary TEXT NOT NULL,"
    "  created_at TEXT NOT NULL,"
    "  PRIMARY KEY (arxiv_id, command_hash)"
    ")",
    # llm.py — relevance_scores
    "CREATE TABLE IF NOT EXISTS relevance_scores ("
    "  arxiv_id TEXT NOT NULL,"
    "  interests_hash TEXT NOT NULL,"
    "  score INTEGER NOT NULL,"
    "  reason TEXT NOT NULL,"
    "  created_at TEXT NOT NULL,"
    "  PRIMARY KEY (arxiv_id, interests_hash)"
    ")",
    # semantic_scholar_cache.py — s2_papers
    "CREATE TABLE IF NOT EXISTS s2_papers ("
    "  arxiv_id TEXT PRIMARY KEY,"
    "  payload_json TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL"
    ")",
    # semantic_scholar_cache.py — s2_paper_fetch_state
    "CREATE TABLE IF NOT EXISTS s2_paper_fetch_state ("
    "  arxiv_id TEXT PRIMARY KEY,"
    "  status TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL"
    ")",
    # semantic_scholar_cache.py — s2_recommendations
    "CREATE TABLE IF NOT EXISTS s2_recommendations ("
    "  source_arxiv_id TEXT NOT NULL,"
    "  rank INTEGER NOT NULL,"
    "  payload_json TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL,"
    "  PRIMARY KEY (source_arxiv_id, rank)"
    ")",
    # semantic_scholar_cache.py — s2_recommendation_fetch_state
    "CREATE TABLE IF NOT EXISTS s2_recommendation_fetch_state ("
    "  source_arxiv_id TEXT PRIMARY KEY,"
    "  status TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL"
    ")",
    # semantic_scholar_cache.py — s2_citation_graph
    "CREATE TABLE IF NOT EXISTS s2_citation_graph ("
    "  paper_id TEXT NOT NULL,"
    "  direction TEXT NOT NULL,"
    "  rank INTEGER NOT NULL,"
    "  payload_json TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL,"
    "  PRIMARY KEY (paper_id, direction, rank)"
    ")",
    # semantic_scholar_cache.py — s2_citation_graph_fetches
    "CREATE TABLE IF NOT EXISTS s2_citation_graph_fetches ("
    "  paper_id TEXT PRIMARY KEY,"
    "  fetched_at TEXT NOT NULL"
    ")",
    # huggingface.py — hf_daily_papers
    "CREATE TABLE IF NOT EXISTS hf_daily_papers ("
    "  arxiv_id TEXT PRIMARY KEY,"
    "  payload_json TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL"
    ")",
    # huggingface.py — hf_daily_fetch_state
    "CREATE TABLE IF NOT EXISTS hf_daily_fetch_state ("
    "  cache_key TEXT PRIMARY KEY,"
    "  status TEXT NOT NULL,"
    "  fetched_at TEXT NOT NULL"
    ")",
)


def get_cache_db_path() -> Path:
    """Return the path to the unified cache database."""
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / CACHE_DB_FILENAME


def resolve_db_path(legacy_filename: str) -> Path:
    """Return the database path for a module, with legacy fallback.

    Resolution order:
    1. If ``cache.db`` exists → use it (unified install).
    2. If the legacy file (e.g. ``summaries.db``) exists → keep using it.
    3. Neither exists (new install) → return ``cache.db``.
    """
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    unified = config_dir / CACHE_DB_FILENAME
    if unified.exists():
        return unified
    legacy = config_dir / legacy_filename
    if legacy.exists():
        return legacy
    return unified


def init_cache_db(db_path: Path) -> None:
    """Create all cache tables and enable performance PRAGMAs.

    Safe to call repeatedly — uses ``CREATE TABLE IF NOT EXISTS``.

    Raises:
        sqlite3.OperationalError: If the parent directory cannot be created.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise sqlite3.OperationalError(f"Cannot create DB directory: {e}") from e
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        for ddl in _CREATE_TABLE_SQL:
            conn.execute(ddl)
    logger.debug("Initialized cache DB at %s (%d tables)", db_path, len(_CREATE_TABLE_SQL))


__all__ = [
    "CACHE_DB_FILENAME",
    "get_cache_db_path",
    "init_cache_db",
    "resolve_db_path",
]
