"""Tests for the unified cache database module."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from arxiv_browser.database import (
    CACHE_DB_FILENAME,
    get_cache_db_path,
    init_cache_db,
    resolve_db_path,
)


class TestGetCacheDbPath:
    """Tests for get_cache_db_path()."""

    def test_returns_path(self):
        result = get_cache_db_path()
        assert isinstance(result, Path)
        assert result.name == CACHE_DB_FILENAME

    def test_filename_is_cache_db(self):
        assert CACHE_DB_FILENAME == "cache.db"


class TestResolveDbPath:
    """Tests for the dual-path resolution logic."""

    def test_unified_db_preferred(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When cache.db exists, it takes precedence over legacy files."""
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        (tmp_path / "cache.db").touch()
        (tmp_path / "summaries.db").touch()
        result = resolve_db_path("summaries.db")
        assert result == tmp_path / "cache.db"

    def test_legacy_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When only legacy file exists, use it."""
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        (tmp_path / "summaries.db").touch()
        result = resolve_db_path("summaries.db")
        assert result == tmp_path / "summaries.db"

    def test_new_install_uses_unified(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When neither file exists (new install), return cache.db path."""
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        result = resolve_db_path("summaries.db")
        assert result == tmp_path / "cache.db"


class TestInitCacheDb:
    """Tests for init_cache_db()."""

    def test_creates_all_tables(self, tmp_path: Path):
        db_path = tmp_path / "test_cache.db"
        init_cache_db(db_path)

        with closing(sqlite3.connect(str(db_path))) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        expected_tables = {
            "summaries",
            "relevance_scores",
            "s2_papers",
            "s2_paper_fetch_state",
            "s2_recommendations",
            "s2_recommendation_fetch_state",
            "s2_citation_graph",
            "s2_citation_graph_fetches",
            "hf_daily_papers",
            "hf_daily_fetch_state",
        }
        assert expected_tables.issubset(tables)

    def test_wal_mode_enabled(self, tmp_path: Path):
        db_path = tmp_path / "test_cache.db"
        init_cache_db(db_path)

        with closing(sqlite3.connect(str(db_path))) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_idempotent(self, tmp_path: Path):
        """Calling init_cache_db twice should not fail."""
        db_path = tmp_path / "test_cache.db"
        init_cache_db(db_path)
        init_cache_db(db_path)

        with closing(sqlite3.connect(str(db_path))) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        assert len(tables) >= 10

    def test_creates_parent_directory(self, tmp_path: Path):
        db_path = tmp_path / "sub" / "dir" / "test_cache.db"
        init_cache_db(db_path)
        assert db_path.exists()

    def test_tables_are_usable(self, tmp_path: Path):
        """Verify we can insert and read from the created tables."""
        db_path = tmp_path / "test_cache.db"
        init_cache_db(db_path)

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT INTO summaries (arxiv_id, command_hash, summary, created_at) "
                "VALUES ('2401.12345', 'abc123', 'Test summary', '2024-01-01')"
            )
            row = conn.execute(
                "SELECT summary FROM summaries WHERE arxiv_id = '2401.12345'"
            ).fetchone()
        assert row[0] == "Test summary"


class TestLegacyModuleIntegration:
    """Verify legacy get_*_db_path() functions use resolve_db_path."""

    def test_llm_summary_path_delegates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        from arxiv_browser.llm import get_summary_db_path

        # New install → should return cache.db
        result = get_summary_db_path()
        assert result.name == CACHE_DB_FILENAME

    def test_llm_relevance_path_delegates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        from arxiv_browser.llm import get_relevance_db_path

        result = get_relevance_db_path()
        assert result.name == CACHE_DB_FILENAME

    def test_s2_path_delegates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        from arxiv_browser.semantic_scholar_cache import get_s2_db_path

        result = get_s2_db_path()
        assert result.name == CACHE_DB_FILENAME

    def test_hf_path_delegates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        from arxiv_browser.huggingface import get_hf_db_path

        result = get_hf_db_path()
        assert result.name == CACHE_DB_FILENAME

    def test_legacy_file_preserved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Existing installs with legacy DB files continue using them."""
        monkeypatch.setattr("arxiv_browser.database.user_config_dir", lambda _: str(tmp_path))
        (tmp_path / "summaries.db").touch()

        from arxiv_browser.llm import get_summary_db_path

        result = get_summary_db_path()
        assert result.name == "summaries.db"
