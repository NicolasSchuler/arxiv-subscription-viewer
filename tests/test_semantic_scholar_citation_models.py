"""Tests for the Semantic Scholar API client, parsing, and cache layer."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from arxiv_browser.semantic_scholar import (
    S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    S2_CITATIONS_PAGE_SIZE,
    S2_DEFAULT_CACHE_TTL_DAYS,
    S2_REC_CACHE_TTL_DAYS,
    CitationEntry,
    SemanticScholarPaper,
    _citation_entry_to_json,
    _is_fresh,
    _json_to_citation_entry,
    _json_to_paper,
    _paper_to_json,
    fetch_s2_citations,
    fetch_s2_paper,
    fetch_s2_recommendations,
    fetch_s2_references,
    get_s2_db_path,
    has_s2_citation_graph_cache,
    init_s2_db,
    load_s2_citation_graph,
    load_s2_paper,
    load_s2_recommendations,
    parse_citation_entry,
    parse_s2_paper_response,
    save_s2_citation_graph,
    save_s2_paper,
    save_s2_recommendations,
)

# ============================================================================
# Test Helpers
# ============================================================================
from tests.support.semantic_scholar_helpers import _make_citation_entry, _make_paper


class TestParseCitationEntry:
    """Tests for parse_citation_entry()."""

    def test_valid_entry(self) -> None:
        data = {
            "paperId": "s2id001",
            "externalIds": {"ArXiv": "2401.12345"},
            "title": "A Great Paper",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2024,
            "citationCount": 42,
            "url": "https://www.semanticscholar.org/paper/s2id001",
        }
        result = parse_citation_entry(data)
        assert result is not None
        assert result.s2_paper_id == "s2id001"
        assert result.arxiv_id == "2401.12345"
        assert result.title == "A Great Paper"
        assert result.authors == "Alice, Bob"
        assert result.year == 2024
        assert result.citation_count == 42
        # arXiv URL preferred over S2 URL
        assert result.url == "https://arxiv.org/abs/2401.12345"

    def test_missing_paper_id_returns_none(self) -> None:
        data = {"title": "No ID Paper"}
        assert parse_citation_entry(data) is None

    def test_no_arxiv_id_uses_s2_url(self) -> None:
        data = {
            "paperId": "s2id002",
            "externalIds": {},
            "title": "Non-arXiv Paper",
            "authors": [],
            "year": 2023,
            "citationCount": 10,
            "url": "https://www.semanticscholar.org/paper/s2id002",
        }
        result = parse_citation_entry(data)
        assert result is not None
        assert result.arxiv_id == ""
        assert result.url == "https://www.semanticscholar.org/paper/s2id002"

    def test_malformed_external_ids_and_url_are_ignored(self) -> None:
        data = {
            "paperId": "s2id-malformed",
            "externalIds": "not-a-mapping",
            "url": 42,
        }
        result = parse_citation_entry(data)
        assert result is not None
        assert result.arxiv_id == ""
        assert result.url == ""

    def test_authors_joining(self) -> None:
        data = {
            "paperId": "s2id003",
            "authors": [{"name": "Alice"}, {"name": ""}, {"name": "Charlie"}],
        }
        result = parse_citation_entry(data)
        assert result is not None
        # Empty name is filtered out
        assert result.authors == "Alice, Charlie"

    def test_empty_authors(self) -> None:
        data = {"paperId": "s2id004", "authors": None}
        result = parse_citation_entry(data)
        assert result is not None
        assert result.authors == ""

    def test_malformed_authors_items_are_ignored(self) -> None:
        data = {
            "paperId": "s2id007",
            "authors": [None, "Alice", {"name": "Bob"}, {"name": ""}],
        }
        result = parse_citation_entry(data)
        assert result is not None
        assert result.authors == "Bob"

    def test_null_fields_default(self) -> None:
        data = {"paperId": "s2id005"}
        result = parse_citation_entry(data)
        assert result is not None
        assert result.title == "Unknown Title"
        assert result.year is None
        assert result.citation_count == 0
        assert result.arxiv_id == ""

    def test_url_preference_arxiv_over_s2(self) -> None:
        """When both arxiv_id and S2 url are present, arxiv URL is preferred."""
        data = {
            "paperId": "s2id006",
            "externalIds": {"ArXiv": "2305.99999"},
            "url": "https://www.semanticscholar.org/paper/s2id006",
        }
        result = parse_citation_entry(data)
        assert result is not None
        assert result.url == "https://arxiv.org/abs/2305.99999"


class TestCitationEntrySerialization:
    """Tests for _citation_entry_to_json / _json_to_citation_entry round-trip."""

    def test_round_trip(self) -> None:
        entry = _make_citation_entry()
        payload = _citation_entry_to_json(entry)
        restored = _json_to_citation_entry(payload)
        assert restored is not None
        assert restored.s2_paper_id == entry.s2_paper_id
        assert restored.arxiv_id == entry.arxiv_id
        assert restored.title == entry.title
        assert restored.authors == entry.authors
        assert restored.year == entry.year
        assert restored.citation_count == entry.citation_count
        assert restored.url == entry.url

    def test_invalid_json_returns_none(self) -> None:
        assert _json_to_citation_entry("not json") is None

    def test_missing_key_returns_none(self) -> None:
        assert _json_to_citation_entry('{"title": "no id"}') is None


class TestCitationGraphCache:
    """Tests for citation graph SQLite cache operations."""

    def test_init_creates_table(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        import sqlite3

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = {t[0] for t in tables}
            assert "s2_citation_graph" in table_names

    def test_save_and_load_references(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        refs = [
            _make_citation_entry(s2_paper_id="ref1", citation_count=100),
            _make_citation_entry(s2_paper_id="ref2", citation_count=50),
        ]
        save_s2_citation_graph(db_path, "paper1", "references", refs)
        loaded = load_s2_citation_graph(db_path, "paper1", "references")
        assert len(loaded) == 2
        assert loaded[0].s2_paper_id == "ref1"
        assert loaded[1].s2_paper_id == "ref2"

    def test_save_and_load_citations(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        cites = [
            _make_citation_entry(s2_paper_id="cite1", citation_count=200),
        ]
        save_s2_citation_graph(db_path, "paper1", "citations", cites)
        loaded = load_s2_citation_graph(db_path, "paper1", "citations")
        assert len(loaded) == 1
        assert loaded[0].s2_paper_id == "cite1"

    def test_stale_ttl_returns_empty(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        refs = [_make_citation_entry(s2_paper_id="ref1")]
        save_s2_citation_graph(db_path, "paper1", "references", refs)
        # TTL=0 → always stale
        assert load_s2_citation_graph(db_path, "paper1", "references", ttl_days=0) == []

    def test_load_missing_returns_empty(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        assert load_s2_citation_graph(db_path, "nonexistent", "references") == []

    def test_replace_old_entries(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        old_refs = [_make_citation_entry(s2_paper_id="old1")]
        save_s2_citation_graph(db_path, "paper1", "references", old_refs)
        new_refs = [
            _make_citation_entry(s2_paper_id="new1"),
            _make_citation_entry(s2_paper_id="new2"),
        ]
        save_s2_citation_graph(db_path, "paper1", "references", new_refs)
        loaded = load_s2_citation_graph(db_path, "paper1", "references")
        assert len(loaded) == 2
        assert loaded[0].s2_paper_id == "new1"

    def test_directions_independent(self, tmp_path) -> None:
        """References and citations for the same paper_id are independent."""
        db_path = tmp_path / "test.db"
        refs = [_make_citation_entry(s2_paper_id="ref1")]
        cites = [
            _make_citation_entry(s2_paper_id="cite1"),
            _make_citation_entry(s2_paper_id="cite2"),
        ]
        save_s2_citation_graph(db_path, "paper1", "references", refs)
        save_s2_citation_graph(db_path, "paper1", "citations", cites)
        assert len(load_s2_citation_graph(db_path, "paper1", "references")) == 1
        assert len(load_s2_citation_graph(db_path, "paper1", "citations")) == 2

    def test_load_when_db_missing(self, tmp_path) -> None:
        db_path = tmp_path / "nonexistent.db"
        assert load_s2_citation_graph(db_path, "paper1", "references") == []

    def test_cache_marker_present_for_empty_graph(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        save_s2_citation_graph(db_path, "paper-empty", "references", [])
        save_s2_citation_graph(db_path, "paper-empty", "citations", [])
        assert has_s2_citation_graph_cache(db_path, "paper-empty") is True

    def test_cache_marker_stale_returns_false(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        save_s2_citation_graph(db_path, "paper-stale", "references", [])
        save_s2_citation_graph(db_path, "paper-stale", "citations", [])
        assert has_s2_citation_graph_cache(db_path, "paper-stale", ttl_days=0) is False
