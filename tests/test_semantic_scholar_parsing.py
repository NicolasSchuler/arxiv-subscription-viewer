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


class TestParseS2PaperResponse:
    """Tests for parse_s2_paper_response()."""

    def test_valid_response(self) -> None:
        data = {
            "paperId": "abc123",
            "citationCount": 42,
            "influentialCitationCount": 5,
            "tldr": {"text": "This paper does X."},
            "fieldsOfStudy": ["Computer Science", "Mathematics"],
            "year": 2024,
            "url": "https://www.semanticscholar.org/paper/abc123",
            "title": "Test Paper",
            "abstract": "Some abstract text.",
        }
        result = parse_s2_paper_response(data, arxiv_id="2401.12345")
        assert result is not None
        assert result.arxiv_id == "2401.12345"
        assert result.s2_paper_id == "abc123"
        assert result.citation_count == 42
        assert result.influential_citation_count == 5
        assert result.tldr == "This paper does X."
        assert result.fields_of_study == ("Computer Science", "Mathematics")
        assert result.year == 2024
        assert result.url == "https://www.semanticscholar.org/paper/abc123"
        assert result.title == "Test Paper"
        assert result.abstract == "Some abstract text."

    def test_missing_paper_id(self) -> None:
        data = {"citationCount": 10}
        assert parse_s2_paper_response(data, arxiv_id="2401.12345") is None

    def test_null_tldr(self) -> None:
        data = {
            "paperId": "abc123",
            "citationCount": 10,
            "influentialCitationCount": 0,
            "tldr": None,
            "fieldsOfStudy": [],
            "year": 2024,
            "url": "https://example.com",
        }
        result = parse_s2_paper_response(data, arxiv_id="2401.12345")
        assert result is not None
        assert result.tldr == ""

    def test_empty_fields_of_study(self) -> None:
        data = {
            "paperId": "abc123",
            "citationCount": 10,
            "influentialCitationCount": 0,
            "tldr": None,
            "fieldsOfStudy": None,
            "year": None,
            "url": "",
        }
        result = parse_s2_paper_response(data, arxiv_id="2401.12345")
        assert result is not None
        assert result.fields_of_study == ()
        assert result.year is None

    def test_arxiv_id_from_external_ids(self) -> None:
        data = {
            "paperId": "abc123",
            "externalIds": {"ArXiv": "2401.99999"},
            "citationCount": 0,
            "influentialCitationCount": 0,
        }
        result = parse_s2_paper_response(data)
        assert result is not None
        assert result.arxiv_id == "2401.99999"

    def test_malformed_external_ids_are_ignored(self) -> None:
        data = {
            "paperId": "abc123",
            "externalIds": ["not", "a", "mapping"],
            "tldr": {"text": 42},
        }
        result = parse_s2_paper_response(data)
        assert result is not None
        assert result.arxiv_id == ""
        assert result.tldr == ""

    def test_missing_optional_fields_default_to_zero(self) -> None:
        data = {"paperId": "abc123"}
        result = parse_s2_paper_response(data, arxiv_id="2401.12345")
        assert result is not None
        assert result.citation_count == 0
        assert result.influential_citation_count == 0
        assert result.tldr == ""
        assert result.fields_of_study == ()

    def test_non_string_fields_of_study_filtered(self) -> None:
        data = {
            "paperId": "abc123",
            "fieldsOfStudy": ["CS", 42, None, "Math"],
        }
        result = parse_s2_paper_response(data, arxiv_id="2401.12345")
        assert result is not None
        assert result.fields_of_study == ("CS", "Math")


class TestSerialization:
    """Tests for _paper_to_json / _json_to_paper round-trip."""

    def test_round_trip(self) -> None:
        paper = _make_paper()
        payload = _paper_to_json(paper)
        restored = _json_to_paper(payload)
        assert restored is not None
        assert restored.arxiv_id == paper.arxiv_id
        assert restored.s2_paper_id == paper.s2_paper_id
        assert restored.citation_count == paper.citation_count
        assert restored.influential_citation_count == paper.influential_citation_count
        assert restored.tldr == paper.tldr
        assert restored.fields_of_study == paper.fields_of_study
        assert restored.year == paper.year
        assert restored.url == paper.url
        assert restored.title == paper.title
        assert restored.abstract == paper.abstract

    def test_invalid_json_returns_none(self) -> None:
        assert _json_to_paper("not json") is None

    def test_missing_key_returns_none(self) -> None:
        assert _json_to_paper('{"citation_count": 1}') is None


class TestIsFresh:
    """Tests for _is_fresh()."""

    def test_fresh_entry(self) -> None:
        now = datetime.now(UTC).isoformat()
        assert _is_fresh(now, ttl_days=7) is True

    def test_stale_entry(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        assert _is_fresh(old, ttl_days=7) is False

    def test_invalid_date(self) -> None:
        assert _is_fresh("not-a-date", ttl_days=7) is False

    def test_boundary_fresh(self) -> None:
        """Entry just under TTL should be fresh."""
        almost_expired = (datetime.now(UTC) - timedelta(days=6, hours=23)).isoformat()
        assert _is_fresh(almost_expired, ttl_days=7) is True

    def test_boundary_stale(self) -> None:
        """Entry just over TTL should be stale."""
        just_expired = (datetime.now(UTC) - timedelta(days=7, hours=1)).isoformat()
        assert _is_fresh(just_expired, ttl_days=7) is False
