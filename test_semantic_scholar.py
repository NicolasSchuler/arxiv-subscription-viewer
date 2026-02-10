"""Tests for the Semantic Scholar API client, parsing, and cache layer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from semantic_scholar import (
    CitationEntry,
    S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    S2_DEFAULT_CACHE_TTL_DAYS,
    S2_REC_CACHE_TTL_DAYS,
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


def _make_paper(**kwargs) -> SemanticScholarPaper:
    """Create a SemanticScholarPaper with sensible defaults for testing."""
    defaults = {
        "arxiv_id": "2401.12345",
        "s2_paper_id": "abc123",
        "citation_count": 42,
        "influential_citation_count": 5,
        "tldr": "Does X.",
        "fields_of_study": ("CS",),
        "year": 2024,
        "url": "https://example.com",
        "title": "Test Paper",
        "abstract": "Abstract text.",
    }
    defaults.update(kwargs)
    return SemanticScholarPaper(**defaults)


# ============================================================================
# Response Parsing Tests
# ============================================================================


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


# ============================================================================
# Serialization Tests
# ============================================================================


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


# ============================================================================
# Cache Freshness Tests
# ============================================================================


class TestIsFresh:
    """Tests for _is_fresh()."""

    def test_fresh_entry(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        assert _is_fresh(now, ttl_days=7) is True

    def test_stale_entry(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        assert _is_fresh(old, ttl_days=7) is False

    def test_invalid_date(self) -> None:
        assert _is_fresh("not-a-date", ttl_days=7) is False

    def test_boundary_fresh(self) -> None:
        """Entry just under TTL should be fresh."""
        almost_expired = (
            datetime.now(timezone.utc) - timedelta(days=6, hours=23)
        ).isoformat()
        assert _is_fresh(almost_expired, ttl_days=7) is True

    def test_boundary_stale(self) -> None:
        """Entry just over TTL should be stale."""
        just_expired = (
            datetime.now(timezone.utc) - timedelta(days=7, hours=1)
        ).isoformat()
        assert _is_fresh(just_expired, ttl_days=7) is False


# ============================================================================
# SQLite Cache CRUD Tests
# ============================================================================


class TestS2Cache:
    """Tests for SQLite cache operations."""

    def test_init_creates_tables(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        assert db_path.exists()
        import sqlite3

        with sqlite3.connect(str(db_path)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
            assert "s2_papers" in table_names
            assert "s2_recommendations" in table_names

    def test_save_and_load_paper(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        paper = _make_paper()
        save_s2_paper(db_path, paper)
        loaded = load_s2_paper(db_path, "2401.12345")
        assert loaded is not None
        assert loaded.arxiv_id == "2401.12345"
        assert loaded.citation_count == 42
        assert loaded.tldr == "Does X."

    def test_load_returns_none_for_missing(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        assert load_s2_paper(db_path, "nonexistent") is None

    def test_load_returns_none_when_db_missing(self, tmp_path) -> None:
        db_path = tmp_path / "nonexistent.db"
        assert load_s2_paper(db_path, "2401.12345") is None

    def test_load_returns_none_when_stale(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        paper = _make_paper()
        save_s2_paper(db_path, paper)
        # Load with TTL=0 so it's always stale
        assert load_s2_paper(db_path, "2401.12345", ttl_days=0) is None

    def test_save_overwrites_existing(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        paper1 = _make_paper(citation_count=10)
        save_s2_paper(db_path, paper1)
        paper2 = _make_paper(citation_count=99)
        save_s2_paper(db_path, paper2)
        loaded = load_s2_paper(db_path, "2401.12345")
        assert loaded is not None
        assert loaded.citation_count == 99

    def test_save_and_load_recommendations(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        recs = [
            _make_paper(arxiv_id="2401.00001", citation_count=100),
            _make_paper(arxiv_id="2401.00002", citation_count=50),
        ]
        save_s2_recommendations(db_path, "2401.12345", recs)
        loaded = load_s2_recommendations(db_path, "2401.12345")
        assert len(loaded) == 2
        assert loaded[0].arxiv_id == "2401.00001"
        assert loaded[1].arxiv_id == "2401.00002"

    def test_load_recommendations_empty_when_missing(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        assert load_s2_recommendations(db_path, "nonexistent") == []

    def test_load_recommendations_empty_when_stale(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        recs = [_make_paper(arxiv_id="2401.00001")]
        save_s2_recommendations(db_path, "2401.12345", recs)
        assert load_s2_recommendations(db_path, "2401.12345", ttl_days=0) == []

    def test_save_recommendations_replaces_old(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        old_recs = [_make_paper(arxiv_id="2401.00001")]
        save_s2_recommendations(db_path, "2401.12345", old_recs)
        new_recs = [
            _make_paper(arxiv_id="2401.00099"),
            _make_paper(arxiv_id="2401.00098"),
        ]
        save_s2_recommendations(db_path, "2401.12345", new_recs)
        loaded = load_s2_recommendations(db_path, "2401.12345")
        assert len(loaded) == 2
        assert loaded[0].arxiv_id == "2401.00099"

    def test_load_recommendations_when_db_missing(self, tmp_path) -> None:
        db_path = tmp_path / "nonexistent.db"
        assert load_s2_recommendations(db_path, "2401.12345") == []


# ============================================================================
# API Function Tests (mocked httpx)
# ============================================================================


class TestFetchS2Paper:
    """Tests for fetch_s2_paper() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        response_data = {
            "paperId": "abc123",
            "citationCount": 42,
            "influentialCitationCount": 5,
            "tldr": {"text": "Does X."},
            "fieldsOfStudy": ["Computer Science"],
            "year": 2024,
            "url": "https://example.com",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_paper("2401.12345", mock_client)
        assert result is not None
        assert result.citation_count == 42
        assert result.s2_paper_id == "abc123"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_404_returns_none(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_paper("2401.12345", mock_client)
        assert result is None
        # 404 should not retry
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_429_retries(self) -> None:
        mock_429 = MagicMock(spec=httpx.Response)
        mock_429.status_code = 429
        mock_200 = MagicMock(spec=httpx.Response)
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "paperId": "abc123",
            "citationCount": 10,
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = [mock_429, mock_200]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_paper("2401.12345", mock_client)

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_retries(self) -> None:
        mock_200 = MagicMock(spec=httpx.Response)
        mock_200.status_code = 200
        mock_200.json.return_value = {"paperId": "abc123", "citationCount": 0}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = [
            httpx.TimeoutException("timeout"),
            mock_200,
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_paper("2401.12345", mock_client)

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        mock_429 = MagicMock(spec=httpx.Response)
        mock_429.status_code = 429

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_429

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_paper("2401.12345", mock_client)

        assert result is None
        assert mock_client.get.call_count == 3  # S2_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        result = await fetch_s2_paper("2401.12345", mock_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_api_key_header(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"paperId": "abc123"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        await fetch_s2_paper("2401.12345", mock_client, api_key="test-key")
        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["headers"]["x-api-key"] == "test-key"

    @pytest.mark.asyncio
    async def test_no_api_key_no_header(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"paperId": "abc123"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        await fetch_s2_paper("2401.12345", mock_client)
        call_kwargs = mock_client.get.call_args
        assert "x-api-key" not in call_kwargs.kwargs["headers"]


class TestFetchS2Recommendations:
    """Tests for fetch_s2_recommendations() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        response_data = {
            "recommendedPapers": [
                {
                    "paperId": "rec1",
                    "externalIds": {"ArXiv": "2401.00001"},
                    "title": "Rec Paper 1",
                    "citationCount": 100,
                    "influentialCitationCount": 10,
                    "year": 2024,
                    "url": "https://example.com/1",
                    "abstract": "Abstract 1.",
                },
                {
                    "paperId": "rec2",
                    "externalIds": {"ArXiv": "2401.00002"},
                    "title": "Rec Paper 2",
                    "citationCount": 50,
                    "influentialCitationCount": 5,
                    "year": 2024,
                    "url": "https://example.com/2",
                    "abstract": "Abstract 2.",
                },
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert len(result) == 2
        assert result[0].arxiv_id == "2401.00001"
        assert result[1].citation_count == 50

    @pytest.mark.asyncio
    async def test_empty_recommendations(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"recommendedPapers": []}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_entries(self) -> None:
        """Invalid entries (missing paperId) should be filtered out."""
        response_data = {
            "recommendedPapers": [
                {
                    "paperId": "rec1",
                    "externalIds": {"ArXiv": "2401.00001"},
                    "title": "Valid",
                    "citationCount": 10,
                },
                {
                    # Missing paperId — should be skipped
                    "title": "Invalid",
                    "citationCount": 5,
                },
                {
                    "paperId": "rec3",
                    "externalIds": {"ArXiv": "2401.00003"},
                    "title": "Also Valid",
                    "citationCount": 20,
                },
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert len(result) == 2
        assert result[0].arxiv_id == "2401.00001"
        assert result[1].arxiv_id == "2401.00003"

    @pytest.mark.asyncio
    async def test_404_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_429_retries(self) -> None:
        mock_429 = MagicMock(spec=httpx.Response)
        mock_429.status_code = 429
        mock_200 = MagicMock(spec=httpx.Response)
        mock_200.status_code = 200
        mock_200.json.return_value = {"recommendedPapers": []}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = [mock_429, mock_200]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_recommendations("2401.12345", mock_client)

        assert result == []
        assert mock_client.get.call_count == 2


# ============================================================================
# get_s2_db_path Test
# ============================================================================


class TestGetS2DbPath:
    def test_returns_path(self) -> None:
        path = get_s2_db_path()
        assert path.name == "semantic_scholar.db"
        assert "arxiv-browser" in str(path)


# ============================================================================
# Citation Graph: Test Helpers
# ============================================================================


def _make_citation_entry(**kwargs) -> CitationEntry:
    """Create a CitationEntry with sensible defaults for testing."""
    defaults = {
        "s2_paper_id": "s2id001",
        "arxiv_id": "2401.12345",
        "title": "Test Paper",
        "authors": "Alice, Bob",
        "year": 2024,
        "citation_count": 42,
        "url": "https://arxiv.org/abs/2401.12345",
    }
    defaults.update(kwargs)
    return CitationEntry(**defaults)


# ============================================================================
# Citation Graph: Response Parsing Tests
# ============================================================================


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


# ============================================================================
# Citation Graph: Serialization Tests
# ============================================================================


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


# ============================================================================
# Citation Graph: SQLite Cache Tests
# ============================================================================


class TestCitationGraphCache:
    """Tests for citation graph SQLite cache operations."""

    def test_init_creates_table(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        import sqlite3

        with sqlite3.connect(str(db_path)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
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


# ============================================================================
# Citation Graph: API Fetch Tests
# ============================================================================


class TestFetchS2References:
    """Tests for fetch_s2_references() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        response_data = {
            "data": [
                {
                    "citedPaper": {
                        "paperId": "ref1",
                        "externalIds": {"ArXiv": "2401.00001"},
                        "title": "Ref Paper 1",
                        "authors": [{"name": "Alice"}],
                        "year": 2023,
                        "citationCount": 100,
                        "url": "https://example.com/1",
                    }
                },
                {
                    "citedPaper": {
                        "paperId": "ref2",
                        "externalIds": {"ArXiv": "2401.00002"},
                        "title": "Ref Paper 2",
                        "authors": [{"name": "Bob"}],
                        "year": 2022,
                        "citationCount": 50,
                        "url": "https://example.com/2",
                    }
                },
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert len(result) == 2
        # Sorted by citation_count desc
        assert result[0].citation_count == 100
        assert result[1].citation_count == 50

    @pytest.mark.asyncio
    async def test_empty_references(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_sorted_by_citations(self) -> None:
        """Results should be sorted by citation_count descending."""
        response_data = {
            "data": [
                {"citedPaper": {"paperId": "a", "citationCount": 10}},
                {"citedPaper": {"paperId": "b", "citationCount": 999}},
                {"citedPaper": {"paperId": "c", "citationCount": 50}},
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert [e.citation_count for e in result] == [999, 50, 10]

    @pytest.mark.asyncio
    async def test_invalid_entries_filtered(self) -> None:
        """Entries with missing paperId should be filtered out."""
        response_data = {
            "data": [
                {"citedPaper": {"paperId": "valid1", "citationCount": 10}},
                {"citedPaper": {"title": "No ID"}},  # Missing paperId
                {"citedPaper": {"paperId": "valid2", "citationCount": 20}},
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert len(result) == 2


class TestFetchS2Citations:
    """Tests for fetch_s2_citations() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        response_data = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "cite1",
                        "externalIds": {"ArXiv": "2401.00010"},
                        "title": "Citing Paper 1",
                        "authors": [{"name": "Charlie"}],
                        "year": 2024,
                        "citationCount": 200,
                        "url": "https://example.com/10",
                    }
                },
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client)
        assert len(result) == 1
        assert result[0].s2_paper_id == "cite1"
        assert result[0].arxiv_id == "2401.00010"

    @pytest.mark.asyncio
    async def test_limited_to_max(self) -> None:
        """Results should be trimmed to the limit parameter."""
        # Create 5 entries, request limit=2
        response_data = {
            "data": [
                {"citingPaper": {"paperId": f"c{i}", "citationCount": i * 10}}
                for i in range(5)
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client, limit=2)
        assert len(result) == 2
        # Should be the top 2 by citation count (40, 30)
        assert result[0].citation_count == 40
        assert result[1].citation_count == 30

    @pytest.mark.asyncio
    async def test_sorted_by_citations(self) -> None:
        response_data = {
            "data": [
                {"citingPaper": {"paperId": "a", "citationCount": 5}},
                {"citingPaper": {"paperId": "b", "citationCount": 500}},
                {"citingPaper": {"paperId": "c", "citationCount": 50}},
            ]
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client)
        assert [e.citation_count for e in result] == [500, 50, 5]

    @pytest.mark.asyncio
    async def test_404_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client)
        assert result == []
