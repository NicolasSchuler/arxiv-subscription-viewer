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


class TestS2Cache:
    """Tests for SQLite cache operations."""

    def test_init_creates_tables(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_s2_db(db_path)
        assert db_path.exists()
        import sqlite3

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
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
        mock_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429",
            request=httpx.Request("GET", "https://api.semanticscholar.org"),
            response=mock_429,
        )
        mock_200 = MagicMock(spec=httpx.Response)
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "paperId": "abc123",
            "citationCount": 10,
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = [mock_429, mock_200]

        with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
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

        with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_paper("2401.12345", mock_client)

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        mock_429 = MagicMock(spec=httpx.Response)
        mock_429.status_code = 429
        mock_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429",
            request=httpx.Request("GET", "https://api.semanticscholar.org"),
            response=mock_429,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_429

        with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_paper("2401.12345", mock_client)

        assert result is None
        assert mock_client.get.call_count == 3  # S2_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_paper("2401.12345", mock_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_include_status_reports_not_found_as_complete(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404",
            request=httpx.Request("GET", "https://api.semanticscholar.org"),
            response=mock_response,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result, complete = await fetch_s2_paper(
            "2401.12345",
            mock_client,
            include_status=True,
        )
        assert result is None
        assert complete is True

    @pytest.mark.asyncio
    async def test_include_status_reports_connection_failures(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result, complete = await fetch_s2_paper(
                "2401.12345",
                mock_client,
                include_status=True,
            )
        assert result is None
        assert complete is False

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_paper("2401.12345", mock_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_object_json_returns_none(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "an", "object"]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

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
        mock_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429",
            request=httpx.Request("GET", "https://api.semanticscholar.org"),
            response=mock_429,
        )
        mock_200 = MagicMock(spec=httpx.Response)
        mock_200.status_code = 200
        mock_200.json.return_value = {"recommendedPapers": []}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = [mock_429, mock_200]

        with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_s2_recommendations("2401.12345", mock_client)

        assert result == []
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_object_json_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "object"]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_list_recommended_papers_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"recommendedPapers": "invalid"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_recommendations("2401.12345", mock_client)
        assert result == []


class TestGetS2DbPath:
    def test_returns_path(self) -> None:
        path = get_s2_db_path()
        assert path.name == "semantic_scholar.db"
        assert "arxiv-browser" in str(path)


class TestInitS2DbOsError:
    """Fix 3: init_s2_db converts mkdir OSError to sqlite3.OperationalError."""

    def test_init_s2_db_permission_error(self, tmp_path):
        """PermissionError during mkdir should raise sqlite3.OperationalError."""
        import sqlite3 as _sqlite3
        from unittest.mock import patch

        db_path = tmp_path / "sub" / "db.sqlite"
        with (
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
            pytest.raises(_sqlite3.OperationalError, match="Cannot create DB directory"),
        ):
            init_s2_db(db_path)
