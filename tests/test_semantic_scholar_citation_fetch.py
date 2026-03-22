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

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_object_json_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "object"]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_list_data_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "not-a-list"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_references("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_include_status_reports_failure(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "not-a-list"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        entries, complete = await fetch_s2_references(
            "s2paper1",
            mock_client,
            include_status=True,
        )
        assert entries == []
        assert complete is False


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
                {"citingPaper": {"paperId": f"c{i}", "citationCount": i * 10}} for i in range(5)
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

    @pytest.mark.asyncio
    async def test_paginates_and_sorts_globally(self) -> None:
        page1 = MagicMock(spec=httpx.Response)
        page1.status_code = 200
        page1.json.return_value = {
            "data": [
                {"citingPaper": {"paperId": f"p1-{i}", "citationCount": i}}
                for i in range(S2_CITATIONS_PAGE_SIZE)
            ]
        }
        page2 = MagicMock(spec=httpx.Response)
        page2.status_code = 200
        page2.json.return_value = {
            "data": [
                {"citingPaper": {"paperId": "p2-top", "citationCount": 10_000}},
                *[
                    {"citingPaper": {"paperId": f"p2-{i}", "citationCount": i + 1}}
                    for i in range(S2_CITATIONS_PAGE_SIZE - 1)
                ],
            ]
        }
        page3 = MagicMock(spec=httpx.Response)
        page3.status_code = 200
        page3.json.return_value = {"data": []}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = [page1, page2, page3]

        result = await fetch_s2_citations("s2paper1", mock_client, limit=3)
        assert len(result) == 3
        assert result[0].citation_count == 10_000
        assert mock_client.get.call_count == 2
        assert mock_client.get.call_args_list[0].kwargs["params"]["offset"] == "0"
        assert mock_client.get.call_args_list[1].kwargs["params"]["offset"] == str(
            S2_CITATIONS_PAGE_SIZE
        )

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_object_json_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "object"]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_list_data_returns_empty(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "not-a-list"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await fetch_s2_citations("s2paper1", mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_positive_limit_returns_empty_without_calls(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        result = await fetch_s2_citations("s2paper1", mock_client, limit=0)
        assert result == []
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_include_status_reports_incomplete_fetch(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        entries, complete = await fetch_s2_citations(
            "s2paper1",
            mock_client,
            include_status=True,
        )
        assert entries == []
        assert complete is False
