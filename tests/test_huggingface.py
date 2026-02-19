"""Tests for the HuggingFace Daily Papers API client, parsing, and cache layer."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from arxiv_browser.huggingface import (
    HF_DEFAULT_CACHE_TTL_HOURS,
    HuggingFacePaper,
    _hf_paper_to_json,
    _is_fresh,
    _json_to_hf_paper,
    fetch_hf_daily_papers,
    get_hf_db_path,
    init_hf_db,
    load_hf_daily_cache,
    parse_hf_paper_response,
    save_hf_daily_cache,
)

# ============================================================================
# Test Helpers
# ============================================================================


def _make_paper(**kwargs) -> HuggingFacePaper:
    """Create a HuggingFacePaper with sensible defaults for testing."""
    defaults = {
        "arxiv_id": "2602.08629",
        "title": "Test Paper",
        "upvotes": 42,
        "num_comments": 5,
        "ai_summary": "This paper does X.",
        "ai_keywords": ("machine learning", "transformers"),
        "github_repo": "https://github.com/test/repo",
        "github_stars": 100,
    }
    defaults.update(kwargs)
    return HuggingFacePaper(**defaults)


def _make_api_item(**overrides) -> dict:
    """Create a valid HF daily papers API response item."""
    item = {
        "paper": {
            "id": "2602.08629",
            "title": "Test Paper",
            "upvotes": 42,
            "ai_summary": "This paper does X.",
            "ai_keywords": ["machine learning", "transformers"],
            "githubRepo": "https://github.com/test/repo",
            "githubStars": 100,
        },
        "numComments": 5,
    }
    for key, value in overrides.items():
        if key.startswith("paper."):
            item["paper"][key[6:]] = value
        else:
            item[key] = value
    return item


# ============================================================================
# Response Parsing Tests
# ============================================================================


class TestParseHfPaperResponse:
    """Tests for parse_hf_paper_response()."""

    def test_valid_response(self) -> None:
        item = _make_api_item()
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.arxiv_id == "2602.08629"
        assert result.title == "Test Paper"
        assert result.upvotes == 42
        assert result.num_comments == 5
        assert result.ai_summary == "This paper does X."
        assert result.ai_keywords == ("machine learning", "transformers")
        assert result.github_repo == "https://github.com/test/repo"
        assert result.github_stars == 100

    def test_missing_paper_id_returns_none(self) -> None:
        item = _make_api_item()
        item["paper"]["id"] = ""
        assert parse_hf_paper_response(item) is None

    def test_missing_paper_key_returns_none(self) -> None:
        assert parse_hf_paper_response({}) is None
        assert parse_hf_paper_response({"paper": None}) is None

    def test_non_dict_item_returns_none(self) -> None:
        assert parse_hf_paper_response("not-a-dict") is None  # type: ignore[arg-type]

    def test_non_dict_paper_returns_none(self) -> None:
        assert parse_hf_paper_response({"paper": "bad"}) is None

    def test_null_ai_summary(self) -> None:
        item = _make_api_item()
        item["paper"]["ai_summary"] = None
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.ai_summary == ""

    def test_non_string_ai_summary_ignored(self) -> None:
        item = _make_api_item()
        item["paper"]["ai_summary"] = 12345
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.ai_summary == ""

    def test_empty_ai_keywords(self) -> None:
        item = _make_api_item()
        item["paper"]["ai_keywords"] = []
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.ai_keywords == ()

    def test_null_ai_keywords(self) -> None:
        item = _make_api_item()
        item["paper"]["ai_keywords"] = None
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.ai_keywords == ()

    def test_non_string_keywords_filtered(self) -> None:
        item = _make_api_item()
        item["paper"]["ai_keywords"] = ["valid", 123, None, "also valid"]
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.ai_keywords == ("valid", "also valid")

    def test_optional_github_fields_absent(self) -> None:
        item = _make_api_item()
        del item["paper"]["githubRepo"]
        del item["paper"]["githubStars"]
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.github_repo == ""
        assert result.github_stars == 0

    def test_non_int_upvotes_default_zero(self) -> None:
        item = _make_api_item()
        item["paper"]["upvotes"] = "not a number"
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.upvotes == 0

    def test_non_int_num_comments_default_zero(self) -> None:
        item = _make_api_item()
        item["numComments"] = "not a number"
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.num_comments == 0

    def test_non_string_github_repo_ignored(self) -> None:
        item = _make_api_item()
        item["paper"]["githubRepo"] = 42
        result = parse_hf_paper_response(item)
        assert result is not None
        assert result.github_repo == ""


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Tests for _hf_paper_to_json / _json_to_hf_paper round-trip."""

    def test_round_trip(self) -> None:
        paper = _make_paper()
        payload = _hf_paper_to_json(paper)
        restored = _json_to_hf_paper(payload)
        assert restored is not None
        assert restored.arxiv_id == paper.arxiv_id
        assert restored.title == paper.title
        assert restored.upvotes == paper.upvotes
        assert restored.num_comments == paper.num_comments
        assert restored.ai_summary == paper.ai_summary
        assert restored.ai_keywords == paper.ai_keywords
        assert restored.github_repo == paper.github_repo
        assert restored.github_stars == paper.github_stars

    def test_invalid_json(self) -> None:
        assert _json_to_hf_paper("not json") is None

    def test_missing_required_key(self) -> None:
        # arxiv_id is required (KeyError path)
        payload = json.dumps({"title": "Test"})
        assert _json_to_hf_paper(payload) is None

    def test_empty_optional_fields(self) -> None:
        paper = _make_paper(
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
        payload = _hf_paper_to_json(paper)
        restored = _json_to_hf_paper(payload)
        assert restored is not None
        assert restored.ai_summary == ""
        assert restored.ai_keywords == ()
        assert restored.github_repo == ""
        assert restored.github_stars == 0

    def test_coerces_bad_types(self) -> None:
        payload = json.dumps(
            {
                "arxiv_id": "2602.08629",
                "title": 123,
                "upvotes": "42",
                "num_comments": None,
                "ai_summary": ["bad"],
                "ai_keywords": ["ok", 123, None],
                "github_repo": 9,
                "github_stars": "77",
            }
        )
        restored = _json_to_hf_paper(payload)
        assert restored is not None
        assert restored.title == ""
        assert restored.upvotes == 0
        assert restored.num_comments == 0
        assert restored.ai_summary == ""
        assert restored.ai_keywords == ("ok",)
        assert restored.github_repo == ""
        assert restored.github_stars == 0


# ============================================================================
# TTL Freshness Tests
# ============================================================================


class TestIsFresh:
    """Tests for _is_fresh() hours-based TTL check."""

    def test_fresh_entry(self) -> None:
        now = datetime.now(UTC)
        assert _is_fresh(now.isoformat(), HF_DEFAULT_CACHE_TTL_HOURS) is True

    def test_stale_entry(self) -> None:
        old = datetime.now(UTC) - timedelta(hours=HF_DEFAULT_CACHE_TTL_HOURS + 1)
        assert _is_fresh(old.isoformat(), HF_DEFAULT_CACHE_TTL_HOURS) is False

    def test_exactly_at_boundary(self) -> None:
        """Entry exactly at TTL boundary should be considered stale."""
        old = datetime.now(UTC) - timedelta(hours=HF_DEFAULT_CACHE_TTL_HOURS, seconds=1)
        assert _is_fresh(old.isoformat(), HF_DEFAULT_CACHE_TTL_HOURS) is False

    def test_invalid_date_returns_false(self) -> None:
        assert _is_fresh("not-a-date", HF_DEFAULT_CACHE_TTL_HOURS) is False

    def test_none_returns_false(self) -> None:
        assert _is_fresh(None, HF_DEFAULT_CACHE_TTL_HOURS) is False  # type: ignore[arg-type]

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetime string should be treated as UTC."""
        now = datetime.now(UTC)
        # Remove timezone info to simulate naive datetime
        naive_str = now.strftime("%Y-%m-%dT%H:%M:%S")
        assert _is_fresh(naive_str, HF_DEFAULT_CACHE_TTL_HOURS) is True


# ============================================================================
# SQLite Cache Tests
# ============================================================================


class TestHfCache:
    """Tests for SQLite cache operations."""

    def test_init_creates_table(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_hf_db(db_path)
        assert db_path.exists()

        import sqlite3

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]
            assert "hf_daily_papers" in table_names

    def test_save_load_round_trip(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        papers = [
            _make_paper(arxiv_id="2602.08629"),
            _make_paper(arxiv_id="2602.12345", upvotes=100),
        ]
        save_hf_daily_cache(db_path, papers)
        result = load_hf_daily_cache(db_path)
        assert result is not None
        assert len(result) == 2
        assert "2602.08629" in result
        assert "2602.12345" in result
        assert result["2602.12345"].upvotes == 100

    def test_stale_data_rejected(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        papers = [_make_paper()]
        save_hf_daily_cache(db_path, papers)

        # Manually set fetched_at to an old time
        import sqlite3

        old_time = (datetime.now(UTC) - timedelta(hours=HF_DEFAULT_CACHE_TTL_HOURS + 1)).isoformat()
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("UPDATE hf_daily_papers SET fetched_at = ?", (old_time,))

        result = load_hf_daily_cache(db_path)
        assert result is None

    def test_save_replaces_all(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        # Save first batch
        papers1 = [_make_paper(arxiv_id="old1"), _make_paper(arxiv_id="old2")]
        save_hf_daily_cache(db_path, papers1)

        # Save second batch (should replace)
        papers2 = [_make_paper(arxiv_id="new1")]
        save_hf_daily_cache(db_path, papers2)

        result = load_hf_daily_cache(db_path)
        assert result is not None
        assert len(result) == 1
        assert "new1" in result
        assert "old1" not in result

    def test_empty_db_returns_none(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        init_hf_db(db_path)
        result = load_hf_daily_cache(db_path)
        assert result is None

    def test_nonexistent_db_returns_none(self, tmp_path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = load_hf_daily_cache(db_path)
        assert result is None

    def test_get_hf_db_path(self) -> None:
        path = get_hf_db_path()
        assert path.name == "huggingface.db"
        assert "arxiv-browser" in str(path)


# ============================================================================
# API Function Tests
# ============================================================================


class TestFetchHfDailyPapers:
    """Tests for fetch_hf_daily_papers() with mocked httpx."""

    @pytest.mark.asyncio
    async def test_success_200(self) -> None:
        items = [_make_api_item(), _make_api_item(**{"paper.id": "2602.99999"})]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = items

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert len(result) == 2
        assert result[0].arxiv_id == "2602.08629"
        assert result[1].arxiv_id == "2602.99999"

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_list_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "unexpected"}

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert result == []

    @pytest.mark.asyncio
    async def test_404_returns_empty(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert result == []

    @pytest.mark.asyncio
    @patch("arxiv_browser.huggingface.asyncio.sleep", new_callable=AsyncMock)
    async def test_429_retries(self, mock_sleep) -> None:
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = [_make_api_item()]

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [mock_429, mock_200]

        result = await fetch_hf_daily_papers(client)
        assert len(result) == 1
        assert mock_sleep.called

    @pytest.mark.asyncio
    @patch("arxiv_browser.huggingface.asyncio.sleep", new_callable=AsyncMock)
    async def test_timeout_retries(self, mock_sleep) -> None:
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = [_make_api_item()]

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [httpx.TimeoutException("timeout"), mock_200]

        result = await fetch_hf_daily_papers(client)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ConnectError("connection failed")

        result = await fetch_hf_daily_papers(client)
        assert result == []

    @pytest.mark.asyncio
    @patch("arxiv_browser.huggingface.asyncio.sleep", new_callable=AsyncMock)
    async def test_exhausted_retries_returns_empty(self, mock_sleep) -> None:
        mock_500 = MagicMock()
        mock_500.status_code = 500

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_500

        result = await fetch_hf_daily_papers(client)
        assert result == []
        # Should have retried HF_MAX_RETRIES - 1 times (3 attempts total)
        assert client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_skips_unparseable_items(self) -> None:
        """Items that fail to parse are silently skipped."""
        items = [
            _make_api_item(),
            {"paper": {"id": ""}},  # Missing ID
            _make_api_item(**{"paper.id": "2602.99999"}),
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = items

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_skips_non_dict_items(self) -> None:
        items = [
            _make_api_item(),
            None,
            "bad",
            42,
            _make_api_item(**{"paper.id": "2602.99999"}),
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = items

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        result = await fetch_hf_daily_papers(client)
        assert len(result) == 2


class TestInitHfDbOsError:
    """Fix 3: init_hf_db converts mkdir OSError to sqlite3.OperationalError."""

    def test_init_hf_db_permission_error(self, tmp_path):
        """PermissionError during mkdir should raise sqlite3.OperationalError."""
        import sqlite3
        from unittest.mock import patch

        db_path = tmp_path / "sub" / "db.sqlite"
        with (
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
            pytest.raises(sqlite3.OperationalError, match="Cannot create DB directory"),
        ):
            init_hf_db(db_path)
