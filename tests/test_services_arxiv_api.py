"""Tests for arXiv API service helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arxiv_browser.models import ArxivSearchRequest
from arxiv_browser.services.arxiv_api_service import (
    ARXIV_API_MIN_INTERVAL_SECONDS,
    enforce_rate_limit,
    fetch_latest_day_digest,
    fetch_page,
    fetch_page_sync,
    format_query_label,
)


def test_format_query_label_valid() -> None:
    request = ArxivSearchRequest(query="transformer", field="title", category="cs.AI")
    assert format_query_label(request) == "ti:transformer AND cat:cs.AI"


def test_format_query_label_fallback_for_invalid_request() -> None:
    request = ArxivSearchRequest(query="fallback", field="invalid", category="")
    assert format_query_label(request) == "fallback"


@pytest.mark.asyncio
async def test_enforce_rate_limit_waits_when_needed() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self._calls = 0

        def now(self) -> float:
            self._calls += 1
            if self._calls == 1:
                return 101.0
            return 104.5

    clock = FakeClock()
    sleep = AsyncMock()

    new_last, waited = await enforce_rate_limit(
        last_request_at=100.0,
        min_interval_seconds=3.0,
        now=clock.now,
        sleep=sleep,
    )

    sleep.assert_awaited_once_with(pytest.approx(2.0))
    assert waited == pytest.approx(2.0)
    assert new_last == pytest.approx(104.5)


@pytest.mark.asyncio
async def test_enforce_rate_limit_skips_wait_when_not_needed() -> None:
    sleep = AsyncMock()

    new_last, waited = await enforce_rate_limit(
        last_request_at=0.0,
        min_interval_seconds=3.0,
        now=lambda: 50.0,
        sleep=sleep,
    )

    sleep.assert_not_awaited()
    assert waited == 0.0
    assert new_last == 50.0


@pytest.mark.asyncio
async def test_fetch_page_uses_shared_client(make_paper) -> None:
    request = ArxivSearchRequest(query="transformers", field="all", category="")
    response = MagicMock()
    response.text = "<feed/>"
    response.raise_for_status = MagicMock()
    client = SimpleNamespace(get=AsyncMock(return_value=response))

    with (
        patch(
            "arxiv_browser.services.arxiv_api_service.build_arxiv_search_query",
            return_value="all:transformers",
        ),
        patch(
            "arxiv_browser.services.arxiv_api_service.parse_arxiv_api_feed",
            return_value=[make_paper()],
        ),
    ):
        papers = await fetch_page(
            client=client,
            request=request,
            start=0,
            max_results=5,
            timeout_seconds=30,
            user_agent="arxiv-subscription-viewer/1.0",
        )

    assert len(papers) == 1
    client.get.assert_awaited_once()
    response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_page_requires_explicit_client(make_paper) -> None:
    request = ArxivSearchRequest(query="transformers", field="all", category="")

    with (
        patch(
            "arxiv_browser.services.arxiv_api_service.build_arxiv_search_query",
            return_value="all:transformers",
        ),
        pytest.raises(AttributeError),
    ):
        await fetch_page(
            client=None,
            request=request,
            start=0,
            max_results=5,
            timeout_seconds=30,
            user_agent="arxiv-subscription-viewer/1.0",
        )


def test_fetch_page_sync_uses_shared_query_building(make_paper) -> None:
    request = ArxivSearchRequest(query="transformers", field="all", category="")
    response = MagicMock()
    response.text = "<feed/>"
    response.raise_for_status = MagicMock()

    with (
        patch(
            "arxiv_browser.services.arxiv_api_service.build_arxiv_search_query",
            return_value="all:transformers",
        ),
        patch(
            "arxiv_browser.services.arxiv_api_service.parse_arxiv_api_feed",
            return_value=[make_paper()],
        ),
        patch("arxiv_browser.services.arxiv_api_service.httpx.get", return_value=response) as get,
    ):
        papers = fetch_page_sync(
            request=request,
            start=0,
            max_results=5,
            timeout_seconds=30,
            user_agent="arxiv-subscription-viewer/1.0",
        )

    assert len(papers) == 1
    get.assert_called_once()
    response.raise_for_status.assert_called_once()


def test_fetch_latest_day_digest_stops_after_older_day(make_paper) -> None:
    request = ArxivSearchRequest(query="graph", field="all", category="")
    latest_day = "Mon, 17 Feb 2026"
    older_day = "Sun, 16 Feb 2026"
    page_calls: list[int] = []
    pages = [
        [
            make_paper(arxiv_id="2602.00010", date=latest_day),
            make_paper(arxiv_id="2602.00011", date=latest_day),
        ],
        [
            make_paper(arxiv_id="2602.00012", date=latest_day),
            make_paper(arxiv_id="2602.00013", date=older_day),
        ],
    ]

    def fake_fetch_page(**kwargs):
        page_calls.append(int(kwargs["start"]))
        return pages.pop(0)

    sleep = MagicMock()
    papers = fetch_latest_day_digest(
        request=request,
        max_results=2,
        fetch_page_fn=fake_fetch_page,
        sleep_fn=sleep,
        min_interval_seconds=ARXIV_API_MIN_INTERVAL_SECONDS,
    )

    assert [paper.arxiv_id for paper in papers] == ["2602.00010", "2602.00011", "2602.00012"]
    assert page_calls == [0, 2]
    sleep.assert_called_once_with(ARXIV_API_MIN_INTERVAL_SECONDS)


def test_fetch_latest_day_digest_skips_invalid_first_date(make_paper) -> None:
    request = ArxivSearchRequest(query="graph", field="all", category="")
    latest_day = "Mon, 17 Feb 2026"
    older_day = "Sun, 16 Feb 2026"
    page_calls: list[int] = []
    pages = [
        [
            make_paper(arxiv_id="2602.00010", date="not-a-date"),
            make_paper(arxiv_id="2602.00011", date=latest_day),
            make_paper(arxiv_id="2602.00012", date=latest_day),
        ],
        [
            make_paper(arxiv_id="2602.00013", date=latest_day),
            make_paper(arxiv_id="2602.00014", date=older_day),
        ],
    ]

    def fake_fetch_page(**kwargs):
        page_calls.append(int(kwargs["start"]))
        return pages.pop(0)

    papers = fetch_latest_day_digest(
        request=request,
        max_results=3,
        fetch_page_fn=fake_fetch_page,
        sleep_fn=MagicMock(),
        min_interval_seconds=ARXIV_API_MIN_INTERVAL_SECONDS,
    )

    assert [paper.arxiv_id for paper in papers] == ["2602.00011", "2602.00012", "2602.00013"]
    assert page_calls == [0, 3]
