"""Tests for service interface adapters and defaults."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arxiv_browser.models import ArxivSearchRequest, UserConfig
from arxiv_browser.services.interfaces import (
    AppServices,
    ArxivApiService,
    DownloadService,
    LlmService,
    build_default_app_services,
)


def test_build_default_app_services_protocol_compatible() -> None:
    services = build_default_app_services()

    assert isinstance(services, AppServices)
    assert isinstance(services.arxiv_api, ArxivApiService)
    assert isinstance(services.llm, LlmService)
    assert isinstance(services.download, DownloadService)


@pytest.mark.asyncio
async def test_default_arxiv_api_adapter_delegates(make_paper) -> None:
    request = ArxivSearchRequest(query="transformers", field="title", category="cs.AI")
    services = build_default_app_services()

    with (
        patch(
            "arxiv_browser.services.interfaces._arxiv_api.format_query_label", return_value="label"
        ) as fmt,
        patch(
            "arxiv_browser.services.interfaces._arxiv_api.enforce_rate_limit",
            new=AsyncMock(return_value=(12.3, 0.4)),
        ) as limit,
        patch(
            "arxiv_browser.services.interfaces._arxiv_api.fetch_page",
            new=AsyncMock(return_value=[make_paper(arxiv_id="2401.22222")]),
        ) as fetch,
    ):
        label = services.arxiv_api.format_query_label(request)
        last, waited = await services.arxiv_api.enforce_rate_limit(
            last_request_at=1.0,
            min_interval_seconds=3.0,
            now=lambda: 10.0,
            sleep=AsyncMock(),
        )
        papers = await services.arxiv_api.fetch_page(
            client=None,
            request=request,
            start=0,
            max_results=10,
            timeout_seconds=30,
            user_agent="arxiv-subscription-viewer/1.0",
        )

    assert label == "label"
    assert (last, waited) == (12.3, 0.4)
    assert len(papers) == 1
    fmt.assert_called_once_with(request)
    limit.assert_awaited_once()
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_llm_adapter_delegates(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.33333")
    provider = AsyncMock()
    services = build_default_app_services()

    with (
        patch(
            "arxiv_browser.services.interfaces._llm.generate_summary",
            new=AsyncMock(return_value=("summary", None)),
        ) as gen,
        patch(
            "arxiv_browser.services.interfaces._llm.score_relevance_once",
            new=AsyncMock(return_value=(8, "fit")),
        ) as score,
        patch(
            "arxiv_browser.services.interfaces._llm.suggest_tags_once",
            new=AsyncMock(return_value=["topic:ml"]),
        ) as tags,
    ):
        summary, error = await services.llm.generate_summary(
            paper=paper,
            prompt_template="{paper_content}",
            provider=provider,
            use_full_paper_content=False,
            summary_timeout_seconds=10,
            fetch_paper_content=AsyncMock(return_value=""),
        )
        relevance = await services.llm.score_relevance_once(
            paper=paper,
            interests="ml",
            provider=provider,
            timeout_seconds=10,
        )
        suggested = await services.llm.suggest_tags_once(
            paper=paper,
            taxonomy=["topic:ml"],
            provider=provider,
            timeout_seconds=10,
        )

    assert summary == "summary"
    assert error is None
    assert relevance == (8, "fit")
    assert suggested == ["topic:ml"]
    gen.assert_awaited_once()
    score.assert_awaited_once()
    tags.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_download_adapter_delegates(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.44444")
    services = build_default_app_services()

    with patch(
        "arxiv_browser.services.interfaces._download.download_pdf",
        new=AsyncMock(return_value=True),
    ) as download:
        ok = await services.download.download_pdf(
            paper=paper,
            config=UserConfig(),
            client=None,
            timeout_seconds=30,
        )

    assert ok is True
    download.assert_awaited_once()
