"""Tests for LLM service helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from arxiv_browser.services.llm_service import (
    LLMExecutionError,
    generate_summary,
    score_relevance_once,
    suggest_tags_once,
)


@pytest.mark.asyncio
async def test_generate_summary_full_content_calls_fetcher(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30001")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="summary", error=""))
    )
    fetcher = AsyncMock(return_value="full paper text")

    summary, error = await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=True,
        summary_timeout_seconds=30,
        fetch_paper_content=fetcher,
    )

    assert summary == "summary"
    assert error is None
    fetcher.assert_awaited_once_with(paper)


@pytest.mark.asyncio
async def test_generate_summary_quick_mode_skips_fetcher(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30002", abstract="short abstract")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="summary", error=""))
    )
    fetcher = AsyncMock(return_value="full paper text")

    summary, error = await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
        summary_timeout_seconds=30,
        fetch_paper_content=fetcher,
    )

    assert summary == "summary"
    assert error is None
    fetcher.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_summary_returns_provider_error(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30003")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=False, output="", error="failed"))
    )

    summary, error = await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
        summary_timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="unused"),
    )

    assert summary is None
    assert error == "failed"


@pytest.mark.asyncio
async def test_score_relevance_once_parses_valid_json(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30004")
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                output='{"score": 9, "reason": "Strong fit"}',
                error="",
            )
        )
    )

    result = await score_relevance_once(
        paper=paper,
        interests="llm",
        provider=provider,
        timeout_seconds=30,
    )

    assert result == (9, "Strong fit")


@pytest.mark.asyncio
async def test_suggest_tags_once_parses_valid_json(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30005")
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                output='{"tags": ["topic:ml", "method:transformer"]}',
                error="",
            )
        )
    )

    result = await suggest_tags_once(
        paper=paper,
        taxonomy=["topic:ml"],
        provider=provider,
        timeout_seconds=30,
    )

    assert result == ["topic:ml", "method:transformer"]


@pytest.mark.asyncio
async def test_suggest_tags_once_raises_on_provider_failure(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30006")
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                success=False,
                output="",
                error="command failed",
            )
        )
    )

    with pytest.raises(LLMExecutionError, match="command failed"):
        await suggest_tags_once(
            paper=paper,
            taxonomy=["topic:ml"],
            provider=provider,
            timeout_seconds=30,
        )
