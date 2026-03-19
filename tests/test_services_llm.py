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


# ── Error-path tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_summary_provider_timeout(make_paper) -> None:
    """Provider returning a timeout error should propagate through generate_summary."""
    paper = make_paper(arxiv_id="2401.40001")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=False, output="", error="timeout"))
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
    assert error == "timeout"


@pytest.mark.asyncio
async def test_generate_summary_empty_output(make_paper) -> None:
    """Provider returning success with empty output should return the empty string."""
    paper = make_paper(arxiv_id="2401.40002")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="", error=""))
    )

    summary, error = await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
        summary_timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="unused"),
    )

    assert summary == ""
    assert error is None


@pytest.mark.asyncio
async def test_generate_summary_whitespace_only_output(make_paper) -> None:
    """Provider returning success with whitespace-only output should return it as-is."""
    paper = make_paper(arxiv_id="2401.40003")
    whitespace = "   \n  "
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output=whitespace, error=""))
    )

    summary, error = await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
        summary_timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="unused"),
    )

    assert summary == whitespace
    assert error is None


@pytest.mark.asyncio
async def test_score_relevance_malformed_json(make_paper) -> None:
    """Non-JSON output from provider should cause score_relevance_once to return None."""
    paper = make_paper(arxiv_id="2401.40004")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="not json", error=""))
    )

    result = await score_relevance_once(
        paper=paper,
        interests="llm",
        provider=provider,
        timeout_seconds=30,
    )

    assert result is None


@pytest.mark.asyncio
async def test_score_relevance_wrong_json_type(make_paper) -> None:
    """Score field containing a non-numeric string should cause parsing to fail."""
    paper = make_paper(arxiv_id="2401.40005")
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                output='{"score": "text", "reason": "interesting"}',
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

    assert result is None


@pytest.mark.asyncio
async def test_score_relevance_missing_fields(make_paper) -> None:
    """Empty JSON object with no score field should return None."""
    paper = make_paper(arxiv_id="2401.40006")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="{}", error=""))
    )

    result = await score_relevance_once(
        paper=paper,
        interests="llm",
        provider=provider,
        timeout_seconds=30,
    )

    assert result is None


@pytest.mark.asyncio
async def test_suggest_tags_null_in_list(make_paper) -> None:
    """Non-string values in the tags list should be ignored."""
    paper = make_paper(arxiv_id="2401.40007")
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                output='{"tags": [null, "valid:tag"]}',
                error="",
            )
        )
    )

    result = await suggest_tags_once(
        paper=paper,
        taxonomy=["valid:tag"],
        provider=provider,
        timeout_seconds=30,
    )

    assert result == ["valid:tag"]


@pytest.mark.asyncio
async def test_suggest_tags_empty_response(make_paper) -> None:
    """Empty string output on success should cause suggest_tags_once to return None."""
    paper = make_paper(arxiv_id="2401.40008")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="", error=""))
    )

    result = await suggest_tags_once(
        paper=paper,
        taxonomy=["topic:ml"],
        provider=provider,
        timeout_seconds=30,
    )

    assert result is None


@pytest.mark.asyncio
async def test_generate_summary_with_custom_timeout(make_paper) -> None:
    """The summary_timeout_seconds parameter should be forwarded to the provider."""
    paper = make_paper(arxiv_id="2401.40009")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="ok", error=""))
    )

    await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
        summary_timeout_seconds=120,
        fetch_paper_content=AsyncMock(return_value="unused"),
    )

    provider.execute.assert_awaited_once()
    _prompt_arg, timeout_arg = provider.execute.call_args[0]
    assert timeout_arg == 120


@pytest.mark.asyncio
async def test_score_relevance_provider_failure(make_paper) -> None:
    """Provider returning success=False should cause score_relevance_once to return None."""
    paper = make_paper(arxiv_id="2401.40010")
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(success=False, output="", error="process crashed")
        )
    )

    result = await score_relevance_once(
        paper=paper,
        interests="llm",
        provider=provider,
        timeout_seconds=30,
    )

    assert result is None
