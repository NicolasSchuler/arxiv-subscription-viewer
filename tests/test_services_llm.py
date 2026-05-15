"""Tests for LLM service helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from arxiv_browser.services.llm_service import (
    LLMExecutionError,
    compare_papers,
    generate_paper_debate,
    generate_paper_remix,
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
async def test_generate_paper_remix_returns_text_and_uses_timeout(make_paper) -> None:
    papers = [
        make_paper(arxiv_id="2401.30101", title="A", abstract="alpha"),
        make_paper(arxiv_id="2401.30102", title="B", abstract="beta"),
    ]
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="idea", error=""))
    )

    result, error = await generate_paper_remix(
        papers=papers,
        research_interests="retrieval",
        provider=provider,
        timeout_seconds=17,
    )

    assert result == "idea"
    assert error is None
    provider.execute.assert_awaited_once()
    prompt, timeout = provider.execute.await_args.args
    assert "Research interests: retrieval" in prompt
    assert "Title: A" in prompt and "Title: B" in prompt
    assert timeout == 17


@pytest.mark.asyncio
async def test_generate_paper_remix_returns_provider_error(make_paper) -> None:
    papers = [
        make_paper(arxiv_id="2401.30103"),
        make_paper(arxiv_id="2401.30104"),
    ]
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=False, output="", error="timeout"))
    )

    result, error = await generate_paper_remix(
        papers=papers,
        research_interests="",
        provider=provider,
        timeout_seconds=30,
    )

    assert result is None
    assert error == "timeout"


@pytest.mark.asyncio
async def test_compare_papers_fetches_content_truncates_and_uses_timeout(make_paper) -> None:
    papers = [
        make_paper(arxiv_id="2401.30201", title="Methods A", abstract="alpha"),
        make_paper(arxiv_id="2401.30202", title="Results B", abstract="beta"),
    ]
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(success=True, output=" comparison ", error="")
        )
    )
    fetcher = AsyncMock(side_effect=["A" * 20, "B" * 20])

    result, error = await compare_papers(
        papers=papers,
        provider=provider,
        timeout_seconds=19,
        fetch_paper_content=fetcher,
        max_content_chars=5,
    )

    assert result == "comparison"
    assert error is None
    assert fetcher.await_count == 2
    prompt, timeout = provider.execute.await_args.args
    assert "Title: Methods A" in prompt
    assert "Title: Results B" in prompt
    assert "AAAAA" in prompt and "AAAAAA" not in prompt
    assert "BBBBB" in prompt and "BBBBBB" not in prompt
    assert "## Key Differences" in prompt
    assert timeout == 19


@pytest.mark.asyncio
async def test_compare_papers_returns_provider_error_and_empty_output(make_paper) -> None:
    papers = [make_paper(arxiv_id="2401.30203"), make_paper(arxiv_id="2401.30204")]
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=False, output="", error="timeout"))
    )

    result, error = await compare_papers(
        papers=papers,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="content"),
        max_content_chars=12000,
    )

    assert result is None
    assert error == "timeout"

    provider.execute.return_value = SimpleNamespace(success=True, output="  ", error="")
    result, error = await compare_papers(
        papers=papers,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="content"),
        max_content_chars=12000,
    )

    assert result is None
    assert error == "Empty response content"


@pytest.mark.asyncio
async def test_compare_papers_falls_back_to_abstract_when_content_fetch_fails(make_paper) -> None:
    papers = [
        make_paper(arxiv_id="2401.30205", title="Fallback A", abstract="alpha fallback"),
        make_paper(arxiv_id="2401.30206", title="Fallback B", abstract="beta fallback"),
    ]
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=True, output="ok", error=""))
    )
    fetcher = AsyncMock(side_effect=RuntimeError("fetch failed"))

    result, error = await compare_papers(
        papers=papers,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=fetcher,
        max_content_chars=12000,
    )

    prompt = provider.execute.await_args.args[0]
    assert result == "ok"
    assert error is None
    assert "alpha fallback" in prompt
    assert "beta fallback" in prompt


@pytest.mark.asyncio
async def test_generate_paper_debate_uses_two_role_calls_and_timeout(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30301", title="Debate Target", abstract="abstract")
    provider = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(success=True, output=" advocate text ", error=""),
                SimpleNamespace(success=True, output=" reviewer text ", error=""),
            ]
        )
    )
    fetcher = AsyncMock(return_value="C" * 20)

    result, error = await generate_paper_debate(
        paper=paper,
        provider=provider,
        timeout_seconds=17,
        fetch_paper_content=fetcher,
        max_content_chars=5,
    )

    assert result is not None
    assert result.advocate == "advocate text"
    assert result.reviewer == "reviewer text"
    assert error is None
    assert fetcher.await_count == 1
    advocate_prompt, advocate_timeout = provider.execute.await_args_list[0].args
    reviewer_prompt, reviewer_timeout = provider.execute.await_args_list[1].args
    assert "Debate Target" in advocate_prompt
    assert "CCCCC" in advocate_prompt and "CCCCCC" not in advocate_prompt
    assert "advocate text" in reviewer_prompt
    assert advocate_timeout == reviewer_timeout == 17


@pytest.mark.asyncio
async def test_generate_paper_debate_error_and_empty_paths(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.30302", abstract="fallback")
    provider = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(success=False, output="", error="timeout"))
    )

    result, error = await generate_paper_debate(
        paper=paper,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="content"),
        max_content_chars=12000,
    )

    assert result is None
    assert error == "timeout"

    provider.execute = AsyncMock(return_value=SimpleNamespace(success=True, output="  ", error=""))
    result, error = await generate_paper_debate(
        paper=paper,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="content"),
        max_content_chars=12000,
    )

    assert result is None
    assert error == "Empty advocate response content"

    provider.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(success=True, output="advocate", error=""),
            SimpleNamespace(success=True, output=" ", error=""),
        ]
    )
    result, error = await generate_paper_debate(
        paper=paper,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="content"),
        max_content_chars=12000,
    )

    assert result is None
    assert error == "Empty reviewer response content"

    provider.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(success=True, output="advocate", error=""),
            SimpleNamespace(success=False, output="", error="reviewer failed"),
        ]
    )
    result, error = await generate_paper_debate(
        paper=paper,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(return_value="content"),
        max_content_chars=12000,
    )

    assert result is None
    assert error == "reviewer failed"


@pytest.mark.asyncio
async def test_generate_paper_debate_falls_back_to_abstract_when_content_fetch_fails(
    make_paper,
) -> None:
    paper = make_paper(arxiv_id="2401.30303", title="Fallback Debate", abstract="abstract fallback")
    provider = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(success=True, output="advocate", error=""),
                SimpleNamespace(success=True, output="reviewer", error=""),
            ]
        )
    )

    result, error = await generate_paper_debate(
        paper=paper,
        provider=provider,
        timeout_seconds=30,
        fetch_paper_content=AsyncMock(side_effect=RuntimeError("fetch failed")),
        max_content_chars=12000,
    )

    prompt = provider.execute.await_args_list[0].args[0]
    assert result is not None
    assert error is None
    assert "abstract fallback" in prompt


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
