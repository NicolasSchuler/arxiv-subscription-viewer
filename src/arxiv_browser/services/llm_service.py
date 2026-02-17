"""Internal LLM service helpers for summary, relevance, and tag suggestion calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from arxiv_browser.llm import (
    _parse_auto_tag_response,
    _parse_relevance_response,
    build_auto_tag_prompt,
    build_llm_prompt,
    build_relevance_prompt,
)
from arxiv_browser.llm_providers import LLMProvider
from arxiv_browser.models import Paper


class LLMExecutionError(RuntimeError):
    """Raised when the underlying LLM command execution fails."""


async def generate_summary(
    *,
    paper: Paper,
    prompt_template: str,
    provider: LLMProvider,
    use_full_paper_content: bool,
    summary_timeout_seconds: int,
    fetch_paper_content: Callable[[Paper], Awaitable[str]],
) -> tuple[str | None, str | None]:
    """Generate a summary, returning (summary, error_message)."""
    if use_full_paper_content:
        paper_content = await fetch_paper_content(paper)
    else:
        abstract = paper.abstract or paper.abstract_raw or ""
        paper_content = f"Abstract:\n{abstract}" if abstract else ""

    prompt = build_llm_prompt(paper, prompt_template, paper_content)
    result = await provider.execute(prompt, summary_timeout_seconds)
    if not result.success:
        return None, result.error
    return result.output, None


async def score_relevance_once(
    *,
    paper: Paper,
    interests: str,
    provider: LLMProvider,
    timeout_seconds: int,
) -> tuple[int, str] | None:
    """Score one paper for relevance, returning parsed score tuple or None."""
    prompt = build_relevance_prompt(paper, interests)
    llm_result = await provider.execute(prompt, timeout_seconds)
    if not llm_result.success:
        return None
    return _parse_relevance_response(llm_result.output)


async def suggest_tags_once(
    *,
    paper: Paper,
    taxonomy: list[str],
    provider: LLMProvider,
    timeout_seconds: int,
) -> list[str] | None:
    """Suggest tags for one paper, returning parsed tags or None."""
    prompt = build_auto_tag_prompt(paper, taxonomy)
    llm_result = await provider.execute(prompt, timeout_seconds)
    if not llm_result.success:
        error = llm_result.error or "LLM command failed"
        raise LLMExecutionError(error)
    return _parse_auto_tag_response(llm_result.output)


__all__ = [
    "LLMExecutionError",
    "generate_summary",
    "score_relevance_once",
    "suggest_tags_once",
]
