"""Internal LLM service helpers for summary, relevance, and tag suggestion calls."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
    """Score one paper for relevance against the user's research interests.

    Args:
        paper: The paper to score.
        interests: Free-text description of the user's research interests.
        provider: An ``LLMProvider`` instance used to execute the prompt.
        timeout_seconds: Maximum seconds to wait for the LLM response.

    Returns:
        A ``(score, reason)`` tuple where ``score`` is an integer 1-10 and
        ``reason`` is the LLM's one-sentence explanation, or ``None`` when the
        LLM call fails *or* the response cannot be parsed.  Callers should
        treat ``None`` as "score unavailable" rather than an error — the
        failure is logged at DEBUG level and the caller can decide whether to
        retry or skip.
    """
    prompt = build_relevance_prompt(paper, interests)
    llm_result = await provider.execute(prompt, timeout_seconds)
    if not llm_result.success:
        logger.debug("Relevance LLM failed for %s: %s", paper.arxiv_id, llm_result.error)
        return None
    parsed = _parse_relevance_response(llm_result.output)
    if parsed is None:
        logger.debug(
            "Relevance parse failed for %s (output: %.200s)", paper.arxiv_id, llm_result.output
        )
    return parsed


async def suggest_tags_once(
    *,
    paper: Paper,
    taxonomy: list[str],
    provider: LLMProvider,
    timeout_seconds: int,
) -> list[str] | None:
    """Suggest tags for one paper using the LLM.

    Unlike ``score_relevance_once``, this function **raises** on LLM execution
    failure rather than returning ``None``, because callers (batch auto-tag
    actions) need to distinguish between "LLM unavailable" (abort the batch)
    and "response unparseable" (skip this paper).

    Args:
        paper: The paper to tag.
        taxonomy: Existing tag strings from the user's library, provided as
            context so the LLM can reuse consistent tag vocabulary.
        provider: An ``LLMProvider`` instance used to execute the prompt.
        timeout_seconds: Maximum seconds to wait for the LLM response.

    Returns:
        A list of suggested tag strings (lowercased, stripped) if the LLM
        responds successfully and the response can be parsed, or ``None`` if
        the response is unparseable (the paper is skipped, not the batch).

    Raises:
        LLMExecutionError: If the underlying LLM command fails (non-zero exit,
            timeout, empty output, etc.).  The caller should abort the batch.
    """
    prompt = build_auto_tag_prompt(paper, taxonomy)
    llm_result = await provider.execute(prompt, timeout_seconds)
    if not llm_result.success:
        error = llm_result.error or "LLM command failed"
        raise LLMExecutionError(error)
    tags = _parse_auto_tag_response(llm_result.output)
    if tags is None:
        logger.debug(
            "Auto-tag parse failed for %s (output: %.200s)", paper.arxiv_id, llm_result.output
        )
    return tags


__all__ = [
    "LLMExecutionError",
    "generate_summary",
    "score_relevance_once",
    "suggest_tags_once",
]
