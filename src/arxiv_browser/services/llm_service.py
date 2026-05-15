"""Internal LLM service helpers for summary, relevance, and tag suggestion calls."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from arxiv_browser.llm import (
    PaperDebateResult,
    _parse_auto_tag_response,
    _parse_relevance_response,
    build_auto_tag_prompt,
    build_llm_prompt,
    build_paper_comparison_prompt,
    build_paper_debate_advocate_prompt,
    build_paper_debate_reviewer_prompt,
    build_paper_remix_prompt,
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


async def generate_paper_remix(
    *,
    papers: list[Paper],
    research_interests: str,
    provider: LLMProvider,
    timeout_seconds: int,
) -> tuple[str | None, str | None]:
    """Generate one research-idea synthesis from 2-3 papers."""
    prompt = build_paper_remix_prompt(papers, research_interests)
    llm_result = await provider.execute(prompt, timeout_seconds)
    if not llm_result.success:
        return None, llm_result.error
    return llm_result.output, None


async def compare_papers(
    *,
    papers: list[Paper],
    provider: LLMProvider,
    timeout_seconds: int,
    fetch_paper_content: Callable[[Paper], Awaitable[str]],
    max_content_chars: int,
) -> tuple[str | None, str | None]:
    """Generate a structured comparison for 2-3 papers."""
    paper_contents: list[str] = []
    for paper in papers:
        try:
            paper_contents.append(await fetch_paper_content(paper))
        except Exception as exc:
            logger.debug("Paper comparison content fetch failed for %s: %s", paper.arxiv_id, exc)
            paper_contents.append(paper.abstract or paper.abstract_raw or "")

    prompt = build_paper_comparison_prompt(papers, paper_contents, max_content_chars)
    llm_result = await provider.execute(prompt, timeout_seconds)
    if not llm_result.success:
        return None, llm_result.error
    output = llm_result.output.strip()
    if not output:
        return None, "Empty response content"
    return output, None


async def generate_paper_debate(
    *,
    paper: Paper,
    provider: LLMProvider,
    timeout_seconds: int,
    fetch_paper_content: Callable[[Paper], Awaitable[str]],
    max_content_chars: int,
) -> tuple[PaperDebateResult | None, str | None]:
    """Generate an advocate-vs-Reviewer-2 debate for one paper."""
    try:
        paper_content = await fetch_paper_content(paper)
    except Exception as exc:
        logger.debug("Paper debate content fetch failed for %s: %s", paper.arxiv_id, exc)
        paper_content = paper.abstract or paper.abstract_raw or ""

    advocate_prompt = build_paper_debate_advocate_prompt(
        paper,
        paper_content,
        max_content_chars,
    )
    advocate_result = await provider.execute(advocate_prompt, timeout_seconds)
    if not advocate_result.success:
        return None, advocate_result.error
    advocate = advocate_result.output.strip()
    if not advocate:
        return None, "Empty advocate response content"

    reviewer_prompt = build_paper_debate_reviewer_prompt(
        paper,
        paper_content,
        advocate,
        max_content_chars,
    )
    reviewer_result = await provider.execute(reviewer_prompt, timeout_seconds)
    if not reviewer_result.success:
        return None, reviewer_result.error
    reviewer = reviewer_result.output.strip()
    if not reviewer:
        return None, "Empty reviewer response content"

    return PaperDebateResult(advocate=advocate, reviewer=reviewer), None


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
    "compare_papers",
    "generate_paper_debate",
    "generate_paper_remix",
    "generate_summary",
    "score_relevance_once",
    "suggest_tags_once",
]
