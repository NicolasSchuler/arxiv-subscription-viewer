"""LLM provider calls for local scientific-impact judging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from arxiv_browser.judging import (
    JudgeBattle,
    JudgeScore,
    build_judge_prompt,
    build_pairwise_judge_prompt,
    parse_judge_response,
    parse_pairwise_judge_response,
)
from arxiv_browser.llm_providers import LLMProvider
from arxiv_browser.models import Paper

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """Inputs for one absolute scientific-impact assessment."""

    paper: Paper
    provider: LLMProvider
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class PairwiseJudgeRequest:
    """Inputs for one pairwise scientific-impact comparison."""

    left: Paper
    right: Paper
    provider: LLMProvider
    timeout_seconds: int


async def score_impact_once(request: JudgeRequest) -> JudgeScore | None:
    """Score one paper with the local configured LLM provider."""
    context = request.paper.abstract or request.paper.abstract_raw or ""
    result = await request.provider.execute(
        build_judge_prompt(request.paper, context),
        request.timeout_seconds,
    )
    if not result.success:
        logger.debug("Impact judge failed for %s: %s", request.paper.arxiv_id, result.error)
        return None
    parsed = parse_judge_response(result.output)
    if parsed is None:
        logger.debug(
            "Impact judge parse failed for %s (output: %.200s)",
            request.paper.arxiv_id,
            result.output,
        )
    return parsed


async def compare_impact_pair(request: PairwiseJudgeRequest) -> JudgeBattle | None:
    """Compare two papers with the configured LLM provider."""
    result = await request.provider.execute(
        build_pairwise_judge_prompt(request.left, request.right),
        request.timeout_seconds,
    )
    if not result.success:
        logger.debug(
            "Pairwise impact judge failed for %s/%s: %s",
            request.left.arxiv_id,
            request.right.arxiv_id,
            result.error,
        )
        return None
    parsed = parse_pairwise_judge_response(
        result.output,
        request.left.arxiv_id,
        request.right.arxiv_id,
    )
    if parsed is None:
        logger.debug("Pairwise impact parse failed (output: %.200s)", result.output)
    return parsed


__all__ = [
    "JudgeRequest",
    "PairwiseJudgeRequest",
    "compare_impact_pair",
    "score_impact_once",
]
