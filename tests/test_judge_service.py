"""Service-level tests for LLM scientific-impact judge delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from arxiv_browser.llm_providers import LLMResult
from arxiv_browser.services import judge_service
from arxiv_browser.services.interfaces import DefaultLlmService


class _Provider:
    def __init__(self, result: LLMResult) -> None:
        self.execute = AsyncMock(return_value=result)


@pytest.mark.asyncio
async def test_score_impact_once_success_failure_and_parse_failure(make_paper) -> None:
    paper = make_paper(title="Impact paper")
    request = judge_service.JudgeRequest(
        paper=paper,
        provider=_Provider(
            LLMResult('{"impact":7,"significance":6,"novelty":5,"rigor":8,"clarity":9}', True)
        ),  # type: ignore[arg-type]
        timeout_seconds=12,
    )
    score = await judge_service.score_impact_once(request)
    assert score is not None and score.impact == 7
    request.provider.execute.assert_awaited_once()  # type: ignore[union-attr]
    assert "Impact paper" in request.provider.execute.call_args.args[0]  # type: ignore[union-attr]

    failed = judge_service.JudgeRequest(paper, _Provider(LLMResult("", False, "offline")), 12)  # type: ignore[arg-type]
    invalid = judge_service.JudgeRequest(paper, _Provider(LLMResult("not-json", True)), 12)  # type: ignore[arg-type]
    assert await judge_service.score_impact_once(failed) is None
    assert await judge_service.score_impact_once(invalid) is None


@pytest.mark.asyncio
async def test_compare_service_and_default_adapter_delegate_at_resolved_symbols(
    make_paper, monkeypatch
) -> None:
    left, right = make_paper(arxiv_id="a"), make_paper(arxiv_id="b")
    request = judge_service.PairwiseJudgeRequest(
        left,
        right,
        _Provider(LLMResult('{"winner":"B","reason":"stronger"}', True)),
        9,  # type: ignore[arg-type]
    )
    battle = await judge_service.compare_impact_pair(request)
    assert battle is not None and battle.winner_arxiv_id == "b"
    assert (
        await judge_service.compare_impact_pair(
            judge_service.PairwiseJudgeRequest(
                left, right, _Provider(LLMResult("", False, "offline")), 9
            )  # type: ignore[arg-type]
        )
        is None
    )
    assert (
        await judge_service.compare_impact_pair(
            judge_service.PairwiseJudgeRequest(left, right, _Provider(LLMResult("bad", True)), 9)  # type: ignore[arg-type]
        )
        is None
    )

    score_call = AsyncMock(return_value="score")
    battle_call = AsyncMock(return_value="battle")
    monkeypatch.setattr("arxiv_browser.services.interfaces._judge.score_impact_once", score_call)
    monkeypatch.setattr("arxiv_browser.services.interfaces._judge.compare_impact_pair", battle_call)
    service = DefaultLlmService()
    assert await service.score_impact_once(request) == "score"
    assert await service.compare_impact_pair(request) == "battle"
    score_call.assert_awaited_once_with(request)
    battle_call.assert_awaited_once_with(request)
