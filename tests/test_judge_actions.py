"""Action orchestration tests for local scientific-impact judging."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arxiv_browser.actions import judge_actions, search_api_actions
from arxiv_browser.judging import CachedJudgeBattle, CachedJudgeScore, JudgeBattle, JudgeScore
from arxiv_browser.models import UserConfig
from tests.support.app_stubs import _new_app_stub


def _score(impact: float = 7.0) -> JudgeScore:
    return JudgeScore(impact, 6, 5, 8, 9)


def _app(papers, tmp_path):
    app = _new_app_stub()
    app.all_papers = list(papers)
    app.filtered_papers = list(papers)
    app._config = UserConfig(
        llm_command="judge {prompt}",
        llm_timeout=9,
        judge_paper_limit=500,
        judge_pairwise_top_k=0,
    )
    app._judge_scores = {}
    app._judge_db_path = tmp_path / "judge.db"
    app._judge_scoring_active = False
    app._judge_cancel_requested = False
    app._judge_progress = None
    app._capture_dataset_epoch = MagicMock(return_value=4)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    app._get_current_paper = MagicMock(return_value=papers[0] if papers else None)
    return app


def test_action_entry_guards_http_and_cli_trust(make_paper, tmp_path, monkeypatch) -> None:
    app = _app([make_paper()], tmp_path)
    app._judge_scoring_active = True
    judge_actions.action_judge_impact(app)
    assert "already in progress" in app.notify.call_args.args[0]

    app._judge_scoring_active = False
    app._config.llm_provider_type = "http"
    monkeypatch.setattr(judge_actions, "resolve_provider", lambda _config: None)
    judge_actions.action_judge_impact(app)
    assert "llm_api_base_url" in app.notify.call_args.args[0]

    provider = object()
    start = MagicMock()
    monkeypatch.setattr(judge_actions, "resolve_provider", lambda _config: provider)
    monkeypatch.setattr(judge_actions, "_start_judge_scoring", start)
    judge_actions.action_judge_impact(app)
    assert app._llm_provider is provider
    start.assert_called_once_with(app, "")

    app._config.llm_provider_type = "cli"
    app._require_llm_command = MagicMock(return_value=None)
    judge_actions.action_judge_impact(app)
    app._require_llm_command = MagicMock(return_value="judge {prompt}")
    app._ensure_llm_command_trusted = MagicMock(return_value=False)
    judge_actions.action_judge_impact(app)
    assert start.call_count == 1
    app._ensure_llm_command_trusted = MagicMock(return_value=True)
    judge_actions.action_judge_impact(app)
    assert start.call_args.args == (app, "judge {prompt}")


def test_start_and_target_selection_limits_and_missing_provider(
    make_paper, tmp_path, monkeypatch
) -> None:
    papers = [make_paper(arxiv_id=f"p{i}") for i in range(4)]
    app = _app(papers, tmp_path)
    app.selected_ids = {"p1", "p3", "missing"}
    app._config.judge_paper_limit = 2
    assert [p.arxiv_id for p in judge_actions._target_judge_papers(app)[0]] == ["p1", "p3"]
    app.selected_ids = set()
    app._config.judge_paper_limit = 2_000
    targeted, truncated = judge_actions._target_judge_papers(app)
    assert targeted == papers and truncated is False
    app._config.judge_paper_limit = 2
    targeted, truncated = judge_actions._target_judge_papers(app)
    assert len(targeted) == 2 and truncated is True

    app._llm_provider = None
    judge_actions._start_judge_scoring(app, "cmd")
    assert "No LLM provider" in app.notify.call_args.args[0]
    app._llm_provider = object()
    app.filtered_papers = []
    judge_actions._start_judge_scoring(app, "cmd")
    assert "No visible papers" in app.notify.call_args.args[0]
    app.filtered_papers = papers
    app._config.judge_paper_limit = 2
    judge_task = MagicMock()

    def track(coro):
        coro.close()
        return judge_task

    tracked = MagicMock(side_effect=track)
    app._track_dataset_task = tracked
    monkeypatch.setattr(judge_actions, "judge_identity_hash", lambda *_args: "identity")
    judge_actions._start_judge_scoring(app, "cmd")
    assert app._judge_scoring_active is True
    assert app._judge_progress == (0, 2)
    assert any("first 2" in call.args[0] for call in app.notify.call_args_list)
    assert tracked.called
    assert app._judge_task is judge_task


def test_cancel_action_cancels_inflight_judge_task(make_paper, tmp_path) -> None:
    app = _app([make_paper()], tmp_path)
    app._get_search_container_widget = MagicMock(return_value=SimpleNamespace(is_open=False))
    app._in_arxiv_api_mode = False
    app._relevance_scoring_active = False
    app._auto_tag_progress = None
    app._judge_scoring_active = True
    app._judge_task = MagicMock()
    app._judge_task.done.return_value = False
    search_api_actions.action_cancel_search(app)
    assert app._judge_cancel_requested is True
    app._judge_task.cancel.assert_called_once_with()
    assert "Cancelling impact judging" in app.notify.call_args.args[0]


def test_prepare_runtime_namespaces_scores_and_clears_old_pairwise_state(
    make_paper, tmp_path
) -> None:
    papers = [make_paper(arxiv_id="a"), make_paper(arxiv_id="b")]
    app = _app(papers, tmp_path)
    app._judge_hash = "old"
    app._judge_scores = {"a": _score(9), "b": _score(8)}
    judge_actions._prepare_judge_runtime(app, [papers[0]], "new")
    assert app._judge_scores == {}
    assert app._judge_hash == "new"

    app._judge_scores = {
        "a": _score(9),
        "b": JudgeScore(8, 6, 5, 8, 9, pairwise_score=9.5, pairwise_wins=1, pairwise_matches=1),
    }
    judge_actions._prepare_judge_runtime(app, [papers[0]], "new")
    assert "a" not in app._judge_scores
    assert app._judge_scores["b"].pairwise_score is None
    assert app._judge_scores["b"].pairwise_matches == 0


@pytest.mark.asyncio
async def test_batch_uses_valid_cache_scores_new_scores_and_reports_progress(
    make_paper, tmp_path, monkeypatch
) -> None:
    papers = [make_paper(arxiv_id="cached"), make_paper(arxiv_id="fresh")]
    app = _app(papers, tmp_path)
    context_hashes = {paper.arxiv_id: "hash-" + paper.arxiv_id for paper in papers}
    cached = {"cached": CachedJudgeScore(context_hashes["cached"], _score(8))}
    app._get_services = MagicMock(
        return_value=SimpleNamespace(
            llm=SimpleNamespace(score_impact_once=AsyncMock(return_value=_score(6)))
        )
    )

    async def immediate_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        judge_actions, "paper_context_hash", lambda paper: context_hashes[paper.arxiv_id]
    )
    monkeypatch.setattr(judge_actions, "load_judge_scores", lambda *_args: cached)
    monkeypatch.setattr(judge_actions.asyncio, "to_thread", immediate_thread)
    await judge_actions._run_judge_batch_async(app, papers, object(), "judge")

    assert app._judge_scores == {"cached": _score(8), "fresh": _score(6)}
    assert app._get_services().llm.score_impact_once.await_count == 1
    assert any("1 scored, 1 cached" in call.args[0] for call in app.notify.call_args_list)
    app._mark_badges_dirty.assert_called_with("judge", immediate=True)
    assert app._judge_scoring_active is False and app._judge_progress is None


@pytest.mark.asyncio
async def test_batch_snapshots_paper_before_cache_await(make_paper, tmp_path, monkeypatch) -> None:
    paper = make_paper(arxiv_id="a", abstract="raw abstract")
    app = _app([paper], tmp_path)
    app._judge_db_path = None

    async def load_then_mutate(*_args):
        paper.abstract = "cleaned later"
        return {}

    async def inspect_request(request):
        assert request.paper is not paper
        assert request.paper.abstract == "raw abstract"
        return _score()

    app._get_services = MagicMock(
        return_value=SimpleNamespace(llm=SimpleNamespace(score_impact_once=inspect_request))
    )
    monkeypatch.setattr(judge_actions, "_load_current_judge_scores", load_then_mutate)
    await judge_actions._run_judge_batch_async(app, [paper], object(), "judge")
    assert paper.abstract == "cleaned later"
    assert app._judge_scores["a"] == _score()


@pytest.mark.asyncio
async def test_batch_honors_cancel_after_all_cached_load(make_paper, tmp_path, monkeypatch) -> None:
    paper = make_paper(arxiv_id="a")
    app = _app([paper], tmp_path)
    app._judge_scoring_active = True

    async def load_then_cancel(*_args):
        app._judge_cancel_requested = True
        return {"a": _score()}

    monkeypatch.setattr(judge_actions, "_load_current_judge_scores", load_then_cancel)
    await judge_actions._run_judge_batch_async(app, [paper], object(), "judge")
    assert "cancelled after 1/1" in app.notify.call_args.args[0]
    assert app._judge_scoring_active is False


@pytest.mark.asyncio
async def test_cancel_failure_stale_epoch_and_pairwise_cached_battle_reuse(
    make_paper, tmp_path, monkeypatch
) -> None:
    papers = [make_paper(arxiv_id="a"), make_paper(arxiv_id="b"), make_paper(arxiv_id="c")]
    app = _app(papers, tmp_path)
    app._judge_cancel_requested = True
    app._judge_scores = {paper.arxiv_id: _score(8 - index) for index, paper in enumerate(papers)}
    app._get_services = MagicMock(
        return_value=SimpleNamespace(
            llm=SimpleNamespace(
                score_impact_once=AsyncMock(side_effect=RuntimeError("nope")),
                compare_impact_pair=AsyncMock(return_value=JudgeBattle("a", "b", "a")),
            )
        )
    )
    context = judge_actions._JudgeBatchContext(
        object(), "judge", None, {paper.arxiv_id: paper.arxiv_id for paper in papers}, 4
    )
    stats = judge_actions._JudgeBatchStats(total=3)
    await judge_actions._score_uncached_papers(app, papers, context, stats)
    assert stats.cancelled is True and stats.done == 0
    assert app._get_services().llm.score_impact_once.await_count == 0
    judge_actions._notify_judge_result(app, stats)
    assert "cancelled" in app.notify.call_args.args[0]

    app._judge_cancel_requested = False
    cached_battle = CachedJudgeBattle(JudgeBattle("a", "b", "a"), "a", "b")
    rounds = (("a", "b"),)
    results = await judge_actions._run_pairwise_round(
        app,
        rounds,
        {paper.arxiv_id: paper for paper in papers},
        {("a", "b"): cached_battle},
        context,
    )
    assert results == [cached_battle.battle]
    app._get_services().llm.compare_impact_pair.assert_not_awaited()

    stale = _app(papers[:1], tmp_path)
    stale._is_current_dataset_epoch = MagicMock(return_value=False)
    stale._judge_scoring_active = True
    stale._judge_progress = (1, 1)
    judge_actions._finish_judge_batch(stale, 4)
    assert stale._judge_scoring_active is True


@pytest.mark.asyncio
async def test_pairwise_refinement_executes_new_battles_and_handles_failure(
    make_paper, tmp_path, monkeypatch
) -> None:
    papers = [make_paper(arxiv_id="a"), make_paper(arxiv_id="b")]
    app = _app(papers, tmp_path)
    app._config.judge_pairwise_top_k = 2
    app._judge_scores = {"a": _score(8), "b": _score(6)}
    app._get_services = MagicMock(
        return_value=SimpleNamespace(
            llm=SimpleNamespace(
                compare_impact_pair=AsyncMock(return_value=JudgeBattle("a", "b", "a"))
            )
        )
    )
    context = judge_actions._JudgeBatchContext(object(), "judge", None, {"a": "a", "b": "b"}, 4)
    stats = judge_actions._JudgeBatchStats(total=2)
    await judge_actions._run_pairwise_refinement(app, papers, context, stats)
    assert stats.battles == 1
    assert app._judge_scores["a"].pairwise_matches == 1

    app._get_services().llm.compare_impact_pair = AsyncMock(return_value=None)
    stats = judge_actions._JudgeBatchStats(total=2)
    await judge_actions._run_pairwise_refinement(app, papers, context, stats)
    assert stats.battle_failed == 1


@pytest.mark.asyncio
async def test_score_one_handles_parse_failure_recoverable_unexpected_and_stale(
    make_paper, tmp_path
) -> None:
    paper = make_paper(arxiv_id="a")
    app = _app([paper], tmp_path)
    service = SimpleNamespace(score_impact_once=AsyncMock(return_value=None))
    app._get_services = MagicMock(return_value=SimpleNamespace(llm=service))
    context = judge_actions._JudgeBatchContext(object(), "judge", None, {"a": "hash"}, 4)
    stats = judge_actions._JudgeBatchStats(total=1)

    await judge_actions._score_one_paper(app, paper, context, stats, asyncio.Semaphore(1))
    assert stats.failed == 1 and stats.done == 1

    service.score_impact_once = AsyncMock(side_effect=OSError("offline"))
    await judge_actions._score_one_paper(app, paper, context, stats, asyncio.Semaphore(1))
    service.score_impact_once = AsyncMock(side_effect=Exception("unexpected"))
    await judge_actions._score_one_paper(app, paper, context, stats, asyncio.Semaphore(1))
    assert stats.failed == 3 and stats.done == 3

    service.score_impact_once = AsyncMock(return_value=_score())
    app._is_current_dataset_epoch = MagicMock(return_value=False)
    await judge_actions._score_one_paper(app, paper, context, stats, asyncio.Semaphore(1))
    assert "a" not in app._judge_scores


@pytest.mark.asyncio
async def test_batch_failures_and_stale_cache_exit_are_soft(
    make_paper, tmp_path, monkeypatch
) -> None:
    paper = make_paper(arxiv_id="a")
    app = _app([paper], tmp_path)

    async def recoverable(*_args):
        raise OSError("locked")

    monkeypatch.setattr(judge_actions, "_load_current_judge_scores", recoverable)
    app._judge_scoring_active = True
    await judge_actions._run_judge_batch_async(app, [paper], object(), "judge")
    assert any("check the configured LLM" in call.args[0] for call in app.notify.call_args_list)
    assert app._judge_scoring_active is False

    async def unexpected(*_args):
        raise Exception("boom")

    monkeypatch.setattr(judge_actions, "_load_current_judge_scores", unexpected)
    app._judge_scoring_active = True
    await judge_actions._run_judge_batch_async(app, [paper], object(), "judge")
    assert any("failed unexpectedly" in call.args[0] for call in app.notify.call_args_list)

    async def empty(*_args):
        return {}

    monkeypatch.setattr(judge_actions, "_load_current_judge_scores", empty)
    app._is_current_dataset_epoch = MagicMock(return_value=False)
    app._judge_scoring_active = True
    await judge_actions._run_judge_batch_async(app, [paper], object(), "judge")
    assert app._judge_scoring_active is True
    assert await judge_actions._load_current_judge_scores(None, "judge", {}) == {}


@pytest.mark.asyncio
async def test_pairwise_cancel_exception_persistence_and_notification_branches(
    make_paper, tmp_path
) -> None:
    papers = [make_paper(arxiv_id="a"), make_paper(arxiv_id="b")]
    app = _app(papers, tmp_path)
    app._judge_scores = {"a": _score(8), "b": _score(6)}
    app._config.judge_pairwise_top_k = 1
    service = SimpleNamespace(
        compare_impact_pair=AsyncMock(return_value=JudgeBattle("a", "b", "a"))
    )
    app._get_services = MagicMock(return_value=SimpleNamespace(llm=service))
    context = judge_actions._JudgeBatchContext(
        object(), "judge", app._judge_db_path, {"a": "ha", "b": "hb"}, 4
    )
    stats = judge_actions._JudgeBatchStats(total=2)
    await judge_actions._run_pairwise_refinement(app, papers, context, stats)
    assert stats.battles == 1
    assert await judge_actions._load_battle_cache(context)

    service.compare_impact_pair = AsyncMock(side_effect=OSError("offline"))
    no_cache_context = judge_actions._JudgeBatchContext(
        object(), "other", None, {"a": "ha", "b": "hb"}, 4
    )
    assert await judge_actions._run_pairwise_round(
        app, (("a", "b"),), {"a": papers[0], "b": papers[1]}, {}, no_cache_context
    ) == [None]
    assert await judge_actions._load_battle_cache(no_cache_context) == {}

    app._judge_cancel_requested = True
    stats = judge_actions._JudgeBatchStats(total=2)
    await judge_actions._run_pairwise_refinement(app, papers, no_cache_context, stats)
    assert stats.cancelled is True
    assert await judge_actions._run_pairwise_round(
        app, (("a", "b"),), {"a": papers[0], "b": papers[1]}, {}, no_cache_context
    ) == [None]

    app._judge_cancel_requested = False
    app._judge_scores = {"a": _score(8), "b": _score(6)}

    async def compare_then_cancel(*_args):
        app._judge_cancel_requested = True
        return JudgeBattle("a", "b", "a")

    service.compare_impact_pair = AsyncMock(side_effect=compare_then_cancel)
    stats = judge_actions._JudgeBatchStats(total=2)
    await judge_actions._run_pairwise_refinement(app, papers, no_cache_context, stats)
    assert stats.cancelled is True
    assert all(score.pairwise_score is None for score in app._judge_scores.values())

    stats = judge_actions._JudgeBatchStats(
        total=2, scored=1, cached=1, failed=1, battles=2, battle_failed=1
    )
    judge_actions._notify_judge_result(app, stats)
    assert "1 failed" in app.notify.call_args.args[0]
    assert "2 comparisons" in app.notify.call_args.args[0]
    app._get_current_paper = MagicMock(return_value=None)
    judge_actions._update_judge_badge(app, "a")
    app._refresh_detail_pane.assert_not_called()
