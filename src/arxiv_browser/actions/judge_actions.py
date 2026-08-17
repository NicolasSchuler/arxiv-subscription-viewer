"""Local LLM-as-judge batch scoring and pairwise tournament actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure
from arxiv_browser.judging import (
    CachedJudgeBattle,
    JudgeBattle,
    JudgeScore,
    build_tournament_rounds,
    cached_battle_matches,
    judge_identity_hash,
    load_judge_battles,
    load_judge_scores,
    paper_context_hash,
    refine_judge_scores,
    save_judge_battle,
    save_judge_score,
)
from arxiv_browser.llm_providers import LLMProvider, resolve_provider
from arxiv_browser.models import Paper
from arxiv_browser.services.judge_service import JudgeRequest, PairwiseJudgeRequest

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser

_JUDGE_CONCURRENCY = 3
_PAIRWISE_CONCURRENCY = 2
_MAX_JUDGE_PAPERS = 500
_MAX_PAIRWISE_PAPERS = 32
_PROGRESS_NOTIFY_INTERVAL = 5


@dataclass(slots=True)
class _JudgeBatchStats:
    total: int
    cached: int = 0
    scored: int = 0
    failed: int = 0
    done: int = 0
    battles: int = 0
    battle_failed: int = 0
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class _JudgeBatchContext:
    provider: LLMProvider
    judge_hash: str
    db_path: Path | None
    context_hashes: dict[str, str]
    task_epoch: int


def action_judge_impact(app: ArxivBrowser) -> None:
    """Score selected or visible papers for potential scientific impact."""
    if getattr(app, "_judge_scoring_active", False):
        app.notify("Impact judging already in progress", title="AI Judge")
        return

    if app._config.llm_provider_type.lower() == "http":
        provider = resolve_provider(app._config)
        if provider is None:
            app.notify(
                "Configure llm_api_base_url before running the AI judge",
                title="LLM not configured",
                severity="warning",
            )
            return
        app._llm_provider = provider
        _start_judge_scoring(app, "")
        return

    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: _start_judge_scoring(app, command_template),
    ):
        return
    _start_judge_scoring(app, command_template)


def _start_judge_scoring(app: ArxivBrowser, command_template: str) -> None:
    """Start a dataset-bound local judge batch after provider trust checks."""
    if getattr(app, "_judge_scoring_active", False):
        app.notify("Impact judging already in progress", title="AI Judge")
        return
    provider = getattr(app, "_llm_provider", None)
    if provider is None:
        app.notify("No LLM provider is configured", title="AI Judge", severity="warning")
        return
    papers, truncated = _target_judge_papers(app)
    if not papers:
        app.notify("No visible papers to judge", title="AI Judge", severity="warning")
        return
    if truncated:
        app.notify(
            f"Judging the first {len(papers)} papers; change judge_paper_limit to score more",
            title="AI Judge",
        )
    app._judge_scoring_active = True
    app._judge_cancel_requested = False
    app._judge_progress = (0, len(papers))
    app._update_footer()
    judge_hash = judge_identity_hash(app._config, command_template)
    app._judge_task = app._track_dataset_task(
        _run_judge_batch_async(app, papers, provider, judge_hash)
    )


def _target_judge_papers(app: ArxivBrowser) -> tuple[list[Paper], bool]:
    """Return selected papers, or the current visible dataset, within the cost cap."""
    if app.selected_ids:
        papers = [paper for paper in app.all_papers if paper.arxiv_id in app.selected_ids]
    else:
        papers = list(app.filtered_papers)
    limit = max(1, min(_MAX_JUDGE_PAPERS, app._config.judge_paper_limit))
    return papers[:limit], len(papers) > limit


async def _run_judge_batch_async(
    app: ArxivBrowser,
    papers: list[Paper],
    provider: LLMProvider,
    judge_hash: str,
) -> None:
    """Load cached scores, judge missing papers, then optionally compare top papers."""
    task_epoch = app._capture_dataset_epoch()
    papers = [replace(paper) for paper in papers]
    context_hashes = {paper.arxiv_id: paper_context_hash(paper) for paper in papers}
    db_path: Path | None = getattr(app, "_judge_db_path", None)
    context = _JudgeBatchContext(
        provider=provider,
        judge_hash=judge_hash,
        db_path=db_path,
        context_hashes=context_hashes,
        task_epoch=task_epoch,
    )
    stats = _JudgeBatchStats(total=len(papers))
    try:
        _prepare_judge_runtime(app, papers, judge_hash)
        cached = await _load_current_judge_scores(db_path, judge_hash, context_hashes)
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _apply_cached_judge_scores(app, cached)
        stats.cached = stats.done = len(cached)
        app._judge_progress = (stats.done, stats.total)
        if getattr(app, "_judge_cancel_requested", False):
            stats.cancelled = True
            _notify_judge_result(app, stats)
            return

        uncached = [paper for paper in papers if paper.arxiv_id not in cached]
        await _score_uncached_papers(app, uncached, context, stats)
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if not stats.cancelled:
            await _run_pairwise_refinement(app, papers, context, stats)
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _notify_judge_result(app, stats)
        _refresh_judge_display(app, immediate=True)
    except asyncio.CancelledError:
        raise
    except RECOVERABLE_ACTION_ERRORS as exc:
        log_action_failure("scientific impact judging", exc)
        if app._is_current_dataset_epoch(task_epoch):
            app.notify(
                "Impact judging failed; check the configured LLM",
                title="AI Judge",
                severity="error",
            )
    except Exception as exc:
        log_action_failure("scientific impact judging", exc, unexpected=True)
        if app._is_current_dataset_epoch(task_epoch):
            app.notify(
                "Impact judging failed unexpectedly",
                title="AI Judge",
                severity="error",
            )
    finally:
        _finish_judge_batch(app, task_epoch)


async def _load_current_judge_scores(
    db_path: Path | None,
    judge_hash: str,
    context_hashes: dict[str, str],
) -> dict[str, JudgeScore]:
    if db_path is None:
        return {}
    cached = await asyncio.to_thread(load_judge_scores, db_path, judge_hash)
    return {
        paper_id: entry.score
        for paper_id, entry in cached.items()
        if context_hashes.get(paper_id) == entry.context_hash
    }


def _apply_cached_judge_scores(app: ArxivBrowser, scores: dict[str, JudgeScore]) -> None:
    app._judge_scores.update(scores)
    if scores:
        _refresh_judge_display(app)


def _prepare_judge_runtime(
    app: ArxivBrowser,
    papers: list[Paper],
    judge_hash: str,
) -> None:
    """Remove scores that cannot be compared safely with the pending batch."""
    scores = app._judge_scores
    changed = False
    if getattr(app, "_judge_hash", "") != judge_hash:
        changed = bool(scores)
        scores.clear()
    else:
        for paper in papers:
            changed = scores.pop(paper.arxiv_id, None) is not None or changed
    for paper_id, score in list(scores.items()):
        if score.pairwise_score is not None:
            scores[paper_id] = replace(
                score,
                pairwise_score=None,
                pairwise_wins=0.0,
                pairwise_matches=0,
            )
            changed = True
    app._judge_hash = judge_hash
    if changed:
        _refresh_judge_display(app, immediate=True)


async def _score_uncached_papers(
    app: ArxivBrowser,
    papers: list[Paper],
    context: _JudgeBatchContext,
    stats: _JudgeBatchStats,
) -> None:
    semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)
    await asyncio.gather(
        *(_score_one_paper(app, paper, context, stats, semaphore) for paper in papers)
    )


async def _score_one_paper(
    app: ArxivBrowser,
    paper: Paper,
    context: _JudgeBatchContext,
    stats: _JudgeBatchStats,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        if getattr(app, "_judge_cancel_requested", False):
            stats.cancelled = True
            return
        try:
            score = await app._get_services().llm.score_impact_once(
                JudgeRequest(
                    paper=paper,
                    provider=context.provider,
                    timeout_seconds=app._config.llm_timeout,
                )
            )
            if not app._is_current_dataset_epoch(context.task_epoch):
                return
            if score is None:
                stats.failed += 1
                return
            app._judge_scores[paper.arxiv_id] = score
            if context.db_path is not None:
                await asyncio.to_thread(
                    save_judge_score,
                    context.db_path,
                    context.judge_hash,
                    paper.arxiv_id,
                    context.context_hashes[paper.arxiv_id],
                    score,
                )
            _update_judge_badge(app, paper.arxiv_id)
            stats.scored += 1
        except asyncio.CancelledError:
            raise
        except RECOVERABLE_ACTION_ERRORS as exc:
            log_action_failure(f"impact judging for {paper.arxiv_id}", exc)
            stats.failed += 1
        except Exception as exc:
            log_action_failure(f"impact judging for {paper.arxiv_id}", exc, unexpected=True)
            stats.failed += 1
        finally:
            if getattr(app, "_judge_cancel_requested", False):
                stats.cancelled = True
            _advance_judge_progress(app, context, stats)


def _advance_judge_progress(
    app: ArxivBrowser,
    context: _JudgeBatchContext,
    stats: _JudgeBatchStats,
) -> None:
    if not app._is_current_dataset_epoch(context.task_epoch):
        return
    stats.done += 1
    app._judge_progress = (stats.done, stats.total)
    app._update_footer()
    if stats.done % _PROGRESS_NOTIFY_INTERVAL == 0 and stats.done < stats.total:
        app.notify(f"Judging impact {stats.done}/{stats.total}...", title="AI Judge")


async def _run_pairwise_refinement(
    app: ArxivBrowser,
    papers: list[Paper],
    context: _JudgeBatchContext,
    stats: _JudgeBatchStats,
) -> None:
    top_k = max(0, min(_MAX_PAIRWISE_PAPERS, app._config.judge_pairwise_top_k))
    if top_k == 1:
        top_k = 2
    available = [paper for paper in papers if paper.arxiv_id in app._judge_scores]
    if top_k < 2 or len(available) < 2:
        return
    available.sort(key=lambda paper: app._judge_scores[paper.arxiv_id].impact, reverse=True)
    finalists = available[: min(top_k, len(available))]
    rounds = build_tournament_rounds(paper.arxiv_id for paper in finalists)
    all_pairs = [pair for round_pairs in rounds for pair in round_pairs]
    if not all_pairs:
        return
    app._judge_progress = (0, len(all_pairs))
    app._update_footer()
    cached = await _load_battle_cache(context)
    paper_by_id = {paper.arxiv_id: paper for paper in finalists}
    battles: list[JudgeBattle] = []
    battle_done = 0
    for round_pairs in rounds:
        if getattr(app, "_judge_cancel_requested", False):
            stats.cancelled = True
            break
        round_results = await _run_pairwise_round(
            app,
            round_pairs,
            paper_by_id,
            cached,
            context,
        )
        for battle in round_results:
            if battle is None:
                stats.battle_failed += 1
            else:
                battles.append(battle)
                stats.battles += 1
            battle_done += 1
            app._judge_progress = (battle_done, len(all_pairs))
            app._update_footer()
        if getattr(app, "_judge_cancel_requested", False):
            stats.cancelled = True
            break
    if not stats.cancelled:
        app._judge_scores.update(refine_judge_scores(app._judge_scores, battles))


async def _load_battle_cache(
    context: _JudgeBatchContext,
) -> dict[tuple[str, str], CachedJudgeBattle]:
    if context.db_path is None:
        return {}
    return await asyncio.to_thread(load_judge_battles, context.db_path, context.judge_hash)


async def _run_pairwise_round(
    app: ArxivBrowser,
    pairs: tuple[tuple[str, str], ...],
    paper_by_id: dict[str, Paper],
    cache: dict[tuple[str, str], CachedJudgeBattle],
    context: _JudgeBatchContext,
) -> list[JudgeBattle | None]:
    semaphore = asyncio.Semaphore(_PAIRWISE_CONCURRENCY)

    async def _resolve(pair: tuple[str, str]) -> JudgeBattle | None:
        async with semaphore:
            cached = cache.get(pair)
            if cached is not None and cached_battle_matches(cached, context.context_hashes):
                return cached.battle
            if getattr(app, "_judge_cancel_requested", False):
                return None
            left, right = pair
            try:
                battle = await app._get_services().llm.compare_impact_pair(
                    PairwiseJudgeRequest(
                        left=paper_by_id[left],
                        right=paper_by_id[right],
                        provider=context.provider,
                        timeout_seconds=app._config.llm_timeout,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_action_failure(f"pairwise impact judging for {left}/{right}", exc)
                return None
            if battle is None or not app._is_current_dataset_epoch(context.task_epoch):
                return None
            cached_battle = CachedJudgeBattle(
                battle=battle,
                left_context_hash=context.context_hashes[battle.left_arxiv_id],
                right_context_hash=context.context_hashes[battle.right_arxiv_id],
            )
            cache[pair] = cached_battle
            if context.db_path is not None:
                await asyncio.to_thread(
                    save_judge_battle,
                    context.db_path,
                    context.judge_hash,
                    cached_battle,
                )
            return battle

    return await asyncio.gather(*(_resolve(pair) for pair in pairs))


def _notify_judge_result(app: ArxivBrowser, stats: _JudgeBatchStats) -> None:
    if stats.cancelled:
        app.notify(
            f"Impact judging cancelled after {stats.done}/{stats.total} papers", title="AI Judge"
        )
        return
    message = f"Impact judging complete: {stats.scored} scored"
    if stats.cached:
        message += f", {stats.cached} cached"
    if stats.failed:
        message += f", {stats.failed} failed"
    if stats.battles:
        message += f", {stats.battles} comparisons"
    if stats.battle_failed:
        message += f", {stats.battle_failed} comparisons failed"
    app.notify(message, title="AI Judge")


def _refresh_judge_display(app: ArxivBrowser, *, immediate: bool = False) -> None:
    app._mark_badges_dirty("judge", immediate=immediate)
    app._refresh_detail_pane()


def _update_judge_badge(app: ArxivBrowser, arxiv_id: str) -> None:
    app._mark_badges_dirty("judge")
    current = app._get_current_paper()
    if current is not None and current.arxiv_id == arxiv_id:
        app._refresh_detail_pane()


def _finish_judge_batch(app: ArxivBrowser, task_epoch: int) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app._judge_scoring_active = False
    app._judge_progress = None
    app._judge_cancel_requested = False
    app._judge_task = None
    app._update_footer()


__all__ = ["action_judge_impact"]
