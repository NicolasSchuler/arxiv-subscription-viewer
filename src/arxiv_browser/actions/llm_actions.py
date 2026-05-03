"""LLM, Semantic Scholar, and command-trust action handlers for ArxivBrowser.

Covers: AI summary generation, paper chat, relevance scoring, auto-tagging,
research-interests editing, Semantic Scholar paper/recommendation/citation-graph
fetching, and the security trust-gate for LLM and PDF-viewer commands.

The command trust-gate (hashing, trusted-hash persistence, confirmation
prompts) lives in :mod:`arxiv_browser.actions.trust_gate`; this module
re-exports its public surface for backwards compatibility.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure, logger
from arxiv_browser.actions.trust_gate import (
    CommandTrustRequest,
    _ensure_command_trusted,
    _ensure_llm_command_trusted,
    _ensure_pdf_viewer_trusted,
    _is_llm_command_trusted,
    _is_pdf_viewer_trusted,
    _remember_trusted_hash,
    _trust_hash,
)
from arxiv_browser.config import get_config_path, save_config
from arxiv_browser.llm import (
    LLM_PRESETS,
    SUMMARY_MODES,
    _compute_command_hash,
    _load_all_relevance_scores,
    _load_summary,
    _resolve_llm_command,
    _save_relevance_score,
    _save_summary,
)
from arxiv_browser.llm_providers import LLMProvider, llm_command_requires_shell, resolve_provider
from arxiv_browser.modals import (
    PaperChatScreen,
    PaperEditModal,
    ResearchInterestsModal,
    SummaryModeModal,
)
from arxiv_browser.modals.editing import PaperEditResult
from arxiv_browser.models import Paper

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser


__all__ = [
    "LLM_PRESETS",
    "RECOVERABLE_ACTION_ERRORS",
    "SUMMARY_MODES",
    "CommandTrustRequest",
    "LLMProvider",
    "Paper",
    "PaperChatScreen",
    "PaperEditModal",
    "PaperEditResult",
    "ResearchInterestsModal",
    "SummaryModeModal",
    "_compute_command_hash",
    "_ensure_command_trusted",
    "_ensure_llm_command_trusted",
    "_ensure_pdf_viewer_trusted",
    "_is_llm_command_trusted",
    "_is_pdf_viewer_trusted",
    "_load_all_relevance_scores",
    "_load_summary",
    "_remember_trusted_hash",
    "_resolve_llm_command",
    "_save_relevance_score",
    "_save_summary",
    "_trust_hash",
    "asyncio",
    "get_config_path",
    "llm_command_requires_shell",
    "log_action_failure",
    "logger",
    "resolve_provider",
    "save_config",
]


_RECOVERABLE_ACTION_ERRORS = RECOVERABLE_ACTION_ERRORS
_TRUST_HASH_LENGTH = 16
_COMMAND_PREVIEW_MAX_LEN = 120
_NOTIFY_TIMEOUT_DEFAULT = 8
_NOTIFY_TIMEOUT_LONG = 10
_NOTIFY_MAX_LENGTH = 200
_RELEVANCE_SCORING_CONCURRENCY = 3
_RELEVANCE_PROGRESS_NOTIFY_INTERVAL = 5


@dataclass(slots=True)
class _RelevanceBatchStats:
    total: int
    scored: int = 0
    failed: int = 0
    done: int = 0
    cancelled: bool = False


@dataclass(slots=True, frozen=True)
class _RelevanceBatchContext:
    interests: str
    provider: LLMProvider
    db_path: Path | None
    interests_hash: str
    task_epoch: int


@dataclass(slots=True)
class _AutoTagBatchStats:
    total: int
    tagged: int = 0
    failed: int = 0


def _log_action_failure(action: str, exc: Exception, *, unexpected: bool = False) -> None:
    return log_action_failure(action, exc, unexpected=unexpected)


def _collect_all_tags(app: ArxivBrowser) -> list[str]:
    """Collect all unique tags across all paper metadata."""
    return list(
        dict.fromkeys(tag for meta in app._config.paper_metadata.values() for tag in meta.tags)
    )


def _require_llm_command(app: ArxivBrowser) -> str | None:
    """Resolve the LLM command template, notifying the user if not configured.

    Also refreshes ``app._llm_provider`` to stay in sync with config changes
    since the app was started.

    Args:
        app: The running ``ArxivBrowser`` application instance.

    Returns:
        The command template string (needed for cache hashing and trust
        checks), or ``None`` if no LLM command is configured or the command
        is blocked by the ``allow_llm_shell_fallback`` setting.
    """
    command_template = _resolve_llm_command(app._config)
    if not command_template:
        preset = app._config.llm_preset
        if preset and preset not in LLM_PRESETS:
            valid = ", ".join(sorted(LLM_PRESETS))
            msg = f"Unknown preset '{preset}'. Valid: {valid}"
        else:
            msg = f"Set llm_command or llm_preset in config.json ({get_config_path()})"
        app.notify(
            msg, title="LLM not configured", severity="warning", timeout=_NOTIFY_TIMEOUT_DEFAULT
        )
        return None
    if not app._config.allow_llm_shell_fallback and llm_command_requires_shell(command_template):
        app.notify(
            "LLM command uses shell syntax, but allow_llm_shell_fallback is disabled in config.json",
            title="LLM command blocked",
            severity="warning",
            timeout=_NOTIFY_TIMEOUT_LONG,
        )
        return None
    app._llm_provider = resolve_provider(app._config)
    return command_template


def action_generate_summary(app: ArxivBrowser) -> None:
    """Generate an AI summary for the currently highlighted paper."""
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_summary_flow(command_template),
    ):
        return
    app._start_summary_flow(command_template)


def _start_summary_flow(app: ArxivBrowser, command_template: str) -> None:
    """Start the summary mode flow after command trust checks pass."""

    paper = app._get_current_paper()
    if not paper:
        app.notify("No paper selected", title="AI Summary", severity="warning")
        return

    if paper.arxiv_id in app._summary_loading:
        app.notify("Summary already generating...", title="AI Summary")
        return

    app.push_screen(
        SummaryModeModal(),
        lambda mode: app._on_summary_mode_selected(mode, paper, command_template),
    )


def _on_summary_mode_selected(app, mode: str | None, paper: Paper, command_template: str) -> None:
    """Handle the mode chosen from SummaryModeModal."""
    if not mode:
        return
    if mode not in SUMMARY_MODES:
        app.notify(f"Unknown summary mode: {mode}", title="AI Summary", severity="error")
        return

    arxiv_id = paper.arxiv_id
    if arxiv_id in app._summary_loading:
        return

    # Resolve prompt template for this mode
    if mode == "default" and app._config.llm_prompt_template:
        prompt_template = app._config.llm_prompt_template
    else:
        prompt_template = SUMMARY_MODES[mode][1]

    cmd_hash = _compute_command_hash(command_template, prompt_template)
    mode_label = mode.upper() if mode != "default" else ""
    use_full_paper_content = mode != "quick"

    # Check SQLite cache first
    cached = _load_summary(app._summary_db_path, arxiv_id, cmd_hash)
    if cached:
        app._paper_summaries[arxiv_id] = cached
        app._summary_mode_label[arxiv_id] = mode_label
        app._summary_command_hash[arxiv_id] = cmd_hash
        app._update_abstract_display(arxiv_id)
        app.notify("Summary loaded from cache", title="AI Summary")
        return

    # Avoid showing stale content under a newly selected mode.
    if app._summary_command_hash.get(arxiv_id) != cmd_hash:
        app._paper_summaries.pop(arxiv_id, None)
        app._summary_command_hash.pop(arxiv_id, None)

    # Start async generation
    app._summary_loading.add(arxiv_id)
    app._summary_mode_label[arxiv_id] = mode_label
    app._update_abstract_display(arxiv_id)
    app._track_dataset_task(
        app._generate_summary_async(
            paper,
            prompt_template,
            cmd_hash,
            mode_label=mode_label,
            use_full_paper_content=use_full_paper_content,
        )
    )


async def _generate_summary_async(
    app,
    paper: Paper,
    prompt_template: str,
    cmd_hash: str,
    mode_label: str = "",
    use_full_paper_content: bool = True,
) -> None:
    """Run the LLM CLI tool asynchronously and update the UI."""
    task_epoch = app._capture_dataset_epoch()
    arxiv_id = paper.arxiv_id
    generated_summary = False
    try:
        provider = _summary_provider(app)
        if provider is None:
            return
        if use_full_paper_content:
            app.notify("Fetching paper content...", title="AI Summary")

        summary, error = await _request_summary(
            app,
            paper=paper,
            prompt_template=prompt_template,
            provider=provider,
            use_full_paper_content=use_full_paper_content,
        )
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if summary is None:
            _notify_summary_error(app, error)
            return

        await _store_generated_summary(app, arxiv_id, summary, mode_label, cmd_hash)
        generated_summary = True
        app.notify("Summary generated", title="AI Summary")

    except asyncio.CancelledError:
        raise
    except ValueError as e:
        _handle_summary_value_error(app, arxiv_id, task_epoch, e)
    except _RECOVERABLE_ACTION_ERRORS as exc:
        _handle_summary_recoverable_error(app, arxiv_id, task_epoch, exc)
    finally:
        _finish_summary_generation(app, arxiv_id, task_epoch, generated_summary)


def _summary_provider(app: ArxivBrowser) -> LLMProvider | None:
    provider = app._llm_provider
    if provider is None:
        logger.warning("LLM provider unexpectedly None in _generate_summary_async")
    return provider


async def _request_summary(
    app: ArxivBrowser,
    *,
    paper: Paper,
    prompt_template: str,
    provider: LLMProvider,
    use_full_paper_content: bool,
) -> tuple[str | None, str | None]:
    return await app._get_services().llm.generate_summary(
        paper=paper,
        prompt_template=prompt_template,
        provider=provider,
        use_full_paper_content=use_full_paper_content,
        summary_timeout_seconds=app._config.llm_timeout,
        fetch_paper_content=app._fetch_paper_content_async,
    )


def _notify_summary_error(app: ArxivBrowser, error: str | None) -> None:
    app.notify(
        (error or "LLM command failed")[:_NOTIFY_MAX_LENGTH],
        title="AI Summary",
        severity="error",
        timeout=_NOTIFY_TIMEOUT_DEFAULT,
    )


async def _store_generated_summary(
    app: ArxivBrowser,
    arxiv_id: str,
    summary: str,
    mode_label: str,
    cmd_hash: str,
) -> None:
    app._paper_summaries[arxiv_id] = summary
    app._summary_mode_label[arxiv_id] = mode_label
    app._summary_command_hash[arxiv_id] = cmd_hash
    await asyncio.to_thread(_save_summary, app._summary_db_path, arxiv_id, summary, cmd_hash)


def _handle_summary_value_error(
    app: ArxivBrowser,
    arxiv_id: str,
    task_epoch: int,
    exc: ValueError,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    logger.warning("Summary config error for %s: %s", arxiv_id, exc)
    app.notify(str(exc), title="AI Summary", severity="error", timeout=_NOTIFY_TIMEOUT_LONG)


def _handle_summary_recoverable_error(
    app: ArxivBrowser,
    arxiv_id: str,
    task_epoch: int,
    exc: Exception,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    _log_action_failure(f"summary generation for {arxiv_id}", exc)
    app.notify("Summary failed", title="AI Summary", severity="error")


def _finish_summary_generation(
    app: ArxivBrowser,
    arxiv_id: str,
    task_epoch: int,
    generated_summary: bool,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app._summary_loading.discard(arxiv_id)
    if not generated_summary and arxiv_id not in app._paper_summaries:
        app._summary_mode_label.pop(arxiv_id, None)
        app._summary_command_hash.pop(arxiv_id, None)
    app._update_abstract_display(arxiv_id)


def action_chat_with_paper(app: ArxivBrowser) -> None:
    """Open an interactive chat session about the current paper."""
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_chat_with_paper(),
    ):
        return
    app._start_chat_with_paper()


def _start_chat_with_paper(app: ArxivBrowser) -> None:
    """Start the chat flow after command trust checks pass."""
    paper = app._get_current_paper()
    if not paper:
        app.notify("No paper selected", title="Chat", severity="warning")
        return
    if app._llm_provider is None:
        logger.warning("LLM provider unexpectedly None in _start_chat_with_paper")
        return
    app.notify("Fetching paper content...", title="Chat")
    app._track_dataset_task(app._open_chat_screen(paper, app._llm_provider))


async def _open_chat_screen(app: ArxivBrowser, paper: Paper, provider: LLMProvider) -> None:
    """Fetch paper content and open the chat modal."""
    task_epoch = app._capture_dataset_epoch()
    paper_content = await app._fetch_paper_content_async(paper)
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app.push_screen(
        PaperChatScreen(paper, provider, paper_content, timeout=app._config.llm_timeout)
    )


def action_score_relevance(app: ArxivBrowser) -> None:
    """Score all loaded papers for relevance using the configured LLM."""
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_score_relevance_flow(command_template),
    ):
        return
    app._start_score_relevance_flow(command_template)


def _start_score_relevance_flow(app: ArxivBrowser, command_template: str) -> None:
    """Start relevance scoring after command trust checks pass."""

    if app._relevance_scoring_active:
        app.notify("Relevance scoring already in progress", title="Relevance")
        return

    interests = app._config.research_interests
    if not interests:
        app.push_screen(
            ResearchInterestsModal(),
            lambda text: app._on_interests_saved_then_score(text, command_template),
        )
        return

    app._start_relevance_scoring(command_template, interests)


def _on_interests_saved_then_score(
    app: ArxivBrowser, interests: str | None, command_template: str
) -> None:
    """Callback after ResearchInterestsModal: save interests then start scoring."""
    if not interests:
        return
    if app._relevance_scoring_active:
        app.notify("Relevance scoring already in progress", title="Relevance")
        return
    app._config.research_interests = interests
    app._save_config_or_warn("research interests")
    app.notify("Research interests saved", title="Relevance")
    app._start_relevance_scoring(command_template, interests)


def _start_relevance_scoring(app: ArxivBrowser, command_template: str, interests: str) -> None:
    """Begin batch relevance scoring for all loaded papers."""
    if app._relevance_scoring_active:
        app.notify("Relevance scoring already in progress", title="Relevance")
        return
    app._relevance_scoring_active = True
    app._update_footer()
    papers = list(app.all_papers)
    app._track_dataset_task(app._score_relevance_batch_async(papers, command_template, interests))


def action_edit_interests(app: ArxivBrowser) -> None:
    """Edit research interests and clear relevance cache."""
    app.push_screen(
        ResearchInterestsModal(app._config.research_interests),
        app._on_interests_edited,
    )


def _on_interests_edited(app: ArxivBrowser, interests: str | None) -> None:
    """Callback after editing interests: save and clear cache."""
    if not interests or interests == app._config.research_interests:
        return
    app._config.research_interests = interests
    app._save_config_or_warn("research interests")
    app._relevance_scores.clear()
    app._mark_badges_dirty("relevance", immediate=True)
    app._refresh_detail_pane()
    if interests:
        app.notify("Research interests updated -- press L to re-score", title="Relevance")
    else:
        app.notify("Research interests cleared", title="Relevance")


async def _score_relevance_batch_async(
    app,
    papers: list[Paper],
    command_template: str,
    interests: str,
) -> None:
    """Background task: batch-score papers for relevance with concurrency control.

    Bulk-loads existing scores from SQLite first, then fires LLM calls for the
    uncached subset.  A ``asyncio.Semaphore(3)`` limits concurrency to three
    simultaneous LLM processes.  Each coroutine checks
    ``app._cancel_batch_requested`` after acquiring the semaphore so a
    user-initiated cancel stops new work without waiting for in-flight calls.

    Args:
        app: The running ``ArxivBrowser`` application instance.
        papers: Full list of loaded papers to score (already-cached papers are
            skipped after the bulk SQLite load).
        command_template: LLM command template; combined with *interests* to
            produce the ``interests_hash`` that namespaces the SQLite cache.
        interests: The user's research-interests text used as the scoring
            prompt context.
    """
    task_epoch = app._capture_dataset_epoch()
    try:
        provider = getattr(app, "_llm_provider", None)
        if provider is None and not hasattr(app, "_relevance_db_path"):
            _warn_missing_relevance_provider()
            return

        interests_hash = _compute_command_hash(command_template, interests)
        db_path: Path | None = getattr(app, "_relevance_db_path", None)

        cached_scores = await _load_cached_relevance_scores(db_path, interests_hash)
        if not app._is_current_dataset_epoch(task_epoch):
            return

        _apply_cached_relevance_scores(app, cached_scores)
        uncached = _uncached_relevance_papers(papers, cached_scores)

        if not uncached:
            _notify_all_relevance_cached(app, papers)
            return

        if provider is None:
            _warn_missing_relevance_provider()
            return

        context = _RelevanceBatchContext(
            interests=interests,
            provider=provider,
            db_path=db_path,
            interests_hash=interests_hash,
            task_epoch=task_epoch,
        )
        stats = _RelevanceBatchStats(total=len(uncached))
        await _score_uncached_relevance_papers(app, uncached, context, stats)
        if not app._is_current_dataset_epoch(task_epoch):
            return

        _notify_relevance_batch_result(app, papers, stats)
        _refresh_relevance_display(app)

    except asyncio.CancelledError:
        raise
    except _RECOVERABLE_ACTION_ERRORS as exc:
        _handle_relevance_batch_error(app, task_epoch, exc)
    finally:
        _finish_relevance_batch(app, task_epoch)


def _warn_missing_relevance_provider() -> None:
    logger.warning("LLM provider unexpectedly None in _score_relevance_batch")


async def _load_cached_relevance_scores(
    db_path: Path | None,
    interests_hash: str,
) -> dict[str, tuple[int, str]]:
    # Lightweight stubs may not define a DB path; treat that as no cache.
    if db_path is None:
        return {}
    return await asyncio.to_thread(_load_all_relevance_scores, db_path, interests_hash)


def _apply_cached_relevance_scores(
    app: ArxivBrowser,
    cached_scores: dict[str, tuple[int, str]],
) -> None:
    for aid, score_data in cached_scores.items():
        app._relevance_scores[aid] = score_data
    _refresh_relevance_display(app)


def _refresh_relevance_display(app: ArxivBrowser) -> None:
    app._mark_badges_dirty("relevance")
    app._refresh_detail_pane()


def _uncached_relevance_papers(
    papers: list[Paper],
    cached_scores: dict[str, tuple[int, str]],
) -> list[Paper]:
    return [paper for paper in papers if paper.arxiv_id not in cached_scores]


def _notify_all_relevance_cached(app: ArxivBrowser, papers: list[Paper]) -> None:
    app.notify(f"All {len(papers)} papers already scored", title="Relevance")


async def _score_uncached_relevance_papers(
    app: ArxivBrowser,
    papers: list[Paper],
    context: _RelevanceBatchContext,
    stats: _RelevanceBatchStats,
) -> None:
    sem = asyncio.Semaphore(_RELEVANCE_SCORING_CONCURRENCY)
    await asyncio.gather(
        *(_score_one_relevance_paper(app, paper, context, stats, sem) for paper in papers)
    )


async def _score_one_relevance_paper(
    app: ArxivBrowser,
    paper: Paper,
    context: _RelevanceBatchContext,
    stats: _RelevanceBatchStats,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        if getattr(app, "_cancel_batch_requested", False):
            stats.cancelled = True
            return
        try:
            parsed = await _request_relevance_score(app, paper, context)
            if not app._is_current_dataset_epoch(context.task_epoch):
                return
            if parsed is None:
                stats.failed += 1
                return
            await _store_relevance_score(app, paper, parsed, context, stats)
        except asyncio.CancelledError:
            raise
        except _RECOVERABLE_ACTION_ERRORS as exc:
            _log_action_failure(f"relevance scoring for {paper.arxiv_id}", exc)
            stats.failed += 1
        except Exception as exc:
            _log_action_failure(f"relevance scoring for {paper.arxiv_id}", exc, unexpected=True)
            stats.failed += 1
        finally:
            _advance_relevance_progress(app, context, stats)


async def _request_relevance_score(
    app: ArxivBrowser,
    paper: Paper,
    context: _RelevanceBatchContext,
) -> tuple[int, str] | None:
    return await app._get_services().llm.score_relevance_once(
        paper=paper,
        interests=context.interests,
        provider=context.provider,
        timeout_seconds=app._config.llm_timeout,
    )


async def _store_relevance_score(
    app: ArxivBrowser,
    paper: Paper,
    parsed: tuple[int, str],
    context: _RelevanceBatchContext,
    stats: _RelevanceBatchStats,
) -> None:
    score, reason = parsed
    app._relevance_scores[paper.arxiv_id] = (score, reason)
    if context.db_path is not None:
        await asyncio.to_thread(
            _save_relevance_score,
            context.db_path,
            paper.arxiv_id,
            context.interests_hash,
            score,
            reason,
        )
    app._update_relevance_badge(paper.arxiv_id)
    stats.scored += 1


def _advance_relevance_progress(
    app: ArxivBrowser,
    context: _RelevanceBatchContext,
    stats: _RelevanceBatchStats,
) -> None:
    if not app._is_current_dataset_epoch(context.task_epoch):
        return
    stats.done += 1
    app._scoring_progress = (stats.done, stats.total)
    app._update_footer()
    if stats.done % _RELEVANCE_PROGRESS_NOTIFY_INTERVAL == 0:
        app.notify(f"Scoring relevance {stats.done}/{stats.total}...", title="Relevance")


def _notify_relevance_batch_result(
    app: ArxivBrowser,
    papers: list[Paper],
    stats: _RelevanceBatchStats,
) -> None:
    if stats.cancelled:
        app.notify(f"Scoring cancelled after {stats.done}/{stats.total} papers", title="Relevance")
        return
    message = f"Relevance scoring complete: {stats.scored} scored"
    if stats.failed:
        message += f", {stats.failed} failed"
    cached_count = len(papers) - stats.total
    if cached_count:
        message += f", {cached_count} cached"
    app.notify(message, title="Relevance")


def _handle_relevance_batch_error(
    app: ArxivBrowser,
    task_epoch: int,
    exc: Exception,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    _log_action_failure("relevance batch scoring", exc)
    app.notify("Relevance scoring failed", title="Relevance", severity="error")


def _finish_relevance_batch(app: ArxivBrowser, task_epoch: int) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app._relevance_scoring_active = False
    app._scoring_progress = None
    app._cancel_batch_requested = False
    app._update_footer()


def _update_relevance_badge(app: ArxivBrowser, arxiv_id: str) -> None:
    """Update a single list item's relevance badge."""
    app._mark_badges_dirty("relevance")
    current = app._get_current_paper()
    if current is not None and current.arxiv_id == arxiv_id:
        app._refresh_detail_pane()


def action_auto_tag(app: ArxivBrowser) -> None:
    """Auto-tag current or selected papers using the configured LLM."""
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_auto_tag_flow(),
    ):
        return
    app._start_auto_tag_flow()


def _start_auto_tag_flow(app: ArxivBrowser) -> None:
    """Start auto-tagging after command trust checks pass."""

    if app._auto_tag_active:
        app.notify("Auto-tagging already in progress", title="Auto-Tag")
        return

    taxonomy = app._collect_all_tags()
    app._auto_tag_active = True

    if app.selected_ids:
        papers = [p for p in app.all_papers if p.arxiv_id in app.selected_ids]
        if not papers:
            app._auto_tag_active = False
            app.notify("No selected papers found", title="Auto-Tag", severity="warning")
            return
        app._auto_tag_progress = (0, len(papers))
        app._update_footer()
        app._track_dataset_task(app._auto_tag_batch_async(papers, taxonomy))
    else:
        paper = app._get_current_paper()
        if not paper:
            app._auto_tag_active = False
            app.notify("No paper selected", title="Auto-Tag", severity="warning")
            return
        current_tags = (app._tags_for(paper.arxiv_id) or [])[:]
        app._track_dataset_task(app._auto_tag_single_async(paper, taxonomy, current_tags))


def _maybe_cancel_auto_tag_batch(
    app: ArxivBrowser,
    *,
    index: int,
    total: int,
    tagged: int,
) -> bool:
    """Handle user-requested batch cancellation and report partial progress."""
    if not getattr(app, "_cancel_batch_requested", False):
        return False
    if tagged > 0:
        app._save_config_or_warn("partial auto-tag results")
    app.notify(
        f"Auto-tagging cancelled after {index - 1}/{total} papers ({tagged} tagged)",
        title="Auto-Tag",
    )
    return True


def _apply_auto_tag_batch_result(
    app: ArxivBrowser,
    *,
    paper: Paper,
    suggested: list[str],
    taxonomy: list[str],
) -> None:
    """Merge one auto-tag suggestion result into paper metadata and taxonomy."""
    meta = app._get_or_create_metadata(paper.arxiv_id)
    meta.tags = list(dict.fromkeys(meta.tags + suggested))
    for tag in suggested:
        if tag not in taxonomy:
            taxonomy.append(tag)


def _auto_tag_failure_message(tagged: int) -> str:
    """Return the standard batch auto-tag failure message."""
    if tagged:
        return f"Auto-tagging failed ({tagged} tagged before error)"
    return "Auto-tagging failed"


async def _auto_tag_single_async(
    app,
    paper: Paper,
    taxonomy: list[str],
    current_tags: list[str],
) -> None:
    """Auto-tag a single paper: call LLM, show suggestion modal."""
    task_epoch = app._capture_dataset_epoch()
    try:
        suggested = await app._call_auto_tag_llm(paper, taxonomy)
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if suggested is None:
            app.notify("Auto-tagging failed", title="Auto-Tag", severity="warning")
            return

        current_notes = ""
        if paper.arxiv_id in app._config.paper_metadata:
            current_notes = app._config.paper_metadata[paper.arxiv_id].notes

        def on_edit_result(result: PaperEditResult | None) -> None:
            if result is None:
                return
            # Save notes if the user edited them
            metadata = app._get_or_create_metadata(paper.arxiv_id)
            metadata.notes = result.notes
            # Delegate tag saving to existing handler
            app._on_auto_tag_accepted(result.tags, paper.arxiv_id)

        app.push_screen(
            PaperEditModal(
                paper.arxiv_id,
                current_notes=current_notes,
                current_tags=current_tags,
                suggested_tags=suggested,
                initial_tab="ai-tags",
            ),
            on_edit_result,
        )
    except asyncio.CancelledError:
        raise
    except _RECOVERABLE_ACTION_ERRORS as exc:
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _log_action_failure(f"auto-tag single for {paper.arxiv_id}", exc)
        app.notify("Auto-tagging failed", title="Auto-Tag", severity="error")
    finally:
        if app._is_current_dataset_epoch(task_epoch):
            app._auto_tag_active = False
            app._update_footer()


async def _auto_tag_batch_async(
    app,
    papers: list[Paper],
    taxonomy: list[str],
) -> None:
    """Batch auto-tag: call LLM for each paper, apply directly."""
    task_epoch = app._capture_dataset_epoch()
    stats = _AutoTagBatchStats(total=len(papers))
    try:
        app._auto_tag_progress = (0, stats.total)
        app._update_footer()
        await _process_auto_tag_batch(app, papers, taxonomy, task_epoch, stats)
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _complete_auto_tag_batch(app, stats)

    except asyncio.CancelledError:
        raise
    except _RECOVERABLE_ACTION_ERRORS as exc:
        _handle_auto_tag_batch_error(app, task_epoch, stats.tagged, exc)
    finally:
        _finish_auto_tag_batch(app, task_epoch)


async def _process_auto_tag_batch(
    app: ArxivBrowser,
    papers: list[Paper],
    taxonomy: list[str],
    task_epoch: int,
    stats: _AutoTagBatchStats,
) -> None:
    for index, paper in enumerate(papers, start=1):
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if _maybe_cancel_auto_tag_batch(
            app,
            index=index,
            total=stats.total,
            tagged=stats.tagged,
        ):
            return
        await _process_auto_tag_batch_paper(app, paper, taxonomy, task_epoch, stats, index)


async def _process_auto_tag_batch_paper(
    app: ArxivBrowser,
    paper: Paper,
    taxonomy: list[str],
    task_epoch: int,
    stats: _AutoTagBatchStats,
    index: int,
) -> None:
    app._auto_tag_progress = (index, stats.total)
    app._update_footer()
    suggested = await app._call_auto_tag_llm(paper, taxonomy)
    if not app._is_current_dataset_epoch(task_epoch):
        return
    if suggested is None:
        stats.failed += 1
        return
    _apply_auto_tag_batch_result(app, paper=paper, suggested=suggested, taxonomy=taxonomy)
    stats.tagged += 1


def _complete_auto_tag_batch(app: ArxivBrowser, stats: _AutoTagBatchStats) -> None:
    app._save_config_or_warn("auto-tag results")
    app._mark_badges_dirty("tags", immediate=True)
    app._refresh_detail_pane()
    app.notify(_auto_tag_success_message(stats), title="Auto-Tag")


def _auto_tag_success_message(stats: _AutoTagBatchStats) -> str:
    message = f"Auto-tagged {stats.tagged} paper{'s' if stats.tagged != 1 else ''}"
    if stats.failed:
        message += f" ({stats.failed} failed)"
    return message


def _handle_auto_tag_batch_error(
    app: ArxivBrowser,
    task_epoch: int,
    tagged: int,
    exc: Exception,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    _log_action_failure("auto-tag batch", exc)
    if tagged > 0:
        app._save_config_or_warn("partial auto-tag results")
    app.notify(_auto_tag_failure_message(tagged), title="Auto-Tag", severity="error")


def _finish_auto_tag_batch(app: ArxivBrowser, task_epoch: int) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app._auto_tag_active = False
    app._auto_tag_progress = None
    app._cancel_batch_requested = False
    app._update_footer()
