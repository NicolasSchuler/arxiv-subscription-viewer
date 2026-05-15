"""Paper-remix action helpers split out of the main LLM action module."""

from __future__ import annotations

import asyncio
from typing import Any

from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure, logger
from arxiv_browser.modals import PaperRemixResultModal
from arxiv_browser.models import Paper

_NOTIFY_TIMEOUT_DEFAULT = 8
_NOTIFY_MAX_LENGTH = 200
_RECOVERABLE_ACTION_ERRORS = RECOVERABLE_ACTION_ERRORS


def action_remix_papers(app: Any) -> None:
    """Generate a research idea from exactly 2-3 selected papers."""
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_paper_remix_flow(),
    ):
        return
    app._start_paper_remix_flow()


def _start_paper_remix_flow(app: Any) -> None:
    """Start the paper-remix generation flow after trust checks pass."""
    if getattr(app, "_paper_remix_active", False):
        app.notify("Paper remix already in progress", title="Paper Remix")
        return

    selected_count = len(app.selected_ids)
    if selected_count not in {2, 3}:
        app.notify(
            f"Select exactly 2 or 3 papers first ({selected_count} selected)",
            title="Paper Remix",
            severity="warning",
        )
        return

    papers = _selected_papers_for_remix(app)
    if len(papers) not in {2, 3}:
        app.notify(
            "Selected papers are no longer available", title="Paper Remix", severity="warning"
        )
        return

    provider = app._llm_provider
    if provider is None:
        logger.warning("LLM provider unexpectedly None in _start_paper_remix_flow")
        app.notify("LLM provider unavailable", title="Paper Remix", severity="error")
        return

    app._paper_remix_active = True
    app._update_footer()
    app.notify("Generating research idea...", title="Paper Remix")
    app._track_dataset_task(app._generate_paper_remix_async(papers, provider))


def _selected_papers_for_remix(app: Any) -> list[Paper]:
    """Return selected papers, preferring current visible order."""
    selected_ids = set(app.selected_ids)
    visible = [paper for paper in app.filtered_papers if paper.arxiv_id in selected_ids]
    visible_ids = {paper.arxiv_id for paper in visible}
    hidden = [
        paper
        for paper in app.all_papers
        if paper.arxiv_id in selected_ids and paper.arxiv_id not in visible_ids
    ]
    return [*visible, *hidden]


async def _generate_paper_remix_async(
    app: Any,
    papers: list[Paper],
    provider: Any,
) -> None:
    """Background task: ask the LLM for a paper-remix idea and open the result modal."""
    task_epoch = app._capture_dataset_epoch()
    try:
        result, error = await app._get_services().llm.generate_paper_remix(
            papers=papers,
            research_interests=app._config.research_interests,
            provider=provider,
            timeout_seconds=app._config.llm_timeout,
        )
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if result is None:
            app.notify(
                (error or "LLM command failed")[:_NOTIFY_MAX_LENGTH],
                title="Paper Remix",
                severity="error",
                timeout=_NOTIFY_TIMEOUT_DEFAULT,
            )
            return
        app.push_screen(PaperRemixResultModal(papers, result))
        app.notify("Research idea generated", title="Paper Remix")
    except asyncio.CancelledError:
        raise
    except _RECOVERABLE_ACTION_ERRORS as exc:
        if app._is_current_dataset_epoch(task_epoch):
            log_action_failure("paper remix generation", exc)
            app.notify("Paper remix failed", title="Paper Remix", severity="error")
    except Exception as exc:
        if app._is_current_dataset_epoch(task_epoch):
            log_action_failure("paper remix generation", exc, unexpected=True)
            app.notify("Paper remix failed", title="Paper Remix", severity="error")
    finally:
        app._paper_remix_active = False
        app._update_footer()
