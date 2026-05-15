"""Paper debate action helpers split out of the main LLM action module."""

from __future__ import annotations

import asyncio
from typing import Any

from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure, logger
from arxiv_browser.llm import PAPER_DEBATE_CONTENT_MAX_CHARS
from arxiv_browser.modals import PaperDebateResultModal
from arxiv_browser.models import Paper

_NOTIFY_TIMEOUT_DEFAULT = 8
_NOTIFY_MAX_LENGTH = 200
_RECOVERABLE_ACTION_ERRORS = RECOVERABLE_ACTION_ERRORS


def action_debate_paper(app: Any) -> None:
    """Generate an advocate-vs-Reviewer-2 debate for the current paper."""
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: _start_paper_debate_flow(app),
    ):
        return
    _start_paper_debate_flow(app)


def _start_paper_debate_flow(app: Any) -> None:
    """Start paper-debate generation after trust checks pass."""
    if getattr(app, "_paper_debate_active", False):
        app.notify("Paper debate already in progress", title="Debate Paper")
        return

    paper = app._get_current_paper()
    if not paper:
        app.notify("No paper selected", title="Debate Paper", severity="warning")
        return

    provider = app._llm_provider
    if provider is None:
        logger.warning("LLM provider unexpectedly None in _start_paper_debate_flow")
        app.notify("LLM provider unavailable", title="Debate Paper", severity="error")
        return

    app._paper_debate_active = True
    app._update_footer()
    app.notify("Generating paper debate...", title="Debate Paper")
    app._track_dataset_task(_generate_paper_debate_async(app, paper, provider))


async def _generate_paper_debate_async(
    app: Any,
    paper: Paper,
    provider: Any,
) -> None:
    """Background task: ask the LLM for a debate and open the result modal."""
    task_epoch = app._capture_dataset_epoch()
    try:
        result, error = await app._get_services().llm.generate_paper_debate(
            paper=paper,
            provider=provider,
            timeout_seconds=app._config.llm_timeout,
            fetch_paper_content=app._fetch_paper_content_async,
            max_content_chars=PAPER_DEBATE_CONTENT_MAX_CHARS,
        )
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if result is None:
            app.notify(
                (error or "LLM command failed")[:_NOTIFY_MAX_LENGTH],
                title="Debate Paper",
                severity="error",
                timeout=_NOTIFY_TIMEOUT_DEFAULT,
            )
            return
        app.push_screen(PaperDebateResultModal(paper, result))
        app.notify("Paper debate generated", title="Debate Paper")
    except asyncio.CancelledError:
        raise
    except _RECOVERABLE_ACTION_ERRORS as exc:
        if app._is_current_dataset_epoch(task_epoch):
            log_action_failure("paper debate generation", exc)
            app.notify("Paper debate failed", title="Debate Paper", severity="error")
    except Exception as exc:
        if app._is_current_dataset_epoch(task_epoch):
            log_action_failure("paper debate generation", exc, unexpected=True)
            app.notify("Paper debate failed", title="Debate Paper", severity="error")
    finally:
        app._paper_debate_active = False
        app._update_footer()


__all__ = [
    "_generate_paper_debate_async",
    "_start_paper_debate_flow",
    "action_debate_paper",
]
