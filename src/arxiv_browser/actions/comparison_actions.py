"""Paper comparison action handlers for ArxivBrowser."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure
from arxiv_browser.io_actions import resolve_target_papers
from arxiv_browser.llm import PAPER_COMPARISON_CONTENT_MAX_CHARS
from arxiv_browser.modals import PaperComparisonScreen
from arxiv_browser.models import Paper

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser
    from arxiv_browser.llm_providers import LLMProvider


_NOTIFY_TIMEOUT_DEFAULT = 8
_NOTIFY_MAX_LENGTH = 200
_RECOVERABLE_ACTION_ERRORS = RECOVERABLE_ACTION_ERRORS


def action_compare_papers(app: ArxivBrowser) -> None:
    """Open a local-first side-by-side comparison for 2-3 selected papers."""
    selected_count = len(app.selected_ids)
    if selected_count not in {2, 3}:
        app.notify("Select 2 or 3 papers to compare", title="Paper Comparison", severity="warning")
        return

    papers = _selected_papers_for_comparison(app)
    if len(papers) != selected_count or len(papers) not in {2, 3}:
        app.notify(
            "Selected papers are no longer available",
            title="Paper Comparison",
            severity="warning",
        )
        return

    abstracts = {
        paper.arxiv_id: app._get_abstract_text(paper, allow_async=False) or "" for paper in papers
    }
    app.push_screen(
        PaperComparisonScreen(
            papers,
            abstracts,
            lambda screen: _request_paper_comparison_ai(app, screen, papers),
        )
    )


def _selected_papers_for_comparison(app: ArxivBrowser) -> list[Paper]:
    """Return selected papers in visible order, then hidden selected IDs alphabetically."""
    return resolve_target_papers(
        filtered_papers=app.filtered_papers,
        selected_ids=app.selected_ids,
        papers_by_id=app._papers_by_id,
        current_paper=None,
    )


def _request_paper_comparison_ai(
    app: ArxivBrowser,
    screen: PaperComparisonScreen,
    papers: list[Paper],
) -> None:
    """Validate LLM configuration/trust and start AI comparison when allowed."""
    command_template = app._require_llm_command()
    if not command_template:
        screen.set_ai_idle("Configure an LLM command to generate an AI comparison.")
        return

    def _start() -> None:
        _start_paper_comparison_ai(app, screen, papers)

    if not app._ensure_llm_command_trusted(command_template, _start):
        return
    _start()


def _start_paper_comparison_ai(
    app: ArxivBrowser,
    screen: PaperComparisonScreen,
    papers: list[Paper],
) -> None:
    """Schedule AI comparison generation for an open comparison screen."""
    if screen.ai_running:
        app.notify("AI comparison already generating", title="Paper Comparison")
        return
    provider = app._llm_provider
    if provider is None:
        screen.set_ai_error("LLM provider unavailable")
        app.notify("LLM provider unavailable", title="Paper Comparison", severity="error")
        return
    screen.set_ai_loading()
    app._track_dataset_task(_generate_paper_comparison_async(app, screen, papers, provider))


def _comparison_screen_is_live(screen: PaperComparisonScreen) -> bool:
    return bool(getattr(screen, "is_mounted", True))


async def _generate_paper_comparison_async(
    app: ArxivBrowser,
    screen: PaperComparisonScreen,
    papers: list[Paper],
    provider: LLMProvider,
) -> None:
    """Background task: ask the LLM for a comparison and update the open modal."""
    task_epoch = app._capture_dataset_epoch()
    try:
        result, error = await app._get_services().llm.compare_papers(
            papers=papers,
            provider=provider,
            timeout_seconds=app._config.llm_timeout,
            fetch_paper_content=app._fetch_paper_content_async,
            max_content_chars=PAPER_COMPARISON_CONTENT_MAX_CHARS,
        )
        if not app._is_current_dataset_epoch(task_epoch) or not _comparison_screen_is_live(screen):
            return
        if result is None:
            message = (error or "LLM command failed")[:_NOTIFY_MAX_LENGTH]
            screen.set_ai_error(message)
            app.notify(
                message,
                title="Paper Comparison",
                severity="error",
                timeout=_NOTIFY_TIMEOUT_DEFAULT,
            )
            return
        screen.set_ai_result(result)
        app.notify("AI comparison generated", title="Paper Comparison")
    except asyncio.CancelledError:
        raise
    except _RECOVERABLE_ACTION_ERRORS as exc:
        if app._is_current_dataset_epoch(task_epoch) and _comparison_screen_is_live(screen):
            log_action_failure("paper comparison generation", exc)
            screen.set_ai_error("Paper comparison failed")
            app.notify("Paper comparison failed", title="Paper Comparison", severity="error")
    except Exception as exc:
        if app._is_current_dataset_epoch(task_epoch) and _comparison_screen_is_live(screen):
            log_action_failure("paper comparison generation", exc, unexpected=True)
            screen.set_ai_error("Paper comparison failed")
            app.notify("Paper comparison failed", title="Paper Comparison", severity="error")


__all__ = [
    "_generate_paper_comparison_async",
    "_request_paper_comparison_ai",
    "_selected_papers_for_comparison",
    "_start_paper_comparison_ai",
    "action_compare_papers",
]
