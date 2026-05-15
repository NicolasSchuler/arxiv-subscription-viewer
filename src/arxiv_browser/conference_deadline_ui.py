"""UI/runtime helpers for imported conference deadlines."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from arxiv_browser.action_messages import (
    build_actionable_error,
    build_actionable_success,
    build_actionable_warning,
)
from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure
from arxiv_browser.conference_deadlines import SubmissionTarget, match_paper_to_deadlines
from arxiv_browser.config import save_config

if TYPE_CHECKING:
    from arxiv_browser.models import Paper, PaperMetadata
    from arxiv_browser.services.enrichment_service import ConferenceDeadlinesFetchResult


_RECOVERABLE_ACTION_ERRORS = RECOVERABLE_ACTION_ERRORS
_ENRICHMENT_FETCH_ERRORS = (httpx.HTTPError, OSError, RuntimeError, ValueError, TypeError)


def init_conference_deadline_state(app: Any) -> None:
    """Initialize deadline tracker state attached to the browser app."""
    app._conference_deadlines_active = False
    app._conference_deadlines = []
    app._conference_deadlines_loading = False
    app._conference_deadlines_db_path = app._cache_db_path
    app._conference_deadlines_api_error = False
    app._deadline_countdown_task = None


def start_conference_deadlines_if_enabled(app: Any) -> None:
    """Start deadline import/countdowns when enabled in config."""
    app._conference_deadlines_active = app._config.conference_deadlines_enabled
    if app._conference_deadlines_active:
        ensure_deadline_countdown_timer(app)
        app._track_dataset_task(fetch_conference_deadlines(app))


def build_detail_submission_targets(
    app: Any,
    paper: Paper,
    metadata: PaperMetadata | None,
) -> tuple[tuple[SubmissionTarget, ...], str]:
    """Return current submission targets and a minute-granularity cache key."""
    deadline_now = datetime.now(UTC)
    if not getattr(app, "_conference_deadlines_active", False):
        return (), deadline_now.strftime("%Y%m%d%H%M")
    targets = match_paper_to_deadlines(
        paper,
        getattr(app, "_conference_deadlines", []),
        metadata,
        deadline_now,
    )
    return tuple(targets), deadline_now.strftime("%Y%m%d%H%M")


async def action_refresh_conference_deadlines(app: Any) -> None:
    """Enable and refresh imported conference deadline data."""
    if not app._conference_deadlines_active:
        app._conference_deadlines_active = True
        app._config.conference_deadlines_enabled = True
        if not save_config(app._config):
            app._conference_deadlines_active = False
            app._config.conference_deadlines_enabled = False
            app.notify(
                "Failed to save conference deadline setting",
                title="Deadlines",
                severity="error",
            )
            return
    ensure_deadline_countdown_timer(app)
    await fetch_conference_deadlines(app)


async def fetch_conference_deadlines(app: Any) -> None:
    """Fetch or load conference deadline data through the service layer."""
    if app._conference_deadlines_loading:
        app.notify("Conference deadlines are already refreshing", title="Deadlines")
        return
    app._conference_deadlines_loading = True
    app._conference_deadlines_api_error = False
    app._update_status_bar()
    app.notify("Refreshing conference deadlines...", title="Deadlines")
    try:
        app._track_dataset_task(fetch_conference_deadlines_async(app))
    except _RECOVERABLE_ACTION_ERRORS as exc:
        app._conference_deadlines_loading = False
        app._conference_deadlines_api_error = True
        app._update_status_bar()
        log_action_failure("conference deadline refresh scheduling", exc)
        app.notify(
            build_actionable_error(
                "refresh conference deadlines",
                why="local cache lookup failed",
                next_step="retry in a moment or disable the feature in config",
            ),
            title="Deadlines",
            severity="error",
        )
    except Exception as exc:
        app._conference_deadlines_loading = False
        app._update_status_bar()
        log_action_failure("conference deadline refresh scheduling", exc, unexpected=True)
        raise


async def fetch_conference_deadlines_async(app: Any) -> None:
    """Background task: import AI Deadlines data and refresh detail matches."""
    task_epoch = app._capture_dataset_epoch()
    try:
        client = app._http_client
        if client is None:
            app._conference_deadlines_api_error = True
            return

        result = await app._get_services().enrichment.load_or_fetch_conference_deadlines(
            db_path=app._conference_deadlines_db_path,
            cache_ttl_hours=app._config.conference_deadlines_cache_ttl_hours,
            client=client,
            source_url=app._config.conference_deadlines_source_url,
        )
        if not app._is_current_dataset_epoch(task_epoch):
            return
        apply_conference_deadlines_result(app, result)
    except asyncio.CancelledError:
        raise
    except _ENRICHMENT_FETCH_ERRORS as exc:
        handle_conference_deadlines_exception(app, task_epoch, exc)
    except Exception as exc:
        handle_conference_deadlines_exception(app, task_epoch, exc, unexpected=True)
    finally:
        finish_conference_deadlines_fetch(app, task_epoch)


async def _deadline_countdown_loop(app: Any) -> None:
    while getattr(app, "_conference_deadlines_active", False):
        await asyncio.sleep(60)
        app._refresh_detail_pane()


def ensure_deadline_countdown_timer(app: Any) -> None:
    """Refresh the detail pane once per minute while deadline countdowns are active."""
    task = getattr(app, "_deadline_countdown_task", None)
    if isinstance(task, asyncio.Task) and not task.done():
        return
    app._deadline_countdown_task = app._track_dataset_task(_deadline_countdown_loop(app))


def apply_conference_deadlines_result(app: Any, result: ConferenceDeadlinesFetchResult) -> None:
    if not result.complete or result.state == "unavailable":
        notify_conference_deadlines_error(app)
        return
    app._conference_deadlines = result.deadlines
    app._conference_deadlines_api_error = False
    ensure_deadline_countdown_timer(app)
    app._refresh_detail_pane()
    if result.state == "empty":
        if not result.from_cache:
            app.notify(
                build_actionable_warning(
                    "No conference deadlines were returned",
                    next_step="check the configured AI Deadlines source URL",
                ),
                title="Deadlines",
                severity="warning",
            )
        return
    if not result.from_cache:
        app.notify(
            build_actionable_success(
                "Conference deadlines loaded",
                detail=f"{len(result.deadlines)} deadline{'s' if len(result.deadlines) != 1 else ''} imported",
                next_step="matching future targets now appear in paper details",
            ),
            title="Deadlines",
        )


def notify_conference_deadlines_error(app: Any) -> None:
    app._conference_deadlines_api_error = True
    app.notify(
        build_actionable_error(
            "refresh conference deadlines",
            why="the AI Deadlines source could not be fetched or parsed",
            next_step="retry later or check conference_deadlines_source_url",
        ),
        title="Deadlines",
        severity="error",
    )


def handle_conference_deadlines_exception(
    app: Any,
    task_epoch: int,
    exc: Exception,
    *,
    unexpected: bool = False,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    log_action_failure("conference deadline refresh", exc, unexpected=unexpected)
    notify_conference_deadlines_error(app)


def finish_conference_deadlines_fetch(app: Any, task_epoch: int) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app._conference_deadlines_loading = False
    app._update_status_bar()
