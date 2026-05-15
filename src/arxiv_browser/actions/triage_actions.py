"""Quick triage action handlers for ArxivBrowser."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arxiv_browser.modals.triage import (
    TRIAGE_LATER_TAG,
    QuickTriageCallbacks,
    QuickTriageItem,
    QuickTriageRequest,
    QuickTriageResult,
    QuickTriageScreen,
    format_quick_triage_summary,
)
from arxiv_browser.models import MAX_PAPERS_PER_COLLECTION, Paper

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser


def action_quick_triage(app: ArxivBrowser) -> None:
    """Launch quick triage for the current visible unread queue."""
    items = _quick_triage_items(app)
    if not items:
        app.notify(
            "No unread papers in the current view",
            title="Quick Triage",
            severity="warning",
        )
        return
    request = QuickTriageRequest(
        items=items,
        callbacks=QuickTriageCallbacks(
            mark_starred_read=lambda paper: _mark_starred_read(app, paper),
            mark_skipped=lambda paper: _mark_skipped(app, paper),
            tag_later=lambda paper: _tag_later(app, paper),
            save_to_collection=lambda paper, name: _save_to_collection(app, paper, name),
        ),
        collections=app._config.collections,
        papers_by_id=app._papers_by_id,
    )
    app.push_screen(QuickTriageScreen(request), lambda result: _finish_quick_triage(app, result))


def _quick_triage_items(app: ArxivBrowser) -> list[QuickTriageItem]:
    """Build the visible unread queue, preserving current order."""
    items: list[QuickTriageItem] = []
    for paper in app.filtered_papers:
        metadata = app._config.paper_metadata.get(paper.arxiv_id)
        if metadata is not None and metadata.is_read:
            continue
        items.append(
            QuickTriageItem(
                paper=paper,
                abstract_text=app._get_abstract_text(paper, allow_async=False) or "",
                relevance=app._relevance_scores.get(paper.arxiv_id),
                triage_prediction=getattr(app, "_triage_predictions", {}).get(paper.arxiv_id),
                watched=paper.arxiv_id in app._watched_paper_ids,
            )
        )
    return items


def _mark_starred_read(app: ArxivBrowser, paper: Paper) -> bool:
    """Mark a paper as read and starred."""
    metadata = app._get_or_create_metadata(paper.arxiv_id)
    if not metadata.is_read:
        app._record_read_velocity_events(1)
    metadata.is_read = True
    metadata.starred = True
    app._save_config_or_warn("quick triage")
    return True


def _mark_skipped(app: ArxivBrowser, paper: Paper) -> bool:
    """Mark a paper as read without changing star state."""
    metadata = app._get_or_create_metadata(paper.arxiv_id)
    if not metadata.is_read:
        app._record_read_velocity_events(1)
    metadata.is_read = True
    app._save_config_or_warn("quick triage")
    return True


def _tag_later(app: ArxivBrowser, paper: Paper) -> bool:
    """Add the triage-later tag and mark the paper read."""
    metadata = app._get_or_create_metadata(paper.arxiv_id)
    if not metadata.is_read:
        app._record_read_velocity_events(1)
    metadata.is_read = True
    if TRIAGE_LATER_TAG not in metadata.tags:
        metadata.tags.append(TRIAGE_LATER_TAG)
    app._save_config_or_warn("quick triage")
    return True


def _save_to_collection(app: ArxivBrowser, paper: Paper, collection_name: str) -> bool:
    """Save the paper to the selected collection and mark it read."""
    collection = next((col for col in app._config.collections if col.name == collection_name), None)
    if collection is None:
        app.notify(
            f"Collection '{collection_name}' no longer exists",
            title="Quick Triage",
            severity="warning",
        )
        return False

    if paper.arxiv_id not in collection.paper_ids:
        if len(collection.paper_ids) >= MAX_PAPERS_PER_COLLECTION:
            app.notify(
                f"Collection '{collection_name}' is full",
                title="Quick Triage",
                severity="warning",
            )
            return False
        collection.paper_ids.append(paper.arxiv_id)

    metadata = app._get_or_create_metadata(paper.arxiv_id)
    if not metadata.is_read:
        app._record_read_velocity_events(1)
    metadata.is_read = True
    app._save_config_or_warn("quick triage")
    return True


def _finish_quick_triage(
    app: ArxivBrowser,
    result: QuickTriageResult | None,
) -> None:
    """Refresh the main view and show the final or partial summary."""
    if result is None:
        return
    _refresh_after_quick_triage(app)
    app.notify(format_quick_triage_summary(result), title="Quick Triage")


def _refresh_after_quick_triage(app: ArxivBrowser) -> None:
    """Refresh list/detail state after decisions have changed metadata."""
    try:
        app._apply_filter(app._get_active_query())
    except (AttributeError, RuntimeError):
        app._refresh_list_view()
        app._refresh_detail_pane()
        app._update_header()


__all__ = [
    "TRIAGE_LATER_TAG",
    "_finish_quick_triage",
    "_quick_triage_items",
    "action_quick_triage",
]
