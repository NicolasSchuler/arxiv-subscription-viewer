# ruff: noqa: UP037
"""Library and organisation action handlers for ArxivBrowser.

Covers: list navigation (cursor up/down), multi-paper selection (select,
select-all, clear), sort-order cycling, read/star toggles, notes and tags
editing, and watch-list management.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

from arxiv_browser.config import save_config
from arxiv_browser.modals import PaperEditModal, WatchListModal
from arxiv_browser.modals.editing import PaperEditResult
from arxiv_browser.models import SORT_OPTIONS, PaperMetadata, WatchListEntry
from arxiv_browser.review import clear_review, mark_reviewed, schedule_review

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser


def action_cursor_down(app: "ArxivBrowser") -> None:
    """Move cursor down (vim-style j key)."""
    if app._move_detail_line_cursor(1):
        return
    list_view = app._get_paper_list_widget()
    list_view.action_cursor_down()


def action_cursor_up(app: "ArxivBrowser") -> None:
    """Move cursor up (vim-style k key)."""
    if app._move_detail_line_cursor(-1):
        return
    list_view = app._get_paper_list_widget()
    list_view.action_cursor_up()


def action_toggle_select(app: "ArxivBrowser") -> None:
    """Toggle selection of the currently highlighted paper."""
    paper = app._get_current_paper()
    if not paper:
        return
    aid = paper.arxiv_id
    if aid in app.selected_ids:
        app.selected_ids.discard(aid)
    else:
        app.selected_ids.add(aid)
    idx = app._get_current_index()
    if idx is not None:
        app._update_option_at_index(idx)
    app._update_header()


def action_select_all(app: "ArxivBrowser") -> None:
    """Select all currently visible papers."""
    if app._open_line_annotation_modal():
        return
    for paper in app.filtered_papers:
        app.selected_ids.add(paper.arxiv_id)
    for i in range(len(app.filtered_papers)):
        app._update_option_at_index(i)
    app._update_header()


def action_clear_selection(app: "ArxivBrowser") -> None:
    """Clear all selections."""
    app.selected_ids.clear()
    for i in range(len(app.filtered_papers)):
        app._update_option_at_index(i)
    app._update_header()


def action_cycle_sort(app: "ArxivBrowser") -> None:
    """Cycle through the configured paper sort options."""
    app._sort_index = (app._sort_index + 1) % len(SORT_OPTIONS)
    sort_key = SORT_OPTIONS[app._sort_index]
    app.notify(f"Sorted by {sort_key}", title="Sort")

    # Re-sort and refresh the list
    app._sort_papers()
    app._refresh_list_view()
    app._update_header()


def action_toggle_read(app: "ArxivBrowser") -> None:
    """Toggle read status of highlighted paper, or bulk toggle for selected papers."""
    if app.selected_ids:
        app._bulk_toggle_bool("is_read", "marked read", "marked unread", "Read Status")
        return

    paper = app._get_current_paper()
    if not paper:
        return
    metadata = app._get_or_create_metadata(paper.arxiv_id)
    metadata.is_read = not metadata.is_read
    if metadata.is_read:
        app._record_read_velocity_events(1)
    idx = app._get_current_index()
    if idx is not None:
        app._update_option_at_index(idx)
    status = "read" if metadata.is_read else "unread"
    app.notify(f"Marked as {status}", title="Read Status")


def action_toggle_star(app: "ArxivBrowser") -> None:
    """Toggle star status of highlighted paper, or bulk toggle for selected papers."""
    if app.selected_ids:
        app._bulk_toggle_bool("starred", "starred", "unstarred", "Star")
        return

    paper = app._get_current_paper()
    if not paper:
        return
    metadata = app._get_or_create_metadata(paper.arxiv_id)
    metadata.starred = not metadata.starred
    idx = app._get_current_index()
    if idx is not None:
        app._update_option_at_index(idx)
    status = "starred" if metadata.starred else "unstarred"
    app.notify(f"Paper {status}", title="Star")


def action_edit_notes(app: "ArxivBrowser") -> None:
    """Open notes editor for the currently highlighted paper."""
    paper = app._get_current_paper()
    if not paper:
        return

    arxiv_id = paper.arxiv_id
    current_notes = ""
    current_tags: list[str] = []
    if arxiv_id in app._config.paper_metadata:
        current_notes = app._config.paper_metadata[arxiv_id].notes
        current_tags = app._config.paper_metadata[arxiv_id].tags.copy()
    all_tags = app._collect_all_tags()

    def on_edit_result(result: PaperEditResult | None) -> None:
        if result is None:
            return
        metadata = app._get_or_create_metadata(arxiv_id)
        metadata.notes = result.notes
        metadata.tags = result.tags
        cur = app._get_current_paper()
        if cur and cur.arxiv_id == arxiv_id:
            idx = app._get_current_index()
            if idx is not None:
                app._update_option_at_index(idx)
        app.notify("Saved", title="Edit")

    app.push_screen(
        PaperEditModal(
            arxiv_id,
            current_notes=current_notes,
            current_tags=current_tags,
            all_tags=all_tags,
            initial_tab="notes",
        ),
        on_edit_result,
    )


def action_edit_tags(app: "ArxivBrowser") -> None:
    """Open tags editor for the current paper, or bulk-tag selected papers."""
    if app.selected_ids:
        app._bulk_edit_tags()
        return

    paper = app._get_current_paper()
    if not paper:
        return

    arxiv_id = paper.arxiv_id
    current_notes = ""
    current_tags: list[str] = []
    if arxiv_id in app._config.paper_metadata:
        current_notes = app._config.paper_metadata[arxiv_id].notes
        current_tags = app._config.paper_metadata[arxiv_id].tags.copy()

    all_tags = app._collect_all_tags()

    def on_edit_result(result: PaperEditResult | None) -> None:
        if result is None:
            return
        metadata = app._get_or_create_metadata(arxiv_id)
        metadata.notes = result.notes
        metadata.tags = result.tags
        cur = app._get_current_paper()
        if cur and cur.arxiv_id == arxiv_id:
            idx = app._get_current_index()
            if idx is not None:
                app._update_option_at_index(idx)
        tag_desc = ", ".join(result.tags) if result.tags else "none"
        app.notify(f"Tags: {tag_desc}", title="Saved")

    app.push_screen(
        PaperEditModal(
            arxiv_id,
            current_notes=current_notes,
            current_tags=current_tags,
            all_tags=all_tags,
            initial_tab="tags",
        ),
        on_edit_result,
    )


def action_toggle_watch_filter(app: "ArxivBrowser") -> None:
    """Toggle filtering to show only watched papers."""
    app._watch_filter_active = not app._watch_filter_active

    if app._watch_filter_active:
        if not app._watched_paper_ids:
            app.notify("Watch list is empty", title="Watch", severity="warning")
            app._watch_filter_active = False
            return
        app.notify("Showing watched papers", title="Watch")
    else:
        app.notify("Showing all papers", title="Watch")

    # Re-apply current filter with watch list consideration
    query = app._get_search_input_widget().value.strip()
    app._apply_filter(query)


def action_manage_watch_list(app: "ArxivBrowser") -> None:
    """Open the watch list manager."""

    def on_watch_list_updated(entries: list[WatchListEntry] | None) -> None:
        if entries is None:
            return
        old_entries = list(app._config.watch_list)
        app._config.watch_list = entries
        if not save_config(app._config):
            app._config.watch_list = old_entries
            app.notify(
                "Failed to save watch list",
                title="Watch",
                severity="error",
            )
            return
        app._compute_watched_papers()
        if app._watch_filter_active and not app._watched_paper_ids:
            app._watch_filter_active = False
        query = app._get_search_input_widget().value.strip()
        app._apply_filter(query)
        app.notify("Watch list updated", title="Watch")

    app.push_screen(WatchListModal(app._config.watch_list), on_watch_list_updated)


def action_schedule_review(app: "ArxivBrowser") -> None:
    """Schedule the current or selected papers for spaced review."""
    _apply_review_metadata_action(
        app,
        schedule_review,
        singular="Scheduled review for 1 paper",
        plural="Scheduled reviews for {count} papers",
    )


def action_mark_reviewed(app: "ArxivBrowser") -> None:
    """Advance the current or selected papers in the spaced-review schedule."""
    _apply_review_metadata_action(
        app,
        mark_reviewed,
        singular="Advanced review schedule for 1 paper",
        plural="Advanced review schedules for {count} papers",
    )


def action_clear_review(app: "ArxivBrowser") -> None:
    """Remove the current or selected papers from the spaced-review queue."""
    _apply_review_metadata_action(
        app,
        lambda metadata, _today: clear_review(metadata),
        singular="Cleared review schedule for 1 paper",
        plural="Cleared review schedules for {count} papers",
    )


def action_show_due_reviews(app: "ArxivBrowser") -> None:
    """Apply the review-due virtual query filter."""
    query = "review-due"
    search_input = app._get_search_input_widget()
    search_input.value = query
    app._apply_filter(query)
    app.notify("Showing due reviews", title="Review")


def action_mark_visible_read(app: "ArxivBrowser") -> None:
    """Mark all currently visible (filtered) papers as read."""
    changed = 0
    for paper in app.filtered_papers:
        meta = app._get_or_create_metadata(paper.arxiv_id)
        if not meta.is_read:
            meta.is_read = True
            changed += 1
    if changed:
        app._record_read_velocity_events(changed)
        app._save_config_or_warn("mark visible read")
        app._mark_badges_dirty("read", immediate=True)
        app._refresh_list_view()
        app._refresh_detail_pane()
    app.notify(
        f"Marked {changed} paper{'s' if changed != 1 else ''} as read",
        title="Read Status",
    )


def _apply_review_metadata_action(
    app: "ArxivBrowser",
    update: Callable[[PaperMetadata, date], None],
    *,
    singular: str,
    plural: str,
) -> None:
    target_ids = _review_target_ids(app)
    if not target_ids:
        app.notify("Select a paper first", title="Review", severity="warning")
        return
    today = date.today()
    for arxiv_id in target_ids:
        update(app._get_or_create_metadata(arxiv_id), today)
    app._save_config_or_warn("review schedule")
    query = app._get_live_query()
    app._apply_filter(query)
    app._refresh_detail_pane()
    count = len(target_ids)
    message = singular if count == 1 else plural.format(count=count)
    app.notify(message, title="Review")


def _review_target_ids(app: "ArxivBrowser") -> set[str]:
    if app.selected_ids:
        return set(app.selected_ids)
    paper = app._get_current_paper()
    return {paper.arxiv_id} if paper else set()
