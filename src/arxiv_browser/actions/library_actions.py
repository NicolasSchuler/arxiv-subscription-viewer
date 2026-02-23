# ruff: noqa: F403, F405, UP037
# pyright: reportUndefinedVariable=false, reportAttributeAccessIssue=false
"""Extracted ArxivBrowser action handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arxiv_browser.actions._runtime import *

if TYPE_CHECKING:
    from arxiv_browser.app import ArxivBrowser


def _sync_app_globals() -> None:
    """Sync patched globals from arxiv_browser.app without importing it."""
    sync_app_globals(globals())


def action_cursor_down(app: "ArxivBrowser") -> None:
    """Move cursor down (vim-style j key)."""
    _sync_app_globals()
    list_view = app._get_paper_list_widget()
    list_view.action_cursor_down()


def action_cursor_up(app: "ArxivBrowser") -> None:
    """Move cursor up (vim-style k key)."""
    _sync_app_globals()
    list_view = app._get_paper_list_widget()
    list_view.action_cursor_up()


def action_toggle_select(app: "ArxivBrowser") -> None:
    """Toggle selection of the currently highlighted paper."""
    _sync_app_globals()
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
    _sync_app_globals()
    for paper in app.filtered_papers:
        app.selected_ids.add(paper.arxiv_id)
    for i in range(len(app.filtered_papers)):
        app._update_option_at_index(i)
    app._update_header()


def action_clear_selection(app: "ArxivBrowser") -> None:
    """Clear all selections."""
    _sync_app_globals()
    app.selected_ids.clear()
    for i in range(len(app.filtered_papers)):
        app._update_option_at_index(i)
    app._update_header()


def action_cycle_sort(app: "ArxivBrowser") -> None:
    """Cycle through sort options: title, date, arxiv_id."""
    _sync_app_globals()
    app._sort_index = (app._sort_index + 1) % len(SORT_OPTIONS)
    sort_key = SORT_OPTIONS[app._sort_index]
    app.notify(f"Sorted by {sort_key}", title="Sort")

    # Re-sort and refresh the list
    app._sort_papers()
    app._refresh_list_view()
    app._update_header()


def action_toggle_read(app: "ArxivBrowser") -> None:
    """Toggle read status of highlighted paper, or bulk toggle for selected papers."""
    _sync_app_globals()
    if app.selected_ids:
        app._bulk_toggle_bool("is_read", "marked read", "marked unread", "Read Status")
        return

    paper = app._get_current_paper()
    if not paper:
        return
    metadata = app._get_or_create_metadata(paper.arxiv_id)
    metadata.is_read = not metadata.is_read
    idx = app._get_current_index()
    if idx is not None:
        app._update_option_at_index(idx)
    status = "read" if metadata.is_read else "unread"
    app.notify(f"Marked as {status}", title="Read Status")


def action_toggle_star(app: "ArxivBrowser") -> None:
    """Toggle star status of highlighted paper, or bulk toggle for selected papers."""
    _sync_app_globals()
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
    _sync_app_globals()
    paper = app._get_current_paper()
    if not paper:
        return

    arxiv_id = paper.arxiv_id
    current_notes = ""
    if arxiv_id in app._config.paper_metadata:
        current_notes = app._config.paper_metadata[arxiv_id].notes

    def on_notes_saved(notes: str | None) -> None:
        if notes is None:
            return
        metadata = app._get_or_create_metadata(arxiv_id)
        metadata.notes = notes
        # Update the option display if still on the same paper
        cur = app._get_current_paper()
        if cur and cur.arxiv_id == arxiv_id:
            idx = app._get_current_index()
            if idx is not None:
                app._update_option_at_index(idx)
        app.notify("Notes saved", title="Notes")

    app.push_screen(NotesModal(arxiv_id, current_notes), on_notes_saved)


def action_edit_tags(app: "ArxivBrowser") -> None:
    """Open tags editor for the current paper, or bulk-tag selected papers."""
    _sync_app_globals()
    if app.selected_ids:
        app._bulk_edit_tags()
        return

    paper = app._get_current_paper()
    if not paper:
        return

    arxiv_id = paper.arxiv_id
    current_tags: list[str] = []
    if arxiv_id in app._config.paper_metadata:
        current_tags = app._config.paper_metadata[arxiv_id].tags.copy()

    # Collect all unique tags across all paper metadata for suggestions
    all_tags = app._collect_all_tags()

    def on_tags_saved(tags: list[str] | None) -> None:
        if tags is None:
            return
        metadata = app._get_or_create_metadata(arxiv_id)
        metadata.tags = tags
        # Update the option display if still on the same paper
        cur = app._get_current_paper()
        if cur and cur.arxiv_id == arxiv_id:
            idx = app._get_current_index()
            if idx is not None:
                app._update_option_at_index(idx)
        app.notify(f"Tags: {', '.join(tags) if tags else 'none'}", title="Tags")

    app.push_screen(TagsModal(arxiv_id, current_tags, all_tags=all_tags), on_tags_saved)


def action_toggle_watch_filter(app: "ArxivBrowser") -> None:
    """Toggle filtering to show only watched papers."""
    _sync_app_globals()
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
    _sync_app_globals()

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
