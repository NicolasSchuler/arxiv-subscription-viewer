"""Header text formatting for the browse paper list."""

from __future__ import annotations

from typing import Any

from arxiv_browser._ascii import is_ascii_mode


def format_header_text(app: Any, query: str = "") -> str:
    """Format the left-pane header text for browser dataset modes."""
    sep = " - " if is_ascii_mode() else " \u00b7 "
    total = len(app.all_papers)
    filtered = len(app.filtered_papers)
    if getattr(app, "_in_arxiv_api_mode", False) and app._arxiv_search_state is not None:
        page = (app._arxiv_search_state.start // app._arxiv_search_state.max_results) + 1
        return f" [bold]Papers[/]{sep}API results{sep}page {page}"
    inbox_context = getattr(app, "_digest_inbox_context", None)
    if inbox_context is not None:
        return f" [bold]Papers[/]{sep}Inbox{sep}{inbox_context.source_label}{sep}{filtered}/{total}"
    if app.selected_ids:
        return f" [bold]Papers[/]{sep}{len(app.selected_ids)} selected"
    if getattr(app, "_watch_filter_active", False):
        return f" [bold]Papers[/]{sep}Watched {filtered}/{total}"
    if query:
        return f" [bold]Papers[/]{sep}Filtered ({filtered}/{total})"
    return f" [bold]Papers[/]{sep}Browse {total}"
