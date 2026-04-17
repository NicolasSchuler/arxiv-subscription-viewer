"""Shared empty-state copy for the paper list and secondary views."""

from __future__ import annotations


def build_list_empty_message(
    *,
    query: str,
    in_arxiv_api_mode: bool,
    watch_filter_active: bool,
    history_mode: bool,
) -> str:
    """Build actionable empty-state copy for the paper list."""
    if query:
        return (
            "[dim italic]No papers match your search.[/]\n"
            "[dim]Try: edit the query or press [bold]Esc[/bold] to clear search.[/]\n"
            "[dim]Next: press [bold]?[/bold] for shortcuts or [bold]Ctrl+p[/bold] for commands.[/]"
        )
    if in_arxiv_api_mode:
        return (
            "[dim italic]No API results on this page.[/]\n"
            "[dim]Try: [bold]][/bold] next page, [bold][[/bold] previous page, "
            "or [bold]A[/bold] for a new query.[/]\n"
            "[dim]Next: press [bold]Esc[/bold] or [bold]Ctrl+e[/bold] to exit API mode.[/]"
        )
    if watch_filter_active:
        return (
            "[dim italic]No watched papers found.[/]\n"
            "[dim]Try: press [bold]w[/bold] to show all papers.[/]\n"
            "[dim]Next: press [bold]W[/bold] to manage watch list patterns.[/]"
        )
    if history_mode:
        return (
            "[dim italic]No papers available for this date.[/]\n"
            "[dim]Try: press [bold][[/bold] or [bold]][/bold] to change dates.[/]\n"
            "[dim]Next: press [bold]A[/bold] to search arXiv.[/]"
        )
    return (
        "[dim italic]No papers available.[/]\n"
        "[dim]Try: press [bold]A[/bold] to search arXiv.[/]\n"
        "[dim]Next: load a history file or run with [bold]-i[/bold] <file>.[/]"
    )


# ── Secondary views ──────────────────────────────────────────────────────────
#
# These helpers centralize empty-state copy for modal list views so the same
# phrasing and Try/Next cadence used by the main paper list flows through the
# rest of the UI. Each returns a single plain-text line suitable for a
# disabled ``ListItem`` placeholder.


COLLECTIONS_MANAGE_EMPTY = "No collections yet. Try: type a name above and press Create."
COLLECTIONS_PICK_EMPTY = (
    "No collections yet. Try: press Ctrl+k in the main view to create one first."
)
COLLECTION_DETAIL_EMPTY = (
    "No papers in this collection yet. "
    "Try: select a paper in the main view and press Ctrl+k to add it."
)
CITATIONS_REFS_EMPTY = (
    "No references recorded for this paper. Try: press e to refresh Semantic Scholar data."
)
CITATIONS_CITES_EMPTY = (
    "No citing papers recorded yet. Try: press e to refresh Semantic Scholar data."
)


__all__ = [
    "CITATIONS_CITES_EMPTY",
    "CITATIONS_REFS_EMPTY",
    "COLLECTIONS_MANAGE_EMPTY",
    "COLLECTIONS_PICK_EMPTY",
    "COLLECTION_DETAIL_EMPTY",
    "build_list_empty_message",
]
