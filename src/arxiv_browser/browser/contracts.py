"""Shared browser UI contracts and compatibility aliases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from arxiv_browser.models import WatchListEntry
from arxiv_browser.widgets import chrome as _widget_chrome


class TaskTrackingApp(Protocol):
    """Protocol for the subset of ArxivBrowser used by modal screens."""

    def _track_task(self, coro: Any, *, dataset_bound: bool = False) -> asyncio.Task[None]: ...


# Context-sensitive footer keybinding hints (compatibility alias kept for tests/imports).
FOOTER_CONTEXTS: dict[str, list[tuple[str, str]]] = {
    "default": _widget_chrome.build_browse_footer_bindings(
        s2_active=False,
        has_starred=False,
        llm_configured=False,
        has_history_navigation=False,
    ),
    "selection": _widget_chrome.build_selection_footer_base_bindings(),
    "search": _widget_chrome.build_search_footer_bindings(),
    "api": _widget_chrome.build_api_footer_bindings(),
}


@dataclass(slots=True, frozen=True)
class _PaletteAppState:
    """Minimal app state snapshot for command-palette availability decisions."""

    in_arxiv_api_mode: bool
    hf_active: bool
    watch_filter_active: bool
    show_abstract_preview: bool
    detail_mode: str
    active_query: str
    has_history_navigation: bool
    watch_list: list[WatchListEntry]
    has_marks: bool
    has_starred: bool
    llm_configured: bool
    has_visible_papers: bool
    has_selection: bool
    has_current_paper: bool
    has_target_papers: bool
    s2_active: bool
    s2_data_loaded: bool


# Command palette registry: (name, description, key_hint, action_name)
# action_name maps to ArxivBrowser.action_* methods (or "" for non-action commands)
COMMAND_PALETTE_COMMANDS: list[tuple[str, str, str, str]] = [
    ("Search Papers", "Filter papers by text, category, or tag", "/", "toggle_search"),
    ("Search Syntax", "Open search examples and operators", "", "show_search_syntax"),
    ("Search arXiv API", "Search all of arXiv online", "A", "arxiv_search"),
    ("Previous Date", "Navigate to older date file", "[", "prev_date"),
    ("Next Date", "Navigate to newer date file", "]", "next_date"),
    ("Open in Browser", "Open selected paper(s) in web browser", "o", "open_url"),
    ("Open PDF", "Open selected paper(s) as PDF", "P", "open_pdf"),
    ("Download PDF", "Download PDF(s) to local folder", "d", "download_pdf"),
    ("Copy to Clipboard", "Copy paper info to clipboard", "c", "copy_selected"),
    ("Toggle Read", "Mark paper(s) as read/unread", "r", "toggle_read"),
    (
        "Mark Visible Read",
        "Mark all visible (filtered) papers as read",
        "Ctrl+r",
        "mark_visible_read",
    ),
    ("Toggle Star", "Star/unstar paper(s)", "x", "toggle_star"),
    ("Edit Notes", "Add or edit notes for current paper", "n", "edit_notes"),
    ("Edit Tags", "Add or edit tags (bulk when selected)", "t", "edit_tags"),
    ("Select All", "Select all visible papers", "a", "select_all"),
    ("Clear Selection", "Deselect all papers", "u", "clear_selection"),
    ("Toggle Selection", "Toggle selection on current paper", "Space", "toggle_select"),
    (
        "Cycle Sort",
        "Cycle sort: title/date/arxiv_id/citations/trending/relevance",
        "s",
        "cycle_sort",
    ),
    (
        "Show Watched Papers",
        "Filter the list to papers matching your watch list",
        "w",
        "toggle_watch_filter",
    ),
    ("Manage Watch List", "Add/remove watch list patterns", "W", "manage_watch_list"),
    ("Show Abstract Preview", "Reveal abstract snippets in the paper list", "p", "toggle_preview"),
    ("Export Menu", "Export as BibTeX, Markdown, RIS, or CSV", "E", "export_menu"),
    ("Export Metadata", "Export all annotations to a JSON snapshot", "", "export_metadata"),
    ("Import Metadata", "Import annotations from a chosen JSON snapshot", "", "import_metadata"),
    (
        "Fetch S2 Data",
        "Fetch Semantic Scholar data for the current paper (requires S2 enabled)",
        "e",
        "fetch_s2",
    ),
    (
        "Toggle Semantic Scholar",
        "Enable or disable Semantic Scholar enrichment",
        "Ctrl+e",
        "ctrl_e_dispatch",
    ),
    (
        "Enable HuggingFace Trending",
        "Show HuggingFace badges and detail-pane matches",
        "Ctrl+h",
        "toggle_hf",
    ),
    ("Check Versions", "Check starred papers for arXiv updates", "V", "check_versions"),
    (
        "Citation Graph",
        "Explore the citation graph for the current paper (requires S2 data)",
        "G",
        "citation_graph",
    ),
    (
        "AI Summary",
        "Generate an LLM-powered paper summary (requires LLM configuration)",
        "Ctrl+s",
        "generate_summary",
    ),
    (
        "Chat with Paper",
        "Interactive Q&A about the current paper (requires LLM configuration)",
        "C",
        "chat_with_paper",
    ),
    (
        "Score Relevance",
        "LLM-score loaded papers by research interests (requires LLM configuration)",
        "L",
        "score_relevance",
    ),
    ("Edit Interests", "Edit research interests for relevance scoring", "Ctrl+l", "edit_interests"),
    (
        "Auto-Tag",
        "Suggest tags for current or selected papers (requires LLM configuration)",
        "Ctrl+g",
        "auto_tag",
    ),
    (
        "Similar Papers",
        "Find similar papers locally or via Semantic Scholar when available",
        "R",
        "show_similar",
    ),
    ("Add Bookmark", "Save current search as bookmark", "Ctrl+b", "add_bookmark"),
    ("Collections", "Manage paper reading lists", "Ctrl+k", "collections"),
    ("Add to Collection", "Add papers to a reading list", "", "add_to_collection"),
    (
        "Toggle Detail Density",
        "Switch between scan and full detail views",
        "v",
        "toggle_detail_mode",
    ),
    ("Cycle Theme", "Switch between Monokai/Catppuccin/Solarized", "Ctrl+t", "cycle_theme"),
    ("Toggle Sections", "Show/hide detail pane sections", "Ctrl+d", "toggle_sections"),
    ("Help", "Show all keyboard shortcuts", "?", "show_help"),
    ("Set Mark", "Set a named mark (a-z) at current position", "m", "start_mark"),
    ("Jump to Mark", "Jump to a named mark (a-z)", "'", "start_goto_mark"),
]


__all__ = ["COMMAND_PALETTE_COMMANDS", "FOOTER_CONTEXTS", "TaskTrackingApp", "_PaletteAppState"]
