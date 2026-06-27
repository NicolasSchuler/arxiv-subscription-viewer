"""Shared browser UI contracts and compatibility aliases."""

from __future__ import annotations

from dataclasses import dataclass

from arxiv_browser.app_protocols import TaskTrackingApp
from arxiv_browser.models import WatchListEntry
from arxiv_browser.widgets import chrome as _widget_chrome

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
    compact_list: bool
    detail_mode: str
    active_query: str
    has_history_files: bool
    has_history_navigation: bool
    watch_list: list[WatchListEntry]
    has_marks: bool
    has_starred: bool
    llm_configured: bool
    has_visible_papers: bool
    has_selection: bool
    selected_count: int
    has_current_paper: bool
    has_target_papers: bool
    s2_active: bool
    s2_data_loaded: bool


_PALETTE_BLOCKED_COPY = {
    "selection": "Select a paper first",
    "visible papers": "Show at least one visible paper first",
    "an active search": "Run a search first",
    "history mode": "Open a history-backed digest first",
    "saved marks": "Set a mark first",
    "watch list entries": "Create a watch-list entry first",
    "starred papers": "Star at least one paper first",
    "Semantic Scholar enabled": "Enable Semantic Scholar first",
    "S2 data": "Fetch S2 data for this paper first",
    "LLM configuration": "Configure an LLM command first",
    "2-3 selected papers": "Select exactly 2 or 3 papers first",
}

TARGET_PAPER_PALETTE_ACTIONS = frozenset(
    {
        "add_to_collection",
        "author_profile",
        "copy_selected",
        "download_pdf",
        "edit_notes",
        "edit_tags",
        "export_menu",
        "open_pdf",
        "open_url",
        "preview_figure",
        "preview_pdf",
        "read_abstract_aloud",
        "show_similar",
        "start_mark",
        "toggle_read",
        "schedule_review",
        "mark_reviewed",
        "clear_review",
        "toggle_star",
        "track_author",
    }
)

COMMAND_PALETTE_GROUPS: dict[str, str] = {
    "toggle_search": "Core",
    "show_search_syntax": "Core",
    "arxiv_search": "Research",
    "prev_date": "Advanced",
    "next_date": "Advanced",
    "open_url": "Core",
    "open_pdf": "Core",
    "preview_figure": "Research",
    "read_abstract_aloud": "Research",
    "download_pdf": "Core",
    "copy_selected": "Core",
    "toggle_read": "Organize",
    "toggle_star": "Organize",
    "quick_triage": "Organize",
    "train_triage_model": "Organize",
    "clear_triage_model": "Organize",
    "triage_model_diagnostics": "Organize",
    "edit_notes": "Organize",
    "edit_tags": "Organize",
    "select_all": "Core",
    "clear_selection": "Core",
    "toggle_select": "Core",
    "cycle_sort": "Core",
    "toggle_watch_filter": "Organize",
    "manage_watch_list": "Organize",
    "toggle_preview": "Advanced",
    "toggle_compact_list": "Advanced",
    "export_menu": "Core",
    "export_metadata": "Advanced",
    "import_metadata": "Advanced",
    "fetch_s2": "Research",
    "ctrl_e_dispatch": "Research",
    "toggle_hf": "Research",
    "refresh_conference_deadlines": "Research",
    "check_versions": "Research",
    "citation_graph": "Research",
    "trend_radar": "Research",
    "generate_summary": "Research",
    "chat_with_paper": "Research",
    "debate_paper": "Research",
    "compare_papers": "Research",
    "remix_papers": "Research",
    "score_relevance": "Research",
    "edit_interests": "Research",
    "auto_tag": "Research",
    "show_similar": "Research",
    "serendipity": "Research",
    "author_profile": "Research",
    "track_author": "Research",
    "add_bookmark": "Organize",
    "collections": "Organize",
    "add_to_collection": "Organize",
    "toggle_detail_mode": "Advanced",
    "grow_detail_pane": "Advanced",
    "grow_list_pane": "Advanced",
    "reset_pane_sizes": "Advanced",
    "cycle_theme": "Advanced",
    "open_settings": "Advanced",
    "toggle_sections": "Advanced",
    "show_help": "Core",
    "start_mark": "Advanced",
    "start_goto_mark": "Advanced",
}


def _palette_ctrl_e_copy(state: _PaletteAppState) -> tuple[str, str]:
    if state.in_arxiv_api_mode:
        return "Exit Search Results", "Return to your local or history papers"
    return "Toggle Semantic Scholar", "Enable or disable Semantic Scholar enrichment"


def _palette_hf_copy(state: _PaletteAppState) -> tuple[str, str]:
    if state.hf_active:
        return (
            "Disable HuggingFace Trending",
            "Hide HuggingFace badges and detail-pane matches",
        )
    return (
        "Enable HuggingFace Trending",
        "Show HuggingFace badges and detail-pane matches",
    )


def _palette_preview_copy(state: _PaletteAppState) -> tuple[str, str]:
    if state.show_abstract_preview:
        return "Hide Abstract Preview", "Return to a denser paper list without snippets"
    return "Show Abstract Preview", "Reveal abstract snippets in the paper list"


def _palette_compact_list_copy(state: _PaletteAppState) -> tuple[str, str]:
    if state.compact_list:
        return "Show Full Rows", "Return to title, authors, and metadata per paper"
    return "Compact List (Titles Only)", "Show one line per paper to skim more titles at once"


def _palette_detail_mode_copy(state: _PaletteAppState) -> tuple[str, str]:
    if state.detail_mode == "scan":
        return "Switch to Full Details", "Expand the detail pane for long-form reading"
    return "Switch to Scan Details", "Return to a faster triage-focused detail view"


def _first_failed_palette_requirement(
    action_name: str,
    requirements: tuple[tuple[set[str], bool, str], ...],
) -> str:
    for actions, available, reason in requirements:
        if action_name in actions and not available:
            return reason
    return ""


def _palette_blocked_copy(reason: str) -> str:
    """Return actionable disabled-command copy for terse internal blockers."""
    return _PALETTE_BLOCKED_COPY.get(reason, reason)


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
    ("Preview PDF", "Render a terminal preview of the current paper PDF", "F", "preview_pdf"),
    (
        "Preview First Figure",
        "Render the first arXiv HTML figure in the terminal",
        "I",
        "preview_figure",
    ),
    (
        "Read Abstract Aloud",
        "Play the current paper title and abstract through system TTS",
        "y",
        "read_abstract_aloud",
    ),
    ("Download PDF", "Download PDF(s) to local folder", "d", "download_pdf"),
    ("Copy to Clipboard", "Copy paper info to clipboard", "c", "copy_selected"),
    ("Toggle Read", "Mark paper(s) as read/unread", "r", "toggle_read"),
    (
        "Mark Visible Read",
        "Mark all visible (filtered) papers as read",
        "Ctrl+r",
        "mark_visible_read",
    ),
    (
        "Quick Triage",
        "Review visible unread papers one at a time",
        "T",
        "quick_triage",
    ),
    (
        "Train Triage Model",
        "Learn likely-star and likely-skip buckets from saved decisions",
        "",
        "train_triage_model",
    ),
    (
        "Clear Triage Model",
        "Delete the local ML triage model and hide prediction badges",
        "",
        "clear_triage_model",
    ),
    (
        "Triage Model Diagnostics",
        "Explain local ML triage training, buckets, uncertainty, and learned terms",
        "",
        "triage_model_diagnostics",
    ),
    (
        "Schedule Review",
        "Add current or selected papers to the spaced-review queue",
        "",
        "schedule_review",
    ),
    (
        "Mark Reviewed",
        "Advance current or selected papers to the next review interval",
        "",
        "mark_reviewed",
    ),
    (
        "Clear Review",
        "Remove current or selected papers from the spaced-review queue",
        "",
        "clear_review",
    ),
    (
        "Show Due Reviews",
        "Filter the list to papers whose next review is due",
        "",
        "show_due_reviews",
    ),
    ("Toggle Star", "Star/unstar paper(s)", "x", "toggle_star"),
    ("Edit Notes", "Add or edit notes for current paper", "n", "edit_notes"),
    ("Edit Tags", "Add or edit tags (bulk when selected)", "t", "edit_tags"),
    ("Select All", "Select all visible papers", "a", "select_all"),
    ("Clear Selection", "Deselect all papers", "u", "clear_selection"),
    ("Toggle Selection", "Toggle selection on current paper", "Space", "toggle_select"),
    (
        "Cycle Sort",
        "Cycle sort: title/date/arxiv_id/citations/trending/relevance/queue/triage",
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
    (
        "Compact List (Titles Only)",
        "Show one line per paper to skim more titles at once",
        "z",
        "toggle_compact_list",
    ),
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
    (
        "Refresh Conference Deadlines",
        "Import upcoming submission deadlines from the configured AI Deadlines source",
        "",
        "refresh_conference_deadlines",
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
        "Debate Paper",
        "Generate an advocate-vs-Reviewer-2 debate for the current paper",
        "",
        "debate_paper",
    ),
    (
        "Compare Papers",
        "Side-by-side comparison for 2-3 selected papers",
        "Ctrl+v",
        "compare_papers",
    ),
    (
        "Paper Remix",
        "Generate one research idea from 2-3 selected papers",
        "",
        "remix_papers",
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
    (
        "Surprise Me",
        "Jump to an unread paper far from your current interests",
        "",
        "serendipity",
    ),
    (
        "Trend Radar",
        "Show category, author, and topic trends across local history",
        "",
        "trend_radar",
    ),
    (
        "Author Profile",
        "Show this author's local papers, co-authors, and cached citations",
        "",
        "author_profile",
    ),
    (
        "Track Author",
        "Highlight future papers from an exact author match",
        "",
        "track_author",
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
    (
        "Grow Detail Pane",
        "Give more space to the paper details pane",
        "Alt+Right",
        "grow_detail_pane",
    ),
    (
        "Grow Paper List",
        "Give more space to the paper list pane",
        "Alt+Left",
        "grow_list_pane",
    ),
    (
        "Reset Pane Sizes",
        "Restore the default list/details split",
        "Alt+0",
        "reset_pane_sizes",
    ),
    ("Cycle Theme", "Switch between installed color themes", "Ctrl+t", "cycle_theme"),
    (
        "Settings",
        "Configure LLM preset, theme, enrichment, and research interests",
        ",",
        "open_settings",
    ),
    ("Toggle Sections", "Show/hide detail pane sections", "Ctrl+d", "toggle_sections"),
    (
        "Focus Details",
        "Move focus between the paper list and detail pane",
        "Tab",
        "toggle_focus_pane",
    ),
    ("Help", "Show all keyboard shortcuts", "?", "show_help"),
    ("Set Mark", "Set a named mark (a-z) at current position", "m", "start_mark"),
    ("Jump to Mark", "Jump to a named mark (a-z)", "'", "start_goto_mark"),
]


__all__ = [
    "COMMAND_PALETTE_COMMANDS",
    "FOOTER_CONTEXTS",
    "TARGET_PAPER_PALETTE_ACTIONS",
    "TaskTrackingApp",
    "_PaletteAppState",
]
