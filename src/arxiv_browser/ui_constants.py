"""Internal UI constants for the ArxivBrowser app."""

from __future__ import annotations

from textual.binding import Binding, BindingType

APP_CSS = """
Screen {
    background: $th-background;
}

Header {
    background: $th-panel-alt;
    color: $th-text;
}



#main-container {
    height: 1fr;
    background: $th-background;
}

#left-pane {
    width: 2fr;
    min-width: 50;
    max-width: 100;
    height: 100%;
    border: solid $th-highlight;
    background: $th-panel;
}

#left-pane:focus-within {
    border: solid $th-highlight;
    border-left: solid $th-accent;
    border-top: solid $th-accent;
}

#right-pane {
    width: 3fr;
    max-width: 100;
    height: 100%;
    border: solid $th-highlight;
    background: $th-panel;
}

#right-pane:focus-within {
    border: solid $th-highlight;
    border-left: solid $th-accent;
    border-top: solid $th-accent;
}

/* Narrow terminals (< 96 cols): stack panes vertically, list first. */
Screen.-narrow #main-container {
    layout: vertical;
}

Screen.-narrow #left-pane,
Screen.-narrow #right-pane {
    width: 100%;
    min-width: 0;
    max-width: 100%;
}

Screen.-narrow #left-pane {
    height: 2fr;
}

Screen.-narrow #right-pane {
    height: 1fr;
}

#list-header {
    padding: 1 1;
    background: $th-panel;
    color: $th-accent;
    text-style: bold;
}

#details-header {
    padding: 1 1;
    background: $th-panel;
    color: $th-accent-alt;
    text-style: bold;
}

#paper-list {
    height: 1fr;
    scrollbar-gutter: stable;
    scrollbar-background: $th-panel;
    scrollbar-color: $th-highlight;
    scrollbar-color-hover: $th-muted;
    scrollbar-color-active: $th-accent;
}

#details-scroll {
    height: 1fr;
    padding: 0 1;
    scrollbar-background: $th-panel;
    scrollbar-color: $th-highlight;
    scrollbar-color-hover: $th-muted;
    scrollbar-color-active: $th-accent;
}

OmniInput {
    padding: 0 1;
    background: $th-panel;
}

OmniInput #omni-input {
    width: 100%;
    border: solid $th-highlight;
    background: $th-background;
}

OmniInput #omni-input:focus {
    border: solid $th-accent;
}

OmniInput #omni-hint {
    color: $th-muted;
    padding: 0 1 1 1;
}

OmniInput #omni-results {
    background: $th-panel;
    max-height: 12;
    border-left: solid $th-highlight;
    scrollbar-background: $th-panel;
    scrollbar-color: $th-highlight;
    scrollbar-color-hover: $th-muted;
    scrollbar-color-active: $th-accent;
}

OmniInput #omni-results > .option-list--option-highlighted {
    background: $th-selection;
    border-left: solid $th-accent;
}

#paper-list > .option-list--option-highlighted {
    background: $th-panel-alt;
    border-left: solid $th-muted;
}

#paper-list:focus > .option-list--option-highlighted {
    background: $th-highlight;
    border-left: solid $th-accent;
}

#paper-list > .selected {
    background: $th-selection;
}

#paper-list > .option-list--option-hover {
    background: $th-panel-alt;
}

PaperDetails {
    padding: 0;
}

VerticalScroll {
    scrollbar-background: $th-panel;
    scrollbar-color: $th-highlight;
    scrollbar-color-hover: $th-muted;
    scrollbar-color-active: $th-scrollbar-active;
}

#status-bar {
    padding: 0 1;
    color: $th-muted;
}
"""

# Horizontal breakpoints (terminal width in cells) used to toggle responsive
# layout classes on the active screen. Below ``NARROW_BREAKPOINT`` cells the
# list/detail panes stack vertically (list first); at or above it they keep the
# side-by-side split.
NARROW_BREAKPOINT = 96
APP_HORIZONTAL_BREAKPOINTS: list[tuple[int, str]] = [
    (0, "-narrow"),
    (NARROW_BREAKPOINT, "-wide"),
]

# ---------------------------------------------------------------------------
# Keybinding Tiers
# ---------------------------------------------------------------------------
# Bindings are categorised into three tiers so that new users see only the
# essentials while power users can discover advanced shortcuts progressively.
#
# Core (~12 keys) - shown in the default footer.
#   /  search          j/k  navigate       Space  select
#   o  open            r    read           x      star
#   E  export          s    sort           Tab    focus details
#   Ctrl+p  commands   q    quit           [/]    dates (history)
#
# Standard (~15-20 keys) - prominent in help overlay.
#   a  select all      u  clear selection  n  notes       t  tags
#   T  quick triage
#   c  copy            d  download         P  PDF         F  preview PDF
#   I  preview figure  z  compact list
#   v  detail mode     w  watch filter     W  watch list  A  API search
#   y  read aloud      1-9  bookmarks      Ctrl+b  add bookmark
#
# Power (remaining) - discoverable via command palette (Ctrl+p).
#   m/'  marks          R  similar          G  citation graph
#   V    versions       e  S2 fetch         Ctrl+s  AI summary
#   C    chat           Ctrl+v compare      Ctrl+p  debate/remix
#   L    relevance
#   Ctrl+g  auto-tag
#   Ctrl+t  theme       Ctrl+d  sections    Ctrl+k  collections
#   Ctrl+h  HF toggle   Ctrl+e  S2 toggle   Ctrl+l  interests
#   Ctrl+Shift+b  remove bookmark
# ---------------------------------------------------------------------------

APP_BINDINGS: list[BindingType] = [
    Binding("q", "quit", "Quit", show=False),
    Binding("slash", "toggle_search", "Search", show=False),
    Binding("A", "arxiv_search", "arXiv Search (@query)", show=False),
    Binding(
        "ctrl+e",
        "ctrl_e_dispatch",
        "Toggle S2 (browse) / Exit API (API mode)",
        show=False,
    ),
    Binding("escape", "cancel_search", "Cancel", show=False),
    Binding("o", "open_url", "Open in Browser", show=False),
    Binding("P", "open_pdf", "Open PDF", show=False),
    Binding("F", "preview_pdf", "Preview PDF", show=False),
    Binding("I", "preview_figure", "Preview Figure", show=False),
    Binding("c", "copy_selected", "Copy", show=False),
    Binding("s", "cycle_sort", "Sort", show=False),
    Binding("tab", "toggle_focus_pane", "Focus Details", show=False, priority=True),
    Binding("space", "toggle_select", "Select", show=False),
    Binding("a", "select_all", "Select All", show=False),
    Binding("u", "clear_selection", "Clear Selection", show=False),
    Binding("j", "cursor_down", "Down", show=False),
    Binding("k", "cursor_up", "Up", show=False),
    # Paper management
    Binding("r", "toggle_read", "Read", show=False),
    Binding("x", "toggle_star", "Star", show=False),
    Binding("n", "edit_notes", "Notes", show=False),
    Binding("t", "edit_tags", "Tags", show=False),
    Binding("T", "quick_triage", "Quick Triage", show=False),
    # Watch list
    Binding("w", "toggle_watch_filter", "Watch", show=False),
    Binding("W", "manage_watch_list", "Watch List", show=False),
    # Bookmarked search tabs
    Binding("1", "goto_bookmark(0)", "Bookmark 1", show=False),
    Binding("2", "goto_bookmark(1)", "Bookmark 2", show=False),
    Binding("3", "goto_bookmark(2)", "Bookmark 3", show=False),
    Binding("4", "goto_bookmark(3)", "Bookmark 4", show=False),
    Binding("5", "goto_bookmark(4)", "Bookmark 5", show=False),
    Binding("6", "goto_bookmark(5)", "Bookmark 6", show=False),
    Binding("7", "goto_bookmark(6)", "Bookmark 7", show=False),
    Binding("8", "goto_bookmark(7)", "Bookmark 8", show=False),
    Binding("9", "goto_bookmark(8)", "Bookmark 9", show=False),
    Binding("ctrl+b", "add_bookmark", "Add Bookmark", show=False),
    Binding("ctrl+shift+b", "remove_bookmark", "Del Bookmark", show=False),
    # Abstract preview
    Binding("p", "toggle_preview", "Preview", show=False),
    Binding("z", "toggle_compact_list", "Compact List", show=False),
    Binding("v", "toggle_detail_mode", "Detail Mode", show=False),
    Binding("y", "read_abstract_aloud", "Read Abstract", show=False),
    # Vim-style marks
    Binding("m", "start_mark", "Mark", show=False),
    Binding("apostrophe", "start_goto_mark", "Goto Mark", show=False),
    # Export features (b/B/M accessible via E -> export menu)
    Binding("E", "export_menu", "Export...", show=False),
    Binding("d", "download_pdf", "Download", show=False),
    # Paper similarity
    Binding("R", "show_similar", "Similar", show=False),
    # LLM summary & chat
    Binding("ctrl+s", "generate_summary", "AI Summary", show=False),
    Binding("C", "chat_with_paper", "Chat", show=False),
    Binding("ctrl+v", "compare_papers", "Compare Papers", show=False),
    # Semantic Scholar enrichment
    Binding("e", "fetch_s2", "Enrich (S2)", show=False),
    # HuggingFace trending
    Binding("ctrl+h", "toggle_hf", "Toggle HF", show=False),
    # Version tracking
    Binding("V", "check_versions", "Check Versions", show=False),
    # Citation graph
    Binding("G", "citation_graph", "Citation Graph", show=False),
    # Relevance scoring
    Binding("L", "score_relevance", "Score Relevance", show=False),
    Binding("ctrl+l", "edit_interests", "Edit Interests", show=False),
    Binding("ctrl+g", "auto_tag", "Auto-Tag", show=False),
    Binding("ctrl+r", "mark_visible_read", "Mark Visible Read", show=False),
    # Theme cycling
    Binding("ctrl+t", "cycle_theme", "Theme", show=False),
    # Collapsible sections
    Binding("ctrl+d", "toggle_sections", "Sections", show=False),
    # History mode: date navigation
    Binding("bracketleft", "prev_date", "Older", show=False),
    Binding("bracketright", "next_date", "Newer", show=False),
    # Help overlay
    Binding("question_mark", "show_help", "Help (?)", show=False),
    # What's New overlay (version-bump release notes)
    Binding("f1", "show_whats_new", "What's New", show=False),
    # Command palette
    Binding("ctrl+p", "command_palette", "Command palette (>cmd)", show=False),
    # Collections
    Binding("ctrl+k", "collections", "Collections", show=False),
]

__all__ = [
    "APP_BINDINGS",
    "APP_CSS",
    "APP_HORIZONTAL_BREAKPOINTS",
    "NARROW_BREAKPOINT",
]
