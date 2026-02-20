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
}

#left-pane {
    width: 2fr;
    min-width: 50;
    max-width: 100;
    height: 100%;
    border: tall $th-highlight;
    background: $th-panel;
}

#left-pane:focus-within {
    border: tall $th-accent;
}

#right-pane {
    width: 3fr;
    height: 100%;
    border: tall $th-highlight;
    background: $th-panel;
}

#right-pane:focus-within {
    border: tall $th-accent;
}

#list-header {
    padding: 0 1;
    background: $th-panel;
    color: $th-accent;
    text-style: bold;
}

#details-header {
    padding: 0 1;
    background: $th-panel;
    color: $th-accent-alt;
    text-style: bold;
}

#paper-list {
    height: 1fr;
    scrollbar-gutter: stable;
}

#details-scroll {
    height: 1fr;
    padding: 0 1;
}

#search-container {
    height: auto;
    padding: 0 1;
    background: $th-panel;
    display: none;
}

#search-container.visible {
    display: block;
}

#search-input {
    width: 100%;
    border: tall $th-accent;
    background: $th-background;
}

#search-input:focus {
    border: tall $th-accent-alt;
}

#paper-list > .option-list--option-highlighted {
    background: $th-highlight;
}

#paper-list:focus > .option-list--option-highlighted {
    background: $th-highlight-focus;
}

#paper-list > .option-list--option-hover {
    background: $th-panel-alt;
}

PaperDetails {
    padding: 0;
}

VerticalScroll {
    scrollbar-background: $th-scrollbar-bg;
    scrollbar-color: $th-scrollbar-thumb;
    scrollbar-color-hover: $th-scrollbar-hover;
    scrollbar-color-active: $th-scrollbar-active;
}

#status-bar {
    padding: 0 1;
    color: $th-muted;
}
"""

APP_BINDINGS: list[BindingType] = [
    Binding("q", "quit", "Quit", show=False),
    Binding("slash", "toggle_search", "Search", show=False),
    Binding("A", "arxiv_search", "arXiv Search", show=False),
    Binding(
        "ctrl+e",
        "ctrl_e_dispatch",
        "Toggle S2 (browse) / Exit API (API mode)",
        show=False,
    ),
    Binding("escape", "cancel_search", "Cancel", show=False),
    Binding("o", "open_url", "Open Selected", show=False),
    Binding("P", "open_pdf", "Open PDF", show=False),
    Binding("c", "copy_selected", "Copy", show=False),
    Binding("s", "cycle_sort", "Sort", show=False),
    Binding("space", "toggle_select", "Select", show=False),
    Binding("a", "select_all", "Select All", show=False),
    Binding("u", "clear_selection", "Clear Selection", show=False),
    Binding("j", "cursor_down", "Down", show=False),
    Binding("k", "cursor_up", "Up", show=False),
    # Phase 2: Read/Star status and Notes/Tags
    Binding("r", "toggle_read", "Read", show=False),
    Binding("x", "toggle_star", "Star", show=False),
    Binding("n", "edit_notes", "Notes", show=False),
    Binding("t", "edit_tags", "Tags", show=False),
    # Phase 3: Watch list
    Binding("w", "toggle_watch_filter", "Watch", show=False),
    Binding("W", "manage_watch_list", "Watch List", show=False),
    # Phase 4: Bookmarked search tabs
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
    # Phase 5: Abstract preview
    Binding("p", "toggle_preview", "Preview", show=False),
    # Phase 7: Vim-style marks
    Binding("m", "start_mark", "Mark", show=False),
    Binding("apostrophe", "start_goto_mark", "Goto Mark", show=False),
    # Phase 8: Export features (b/B/M accessible via E -> export menu)
    Binding("E", "export_menu", "Export...", show=False),
    Binding("d", "download_pdf", "Download", show=False),
    # Phase 9: Paper similarity
    Binding("R", "show_similar", "Similar", show=False),
    # LLM summary & chat
    Binding("ctrl+s", "generate_summary", "AI Summary", show=False),
    Binding("C", "chat_with_paper", "Chat", show=False),
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
    # Theme cycling
    Binding("ctrl+t", "cycle_theme", "Theme", show=False),
    # Collapsible sections
    Binding("ctrl+d", "toggle_sections", "Sections", show=False),
    # History mode: date navigation
    Binding("bracketleft", "prev_date", "Older", show=False),
    Binding("bracketright", "next_date", "Newer", show=False),
    # Help overlay
    Binding("question_mark", "show_help", "Help (?)", show=False),
    # Command palette
    Binding("ctrl+p", "command_palette", "Commands", show=False),
    # Collections
    Binding("ctrl+k", "collections", "Collections", show=False),
]

__all__ = [
    "APP_BINDINGS",
    "APP_CSS",
]
