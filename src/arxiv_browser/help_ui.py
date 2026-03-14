"""Help screen section builders derived from runtime key bindings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textual.binding import Binding

HELP_SECTION_ACTIONS: list[tuple[str, list[str]]] = [
    (
        "Core Actions",
        [
            "toggle_search",
            "show_search_syntax",
            "cursor_down",
            "cursor_up",
            "toggle_select",
            "select_all",
            "clear_selection",
            "open_url",
            "open_pdf",
            "cycle_sort",
            "copy_selected",
            "export_menu",
            "download_pdf",
            "command_palette",
            "show_help",
        ],
    ),
    (
        "Standard · Organize",
        [
            "toggle_read",
            "toggle_star",
            "edit_notes",
            "edit_tags",
            "toggle_watch_filter",
            "manage_watch_list",
            "goto_bookmark",
            "add_bookmark",
            "remove_bookmark",
            "collections",
        ],
    ),
    (
        "Power · Research Tools",
        [
            "arxiv_search",
            "ctrl_e_dispatch",
            "fetch_s2",
            "show_similar",
            "citation_graph",
            "check_versions",
            "toggle_hf",
            "generate_summary",
            "chat_with_paper",
            "score_relevance",
            "edit_interests",
            "auto_tag",
        ],
    ),
    (
        "Power · Advanced",
        [
            "prev_date",
            "next_date",
            "toggle_preview",
            "toggle_detail_mode",
            "cycle_theme",
            "toggle_sections",
            "start_mark",
            "start_goto_mark",
            "quit",
        ],
    ),
]

HELP_SEARCH_SYNTAX: list[tuple[str, str]] = [
    ("cat:cs.AI", "Category filter"),
    ("tag:to-read", "Tag filter"),
    ("author:hinton", "Author filter"),
    ("title:transformer", "Title filter"),
    ("abstract:attention", "Abstract filter"),
    ("unread / starred", "State filters"),
    ("AND / OR / NOT", "Boolean operators"),
]

HELP_GETTING_STARTED: list[tuple[str, str]] = [
    ("/", "Search papers"),
    ("A", "Search all arXiv"),
    ("j / k", "Move selection"),
    ("Space", "Select current paper"),
    ("o", "Open selected paper(s)"),
    ("r / x", "Read or star the current paper"),
    ("E", "Export current or selected papers"),
    ("Ctrl+p", "Open commands"),
    ("[ / ]", "Change dates (history mode)"),
    ("?", "Show full shortcuts"),
]

HELP_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "ctrl_e_dispatch": "Toggle S2 (browse) / Exit API (API mode)",
    "command_palette": "Commands",
    "show_help": "Help overlay",
    "show_search_syntax": "Search examples & operators",
    "open_url": "Open in Browser",
    "open_pdf": "Open PDF",
    "toggle_detail_mode": "Toggle detail density (scan/full)",
}


# Actions accessible only via command palette (Ctrl+p) — no direct keybinding.
HELP_PALETTE_ONLY_KEYS: dict[str, str] = {
    "show_search_syntax": "Ctrl+p",
}


def _format_help_key(key: str) -> str:
    """Normalize Textual key names for user-facing help text."""
    replacements = {
        "slash": "/",
        "space": "Space",
        "question_mark": "?",
        "apostrophe": "'",
        "bracketleft": "[",
        "bracketright": "]",
    }
    key = replacements.get(key, key)
    if key.startswith("ctrl+"):
        rest = key.removeprefix("ctrl+")
        rest = rest.replace("shift+", "Shift+")
        return "Ctrl+" + rest
    return key


def _iter_binding_definitions(
    bindings: Sequence[Binding | tuple[Any, ...]],
) -> list[Binding]:
    """Normalize App.BINDINGS entries into Binding objects."""
    normalized: list[Binding] = []
    for binding_item in bindings:
        if isinstance(binding_item, Binding):
            normalized.append(binding_item)
            continue
        key = str(binding_item[0]) if len(binding_item) > 0 else ""
        action = str(binding_item[1]) if len(binding_item) > 1 else ""
        description = str(binding_item[2]) if len(binding_item) > 2 else ""
        normalized.append(Binding(key, action, description, show=False))
    return normalized


def _binding_for_help_action(
    bindings: Sequence[Binding | tuple[Any, ...]],
    action_name: str,
) -> Binding | None:
    """Resolve a Binding by action name, supporting parameterized actions."""
    for binding in _iter_binding_definitions(bindings):
        if binding.action == action_name:
            return binding
        if binding.action.startswith(f"{action_name}("):
            return binding
    return None


def build_help_sections(
    bindings: Sequence[Binding | tuple[Any, ...]],
    *,
    search_first: bool = False,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Build help sections from runtime key bindings."""
    from arxiv_browser._ascii import is_ascii_mode

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        ("Getting Started", list(HELP_GETTING_STARTED))
    ]
    if search_first:
        sections.append(("Search Syntax", HELP_SEARCH_SYNTAX))
    for section_name, actions in HELP_SECTION_ACTIONS:
        # Replace middle-dot separator with ASCII dash when in ASCII mode
        display_name = section_name.replace("\u00b7", "-") if is_ascii_mode() else section_name
        entries: list[tuple[str, str]] = []
        for action_name in actions:
            if action_name == "goto_bookmark":
                entries.append(("1-9", "Jump to bookmark"))
                continue
            binding = _binding_for_help_action(bindings, action_name)
            if binding is None:
                if action_name in HELP_PALETTE_ONLY_KEYS:
                    key = _format_help_key(HELP_PALETTE_ONLY_KEYS[action_name])
                    description = HELP_DESCRIPTION_OVERRIDES.get(action_name, action_name)
                    entries.append((key, description))
                continue
            key = _format_help_key(binding.key)
            description = HELP_DESCRIPTION_OVERRIDES.get(action_name, binding.description)
            entries.append((key, description))
        sections.append((display_name, entries))

    if not search_first:
        sections.insert(1, ("Search Syntax", HELP_SEARCH_SYNTAX))
    return sections


__all__ = ["build_help_sections"]
