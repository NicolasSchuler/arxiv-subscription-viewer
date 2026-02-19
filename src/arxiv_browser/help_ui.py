"""Help screen section builders derived from runtime key bindings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textual.binding import Binding

HELP_SECTION_ACTIONS: list[tuple[str, list[str]]] = [
    (
        "Navigation",
        [
            "cursor_down",
            "cursor_up",
            "prev_date",
            "next_date",
            "goto_bookmark",
            "start_mark",
            "start_goto_mark",
        ],
    ),
    (
        "Search & Filter",
        [
            "toggle_search",
            "cancel_search",
            "arxiv_search",
            "ctrl_e_dispatch",
            "toggle_watch_filter",
            "manage_watch_list",
            "add_bookmark",
            "remove_bookmark",
        ],
    ),
    (
        "Selection & Core Actions",
        [
            "toggle_select",
            "select_all",
            "clear_selection",
            "open_url",
            "open_pdf",
            "copy_selected",
            "cycle_sort",
            "toggle_read",
            "toggle_star",
            "edit_notes",
            "edit_tags",
        ],
    ),
    (
        "Research & AI",
        [
            "show_similar",
            "citation_graph",
            "check_versions",
            "fetch_s2",
            "toggle_hf",
            "generate_summary",
            "chat_with_paper",
            "score_relevance",
            "edit_interests",
            "auto_tag",
        ],
    ),
    (
        "View & Utilities",
        [
            "toggle_preview",
            "export_menu",
            "download_pdf",
            "collections",
            "command_palette",
            "cycle_theme",
            "toggle_sections",
            "show_help",
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
    ("j / k", "Move selection"),
    ("Space", "Select current paper"),
    ("o", "Open selected paper(s)"),
    ("Ctrl+p", "Open command palette"),
    ("?", "Show full shortcuts"),
]

HELP_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "ctrl_e_dispatch": "Toggle S2 (browse) / Exit API (API mode)",
    "command_palette": "Command palette",
    "show_help": "Help overlay",
    "open_url": "Open in browser",
    "open_pdf": "Open PDF",
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
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Build help sections from runtime key bindings."""
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        ("Getting Started", list(HELP_GETTING_STARTED))
    ]
    for section_name, actions in HELP_SECTION_ACTIONS:
        entries: list[tuple[str, str]] = []
        for action_name in actions:
            if action_name == "goto_bookmark":
                entries.append(("1-9", "Jump to bookmark"))
                continue
            binding = _binding_for_help_action(bindings, action_name)
            if binding is None:
                continue
            key = _format_help_key(binding.key)
            description = HELP_DESCRIPTION_OVERRIDES.get(action_name, binding.description)
            entries.append((key, description))
        sections.append((section_name, entries))

    sections.append(("Search Syntax", HELP_SEARCH_SYNTAX))
    return sections


__all__ = ["build_help_sections"]
