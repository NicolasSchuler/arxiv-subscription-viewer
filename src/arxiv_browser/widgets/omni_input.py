"""Unified OmniInput — VS Code-style search with mode prefixes.

Modes:
- No prefix  → local paper search (default)
- ``>``      → command palette (inline results)
- ``@``      → arXiv API search (triggers on Enter)
- ``~``      → semantic local paper search
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.message import Message
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from arxiv_browser.fuzzy import partial_fuzzy_score
from arxiv_browser.palette import (
    PALETTE_DESC_MAX_LEN,
    PALETTE_KEY_MAX_LEN,
    PALETTE_NAME_MAX_LEN,
    PaletteCommand,
    truncate_palette_text,
)
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)

PREFIX_COMMAND = ">"
PREFIX_API = "@"
PREFIX_SEMANTIC = "~"

OMNI_PLACEHOLDER = "Search papers · ~ semantic · > commands · @ arXiv API"
OMNI_HINT_LOCAL = 'Examples: cat:cs.AI  author:hinton  ~ hallucination in RAG  "large language"'
OMNI_HINT_COMMAND = "↑↓ move · Enter run · Esc close"
OMNI_HINT_API = "Enter to search arXiv · Esc cancel"
OMNI_HINT_SEMANTIC = "Semantic search over titles + abstracts · Esc cancel"

FUZZY_THRESHOLD = 40
COMMAND_NAME_WIDTH = 20
COMMAND_GROUP_WIDTH = 8
COMMAND_DESC_WIDTH = 26

CommandMatch = tuple[float, int, int, PaletteCommand]


@dataclass(slots=True, frozen=True)
class OmniMode:
    """Parsed state of the OmniInput prefix and query."""

    mode: str  # "local", "command", "api", "semantic"
    query: str  # text after prefix (stripped)


def parse_omni_mode(raw: str) -> OmniMode:
    """Parse raw input text into an OmniMode."""
    if raw.startswith(PREFIX_COMMAND):
        return OmniMode(mode="command", query=raw[1:].lstrip())
    if raw.startswith(PREFIX_API):
        return OmniMode(mode="api", query=raw[1:].lstrip())
    if raw.startswith(PREFIX_SEMANTIC):
        return OmniMode(mode="semantic", query=raw[1:].lstrip())
    return OmniMode(mode="local", query=raw)


class OmniInput(Vertical):
    """Unified search container with mode-prefix detection.

    Replaces the old ``#search-container`` + ``#search-input`` + ``#search-hint``.
    """

    DEFAULT_CSS = """
    OmniInput {
        height: auto;
        display: none;
    }

    OmniInput.visible {
        display: block;
    }

    OmniInput #omni-input {
        margin: 0;
    }

    OmniInput #omni-hint {
        color: $th-muted;
        height: 1;
        padding: 0 1;
    }

    OmniInput #omni-results {
        max-height: 12;
        display: none;
    }

    OmniInput #omni-results.visible {
        display: block;
    }
    """

    # --- messages ---

    class LocalSearch(Message):
        """Emitted when the user types a local search query (no prefix)."""

        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query

    class ApiSearch(Message):
        """Emitted when the user submits an arXiv API query (@ prefix + Enter)."""

        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query

    class CommandSelected(Message):
        """Emitted when the user selects a command (> prefix)."""

        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    class LocalSearchSubmitted(Message):
        """Emitted when the user presses Enter in local search mode."""

        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query

    class Dismissed(Message):
        """Emitted when the user dismisses the OmniInput (Esc)."""

    # --- state ---

    def __init__(self) -> None:
        super().__init__()
        self._commands: list[PaletteCommand] = []
        self._filtered_commands: list[PaletteCommand] = []
        self._command_option_indexes: dict[int, int] = {}
        self._current_mode: str = "local"

    def compose(self) -> ComposeResult:
        """Build the omni input: text field, hint line, and results panel."""
        yield Input(
            placeholder=_mode_safe_hint(OMNI_PLACEHOLDER),
            id="omni-input",
            disabled=True,
            select_on_focus=False,
        )
        yield Static(_hint_for_mode("local"), id="omni-hint")
        yield OptionList(id="omni-results", disabled=True)

    def set_commands(self, commands: list[PaletteCommand]) -> None:
        """Update the available command palette commands."""
        self._commands = commands

    def open(self, initial_text: str = "") -> None:
        """Show the OmniInput and focus it."""
        self.add_class("visible")
        inp = self.query_one("#omni-input", Input)
        inp.disabled = False
        inp.value = initial_text
        inp.cursor_position = len(initial_text)
        inp.focus()

    def close(self) -> None:
        """Hide the OmniInput and clear state."""
        self.remove_class("visible")
        inp = self.query_one("#omni-input", Input)
        inp.value = ""
        inp.disabled = True
        self._hide_results()
        self._current_mode = "local"

    @property
    def is_open(self) -> bool:
        """Whether the OmniInput is currently visible."""
        return self.has_class("visible")

    @property
    def value(self) -> str:
        """Current text in the input field."""
        return self.query_one("#omni-input", Input).value

    @value.setter
    def value(self, text: str) -> None:
        """Replace the current input text."""
        self.query_one("#omni-input", Input).value = text

    def hide(self) -> None:
        """Hide the OmniInput without clearing the input value."""
        self.remove_class("visible")
        self.query_one("#omni-input", Input).disabled = True
        self._hide_results()

    def focus_input(self) -> None:
        """Focus the inner Input widget."""
        self.query_one("#omni-input", Input).focus()

    # --- internal ---

    def _show_results(self) -> None:
        results = self.query_one("#omni-results", OptionList)
        results.disabled = False
        results.add_class("visible")

    def _hide_results(self) -> None:
        results = self.query_one("#omni-results", OptionList)
        results.remove_class("visible")
        results.disabled = True
        results.clear_options()
        self._filtered_commands = []
        self._command_option_indexes = {}

    def _update_hint(self, mode: str) -> None:
        hint = self.query_one("#omni-hint", Static)
        hint.update(_hint_for_mode(mode))

    def _populate_command_results(self, query: str) -> None:
        """Populate the inline results list with matching commands."""
        results = self.query_one("#omni-results", OptionList)
        results.clear_options()
        self._filtered_commands = self._filter_commands(query)
        self._command_option_indexes = {}

        show_group_headers = len({cmd.group for cmd in self._filtered_commands}) > 1
        last_group: str | None = None
        for command_index, cmd in enumerate(self._filtered_commands):
            if show_group_headers and cmd.group != last_group:
                results.add_option(_command_group_header(cmd.group))
                last_group = cmd.group
            option_index = results.option_count
            results.add_option(self._command_option(cmd, show_group=not show_group_headers))
            self._command_option_indexes[option_index] = command_index
        self._add_empty_command_result(results, query)

        self._show_results()

    def _filter_commands(self, query: str) -> list[PaletteCommand]:
        if not query:
            return list(self._commands)

        q = query.lower()
        scored = [match for cmd in self._commands if (match := self._command_match(q, cmd))]
        scored.sort(key=_command_match_sort_key, reverse=True)
        return [cmd for _, _, _, cmd in scored]

    def _command_match(self, query: str, command: PaletteCommand) -> CommandMatch | None:
        score = max(
            partial_fuzzy_score(query, command.name),
            partial_fuzzy_score(query, command.description),
            partial_fuzzy_score(query, command.group),
            partial_fuzzy_score(query, command.key_hint),
            partial_fuzzy_score(query, command.action),
        )
        if score < FUZZY_THRESHOLD:
            return None
        return score, int(command.enabled), int(command.suggested), command

    def _add_empty_command_result(self, results: OptionList, query: str) -> None:
        if self._filtered_commands or not query:
            return

        safe = escape_rich_text(query)
        # Kept short so it fits one row of the narrow results dropdown (the Esc
        # affordance already lives in the persistent hint line below the input).
        results.add_option(Option(f'[dim]No matching commands for "{safe}"[/]', disabled=True))

    def _command_option(self, command: PaletteCommand, *, show_group: bool = True) -> Option:
        return Option(
            self._command_option_label(command, show_group=show_group),
            disabled=not command.enabled,
        )

    def _command_option_label(self, command: PaletteCommand, *, show_group: bool = True) -> str:
        safe_name, safe_desc, safe_hint, safe_group, safe_blocked = self._safe_command_parts(
            command
        )
        if not command.enabled:
            blocked = f"  Requires: {safe_blocked}" if safe_blocked else ""
            muted = theme_colors_for(self)["muted"]
            disabled_parts = [safe_name, safe_desc]
            if show_group:
                disabled_parts.insert(1, safe_group)
            return f"[dim]{'  '.join(disabled_parts)}[/][{muted}]{blocked}[/]"

        colors = theme_colors_for(self)
        parts = [f"[bold {colors['accent']}]{safe_name}[/]"]
        if show_group:
            parts.append(f"[dim {colors['purple']}]{safe_group}[/]")
        parts.append(f"[{colors['muted']}]{safe_desc}[/]")
        if command.suggested:
            parts.append(f"[{colors['green']}]* suggested[/]")
        if safe_hint:
            parts.append(f"[{colors['green']}]{safe_hint}[/]")
        return "  ".join(parts)

    def _safe_command_parts(self, command: PaletteCommand) -> tuple[str, str, str, str, str]:
        name = _palette_cell(command.name, PALETTE_NAME_MAX_LEN, COMMAND_NAME_WIDTH)
        desc = _palette_cell(command.description, PALETTE_DESC_MAX_LEN, COMMAND_DESC_WIDTH)
        hint = (
            _palette_cell(command.key_hint, PALETTE_KEY_MAX_LEN, PALETTE_KEY_MAX_LEN)
            if command.key_hint
            else ""
        )
        group = _palette_cell(command.group, PALETTE_KEY_MAX_LEN, COMMAND_GROUP_WIDTH)
        blocked = (
            truncate_palette_text(command.blocked_reason, PALETTE_DESC_MAX_LEN)
            if command.blocked_reason
            else ""
        )
        return (
            escape_rich_text(name),
            escape_rich_text(desc),
            escape_rich_text(hint),
            escape_rich_text(group),
            escape_rich_text(blocked),
        )

    def on_key(self, event: Key) -> None:
        """Let the focused input move through command results with arrow keys."""
        if self._current_mode != "command" or event.key not in {"up", "down"}:
            return
        if not self._filtered_commands:
            return
        self._move_command_highlight(1 if event.key == "down" else -1)
        event.prevent_default()
        event.stop()

    def _move_command_highlight(self, direction: int) -> None:
        """Move the command-result highlight to the next enabled command."""
        results = self.query_one("#omni-results", OptionList)
        option_indexes = sorted(self._command_option_indexes)
        count = len(option_indexes)
        if count == 0:
            return
        start = results.highlighted
        if start is None:
            index = -1 if direction > 0 else count
        elif start in self._command_option_indexes:
            index = option_indexes.index(start)
        else:
            before = [
                pos for pos, option_index in enumerate(option_indexes) if option_index < start
            ]
            index = len(before) - 1 if direction > 0 else len(before)
        for _ in range(count):
            index = (index + direction) % count
            option_index = option_indexes[index]
            command_index = self._command_option_indexes[option_index]
            if self._filtered_commands[command_index].enabled:
                results.highlighted = option_index
                return

    # --- event handlers ---

    @on(Input.Changed, "#omni-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        """Route input changes based on mode prefix."""
        parsed = parse_omni_mode(event.value)
        old_mode = self._current_mode
        self._current_mode = parsed.mode

        if parsed.mode != old_mode:
            self._update_hint(parsed.mode)

        if parsed.mode == "command":
            self._populate_command_results(parsed.query)
        elif parsed.mode == "api":
            self._hide_results()
        elif parsed.mode == "semantic":
            self._hide_results()
            self.post_message(self.LocalSearch(f"{PREFIX_SEMANTIC} {parsed.query}".rstrip()))
        else:
            self._hide_results()
            self.post_message(self.LocalSearch(parsed.query))

    @on(Input.Submitted, "#omni-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in the omni input."""
        parsed = parse_omni_mode(event.value)
        if parsed.mode == "api" and parsed.query.strip():
            self.post_message(self.ApiSearch(parsed.query.strip()))
        elif parsed.mode == "command":
            self._select_highlighted_command()
        elif parsed.mode == "semantic":
            self.post_message(
                self.LocalSearchSubmitted(f"{PREFIX_SEMANTIC} {parsed.query}".rstrip())
            )
        elif parsed.mode == "local":
            self.post_message(self.LocalSearchSubmitted(parsed.query))

    @on(OptionList.OptionSelected, "#omni-results")
    def _on_result_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle command selection from the results list."""
        command_index = self._command_option_indexes.get(event.option_index)
        if command_index is not None:
            cmd = self._filtered_commands[command_index]
            if cmd.enabled:
                self.post_message(self.CommandSelected(cmd.action))

    def _select_highlighted_command(self) -> None:
        """Select the currently highlighted command in the results list."""
        results = self.query_one("#omni-results", OptionList)
        idx = results.highlighted
        if idx is None:
            idx = self._first_enabled_command_option_index()
        command_index = self._command_option_indexes.get(idx) if idx is not None else None
        if command_index is not None:
            cmd = self._filtered_commands[command_index]
            if cmd.enabled:
                self.post_message(self.CommandSelected(cmd.action))

    def _first_enabled_command_option_index(self) -> int | None:
        """Return the first selectable option index in command mode."""
        for option_index, command_index in sorted(self._command_option_indexes.items()):
            if self._filtered_commands[command_index].enabled:
                return option_index
        return None


def _palette_cell(text: str, max_len: int, width: int) -> str:
    """Return width-stable text for one command result column."""
    return truncate_palette_text(text, max_len).ljust(width)


def _command_group_header(group: str) -> Option:
    """Return a non-selectable command group header row."""
    safe_group = escape_rich_text(truncate_palette_text(group, PALETTE_KEY_MAX_LEN))
    return Option(f"[dim]{safe_group}[/]", disabled=True)


def _command_match_sort_key(match: CommandMatch) -> tuple[int, int, float]:
    score, enabled, suggested, _command = match
    return enabled, suggested, score


def _mode_safe_hint(text: str) -> str:
    from arxiv_browser._ascii import is_ascii_mode

    if not is_ascii_mode():
        return text
    return text.replace("·", "-").replace("↑↓", "^v")


def _hint_for_mode(mode: str) -> str:
    if mode == "command":
        return _mode_safe_hint(OMNI_HINT_COMMAND)
    if mode == "api":
        return _mode_safe_hint(OMNI_HINT_API)
    if mode == "semantic":
        return _mode_safe_hint(OMNI_HINT_SEMANTIC)
    return _mode_safe_hint(OMNI_HINT_LOCAL)


__all__ = [
    "OMNI_HINT_API",
    "OMNI_HINT_COMMAND",
    "OMNI_HINT_LOCAL",
    "OMNI_HINT_SEMANTIC",
    "OMNI_PLACEHOLDER",
    "OmniInput",
    "OmniMode",
    "parse_omni_mode",
]
