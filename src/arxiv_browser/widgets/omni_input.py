"""Unified OmniInput — VS Code-style search with mode prefixes.

Modes:
- No prefix  → local paper search (default)
- ``>``      → command palette (inline results)
- ``@``      → arXiv API search (triggers on Enter)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, OptionList, Static

from arxiv_browser.fuzzy import partial_fuzzy_score
from arxiv_browser.modals.search import (
    PALETTE_DESC_MAX_LEN,
    PALETTE_KEY_MAX_LEN,
    PALETTE_NAME_MAX_LEN,
    PaletteCommand,
    _truncate_palette_text,
)
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)

PREFIX_COMMAND = ">"
PREFIX_API = "@"

OMNI_PLACEHOLDER = "Search papers · > commands · @ arXiv API"
OMNI_HINT_LOCAL = 'Examples: cat:cs.AI  author:hinton  unread  "large language"'
OMNI_HINT_COMMAND = "↑↓ move · Enter run · Esc close"
OMNI_HINT_API = "Enter to search arXiv · Esc cancel"

FUZZY_THRESHOLD = 40


@dataclass(slots=True, frozen=True)
class OmniMode:
    """Parsed state of the OmniInput prefix and query."""

    mode: str  # "local", "command", "api"
    query: str  # text after prefix (stripped)


def parse_omni_mode(raw: str) -> OmniMode:
    """Parse raw input text into an OmniMode."""
    if raw.startswith(PREFIX_COMMAND):
        return OmniMode(mode="command", query=raw[1:].lstrip())
    if raw.startswith(PREFIX_API):
        return OmniMode(mode="api", query=raw[1:].lstrip())
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
        color: $text-muted;
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
        self._current_mode: str = "local"

    def compose(self) -> ComposeResult:
        """Build the omni input: text field, hint line, and results panel."""
        yield Input(
            placeholder=OMNI_PLACEHOLDER, id="omni-input", disabled=True, select_on_focus=False
        )
        yield Static(OMNI_HINT_LOCAL, id="omni-hint")
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

    def _update_hint(self, mode: str) -> None:
        hint = self.query_one("#omni-hint", Static)
        if mode == "command":
            hint.update(OMNI_HINT_COMMAND)
        elif mode == "api":
            hint.update(OMNI_HINT_API)
        else:
            hint.update(OMNI_HINT_LOCAL)

    def _populate_command_results(self, query: str) -> None:
        """Populate the inline results list with matching commands."""
        results = self.query_one("#omni-results", OptionList)
        results.clear_options()

        if query:
            q = query.lower()
            scored: list[tuple[float, int, int, PaletteCommand]] = []
            for cmd in self._commands:
                score = max(
                    partial_fuzzy_score(q, cmd.name),
                    partial_fuzzy_score(q, cmd.description),
                )
                if score >= FUZZY_THRESHOLD:
                    scored.append((score, int(cmd.enabled), int(cmd.suggested), cmd))
            scored.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
            self._filtered_commands = [cmd for _, _, _, cmd in scored]
        else:
            self._filtered_commands = list(self._commands)

        if not self._filtered_commands:
            if query:
                safe = escape_rich_text(query)
                from textual.widgets.option_list import Option

                results.add_option(Option(f'[dim]No matches for "{safe}"[/]', disabled=True))
            self._show_results()
            return

        colors = theme_colors_for(self)
        green = colors["green"]
        accent = colors["accent"]
        muted = colors["muted"]

        from textual.widgets.option_list import Option

        for cmd in self._filtered_commands:
            name = _truncate_palette_text(cmd.name, PALETTE_NAME_MAX_LEN)
            desc = _truncate_palette_text(cmd.description, PALETTE_DESC_MAX_LEN)
            hint = _truncate_palette_text(cmd.key_hint, PALETTE_KEY_MAX_LEN) if cmd.key_hint else ""
            safe_name = escape_rich_text(name)
            safe_desc = escape_rich_text(desc)
            safe_hint = escape_rich_text(hint)

            if not cmd.enabled:
                label = f"[dim]{safe_name}  {safe_desc}[/]"
            else:
                parts = [f"[bold {accent}]{safe_name}[/]", f"[{muted}]{safe_desc}[/]"]
                if safe_hint:
                    parts.append(f"[{green}]{safe_hint}[/]")
                label = "  ".join(parts)

            results.add_option(Option(label, disabled=not cmd.enabled))

        self._show_results()

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
        elif parsed.mode == "local":
            self.post_message(self.LocalSearchSubmitted(parsed.query))

    @on(OptionList.OptionSelected, "#omni-results")
    def _on_result_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle command selection from the results list."""
        idx = event.option_index
        if 0 <= idx < len(self._filtered_commands):
            cmd = self._filtered_commands[idx]
            if cmd.enabled:
                self.post_message(self.CommandSelected(cmd.action))

    def _select_highlighted_command(self) -> None:
        """Select the currently highlighted command in the results list."""
        results = self.query_one("#omni-results", OptionList)
        idx = results.highlighted
        if idx is None and self._filtered_commands:
            idx = 0
        if idx is not None and 0 <= idx < len(self._filtered_commands):
            cmd = self._filtered_commands[idx]
            if cmd.enabled:
                self.post_message(self.CommandSelected(cmd.action))


__all__ = [
    "OMNI_HINT_API",
    "OMNI_HINT_COMMAND",
    "OMNI_HINT_LOCAL",
    "OMNI_PLACEHOLDER",
    "OmniInput",
    "OmniMode",
    "parse_omni_mode",
]
