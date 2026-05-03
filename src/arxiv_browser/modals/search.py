"""Search and command palette modals."""

from __future__ import annotations

import logging

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from arxiv_browser.fuzzy import partial_fuzzy_score
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import ArxivSearchRequest
from arxiv_browser.palette import (
    PALETTE_DESC_MAX_LEN,
    PALETTE_KEY_MAX_LEN,
    PALETTE_NAME_MAX_LEN,
    PaletteCommand,
    truncate_palette_text,
)
from arxiv_browser.parsing import ARXIV_QUERY_FIELDS, build_arxiv_search_query
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)

_truncate_palette_text = truncate_palette_text
PALETTE_FUZZY_THRESHOLD = 40
PaletteMatch = tuple[float, int, int, PaletteCommand]


class ArxivSearchModal(ModalBase[ArxivSearchRequest | None]):
    """Modal dialog for searching the full arXiv API.

    .. deprecated::
        Replaced by :class:`~arxiv_browser.widgets.omni_input.OmniInput` with
        ``@`` prefix mode.  Retained for backward compatibility; will be removed
        in a future release.
    """

    BINDINGS = [
        Binding("enter", "search", "Search"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ArxivSearchModal {
        align: center middle;
    }

    #arxiv-search-dialog {
        width: 70;
        height: auto;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #arxiv-search-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #arxiv-search-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #arxiv-search-query,
    #arxiv-search-field,
    #arxiv-search-category {
        width: 100%;
        background: $th-panel;
        border: none;
        margin-bottom: 1;
    }

    #arxiv-search-query:focus,
    #arxiv-search-field:focus,
    #arxiv-search-category:focus {
        border-left: tall $th-accent;
    }

    #arxiv-search-buttons {
        height: auto;
        align: right middle;
    }

    #arxiv-search-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        initial_query: str = "",
        initial_field: str = "all",
        initial_category: str = "",
    ) -> None:
        """Initialize the search modal with optional pre-filled field values."""
        super().__init__()
        self._initial_query = initial_query
        self._initial_field = initial_field if initial_field in ARXIV_QUERY_FIELDS else "all"
        self._initial_category = initial_category

    def compose(self) -> ComposeResult:
        """Yield the search dialog with query input, field selector, category filter, and buttons."""
        with Vertical(id="arxiv-search-dialog"):
            yield Label("Search All arXiv", id="arxiv-search-title")
            yield Label(
                "Query all arXiv by field, with optional category filter.",
                id="arxiv-search-help",
            )
            yield Input(
                value=self._initial_query,
                placeholder="Search query (e.g., diffusion transformers)",
                id="arxiv-search-query",
            )
            yield Select(
                [
                    ("All fields", "all"),
                    ("Title", "title"),
                    ("Author", "author"),
                    ("Abstract", "abstract"),
                ],
                id="arxiv-search-field",
            )
            yield Input(
                value=self._initial_category,
                placeholder="Optional category (e.g., cs.AI)",
                id="arxiv-search-category",
            )
            with Horizontal(id="arxiv-search-buttons"):
                yield Button("Cancel (Esc)", variant="default", id="arxiv-cancel")
                yield Button("Search (Enter)", variant="primary", id="arxiv-search")

    def on_mount(self) -> None:
        """Set the initial field selection and focus the query input."""
        self.query_one("#arxiv-search-field", Select).value = self._initial_field
        self._focus_widget("#arxiv-search-query")

    def action_search(self) -> None:
        """Validate the query and dismiss the modal with an ArxivSearchRequest."""
        query = self.query_one("#arxiv-search-query", Input).value.strip()
        category = self.query_one("#arxiv-search-category", Input).value.strip()
        field_value = self.query_one("#arxiv-search-field", Select).value
        field = field_value if isinstance(field_value, str) else "all"

        try:
            build_arxiv_search_query(query, field, category)
        except ValueError as exc:
            self.notify(str(exc), title="arXiv Search", severity="warning")
            return

        self.dismiss(ArxivSearchRequest(query=query, field=field, category=category))

    @on(Button.Pressed, "#arxiv-search")
    def on_search_pressed(self) -> None:
        """Handle the Search button press by triggering the search action."""
        self.action_search()

    @on(Button.Pressed, "#arxiv-cancel")
    def on_cancel_pressed(self) -> None:
        """Handle the Cancel button press by dismissing the modal."""
        self.action_cancel()

    @on(Input.Submitted, "#arxiv-search-query")
    def on_query_submitted(self) -> None:
        """Handle Enter key in the query input by triggering the search action."""
        self.action_search()

    @on(Input.Submitted, "#arxiv-search-category")
    def on_category_submitted(self) -> None:
        """Handle Enter key in the category input by triggering the search action."""
        self.action_search()


class CommandPaletteModal(ModalBase[str]):
    """Fuzzy-searchable command palette for discovering and executing actions.

    .. deprecated::
        Replaced by :class:`~arxiv_browser.widgets.omni_input.OmniInput` with
        ``>`` prefix mode.  Retained for backward compatibility; will be removed
        in a future release.

    Parameters
    ----------
    commands:
        List of ``(name, description, key_hint, action)`` tuples.  The caller
        must supply this because the canonical command list lives in
        ``app.py`` and this module cannot import from ``app.py`` (DAG
        constraint).
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("q", "cancel", "Close"),
    ]

    DEFAULT_CSS = """
    CommandPaletteModal {
        align: center middle;
    }

    CommandPaletteModal > Vertical {
        width: 70;
        max-height: 28;
        background: $th-panel;
        border: thick $th-accent;
        padding: 0 2;
    }

    CommandPaletteModal #palette-title {
        margin-bottom: 1;
    }

    CommandPaletteModal #palette-search {
        margin-bottom: 1;
    }

    CommandPaletteModal #palette-results {
        height: 1fr;
    }

    CommandPaletteModal #palette-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, commands: list[PaletteCommand]) -> None:
        """Initialize the command palette with the given list of available commands."""
        super().__init__()
        self._commands = commands
        self._filtered: list[PaletteCommand] = list(self._commands)

    def compose(self) -> ComposeResult:
        """Yield the palette UI with a search input, scrollable results list, and footer hints."""
        from arxiv_browser._ascii import is_ascii_mode

        arrows = "^v" if is_ascii_mode() else "\u2191\u2193"
        accent = theme_colors_for(self)["accent"]
        with Vertical():
            yield Label(f"[bold {accent}]Command palette[/]", id="palette-title")
            yield Input(
                placeholder="Type to search command palette actions...",
                id="palette-search",
            )
            yield OptionList(id="palette-results")
            yield Static(
                f"{arrows} move   Enter run   Esc/q close   ? shortcuts", id="palette-footer"
            )

    def on_mount(self) -> None:
        """Populate the initial unfiltered results and focus the search input."""
        self._populate_results("")
        self._focus_widget("#palette-search")

    @on(Input.Changed, "#palette-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Re-filter the command list whenever the search input text changes."""
        self._populate_results(event.value.strip())

    def _populate_results(self, query: str) -> None:
        """Populate the results list, optionally filtered by fuzzy query."""
        option_list = self.query_one("#palette-results", OptionList)
        option_list.clear_options()
        self._filtered = self._filter_commands(query)

        if not self._filtered:
            option_list.add_option(self._empty_results_option(query))
            return

        if not query:
            self._populate_grouped_results(option_list)
            self._highlight_first_enabled(option_list)
            return

        self._add_command_options(option_list, self._filtered)
        self._highlight_first_enabled(option_list)

    def _filter_commands(self, query: str) -> list[PaletteCommand]:
        if not query:
            return list(self._commands)

        normalized_query = query.lower()
        scored = [
            match
            for command in self._commands
            if (match := _palette_match(normalized_query, command))
        ]
        scored.sort(key=_palette_match_sort_key, reverse=True)
        return [command for _, _, _, command in scored]

    def _empty_results_option(self, query: str) -> Option:
        if query:
            safe_query = escape_rich_text(query)
            return Option(
                "[dim]No commands match "
                f'[bold]"{safe_query}"[/bold].[/]\n'
                "[dim]Try: use a shorter term.[/]\n"
                "[dim]Next: press [bold]Esc[/bold]/[bold]q[/bold] to close or [bold]?[/bold] for shortcuts.[/]",
                disabled=True,
            )
        return Option(
            "[dim]No commands available.[/]\n"
            "[dim]Try: reopen with [bold]Ctrl+p[/bold].[/]\n"
            "[dim]Next: press [bold]?[/bold] for shortcuts.[/]",
            disabled=True,
        )

    def _populate_grouped_results(self, option_list: OptionList) -> None:
        suggested = [command for command in self._filtered if command.suggested]
        remaining = [command for command in self._filtered if not command.suggested]
        self._add_command_group(option_list, "Suggested now", suggested)
        self._add_command_group(option_list, "All commands", remaining)

    def _add_command_group(
        self,
        option_list: OptionList,
        heading: str,
        commands: list[PaletteCommand],
    ) -> None:
        if not commands:
            return
        option_list.add_option(Option(f"[bold]{heading}[/]", disabled=True))
        self._add_command_options(option_list, commands)

    def _add_command_options(
        self,
        option_list: OptionList,
        commands: list[PaletteCommand],
    ) -> None:
        for command in commands:
            option_list.add_option(self._build_command_option(command))

    def _build_command_option(self, command: PaletteCommand) -> Option:
        """Render one command palette row."""
        colors = theme_colors_for(self)
        accent = colors["accent"]
        muted = colors["muted"]
        safe_name = escape_rich_text(_truncate_palette_text(command.name, PALETTE_NAME_MAX_LEN))
        safe_desc = escape_rich_text(
            _truncate_palette_text(command.description, PALETTE_DESC_MAX_LEN)
        )
        safe_key = escape_rich_text(_truncate_palette_text(command.key_hint, PALETTE_KEY_MAX_LEN))
        safe_group = escape_rich_text(command.group)
        if command.enabled:
            meta = f"[{muted}]{safe_group}[/]"
            if safe_key:
                meta += f"  [{accent}]{safe_key}[/]"
        else:
            blocked = escape_rich_text(command.blocked_reason or "Unavailable")
            meta = f"[{muted}]Requires: {blocked}[/]"
        markup = f"[bold]{safe_name}[/]  [{muted}]{safe_desc}[/]\n  {meta}"
        return Option(markup, id=command.action, disabled=not command.enabled)

    @staticmethod
    def _highlight_first_enabled(option_list: OptionList) -> None:
        """Move the cursor to the first selectable command."""
        for idx in range(option_list.option_count):
            if not option_list.get_option_at_index(idx).disabled:
                option_list.highlighted = idx
                return
        option_list.highlighted = None

    @on(OptionList.OptionSelected, "#palette-results")
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection of a command option by dismissing with its action string."""
        if event.option_id is not None:
            self.dismiss(str(event.option_id))

    def action_cancel(self) -> None:
        """Dismiss the command palette without executing any command."""
        self.dismiss("")

    def key_enter(self) -> None:
        """Execute the currently highlighted command."""
        option_list = self.query_one("#palette-results", OptionList)
        idx = option_list.highlighted
        if idx is None or idx < 0:
            return
        option = option_list.get_option_at_index(idx)
        if option.disabled or option.id is None:
            return
        self.dismiss(str(option.id))


def _palette_match(query: str, command: PaletteCommand) -> PaletteMatch | None:
    score = max(
        partial_fuzzy_score(query, command.name),
        partial_fuzzy_score(query, command.description),
    )
    if score < PALETTE_FUZZY_THRESHOLD:
        return None
    return score, int(command.enabled), int(command.suggested), command


def _palette_match_sort_key(match: PaletteMatch) -> tuple[int, int, float]:
    score, enabled, suggested, _command = match
    return enabled, suggested, score
