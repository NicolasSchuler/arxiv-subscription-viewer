"""Search and command palette modals."""

from __future__ import annotations

import logging

from rapidfuzz import fuzz
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from arxiv_browser.models import ArxivSearchRequest
from arxiv_browser.parsing import ARXIV_QUERY_FIELDS, build_arxiv_search_query
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)


class ArxivSearchModal(ModalScreen[ArxivSearchRequest | None]):
    """Modal dialog for searching the full arXiv API."""

    BINDINGS = [
        Binding("enter", "search", "Search"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ArxivSearchModal {
        align: center middle;
    }

    #arxiv-search-dialog {
        width: 70%;
        height: auto;
        min-width: 60;
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
        super().__init__()
        self._initial_query = initial_query
        self._initial_field = initial_field if initial_field in ARXIV_QUERY_FIELDS else "all"
        self._initial_category = initial_category

    def compose(self) -> ComposeResult:
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
        self.query_one("#arxiv-search-field", Select).value = self._initial_field
        self.query_one("#arxiv-search-query", Input).focus()

    def action_search(self) -> None:
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

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#arxiv-search")
    def on_search_pressed(self) -> None:
        self.action_search()

    @on(Button.Pressed, "#arxiv-cancel")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    @on(Input.Submitted, "#arxiv-search-query")
    def on_query_submitted(self) -> None:
        self.action_search()

    @on(Input.Submitted, "#arxiv-search-category")
    def on_category_submitted(self) -> None:
        self.action_search()


class CommandPaletteModal(ModalScreen[str]):
    """Fuzzy-searchable command palette for discovering and executing actions.

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
        padding: 1 2;
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

    def __init__(self, commands: list[tuple[str, str, str, str]]) -> None:
        super().__init__()
        self._commands = commands
        self._filtered: list[tuple[str, str, str, str]] = list(self._commands)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"[bold {THEME_COLORS['accent']}]Command palette[/]")
            yield Input(
                placeholder="Type to search command palette actions...",
                id="palette-search",
            )
            yield OptionList(id="palette-results")
            yield Static("Close: Esc", id="palette-footer")

    def on_mount(self) -> None:
        self._populate_results("")
        self.query_one("#palette-search", Input).focus()

    @on(Input.Changed, "#palette-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._populate_results(event.value.strip())

    def _populate_results(self, query: str) -> None:
        """Populate the results list, optionally filtered by fuzzy query."""
        option_list = self.query_one("#palette-results", OptionList)
        option_list.clear_options()

        if query:
            q = query.lower()
            scored: list[tuple[float, tuple[str, str, str, str]]] = []
            for cmd in self._commands:
                name, desc, _, _ = cmd
                score = max(
                    fuzz.partial_ratio(q, name.lower()),
                    fuzz.partial_ratio(q, desc.lower()),
                )
                if score >= 40:
                    scored.append((score, cmd))
            scored.sort(key=lambda x: x[0], reverse=True)
            self._filtered = [cmd for _, cmd in scored]
        else:
            self._filtered = list(self._commands)

        if not self._filtered:
            if query:
                safe_query = escape_rich_text(query)
                option_list.add_option(
                    Option(
                        "[dim]No commands match "
                        f'[bold]"{safe_query}"[/bold].[/]\n'
                        "[dim]Try: use a shorter term.[/]\n"
                        "[dim]Next: press [bold]Esc[/bold] to close or [bold]?[/bold] for shortcuts.[/]",
                        disabled=True,
                    )
                )
            else:
                option_list.add_option(
                    Option(
                        "[dim]No commands available.[/]\n"
                        "[dim]Try: reopen with [bold]Ctrl+p[/bold].[/]\n"
                        "[dim]Next: press [bold]?[/bold] for shortcuts.[/]",
                        disabled=True,
                    )
                )
            return

        accent = THEME_COLORS["accent"]
        muted = THEME_COLORS["muted"]
        for name, desc, key_hint, action in self._filtered:
            safe_name = escape_rich_text(name)
            safe_desc = escape_rich_text(desc)
            safe_key = escape_rich_text(key_hint)
            markup = f"[bold]{safe_name}[/]  [{muted}]{safe_desc}[/]\n  [{accent}]{safe_key}[/]"
            option_list.add_option(Option(markup, id=action))

        if option_list.option_count > 0:
            option_list.highlighted = 0

    @on(OptionList.OptionSelected, "#palette-results")
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self.dismiss(str(event.option_id))

    def action_cancel(self) -> None:
        self.dismiss("")

    def key_enter(self) -> None:
        """Execute the currently highlighted command."""
        option_list = self.query_one("#palette-results", OptionList)
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self._filtered):
            _, _, _, action = self._filtered[idx]
            self.dismiss(action)
