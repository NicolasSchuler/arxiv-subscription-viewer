"""Full-screen help overlay modal.

HelpScreen — standalone keyboard-shortcut reference extracted from common.py.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

from arxiv_browser.themes import theme_colors_for


class HelpScreen(ModalScreen[None]):
    """Full-screen help overlay showing all keyboard shortcuts by category."""

    _DEFAULT_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Getting Started",
            [
                ("/", "Search papers"),
                ("A", "Search all arXiv"),
                ("j / k", "Move selection"),
                ("Space", "Select current paper"),
                ("o", "Open selected paper(s)"),
                ("Ctrl+p", "Open commands"),
                ("?", "Show full shortcuts"),
            ],
        ),
        (
            "Search Syntax",
            [
                ("cat:cs.AI", "Category filter"),
                ("author:hinton", "Author filter"),
                ("unread / starred", "State filters"),
                ("AND / OR / NOT", "Boolean operators"),
            ],
        ),
        (
            "Core Actions",
            [
                ("Space", "Toggle selection"),
                ("a", "Select all visible"),
                ("u", "Clear selection"),
                ("o", "Open in Browser"),
                ("P", "Open as PDF"),
                ("E", "Export menu"),
                ("d", "Download PDFs"),
                ("v", "Toggle detail density (scan/full)"),
                ("?", "Help overlay"),
            ],
        ),
        (
            "Standard · Organize",
            [
                ("r", "Toggle read"),
                ("x", "Toggle star"),
                ("n", "Edit notes"),
                ("t", "Edit tags"),
                ("Ctrl+b", "Save search as bookmark"),
            ],
        ),
    ]

    BINDINGS = [
        Binding("question_mark", "dismiss", "Close", show=False),
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close", show=False),
    ]

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
        overflow-y: auto;
    }

    #help-title {
        text-style: bold;
        color: $th-accent-alt;
        text-align: center;
        margin-bottom: 1;
    }

    #help-filter {
        margin-bottom: 1;
    }

    #help-sections {
        height: auto;
    }

    .help-section {
        margin-bottom: 1;
    }

    .help-section-title {
        text-style: bold;
        margin-bottom: 0;
    }

    .help-keys {
        padding-left: 2;
        color: $th-text;
    }

    #help-no-matches {
        text-align: center;
        color: $th-muted;
        margin: 2 0;
    }

    #help-footer {
        text-align: center;
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        sections: list[tuple[str, list[tuple[str, str]]]] | None = None,
        footer_note: str = "Close: ? / Esc / q",
    ) -> None:
        """Initialise the help screen with optional custom sections and footer."""
        super().__init__()
        raw = sections or list(self._DEFAULT_SECTIONS)
        # Replace middle-dot separator with ASCII dash when in ASCII mode
        from arxiv_browser._ascii import is_ascii_mode

        if is_ascii_mode():
            raw = [(name.replace("\u00b7", "-"), entries) for name, entries in raw]
        self._sections = raw
        self._footer_note = footer_note

    def _render_section_lines(self, entries: list[tuple[str, str]]) -> str:
        """Render key-description pairs as theme-coloured markup lines."""
        green = theme_colors_for(self)["green"]
        lines = [f"  [{green}]{key}[/]  {description}" for key, description in entries]
        return "\n".join(lines)

    def _filter_sections(self, query: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """Return sections filtered to entries matching the query (case-insensitive)."""
        if not query:
            return self._sections
        q = query.lower()
        filtered: list[tuple[str, list[tuple[str, str]]]] = []
        for section_name, entries in self._sections:
            matching = [
                (key, desc) for key, desc in entries if q in key.lower() or q in desc.lower()
            ]
            if matching:
                filtered.append((section_name, matching))
        return filtered

    def compose(self) -> ComposeResult:
        """Yield a scrollable dialog with a filter input and categorised shortcut tables."""
        with VerticalScroll(id="help-dialog"):
            yield Label("Keyboard Shortcuts", id="help-title")
            yield Input(
                placeholder="Filter help... (type to search)",
                id="help-filter",
            )
            yield Vertical(id="help-sections")
            yield Label(self._footer_note, id="help-footer")

    async def on_mount(self) -> None:
        """Populate help sections and focus the filter input."""
        await self._populate_sections("")
        self.query_one("#help-filter", Input).focus()

    @on(Input.Changed, "#help-filter")
    async def _on_filter_changed(self, event: Input.Changed) -> None:
        """Re-filter help sections when the filter input changes."""
        await self._populate_sections(event.value.strip())

    async def _populate_sections(self, query: str) -> None:
        """Rebuild the help section widgets based on the current filter query."""
        container = self.query_one("#help-sections", Vertical)
        await container.remove_children()

        filtered = self._filter_sections(query)

        if not filtered and query:
            safe = query.replace("[", "\\[")
            await container.mount(
                Static(
                    f'[dim]No matches for [bold]"{safe}"[/bold][/]',
                    id="help-no-matches",
                )
            )
            return

        widgets: list[Label | Static] = []
        colors = theme_colors_for(self)
        for section_name, entries in filtered:
            widgets.append(
                Label(
                    f"[{colors['accent']}]{section_name}[/]",
                    classes="help-section-title",
                )
            )
            widgets.append(Static(self._render_section_lines(entries), classes="help-keys"))
        if widgets:
            await container.mount(*widgets)

    def action_dismiss(self) -> None:
        """Close the help screen."""
        self.dismiss(None)
