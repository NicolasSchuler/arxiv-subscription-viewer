"""First-run welcome overlay showing essential keybindings."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Label, Static

from arxiv_browser.modals.base import ModalBase
from arxiv_browser.themes import theme_colors_for


class WelcomeScreen(ModalBase[None]):
    """First-run tutorial overlay showing core keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss_welcome", "Close", show=False),
        Binding("enter", "dismiss_welcome", "Close", show=False),
        Binding("space", "dismiss_welcome", "Close", show=False),
        Binding("question_mark", "dismiss_welcome", "Close", show=False),
    ]

    CSS = """
    WelcomeScreen {
        align: center middle;
    }

    #welcome-dialog {
        width: 60;
        max-height: 80%;
        background: $th-background;
        border: tall $th-accent;
        padding: 1 2;
    }

    #welcome-title {
        text-style: bold;
        color: $th-accent-alt;
        text-align: center;
        margin-bottom: 1;
    }

    #welcome-subtitle {
        text-align: center;
        color: $th-muted;
        margin-bottom: 1;
    }

    .welcome-keys {
        padding-left: 2;
        color: $th-text;
    }

    #welcome-footer {
        text-align: center;
        color: $th-muted;
        margin-top: 1;
        text-style: italic;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield a focused welcome dialog with essential keybindings."""
        with VerticalScroll(id="welcome-dialog"):
            yield Label("Welcome to arXiv Viewer", id="welcome-title")
            yield Label(
                "Here are the essential shortcuts to get started:",
                id="welcome-subtitle",
            )
            yield Static(id="welcome-content")
            yield Label(
                "Press any key to start browsing",
                id="welcome-footer",
            )

    def on_mount(self) -> None:
        """Populate the welcome content with themed keybinding hints."""
        colors = theme_colors_for(self)
        green = colors["green"]
        accent = colors["accent"]

        sections = [
            (
                "Navigate",
                [
                    ("j / k", "Move up and down"),
                    ("[ / ]", "Change dates (history mode)"),
                ],
            ),
            (
                "Search",
                [
                    ("/", "Search and filter papers"),
                    ("Ctrl+p", "Open command palette"),
                ],
            ),
            (
                "Actions",
                [
                    ("Space", "Select paper"),
                    ("o", "Open in browser"),
                    ("r / x", "Mark read / star"),
                    ("E", "Export selected papers"),
                ],
            ),
            (
                "Help",
                [
                    ("?", "Show all keyboard shortcuts"),
                ],
            ),
        ]

        lines: list[str] = []
        for section_name, entries in sections:
            lines.append(f"\n[{accent}]{section_name}[/]")
            for key, desc in entries:
                lines.append(f"  [{green}]{key:<12}[/]  {desc}")

        content = self.query_one("#welcome-content", Static)
        content.update("\n".join(lines))

    def action_dismiss_welcome(self) -> None:
        """Close the welcome screen."""
        self.dismiss(None)
