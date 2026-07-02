"""First-run welcome overlay showing essential keybindings."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Label, Static

from arxiv_browser._ascii import is_ascii_mode
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.themes import theme_colors_for


class WelcomeScreen(ModalBase[None]):
    """First-run tutorial overlay showing core keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss_welcome", "Close", show=False),
        Binding("enter", "dismiss_welcome", "Close", show=False),
        Binding("space", "dismiss_welcome", "Close", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
    ]

    CSS = """
    #welcome-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        /* The dialog itself does not scroll — title, subtitle, and footer stay
           pinned; only the content between them scrolls when space is tight. */
    }

    #welcome-title {
        color: $th-accent-alt;
        text-align: center;
    }

    #welcome-subtitle {
        text-align: center;
        color: $th-muted;
        margin-bottom: 1;
    }

    #welcome-scroll {
        height: 1fr;
    }

    .welcome-keys {
        padding-left: 2;
        color: $th-text;
    }

    #welcome-footer {
        text-align: center;
        margin-top: 1;
        text-style: italic;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield a focused welcome dialog with essential keybindings."""
        with Vertical(id="welcome-dialog", classes="modal-dialog"):
            yield Label("Welcome to arXiv Viewer", id="welcome-title", classes="modal-title")
            yield Label(
                "Here are the essential shortcuts to get started:",
                id="welcome-subtitle",
            )
            with VerticalScroll(id="welcome-scroll"):
                yield Static(id="welcome-content")
            yield Label(
                self._footer_text(),
                id="welcome-footer",
                classes="modal-footer",
            )

    @staticmethod
    def _footer_text() -> str:
        """Return a compact, width-safe dismiss hint (ASCII-aware separator)."""
        sep = "-" if is_ascii_mode() else "·"
        return f"Enter / Esc: start {sep} ?: full help"

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
                ],
            ),
            (
                "Search",
                [
                    ("/", "Search and filter papers"),
                    ("A", "Search all arXiv"),
                    ("Ctrl+p", "Open command palette"),
                ],
            ),
            (
                "Actions",
                [
                    ("Space", "Select paper"),
                    ("o", "Open in browser"),
                    ("r / x", "Toggle read / star"),
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
        for index, (section_name, entries) in enumerate(sections):
            if index:
                lines.append("")
            lines.append(f"[{accent}]{section_name}[/]")
            for key, desc in entries:
                lines.append(f"  [{green}]{key:<12}[/]  {desc}")

        content = self.query_one("#welcome-content", Static)
        content.update("\n".join(lines))

    def action_dismiss_welcome(self) -> None:
        """Close the welcome screen."""
        self.dismiss(None)

    def action_show_help(self) -> None:
        """Close onboarding and open the full help overlay."""

        def _open_help() -> None:
            action = getattr(self.app, "action_show_help", None)
            if callable(action):
                action()

        self.dismiss(None)
        self.app.call_later(_open_help)
