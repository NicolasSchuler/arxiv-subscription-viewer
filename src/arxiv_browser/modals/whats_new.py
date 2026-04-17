"""Version-bump 'What's New' overlay.

Surfaces the headline changes from the current release. The modal is
shown automatically when ``UserConfig.last_seen_whats_new`` differs
from :data:`arxiv_browser.whats_new.WHATS_NEW_VERSION`, and can also
be opened on demand via ``F1``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Label, Static

from arxiv_browser.modals.base import ModalBase
from arxiv_browser.themes import theme_colors_for
from arxiv_browser.whats_new import (
    WHATS_NEW_ENTRIES,
    WHATS_NEW_HEADLINE,
    WHATS_NEW_VERSION,
)


class WhatsNewScreen(ModalBase[None]):
    """Modal listing the headline changes shipped with the current release."""

    BINDINGS = [
        Binding("escape", "dismiss_whats_new", "Close", show=False),
        Binding("enter", "dismiss_whats_new", "Close", show=False),
        Binding("space", "dismiss_whats_new", "Close", show=False),
        Binding("q", "dismiss_whats_new", "Close", show=False),
    ]

    CSS = """
    WhatsNewScreen {
        align: center middle;
    }

    #whats-new-dialog {
        width: 70;
        max-height: 80%;
        background: $th-background;
        border: tall $th-accent;
        padding: 1 2;
    }

    #whats-new-title {
        text-style: bold;
        color: $th-accent-alt;
        text-align: center;
        margin-bottom: 1;
    }

    #whats-new-version {
        text-align: center;
        color: $th-muted;
        margin-bottom: 1;
    }

    #whats-new-footer {
        text-align: center;
        color: $th-muted;
        margin-top: 1;
        text-style: italic;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield a scrolling dialog rendering the release highlights."""
        with VerticalScroll(id="whats-new-dialog"):
            yield Label(WHATS_NEW_HEADLINE, id="whats-new-title")
            yield Label(f"Version {WHATS_NEW_VERSION}", id="whats-new-version")
            yield Static(id="whats-new-content")
            yield Label(
                "Press Esc / Enter / Space to close",
                id="whats-new-footer",
            )

    def on_mount(self) -> None:
        """Render themed bullet list for the current release entries."""
        colors = theme_colors_for(self)
        accent = colors["accent"]

        lines: list[str] = []
        for title, description in WHATS_NEW_ENTRIES:
            lines.append(f"[{accent}]• {title}[/]")
            lines.append(f"  {description}")
            lines.append("")

        content = self.query_one("#whats-new-content", Static)
        content.update("\n".join(lines).rstrip())

    def action_dismiss_whats_new(self) -> None:
        """Close the What's New screen."""
        self.dismiss(None)


__all__ = ["WhatsNewScreen"]
