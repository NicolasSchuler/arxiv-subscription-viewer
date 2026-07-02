"""Version-bump 'What's New' overlay.

Surfaces the headline changes from the current release. The modal is
shown automatically when ``UserConfig.last_seen_whats_new`` differs
from :data:`arxiv_browser.whats_new.WHATS_NEW_VERSION`, and can also
be opened on demand via ``F1``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Label, Static

from arxiv_browser._ascii import is_ascii_mode
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
    #whats-new-dialog {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        /* The dialog itself does not scroll — title, version, and footer stay
           pinned; only the release-notes body between them scrolls. */
    }

    #whats-new-title {
        color: $th-accent-alt;
        text-align: center;
    }

    #whats-new-version {
        text-align: center;
        color: $th-muted;
        margin-bottom: 1;
    }

    #whats-new-scroll {
        height: 1fr;
    }

    #whats-new-footer {
        text-align: center;
        margin-top: 1;
        text-style: italic;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield a dialog rendering the release highlights with a pinned footer."""
        with Vertical(id="whats-new-dialog", classes="modal-dialog"):
            yield Label(WHATS_NEW_HEADLINE, id="whats-new-title", classes="modal-title")
            yield Label(f"Version {WHATS_NEW_VERSION}", id="whats-new-version")
            with VerticalScroll(id="whats-new-scroll"):
                yield Static(id="whats-new-content")
            yield Label(
                "Close: Enter / Esc / q",
                id="whats-new-footer",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        """Render themed bullet list for the current release entries."""
        colors = theme_colors_for(self)
        accent = colors["accent"]
        bullet = "-" if is_ascii_mode() else "•"

        lines: list[str] = []
        for title, description in WHATS_NEW_ENTRIES:
            lines.append(f"[{accent}]{bullet} {title}[/]")
            lines.append(f"  {description}")
            lines.append("")

        content = self.query_one("#whats-new-content", Static)
        content.update("\n".join(lines).rstrip())

    def action_dismiss_whats_new(self) -> None:
        """Close the What's New screen."""
        self.dismiss(None)


__all__ = ["WhatsNewScreen"]
