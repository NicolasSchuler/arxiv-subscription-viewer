"""Paper metadata editing modals — notes, tags, auto-tag suggestions."""

from __future__ import annotations

import logging

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, TextArea

from arxiv_browser.themes import get_tag_color, parse_tag_namespace

logger = logging.getLogger(__name__)


class NotesModal(ModalScreen[str | None]):
    """Modal dialog for editing paper notes."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    NotesModal {
        align: center middle;
    }

    #notes-dialog {
        width: 60%;
        height: 60%;
        min-width: 50;
        min-height: 15;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #notes-title {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #notes-textarea {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #notes-textarea:focus {
        border-left: tall $th-accent;
    }

    #notes-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #notes-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, arxiv_id: str, current_notes: str = "") -> None:
        super().__init__()
        self._arxiv_id = arxiv_id
        self._current_notes = current_notes

    def compose(self) -> ComposeResult:
        with Vertical(id="notes-dialog"):
            yield Label(f"Notes for {self._arxiv_id}", id="notes-title")
            yield TextArea(self._current_notes, id="notes-textarea")
            with Horizontal(id="notes-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")

    def on_mount(self) -> None:
        self.query_one("#notes-textarea", TextArea).focus()

    def action_save(self) -> None:
        text = self.query_one("#notes-textarea", TextArea).text
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


class TagsModal(ModalScreen[list[str] | None]):
    """Modal dialog for editing paper tags."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    TagsModal {
        align: center middle;
    }

    #tags-dialog {
        width: 50%;
        height: auto;
        min-width: 40;
        background: $th-background;
        border: tall $th-green;
        padding: 0 2;
    }

    #tags-title {
        text-style: bold;
        color: $th-green;
        margin-bottom: 1;
    }

    #tags-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #tags-input {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #tags-input:focus {
        border-left: tall $th-green;
    }

    #tags-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #tags-buttons Button {
        margin-left: 1;
    }

    #tags-suggestions {
        color: $th-muted;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        arxiv_id: str,
        current_tags: list[str] | None = None,
        all_tags: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._arxiv_id = arxiv_id
        self._current_tags = current_tags or []
        self._all_tags = all_tags or []

    def _build_suggestions_markup(self) -> str:
        """Build Rich markup for tag suggestions grouped by namespace."""
        if not self._all_tags:
            return ""
        # Group unique tags by namespace
        namespaced: dict[str, list[str]] = {}
        unnamespaced: list[str] = []
        for tag in sorted(set(self._all_tags)):
            ns, val = parse_tag_namespace(tag)
            if ns:
                namespaced.setdefault(ns, []).append(val)
            else:
                unnamespaced.append(val)
        parts = []
        for ns in sorted(namespaced):
            color = get_tag_color(f"{ns}:")
            vals = ", ".join(namespaced[ns])
            parts.append(f"[{color}]{ns}:[/] {vals}")
        if unnamespaced:
            parts.append(", ".join(unnamespaced))
        return " | ".join(parts)

    def compose(self) -> ComposeResult:
        with Vertical(id="tags-dialog"):
            yield Label(f"Tags for {self._arxiv_id}", id="tags-title")
            yield Label(
                "Use namespace:tag format (e.g., topic:ml, status:to-read)",
                id="tags-help",
            )
            suggestions = self._build_suggestions_markup()
            if suggestions:
                yield Label(suggestions, id="tags-suggestions")
            yield Input(
                value=", ".join(self._current_tags),
                placeholder="Enter tags...",
                id="tags-input",
            )
            with Horizontal(id="tags-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")

    def on_mount(self) -> None:
        self.query_one("#tags-input", Input).focus()

    def _parse_tags(self, text: str) -> list[str]:
        """Parse comma-separated tags, stripping whitespace."""
        return [tag.strip() for tag in text.split(",") if tag.strip()]

    def action_save(self) -> None:
        text = self.query_one("#tags-input", Input).value
        self.dismiss(self._parse_tags(text))

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    @on(Input.Submitted, "#tags-input")
    def on_input_submitted(self) -> None:
        self.action_save()


class AutoTagSuggestModal(ModalScreen[list[str] | None]):
    """Modal showing LLM-suggested tags for user to accept or modify."""

    BINDINGS = [
        Binding("ctrl+s", "accept", "Accept"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    AutoTagSuggestModal {
        align: center middle;
    }

    #autotag-dialog {
        width: 55%;
        height: auto;
        min-width: 45;
        max-height: 80%;
        background: $th-background;
        border: tall $th-green;
        padding: 0 2;
    }

    #autotag-title {
        text-style: bold;
        color: $th-green;
        margin-bottom: 1;
    }

    #autotag-current {
        color: $th-muted;
        margin-bottom: 1;
    }

    #autotag-input {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #autotag-input:focus {
        border-left: tall $th-green;
    }

    #autotag-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #autotag-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        paper_title: str,
        suggested_tags: list[str],
        current_tags: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._paper_title = paper_title
        self._suggested = suggested_tags
        self._current = current_tags or []

    def compose(self) -> ComposeResult:
        with Vertical(id="autotag-dialog"):
            yield Static(f"Auto-Tag: {self._paper_title[:60]}", id="autotag-title")
            if self._current:
                yield Static(
                    f"Current: [bold]{', '.join(self._current)}[/bold]",
                    id="autotag-current",
                )
            # Merge current + suggested, dedup
            merged = list(dict.fromkeys(self._current + self._suggested))
            yield Input(
                value=", ".join(merged),
                placeholder="Edit tags (comma-separated)",
                id="autotag-input",
            )
            with Horizontal(id="autotag-buttons"):
                yield Button("Accept [Ctrl+s]", id="accept-btn", variant="success")
                yield Button("Cancel [Esc]", id="cancel-btn")

    def action_accept(self) -> None:
        text_input = self.query_one("#autotag-input", Input)
        raw = text_input.value
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        self.dismiss(tags)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#accept-btn")
    def on_accept_pressed(self) -> None:
        self.action_accept()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()
