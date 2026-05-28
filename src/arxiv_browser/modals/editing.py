"""Unified paper metadata editing modal — notes, tags, auto-tag suggestions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Label, Static, TabbedContent, TabPane, TextArea

from arxiv_browser.modals.base import ModalBase
from arxiv_browser.themes import get_tag_color, parse_tag_namespace

logger = logging.getLogger(__name__)


@dataclass
class PaperEditResult:
    """Result from the unified paper editing modal."""

    __slots__ = ("active_tab", "notes", "tags")
    notes: str
    tags: list[str]
    active_tab: str


@dataclass
class LineAnnotationResult:
    """Result from the quick detail-line annotation modal."""

    __slots__ = ("line", "text")
    line: int
    text: str


class TagChip(Button):
    """Clickable tag suggestion chip in the paper editing modal."""

    def __init__(self, tag: str, *, selected: bool = False) -> None:
        self.tag = tag
        super().__init__(
            _format_tag_chip_label(tag, selected),
            variant="primary" if selected else "default",
            classes="tag-chip",
        )


def _format_tag_chip_label(tag: str, selected: bool) -> str:
    prefix = "[x]" if selected else "[ ]"
    return f"{prefix} {tag}"


def _chunked_tags(tags: list[str], size: int) -> list[list[str]]:
    return [tags[index : index + size] for index in range(0, len(tags), size)]


class PaperEditModal(ModalBase["PaperEditResult | None"]):
    """Unified modal for editing paper notes, tags, and AI-suggested tags.

    Combines the former NotesModal, TagsModal, and AutoTagSuggestModal into
    a single TabbedContent dialog.  Callers specify *initial_tab* to control
    which pane is shown first.
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    PaperEditModal {
        align: center middle;
    }

    #edit-dialog {
        width: 70;
        max-width: 90%;
        height: 70%;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #edit-title {
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

    #tags-input, #autotag-input {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #tags-input:focus, #autotag-input:focus {
        border-left: tall $th-green;
    }

    #tags-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #tags-suggestions,
    #tag-filter {
        margin-bottom: 1;
    }

    #tags-suggestions,
    #edit-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #tag-chip-box {
        height: auto;
        max-height: 8;
        margin-bottom: 1;
    }

    .tag-chip-row {
        height: auto;
    }

    .tag-chip {
        margin-right: 1;
        min-width: 10;
    }

    #autotag-current {
        color: $th-muted;
        margin-bottom: 1;
    }

    #edit-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #edit-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        arxiv_id: str,
        current_notes: str = "",
        current_tags: list[str] | None = None,
        all_tags: list[str] | None = None,
        suggested_tags: list[str] | None = None,
        initial_tab: str = "notes",
    ) -> None:
        """Initialize the editing modal with paper data and the tab to show first."""
        super().__init__()
        self._arxiv_id = arxiv_id
        self._current_notes = current_notes
        self._current_tags = current_tags or []
        self._all_tags = all_tags or []
        self._suggested_tags = suggested_tags
        self._initial_tab = initial_tab
        self._initial_notes = current_notes
        self._initial_tags_value = ", ".join(self._current_tags)

    def _build_suggestions_markup(self) -> str:
        """Build Rich markup for tag suggestions grouped by namespace."""
        if not self._all_tags:
            return ""
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
        """Yield the tabbed editing dialog with notes, tags, and optional AI tags."""
        with Vertical(id="edit-dialog"):
            yield Label(f"Edit: {self._arxiv_id}", id="edit-title")
            with TabbedContent(initial=self._initial_tab):
                with TabPane("Notes", id="notes"):
                    yield TextArea(self._current_notes, id="notes-textarea")
                with TabPane("Tags", id="tags"):
                    yield Label(
                        "Use namespace:tag format (e.g., topic:ml, status:to-read)",
                        id="tags-help",
                    )
                    suggestions = self._build_suggestions_markup()
                    if suggestions:
                        yield Label(suggestions, id="tags-suggestions")
                        yield Input(placeholder="Filter tag chips...", id="tag-filter")
                        with Vertical(id="tag-chip-box"):
                            for row in _chunked_tags(sorted(set(self._all_tags)), 3):
                                with Horizontal(classes="tag-chip-row"):
                                    for tag in row:
                                        yield TagChip(tag, selected=tag in self._current_tags)
                    yield Input(
                        value=", ".join(self._current_tags),
                        placeholder="Enter tags...",
                        id="tags-input",
                    )
                if self._suggested_tags:
                    with TabPane("AI Tags", id="ai-tags"):
                        if self._current_tags:
                            yield Static(
                                f"Current: [bold]{', '.join(self._current_tags)}[/bold]",
                                id="autotag-current",
                            )
                        merged = list(dict.fromkeys(self._current_tags + self._suggested_tags))
                        yield Input(
                            value=", ".join(merged),
                            placeholder="Edit tags (comma-separated)",
                            id="autotag-input",
                        )
            with Horizontal(id="edit-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")
            yield Static("[dim]Saved | Ctrl+S save | Esc cancel[/dim]", id="edit-help")

    def on_mount(self) -> None:
        """Focus the appropriate widget based on the initial tab."""
        if self._initial_tab == "notes":
            self._focus_widget("#notes-textarea")
        elif self._initial_tab == "ai-tags" and self._suggested_tags:
            self._focus_widget("#autotag-input")
        else:
            self._focus_widget("#tags-input")

    def _get_active_tab_id(self) -> str:
        """Return the ID of the currently active tab pane."""
        try:
            tc = self.query_one(TabbedContent)
            if tc.active:
                return tc.active
        except NoMatches:  # DOM may not be ready
            return self._initial_tab
        return self._initial_tab

    def _parse_tags(self, text: str) -> list[str]:
        """Parse comma-separated tags, stripping whitespace."""
        return [tag.strip() for tag in text.split(",") if tag.strip()]

    def _current_tag_input(self) -> Input:
        active = self._get_active_tab_id()
        if active == "ai-tags":
            try:
                return self.query_one("#autotag-input", Input)
            except NoMatches:
                return self.query_one("#tags-input", Input)
        return self.query_one("#tags-input", Input)

    def _set_tags_value(self, tags: list[str]) -> None:
        self._current_tag_input().value = ", ".join(tags)
        self._sync_tag_chips(tags)
        self._update_dirty_state()

    def _sync_tag_chips(self, tags: list[str] | None = None) -> None:
        selected = set(
            tags if tags is not None else self._parse_tags(self._current_tag_input().value)
        )
        for widget in self.query(".tag-chip"):
            if not isinstance(widget, TagChip):
                continue
            is_selected = widget.tag in selected
            widget.label = _format_tag_chip_label(widget.tag, is_selected)
            widget.variant = "primary" if is_selected else "default"

    def _update_dirty_state(self) -> None:
        try:
            notes = self.query_one("#notes-textarea", TextArea).text
            tags = self.query_one("#tags-input", Input).value
            dirty = notes != self._initial_notes or tags != self._initial_tags_value
            self.query_one("#edit-help", Static).update(
                "[bold]Unsaved[/bold] | Ctrl+S save | Esc cancel"
                if dirty
                else "[dim]Saved | Ctrl+S save | Esc cancel[/dim]"
            )
        except Exception:
            return

    def action_save(self) -> None:
        """Collect values from all tabs and dismiss with the result."""
        notes = self.query_one("#notes-textarea", TextArea).text
        active = self._get_active_tab_id()
        # Use autotag input when AI Tags tab is active, otherwise regular tags
        if active == "ai-tags":
            try:
                raw = self.query_one("#autotag-input", Input).value
                tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
            except Exception:
                tags = self._parse_tags(self.query_one("#tags-input", Input).value)
        else:
            tags = self._parse_tags(self.query_one("#tags-input", Input).value)
        self.dismiss(PaperEditResult(notes=notes, tags=tags, active_tab=active))

    @on(TextArea.Changed, "#notes-textarea")
    def on_notes_changed(self) -> None:
        """Refresh saved/unsaved state when note text changes."""
        self._update_dirty_state()

    @on(Input.Changed, "#tags-input")
    def on_tags_changed(self) -> None:
        """Refresh chips and saved/unsaved state when tag text changes."""
        self._sync_tag_chips()
        self._update_dirty_state()

    @on(Input.Changed, "#autotag-input")
    def on_autotag_changed(self) -> None:
        """Refresh chips when the AI-tag input changes."""
        self._sync_tag_chips()

    @on(Input.Changed, "#tag-filter")
    def on_tag_filter_changed(self, event: Input.Changed) -> None:
        """Filter visible tag chips by substring."""
        query = event.value.strip().lower()
        for widget in self.query(".tag-chip"):
            if isinstance(widget, TagChip):
                widget.display = query in widget.tag.lower()

    @on(Button.Pressed, ".tag-chip")
    def on_tag_chip_pressed(self, event: Button.Pressed) -> None:
        """Toggle a suggested tag in the active tag input."""
        chip = event.button
        if not isinstance(chip, TagChip):
            return
        tags = self._parse_tags(self._current_tag_input().value)
        if chip.tag in tags:
            tags = [tag for tag in tags if tag != chip.tag]
        else:
            tags.append(chip.tag)
        self._set_tags_value(tags)

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        """Handle the save button press by triggering the save action."""
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        """Handle the cancel button press by dismissing without saving."""
        self.action_cancel()

    @on(Input.Submitted, "#tags-input")
    def on_tags_submitted(self) -> None:
        """Handle Enter key in the tags input by triggering save."""
        self.action_save()

    @on(Input.Submitted, "#autotag-input")
    def on_autotag_submitted(self) -> None:
        """Handle Enter key in the autotag input by triggering save."""
        self.action_save()


class LineAnnotationModal(ModalBase["LineAnnotationResult | None"]):
    """Compact modal for adding an inline detail-pane annotation."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    LineAnnotationModal {
        align: center middle;
    }

    #line-annotation-dialog {
        width: 68;
        height: auto;
        background: $th-background;
        border: tall $th-accent;
        padding: 1 2;
    }

    #line-annotation-title {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #line-annotation-input {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #line-annotation-input:focus {
        border-left: tall $th-accent;
    }

    #line-annotation-help {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, line: int) -> None:
        super().__init__()
        self._line = max(1, line)

    def compose(self) -> ComposeResult:
        """Yield a single-line annotation input."""
        with Vertical(id="line-annotation-dialog"):
            yield Label(f"Annotate line {self._line}", id="line-annotation-title")
            yield Input(placeholder="Margin note...", id="line-annotation-input")
            yield Static("[dim]Enter save | Esc cancel[/dim]", id="line-annotation-help")

    def on_mount(self) -> None:
        """Focus the note input."""
        self._focus_widget("#line-annotation-input")

    @on(Input.Submitted, "#line-annotation-input")
    def on_annotation_submitted(self, event: Input.Submitted) -> None:
        """Save a non-empty annotation."""
        text = event.value.strip()
        if not text:
            self.dismiss(None)
            return
        self.dismiss(LineAnnotationResult(line=self._line, text=text))
