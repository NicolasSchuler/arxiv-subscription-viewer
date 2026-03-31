"""Watch list management modals.

WatchListItem and WatchListModal — extracted from common.py.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, ListItem, ListView, Select, Static

from arxiv_browser.models import WATCH_MATCH_TYPES, WatchListEntry


class WatchListItem(ListItem):
    """List item for watch list entries."""

    def __init__(self, entry: WatchListEntry, *children, **kwargs) -> None:
        """Initialise with the associated watch list entry."""
        super().__init__(*children, **kwargs)
        self.entry = entry


class WatchListModal(ModalScreen[list[WatchListEntry] | None]):
    """Modal dialog for managing watch list entries."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    WatchListModal {
        align: center middle;
    }

    #watch-dialog {
        width: 70;
        height: 70%;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #watch-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #watch-body {
        height: 1fr;
    }

    #watch-list {
        width: 100%;
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #watch-list-column {
        width: 2fr;
        height: 1fr;
        margin-right: 2;
    }

    #watch-empty {
        color: $th-muted;
        padding: 0 1;
        margin-top: 1;
        display: none;
    }

    #watch-empty.visible {
        display: block;
    }

    #watch-form {
        width: 1fr;
        height: 1fr;
    }

    #watch-form Label {
        color: $th-muted;
        margin-top: 1;
    }

    #watch-pattern,
    #watch-type {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #watch-pattern:focus,
    #watch-type:focus {
        border-left: tall $th-accent;
    }

    #watch-case {
        margin-top: 1;
    }

    #watch-actions {
        height: auto;
        margin-top: 1;
        align: left middle;
    }

    #watch-actions Button {
        margin-right: 1;
    }

    #watch-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #watch-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, entries: list[WatchListEntry]) -> None:
        """Initialise the modal with a defensive copy of watch list entries."""
        super().__init__()
        self._entries = [
            WatchListEntry(
                pattern=entry.pattern,
                match_type=entry.match_type,
                case_sensitive=entry.case_sensitive,
            )
            for entry in entries
        ]

    def compose(self) -> ComposeResult:
        """Yield the watch list view, entry form, and save/cancel buttons."""
        with Vertical(id="watch-dialog"):
            yield Label("Watch List Manager", id="watch-title")
            with Horizontal(id="watch-body"):
                with Vertical(id="watch-list-column"):
                    yield ListView(id="watch-list")
                    yield Static(
                        "No watch entries yet.\nTry: add a pattern on the right, then press Add.",
                        id="watch-empty",
                    )
                with Vertical(id="watch-form"):
                    yield Label("Pattern")
                    yield Input(placeholder="e.g., diffusion", id="watch-pattern")
                    yield Label("Match Type")
                    yield Select(
                        [(value, value) for value in WATCH_MATCH_TYPES],
                        id="watch-type",
                    )
                    yield Checkbox("Case sensitive", id="watch-case")
                    with Horizontal(id="watch-actions"):
                        yield Button("Add", variant="primary", id="watch-add")
                        yield Button("Update", variant="default", id="watch-update")
                        yield Button("Delete", variant="default", id="watch-delete")
            with Horizontal(id="watch-buttons"):
                yield Button("Cancel", variant="default", id="watch-cancel")
                yield Button("Save (Ctrl+S)", variant="primary", id="watch-save")

    def on_mount(self) -> None:
        """Populate the list view and focus the pattern input on mount."""
        self._refresh_list()
        self.query_one("#watch-pattern", Input).focus()

    def _refresh_list(self) -> None:
        """Rebuild the list view from the current entries and update the empty hint."""
        list_view = self.query_one("#watch-list", ListView)
        empty_hint = self.query_one("#watch-empty", Static)
        list_view.clear()
        for entry in self._entries:
            label = f"{entry.match_type}: {entry.pattern}"
            if entry.case_sensitive:
                label = f"{label} (Aa)"
            list_view.mount(WatchListItem(entry, Label(label)))
        if list_view.children:
            list_view.index = 0
            self._populate_form(list_view.highlighted_child)
            empty_hint.remove_class("visible")
        else:
            empty_hint.add_class("visible")

    def _populate_form(self, item: ListItem | None) -> None:
        """Fill the pattern, match-type, and case-sensitivity fields from a list item."""
        if not isinstance(item, WatchListItem):
            return
        self.query_one("#watch-pattern", Input).value = item.entry.pattern
        self.query_one("#watch-type", Select).value = item.entry.match_type
        self.query_one("#watch-case", Checkbox).value = item.entry.case_sensitive

    def _build_entry_from_form(self) -> WatchListEntry | None:
        """Read the form fields and return a new ``WatchListEntry``, or ``None`` if invalid.

        A form entry is considered invalid only when the pattern field is
        empty after stripping whitespace — all other validation (unknown
        ``match_type``) silently falls back to ``"author"``.

        Returns:
            A ``WatchListEntry`` populated from the current form state, or
            ``None`` (with a warning notification) when the pattern is empty.
        """
        pattern = self.query_one("#watch-pattern", Input).value.strip()
        match_value = self.query_one("#watch-type", Select).value
        match_type = match_value if isinstance(match_value, str) else "author"
        case_sensitive = self.query_one("#watch-case", Checkbox).value
        if not pattern:
            self.notify("Pattern cannot be empty", title="Watch", severity="warning")
            return None
        if match_type not in WATCH_MATCH_TYPES:
            match_type = "author"
        return WatchListEntry(
            pattern=pattern,
            match_type=match_type,
            case_sensitive=case_sensitive,
        )

    def action_save(self) -> None:
        """Dismiss the modal and return the current list of entries."""
        self.dismiss(self._entries)

    def action_cancel(self) -> None:
        """Dismiss the modal without saving changes."""
        self.dismiss(None)

    @on(ListView.Highlighted, "#watch-list")
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        """Sync the form fields when a different list entry is highlighted."""
        self._populate_form(event.item)

    @on(Button.Pressed, "#watch-add")
    def on_add_pressed(self) -> None:
        """Create a new watch entry from the form and append it to the list."""
        entry = self._build_entry_from_form()
        if not entry:
            return
        self._entries.append(entry)
        self._refresh_list()

    @on(Button.Pressed, "#watch-update")
    def on_update_pressed(self) -> None:
        """Replace the highlighted watch entry with current form values."""
        list_view = self.query_one("#watch-list", ListView)
        if not isinstance(list_view.highlighted_child, WatchListItem):
            self.notify("Select a watch entry to update", title="Watch")
            return
        entry = self._build_entry_from_form()
        if not entry:
            return
        index = list_view.index if list_view.index is not None else 0
        self._entries[index] = entry
        self._refresh_list()

    @on(Button.Pressed, "#watch-delete")
    def on_delete_pressed(self) -> None:
        """Remove the highlighted watch entry from the list."""
        list_view = self.query_one("#watch-list", ListView)
        if not isinstance(list_view.highlighted_child, WatchListItem):
            self.notify("Select a watch entry to delete", title="Watch")
            return
        index = list_view.index if list_view.index is not None else 0
        self._entries.pop(index)
        self._refresh_list()

    @on(Button.Pressed, "#watch-save")
    def on_save_pressed(self) -> None:
        """Handle the Save button press by delegating to action_save."""
        self.action_save()

    @on(Button.Pressed, "#watch-cancel")
    def on_cancel_pressed(self) -> None:
        """Handle the Cancel button press by delegating to action_cancel."""
        self.action_cancel()
