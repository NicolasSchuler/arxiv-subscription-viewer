"""Watch list management modals.

WatchListItem and WatchListModal — extracted from common.py.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Label, ListItem, ListView, Select, Static

from arxiv_browser._ascii import is_ascii_mode
from arxiv_browser.modals.base import ModalBase, build_empty_placeholder
from arxiv_browser.models import WATCH_MATCH_TYPES, WatchListEntry

# Match-type placeholder / prompt shown before a type is chosen. Lists the
# actual options so the Select is a guessable affordance instead of a blank bar.
WATCH_MATCH_PROMPT = " / ".join(WATCH_MATCH_TYPES)

# Manage-view empty state rendered inside the list region (mirrors the
# Collections manager). Keeps the §7 Try:/Next: template intact.
WATCH_LIST_EMPTY = (
    "No watch entries yet. Try: add a pattern on the right, then press Add. "
    "Next: press Save to persist."
)


class WatchListItem(ListItem):
    """List item for watch list entries."""

    def __init__(self, entry: WatchListEntry, *children, **kwargs) -> None:
        """Initialise with the associated watch list entry."""
        super().__init__(*children, **kwargs)
        self.entry = entry


class WatchListModal(ModalBase[list[WatchListEntry] | None]):
    """Modal dialog for managing watch list entries."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    #watch-dialog {
        width: 70;
        max-width: 90%;
        height: 70%;
        min-height: 20;
        /* tighter vertical padding than the shared .modal-dialog default (1 2) */
        padding: 0 2;
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

    /* Full-width, wrapping empty-state placeholder rendered inside the list
       region (avoids single-line truncation of the Try:/Next: hint). */
    #watch-list > ListItem.-empty {
        height: auto;
    }

    #watch-list > ListItem.-empty > Label {
        width: 1fr;
        height: auto;
    }

    #watch-form {
        width: 1fr;
        height: 1fr;
    }

    #watch-form Label {
        color: $th-muted;
        margin-top: 1;
    }

    /* Single-line height matches the Collections form inputs so both list
       managers share one field rhythm. */
    #watch-pattern {
        width: 100%;
        height: 1;
        background: $th-panel;
        border: none;
    }

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

    /* min-width:0 keeps all three action buttons inside the dialog interior
       (default min-width:16 overflows the ~62-col row). */
    #watch-actions Button {
        margin-right: 1;
        min-width: 0;
    }

    /* Destructive Delete stays dim/outlined at rest so Save keeps the only
       primary weight; the error color is reserved for hover/focus. */
    #watch-delete {
        color: $th-muted;
        background: transparent;
        text-style: none;
    }

    #watch-delete:hover,
    #watch-delete:focus {
        color: $error;
        background: $error-muted;
        text-style: bold;
    }

    #watch-buttons {
        margin-top: 1;
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
        self._dirty = False

    def compose(self) -> ComposeResult:
        """Yield the watch list view, entry form, and save/cancel buttons."""
        with Vertical(id="watch-dialog", classes="modal-dialog"):
            yield Label("Watch List Manager", id="watch-title", classes="modal-title")
            with Horizontal(id="watch-body"):
                with Vertical(id="watch-list-column"):
                    yield ListView(id="watch-list")
                with Vertical(id="watch-form"):
                    yield Label("Pattern")
                    yield Input(placeholder="e.g., diffusion", id="watch-pattern")
                    yield Label("Match Type")
                    yield Select(
                        [(value, value) for value in WATCH_MATCH_TYPES],
                        prompt=WATCH_MATCH_PROMPT,
                        id="watch-type",
                    )
                    yield Checkbox("Case sensitive", id="watch-case")
            # Action row lives at the dialog level (not nested in the narrow
            # 1fr form column) so all three buttons stay on-dialog.
            with Horizontal(id="watch-actions"):
                yield Button("Add", variant="default", id="watch-add")
                yield Button("Update", variant="default", id="watch-update")
                yield Button("Delete", variant="default", id="watch-delete")
            with Horizontal(id="watch-buttons", classes="modal-buttons"):
                yield Button("Cancel", variant="default", id="watch-cancel")
                yield Button("Save (Ctrl+S)", variant="primary", id="watch-save")
            yield Static(
                "No unsaved changes | Esc discards edits",
                id="watch-help",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        """Populate the list view, ASCII-fix the Select arrow, and focus the pattern."""
        self._refresh_list()
        self._focus_widget("#watch-pattern")
        # Deferred so the Select's arrow children (mounted by the framework
        # during compose) exist before we rewrite their glyphs.
        self.call_after_refresh(self._apply_ascii_select_arrow)

    def _apply_ascii_select_arrow(self) -> None:
        """Swap the Select's Unicode chevron for an ASCII fallback in ASCII mode.

        The ``▼``/``▲`` glyphs are framework-emitted (see the style guide's
        "Framework Chrome Glyphs" note); we override them here so the dropdown
        affordance survives ``--ascii``.
        """
        if not is_ascii_mode():
            return
        for arrow in self.query("#watch-type .down-arrow"):
            if isinstance(arrow, Static):
                arrow.update("v")
        for arrow in self.query("#watch-type .up-arrow"):
            if isinstance(arrow, Static):
                arrow.update("^")

    def _refresh_list(self) -> None:
        """Rebuild the list view from the current entries.

        When there are no entries an in-list empty-state placeholder is mounted
        so the guidance sits inside the list region rather than floating below
        an empty box.
        """
        list_view = self.query_one("#watch-list", ListView)
        list_view.clear()
        if not self._entries:
            list_view.mount(build_empty_placeholder(WATCH_LIST_EMPTY))
            return
        for entry in self._entries:
            label = f"{entry.match_type}: {entry.pattern}"
            if entry.case_sensitive:
                label = f"{label} (Aa)"
            list_view.mount(WatchListItem(entry, Label(label)))
        if list_view.children:
            list_view.index = 0
            self._populate_form(list_view.highlighted_child)

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

    def _mark_dirty(self) -> None:
        """Mark watch-list edits as unsaved in the modal footer."""
        self._dirty = True
        try:
            self.query_one("#watch-help", Static).update(
                "[bold]Unsaved changes[/bold] | Esc discards edits"
            )
        except NoMatches:
            return

    def action_save(self) -> None:
        """Dismiss the modal and return the current list of entries."""
        self.dismiss(self._entries)

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
        self._mark_dirty()
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
        self._mark_dirty()
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
        self._mark_dirty()
        self._refresh_list()

    @on(Button.Pressed, "#watch-save")
    def on_save_pressed(self) -> None:
        """Handle the Save button press by delegating to action_save."""
        self.action_save()

    @on(Button.Pressed, "#watch-cancel")
    def on_cancel_pressed(self) -> None:
        """Handle the Cancel button press by delegating to action_cancel."""
        self.action_cancel()
