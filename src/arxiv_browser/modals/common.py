"""General-purpose modal dialogs.

HelpScreen, ConfirmModal, ExportMenuModal, MetadataSnapshotPickerModal,
SectionToggleModal, WatchListItem, and WatchListModal — extracted from app.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from arxiv_browser.models import DETAIL_SECTION_NAMES, WATCH_MATCH_TYPES, WatchListEntry
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)

# ============================================================================
# Help Overlay
# ============================================================================


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

    @staticmethod
    def _render_section_lines(entries: list[tuple[str, str]]) -> str:
        """Render key-description pairs as theme-coloured markup lines."""
        green = THEME_COLORS["green"]
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
        for section_name, entries in filtered:
            widgets.append(
                Label(
                    f"[{THEME_COLORS['accent']}]{section_name}[/]",
                    classes="help-section-title",
                )
            )
            widgets.append(Static(self._render_section_lines(entries), classes="help-keys"))
        if widgets:
            await container.mount(*widgets)

    def action_dismiss(self) -> None:
        """Close the help screen."""
        self.dismiss(None)


# ============================================================================
# Confirm Modal
# ============================================================================


class ConfirmModal(ModalScreen[bool]):
    """Modal dialog for confirming batch operations."""

    BINDINGS = [
        Binding("y", "confirm", "Confirm"),
        Binding("n", "cancel", "Cancel"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmModal {
        align: center middle;
    }

    #confirm-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #confirm-message {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #confirm-buttons {
        height: auto;
        align: right middle;
    }

    #confirm-buttons Button {
        margin-left: 1;
    }

    #confirm-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, message: str) -> None:
        """Initialise the confirmation modal with the given prompt message."""
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        """Yield the confirmation message, confirm/cancel buttons, and footer hint."""
        with Vertical(id="confirm-dialog"):
            yield Label(self._message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Confirm (y)", variant="warning", id="confirm-yes")
                yield Button("Cancel (Esc)", variant="default", id="confirm-no")
            yield Static("Confirm: y  Cancel: n / Esc", id="confirm-footer")

    def action_confirm(self) -> None:
        """Dismiss the modal with a positive (True) result."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss the modal with a negative (False) result."""
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-yes")
    def on_yes(self) -> None:
        """Handle the Confirm button press by dismissing with True."""
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def on_no(self) -> None:
        """Handle the Cancel button press by dismissing with False."""
        self.dismiss(False)


# ============================================================================
# Export Menu Modal
# ============================================================================


class ExportMenuModal(ModalScreen[str]):
    """Unified export menu offering all clipboard and file export formats."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
        Binding("c", "do_clipboard_plain", "Plain text", show=False),
        Binding("b", "do_clipboard_bibtex", "BibTeX", show=False),
        Binding("m", "do_clipboard_markdown", "Markdown", show=False),
        Binding("r", "do_clipboard_ris", "RIS", show=False),
        Binding("v", "do_clipboard_csv", "CSV", show=False),
        Binding("t", "do_clipboard_mdtable", "Markdown table", show=False),
        Binding("B", "do_file_bibtex", "BibTeX file", show=False),
        Binding("R", "do_file_ris", "RIS file", show=False),
        Binding("C", "do_file_csv", "CSV file", show=False),
    ]

    CSS = """
    ExportMenuModal {
        align: center middle;
    }

    #export-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #export-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
    }

    .export-section {
        color: $th-muted;
        margin-top: 1;
    }

    .export-keys {
        padding-left: 2;
        color: $th-text;
    }

    #export-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, paper_count: int) -> None:
        """Initialise the export menu with the number of selected papers."""
        super().__init__()
        self._paper_count = paper_count

    def compose(self) -> ComposeResult:
        """Yield the export dialog with clipboard and file format options."""
        s = "s" if self._paper_count != 1 else ""
        with Vertical(id="export-dialog"):
            yield Label(
                f"Export Papers ({self._paper_count} selected{s})",
                id="export-title",
            )
            yield Label("[bold]Clipboard[/bold]", classes="export-section")
            g = THEME_COLORS["green"]
            yield Static(
                f"  [{g}]c[/]  Plain text     [{g}]b[/]  BibTeX\n"
                f"  [{g}]m[/]  Markdown       [{g}]r[/]  RIS\n"
                f"  [{g}]v[/]  CSV            [{g}]t[/]  Markdown table",
                classes="export-keys",
            )
            yield Label("[bold]File[/bold]", classes="export-section")
            yield Static(
                f"  [{g}]B[/]  BibTeX (.bib)  [{g}]R[/]  RIS (.ris)\n  [{g}]C[/]  CSV (.csv)",
                classes="export-keys",
            )
            yield Static("[dim]Cancel: Esc/q[/dim]", id="export-footer")

    def action_cancel(self) -> None:
        """Close the export menu without selecting a format."""
        self.dismiss("")

    def action_do_clipboard_plain(self) -> None:
        """Export selected papers as plain text to the clipboard."""
        self.dismiss("clipboard-plain")

    def action_do_clipboard_bibtex(self) -> None:
        """Export selected papers as BibTeX to the clipboard."""
        self.dismiss("clipboard-bibtex")

    def action_do_clipboard_markdown(self) -> None:
        """Export selected papers as Markdown to the clipboard."""
        self.dismiss("clipboard-markdown")

    def action_do_clipboard_ris(self) -> None:
        """Export selected papers as RIS to the clipboard."""
        self.dismiss("clipboard-ris")

    def action_do_clipboard_csv(self) -> None:
        """Export selected papers as CSV to the clipboard."""
        self.dismiss("clipboard-csv")

    def action_do_clipboard_mdtable(self) -> None:
        """Export selected papers as a Markdown table to the clipboard."""
        self.dismiss("clipboard-mdtable")

    def action_do_file_bibtex(self) -> None:
        """Export selected papers to a BibTeX (.bib) file."""
        self.dismiss("file-bibtex")

    def action_do_file_ris(self) -> None:
        """Export selected papers to a RIS (.ris) file."""
        self.dismiss("file-ris")

    def action_do_file_csv(self) -> None:
        """Export selected papers to a CSV (.csv) file."""
        self.dismiss("file-csv")


# ============================================================================
# Metadata Snapshot Picker
# ============================================================================


class MetadataSnapshotItem(ListItem):
    """List item for a metadata snapshot import choice."""

    def __init__(self, snapshot_path: Path, *children, **kwargs) -> None:
        """Initialise with the filesystem path to a metadata snapshot."""
        super().__init__(*children, **kwargs)
        self.snapshot_path = snapshot_path


class MetadataSnapshotPickerModal(ModalScreen[Path | None]):
    """Modal for choosing which metadata snapshot to import."""

    BINDINGS = [
        Binding("enter", "choose", "Import"),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    MetadataSnapshotPickerModal {
        align: center middle;
    }

    #metadata-snapshot-dialog {
        width: 52;
        height: 24;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #metadata-snapshot-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
    }

    #metadata-snapshot-subtitle {
        color: $th-muted;
        margin-bottom: 1;
    }

    #metadata-snapshot-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #metadata-snapshot-list > ListItem {
        padding: 0 1;
    }

    #metadata-snapshot-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, snapshots: list[Path]) -> None:
        """Initialise the picker with a list of available snapshot paths."""
        super().__init__()
        self._snapshots = list(snapshots)

    def compose(self) -> ComposeResult:
        """Yield the snapshot list dialog with title, subtitle, and footer."""
        with Vertical(id="metadata-snapshot-dialog"):
            yield Label("Import Metadata Snapshot", id="metadata-snapshot-title")
            yield Static(
                "Choose the JSON snapshot to import. Newest snapshots appear first.",
                id="metadata-snapshot-subtitle",
            )
            yield ListView(id="metadata-snapshot-list")
            yield Static(
                "[dim]Select: Enter | Cancel: Esc/q[/]",
                id="metadata-snapshot-footer",
            )

    def on_mount(self) -> None:
        """Populate the list view with snapshot entries and focus it."""
        list_view = self.query_one("#metadata-snapshot-list", ListView)
        list_view.clear()
        for snapshot in self._snapshots:
            label = self._format_snapshot_label(snapshot)
            list_view.mount(MetadataSnapshotItem(snapshot, Label(label)))
        if list_view.children:
            list_view.index = 0
        list_view.focus()

    @staticmethod
    def _format_snapshot_label(snapshot: Path) -> str:
        """Format a snapshot filename with its last-modified timestamp."""
        try:
            modified = datetime.fromtimestamp(snapshot.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            modified = "unknown time"
        return f"{snapshot.name}  [{THEME_COLORS['muted']}]modified {modified}[/]"

    def action_choose(self) -> None:
        """Dismiss with the currently highlighted snapshot path."""
        list_view = self.query_one("#metadata-snapshot-list", ListView)
        if isinstance(list_view.highlighted_child, MetadataSnapshotItem):
            self.dismiss(list_view.highlighted_child.snapshot_path)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Close the picker without selecting a snapshot."""
        self.dismiss(None)

    @on(ListView.Selected, "#metadata-snapshot-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection by dismissing with the chosen snapshot."""
        if isinstance(event.item, MetadataSnapshotItem):
            self.dismiss(event.item.snapshot_path)


# ============================================================================
# Watch List
# ============================================================================


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


# ============================================================================
# Section Toggle Modal
# ============================================================================

# Section toggle hotkeys: single key -> section key
_SECTION_TOGGLE_KEYS: dict[str, str] = {
    "a": "authors",
    "b": "abstract",
    "t": "tags",
    "r": "relevance",
    "s": "summary",
    "e": "s2",
    "h": "hf",
    "v": "version",
}


class SectionToggleModal(ModalScreen[list[str] | None]):
    """Modal for toggling collapsible detail pane sections.

    Dismisses with a **sorted list of the section keys that should remain
    collapsed** (e.g. ``["abstract", "summary"]``), or ``None`` when the
    user cancels.  The caller is responsible for writing this list to
    ``config.collapsed_sections``.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
        Binding("a", "toggle_a", "", show=False),
        Binding("b", "toggle_b", "", show=False),
        Binding("t", "toggle_t", "", show=False),
        Binding("r", "toggle_r", "", show=False),
        Binding("s", "toggle_s", "", show=False),
        Binding("e", "toggle_e", "", show=False),
        Binding("h", "toggle_h", "", show=False),
        Binding("v", "toggle_v", "", show=False),
    ]

    CSS = """
    SectionToggleModal {
        align: center middle;
    }

    #section-toggle-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #section-toggle-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    .section-toggle-list {
        padding-left: 2;
        color: $th-text;
    }

    #section-toggle-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, collapsed: list[str]) -> None:
        """Initialise with the list of currently collapsed section keys."""
        super().__init__()
        self._collapsed: set[str] = set(collapsed)

    def compose(self) -> ComposeResult:
        """Yield the section toggle list with expand/collapse indicators."""
        with Vertical(id="section-toggle-dialog"):
            yield Label("Detail Pane Sections", id="section-toggle-title")
            yield Static(
                self._render_list(), id="section-toggle-list", classes="section-toggle-list"
            )
            yield Static(
                "[dim]Toggle: key | Save: Enter | Cancel: Esc/q[/]",
                id="section-toggle-footer",
            )

    def _render_list(self) -> str:
        """Render all section names with their current expanded/collapsed state."""
        g = THEME_COLORS["green"]
        lines = []
        for key, section in _SECTION_TOGGLE_KEYS.items():
            name = DETAIL_SECTION_NAMES[section]
            indicator = "\u25b8" if section in self._collapsed else "\u25be"
            state = "[dim]collapsed[/]" if section in self._collapsed else f"[{g}]expanded[/]"
            lines.append(f"  [{g}]{key}[/]  {indicator} {name:<18s} {state}")
        return "\n".join(lines)

    def _toggle(self, key: str) -> None:
        """Toggle a section's collapsed state by hotkey and refresh the display."""
        section = _SECTION_TOGGLE_KEYS.get(key)
        if section is None:
            return
        if section in self._collapsed:
            self._collapsed.discard(section)
        else:
            self._collapsed.add(section)
        try:
            self.query_one("#section-toggle-list", Static).update(self._render_list())
        except NoMatches:
            pass

    def action_toggle_a(self) -> None:
        """Toggle the authors section."""
        self._toggle("a")

    def action_toggle_b(self) -> None:
        """Toggle the abstract section."""
        self._toggle("b")

    def action_toggle_t(self) -> None:
        """Toggle the tags section."""
        self._toggle("t")

    def action_toggle_r(self) -> None:
        """Toggle the relevance section."""
        self._toggle("r")

    def action_toggle_s(self) -> None:
        """Toggle the summary section."""
        self._toggle("s")

    def action_toggle_e(self) -> None:
        """Toggle the Semantic Scholar section."""
        self._toggle("e")

    def action_toggle_h(self) -> None:
        """Toggle the Hugging Face section."""
        self._toggle("h")

    def action_toggle_v(self) -> None:
        """Toggle the version section."""
        self._toggle("v")

    def action_save(self) -> None:
        """Dismiss and return the sorted list of collapsed section keys."""
        self.dismiss(sorted(self._collapsed))

    def action_cancel(self) -> None:
        """Dismiss without applying any changes."""
        self.dismiss(None)
