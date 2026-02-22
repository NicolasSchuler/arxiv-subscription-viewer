"""General-purpose modal dialogs.

HelpScreen, ConfirmModal, ExportMenuModal, SectionToggleModal,
WatchListItem, and WatchListModal — extracted from app.py.
"""

from __future__ import annotations

import logging

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
            "Navigation",
            [
                ("j / k", "Navigate down / up"),
                ("[ / ]", "Previous / next date"),
                ("1-9", "Jump to bookmark"),
                ("m then a-z", "Set mark"),
                ("' then a-z", "Jump to mark"),
            ],
        ),
        (
            "Search & Filter",
            [
                ("/", "Toggle search"),
                ("Esc", "Clear search / exit API"),
                ("A", "Search all arXiv (API mode)"),
                ("Ctrl+e", "Toggle S2 (browse) / Exit API (API mode)"),
                ("w", "Toggle watch list filter"),
                ("Ctrl+b", "Save search as bookmark"),
            ],
        ),
        (
            "Selection",
            [
                ("Space", "Toggle selection"),
                ("a", "Select all visible"),
                ("u", "Clear selection"),
            ],
        ),
        (
            "Actions",
            [
                ("o", "Open in browser"),
                ("P", "Open as PDF"),
                ("n", "Edit notes"),
                ("t", "Edit tags"),
                ("E", "Export menu"),
                ("d", "Download PDFs"),
                ("?", "Help overlay"),
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
        super().__init__()
        self._sections = sections or list(self._DEFAULT_SECTIONS)
        self._footer_note = footer_note

    @staticmethod
    def _render_section_lines(entries: list[tuple[str, str]]) -> str:
        green = THEME_COLORS["green"]
        lines = [f"  [{green}]{key}[/]  {description}" for key, description in entries]
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-dialog"):
            yield Label("Keyboard Shortcuts", id="help-title")
            for section_name, entries in self._sections:
                if not entries:
                    continue
                yield Label(
                    f"[{THEME_COLORS['accent']}]{section_name}[/]",
                    classes="help-section-title",
                )
                yield Static(self._render_section_lines(entries), classes="help-keys")

            yield Label(self._footer_note, id="help-footer")

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
        width: 50%;
        min-width: 40;
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
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self._message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Confirm (y)", variant="warning", id="confirm-yes")
                yield Button("Cancel (Esc)", variant="default", id="confirm-no")
            yield Static("Confirm: y  Cancel: n / Esc", id="confirm-footer")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-yes")
    def on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def on_no(self) -> None:
        self.dismiss(False)


# ============================================================================
# Export Menu Modal
# ============================================================================


class ExportMenuModal(ModalScreen[str]):
    """Unified export menu offering all clipboard and file export formats."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
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
        super().__init__()
        self._paper_count = paper_count

    def compose(self) -> ComposeResult:
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
            yield Static("[dim]Cancel: Esc[/dim]", id="export-footer")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_do_clipboard_plain(self) -> None:
        self.dismiss("clipboard-plain")

    def action_do_clipboard_bibtex(self) -> None:
        self.dismiss("clipboard-bibtex")

    def action_do_clipboard_markdown(self) -> None:
        self.dismiss("clipboard-markdown")

    def action_do_clipboard_ris(self) -> None:
        self.dismiss("clipboard-ris")

    def action_do_clipboard_csv(self) -> None:
        self.dismiss("clipboard-csv")

    def action_do_clipboard_mdtable(self) -> None:
        self.dismiss("clipboard-mdtable")

    def action_do_file_bibtex(self) -> None:
        self.dismiss("file-bibtex")

    def action_do_file_ris(self) -> None:
        self.dismiss("file-ris")

    def action_do_file_csv(self) -> None:
        self.dismiss("file-csv")


# ============================================================================
# Watch List
# ============================================================================


class WatchListItem(ListItem):
    """List item for watch list entries."""

    def __init__(self, entry: WatchListEntry, *children, **kwargs) -> None:
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
        width: 80%;
        height: 70%;
        min-width: 60;
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
        self._refresh_list()
        self.query_one("#watch-pattern", Input).focus()

    def _refresh_list(self) -> None:
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
        if not isinstance(item, WatchListItem):
            return
        self.query_one("#watch-pattern", Input).value = item.entry.pattern
        self.query_one("#watch-type", Select).value = item.entry.match_type
        self.query_one("#watch-case", Checkbox).value = item.entry.case_sensitive

    def _build_entry_from_form(self) -> WatchListEntry | None:
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
        self.dismiss(self._entries)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(ListView.Highlighted, "#watch-list")
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        self._populate_form(event.item)

    @on(Button.Pressed, "#watch-add")
    def on_add_pressed(self) -> None:
        entry = self._build_entry_from_form()
        if not entry:
            return
        self._entries.append(entry)
        self._refresh_list()

    @on(Button.Pressed, "#watch-update")
    def on_update_pressed(self) -> None:
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
        list_view = self.query_one("#watch-list", ListView)
        if not isinstance(list_view.highlighted_child, WatchListItem):
            self.notify("Select a watch entry to delete", title="Watch")
            return
        index = list_view.index if list_view.index is not None else 0
        self._entries.pop(index)
        self._refresh_list()

    @on(Button.Pressed, "#watch-save")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#watch-cancel")
    def on_cancel_pressed(self) -> None:
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
    """Modal for toggling collapsible detail pane sections."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
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
        super().__init__()
        self._collapsed: set[str] = set(collapsed)

    def compose(self) -> ComposeResult:
        with Vertical(id="section-toggle-dialog"):
            yield Label("Detail Pane Sections", id="section-toggle-title")
            yield Static(
                self._render_list(), id="section-toggle-list", classes="section-toggle-list"
            )
            yield Static(
                "[dim]Toggle: key \u00b7 Save: Enter \u00b7 Cancel: Esc[/]",
                id="section-toggle-footer",
            )

    def _render_list(self) -> str:
        g = THEME_COLORS["green"]
        lines = []
        for key, section in _SECTION_TOGGLE_KEYS.items():
            name = DETAIL_SECTION_NAMES[section]
            indicator = "\u25b8" if section in self._collapsed else "\u25be"
            state = "[dim]collapsed[/]" if section in self._collapsed else f"[{g}]expanded[/]"
            lines.append(f"  [{g}]{key}[/]  {indicator} {name:<18s} {state}")
        return "\n".join(lines)

    def _toggle(self, key: str) -> None:
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
        self._toggle("a")

    def action_toggle_b(self) -> None:
        self._toggle("b")

    def action_toggle_t(self) -> None:
        self._toggle("t")

    def action_toggle_r(self) -> None:
        self._toggle("r")

    def action_toggle_s(self) -> None:
        self._toggle("s")

    def action_toggle_e(self) -> None:
        self._toggle("e")

    def action_toggle_h(self) -> None:
        self._toggle("h")

    def action_toggle_v(self) -> None:
        self._toggle("v")

    def action_save(self) -> None:
        self.dismiss(sorted(self._collapsed))

    def action_cancel(self) -> None:
        self.dismiss(None)
