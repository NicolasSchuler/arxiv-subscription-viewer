"""General-purpose modal dialogs.

ConfirmModal, ExportMenuModal, MetadataSnapshotPickerModal, and
SectionToggleModal — extracted from app.py.

HelpScreen has moved to modals/help.py.
WatchListItem and WatchListModal have moved to modals/watchlist.py.
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
from textual.widgets import (
    Button,
    Label,
    ListItem,
    ListView,
    Static,
)

from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import DETAIL_SECTION_NAMES
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)

# ============================================================================
# Confirm Modal
# ============================================================================


class ConfirmModal(ModalBase[bool]):
    """Modal dialog for confirming batch operations."""

    BINDINGS = [
        Binding("y", "confirm", "Confirm"),
        Binding("n", "cancel", "Cancel"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    #confirm-dialog {
        width: 52;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        /* destructive action: keep the warning-orange border, not accent */
        border: tall $th-orange;
    }

    #confirm-message-scroll {
        height: auto;
        max-height: 16;
    }

    #confirm-message {
        color: $th-accent-alt;
    }

    #confirm-buttons Button {
        margin-left: 1;
    }

    #confirm-footer {
        margin-top: 1;
    }
    """

    def __init__(self, message: str) -> None:
        """Initialise the confirmation modal with the given prompt message."""
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        """Yield the confirmation message, confirm/cancel buttons, and footer hint."""
        with Vertical(id="confirm-dialog", classes="modal-dialog"):
            with VerticalScroll(id="confirm-message-scroll"):
                yield Label(self._message, id="confirm-message", classes="modal-title")
            with Horizontal(id="confirm-buttons", classes="modal-buttons"):
                yield Button("Confirm (y)", variant="warning", id="confirm-yes")
                yield Button("Cancel (Esc)", variant="default", id="confirm-no")
            yield Static("y confirm | Esc/n cancel", id="confirm-footer", classes="modal-footer")

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


class ExportMenuModal(ModalBase[str]):
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
    #export-dialog {
        width: 52;
        max-width: 90%;
        height: auto;
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
        g = theme_colors_for(self)["green"]
        with Vertical(id="export-dialog", classes="modal-dialog"):
            yield Label(
                f"Export Papers ({self._paper_count} selected{s})",
                id="export-title",
                classes="modal-title",
            )
            yield Label("[bold]Clipboard[/bold]", classes="export-section")
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
            yield Static("Cancel: Esc/q", id="export-footer", classes="modal-footer")

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


class MetadataSnapshotPickerModal(ModalBase[Path | None]):
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
        max-width: 90%;
        height: 24;
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
        margin-top: 1;
    }
    """

    def __init__(self, snapshots: list[Path]) -> None:
        """Initialise the picker with a list of available snapshot paths."""
        super().__init__()
        self._snapshots = list(snapshots)

    def compose(self) -> ComposeResult:
        """Yield the snapshot list dialog with title, subtitle, and footer."""
        with Vertical(id="metadata-snapshot-dialog", classes="modal-dialog"):
            yield Label(
                "Import Metadata Snapshot",
                id="metadata-snapshot-title",
                classes="modal-title",
            )
            subtitle = (
                "Choose the JSON snapshot to import. Newest snapshots appear first."
                if self._snapshots
                else "No metadata snapshots found. They are created when you export metadata."
            )
            yield Static(subtitle, id="metadata-snapshot-subtitle")
            yield ListView(id="metadata-snapshot-list")
            yield Static(
                "Select: Enter | Cancel: Esc/q",
                id="metadata-snapshot-footer",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        """Populate the list view with snapshot entries and focus it."""
        list_view = self.query_one("#metadata-snapshot-list", ListView)
        list_view.clear()
        muted = theme_colors_for(self)["muted"]
        if not self._snapshots:
            placeholder = ListItem(Label(f"[{muted}]No snapshots found.[/]"))
            placeholder.disabled = True
            list_view.mount(placeholder)
            return
        for snapshot in self._snapshots:
            label = self._format_snapshot_label(snapshot, muted)
            list_view.mount(MetadataSnapshotItem(snapshot, Label(label)))
        list_view.index = 0
        list_view.focus()

    @staticmethod
    def _format_snapshot_label(snapshot: Path, muted_color: str | None = None) -> str:
        """Format a snapshot filename with its last-modified timestamp."""
        try:
            modified = datetime.fromtimestamp(snapshot.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            modified = "unknown time"
        muted = muted_color or theme_colors_for(None)["muted"]
        return f"{snapshot.name}  [{muted}]modified {modified}[/]"

    def action_choose(self) -> None:
        """Dismiss with the currently highlighted snapshot path."""
        list_view = self.query_one("#metadata-snapshot-list", ListView)
        if isinstance(list_view.highlighted_child, MetadataSnapshotItem):
            self.dismiss(list_view.highlighted_child.snapshot_path)
        else:
            self.dismiss(None)

    @on(ListView.Selected, "#metadata-snapshot-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection by dismissing with the chosen snapshot."""
        if isinstance(event.item, MetadataSnapshotItem):
            self.dismiss(event.item.snapshot_path)


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


class SectionToggleModal(ModalBase[list[str] | None]):
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
    #section-toggle-dialog {
        width: 52;
        max-width: 90%;
        height: auto;
    }

    .section-toggle-list {
        padding-left: 2;
        color: $th-text;
    }

    #section-toggle-footer {
        margin-top: 1;
    }
    """

    def __init__(self, collapsed: list[str]) -> None:
        """Initialise with the list of currently collapsed section keys."""
        super().__init__()
        self._collapsed: set[str] = set(collapsed)

    def compose(self) -> ComposeResult:
        """Yield the section toggle list with expand/collapse indicators."""
        with Vertical(id="section-toggle-dialog", classes="modal-dialog"):
            yield Label("Detail Pane Sections", id="section-toggle-title", classes="modal-title")
            yield Static(
                self._render_list(), id="section-toggle-list", classes="section-toggle-list"
            )
            yield Static(
                "Toggle: key | Save: Enter | Cancel: Esc/q",
                id="section-toggle-footer",
                classes="modal-footer",
            )

    def _render_list(self) -> str:
        """Render all section names with their current expanded/collapsed state."""
        from arxiv_browser._ascii import is_ascii_mode

        g = theme_colors_for(self)["green"]
        collapsed_glyph, expanded_glyph = (">", "v") if is_ascii_mode() else ("\u25b8", "\u25be")
        lines = []
        for key, section in _SECTION_TOGGLE_KEYS.items():
            name = DETAIL_SECTION_NAMES[section]
            indicator = collapsed_glyph if section in self._collapsed else expanded_glyph
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
