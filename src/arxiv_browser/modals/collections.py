"""Paper collections (reading lists) modals."""

from __future__ import annotations

import logging
from datetime import datetime

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView

from arxiv_browser.action_messages import build_actionable_warning
from arxiv_browser.models import MAX_COLLECTIONS, Paper, PaperCollection

logger = logging.getLogger(__name__)


class CollectionsModal(ModalScreen[str | None]):
    """Modal dialog for managing paper collections (reading lists)."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    CollectionsModal {
        align: center middle;
    }

    #col-dialog {
        width: 70;
        height: 70%;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #col-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #col-body {
        height: 1fr;
    }

    #col-list {
        width: 2fr;
        height: 1fr;
        background: $th-panel;
        border: none;
        margin-right: 2;
    }

    #col-form {
        width: 1fr;
        height: 1fr;
    }

    #col-form Label {
        color: $th-muted;
        margin-top: 1;
    }

    #col-name,
    #col-desc {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #col-name:focus,
    #col-desc:focus {
        border-left: tall $th-accent;
    }

    #col-actions {
        height: auto;
        margin-top: 1;
        align: left middle;
    }

    #col-actions Button {
        margin-right: 1;
    }

    #col-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #col-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        collections: list[PaperCollection],
        papers_by_id: dict[str, Paper] | None = None,
    ) -> None:
        """Initialize with deep-copied collections and an optional paper lookup."""
        super().__init__()
        self._collections = [
            PaperCollection(
                name=c.name,
                description=c.description,
                paper_ids=list(c.paper_ids),
                created=c.created,
            )
            for c in collections
        ]
        self._papers_by_id = papers_by_id or {}

    def compose(self) -> ComposeResult:
        """Yield the collections dialog with a list view, name/description form, and action buttons."""
        with Vertical(id="col-dialog"):
            yield Label("Collections Manager", id="col-title")
            with Horizontal(id="col-body"):
                yield ListView(id="col-list")
                with Vertical(id="col-form"):
                    yield Label("Name")
                    yield Input(placeholder="e.g., ML Reading List", id="col-name")
                    yield Label("Description")
                    yield Input(placeholder="Optional description", id="col-desc")
                    with Horizontal(id="col-actions"):
                        yield Button("Create", variant="primary", id="col-create")
                        yield Button("Rename", variant="default", id="col-rename")
                        yield Button("Delete", variant="default", id="col-delete")
                        yield Button("View", variant="default", id="col-view")
            with Horizontal(id="col-buttons"):
                yield Button("Close", variant="default", id="col-close")
                yield Button("Save", variant="primary", id="col-save")

    def on_mount(self) -> None:
        """Populate the collections list and focus the name input on mount."""
        self._refresh_list()
        self.query_one("#col-name", Input).focus()

    def _refresh_list(self) -> None:
        """Clear and repopulate the collections list view with current collection data."""
        list_view = self.query_one("#col-list", ListView)
        list_view.clear()
        for col in self._collections:
            count = len(col.paper_ids)
            label = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0
            self._populate_form(0)

    def _populate_form(self, index: int) -> None:
        """Fill the name and description inputs from the collection at the given index."""
        if index < 0 or index >= len(self._collections):
            return
        col = self._collections[index]
        self.query_one("#col-name", Input).value = col.name
        self.query_one("#col-desc", Input).value = col.description

    def _get_selected_index(self) -> int | None:
        """Return the currently highlighted list index, or None if nothing is selected."""
        list_view = self.query_one("#col-list", ListView)
        idx = list_view.index
        if idx is None or idx < 0 or idx >= len(self._collections):
            return None
        return idx

    def _notify_warning(self, message: str, *, next_step: str) -> None:
        """Emit a standardized collections warning with actionable next step."""
        self.notify(
            build_actionable_warning(message, next_step=next_step),
            title="Collections",
            severity="warning",
        )

    def action_cancel(self) -> None:
        """Dismiss the modal without saving changes."""
        self.dismiss(None)

    @on(ListView.Highlighted, "#col-list")
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle list highlight changes by populating the form with the selected collection."""
        list_view = self.query_one("#col-list", ListView)
        idx = list_view.index
        if idx is not None:
            self._populate_form(idx)

    @on(Button.Pressed, "#col-create")
    def on_create_pressed(self) -> None:
        """Create a new collection from the form inputs after validating name and limits."""
        name = self.query_one("#col-name", Input).value.strip()
        if not name:
            self._notify_warning(
                "Collection name cannot be empty",
                next_step="enter a name, then press Create",
            )
            return
        if any(c.name == name for c in self._collections):
            self._notify_warning(
                f"Collection '{name}' already exists",
                next_step="choose a different name or rename the existing collection",
            )
            return
        if len(self._collections) >= MAX_COLLECTIONS:
            self._notify_warning(
                f"Collection limit reached ({MAX_COLLECTIONS})",
                next_step="delete an unused collection, then create a new one",
            )
            return
        desc = self.query_one("#col-desc", Input).value.strip()
        self._collections.append(
            PaperCollection(
                name=name,
                description=desc,
                created=datetime.now().isoformat(),
            )
        )
        self._refresh_list()

    @on(Button.Pressed, "#col-rename")
    def on_rename_pressed(self) -> None:
        """Rename the selected collection and update its description from the form inputs."""
        idx = self._get_selected_index()
        if idx is None:
            self._notify_warning(
                "No collection is selected",
                next_step="highlight a collection in the list, then press Rename",
            )
            return
        name = self.query_one("#col-name", Input).value.strip()
        if not name:
            self._notify_warning(
                "Collection name cannot be empty",
                next_step="enter a name, then press Rename",
            )
            return
        desc = self.query_one("#col-desc", Input).value.strip()
        self._collections[idx].name = name
        self._collections[idx].description = desc
        self._refresh_list()

    @on(Button.Pressed, "#col-delete")
    def on_delete_pressed(self) -> None:
        """Delete the currently selected collection from the list."""
        idx = self._get_selected_index()
        if idx is None:
            self._notify_warning(
                "No collection is selected",
                next_step="highlight a collection in the list, then press Delete",
            )
            return
        self._collections.pop(idx)
        self._refresh_list()

    @on(Button.Pressed, "#col-view")
    def on_view_pressed(self) -> None:
        """Open the CollectionViewModal for the selected collection's papers."""
        idx = self._get_selected_index()
        if idx is None:
            self._notify_warning(
                "No collection is selected",
                next_step="highlight a collection in the list, then press View",
            )
            return
        col = self._collections[idx]
        self.app.push_screen(
            CollectionViewModal(col, self._papers_by_id),
            callback=self._on_view_result,
        )

    def _on_view_result(self, result: PaperCollection | None) -> None:
        """Handle the result from CollectionViewModal by updating the modified collection."""
        if result is not None:
            # Find and update the collection
            for i, c in enumerate(self._collections):
                if c.name == result.name:
                    self._collections[i] = result
                    break
            self._refresh_list()

    @on(Button.Pressed, "#col-save")
    def on_save_pressed(self) -> None:
        """Dismiss the modal with a save signal to persist collection changes."""
        self.dismiss("save")

    @on(Button.Pressed, "#col-close")
    def on_close_pressed(self) -> None:
        """Dismiss the modal without saving changes."""
        self.dismiss(None)

    @property
    def collections(self) -> list[PaperCollection]:
        """Return the current list of collections, including any unsaved edits."""
        return self._collections


class CollectionViewModal(ModalScreen[PaperCollection | None]):
    """Modal for viewing and editing papers in a collection."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    CollectionViewModal {
        align: center middle;
    }

    #colview-dialog {
        width: 70;
        height: 65%;
        min-height: 15;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #colview-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #colview-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #colview-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #colview-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        collection: PaperCollection,
        papers_by_id: dict[str, Paper] | None = None,
    ) -> None:
        """Initialize with a deep-copied collection and an optional paper lookup."""
        super().__init__()
        self._collection = PaperCollection(
            name=collection.name,
            description=collection.description,
            paper_ids=list(collection.paper_ids),
            created=collection.created,
        )
        self._papers_by_id = papers_by_id or {}

    def compose(self) -> ComposeResult:
        """Yield the collection view dialog with a paper list and remove/done buttons."""
        count = len(self._collection.paper_ids)
        title = f"{self._collection.name} ({count} paper{'s' if count != 1 else ''})"
        with Vertical(id="colview-dialog"):
            yield Label(title, id="colview-title")
            yield ListView(id="colview-list")
            with Horizontal(id="colview-buttons"):
                yield Button("Remove Selected", variant="default", id="colview-remove")
                yield Button("Done", variant="primary", id="colview-done")

    def on_mount(self) -> None:
        """Populate the paper list on mount."""
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Clear and repopulate the paper list view from the collection's paper IDs."""
        list_view = self.query_one("#colview-list", ListView)
        list_view.clear()
        for pid in self._collection.paper_ids:
            paper = self._papers_by_id.get(pid)
            label = paper.title if paper else pid
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0

    def action_cancel(self) -> None:
        """Dismiss the modal without applying paper removals."""
        self.dismiss(None)

    def _notify_warning(self, message: str, *, next_step: str) -> None:
        """Emit a standardized collection warning with actionable next step."""
        self.notify(
            build_actionable_warning(message, next_step=next_step),
            title="Collection",
            severity="warning",
        )

    @on(Button.Pressed, "#colview-remove")
    def on_remove_pressed(self) -> None:
        """Remove the highlighted paper from the collection and refresh the list."""
        list_view = self.query_one("#colview-list", ListView)
        idx = list_view.index
        if idx is None or idx < 0 or idx >= len(self._collection.paper_ids):
            self._notify_warning(
                "No paper is selected",
                next_step="highlight a paper in the list, then press Remove Selected",
            )
            return
        self._collection.paper_ids.pop(idx)
        self._refresh_list()
        # Update title
        count = len(self._collection.paper_ids)
        self.query_one("#colview-title", Label).update(
            f"{self._collection.name} ({count} paper{'s' if count != 1 else ''})"
        )

    @on(Button.Pressed, "#colview-done")
    def on_done_pressed(self) -> None:
        """Dismiss the modal and return the updated collection."""
        self.dismiss(self._collection)


class AddToCollectionModal(ModalScreen[str | None]):
    """Quick picker to select a collection to add papers to."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    AddToCollectionModal {
        align: center middle;
    }

    #addcol-dialog {
        width: 52;
        height: 50%;
        min-height: 12;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #addcol-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #addcol-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #addcol-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #addcol-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, collections: list[PaperCollection]) -> None:
        """Initialize with the list of available collections to choose from."""
        super().__init__()
        self._collections = collections

    def compose(self) -> ComposeResult:
        """Yield the collection picker dialog with a selectable list and cancel button."""
        with Vertical(id="addcol-dialog"):
            yield Label("Add to Collection", id="addcol-title")
            yield ListView(id="addcol-list")
            with Horizontal(id="addcol-buttons"):
                yield Button("Cancel (Esc/q)", variant="default", id="addcol-cancel")

    def on_mount(self) -> None:
        """Populate the list view with available collections on mount."""
        list_view = self.query_one("#addcol-list", ListView)
        for col in self._collections:
            count = len(col.paper_ids)
            label = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0

    def action_cancel(self) -> None:
        """Dismiss the picker without selecting a collection."""
        self.dismiss(None)

    @on(ListView.Selected, "#addcol-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection by dismissing with the chosen collection name."""
        list_view = self.query_one("#addcol-list", ListView)
        idx = list_view.index
        if idx is not None and 0 <= idx < len(self._collections):
            self.dismiss(self._collections[idx].name)

    @on(Button.Pressed, "#addcol-cancel")
    def on_cancel_pressed(self) -> None:
        """Handle the cancel button press by dismissing without a selection."""
        self.dismiss(None)
