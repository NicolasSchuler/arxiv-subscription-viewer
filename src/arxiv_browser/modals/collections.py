"""Paper collections (reading lists) modal with manage/pick modes."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, ListItem, ListView

from arxiv_browser.action_messages import build_actionable_warning
from arxiv_browser.empty_state import (
    COLLECTION_DETAIL_EMPTY,
    COLLECTIONS_MANAGE_EMPTY,
    COLLECTIONS_PICK_EMPTY,
)
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import MAX_COLLECTIONS, Paper, PaperCollection

logger = logging.getLogger(__name__)


def _build_empty_placeholder(message: str) -> ListItem:
    """Build a disabled ListItem that communicates an empty-state hint.

    The placeholder is disabled so that keyboard navigation skips it and
    selection handlers bail out on the bounds check; the surrounding CSS
    dims it further via the ``-empty`` class.
    """
    item = ListItem(Label(f"[dim italic]{message}[/]"), classes="-empty")
    item.disabled = True
    return item


class CollectionsModal(ModalBase[str | None]):
    """Unified modal for managing and picking paper collections.

    ``mode="manage"`` (default) shows a full collection manager with
    create/rename/delete/view actions.  ``mode="pick"`` shows a simplified
    collection picker for "add paper to collection" flows.

    The manage view includes an inline detail sub-view for viewing papers in
    a collection, accessible via the "View" button (no nested modal).
    """

    BINDINGS = [
        Binding("escape", "cancel_or_back", "Cancel"),
        Binding("q", "cancel_or_back", "Cancel", show=False),
    ]

    CSS = """
    CollectionsModal {
        align: center middle;
    }

    /* ── shared dialog chrome ────────────────────────────── */
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

    /* ── manage-view ─────────────────────────────────────── */
    #manage-view { height: 1fr; }

    #col-body { height: 1fr; }

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

    #col-actions Button { margin-right: 1; }

    #col-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #col-buttons Button { margin-left: 1; }

    /* ── detail-view ─────────────────────────────────────── */
    #detail-view { height: 1fr; }

    #detail-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #detail-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #detail-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #detail-buttons Button { margin-left: 1; }

    /* ── pick-view ───────────────────────────────────────── */
    #pick-view { height: 1fr; }

    #pick-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #pick-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #pick-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #pick-buttons Button { margin-left: 1; }
    """

    def __init__(
        self,
        collections: list[PaperCollection],
        papers_by_id: dict[str, Paper] | None = None,
        *,
        mode: Literal["manage", "pick"] = "manage",
    ) -> None:
        """Initialize with deep-copied collections, optional paper lookup, and mode."""
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
        self._mode: Literal["manage", "pick"] = mode
        self._viewing_collection: PaperCollection | None = None

    # ── compose ──────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        """Yield all three sub-views; visibility is toggled in on_mount."""
        with Vertical(id="col-dialog"):
            yield Label("Collections Manager", id="col-title")

            # manage-view
            with Vertical(id="manage-view"):
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

            # detail-view (papers in a collection)
            with Vertical(id="detail-view"):
                yield Label("", id="detail-title")
                yield ListView(id="detail-list")
                with Horizontal(id="detail-buttons"):
                    yield Button("Remove Selected", variant="default", id="detail-remove")
                    yield Button("Back", variant="primary", id="detail-back")

            # pick-view (quick collection picker)
            with Vertical(id="pick-view"):
                yield Label("Add to Collection", id="pick-title")
                yield ListView(id="pick-list")
                with Horizontal(id="pick-buttons"):
                    yield Button("Cancel (Esc/q)", variant="default", id="pick-cancel")

    # ── lifecycle ────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Show the correct initial view and populate lists."""
        self._show_view(self._mode)
        if self._mode == "manage":
            self._refresh_manage_list()
            self._focus_widget("#col-name")
        else:
            self._refresh_pick_list()

    # ── view switching ───────────────────────────────────────────────

    def _show_view(self, view: Literal["manage", "detail", "pick"]) -> None:
        """Toggle visibility of the three sub-views."""
        self.query_one("#manage-view", Vertical).display = view == "manage"
        self.query_one("#detail-view", Vertical).display = view == "detail"
        self.query_one("#pick-view", Vertical).display = view == "pick"
        # Update dialog title
        titles = {
            "manage": "Collections Manager",
            "detail": "",  # set separately with collection name
            "pick": "Add to Collection",
        }
        self.query_one("#col-title", Label).update(titles[view])

    # ── manage-view helpers ──────────────────────────────────────────

    def _refresh_manage_list(self) -> None:
        """Clear and repopulate the manage-view collections list."""
        list_view = self.query_one("#col-list", ListView)
        list_view.clear()
        if not self._collections:
            list_view.mount(_build_empty_placeholder(COLLECTIONS_MANAGE_EMPTY))
            return
        for col in self._collections:
            count = len(col.paper_ids)
            label = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0
            self._populate_form(0)

    # Keep legacy alias for tests that mock _refresh_list
    _refresh_list = _refresh_manage_list

    def _populate_form(self, index: int) -> None:
        """Fill the name and description inputs from the collection at the given index."""
        if index < 0 or index >= len(self._collections):
            return
        col = self._collections[index]
        self.query_one("#col-name", Input).value = col.name
        self.query_one("#col-desc", Input).value = col.description

    def _get_selected_index(self) -> int | None:
        """Return the currently highlighted manage-list index, or None."""
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

    # ── manage-view event handlers ───────────────────────────────────

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
        self._refresh_manage_list()

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
        self._refresh_manage_list()

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
        self._refresh_manage_list()

    @on(Button.Pressed, "#col-view")
    def on_view_pressed(self) -> None:
        """Switch to the detail-view for the selected collection's papers."""
        idx = self._get_selected_index()
        if idx is None:
            self._notify_warning(
                "No collection is selected",
                next_step="highlight a collection in the list, then press View",
            )
            return
        col = self._collections[idx]
        self._viewing_collection = PaperCollection(
            name=col.name,
            description=col.description,
            paper_ids=list(col.paper_ids),
            created=col.created,
        )
        self._show_detail_view()

    @on(Button.Pressed, "#col-save")
    def on_save_pressed(self) -> None:
        """Dismiss the modal with a save signal to persist collection changes."""
        self.dismiss("save")

    @on(Button.Pressed, "#col-close")
    def on_close_pressed(self) -> None:
        """Dismiss the modal without saving changes."""
        self.dismiss(None)

    # ── detail-view helpers & handlers ───────────────────────────────

    def _show_detail_view(self) -> None:
        """Activate the detail-view and populate it with the viewed collection."""
        assert self._viewing_collection is not None
        col = self._viewing_collection
        count = len(col.paper_ids)
        title = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
        self.query_one("#detail-title", Label).update(title)
        self._show_view("detail")
        self._refresh_detail_list()

    def _refresh_detail_list(self) -> None:
        """Clear and repopulate the detail-view paper list."""
        assert self._viewing_collection is not None
        list_view = self.query_one("#detail-list", ListView)
        list_view.clear()
        if not self._viewing_collection.paper_ids:
            list_view.mount(_build_empty_placeholder(COLLECTION_DETAIL_EMPTY))
            return
        for pid in self._viewing_collection.paper_ids:
            paper = self._papers_by_id.get(pid)
            label = paper.title if paper else pid
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0

    @on(Button.Pressed, "#detail-remove")
    def on_detail_remove_pressed(self) -> None:
        """Remove the highlighted paper from the viewed collection."""
        assert self._viewing_collection is not None
        list_view = self.query_one("#detail-list", ListView)
        idx = list_view.index
        if idx is None or idx < 0 or idx >= len(self._viewing_collection.paper_ids):
            self._notify_warning(
                "No paper is selected",
                next_step="highlight a paper in the list, then press Remove Selected",
            )
            return
        self._viewing_collection.paper_ids.pop(idx)
        self._refresh_detail_list()
        count = len(self._viewing_collection.paper_ids)
        self.query_one("#detail-title", Label).update(
            f"{self._viewing_collection.name} ({count} paper{'s' if count != 1 else ''})"
        )

    @on(Button.Pressed, "#detail-back")
    def on_detail_back_pressed(self) -> None:
        """Return to manage-view, applying any paper removals."""
        if self._viewing_collection is not None:
            for i, c in enumerate(self._collections):
                if c.name == self._viewing_collection.name:
                    self._collections[i] = self._viewing_collection
                    break
            self._viewing_collection = None
        self._show_view("manage")
        self._refresh_manage_list()

    # ── pick-view helpers & handlers ─────────────────────────────────

    def _refresh_pick_list(self) -> None:
        """Populate the pick-view collection list."""
        list_view = self.query_one("#pick-list", ListView)
        list_view.clear()
        if not self._collections:
            list_view.mount(_build_empty_placeholder(COLLECTIONS_PICK_EMPTY))
            return
        for col in self._collections:
            count = len(col.paper_ids)
            label = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0

    @on(ListView.Selected, "#pick-list")
    def on_pick_list_selected(self, event: ListView.Selected) -> None:
        """Handle pick-list item selection by dismissing with the chosen collection name."""
        list_view = self.query_one("#pick-list", ListView)
        idx = list_view.index
        if idx is not None and 0 <= idx < len(self._collections):
            self.dismiss(self._collections[idx].name)

    @on(Button.Pressed, "#pick-cancel")
    def on_pick_cancel_pressed(self) -> None:
        """Handle the cancel button press in pick mode."""
        self.dismiss(None)

    # ── shared escape/q handler ──────────────────────────────────────

    def action_cancel_or_back(self) -> None:
        """Escape returns to manage-view from detail-view, otherwise dismisses."""
        if self._viewing_collection is not None:
            self.on_detail_back_pressed()
        else:
            self.dismiss(None)

    # ── public API ───────────────────────────────────────────────────

    @property
    def collections(self) -> list[PaperCollection]:
        """Return the current list of collections, including any unsaved edits."""
        return self._collections
