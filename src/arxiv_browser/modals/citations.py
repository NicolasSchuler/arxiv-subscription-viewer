"""Recommendation and citation graph modals."""

from __future__ import annotations

import logging
import sqlite3
import webbrowser
from collections.abc import Callable
from typing import cast

import httpx
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ListItem, ListView, Static

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.app_protocols import TaskTrackingApp
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text, truncate_text
from arxiv_browser.semantic_scholar import CitationEntry
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)

RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal


class RecommendationListItem(ListItem):
    """A list item for the recommendations screen that stores a paper reference."""

    def __init__(self, paper: Paper, *children, **kwargs) -> None:
        """Initialize with an associated paper reference."""
        super().__init__(*children, **kwargs)
        self.paper = paper


class RecommendationsScreen(ModalBase[str | None]):
    """Modal screen displaying similar papers and allowing the user to jump to one.

    Accepts a target paper and a ranked list of ``(paper, score)`` pairs.
    When ``s2_available`` is ``True``, an inline source toggle bar is shown
    so the user can switch between local TF-IDF and Semantic Scholar
    recommendations without a separate pre-flight modal.

    Dismisses with the arXiv ID of the selected paper, ``None`` if the
    user cancels, or ``"switch:<source>"`` to request a source change.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("q", "cancel", "Close"),
        Binding("enter", "select", "Select"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("l", "switch_local", "Local", show=False),
        Binding("s", "switch_s2", "S2", show=False),
    ]

    CSS = """
    RecommendationsScreen {
        align: center middle;
    }

    #recommendations-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #recommendations-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
    }

    #source-bar {
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    #source-bar Button {
        margin: 0 1;
    }

    #recommendations-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #recommendations-list > ListItem {
        padding: 0 1;
    }

    #recommendations-list > ListItem.--highlight {
        background: $th-highlight;
    }

    .rec-title {
        color: $th-text;
    }

    .rec-meta {
        color: $th-muted;
    }

    .rec-score {
        color: $th-green;
        text-style: bold;
    }

    #recommendations-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(
        self,
        target_paper: Paper,
        similar_papers: list[tuple[Paper, float]],
        source: str = "local",
        s2_available: bool = False,
    ) -> None:
        """Initialize with the target paper and its ranked similar papers."""
        super().__init__()
        self._target_paper = target_paper
        self._similar_papers = similar_papers
        self._source = source
        self._s2_available = s2_available

    def compose(self) -> ComposeResult:
        """Yield title label, optional source toggle, paper list, and buttons."""
        with Vertical(id="recommendations-dialog"):
            truncated_title = truncate_text(self._target_paper.title, RECOMMENDATION_TITLE_MAX_LEN)
            yield Label(f"Similar to: {truncated_title}", id="recommendations-title")
            if self._s2_available:
                with Horizontal(id="source-bar"):
                    yield Button(
                        "Local (TF-IDF)",
                        variant="primary" if self._source == "local" else "default",
                        id="source-local-btn",
                    )
                    yield Button(
                        "Semantic Scholar",
                        variant="primary" if self._source == "s2" else "default",
                        id="source-s2-btn",
                    )
            yield ListView(id="recommendations-list")
            with Horizontal(id="recommendations-buttons"):
                yield Button("Close (Esc/q)", variant="default", id="close-btn")
                yield Button("Go to Paper (Enter)", variant="primary", id="select-btn")

    def on_mount(self) -> None:
        """Populate the list view with similar papers and focus it."""
        list_view = self.query_one("#recommendations-list", ListView)
        green = theme_colors_for(self)["green"]
        for paper, score in self._similar_papers:
            safe_title = escape_rich_text(paper.title)
            safe_categories = escape_rich_text(paper.categories)
            item = RecommendationListItem(
                paper,
                Static(f"[bold]{safe_title}[/]", classes="rec-title"),
                Static(
                    f"[dim]{paper.arxiv_id}[/] | {safe_categories} | [{green}]{score:.0%}[/] match",
                    classes="rec-meta",
                ),
            )
            list_view.mount(item)
        if list_view.children:
            list_view.index = 0
        list_view.focus()

    def action_select(self) -> None:
        """Dismiss with the highlighted paper's arxiv_id, or None if nothing is highlighted."""
        list_view = self.query_one("#recommendations-list", ListView)
        if isinstance(list_view.highlighted_child, RecommendationListItem):
            self.dismiss(list_view.highlighted_child.paper.arxiv_id)
        else:
            self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move the highlight down in the recommendations list."""
        self.query_one("#recommendations-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the highlight up in the recommendations list."""
        self.query_one("#recommendations-list", ListView).action_cursor_up()

    @on(Button.Pressed, "#close-btn")
    def on_close_pressed(self) -> None:
        """Handle the close button press."""
        self.action_cancel()

    @on(Button.Pressed, "#select-btn")
    def on_select_pressed(self) -> None:
        """Handle the select button press."""
        self.action_select()

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection by dismissing with the chosen paper's arxiv_id."""
        if isinstance(event.item, RecommendationListItem):
            self.dismiss(event.item.paper.arxiv_id)

    # -- Source toggle (inline replacement for RecommendationSourceModal) --

    def action_switch_local(self) -> None:
        """Switch to local recommendations via keybinding."""
        if self._s2_available and self._source != "local":
            self.dismiss("switch:local")

    def action_switch_s2(self) -> None:
        """Switch to Semantic Scholar recommendations via keybinding."""
        if self._s2_available and self._source != "s2":
            self.dismiss("switch:s2")

    @on(Button.Pressed, "#source-local-btn")
    def on_source_local_pressed(self) -> None:
        """Handle the Local source button press."""
        self.action_switch_local()

    @on(Button.Pressed, "#source-s2-btn")
    def on_source_s2_pressed(self) -> None:
        """Handle the Semantic Scholar source button press."""
        self.action_switch_s2()


# ============================================================================
# Citation Graph Modal
# ============================================================================


class CitationGraphListItem(ListItem):
    """A list item in the citation graph modal storing a CitationEntry."""

    def __init__(
        self,
        entry: CitationEntry,
        *children,
        is_local: bool = False,
        **kwargs,
    ) -> None:
        """Initialize with a citation entry and a flag for local availability.

        Args:
            entry: The ``CitationEntry`` (paper metadata from Semantic Scholar)
                to associate with this list item.
            *children: Child widgets to pass to the ``ListItem`` base class.
            is_local: ``True`` when *entry* has an arXiv ID that is present in
                the currently loaded paper list, enabling the "go to local
                paper" (``g``) action.
            **kwargs: Additional keyword arguments forwarded to ``ListItem``.
        """
        super().__init__(*children, **kwargs)
        self.entry = entry
        self.is_local = is_local


class CitationGraphScreen(ModalBase[str | None]):
    """Modal screen for exploring citation graphs with depth-limited drill-down.

    Displays a two-panel layout: **References** (papers this paper cites) on
    the left and **Cited By** (papers that cite this paper) on the right.
    The user can drill into any entry to explore its own citation graph; each
    drill-down level is pushed onto ``_stack`` so that ``Esc``/``q`` steps
    back one level at a time.

    ``_active_panel`` tracks which panel (``"refs"`` or ``"cites"``) currently
    has keyboard focus; ``Tab`` switches between them.

    Dismisses with the arXiv ID of a local paper to jump to (via ``g``), or
    ``None`` when the user closes without navigating.
    """

    BINDINGS = [
        Binding("escape", "back_or_close", "Back / Close"),
        Binding("q", "back_or_close", "Back / Close"),
        Binding("enter", "drill_down", "Drill down"),
        Binding("o", "open_url", "Open in browser", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("tab", "switch_panel", "Switch panel", show=False),
        Binding("g", "go_to_local", "Go to paper", show=False),
    ]

    CSS = """
    CitationGraphScreen {
        align: center middle;
    }

    #citation-graph-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-purple;
        padding: 0 2;
    }

    #citation-graph-breadcrumb {
        text-style: bold;
        color: $th-purple;
        margin-bottom: 1;
        height: auto;
    }

    #citation-graph-panels {
        height: 1fr;
    }

    .citation-panel {
        width: 1fr;
        height: 1fr;
    }

    .citation-panel-title {
        text-style: bold;
        color: $th-accent;
        height: auto;
    }

    .citation-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    .citation-list.active-panel {
        border-left: tall $th-purple;
    }

    .citation-list > ListItem {
        padding: 0 1;
    }

    .citation-list > ListItem.--highlight {
        background: $th-highlight;
    }

    .cite-title {
        color: $th-text;
    }

    .cite-meta {
        color: $th-muted;
    }

    #citation-graph-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #citation-graph-status {
        color: $th-muted;
        height: auto;
    }
    """

    def __init__(
        self,
        root_title: str,
        root_paper_id: str,
        references: list[CitationEntry],
        citations: list[CitationEntry],
        fetch_callback: Callable,
        local_arxiv_ids: frozenset[str],
    ) -> None:
        """Initialize with root paper data, a fetch callback, and the local paper set.

        Args:
            root_title: Display title of the root paper shown in the breadcrumb.
            root_paper_id: Semantic Scholar paper ID (or ``"ARXIV:<id>"`` fallback)
                used as the starting node.
            references: Papers that the root paper cites.
            citations: Papers that cite the root paper.
            fetch_callback: Async callable ``(s2_paper_id: str) ->
                (refs, cites)`` invoked when the user drills into an entry.
            local_arxiv_ids: Frozenset of arXiv IDs currently loaded in the
                app, used to mark entries that can be jumped to with ``g``.
        """
        super().__init__()
        self._root_title = root_title
        self._root_paper_id = root_paper_id
        self._fetch_callback = fetch_callback
        self._local_arxiv_ids = local_arxiv_ids
        # Stack: list of (paper_id, title, refs, cites)
        self._stack: list[tuple[str, str, list, list]] = []
        # Current state
        self._current_refs = references
        self._current_cites = citations
        self._current_title = root_title
        self._current_paper_id = root_paper_id
        self._active_panel: str = "refs"  # "refs" or "cites"
        self._loading = False

    def compose(self) -> ComposeResult:
        """Yield breadcrumb, side-by-side references/citations panels, status bar, and buttons."""
        with Vertical(id="citation-graph-dialog"):
            yield Static("", id="citation-graph-breadcrumb")
            with Horizontal(id="citation-graph-panels"):
                with Vertical(classes="citation-panel"):
                    yield Static("", id="refs-title", classes="citation-panel-title")
                    yield ListView(id="refs-list", classes="citation-list active-panel")
                with Vertical(classes="citation-panel"):
                    yield Static("", id="cites-title", classes="citation-panel-title")
                    yield ListView(id="cites-list", classes="citation-list")
            yield Static("", id="citation-graph-status")
            with Horizontal(id="citation-graph-buttons"):
                yield Button("Close (Esc/q)", variant="default", id="cg-close-btn")
                yield Button("Drill Down (Enter)", variant="primary", id="cg-drill-btn")

    def on_mount(self) -> None:
        """Populate both citation lists and focus the references panel."""
        self._populate_lists()
        self._update_breadcrumb()
        refs_list = self.query_one("#refs-list", ListView)
        refs_list.focus()

    def _build_citation_item(self, entry: CitationEntry) -> CitationGraphListItem:
        """Build a list item widget for a single citation graph entry."""
        colors = theme_colors_for(self)
        is_local = entry.arxiv_id != "" and entry.arxiv_id in self._local_arxiv_ids
        safe_title = escape_rich_text(entry.title)
        local_badge = f" [{colors['green']}]\\[LOCAL][/]" if is_local else ""
        year_str = str(entry.year) if entry.year else "?"
        authors_short = truncate_text(entry.authors, 50) if entry.authors else ""
        return CitationGraphListItem(
            entry,
            Static(
                f"[bold]{safe_title}[/]{local_badge}",
                classes="cite-title",
            ),
            Static(
                f"[dim]{year_str}[/] | {escape_rich_text(authors_short)}"
                f" | [{colors['accent']}]{entry.citation_count} cites[/]",
                classes="cite-meta",
            ),
            is_local=is_local,
        )

    def _populate_lists(self) -> None:
        """Fill both list views with the current references and citations.

        Side effects (beyond clearing and mounting list items):
        - Updates the panel title labels (``#refs-title``, ``#cites-title``)
          to show the current entry counts.
        - Calls ``_update_status`` to refresh the status bar.
        """
        refs_list = self.query_one("#refs-list", ListView)
        cites_list = self.query_one("#cites-list", ListView)
        refs_list.clear()
        cites_list.clear()

        for entry in self._current_refs:
            refs_list.mount(self._build_citation_item(entry))
        for entry in self._current_cites:
            cites_list.mount(self._build_citation_item(entry))

        if refs_list.children:
            refs_list.index = 0
        if cites_list.children:
            cites_list.index = 0

        # Update panel titles
        self.query_one("#refs-title", Static).update(f"References ({len(self._current_refs)})")
        self.query_one("#cites-title", Static).update(f"Cited By ({len(self._current_cites)})")

        # Update status
        self._update_status()

    def _update_breadcrumb(self) -> None:
        """Update the breadcrumb trail."""
        parts = [truncate_text(t, 40) for _, t, _, _ in self._stack]
        parts.append(truncate_text(self._current_title, 40))
        from arxiv_browser._ascii import is_ascii_mode

        arrow = " -> " if is_ascii_mode() else " \u2192 "
        purple = theme_colors_for(self)["purple"]
        breadcrumb = arrow.join(f"[{purple}]{escape_rich_text(p)}[/]" for p in parts)
        self.query_one("#citation-graph-breadcrumb", Static).update(f"Citation Graph: {breadcrumb}")

    def _update_status(self) -> None:
        """Update status bar with navigation hints."""
        if self._loading:
            self.query_one("#citation-graph-status", Static).update(
                "[dim]Loading citation graph...[/]"
            )
            return
        active = self._active_panel
        panel_name = "references" if active == "refs" else "cited by"
        depth = len(self._stack)
        depth_str = f" [dim](depth {depth})[/]" if depth > 0 else ""
        purple = theme_colors_for(self)["purple"]
        self.query_one("#citation-graph-status", Static).update(
            f"[dim]Tab: switch panel | Enter: drill down | "
            f"o: open | g: go to local | Esc/q: back[/]"
            f"  Active: [{purple}]{panel_name}[/]{depth_str}"
        )

    def _get_active_list(self) -> ListView:
        """Return the currently active list view."""
        list_id = "#refs-list" if self._active_panel == "refs" else "#cites-list"
        return self.query_one(list_id, ListView)

    def _get_highlighted_entry(self) -> CitationGraphListItem | None:
        """Get the highlighted item from the active list."""
        lv = self._get_active_list()
        child = lv.highlighted_child
        if isinstance(child, CitationGraphListItem):
            return child
        return None

    def action_back_or_close(self) -> None:
        """Pop one level or close the modal."""
        if self._stack:
            paper_id, title, refs, cites = self._stack.pop()
            self._current_paper_id = paper_id
            self._current_title = title
            self._current_refs = refs
            self._current_cites = cites
            self._populate_lists()
            self._update_breadcrumb()
            self._get_active_list().focus()
        else:
            self.dismiss(None)

    async def action_drill_down(self) -> None:
        """Drill into the highlighted entry's citation graph."""
        if self._loading:
            return
        item = self._get_highlighted_entry()
        if not item:
            return
        entry = item.entry
        # Push current state
        self._stack.append(
            (
                self._current_paper_id,
                self._current_title,
                self._current_refs,
                self._current_cites,
            )
        )
        self._loading = True
        self._update_status()
        try:
            refs, cites = await self._fetch_callback(entry.s2_paper_id)
            self._current_paper_id = entry.s2_paper_id
            self._current_title = entry.title
            self._current_refs = refs
            self._current_cites = cites
            self._populate_lists()
            self._update_breadcrumb()
        except (httpx.HTTPError, OSError, sqlite3.Error):
            logger.warning(
                "Citation graph fetch failed for %s",
                entry.s2_paper_id,
                exc_info=True,
            )
            self.app.notify(
                build_actionable_error(
                    "load citation graph data",
                    why="a network, API, or local cache error occurred",
                    next_step="retry drill-down or return to browse mode and try again",
                ),
                severity="error",
            )
            # Undo the push
            self._stack.pop()
        finally:
            self._loading = False
            self._update_status()
            self._get_active_list().focus()

    def action_switch_panel(self) -> None:
        """Toggle between references and citations panels."""
        refs_list = self.query_one("#refs-list", ListView)
        cites_list = self.query_one("#cites-list", ListView)
        if self._active_panel == "refs":
            self._active_panel = "cites"
            refs_list.remove_class("active-panel")
            cites_list.add_class("active-panel")
            cites_list.focus()
        else:
            self._active_panel = "refs"
            cites_list.remove_class("active-panel")
            refs_list.add_class("active-panel")
            refs_list.focus()
        self._update_status()

    def action_open_url(self) -> None:
        """Open the highlighted entry's URL in the browser."""
        item = self._get_highlighted_entry()
        if item and item.entry.url:
            webbrowser.open(item.entry.url)

    def action_go_to_local(self) -> None:
        """If highlighted entry is local, dismiss with its arxiv_id to jump to it."""
        item = self._get_highlighted_entry()
        if item and item.is_local and item.entry.arxiv_id:
            self.dismiss(item.entry.arxiv_id)

    def action_cursor_down(self) -> None:
        """Move the highlight down in the active panel's list."""
        self._get_active_list().action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the highlight up in the active panel's list."""
        self._get_active_list().action_cursor_up()

    @on(Button.Pressed, "#cg-close-btn")
    def on_close_pressed(self) -> None:
        """Handle the close button press by navigating back or closing."""
        self.action_back_or_close()

    @on(Button.Pressed, "#cg-drill-btn")
    def on_drill_pressed(self) -> None:
        """Handle the drill-down button press by invoking the async drill-down action."""
        # Button click needs to invoke the async action; use app's tracked task
        cast(TaskTrackingApp, self.app)._track_task(self.action_drill_down())
