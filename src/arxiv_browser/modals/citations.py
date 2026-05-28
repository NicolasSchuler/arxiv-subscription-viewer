"""Recommendation and citation graph modals."""

from __future__ import annotations

import logging
import sqlite3
import webbrowser
from collections.abc import Callable
from typing import Literal, cast

import httpx
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ListItem, ListView, Static, Tree

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.app_protocols import TaskTrackingApp
from arxiv_browser.citation_genealogy import (
    GenealogyContext,
    GenealogyDirection,
    GenealogyNode,
    GenealogyOptions,
    GenealogyRoot,
    build_genealogy_tree,
)
from arxiv_browser.empty_state import (
    CITATIONS_CITES_EMPTY,
    CITATIONS_REFS_EMPTY,
)
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text, truncate_text
from arxiv_browser.semantic_scholar import CitationEntry
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)

RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal
CitationViewMode = Literal["graph", "ancestors", "descendants"]


def _build_empty_placeholder(message: str) -> ListItem:
    """Build a disabled ListItem that communicates an empty-state hint."""
    item = ListItem(Label(f"[dim italic]{message}[/]"), classes="-empty")
    item.disabled = True
    return item


def _format_genealogy_label(node: GenealogyNode) -> str:
    """Return compact display text for one genealogy tree node."""
    title = truncate_text(node.paper.title, 72)
    year = str(node.paper.year) if node.paper.year is not None else "?"
    parts = [f"{title} ({year}; {node.paper.citation_count} cites)"]
    badges: list[str] = []
    if node.is_target:
        badges.append("target")
    if node.is_local:
        badges.append("local")
    if node.is_starred:
        badges.append("starred")
    if node.repeated:
        badges.append("repeat")
    if node.truncated:
        badges.append("more")
    if badges:
        parts.append(f"({'/'.join(badges)})")
    return " ".join(parts)


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
        Binding("1", "show_graph", "Graph", show=False),
        Binding("a", "show_ancestors", "Ancestors", show=False),
        Binding("d", "show_descendants", "Descendants", show=False),
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

    #citation-mode-buttons {
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    #citation-mode-buttons Button {
        margin: 0 1;
    }

    #citation-graph-panels {
        height: 1fr;
    }

    #genealogy-tree {
        height: 1fr;
        background: $th-panel;
        border-left: tall $th-purple;
        padding: 0 1;
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
        root_arxiv_id = (
            root_paper_id.removeprefix("ARXIV:") if root_paper_id.startswith("ARXIV:") else ""
        )
        self._view_mode: CitationViewMode = "graph"
        self._genealogy_root = GenealogyRoot(
            paper_id=root_paper_id,
            title=root_title,
            arxiv_id=root_arxiv_id,
        )
        self._genealogy_options = GenealogyOptions()
        self._genealogy_cache: dict[GenealogyDirection, GenealogyNode] = {}
        self._starred_arxiv_ids: frozenset[str] = frozenset()

    def configure_genealogy(
        self,
        root: GenealogyRoot,
        starred_arxiv_ids: frozenset[str] = frozenset(),
    ) -> None:
        """Set richer root/user context for the genealogy tree modes."""
        self._genealogy_root = root
        self._starred_arxiv_ids = starred_arxiv_ids
        self._genealogy_cache.clear()

    def compose(self) -> ComposeResult:
        """Yield breadcrumb, side-by-side references/citations panels, status bar, and buttons."""
        with Vertical(id="citation-graph-dialog"):
            yield Static("", id="citation-graph-breadcrumb")
            with Horizontal(id="citation-mode-buttons"):
                yield Button("Graph", variant="primary", id="cg-mode-graph")
                yield Button("Ancestors", variant="default", id="cg-mode-ancestors")
                yield Button("Descendants", variant="default", id="cg-mode-descendants")
            with Horizontal(id="citation-graph-panels"):
                with Vertical(classes="citation-panel"):
                    yield Static("", id="refs-title", classes="citation-panel-title")
                    yield ListView(id="refs-list", classes="citation-list active-panel")
                with Vertical(classes="citation-panel"):
                    yield Static("", id="cites-title", classes="citation-panel-title")
                    yield ListView(id="cites-list", classes="citation-list")
            yield Tree("", id="genealogy-tree")
            yield Static("", id="citation-graph-status")
            with Horizontal(id="citation-graph-buttons"):
                yield Button("Close (Esc/q)", variant="default", id="cg-close-btn")
                yield Button("Drill Down (Enter)", variant="primary", id="cg-drill-btn")

    def on_mount(self) -> None:
        """Populate both citation lists and focus the references panel."""
        self._populate_lists()
        self._update_breadcrumb()
        self.query_one("#genealogy-tree", Tree).display = False
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

        if self._current_refs:
            for entry in self._current_refs:
                refs_list.mount(self._build_citation_item(entry))
        else:
            refs_list.mount(_build_empty_placeholder(CITATIONS_REFS_EMPTY))

        if self._current_cites:
            for entry in self._current_cites:
                cites_list.mount(self._build_citation_item(entry))
        else:
            cites_list.mount(_build_empty_placeholder(CITATIONS_CITES_EMPTY))

        if self._current_refs and refs_list.children:
            refs_list.index = 0
        if self._current_cites and cites_list.children:
            cites_list.index = 0

        # Update panel titles
        self.query_one("#refs-title", Static).update(f"References ({len(self._current_refs)})")
        self.query_one("#cites-title", Static).update(f"Cited By ({len(self._current_cites)})")

        # Update status
        self._update_status()

    def _update_breadcrumb(self) -> None:
        """Update the breadcrumb trail."""
        if self._view_mode != "graph":
            label = "Ancestors" if self._view_mode == "ancestors" else "Descendants"
            title = truncate_text(self._genealogy_root.title, 56)
            purple = theme_colors_for(self)["purple"]
            self.query_one("#citation-graph-breadcrumb", Static).update(
                f"Citation Genealogy: [{purple}]{label}[/] of {escape_rich_text(title)}"
            )
            return
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
            if self._view_mode == "graph":
                loading_message = "Loading citation graph..."
            else:
                loading_message = f"Loading {self._view_mode} genealogy..."
            self.query_one("#citation-graph-status", Static).update(f"[dim]{loading_message}[/]")
            return
        if self._view_mode != "graph":
            purple = theme_colors_for(self)["purple"]
            self.query_one("#citation-graph-status", Static).update(
                f"[dim]1: graph | a: ancestors | d: descendants | "
                f"Enter/click: go/open | o: open | g: go to local | Esc/q: close[/]"
                f"  Active: [{purple}]{self._view_mode}[/]"
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

    def _get_genealogy_tree(self) -> Tree:
        """Return the genealogy tree widget."""
        return self.query_one("#genealogy-tree", Tree)

    def _get_highlighted_genealogy_node(self) -> GenealogyNode | None:
        """Return the selected genealogy node payload, if the cursor is on one."""
        cursor_node = self._get_genealogy_tree().cursor_node
        if cursor_node is None:
            return None
        data = cursor_node.data
        return data if isinstance(data, GenealogyNode) else None

    def _set_view_mode(self, mode: CitationViewMode) -> None:
        """Toggle graph/tree widgets and mode buttons."""
        self._view_mode = mode
        graph_visible = mode == "graph"
        self.query_one("#citation-graph-panels", Horizontal).display = graph_visible
        self._get_genealogy_tree().display = not graph_visible
        self.query_one("#cg-mode-graph", Button).variant = (
            "primary" if mode == "graph" else "default"
        )
        self.query_one("#cg-mode-ancestors", Button).variant = (
            "primary" if mode == "ancestors" else "default"
        )
        self.query_one("#cg-mode-descendants", Button).variant = (
            "primary" if mode == "descendants" else "default"
        )
        self._update_breadcrumb()
        self._update_status()

    def action_show_graph(self) -> None:
        """Switch back to the two-panel citation graph view."""
        self._set_view_mode("graph")
        self._get_active_list().focus()

    async def action_show_ancestors(self) -> None:
        """Build or show the ancestor genealogy tree."""
        self._show_genealogy_worker("ancestors")

    async def action_show_descendants(self) -> None:
        """Build or show the descendant genealogy tree."""
        self._show_genealogy_worker("descendants")

    def _set_loading_state(self, loading: bool) -> None:
        self._loading = loading
        try:
            self.query_one("#citation-graph-dialog").loading = loading
        except Exception:
            pass

    @work(exclusive=True, group="citation-genealogy", exit_on_error=False)
    async def _show_genealogy_worker(self, direction: GenealogyDirection) -> None:
        """Worker wrapper for citation genealogy fetches."""
        await self._show_genealogy(direction)

    async def _show_genealogy(self, direction: GenealogyDirection) -> None:
        """Build the requested genealogy view, respecting S2 cache/fetch bounds."""
        if self._loading:
            return
        self._set_view_mode(direction)
        cached = self._genealogy_cache.get(direction)
        if cached is None:
            self._set_loading_state(True)
            self._update_status()
            try:
                cached = await build_genealogy_tree(
                    self._genealogy_root,
                    direction,
                    self._fetch_callback,
                    self._genealogy_options,
                    GenealogyContext(
                        local_arxiv_ids=self._local_arxiv_ids,
                        starred_arxiv_ids=self._starred_arxiv_ids,
                    ),
                )
                self._genealogy_cache[direction] = cached
            except (httpx.HTTPError, OSError, sqlite3.Error):
                logger.warning(
                    "Citation genealogy fetch failed for %s/%s",
                    self._genealogy_root.paper_id,
                    direction,
                    exc_info=True,
                )
                self.app.notify(
                    build_actionable_error(
                        "load citation genealogy data",
                        why="a network, API, or local cache error occurred",
                        next_step="retry the genealogy mode or return to the graph view",
                    ),
                    severity="error",
                )
                self._set_view_mode("graph")
                return
            finally:
                self._set_loading_state(False)
                self._update_status()
        self._render_genealogy_tree(cached)
        self._get_genealogy_tree().focus()

    def _render_genealogy_tree(self, root: GenealogyNode) -> None:
        """Render a GenealogyNode tree into the Textual Tree widget."""
        tree = self._get_genealogy_tree()
        root_data = root if root.paper.paper_id else None
        tree.reset(_format_genealogy_label(root), data=root_data)
        tree.root.expand()
        for child in root.children:
            self._add_genealogy_tree_node(tree.root, child)

    def _add_genealogy_tree_node(self, parent, node: GenealogyNode) -> None:
        child = parent.add(_format_genealogy_label(node), data=node, expand=True)
        for grandchild in node.children:
            self._add_genealogy_tree_node(child, grandchild)

    def _activate_genealogy_node(self, node: GenealogyNode | None) -> None:
        """Jump to local nodes or open remote genealogy nodes in a browser."""
        if node is None:
            return
        if node.is_local and node.paper.arxiv_id:
            self.dismiss(node.paper.arxiv_id)
        elif node.paper.url:
            webbrowser.open(node.paper.url)

    def action_back_or_close(self) -> None:
        """Pop one level or close the modal."""
        if self._view_mode != "graph":
            self.dismiss(None)
            return
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
        if self._view_mode != "graph":
            self._activate_genealogy_node(self._get_highlighted_genealogy_node())
            return
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
        self._drill_down_worker(entry)

    @work(exclusive=True, group="citation-drill", exit_on_error=False)
    async def _drill_down_worker(self, entry: CitationEntry) -> None:
        """Worker wrapper for citation graph drill-down fetches."""
        self._set_loading_state(True)
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
            self._set_loading_state(False)
            self._update_status()
            self._get_active_list().focus()

    def action_switch_panel(self) -> None:
        """Toggle between references and citations panels."""
        if self._view_mode != "graph":
            self._get_genealogy_tree().focus()
            return
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
        if self._view_mode != "graph":
            node = self._get_highlighted_genealogy_node()
            if node and node.paper.url:
                webbrowser.open(node.paper.url)
            return
        item = self._get_highlighted_entry()
        if item and item.entry.url:
            webbrowser.open(item.entry.url)

    def action_go_to_local(self) -> None:
        """If highlighted entry is local, dismiss with its arxiv_id to jump to it."""
        if self._view_mode != "graph":
            node = self._get_highlighted_genealogy_node()
            if node and node.is_local and node.paper.arxiv_id:
                self.dismiss(node.paper.arxiv_id)
            return
        item = self._get_highlighted_entry()
        if item and item.is_local and item.entry.arxiv_id:
            self.dismiss(item.entry.arxiv_id)

    def action_cursor_down(self) -> None:
        """Move the highlight down in the active panel's list."""
        if self._view_mode != "graph":
            self._get_genealogy_tree().action_cursor_down()
            return
        self._get_active_list().action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the highlight up in the active panel's list."""
        if self._view_mode != "graph":
            self._get_genealogy_tree().action_cursor_up()
            return
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

    @on(Button.Pressed, "#cg-mode-graph")
    def on_mode_graph_pressed(self) -> None:
        """Handle the Graph mode button."""
        self.action_show_graph()

    @on(Button.Pressed, "#cg-mode-ancestors")
    def on_mode_ancestors_pressed(self) -> None:
        """Handle the Ancestors mode button."""
        cast(TaskTrackingApp, self.app)._track_task(self.action_show_ancestors())

    @on(Button.Pressed, "#cg-mode-descendants")
    def on_mode_descendants_pressed(self) -> None:
        """Handle the Descendants mode button."""
        cast(TaskTrackingApp, self.app)._track_task(self.action_show_descendants())

    @on(Tree.NodeSelected, "#genealogy-tree")
    def on_genealogy_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle click/Enter selection in the genealogy tree."""
        data = event.node.data
        node = data if isinstance(data, GenealogyNode) else None
        self._activate_genealogy_node(node)
