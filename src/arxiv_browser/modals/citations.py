"""Recommendation and citation graph modals."""

from __future__ import annotations

import logging
import sqlite3
import webbrowser
from collections.abc import Callable

import httpx
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView, Static

from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text, truncate_text
from arxiv_browser.semantic_scholar import CitationEntry
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)

RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal


class RecommendationSourceModal(ModalScreen[str]):
    """Simple choice dialog: local or Semantic Scholar recommendations."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("l", "local", "Local", show=False),
        Binding("s", "s2", "S2", show=False),
    ]

    CSS = """
    RecommendationSourceModal {
        align: center middle;
    }

    #rec-source-dialog {
        width: 50;
        height: auto;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #rec-source-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
    }

    #rec-source-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #rec-source-buttons Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="rec-source-dialog"):
            yield Label("Recommendation Source", id="rec-source-title")
            with Horizontal(id="rec-source-buttons"):
                yield Button("Local (TF-IDF)", variant="default", id="local-btn")
                yield Button("Semantic Scholar", variant="primary", id="s2-btn")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_local(self) -> None:
        self.dismiss("local")

    def action_s2(self) -> None:
        self.dismiss("s2")

    @on(Button.Pressed, "#local-btn")
    def on_local_pressed(self) -> None:
        self.action_local()

    @on(Button.Pressed, "#s2-btn")
    def on_s2_pressed(self) -> None:
        self.action_s2()


class RecommendationListItem(ListItem):
    """A list item for the recommendations screen that stores a paper reference."""

    def __init__(self, paper: Paper, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.paper = paper


class RecommendationsScreen(ModalScreen[str | None]):
    """Modal screen displaying similar papers."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("enter", "select", "Select"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    CSS = """
    RecommendationsScreen {
        align: center middle;
    }

    #recommendations-dialog {
        width: 80%;
        height: 80%;
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

    def __init__(self, target_paper: Paper, similar_papers: list[tuple[Paper, float]]) -> None:
        super().__init__()
        self._target_paper = target_paper
        self._similar_papers = similar_papers

    def compose(self) -> ComposeResult:
        with Vertical(id="recommendations-dialog"):
            truncated_title = truncate_text(self._target_paper.title, RECOMMENDATION_TITLE_MAX_LEN)
            yield Label(f"Similar to: {truncated_title}", id="recommendations-title")
            yield ListView(id="recommendations-list")
            with Horizontal(id="recommendations-buttons"):
                yield Button("Close (Esc)", variant="default", id="close-btn")
                yield Button("Go to Paper (Enter)", variant="primary", id="select-btn")

    def on_mount(self) -> None:
        list_view = self.query_one("#recommendations-list", ListView)
        for paper, score in self._similar_papers:
            safe_title = escape_rich_text(paper.title)
            safe_categories = escape_rich_text(paper.categories)
            item = RecommendationListItem(
                paper,
                Static(f"[bold]{safe_title}[/]", classes="rec-title"),
                Static(
                    f"[dim]{paper.arxiv_id}[/] | {safe_categories} | [{THEME_COLORS['green']}]{score:.0%}[/] match",
                    classes="rec-meta",
                ),
            )
            list_view.mount(item)
        if list_view.children:
            list_view.index = 0
        list_view.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        list_view = self.query_one("#recommendations-list", ListView)
        if isinstance(list_view.highlighted_child, RecommendationListItem):
            self.dismiss(list_view.highlighted_child.paper.arxiv_id)
        else:
            self.dismiss(None)

    def action_cursor_down(self) -> None:
        self.query_one("#recommendations-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#recommendations-list", ListView).action_cursor_up()

    @on(Button.Pressed, "#close-btn")
    def on_close_pressed(self) -> None:
        self.action_cancel()

    @on(Button.Pressed, "#select-btn")
    def on_select_pressed(self) -> None:
        self.action_select()

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, RecommendationListItem):
            self.dismiss(event.item.paper.arxiv_id)


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
        super().__init__(*children, **kwargs)
        self.entry = entry
        self.is_local = is_local


class CitationGraphScreen(ModalScreen[str | None]):
    """Modal screen for exploring citation graphs with drill-down navigation.

    Returns the arxiv_id of a local paper to jump to, or None.
    """

    BINDINGS = [
        Binding("escape", "back_or_close", "Back / Close"),
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
        width: 85%;
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
                yield Button("Close (Esc)", variant="default", id="cg-close-btn")
                yield Button("Drill Down (Enter)", variant="primary", id="cg-drill-btn")

    def on_mount(self) -> None:
        self._populate_lists()
        self._update_breadcrumb()
        refs_list = self.query_one("#refs-list", ListView)
        refs_list.focus()

    def _build_citation_item(self, entry: CitationEntry) -> CitationGraphListItem:
        """Build a list item widget for a single citation graph entry."""
        is_local = entry.arxiv_id != "" and entry.arxiv_id in self._local_arxiv_ids
        safe_title = escape_rich_text(entry.title)
        local_badge = f" [{THEME_COLORS['green']}]\\[LOCAL][/]" if is_local else ""
        year_str = str(entry.year) if entry.year else "?"
        authors_short = truncate_text(entry.authors, 50) if entry.authors else ""
        return CitationGraphListItem(
            entry,
            Static(
                f"[bold]{safe_title}[/]{local_badge}",
                classes="cite-title",
            ),
            Static(
                f"[dim]{year_str}[/] · {escape_rich_text(authors_short)}"
                f" · [{THEME_COLORS['accent']}]{entry.citation_count} cites[/]",
                classes="cite-meta",
            ),
            is_local=is_local,
        )

    def _populate_lists(self) -> None:
        """Fill both list views with current references and citations."""
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
        breadcrumb = " → ".join(
            f"[{THEME_COLORS['purple']}]{escape_rich_text(p)}[/]" for p in parts
        )
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
        self.query_one("#citation-graph-status", Static).update(
            f"[dim]Tab: switch panel · Enter: drill down · "
            f"o: open · g: go to local · Esc: back[/]"
            f"  Active: [{THEME_COLORS['purple']}]{panel_name}[/]{depth_str}"
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
                "Failed to load citations. Check your connection.",
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
        self._get_active_list().action_cursor_down()

    def action_cursor_up(self) -> None:
        self._get_active_list().action_cursor_up()

    @on(Button.Pressed, "#cg-close-btn")
    def on_close_pressed(self) -> None:
        self.action_back_or_close()

    @on(Button.Pressed, "#cg-drill-btn")
    def on_drill_pressed(self) -> None:
        # Button click needs to invoke the async action; use app's tracked task
        self.app._track_task(self.action_drill_down())  # type: ignore[attr-defined]
