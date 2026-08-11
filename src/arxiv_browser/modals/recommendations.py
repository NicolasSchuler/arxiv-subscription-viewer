"""Recommendation modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ListItem, ListView, Static

from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text, truncate_text
from arxiv_browser.themes import theme_colors_for

RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal
ROW_TITLE_MAX_LEN = 80  # Max title length per result row (single-line, ellipsised)


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
    #recommendations-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        /* intentional override: orange border distinguishes recommendations */
        border: tall $th-orange;
    }

    #recommendations-title {
        color: $th-orange;
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
        background: $th-highlight-focus;
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
        margin-top: 1;
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
        with Vertical(id="recommendations-dialog", classes="modal-dialog"):
            truncated_title = truncate_text(self._target_paper.title, RECOMMENDATION_TITLE_MAX_LEN)
            yield Label(
                f"Similar to: {truncated_title}",
                id="recommendations-title",
                classes="modal-title",
            )
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
            with Horizontal(id="recommendations-buttons", classes="modal-buttons"):
                yield Button("Close (Esc/q)", variant="default", id="close-btn")
                yield Button("Go to Paper (Enter)", variant="primary", id="select-btn")

    def on_mount(self) -> None:
        """Populate the list view with similar papers and focus it."""
        list_view = self.query_one("#recommendations-list", ListView)
        green = theme_colors_for(self)["green"]
        for paper, score in self._similar_papers:
            safe_title = escape_rich_text(truncate_text(paper.title, ROW_TITLE_MAX_LEN))
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
