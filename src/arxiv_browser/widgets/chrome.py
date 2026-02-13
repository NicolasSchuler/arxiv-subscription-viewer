"""Widget chrome for date navigation, bookmarks, filters, and footer hints."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Label, Static

from arxiv_browser.models import QueryToken, SearchBookmark
from arxiv_browser.parsing import count_papers_in_file
from arxiv_browser.query import escape_rich_text, pill_label_for_token
from arxiv_browser.themes import THEME_COLORS

DATE_NAV_WINDOW_SIZE = 5


class ContextFooter(Static):
    """Context-sensitive footer showing relevant keybindings."""

    DEFAULT_CSS = """
    ContextFooter {
        dock: bottom;
        height: 1;
        background: $th-background;
        color: $th-muted;
        padding: 0 1;
        border-top: solid $th-panel-alt;
    }
    """

    def render_bindings(self, bindings: list[tuple[str, str]], mode_badge: str = "") -> None:
        """Update the footer with a list of (key, label) binding hints."""
        accent = THEME_COLORS["accent"]
        muted = THEME_COLORS["muted"]
        parts = []
        if mode_badge:
            parts.append(mode_badge)
        for key, label in bindings:
            safe_key = escape_rich_text(key)
            if key and label:
                parts.append(f"[bold {accent}]{safe_key}[/] [{muted}]{label}[/]")
            elif label:
                # Label-only entry (e.g., progress indicator)
                parts.append(f"[italic {muted}]{label}[/]")
            else:
                # Key-only entry (e.g., "type to filter" hint)
                parts.append(f"[italic {muted}]{safe_key}[/]")
        self.update("  ".join(parts))


class DateNavigator(Horizontal):
    """Horizontal date strip showing available dates with sliding window."""

    class NavigateDate(Message):
        """Request to navigate by direction (+1 = older, -1 = newer)."""

        def __init__(self, direction: int) -> None:
            super().__init__()
            self.direction = direction

    class JumpToDate(Message):
        """Request to jump to a specific date index."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    DEFAULT_CSS = """
    DateNavigator {
        height: auto;
        padding: 0 1;
        background: $th-panel;
        display: none;
    }

    DateNavigator.visible {
        display: block;
    }

    DateNavigator .date-nav-arrow {
        padding: 0 1;
        color: $th-muted;
    }

    DateNavigator .date-nav-arrow:hover {
        color: $th-text;
    }

    DateNavigator .date-nav-item {
        padding: 0 1;
        color: $th-muted;
    }

    DateNavigator .date-nav-item:hover {
        color: $th-text;
    }

    DateNavigator .date-nav-item.current {
        color: $th-accent;
        text-style: bold;
    }
    """

    def __init__(
        self,
        history_files: list[tuple[date, Path]],
        current_index: int = 0,
    ) -> None:
        super().__init__()
        self._history_files = history_files
        self._current_index = current_index
        self._paper_counts: dict[Path, int] = {}

    def compose(self) -> ComposeResult:
        yield Label("<", classes="date-nav-arrow", id="date-nav-prev")
        yield Label(">", classes="date-nav-arrow", id="date-nav-next")

    def _get_paper_count(self, index: int) -> int:
        _, path = self._history_files[index]
        if path not in self._paper_counts:
            self._paper_counts[path] = count_papers_in_file(path)
        return self._paper_counts[path]

    async def update_dates(
        self,
        history_files: list[tuple[date, Path]],
        current_index: int,
    ) -> None:
        """Update the displayed dates with a sliding window."""
        self._history_files = history_files
        self._current_index = current_index
        active_paths = {path for _, path in history_files}
        self._paper_counts = {
            path: count for path, count in self._paper_counts.items() if path in active_paths
        }

        if len(history_files) <= 1:
            self.remove_class("visible")
            return

        self.add_class("visible")

        # Remove old date labels (keep arrows)
        for child in list(self.children):
            if "date-nav-item" in child.classes:
                await child.remove()

        # Compute sliding window centered on current
        total = len(history_files)
        half = DATE_NAV_WINDOW_SIZE // 2
        start = max(0, current_index - half)
        end = min(total, start + DATE_NAV_WINDOW_SIZE)
        if end - start < DATE_NAV_WINDOW_SIZE:
            start = max(0, end - DATE_NAV_WINDOW_SIZE)

        # Mount date labels between the two arrows
        next_arrow = self.query_one("#date-nav-next")
        for i in range(start, end):
            d, _ = history_files[i]
            count = self._get_paper_count(i)
            label_text = f"{d.strftime('%b %d')}({count})"
            if i == current_index:
                label_text = f"[{label_text}]"
            classes = "date-nav-item current" if i == current_index else "date-nav-item"
            self.mount(Label(label_text, classes=classes, id=f"date-nav-{i}"), before=next_arrow)

    def on_click(self, event: object) -> None:
        """Handle clicks on arrows and date labels."""
        from textual.events import Click

        if not isinstance(event, Click):
            return
        widget = event.widget
        if widget is None:
            return
        widget_id = widget.id or ""
        if widget_id == "date-nav-prev":
            self.post_message(self.NavigateDate(1))
        elif widget_id == "date-nav-next":
            self.post_message(self.NavigateDate(-1))
        elif widget_id.startswith("date-nav-"):
            try:
                index = int(widget_id.removeprefix("date-nav-"))
                self.post_message(self.JumpToDate(index))
            except ValueError:
                pass


class BookmarkTabBar(Horizontal):
    """Horizontal bar displaying search bookmarks as numbered tabs."""

    DEFAULT_CSS = """
    BookmarkTabBar {
        height: auto;
        padding: 0 1;
        background: $th-panel;
        border-bottom: solid $th-panel-alt;
    }

    BookmarkTabBar .bookmark-tab {
        padding: 0 2;
        margin-right: 1;
        color: $th-muted;
    }

    BookmarkTabBar .bookmark-tab:hover {
        color: $th-text;
    }

    BookmarkTabBar .bookmark-tab.active {
        color: $th-accent-alt;
        text-style: bold;
    }

    BookmarkTabBar .bookmark-add {
        color: $th-muted;
        padding: 0 1;
    }

    BookmarkTabBar .bookmark-add:hover {
        color: $th-green;
    }
    """

    def __init__(self, bookmarks: list[SearchBookmark], active_index: int = -1) -> None:
        super().__init__()
        self._bookmarks = bookmarks
        self._active_index = active_index

    def compose(self) -> ComposeResult:
        for i, bookmark in enumerate(self._bookmarks[:9]):  # Max 9 bookmarks
            classes = "bookmark-tab active" if i == self._active_index else "bookmark-tab"
            yield Label(f"{i + 1}: {bookmark.name}", classes=classes, id=f"bookmark-{i}")
        yield Label("[+]", classes="bookmark-add", id="bookmark-add")

    async def update_bookmarks(
        self, bookmarks: list[SearchBookmark], active_index: int = -1
    ) -> None:
        """Update the displayed bookmarks."""
        self._bookmarks = bookmarks
        self._active_index = active_index
        await self.remove_children()
        for i, bookmark in enumerate(bookmarks[:9]):
            classes = "bookmark-tab active" if i == self._active_index else "bookmark-tab"
            self.mount(Label(f"{i + 1}: {bookmark.name}", classes=classes, id=f"bookmark-{i}"))
        self.mount(Label("[+]", classes="bookmark-add", id="bookmark-add"))


class FilterPillBar(Horizontal):
    """Horizontal bar displaying active search filters as removable pills."""

    DEFAULT_CSS = """
    FilterPillBar {
        height: auto;
        padding: 0 1;
        background: $th-panel;
        display: none;
    }

    FilterPillBar.visible {
        display: block;
    }

    FilterPillBar .filter-pill {
        padding: 0 1;
        margin-right: 1;
        color: $th-accent;
    }

    FilterPillBar .filter-pill:hover {
        color: $th-text;
        text-style: bold;
    }

    FilterPillBar .filter-pill-watch {
        padding: 0 1;
        margin-right: 1;
        color: $th-orange;
    }

    FilterPillBar .filter-pill-watch:hover {
        color: $th-text;
        text-style: bold;
    }
    """

    class RemoveFilter(Message):
        """Message sent when a filter pill is clicked to remove it."""

        def __init__(self, token_index: int) -> None:
            super().__init__()
            self.token_index = token_index

    class RemoveWatchFilter(Message):
        """Message sent when the watch filter pill is clicked to remove it."""

    async def update_pills(self, tokens: list[QueryToken], watch_active: bool) -> None:
        """Update the displayed filter pills."""
        await self.remove_children()
        has_pills = False
        for i, token in enumerate(tokens):
            if token.kind == "op":
                continue
            label_text = escape_rich_text(pill_label_for_token(token))
            self.mount(Label(f"{label_text} \u00d7", classes="filter-pill", id=f"pill-{i}"))
            has_pills = True
        if watch_active:
            self.mount(Label("watched \u00d7", classes="filter-pill-watch", id="pill-watch"))
            has_pills = True
        if has_pills:
            self.add_class("visible")
        else:
            self.remove_class("visible")

    def on_click(self, event: object) -> None:
        """Handle click on a filter pill to remove it."""
        from textual.events import Click

        if not isinstance(event, Click):
            return
        widget = event.widget
        if not isinstance(widget, Label):
            return
        widget_id = widget.id or ""
        if widget_id == "pill-watch":
            self.post_message(self.RemoveWatchFilter())
        elif widget_id.startswith("pill-"):
            try:
                index = int(widget_id.split("-", 1)[1])
                self.post_message(self.RemoveFilter(index))
            except (ValueError, IndexError):
                pass


__all__ = [
    "DATE_NAV_WINDOW_SIZE",
    "BookmarkTabBar",
    "ContextFooter",
    "DateNavigator",
    "FilterPillBar",
]
