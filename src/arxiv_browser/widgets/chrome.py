"""Widget chrome for date navigation, bookmarks, filters, and footer hints."""

from __future__ import annotations

import re
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


def build_selection_footer_bindings(
    selection_bindings: list[tuple[str, str]], selected_count: int
) -> list[tuple[str, str]]:
    """Build selection-mode footer bindings with dynamic open(n) label."""
    bindings = list(selection_bindings)
    if bindings:
        bindings[0] = ("o", f"open({selected_count})")
    return bindings


def build_browse_footer_bindings(
    *,
    s2_active: bool,
    has_starred: bool,
    llm_configured: bool,
    has_history_navigation: bool,
) -> list[tuple[str, str]]:
    """Build default browsing footer bindings with stable core actions."""
    bindings: list[tuple[str, str]] = [
        ("/", "search"),
        ("o", "open"),
        ("s", "sort"),
        ("r", "read"),
        ("x", "star"),
        ("n", "notes"),
        ("t", "tags"),
    ]
    if s2_active:
        bindings.extend([("e", "S2"), ("G", "graph")])
    if has_starred:
        bindings.append(("V", "versions"))
    if llm_configured:
        bindings.append(("L", "relevance"))
    if has_history_navigation:
        bindings.append(("[/]", "dates"))
    bindings.extend([("E", "export"), ("Ctrl+p", "commands"), ("?", "help")])
    return bindings


def build_footer_mode_badge(
    *,
    relevance_scoring_active: bool,
    version_checking: bool,
    search_visible: bool,
    in_arxiv_api_mode: bool,
    selected_count: int,
) -> str:
    """Build Rich-markup mode badge text for footer state."""
    pink = THEME_COLORS["pink"]
    accent = THEME_COLORS["accent"]
    orange = THEME_COLORS["orange"]
    green = THEME_COLORS["green"]
    panel_alt = THEME_COLORS["panel_alt"]
    if relevance_scoring_active:
        return f"[bold {pink} on {panel_alt}] SCORING [/]"
    if version_checking:
        return f"[bold {pink} on {panel_alt}] VERSIONS [/]"
    if search_visible:
        return f"[bold {accent} on {panel_alt}] SEARCH [/]"
    if in_arxiv_api_mode:
        return f"[bold {orange} on {panel_alt}] API [/]"
    if selected_count > 0:
        return f"[bold {green} on {panel_alt}] {selected_count} SEL [/]"
    return ""


def build_status_bar_text(
    *,
    total: int,
    filtered: int,
    query: str,
    watch_filter_active: bool,
    selected_count: int,
    sort_label: str,
    in_arxiv_api_mode: bool,
    api_page: int | None,
    arxiv_api_loading: bool,
    show_abstract_preview: bool,
    s2_active: bool,
    s2_loading: bool,
    s2_count: int,
    hf_active: bool,
    hf_loading: bool,
    hf_match_count: int,
    version_checking: bool,
    version_update_count: int,
    max_width: int | None = None,
) -> str:
    """Build semantic status bar text for current UI/application state."""
    if max_width is not None and max_width <= 100:
        compact_parts: list[str] = []
        if query:
            compact_parts.append(f"{filtered}/{total} match")
        elif watch_filter_active:
            compact_parts.append(f"{filtered}/{total} watched")
        else:
            compact_parts.append(f"{total} papers")

        if selected_count > 0:
            compact_parts.append(f"{selected_count} selected")

        compact_parts.append(f"sort:{sort_label}")

        if in_arxiv_api_mode and api_page is not None:
            compact_parts.append(f"api p{api_page}")
            if arxiv_api_loading:
                compact_parts.append("loading")

        if s2_active:
            if s2_loading:
                compact_parts.append("S2 loading")
            elif s2_count > 0:
                compact_parts.append(f"S2:{s2_count}")
            else:
                compact_parts.append("S2")

        if hf_active and max_width >= 90:
            if hf_loading:
                compact_parts.append("HF loading")
            elif hf_match_count > 0:
                compact_parts.append(f"HF:{hf_match_count}")
            else:
                compact_parts.append("HF")

        if show_abstract_preview and max_width >= 95:
            compact_parts.append("preview")

        if version_checking:
            compact_parts.append("checking versions")
        elif version_update_count > 0:
            compact_parts.append(f"{version_update_count} updated")

        compact = " | ".join(compact_parts)
        if len(compact) <= max_width:
            return compact

        while len(compact_parts) > 1 and len(" | ".join(compact_parts) + " ...") > max_width:
            compact_parts.pop()
        return " | ".join(compact_parts) + " ..."

    parts: list[str] = []

    if query:
        truncated_query = query if len(query) <= 30 else query[:27] + "..."
        safe_query = escape_rich_text(truncated_query)
        parts.append(
            f'[{THEME_COLORS["accent"]}]{filtered}[/][dim]/{total} matching [/][{THEME_COLORS["accent"]}]"{safe_query}"[/]'
        )
    elif watch_filter_active:
        parts.append(f"[{THEME_COLORS['orange']}]{filtered}[/][dim]/{total} watched[/]")
    else:
        parts.append(f"[dim]{total} papers[/]")

    if selected_count > 0:
        parts.append(f"[bold {THEME_COLORS['green']}]{selected_count} selected[/]")

    parts.append(f"[dim]Sort: {sort_label}[/]")

    if in_arxiv_api_mode and api_page is not None:
        parts.append(f"[{THEME_COLORS['orange']}]API[/]")
        parts.append(f"[dim]Page: {api_page}[/]")
        if arxiv_api_loading:
            parts.append(f"[{THEME_COLORS['orange']}]Loading...[/]")
    if show_abstract_preview:
        parts.append(f"[{THEME_COLORS['purple']}]Preview[/]")
    if s2_active:
        if s2_loading:
            parts.append(f"[{THEME_COLORS['green']}]S2 loading...[/]")
        elif s2_count > 0:
            parts.append(f"[{THEME_COLORS['green']}]S2:{s2_count}[/]")
        else:
            parts.append(f"[{THEME_COLORS['green']}]S2[/]")
    if hf_active:
        if hf_loading:
            parts.append(f"[{THEME_COLORS['orange']}]HF loading...[/]")
        elif hf_match_count > 0:
            parts.append(f"[{THEME_COLORS['orange']}]HF:{hf_match_count}[/]")
        else:
            parts.append(f"[{THEME_COLORS['orange']}]HF[/]")
    if version_checking:
        parts.append(f"[{THEME_COLORS['pink']}]Checking versions...[/]")
    elif version_update_count > 0:
        parts.append(f"[{THEME_COLORS['pink']}]{version_update_count} updated[/]")

    rendered = " [dim]│[/] ".join(parts)
    if max_width is not None and max_width > 0:
        visible = re.sub(r"\[[^\]]*]", "", rendered)
        if len(visible) > max_width:
            return visible[: max(0, max_width - 3)] + "..."
    return rendered


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

    def _prune_count_cache(self, active_paths: set[Path]) -> None:
        """Drop cached counts for history files no longer present."""
        self._paper_counts = {
            path: count for path, count in self._paper_counts.items() if path in active_paths
        }

    async def _clear_date_items(self) -> None:
        """Remove only date labels, leaving navigation arrows intact."""
        for child in self._get_existing_date_items():
            await child.remove()

    def _compute_window(self, total: int, current_index: int) -> tuple[int, int]:
        """Compute a centered sliding window clamped to history bounds."""
        half = DATE_NAV_WINDOW_SIZE // 2
        start = max(0, current_index - half)
        end = min(total, start + DATE_NAV_WINDOW_SIZE)
        if end - start < DATE_NAV_WINDOW_SIZE:
            start = max(0, end - DATE_NAV_WINDOW_SIZE)
        return (start, end)

    def _build_desired_items(
        self,
        history_files: list[tuple[date, Path]],
        current_index: int,
        start: int,
        end: int,
    ) -> list[tuple[str, str, bool]]:
        """Build desired date labels for the visible window."""
        desired: list[tuple[str, str, bool]] = []
        for i in range(start, end):
            d, _ = history_files[i]
            count = self._get_paper_count(i)
            label_text = f"{d.strftime('%b %d')}({count})"
            desired.append((f"date-nav-{i}", label_text, i == current_index))
        return desired

    def _get_existing_date_items(self) -> list[Label]:
        """Return currently mounted date label widgets."""
        return [
            child
            for child in self.children
            if isinstance(child, Label) and "date-nav-item" in child.classes
        ]

    def _can_patch_in_place(
        self,
        existing_items: list[Label],
        desired: list[tuple[str, str, bool]],
    ) -> bool:
        """Return True when existing and desired IDs match in order."""
        existing_order = [child.id for child in existing_items if child.id is not None]
        desired_order = [item_id for item_id, _, _ in desired]
        return existing_order == desired_order

    @staticmethod
    def _render_label_text(label_text: str, is_current: bool) -> str:
        """Render one date label, highlighting the currently selected date."""
        return f"[{label_text}]" if is_current else label_text

    def _patch_items_in_place(
        self,
        existing_items: list[Label],
        desired: list[tuple[str, str, bool]],
    ) -> None:
        """Patch existing date labels without unmount/remount churn."""
        existing_by_id = {child.id: child for child in existing_items}
        for item_id, label_text, is_current in desired:
            child = existing_by_id.get(item_id)
            if child is None:
                continue
            child.update(self._render_label_text(label_text, is_current))
            if is_current:
                child.add_class("current")
            else:
                child.remove_class("current")

    async def _rebuild_items(
        self,
        existing_items: list[Label],
        desired: list[tuple[str, str, bool]],
    ) -> None:
        """Rebuild date labels when the visible window changed."""
        for child in existing_items:
            await child.remove()
        next_arrow = self.query_one("#date-nav-next")
        for item_id, label_text, is_current in desired:
            classes = "date-nav-item current" if is_current else "date-nav-item"
            self.mount(
                Label(self._render_label_text(label_text, is_current), classes=classes, id=item_id),
                before=next_arrow,
            )

    async def update_dates(
        self,
        history_files: list[tuple[date, Path]],
        current_index: int,
    ) -> None:
        """Update the displayed dates with a sliding window."""
        self._history_files = history_files
        self._current_index = current_index
        self._prune_count_cache({path for _, path in history_files})

        if len(history_files) <= 1:
            self.remove_class("visible")
            await self._clear_date_items()
            return

        self.add_class("visible")
        start, end = self._compute_window(len(history_files), current_index)
        desired = self._build_desired_items(history_files, current_index, start, end)
        existing_items = self._get_existing_date_items()

        if self._can_patch_in_place(existing_items, desired):
            self._patch_items_in_place(existing_items, desired)
            return

        await self._rebuild_items(existing_items, desired)

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
        desired: list[tuple[str, str, str]] = []
        for i, token in enumerate(tokens):
            if token.kind == "op":
                continue
            label_text = escape_rich_text(pill_label_for_token(token))
            desired.append((f"pill-{i}", f"{label_text} \u00d7", "filter-pill"))
        if watch_active:
            desired.append(("pill-watch", "watched \u00d7", "filter-pill-watch"))

        existing_items = [
            child
            for child in self.children
            if isinstance(child, Label) and child.id is not None and child.id.startswith("pill-")
        ]
        existing_order = [child.id for child in existing_items]
        desired_order = [item_id for item_id, _, _ in desired]

        if existing_order == desired_order:
            existing_by_id = {child.id: child for child in existing_items}
            for item_id, text, class_name in desired:
                child = existing_by_id.get(item_id)
                if child is None:
                    continue
                child.update(text)
                if class_name == "filter-pill-watch":
                    child.remove_class("filter-pill")
                    child.add_class("filter-pill-watch")
                else:
                    child.remove_class("filter-pill-watch")
                    child.add_class("filter-pill")
        else:
            await self.remove_children()
            for item_id, text, class_name in desired:
                self.mount(Label(text, classes=class_name, id=item_id))

        has_pills = bool(desired)
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
    "build_browse_footer_bindings",
    "build_footer_mode_badge",
    "build_selection_footer_bindings",
    "build_status_bar_text",
]
