"""Widget chrome for date navigation, bookmarks, filters, and footer hints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Label, Static

from arxiv_browser.models import QueryToken, SearchBookmark
from arxiv_browser.parsing import count_papers_in_file
from arxiv_browser.query import escape_rich_text, pill_label_for_token
from arxiv_browser.themes import theme_colors_for
from arxiv_browser.widgets import footer_status as _footer_status

DEFAULT_THEME = _footer_status.DEFAULT_THEME
FooterModeBadgeState = _footer_status.FooterModeBadgeState
StatusBarState = _footer_status.StatusBarState
_build_compact_status_parts = _footer_status._build_compact_status_parts
_build_full_status_parts = _footer_status._build_full_status_parts
_compact_primary_segment = _footer_status._compact_primary_segment
_compact_flag_segment = _footer_status._compact_flag_segment
_coerce_status_bar_state = _footer_status._coerce_status_bar_state
_full_primary_segment = _footer_status._full_primary_segment
_render_compact_status = _footer_status._render_compact_status
_truncate_rich_text = _footer_status._truncate_rich_text
build_api_footer_bindings = _footer_status.build_api_footer_bindings
build_browse_footer_bindings = _footer_status.build_browse_footer_bindings
build_detail_focus_footer_bindings = _footer_status.build_detail_focus_footer_bindings
build_footer_mode_badge = _footer_status.build_footer_mode_badge
build_search_footer_bindings = _footer_status.build_search_footer_bindings
build_selection_footer_base_bindings = _footer_status.build_selection_footer_base_bindings
build_selection_footer_bindings = _footer_status.build_selection_footer_bindings
build_status_bar_text = _footer_status.build_status_bar_text
get_filter_pill_remove_glyph = _footer_status.get_filter_pill_remove_glyph
set_ascii_glyphs = _footer_status.set_ascii_glyphs

DATE_NAV_WINDOW_SIZE = 5
DATE_NAV_ARROW_WIDTH = 3
DATE_NAV_ITEM_PADDING = 2
DATE_NAV_CONTAINER_PADDING = 2
DATE_NAV_LABEL_WIDTH = 9
DATE_NAV_LABEL_WITH_COUNTS = "with_counts"
DATE_NAV_LABEL_MONTH_DAY = "month_day"
DATE_NAV_LABEL_NUMERIC = "numeric"
DATE_NAV_LABEL_MODES: tuple[str, ...] = (
    DATE_NAV_LABEL_WITH_COUNTS,
    DATE_NAV_LABEL_MONTH_DAY,
    DATE_NAV_LABEL_NUMERIC,
)
MAX_FOOTER_HINTS = 9

_FOOTER_ACTION_BY_KEY: dict[str, str] = {
    "/": "toggle_search",
    "A": "arxiv_search",
    "Esc": "cancel_search",
    "o": "open_url",
    "P": "open_pdf",
    "F": "preview_pdf",
    "I": "preview_figure",
    "c": "copy_selected",
    "s": "cycle_sort",
    "Tab": "toggle_focus_pane",
    "Space": "toggle_select",
    "u": "clear_selection",
    "r": "toggle_read",
    "x": "toggle_star",
    "n": "edit_notes",
    "t": "edit_tags",
    "w": "toggle_watch_filter",
    "W": "manage_watch_list",
    "Ctrl+b": "add_bookmark",
    "E": "export_menu",
    "d": "download_pdf",
    "v": "toggle_detail_mode",
    "Ctrl+d": "toggle_sections",
    "e": "fetch_s2",
    "L": "score_relevance",
    "V": "check_versions",
    "?": "show_help",
    "Ctrl+p": "command_palette",
}


@dataclass(frozen=True, slots=True)
class FilterPillSpec:
    """Renderable state for one active filter pill."""

    item_id: str
    text: str
    class_name: str
    tooltip: str


class ContextFooter(Static):
    """Context-sensitive footer showing relevant keybindings.

    Hints render as a single ``Static`` renderable. Clickable hints use
    Textual ``@click`` action-link markup so a click invokes the bound app
    action directly, without mounting per-hint child widgets (re-mounting on
    every state-driven footer refresh churns the message queue and can stall
    the UI).
    """

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
        colors = theme_colors_for(self)
        parts = _build_footer_parts(bindings[:MAX_FOOTER_HINTS], mode_badge, colors)
        self.update("  ".join(parts))


def _chrome_label(content: str, classes: str, item_id: str, tooltip: str) -> Label:
    label = Label(content, classes=classes, id=item_id)
    label.tooltip = tooltip
    return label


def _build_footer_parts(
    bindings: list[tuple[str, str]],
    mode_badge: str,
    colors: Mapping[str, str],
) -> list[str]:
    """Build footer markup parts: an optional mode badge then hint strings."""
    accent = colors["accent"]
    muted = colors["muted"]
    parts: list[str] = []
    if mode_badge:
        parts.append(mode_badge)
    parts.extend(_format_footer_hint(key, label, accent, muted) for key, label in bindings)
    return parts


def _format_footer_hint(key: str, label: str, accent: str, muted: str) -> str:
    """Format one footer hint, wrapping clickable hints in @click action links."""
    safe_key = escape_rich_text(key)
    if key and label:
        hint = f"[bold {accent}]{safe_key}[/] [{muted}]{label}[/]"
    elif label:
        hint = f"[italic {muted}]{label}[/]"
    else:
        hint = f"[italic {muted}]{safe_key}[/]"
    action = _footer_action(key)
    if action is not None:
        return f"[@click=app.{action}]{hint}[/]"
    return hint


def _footer_action(key: str) -> str | None:
    return _FOOTER_ACTION_BY_KEY.get(key)


def _compute_window_bounds(total: int, current_index: int, window_size: int) -> tuple[int, int]:
    """Compute a centered sliding window clamped to history bounds."""
    if total <= 0:
        return (0, 0)
    window_size = max(1, min(window_size, total))
    half = window_size // 2
    start = max(0, current_index - half)
    end = min(total, start + window_size)
    if end - start < window_size:
        start = max(0, end - window_size)
    return (start, end)


def _format_date_nav_label(
    current_date: date,
    *,
    count: int | None,
    mode: str,
) -> str:
    """Format one date label using the requested compaction mode."""
    if mode == DATE_NAV_LABEL_WITH_COUNTS:
        safe_count = count or 0
        return f"{current_date.strftime('%b %d')}({safe_count})"
    if mode == DATE_NAV_LABEL_MONTH_DAY:
        return current_date.strftime("%b %d")
    return current_date.strftime("%m-%d")


def _estimate_date_nav_width(labels: list[str]) -> int:
    """Estimate the rendered width for arrows plus the given labels."""
    return (
        DATE_NAV_CONTAINER_PADDING
        + (DATE_NAV_ARROW_WIDTH * 2)
        + sum(len(label) + DATE_NAV_ITEM_PADDING for label in labels)
    )


def _compute_responsive_date_plan(
    history_files: list[tuple[date, Path]],
    current_index: int,
    width: int,
    get_count: Callable[[int], int],
) -> tuple[int, int, str]:
    """Choose a centered date window and label mode that fits the available width."""
    total = len(history_files)
    if total <= 0:
        return (0, 0, DATE_NAV_LABEL_WITH_COUNTS)

    if width <= 0:
        start, end = _compute_window_bounds(total, current_index, min(DATE_NAV_WINDOW_SIZE, total))
        return (start, end, DATE_NAV_LABEL_WITH_COUNTS)

    max_window = min(DATE_NAV_WINDOW_SIZE, total)
    for window_size in range(max_window, 0, -1):
        start, end = _compute_window_bounds(total, current_index, window_size)
        counts = {i: get_count(i) for i in range(start, end)}
        for mode in DATE_NAV_LABEL_MODES:
            labels = [
                _format_date_nav_label(
                    history_files[i][0],
                    count=counts.get(i),
                    mode=mode,
                )
                for i in range(start, end)
            ]
            if _estimate_date_nav_width(labels) <= width:
                return (start, end, mode)

    start, end = _compute_window_bounds(total, current_index, 1)
    return (start, end, DATE_NAV_LABEL_NUMERIC)


class DateNavigator(Horizontal):
    """Horizontal date strip showing available dates with sliding window."""

    class NavigateDate(Message):
        """Request to navigate by direction (+1 = older, -1 = newer)."""

        def __init__(self, direction: int) -> None:
            """Initialize with a navigation direction (+1 older, -1 newer)."""
            super().__init__()
            self.direction = direction

    class JumpToDate(Message):
        """Request to jump to a specific date index."""

        def __init__(self, index: int) -> None:
            """Initialize with the target date index to jump to."""
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

    DateNavigator .chrome-label {
        padding-right: 1;
        color: $th-muted;
        text-style: bold;
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
        """Initialize the date navigator with history files and selected index."""
        super().__init__()
        self._history_files = history_files
        self._current_index = current_index
        self._paper_counts: dict[Path, int] = {}

    def compose(self) -> ComposeResult:
        """Compose the static label and navigation arrow widgets."""
        yield Label("History", classes="chrome-label", id="date-nav-label")
        yield _chrome_label("<", "date-nav-arrow", "date-nav-prev", "Older (])")
        yield _chrome_label(">", "date-nav-arrow", "date-nav-next", "Newer ([)")

    def _get_paper_count(self, index: int) -> int:
        """Return the cached paper count for the history file at index."""
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
        return _compute_window_bounds(total, current_index, DATE_NAV_WINDOW_SIZE)

    def _build_desired_items(
        self,
        history_files: list[tuple[date, Path]],
        current_index: int,
        start: int,
        end: int,
        label_mode: str,
    ) -> list[tuple[str, str, bool, str]]:
        """Build desired date labels for the visible window."""
        desired: list[tuple[str, str, bool, str]] = []
        for i in range(start, end):
            d, _ = history_files[i]
            count = self._get_paper_count(i) if label_mode == DATE_NAV_LABEL_WITH_COUNTS else None
            label_text = _format_date_nav_label(d, count=count, mode=label_mode)
            tooltip = f"Jump to {d.isoformat()}"
            desired.append((f"date-nav-{i}", label_text, i == current_index, tooltip))
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
        desired: list[tuple[str, str, bool, str]],
    ) -> bool:
        """Return True when existing and desired IDs match in order."""
        existing_order = [child.id for child in existing_items if child.id is not None]
        desired_order = [item_id for item_id, _, _, _ in desired]
        return existing_order == desired_order

    @staticmethod
    def _render_label_text(label_text: str, is_current: bool) -> str:
        """Render one date label, highlighting the currently selected date."""
        return f"[{label_text}]" if is_current else label_text

    def _patch_items_in_place(
        self,
        existing_items: list[Label],
        desired: list[tuple[str, str, bool, str]],
    ) -> None:
        """Patch existing date labels without unmount/remount churn."""
        existing_by_id = {child.id: child for child in existing_items}
        for item_id, label_text, is_current, tooltip in desired:
            child = existing_by_id.get(item_id)
            if child is None:
                continue
            child.update(self._render_label_text(label_text, is_current))
            child.tooltip = tooltip
            if is_current:
                child.add_class("current")
            else:
                child.remove_class("current")

    async def _rebuild_items(
        self,
        existing_items: list[Label],
        desired: list[tuple[str, str, bool, str]],
    ) -> None:
        """Rebuild date labels when the visible window changed."""
        for child in existing_items:
            await child.remove()
        next_arrow = self.query_one("#date-nav-next")
        for item_id, label_text, is_current, tooltip in desired:
            classes = "date-nav-item current" if is_current else "date-nav-item"
            label = _chrome_label(
                self._render_label_text(label_text, is_current),
                classes,
                item_id,
                tooltip,
            )
            self.mount(label, before=next_arrow)

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
        width = getattr(getattr(self, "size", None), "width", 0)
        start, end, label_mode = _compute_responsive_date_plan(
            history_files,
            current_index,
            max(0, width - DATE_NAV_LABEL_WIDTH),
            self._get_paper_count,
        )
        desired = self._build_desired_items(
            history_files,
            current_index,
            start,
            end,
            label_mode,
        )
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
        display: none;
    }

    BookmarkTabBar.visible {
        display: block;
    }

    BookmarkTabBar .chrome-label {
        padding-right: 1;
        color: $th-muted;
        text-style: bold;
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
        color: $th-text;
    }

    BookmarkTabBar .bookmark-hint {
        color: $th-muted;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        bookmarks: list[SearchBookmark],
        active_index: int = -1,
        active_search: bool = False,
    ) -> None:
        """Initialize the bookmark bar with bookmarks and active state."""
        super().__init__()
        self._bookmarks = bookmarks
        self._active_index = active_index
        self._active_search = active_search
        if bookmarks or active_search:
            self.add_class("visible")

    def compose(self) -> ComposeResult:
        """Compose the bookmark label, numbered tabs, and save hint."""
        yield Label("Saved searches", classes="chrome-label", id="bookmark-label")
        if self._bookmarks:
            for i, bookmark in enumerate(self._bookmarks[:9]):  # Max 9 bookmarks
                classes = "bookmark-tab active" if i == self._active_index else "bookmark-tab"
                yield _chrome_label(
                    f"{i + 1}: {bookmark.name}",
                    classes,
                    f"bookmark-{i}",
                    f"Saved search {i + 1} - press {i + 1} to load",
                )
            yield _chrome_label("Ctrl+b save", "bookmark-add", "bookmark-add", "Save search")
        elif self._active_search:
            yield _chrome_label(
                "Ctrl+b save current search",
                "bookmark-hint",
                "bookmark-hint",
                "Save current search",
            )

    async def update_bookmarks(
        self,
        bookmarks: list[SearchBookmark],
        active_index: int = -1,
        active_search: bool = False,
    ) -> None:
        """Update the displayed bookmarks."""
        self._bookmarks = bookmarks
        self._active_index = active_index
        self._active_search = active_search
        await self.remove_children()
        if bookmarks or active_search:
            self.add_class("visible")
        else:
            self.remove_class("visible")
            return
        self.mount(Label("Saved searches", classes="chrome-label", id="bookmark-label"))
        if bookmarks:
            for i, bookmark in enumerate(bookmarks[:9]):
                classes = "bookmark-tab active" if i == self._active_index else "bookmark-tab"
                self.mount(
                    _chrome_label(
                        f"{i + 1}: {bookmark.name}",
                        classes,
                        f"bookmark-{i}",
                        f"Saved search {i + 1} - press {i + 1} to load",
                    )
                )
            self.mount(_chrome_label("Ctrl+b save", "bookmark-add", "bookmark-add", "Save search"))
        else:
            self.mount(
                _chrome_label(
                    "Ctrl+b save current search",
                    "bookmark-hint",
                    "bookmark-hint",
                    "Save current search",
                )
            )


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

    FilterPillBar .chrome-label {
        padding-right: 1;
        color: $th-muted;
        text-style: bold;
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
            """Initialize with the query token index to remove."""
            super().__init__()
            self.token_index = token_index

    class RemoveWatchFilter(Message):
        """Message sent when the watch filter pill is clicked to remove it."""

    def compose(self) -> ComposeResult:
        """Compose the filter label prefix widget."""
        yield Label("Filters", classes="chrome-label", id="filter-pill-prefix")

    async def update_pills(self, tokens: list[QueryToken], watch_active: bool) -> None:
        """Update the displayed filter pills."""
        desired = self._desired_filter_pills(tokens, watch_active)
        existing_items = self._existing_filter_pills()

        if _pill_order(existing_items) == [pill.item_id for pill in desired]:
            self._update_existing_filter_pills(existing_items, desired)
        else:
            await self._rebuild_filter_pills(existing_items, desired)

        if desired:
            self.add_class("visible")
        else:
            self.remove_class("visible")

    def _desired_filter_pills(
        self,
        tokens: list[QueryToken],
        watch_active: bool,
    ) -> list[FilterPillSpec]:
        desired = [self._filter_pill_spec(index, token) for index, token in enumerate(tokens)]
        desired = [pill for pill in desired if pill is not None]
        if watch_active:
            desired.append(_watch_filter_pill_spec())
        return desired

    def _filter_pill_spec(self, index: int, token: QueryToken) -> FilterPillSpec | None:
        if token.kind == "op":
            return None
        label_text = escape_rich_text(pill_label_for_token(token))
        return FilterPillSpec(
            item_id=f"pill-{index}",
            text=f"{label_text} {get_filter_pill_remove_glyph()}",
            class_name="filter-pill",
            tooltip="Click to remove filter",
        )

    def _existing_filter_pills(self) -> list[Label]:
        return [
            child
            for child in self.children
            if isinstance(child, Label) and child.id is not None and child.id.startswith("pill-")
        ]

    def _update_existing_filter_pills(
        self,
        existing_items: list[Label],
        desired: list[FilterPillSpec],
    ) -> None:
        existing_by_id = {child.id: child for child in existing_items}
        for pill in desired:
            child = existing_by_id.get(pill.item_id)
            if child is not None:
                _update_filter_pill(child, pill)

    async def _rebuild_filter_pills(
        self,
        existing_items: list[Label],
        desired: list[FilterPillSpec],
    ) -> None:
        for child in existing_items:
            await child.remove()
        for pill in desired:
            self.mount(_chrome_label(pill.text, pill.class_name, pill.item_id, pill.tooltip))

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


def _pill_order(pills: list[Label]) -> list[str | None]:
    return [pill.id for pill in pills]


def _update_filter_pill(label: Label, pill: FilterPillSpec) -> None:
    label.update(pill.text)
    label.remove_class("filter-pill-watch")
    label.remove_class("filter-pill")
    label.add_class(pill.class_name)
    label.tooltip = pill.tooltip


def _watch_filter_pill_spec() -> FilterPillSpec:
    return FilterPillSpec(
        item_id="pill-watch",
        text=f"watched {get_filter_pill_remove_glyph()}",
        class_name="filter-pill-watch",
        tooltip="Click to remove watch filter",
    )


__all__ = [
    "DATE_NAV_WINDOW_SIZE",
    "BookmarkTabBar",
    "ContextFooter",
    "DateNavigator",
    "FilterPillBar",
    "FooterModeBadgeState",
    "StatusBarState",
    "build_api_footer_bindings",
    "build_browse_footer_bindings",
    "build_detail_focus_footer_bindings",
    "build_footer_mode_badge",
    "build_search_footer_bindings",
    "build_selection_footer_base_bindings",
    "build_selection_footer_bindings",
    "build_status_bar_text",
    "set_ascii_glyphs",
]
