"""Bookmark tabs and active-search filter controls."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Label

from arxiv_browser._ascii import is_ascii_mode
from arxiv_browser.models import QueryToken, SearchBookmark
from arxiv_browser.query import escape_rich_text, pill_label_for_token
from arxiv_browser.widgets.footer_status import get_filter_pill_remove_glyph

BOOKMARK_MAX_TABS = 9
_BOOKMARK_PREFIX_LABEL = "Saved searches"
_BOOKMARK_SAVE_LABEL = "Ctrl+b save"
_BOOKMARK_SAVE_CURRENT_LABEL = "Ctrl+b save current search"
# Cell budget from each part's CSS padding/margin (see BookmarkTabBar DEFAULT_CSS).
_BOOKMARK_CHIP_CHROME = 5
_BOOKMARK_PREFIX_CHROME = 1
_BOOKMARK_SIDE_CHROME = 2


@dataclass(frozen=True, slots=True)
class FilterPillSpec:
    """Renderable state for one active filter pill."""

    item_id: str
    text: str
    class_name: str
    tooltip: str


def _search_chrome_label(content: str, classes: str, item_id: str, tooltip: str) -> Label:
    label = Label(content, classes=classes, id=item_id)
    label.tooltip = tooltip
    return label


def _bookmark_chip_text(index: int, name: str) -> str:
    """Return the visible label for a numbered bookmark chip."""
    return f"{index + 1}: {name}"


def _elide_bookmark_name(index: int, name: str, budget: int, ellipsis: str) -> str:
    """Elide a single bookmark name so its chip fits ``budget`` cells."""
    prefix = f"{index + 1}: "
    avail = budget - _BOOKMARK_CHIP_CHROME - len(prefix) - len(ellipsis)
    if avail < 1:
        avail = 1
    return name[:avail] + ellipsis


def _fit_bookmark_chips(
    names: list[str],
    width: int,
) -> tuple[list[tuple[int, str]], int]:
    """Fit numbered bookmark chips into ``width`` content cells.

    Returns ``(visible, hidden)`` — ``visible`` is ``(index, display_name)`` per
    rendered chip (name may be ellipsis-elided), ``hidden`` is the count folded
    into a trailing ``+N`` chip. Whole chips are dropped from the right before any
    chip is elided (no mid-word fragment); ``width`` <= 0 keeps every chip.
    """
    indexed = list(enumerate(names))
    if width <= 0 or not indexed:
        return indexed, 0

    ellipsis = "..." if is_ascii_mode() else "…"
    reserved = (
        len(_BOOKMARK_PREFIX_LABEL)
        + _BOOKMARK_PREFIX_CHROME
        + len(_BOOKMARK_SAVE_LABEL)
        + _BOOKMARK_SIDE_CHROME
    )
    budget = width - reserved

    def chip_width(index: int, name: str) -> int:
        return len(_bookmark_chip_text(index, name)) + _BOOKMARK_CHIP_CHROME

    visible: list[tuple[int, str]] = []
    used = 0
    for index, name in indexed:
        width_needed = chip_width(index, name)
        if not visible and width_needed > budget:
            # The first chip alone overflows: elide it rather than clip a word.
            elided = _elide_bookmark_name(index, name, budget, ellipsis)
            visible.append((index, elided))
            used += chip_width(index, elided)
            break
        if used + width_needed > budget:
            break
        visible.append((index, name))
        used += width_needed

    hidden = len(indexed) - len(visible)
    if hidden == 0:
        return visible, 0

    # Reserve room for the "+N" marker, dropping whole chips (never the last) to fit.
    def marker_width(count: int) -> int:
        return len(f"+{count}") + _BOOKMARK_CHIP_CHROME

    while len(visible) > 1 and used + marker_width(hidden) > budget:
        drop_index, drop_name = visible.pop()
        used -= chip_width(drop_index, drop_name)
        hidden = len(indexed) - len(visible)
    return visible, hidden


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
        yield from self._build_children(self._content_width())

    def _content_width(self) -> int:
        """Return the current content width, or 0 before the first layout pass."""
        return getattr(getattr(self, "content_size", None), "width", 0)

    def _build_children(self, width: int) -> list[Label]:
        """Build the prefix, width-fitted numbered chips, and trailing hint."""
        children: list[Label] = [
            Label(_BOOKMARK_PREFIX_LABEL, classes="chrome-label", id="bookmark-label")
        ]
        if self._bookmarks:
            names = [bookmark.name for bookmark in self._bookmarks[:BOOKMARK_MAX_TABS]]
            visible, hidden = _fit_bookmark_chips(names, width)
            for index, display_name in visible:
                classes = "bookmark-tab active" if index == self._active_index else "bookmark-tab"
                children.append(
                    _search_chrome_label(
                        _bookmark_chip_text(index, display_name),
                        classes,
                        f"bookmark-{index}",
                        f"Saved search {index + 1} - press {index + 1} to load",
                    )
                )
            if hidden:
                plural = "es" if hidden != 1 else ""
                children.append(
                    _search_chrome_label(
                        f"+{hidden}",
                        "bookmark-add",
                        "bookmark-overflow",
                        f"{hidden} more saved search{plural} - press 1-9 to load",
                    )
                )
            children.append(
                _search_chrome_label(
                    _BOOKMARK_SAVE_LABEL,
                    "bookmark-add",
                    "bookmark-add",
                    "Save search",
                )
            )
        elif self._active_search:
            children.append(
                _search_chrome_label(
                    _BOOKMARK_SAVE_CURRENT_LABEL,
                    "bookmark-hint",
                    "bookmark-hint",
                    "Save current search",
                )
            )
        return children

    def _fit_signature(self, width: int) -> tuple[tuple[tuple[int, str], ...], int] | None:
        """Return a signature of the width-derived chip layout, or None."""
        if not self._bookmarks:
            return None
        names = [bookmark.name for bookmark in self._bookmarks[:BOOKMARK_MAX_TABS]]
        visible, hidden = _fit_bookmark_chips(names, width)
        return (tuple(visible), hidden)

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
        # Invalidate any pending resize-driven rebuild so it cannot race this one.
        token = self._next_bookmark_rebuild_token()
        await self.remove_children()
        if token != self._bookmark_rebuild_token:
            return
        if not (bookmarks or active_search):
            self.remove_class("visible")
            return
        self.add_class("visible")
        width = self._content_width()
        self._bookmark_fit_signature = self._fit_signature(width)
        for child in self._build_children(width):
            self.mount(child)

    def _next_bookmark_rebuild_token(self) -> int:
        """Bump and return the rebuild token; the newest request always wins."""
        token = getattr(self, "_bookmark_rebuild_token", 0) + 1
        self._bookmark_rebuild_token = token
        return token

    def on_resize(self, event: object) -> None:
        """Re-fit numbered chips when the bar's available width changes."""
        if not self._bookmarks:
            return
        signature = self._fit_signature(self._content_width())
        if signature == getattr(self, "_bookmark_fit_signature", None):
            return
        self._bookmark_fit_signature = signature
        token = self._next_bookmark_rebuild_token()
        self.call_later(self._rerender_bookmarks, token)

    async def _rerender_bookmarks(self, token: int) -> None:
        """Rebuild chips against the current width (used after a resize)."""
        if token != self._bookmark_rebuild_token:
            return
        await self.remove_children()
        if token != self._bookmark_rebuild_token:
            return
        for child in self._build_children(self._content_width()):
            self.mount(child)


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
            self.mount(_search_chrome_label(pill.text, pill.class_name, pill.item_id, pill.tooltip))

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


__all__ = ["BookmarkTabBar", "FilterPillBar", "FilterPillSpec"]
