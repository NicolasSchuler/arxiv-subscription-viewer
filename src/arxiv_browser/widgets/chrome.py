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

_SELECTION_FOOTER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("o", "open"),
    ("r", "read"),
    ("x", "star"),
    ("t", "tags"),
    ("E", "export"),
    ("d", "download"),
    ("u", "clear"),
    ("?", "help"),
)

_SEARCH_FOOTER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("type to search", ""),
    ("Enter", "apply"),
    ("Esc", "clear"),
    ("↑↓", "move"),
    ("?", "help"),
)

_API_FOOTER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("[/]", "page"),
    ("Esc/Ctrl+e", "exit"),
    ("A", "new query"),
    ("o", "open"),
    ("?", "help"),
)


def build_selection_footer_base_bindings() -> list[tuple[str, str]]:
    """Return canonical selection-mode footer hints."""
    return list(_SELECTION_FOOTER_BINDINGS)


def build_search_footer_bindings() -> list[tuple[str, str]]:
    """Return canonical search-mode footer hints."""
    return list(_SEARCH_FOOTER_BINDINGS)


def build_api_footer_bindings() -> list[tuple[str, str]]:
    """Return canonical API-mode footer hints."""
    return list(_API_FOOTER_BINDINGS)


def build_selection_footer_bindings(selected_count: int) -> list[tuple[str, str]]:
    """Build selection-mode footer bindings with dynamic open(n) label."""
    bindings = build_selection_footer_base_bindings()
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
    """Build a capped default browsing footer with deterministic priority."""
    slot_a = ("[/]", "history") if has_history_navigation else ("n", "notes")
    if s2_active:
        slot_b = ("e", "S2")
    elif has_starred:
        slot_b = ("V", "versions")
    elif llm_configured:
        slot_b = ("L", "relevance")
    else:
        slot_b = ("t", "tags")

    return [
        ("/", "search"),
        ("o", "open"),
        ("s", "sort"),
        ("r", "read"),
        ("x", "star"),
        slot_a,
        slot_b,
        ("E", "export"),
        ("Ctrl+p", "palette"),
        ("?", "help"),
    ]


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
        compact_parts = _build_compact_status_parts(
            total=total,
            filtered=filtered,
            query=query,
            watch_filter_active=watch_filter_active,
            selected_count=selected_count,
            sort_label=sort_label,
            in_arxiv_api_mode=in_arxiv_api_mode,
            api_page=api_page,
            arxiv_api_loading=arxiv_api_loading,
            show_abstract_preview=show_abstract_preview,
            s2_active=s2_active,
            s2_loading=s2_loading,
            s2_count=s2_count,
            hf_active=hf_active,
            hf_loading=hf_loading,
            hf_match_count=hf_match_count,
            version_checking=version_checking,
            version_update_count=version_update_count,
            max_width=max_width,
        )
        return _render_compact_status(compact_parts, max_width)

    parts = _build_full_status_parts(
        total=total,
        filtered=filtered,
        query=query,
        watch_filter_active=watch_filter_active,
        selected_count=selected_count,
        sort_label=sort_label,
        in_arxiv_api_mode=in_arxiv_api_mode,
        api_page=api_page,
        arxiv_api_loading=arxiv_api_loading,
        show_abstract_preview=show_abstract_preview,
        s2_active=s2_active,
        s2_loading=s2_loading,
        s2_count=s2_count,
        hf_active=hf_active,
        hf_loading=hf_loading,
        hf_match_count=hf_match_count,
        version_checking=version_checking,
        version_update_count=version_update_count,
    )
    rendered = " [dim]│[/] ".join(parts)
    return _truncate_rich_text(rendered, max_width)


def _compact_primary_segment(
    *, total: int, filtered: int, query: str, watch_filter_active: bool
) -> str:
    """Build the first compact segment (query/watch/default)."""
    if query:
        return f"{filtered}/{total} match"
    if watch_filter_active:
        return f"{filtered}/{total} watched"
    return f"{total} papers"


def _full_primary_segment(
    *, total: int, filtered: int, query: str, watch_filter_active: bool
) -> str:
    """Build the first rich segment (query/watch/default)."""
    if query:
        truncated_query = query if len(query) <= 30 else query[:27] + "..."
        safe_query = escape_rich_text(truncated_query)
        return (
            f"[{THEME_COLORS['accent']}]{filtered}[/][dim]/{total} matching [/]"
            f'[{THEME_COLORS["accent"]}]"{safe_query}"[/]'
        )
    if watch_filter_active:
        return f"[{THEME_COLORS['orange']}]{filtered}[/][dim]/{total} watched[/]"
    return f"[dim]{total} papers[/]"


def _compact_flag_segment(*, active: bool, loading: bool, count: int, label: str) -> str | None:
    """Return compact flag text like S2/HF status, or None when inactive."""
    if not active:
        return None
    if loading:
        return f"{label} Loading..."
    if count > 0:
        return f"{label}:{count}"
    return label


def _full_flag_segment(
    *, active: bool, loading: bool, count: int, label: str, color: str
) -> str | None:
    """Return rich-markup status text for S2/HF style toggles."""
    if not active:
        return None
    if loading:
        return f"[{color}]{label} loading...[/]"
    if count > 0:
        return f"[{color}]{label}:{count}[/]"
    return f"[{color}]{label}[/]"


def _build_compact_status_parts(
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
    max_width: int,
) -> list[str]:
    """Build compact status tokens for narrow terminals."""
    parts = [
        _compact_primary_segment(
            total=total,
            filtered=filtered,
            query=query,
            watch_filter_active=watch_filter_active,
        ),
        f"sort:{sort_label}",
    ]
    if selected_count > 0:
        parts.insert(1, f"{selected_count} selected")
    if in_arxiv_api_mode and api_page is not None:
        parts.extend([f"API p{api_page}"])
        if arxiv_api_loading:
            parts.append("Loading...")

    s2_segment = _compact_flag_segment(
        active=s2_active, loading=s2_loading, count=s2_count, label="S2"
    )
    if s2_segment:
        parts.append(s2_segment)

    if max_width >= 90:
        hf_segment = _compact_flag_segment(
            active=hf_active, loading=hf_loading, count=hf_match_count, label="HF"
        )
        if hf_segment:
            parts.append(hf_segment)

    # Keep compact mode focused on immediate context. Preview/version details
    # stay in full-width mode to reduce narrow-screen cognitive load.
    _ = (show_abstract_preview, version_checking, version_update_count)
    return parts


def _build_full_status_parts(
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
) -> list[str]:
    """Build rich status tokens for regular widths."""
    parts = [
        _full_primary_segment(
            total=total,
            filtered=filtered,
            query=query,
            watch_filter_active=watch_filter_active,
        ),
        f"[dim]Sort: {sort_label}[/]",
    ]
    if selected_count > 0:
        parts.insert(1, f"[bold {THEME_COLORS['green']}]{selected_count} selected[/]")
    if in_arxiv_api_mode and api_page is not None:
        parts.extend(
            [
                f"[{THEME_COLORS['orange']}]API[/]",
                f"[dim]Page: {api_page}[/]",
            ]
        )
        if arxiv_api_loading:
            parts.append(f"[{THEME_COLORS['orange']}]Loading...[/]")
    if show_abstract_preview:
        parts.append(f"[{THEME_COLORS['purple']}]Preview[/]")

    s2_segment = _full_flag_segment(
        active=s2_active,
        loading=s2_loading,
        count=s2_count,
        label="S2",
        color=THEME_COLORS["green"],
    )
    if s2_segment:
        parts.append(s2_segment)

    hf_segment = _full_flag_segment(
        active=hf_active,
        loading=hf_loading,
        count=hf_match_count,
        label="HF",
        color=THEME_COLORS["orange"],
    )
    if hf_segment:
        parts.append(hf_segment)

    if version_checking:
        parts.append(f"[{THEME_COLORS['pink']}]Checking versions...[/]")
    elif version_update_count > 0:
        parts.append(f"[{THEME_COLORS['pink']}]{version_update_count} updated[/]")
    return parts


def _render_compact_status(parts: list[str], max_width: int) -> str:
    """Render compact parts and shrink if necessary."""
    compact = " | ".join(parts)
    if len(compact) <= max_width:
        return compact

    while len(parts) > 1 and len(" | ".join(parts) + " ...") > max_width:
        parts.pop()
    return " | ".join(parts) + " ..."


def _truncate_rich_text(text: str, max_width: int | None) -> str:
    """Truncate rendered Rich markup by visible width when constrained."""
    if max_width is None or max_width <= 0:
        return text
    visible = re.sub(r"\[[^\]]*]", "", text)
    if len(visible) <= max_width:
        return text
    return visible[: max(0, max_width - 3)] + "..."


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
    "build_api_footer_bindings",
    "build_browse_footer_bindings",
    "build_footer_mode_badge",
    "build_search_footer_bindings",
    "build_selection_footer_base_bindings",
    "build_selection_footer_bindings",
    "build_status_bar_text",
]
