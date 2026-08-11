"""Stable widget-chrome imports plus context-sensitive footer rendering."""

from __future__ import annotations

from collections.abc import Mapping

from rich.text import Text
from textual.widgets import Static

import arxiv_browser.widgets.footer_status as _footer_status
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import theme_colors_for
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_ARROW_WIDTH as DATE_NAV_ARROW_WIDTH,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_CONTAINER_PADDING as DATE_NAV_CONTAINER_PADDING,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_ITEM_PADDING as DATE_NAV_ITEM_PADDING,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_LABEL_MODES as DATE_NAV_LABEL_MODES,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_LABEL_MONTH_DAY as DATE_NAV_LABEL_MONTH_DAY,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_LABEL_NUMERIC as DATE_NAV_LABEL_NUMERIC,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_LABEL_WIDTH as DATE_NAV_LABEL_WIDTH,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_LABEL_WITH_COUNTS as DATE_NAV_LABEL_WITH_COUNTS,
)
from arxiv_browser.widgets.date_navigation import (
    DATE_NAV_WINDOW_SIZE as DATE_NAV_WINDOW_SIZE,
)
from arxiv_browser.widgets.date_navigation import DateNavigator as DateNavigator
from arxiv_browser.widgets.date_navigation import (
    _compute_responsive_date_plan as _compute_responsive_date_plan,
)
from arxiv_browser.widgets.date_navigation import (
    _compute_window_bounds as _compute_window_bounds,
)
from arxiv_browser.widgets.date_navigation import (
    _estimate_date_nav_width as _estimate_date_nav_width,
)
from arxiv_browser.widgets.date_navigation import (
    _format_date_nav_label as _format_date_nav_label,
)
from arxiv_browser.widgets.search_chrome import BOOKMARK_MAX_TABS as BOOKMARK_MAX_TABS
from arxiv_browser.widgets.search_chrome import BookmarkTabBar as BookmarkTabBar
from arxiv_browser.widgets.search_chrome import FilterPillBar as FilterPillBar
from arxiv_browser.widgets.search_chrome import FilterPillSpec as FilterPillSpec
from arxiv_browser.widgets.search_chrome import (
    _bookmark_chip_text as _bookmark_chip_text,
)
from arxiv_browser.widgets.search_chrome import (
    _elide_bookmark_name as _elide_bookmark_name,
)
from arxiv_browser.widgets.search_chrome import (
    _fit_bookmark_chips as _fit_bookmark_chips,
)
from arxiv_browser.widgets.search_chrome import _pill_order as _pill_order
from arxiv_browser.widgets.search_chrome import (
    _update_filter_pill as _update_filter_pill,
)
from arxiv_browser.widgets.search_chrome import (
    _watch_filter_pill_spec as _watch_filter_pill_spec,
)

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
        /* panel background separates the key-hint band from the adjacent
           status bar so the two bottom strips don't blur together. */
        background: $th-panel;
        color: $th-muted;
        padding: 0 1;
    }
    """

    def render_bindings(self, bindings: list[tuple[str, str]], mode_badge: str = "") -> None:
        """Update the footer with a list of (key, label) binding hints."""
        self._footer_bindings = list(bindings[:MAX_FOOTER_HINTS])
        self._footer_mode_badge = mode_badge
        self._render_footer()

    def on_resize(self, event: object) -> None:
        """Re-fit stored hints when the footer's available width changes."""
        if getattr(self, "_footer_bindings", None) is not None:
            self._render_footer()

    def _render_footer(self) -> None:
        """Render stored hints, dropping the lowest-priority ones to fit the width."""
        colors = theme_colors_for(self)
        badge = getattr(self, "_footer_mode_badge", "")
        hints = _fit_footer_hints(
            getattr(self, "_footer_bindings", []),
            badge,
            self.content_size.width,
        )
        parts = _build_footer_parts(hints, badge, colors)
        self.update("  ".join(parts))


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


# Help and the command palette are the two escape hatches; never drop them.
_FOOTER_PROTECTED_KEYS = frozenset({"?", "Ctrl+p"})
# Lower rank drops first when the footer is too narrow; unlisted keys use the default (last).
_FOOTER_DROP_RANKS: dict[str, int] = {
    "[/]": 0,
    "e": 0,
    "L": 0,
    "V": 0,
    "x": 0,
    "E": 1,
    "s": 2,
    "Space": 3,
}
_FOOTER_DEFAULT_DROP_RANK = 4
_FOOTER_HINT_GAP = 2  # cells inserted between rendered parts by "  ".join()


def _footer_hint_width(key: str, label: str) -> int:
    """Return the painted cell width of one footer hint (no markup)."""
    if key and label:
        return len(key) + 1 + len(label)
    return len(label or key)


def _footer_total_width(hints: list[tuple[str, str]], badge: str) -> int:
    """Return the total painted width of a badge plus hint list."""
    widths = [Text.from_markup(badge).cell_len] if badge else []
    widths.extend(_footer_hint_width(key, label) for key, label in hints)
    if not widths:
        return 0
    return sum(widths) + _FOOTER_HINT_GAP * (len(widths) - 1)


def _lowest_priority_hint_index(hints: list[tuple[str, str]]) -> int | None:
    """Return the index of the next hint to drop, or None if all are protected."""
    best: tuple[int, int] | None = None
    for index, (key, _label) in enumerate(hints):
        if key in _FOOTER_PROTECTED_KEYS:
            continue
        rank = _FOOTER_DROP_RANKS.get(key, _FOOTER_DEFAULT_DROP_RANK)
        if best is None or rank < best[0] or (rank == best[0] and index > best[1]):
            best = (rank, index)
    return None if best is None else best[1]


def _fit_footer_hints(
    bindings: list[tuple[str, str]],
    badge: str,
    width: int,
) -> list[tuple[str, str]]:
    """Drop lowest-priority hints until the footer fits ``width``.

    Help and command palette are never dropped; ``width`` <= 0 keeps every hint.
    """
    hints = list(bindings)
    if width <= 0:
        return hints
    while _footer_total_width(hints, badge) > width:
        index = _lowest_priority_hint_index(hints)
        if index is None:
            break
        hints.pop(index)
    return hints


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
