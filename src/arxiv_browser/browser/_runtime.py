# ruff: noqa: F401, F403
"""Shared runtime imports for browser modules."""

from __future__ import annotations

import logging
import time
from collections import deque

from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.timer import Timer
from textual.widgets import Header, Input, Label, OptionList, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from arxiv_browser import widgets as _widgets
from arxiv_browser.actions._runtime import *
from arxiv_browser.browser.contracts import COMMAND_PALETTE_COMMANDS, _PaletteAppState
from arxiv_browser.query import (
    _HIGHLIGHT_PATTERN_CACHE,
    apply_watch_filter,
    execute_query_filter,
    get_query_tokens,
    remove_query_token,
)
from arxiv_browser.services.interfaces import AppServices, build_default_app_services
from arxiv_browser.ui_constants import APP_BINDINGS, APP_CSS
from arxiv_browser.ui_runtime import UiRefreshCoordinator, UiRefs
from arxiv_browser.widgets import chrome as _widget_chrome
from arxiv_browser.widgets import details as _widget_details
from arxiv_browser.widgets import listing as _widget_listing

BookmarkTabBar = _widgets.BookmarkTabBar
ContextFooter = _widgets.ContextFooter
DATE_NAV_WINDOW_SIZE = _widgets.DATE_NAV_WINDOW_SIZE
DateNavigator = _widgets.DateNavigator
DETAIL_CACHE_MAX = _widgets.DETAIL_CACHE_MAX
DetailRenderState = _widget_details.DetailRenderState
FilterPillBar = _widgets.FilterPillBar
PaperDetails = _widgets.PaperDetails
PaperHighlightTerms = _widget_listing.PaperHighlightTerms
PaperListItem = _widgets.PaperListItem
PaperRowRenderState = _widget_listing.PaperRowRenderState
PREVIEW_ABSTRACT_MAX_LEN = _widgets.PREVIEW_ABSTRACT_MAX_LEN
StatusBarState = _widget_chrome.StatusBarState
render_paper_option = _widgets.render_paper_option
_detail_cache_key = _widget_details._detail_cache_key

logger = logging.getLogger("arxiv_browser.browser")

# Shared browser-package constants used by extracted mixins and class helpers.
FUZZY_SCORE_CUTOFF = 60
FUZZY_LIMIT = 100
ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_ABSTRACT_LOADS = 32
BADGE_COALESCE_DELAY = 0.05
PDF_DOWNLOAD_TIMEOUT = 60


def build_list_empty_message(
    *,
    query: str,
    in_arxiv_api_mode: bool,
    watch_filter_active: bool,
    history_mode: bool,
) -> str:
    """Build actionable empty-state copy for the paper list."""
    if query:
        return (
            "[dim italic]No papers match your search.[/]\n"
            "[dim]Try: edit the query or press [bold]Esc[/bold] to clear search.[/]\n"
            "[dim]Next: press [bold]?[/bold] for shortcuts or [bold]Ctrl+p[/bold] for commands.[/]"
        )
    if in_arxiv_api_mode:
        return (
            "[dim italic]No API results on this page.[/]\n"
            "[dim]Try: adjust the query or press [bold][[/bold]/[bold]][/bold] to change pages.[/]\n"
            "[dim]Next: press [bold]Esc[/bold] to return to your local library.[/]"
        )
    if watch_filter_active:
        return (
            "[dim italic]No watched papers found.[/]\n"
            "[dim]Try: press [bold]w[/bold] to show all papers.[/]\n"
            "[dim]Next: press [bold]W[/bold] to manage watch list patterns.[/]"
        )
    if history_mode:
        return (
            "[dim italic]No papers available for this date.[/]\n"
            "[dim]Try: press [bold][[/bold] or [bold]][/bold] to change dates.[/]\n"
            "[dim]Next: press [bold]A[/bold] to search arXiv.[/]"
        )
    return (
        "[dim italic]No papers available.[/]\n"
        "[dim]Try: press [bold]A[/bold] to search arXiv.[/]\n"
        "[dim]Next: load a history file or run with [bold]-i[/bold] <file>.[/]"
    )


__all__ = [name for name in globals() if not name.startswith("__")]  # pyright: ignore[reportUnsupportedDunderAll]
