# ruff: noqa: F401, F403, F405
"""Shared runtime imports and patch-surface syncing for browser modules.

This module is the compatibility seam between the newer browser-package mixins
and the legacy ``arxiv_browser.app`` patch surface that some tests still target.
Browser mixins import from here so they can share one dependency bundle and so
their callables can refresh same-named globals from ``app.py`` immediately
before execution.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass

from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.timer import Timer
from textual.widgets import Header, Input, Label, OptionList, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from arxiv_browser import widgets as _widgets
from arxiv_browser.actions._runtime import *
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

# Command palette registry: (name, description, key_hint, action_name)
COMMAND_PALETTE_COMMANDS: list[tuple[str, str, str, str]] = [
    ("Search Papers", "Filter papers by text, category, or tag", "/", "toggle_search"),
    ("Search Syntax", "Open search examples and operators", "", "show_search_syntax"),
    ("Search arXiv API", "Search all of arXiv online", "A", "arxiv_search"),
    ("Previous Date", "Navigate to older date file", "[", "prev_date"),
    ("Next Date", "Navigate to newer date file", "]", "next_date"),
    ("Open in Browser", "Open selected paper(s) in web browser", "o", "open_url"),
    ("Open PDF", "Open selected paper(s) as PDF", "P", "open_pdf"),
    ("Download PDF", "Download PDF(s) to local folder", "d", "download_pdf"),
    ("Copy to Clipboard", "Copy paper info to clipboard", "c", "copy_selected"),
    ("Toggle Read", "Mark paper(s) as read/unread", "r", "toggle_read"),
    ("Toggle Star", "Star/unstar paper(s)", "x", "toggle_star"),
    ("Edit Notes", "Add or edit notes for current paper", "n", "edit_notes"),
    ("Edit Tags", "Add or edit tags (bulk when selected)", "t", "edit_tags"),
    ("Select All", "Select all visible papers", "a", "select_all"),
    ("Clear Selection", "Deselect all papers", "u", "clear_selection"),
    ("Toggle Selection", "Toggle selection on current paper", "Space", "toggle_select"),
    (
        "Cycle Sort",
        "Cycle sort: title/date/arxiv_id/citations/trending/relevance",
        "s",
        "cycle_sort",
    ),
    (
        "Show Watched Papers",
        "Filter the list to papers matching your watch list",
        "w",
        "toggle_watch_filter",
    ),
    ("Manage Watch List", "Add/remove watch list patterns", "W", "manage_watch_list"),
    ("Show Abstract Preview", "Reveal abstract snippets in the paper list", "p", "toggle_preview"),
    ("Export Menu", "Export as BibTeX, Markdown, RIS, or CSV", "E", "export_menu"),
    ("Export Metadata", "Export all annotations to a JSON snapshot", "", "export_metadata"),
    ("Import Metadata", "Import annotations from a chosen JSON snapshot", "", "import_metadata"),
    (
        "Fetch S2 Data",
        "Fetch Semantic Scholar data for the current paper (requires S2 enabled)",
        "e",
        "fetch_s2",
    ),
    (
        "Toggle Semantic Scholar",
        "Enable or disable Semantic Scholar enrichment",
        "Ctrl+e",
        "ctrl_e_dispatch",
    ),
    (
        "Enable HuggingFace Trending",
        "Show HuggingFace badges and detail-pane matches",
        "Ctrl+h",
        "toggle_hf",
    ),
    ("Check Versions", "Check starred papers for arXiv updates", "V", "check_versions"),
    (
        "Citation Graph",
        "Explore the citation graph for the current paper (requires S2 data)",
        "G",
        "citation_graph",
    ),
    (
        "AI Summary",
        "Generate an LLM-powered paper summary (requires LLM configuration)",
        "Ctrl+s",
        "generate_summary",
    ),
    (
        "Chat with Paper",
        "Interactive Q&A about the current paper (requires LLM configuration)",
        "C",
        "chat_with_paper",
    ),
    (
        "Score Relevance",
        "LLM-score loaded papers by research interests (requires LLM configuration)",
        "L",
        "score_relevance",
    ),
    ("Edit Interests", "Edit research interests for relevance scoring", "Ctrl+l", "edit_interests"),
    (
        "Auto-Tag",
        "Suggest tags for current or selected papers (requires LLM configuration)",
        "Ctrl+g",
        "auto_tag",
    ),
    (
        "Similar Papers",
        "Find similar papers locally or via Semantic Scholar when available",
        "R",
        "show_similar",
    ),
    ("Add Bookmark", "Save current search as bookmark", "Ctrl+b", "add_bookmark"),
    ("Collections", "Manage paper reading lists", "Ctrl+k", "collections"),
    ("Add to Collection", "Add papers to a reading list", "", "add_to_collection"),
    (
        "Toggle Detail Density",
        "Switch between scan and full detail views",
        "v",
        "toggle_detail_mode",
    ),
    ("Cycle Theme", "Switch between Monokai/Catppuccin/Solarized", "Ctrl+t", "cycle_theme"),
    ("Toggle Sections", "Show/hide detail pane sections", "Ctrl+d", "toggle_sections"),
    ("Help", "Show all keyboard shortcuts", "?", "show_help"),
    ("Set Mark", "Set a named mark (a-z) at current position", "m", "start_mark"),
    ("Jump to Mark", "Jump to a named mark (a-z)", "'", "start_goto_mark"),
]


@dataclass(slots=True, frozen=True)
class _PaletteAppState:
    """Minimal app state snapshot for command-palette availability decisions."""

    in_arxiv_api_mode: bool
    hf_active: bool
    watch_filter_active: bool
    show_abstract_preview: bool
    detail_mode: str
    active_query: str
    has_history_navigation: bool
    watch_list: list[WatchListEntry]
    has_marks: bool
    has_starred: bool
    llm_configured: bool
    has_visible_papers: bool
    has_selection: bool
    has_current_paper: bool
    has_target_papers: bool
    s2_active: bool
    s2_data_loaded: bool


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


def sync_browser_globals(namespace: dict[str, object]) -> None:
    """Refresh one browser module's globals from the legacy app patch surface.

    During the compatibility window, tests may monkeypatch symbols on
    ``arxiv_browser.app`` rather than on the extracted browser modules directly.
    Re-syncing just before a method call keeps those patches visible inside the
    browser package without reintroducing app-module imports in every mixin.
    """
    sync_app_globals(namespace)


def _sync_wrapped_callable(func):
    """Wrap a callable so browser globals are refreshed before each invocation.

    The wrapper preserves sync vs async behavior. The important invariant is
    that every public mixin method sees the latest compatibility patches before
    any imported helper is resolved from its defining module's globals.
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _async_wrapped(*args, **kwargs):
            sync_browser_globals(func.__globals__)
            return await func(*args, **kwargs)

        return _async_wrapped

    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        sync_browser_globals(func.__globals__)
        return func(*args, **kwargs)

    return _wrapped


def sync_app_methods(cls):
    """Decorate a mixin/class so every callable re-syncs compatibility globals.

    This is applied to extracted mixins whose methods previously lived in
    ``arxiv_browser.app``. It keeps constructor-time imports static while still
    honoring one-release shims and test patches aimed at the old module path.
    """
    for name, value in list(cls.__dict__.items()):
        if name.startswith("__"):
            continue
        if isinstance(value, staticmethod):
            setattr(cls, name, staticmethod(_sync_wrapped_callable(value.__func__)))
            continue
        if isinstance(value, classmethod):
            setattr(cls, name, classmethod(_sync_wrapped_callable(value.__func__)))
            continue
        if callable(value):
            setattr(cls, name, _sync_wrapped_callable(value))
    return cls


__all__ = [name for name in globals() if not name.startswith("__")]  # pyright: ignore[reportUnsupportedDunderAll]
