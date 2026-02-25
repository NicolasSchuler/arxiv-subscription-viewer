#!/usr/bin/env python3
"""arXiv Paper Browser TUI - Browse arXiv papers from a text file.

Usage:
    python arxiv_browser.py                    # Use default arxiv.txt
    python arxiv_browser.py -i papers.txt      # Use custom file
    python arxiv_browser.py --no-restore       # Start fresh session

Key bindings:
    /       - Toggle search (fuzzy matching)
    A       - Search all arXiv (API mode)
    o       - Open selected paper(s) in browser
    P       - Open selected paper(s) as PDF
    c       - Copy selected paper(s) to clipboard
    d       - Download PDF(s) to local folder
    E       - Export menu (BibTeX, Markdown, RIS, CSV + more)
    space   - Toggle selection
    a       - Select all visible
    u       - Clear selection
    s       - Cycle sort order (title/date/arxiv_id/citations/trending/relevance)
    j/k     - Navigate down/up (vim-style)
    r       - Toggle read status
    x       - Toggle star
    n       - Edit notes
    t       - Edit tags
    w       - Toggle watch list filter
    W       - Manage watch list
    p       - Toggle abstract preview
    m       - Set mark (then press a-z)
    '       - Jump to mark (then press a-z)
    R       - Show similar papers
    1-9     - Jump to bookmark
    Ctrl+b  - Add current search as bookmark
    V       - Check starred papers for version updates
    Ctrl+e  - Toggle S2 (browse) / Exit API (API mode)
    e       - Fetch Semantic Scholar data
    [       - Previous date (history) / previous API page (API mode)
    ]       - Next date (history) / next API page (API mode)
    q       - Quit

Search filters:
    cat:<category>  - Filter by category (e.g., cat:cs.AI)
    tag:<tag>       - Filter by tag (e.g., tag:to-read)
    author:<name>   - Filter by author substring
    title:<text>    - Filter by title substring
    abstract:<text> - Filter by abstract substring
    unread          - Show only unread papers
    starred         - Show only starred papers
    <text>          - Filter by title/author
    "quoted phrase" - Match exact phrases
    AND/OR/NOT      - Combine terms with boolean operators
"""

# ruff: noqa: F401

import asyncio
import hashlib
import logging
import platform
import sqlite3
import subprocess
import sys
import time
import webbrowser
from collections import deque
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from rapidfuzz import fuzz
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.timer import Timer
from textual.widgets import (
    Header,
    Input,
    Label,
    OptionList,
)
from textual.widgets.option_list import Option, OptionDoesNotExist

from arxiv_browser import widgets as _widgets
from arxiv_browser.action_messages import (
    build_actionable_error,
    build_actionable_warning,
    build_download_pdfs_confirmation_prompt,
    build_download_start_notification,
    build_open_papers_confirmation_prompt,
    build_open_papers_notification,
    build_open_pdfs_confirmation_prompt,
    build_open_pdfs_notification,
    requires_batch_confirmation,
)
from arxiv_browser.cli import (
    _configure_color_mode,
    _configure_logging,
    _resolve_legacy_fallback,
    _resolve_papers,
    _validate_interactive_tty,
)
from arxiv_browser.cli import (
    main as _cli_main,
)
from arxiv_browser.config import *  # noqa: F403
from arxiv_browser.config import (
    CONFIG_APP_NAME,
    CONFIG_FILENAME,
    _coerce_arxiv_api_max_results,
    _config_to_dict,
    _dict_to_config,
    _parse_collapsed_sections,
    _safe_get,
    export_metadata,
    get_config_path,
    import_metadata,
    load_config,
    save_config,
)
from arxiv_browser.enrichment import (
    apply_version_updates,
    count_hf_matches,
    get_starred_paper_ids_for_version_check,
)
from arxiv_browser.export import *  # noqa: F403
from arxiv_browser.export import (
    DEFAULT_BIBTEX_EXPORT_DIR,
    DEFAULT_PDF_DOWNLOAD_DIR,
    escape_bibtex,
    extract_year,
    format_authors_bibtex,
    format_collection_as_markdown,
    format_paper_as_bibtex,
    format_paper_as_markdown,
    format_paper_as_ris,
    format_paper_for_clipboard,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    generate_citation_key,
    get_paper_url,
    get_pdf_download_path,
    get_pdf_url,
)
from arxiv_browser.help_ui import build_help_sections
from arxiv_browser.huggingface import (
    HuggingFacePaper,
    get_hf_db_path,
    load_hf_daily_cache,
)
from arxiv_browser.io_actions import (
    build_clipboard_payload,
    build_markdown_export_document,
    build_viewer_args,
    filter_papers_needing_download,
    get_clipboard_command_plan,
    resolve_target_papers,
    write_timestamped_export_file,
)
from arxiv_browser.llm import *  # noqa: F403
from arxiv_browser.llm import (
    AUTO_TAG_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT,
    DEFAULT_LLM_PROMPT,
    LLM_COMMAND_TIMEOUT,
    LLM_PRESETS,
    RELEVANCE_PROMPT_TEMPLATE,
    SUMMARY_MODES,
    _build_llm_shell_command,
    _compute_command_hash,
    _extract_tags_from_json,
    _init_relevance_db,
    _init_summary_db,
    _load_all_relevance_scores,
    _load_relevance_score,
    _load_summary,
    _parse_auto_tag_response,
    _parse_relevance_response,
    _resolve_llm_command,
    _save_relevance_score,
    _save_summary,
    build_auto_tag_prompt,
    build_llm_prompt,
    build_relevance_prompt,
    get_relevance_db_path,
    get_summary_db_path,
)
from arxiv_browser.llm_providers import CLIProvider, LLMResult, resolve_provider
from arxiv_browser.modals import (
    AddToCollectionModal,
    ArxivSearchModal,
    AutoTagSuggestModal,
    CitationGraphScreen,
    CollectionsModal,
    CollectionViewModal,
    CommandPaletteModal,
    ConfirmModal,
    ExportMenuModal,
    HelpScreen,
    NotesModal,
    PaperChatScreen,
    RecommendationSourceModal,
    RecommendationsScreen,
    ResearchInterestsModal,
    SectionToggleModal,
    SummaryModeModal,
    TagsModal,
    WatchListModal,
)
from arxiv_browser.models import *  # noqa: F403
from arxiv_browser.models import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    ARXIV_API_MAX_RESULTS_LIMIT,
    DEFAULT_COLLAPSED_SECTIONS,
    DETAIL_SECTION_KEYS,
    DETAIL_SECTION_NAMES,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    STOPWORDS,
    WATCH_MATCH_TYPES,
    ArxivSearchModeState,
    ArxivSearchRequest,
    LocalBrowseSnapshot,
    Paper,
    PaperCollection,
    PaperMetadata,
    QueryToken,
    SearchBookmark,
    SessionState,
    UserConfig,
    WatchListEntry,
)
from arxiv_browser.parsing import *  # noqa: F403
from arxiv_browser.parsing import (
    ARXIV_DATE_FORMAT,
    ARXIV_QUERY_FIELDS,
    ATOM_NS,
    HISTORY_DATE_FORMAT,
    build_arxiv_search_query,
    build_daily_digest,
    clean_latex,
    count_papers_in_file,
    discover_history_files,
    extract_text_from_html,
    normalize_arxiv_id,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
    parse_arxiv_version_map,
)
from arxiv_browser.query import *  # noqa: F403
from arxiv_browser.query import (
    _HIGHLIGHT_PATTERN_CACHE,
    apply_watch_filter,
    build_highlight_terms,
    escape_rich_text,
    execute_query_filter,
    format_categories,
    format_summary_as_rich,
    get_query_tokens,
    highlight_text,
    insert_implicit_and,
    is_advanced_query,
    match_query_term,
    matches_advanced_query,
    paper_matches_watch_entry,
    pill_label_for_token,
    reconstruct_query,
    remove_query_token,
    render_progress_bar,
    sort_papers,
    to_rpn,
    tokenize_query,
    truncate_text,
)
from arxiv_browser.semantic_scholar import (
    S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    S2_REC_CACHE_TTL_DAYS,
    CitationEntry,
    SemanticScholarPaper,
    fetch_s2_citations,
    fetch_s2_recommendations,
    fetch_s2_references,
    get_s2_db_path,
    has_s2_citation_graph_cache,
    load_s2_citation_graph,
    load_s2_paper,
    load_s2_recommendations,
    save_s2_citation_graph,
    save_s2_recommendations,
)
from arxiv_browser.services.enrichment_service import (
    load_or_fetch_hf_daily_cached as _load_or_fetch_hf_daily_cached,
)
from arxiv_browser.services.enrichment_service import (
    load_or_fetch_s2_paper_cached as _load_or_fetch_s2_paper_cached,
)
from arxiv_browser.services.interfaces import (
    AppServices,
    build_default_app_services,
)
from arxiv_browser.services.llm_service import (
    LLMExecutionError as _LLMExecutionError,
)
from arxiv_browser.similarity import *  # noqa: F403
from arxiv_browser.similarity import (
    SIMILARITY_READ_PENALTY,
    SIMILARITY_RECENCY_DAYS,
    SIMILARITY_RECENCY_WEIGHT,
    SIMILARITY_STARRED_BOOST,
    SIMILARITY_TOP_N,
    SIMILARITY_UNREAD_BOOST,
    SIMILARITY_WEIGHT_AUTHOR,
    SIMILARITY_WEIGHT_CATEGORY,
    SIMILARITY_WEIGHT_TEXT,
    TfidfIndex,
    _compute_tf,
    _extract_author_lastnames,
    _extract_keywords,
    _jaccard_similarity,
    _tokenize_for_tfidf,
    build_similarity_corpus_key,
    compute_paper_similarity,
    find_similar_papers,
)
from arxiv_browser.themes import *  # noqa: F403
from arxiv_browser.themes import (
    CATEGORY_COLORS,
    CATPPUCCIN_MOCHA_THEME,
    DEFAULT_CATEGORY_COLOR,
    DEFAULT_CATEGORY_COLORS,
    DEFAULT_THEME,
    SOLARIZED_DARK_THEME,
    TAG_NAMESPACE_COLORS,
    TEXTUAL_THEMES,
    THEME_CATEGORY_COLORS,
    THEME_COLORS,
    THEME_NAMES,
    THEME_TAG_NAMESPACE_COLORS,
    THEMES,
    _build_textual_theme,
    get_tag_color,
    parse_tag_namespace,
)
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
FilterPillBar = _widgets.FilterPillBar
PaperDetails = _widgets.PaperDetails
PaperListItem = _widgets.PaperListItem
PREVIEW_ABSTRACT_MAX_LEN = _widgets.PREVIEW_ABSTRACT_MAX_LEN
render_paper_option = _widgets.render_paper_option
_detail_cache_key = _widget_details._detail_cache_key

# Public API for this module — includes all re-exported sub-module symbols
# so that `from arxiv_browser import X` works for the full public surface.
__all__ = [
    "ARXIV_API_DEFAULT_MAX_RESULTS",
    "ARXIV_API_MAX_RESULTS_LIMIT",
    "ARXIV_DATE_FORMAT",
    "ARXIV_QUERY_FIELDS",
    "ATOM_NS",
    "AUTO_TAG_PROMPT_TEMPLATE",
    "CATEGORY_COLORS",
    "CATPPUCCIN_MOCHA_THEME",
    "CHAT_SYSTEM_PROMPT",
    "COMMAND_PALETTE_COMMANDS",
    "CONFIG_APP_NAME",
    "CONFIG_FILENAME",
    "DEFAULT_BIBTEX_EXPORT_DIR",
    "DEFAULT_CATEGORY_COLOR",
    "DEFAULT_CATEGORY_COLORS",
    "DEFAULT_COLLAPSED_SECTIONS",
    "DEFAULT_LLM_PROMPT",
    "DEFAULT_PDF_DOWNLOAD_DIR",
    "DEFAULT_THEME",
    "DETAIL_SECTION_KEYS",
    "DETAIL_SECTION_NAMES",
    "HISTORY_DATE_FORMAT",
    "LLM_PRESETS",
    "MAX_COLLECTIONS",
    "MAX_PAPERS_PER_COLLECTION",
    "RELEVANCE_PROMPT_TEMPLATE",
    "SIMILARITY_READ_PENALTY",
    "SIMILARITY_RECENCY_DAYS",
    "SIMILARITY_RECENCY_WEIGHT",
    "SIMILARITY_STARRED_BOOST",
    "SIMILARITY_TOP_N",
    "SIMILARITY_UNREAD_BOOST",
    "SIMILARITY_WEIGHT_AUTHOR",
    "SIMILARITY_WEIGHT_CATEGORY",
    "SIMILARITY_WEIGHT_TEXT",
    "SOLARIZED_DARK_THEME",
    "SORT_OPTIONS",
    "STOPWORDS",
    "SUMMARY_MODES",
    "TAG_NAMESPACE_COLORS",
    "TEXTUAL_THEMES",
    "THEMES",
    "THEME_CATEGORY_COLORS",
    "THEME_COLORS",
    "THEME_NAMES",
    "THEME_TAG_NAMESPACE_COLORS",
    "WATCH_MATCH_TYPES",
    "AddToCollectionModal",
    "ArxivBrowser",
    "ArxivSearchModeState",
    "ArxivSearchRequest",
    "CLIProvider",
    "CollectionViewModal",
    "CollectionsModal",
    "CommandPaletteModal",
    "FilterPillBar",
    "LLMResult",
    "LocalBrowseSnapshot",
    "Paper",
    "PaperChatScreen",
    "PaperCollection",
    "PaperMetadata",
    "QueryToken",
    "RecommendationSourceModal",
    "SearchBookmark",
    "SectionToggleModal",
    "SessionState",
    "TfidfIndex",
    "UserConfig",
    "WatchListEntry",
    "_configure_logging",
    "build_arxiv_search_query",
    "build_auto_tag_prompt",
    "build_daily_digest",
    "build_highlight_terms",
    "build_llm_prompt",
    "build_relevance_prompt",
    "clean_latex",
    "compute_paper_similarity",
    "count_papers_in_file",
    "discover_history_files",
    "escape_bibtex",
    "escape_rich_text",
    "export_metadata",
    "extract_text_from_html",
    "extract_year",
    "find_similar_papers",
    "format_authors_bibtex",
    "format_categories",
    "format_collection_as_markdown",
    "format_paper_as_bibtex",
    "format_paper_as_markdown",
    "format_paper_as_ris",
    "format_paper_for_clipboard",
    "format_papers_as_csv",
    "format_papers_as_markdown_table",
    "format_summary_as_rich",
    "generate_citation_key",
    "get_config_path",
    "get_paper_url",
    "get_pdf_download_path",
    "get_pdf_url",
    "get_relevance_db_path",
    "get_summary_db_path",
    "get_tag_color",
    "highlight_text",
    "import_metadata",
    "insert_implicit_and",
    "is_advanced_query",
    "load_config",
    "main",
    "match_query_term",
    "matches_advanced_query",
    "normalize_arxiv_id",
    "paper_matches_watch_entry",
    "parse_arxiv_api_feed",
    "parse_arxiv_date",
    "parse_arxiv_file",
    "parse_arxiv_version_map",
    "parse_tag_namespace",
    "pill_label_for_token",
    "reconstruct_query",
    "render_paper_option",
    "render_progress_bar",
    "resolve_provider",
    "save_config",
    "sort_papers",
    "to_rpn",
    "tokenize_query",
    "truncate_text",
]

# Module logger for debugging
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# UI Layout constants
MIN_LIST_WIDTH = 50
MAX_LIST_WIDTH = 100
CLIPBOARD_SEPARATOR = "=" * 80

# Context-sensitive footer keybinding hints (compatibility alias kept for tests/imports).
FOOTER_CONTEXTS: dict[str, list[tuple[str, str]]] = {
    "default": _widget_chrome.build_browse_footer_bindings(
        s2_active=False,
        has_starred=False,
        llm_configured=False,
        has_history_navigation=False,
    ),
    "selection": _widget_chrome.build_selection_footer_base_bindings(),
    "search": _widget_chrome.build_search_footer_bindings(),
    "api": _widget_chrome.build_api_footer_bindings(),
}

# Command palette registry: (name, description, key_hint, action_name)
# action_name maps to ArxivBrowser.action_* methods (or "" for non-action commands)
COMMAND_PALETTE_COMMANDS: list[tuple[str, str, str, str]] = [
    # Navigation
    ("Search Papers", "Filter papers by text, category, or tag", "/", "toggle_search"),
    ("Search arXiv API", "Search all of arXiv online", "A", "arxiv_search"),
    ("Previous Date", "Navigate to older date file", "[", "prev_date"),
    ("Next Date", "Navigate to newer date file", "]", "next_date"),
    # Paper actions
    ("Open in Browser", "Open selected paper(s) in web browser", "o", "open_url"),
    ("Open PDF", "Open selected paper(s) as PDF", "P", "open_pdf"),
    ("Download PDF", "Download PDF(s) to local folder", "d", "download_pdf"),
    ("Copy to Clipboard", "Copy paper info to clipboard", "c", "copy_selected"),
    # Metadata
    ("Toggle Read", "Mark paper(s) as read/unread", "r", "toggle_read"),
    ("Toggle Star", "Star/unstar paper(s)", "x", "toggle_star"),
    ("Edit Notes", "Add or edit notes for current paper", "n", "edit_notes"),
    ("Edit Tags", "Add or edit tags (bulk when selected)", "t", "edit_tags"),
    # Selection
    ("Select All", "Select all visible papers", "a", "select_all"),
    ("Clear Selection", "Deselect all papers", "u", "clear_selection"),
    ("Toggle Selection", "Toggle selection on current paper", "Space", "toggle_select"),
    # Sort & Filter
    (
        "Cycle Sort",
        "Cycle sort: title/date/arxiv_id/citations/trending/relevance",
        "s",
        "cycle_sort",
    ),
    ("Toggle Watch Filter", "Show only watched papers", "w", "toggle_watch_filter"),
    ("Manage Watch List", "Add/remove watch list patterns", "W", "manage_watch_list"),
    ("Toggle Preview", "Show/hide abstract preview in list", "p", "toggle_preview"),
    # Export
    ("Export Menu", "Export as BibTeX, Markdown, RIS, or CSV", "E", "export_menu"),
    ("Export Metadata", "Export all annotations to portable JSON file", "", "export_metadata"),
    ("Import Metadata", "Import annotations from JSON file", "", "import_metadata"),
    # Enrichment
    ("Fetch S2 Data", "Fetch Semantic Scholar data for current paper", "e", "fetch_s2"),
    (
        "Toggle S2 / Exit API",
        "Toggle S2 in browse mode or exit API mode",
        "Ctrl+e",
        "ctrl_e_dispatch",
    ),
    ("Toggle HuggingFace", "Enable/disable HuggingFace trending", "Ctrl+h", "toggle_hf"),
    ("Check Versions", "Check starred papers for arXiv updates", "V", "check_versions"),
    ("Citation Graph", "Explore citation graph (S2-powered)", "G", "citation_graph"),
    # AI features
    ("AI Summary", "Generate LLM-powered paper summary", "Ctrl+s", "generate_summary"),
    ("Chat with Paper", "Interactive Q&A about current paper", "C", "chat_with_paper"),
    ("Score Relevance", "LLM-score papers by research interests", "L", "score_relevance"),
    ("Edit Interests", "Edit research interests for relevance scoring", "Ctrl+l", "edit_interests"),
    ("Auto-Tag", "LLM-suggest tags for current or selected papers", "Ctrl+g", "auto_tag"),
    # Recommendations
    ("Similar Papers", "Find similar papers (local or S2)", "R", "show_similar"),
    # Bookmarks
    ("Add Bookmark", "Save current search as bookmark", "Ctrl+b", "add_bookmark"),
    # Collections
    ("Collections", "Manage paper reading lists", "Ctrl+k", "collections"),
    ("Add to Collection", "Add papers to a reading list", "", "add_to_collection"),
    # UI
    ("Cycle Theme", "Switch between Monokai/Catppuccin/Solarized", "Ctrl+t", "cycle_theme"),
    ("Toggle Sections", "Show/hide detail pane sections", "Ctrl+d", "toggle_sections"),
    ("Help", "Show all keyboard shortcuts", "?", "show_help"),
    # Vim marks
    ("Set Mark", "Set a named mark (a-z) at current position", "m", "start_mark"),
    ("Jump to Mark", "Jump to a named mark (a-z)", "'", "start_goto_mark"),
]

# Subprocess timeout in seconds
SUBPROCESS_TIMEOUT = 5

# Fuzzy search settings
FUZZY_SCORE_CUTOFF = 60  # Minimum score (0-100) to include in results
FUZZY_LIMIT = 100  # Maximum number of results to return

# Paper similarity settings

# UI truncation limits
BOOKMARK_NAME_MAX_LEN = 15  # Max bookmark name display length
MAX_ABSTRACT_LOADS = 32  # Maximum concurrent abstract loads

# History file discovery limit
MAX_HISTORY_FILES = 365  # Optional caller-provided cap for history discovery.

# arXiv API search settings
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_API_MIN_INTERVAL_SECONDS = 3.0  # arXiv guidance: max 1 request / 3 seconds
ARXIV_API_TIMEOUT = 30  # Seconds to wait for arXiv API responses

PDF_DOWNLOAD_TIMEOUT = 60  # Seconds per download
MAX_CONCURRENT_DOWNLOADS = 3  # Limit parallel downloads
BATCH_CONFIRM_THRESHOLD = (
    10  # Ask for confirmation when batch operating on more than this many papers
)

# LLM summary settings
SUMMARY_DB_FILENAME = "summaries.db"
MAX_PAPER_CONTENT_LENGTH = 60_000  # ~15k tokens; truncate fetched paper content
ARXIV_HTML_TIMEOUT = 30  # Seconds to wait for arXiv HTML fetch
SUMMARY_HTML_TIMEOUT = 10  # Faster timeout for summary generation path
RELEVANCE_SCORE_TIMEOUT = 30  # Seconds to wait for relevance scoring LLM response
RELEVANCE_DB_FILENAME = "relevance.db"
AUTO_TAG_TIMEOUT = 30  # Seconds to wait for auto-tag LLM response

# Search debounce delay in seconds
SEARCH_DEBOUNCE_DELAY = 0.3
# Detail pane update debounce delay in seconds (shorter — must feel responsive)
DETAIL_PANE_DEBOUNCE_DELAY = 0.1
# Badge refresh coalesce delay — multiple badge sources within this window
# are merged into a single list iteration (50ms is imperceptible)
BADGE_COALESCE_DELAY = 0.05


async def _fetch_paper_content_async(
    paper: Paper,
    client: httpx.AsyncClient | None = None,
    timeout: int = ARXIV_HTML_TIMEOUT,
) -> str:
    """Fetch the full paper content from the arXiv HTML version.

    Falls back to the abstract if the HTML version is not available.
    If *client* is None, a temporary AsyncClient is created for this request.
    """
    html_url = f"https://arxiv.org/html/{paper.arxiv_id}"
    try:
        if client is not None:
            response = await client.get(html_url, timeout=timeout, follow_redirects=True)
        else:
            async with httpx.AsyncClient() as tmp_client:
                response = await tmp_client.get(html_url, timeout=timeout, follow_redirects=True)
        if response.status_code == 200:
            text = await asyncio.to_thread(extract_text_from_html, response.text)
            if text:
                return text[:MAX_PAPER_CONTENT_LENGTH]
        else:
            logger.warning(
                "arXiv HTML fetch returned %d for %s",
                response.status_code,
                paper.arxiv_id,
            )
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to fetch HTML for %s", paper.arxiv_id, exc_info=True)
    # Fallback to abstract
    abstract = paper.abstract or paper.abstract_raw or ""
    return f"Abstract:\n{abstract}" if abstract else ""


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
            "[dim]Try: [bold]][/bold] next page, [bold][[/bold] previous page, "
            "or [bold]A[/bold] for a new query.[/]\n"
            "[dim]Next: press [bold]Esc[/bold] or [bold]Ctrl+e[/bold] to exit API mode.[/]"
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


class ArxivBrowser(App):
    """A TUI application to browse arXiv papers."""

    TITLE = "arXiv Paper Browser"

    # Theme-aware CSS and key bindings are defined in ui_constants for maintainability.
    CSS = APP_CSS

    BINDINGS = APP_BINDINGS

    def __init__(
        self,
        papers: list[Paper],
        config: UserConfig | None = None,
        restore_session: bool = True,
        history_files: list[tuple[date, Path]] | None = None,
        current_date_index: int = 0,
        ascii_icons: bool = False,
        services: AppServices | None = None,
    ) -> None:
        super().__init__()
        # Register all Textual themes so $th-* CSS variables resolve before compose()
        for textual_theme in TEXTUAL_THEMES.values():
            self.register_theme(textual_theme)
        self.all_papers = papers
        self.filtered_papers = papers.copy()
        # Build O(1) lookup dict for papers by arxiv_id
        self._papers_by_id: dict[str, Paper] = {p.arxiv_id: p for p in papers}
        self.selected_ids: set[str] = set()  # Track selected arxiv_ids
        self._search_timer: Timer | None = None
        self._pending_query: str = ""
        self._detail_timer: Timer | None = None
        self._pending_detail_paper: Paper | None = None
        self._pending_detail_started_at: float | None = None
        self._badges_dirty: set[str] = set()
        self._badge_timer: Timer | None = None
        self._sort_index: int = 0  # Index into SORT_OPTIONS
        self._services: AppServices = services or build_default_app_services()

        # Configuration and persistence
        self._config = config or UserConfig()
        self._config.arxiv_api_max_results = _coerce_arxiv_api_max_results(
            self._config.arxiv_api_max_results
        )
        self._restore_session = restore_session

        # Theme and category overrides
        self._apply_category_overrides()
        self._apply_theme_overrides()

        # History mode: multiple date files
        self._history_files: list[tuple[date, Path]] = history_files or []
        self._current_date_index: int = current_date_index

        # Watch list: pre-compute matching papers for O(1) lookup
        self._watched_paper_ids: set[str] = set()
        self._watch_filter_active: bool = False
        self._compute_watched_papers()

        # Abstract preview toggle
        self._show_abstract_preview: bool = self._config.show_abstract_preview

        # Abstract cache for lazy loading
        self._abstract_cache: dict[str, str] = {}
        self._abstract_loading: set[str] = set()
        self._abstract_queue: deque[Paper] = deque()
        self._abstract_pending_ids: set[str] = set()

        # Bookmark state
        self._active_bookmark_index: int = -1  # -1 means no active bookmark

        # Fuzzy search match scores
        self._match_scores: dict[str, float] = {}

        # Highlight terms for incremental search
        self._highlight_terms: dict[str, list[str]] = {
            "title": [],
            "author": [],
            "abstract": [],
        }

        # Vim-style marks state
        self._pending_mark_action: str | None = None  # "set" or "goto"

        # Background task tracking (prevent GC of fire-and-forget tasks)
        self._background_tasks: set[asyncio.Task[None]] = set()

        # PDF download state
        self._download_queue: deque[Paper] = deque()
        self._downloading: set[str] = set()  # arxiv_ids currently downloading
        self._download_results: dict[str, bool] = {}  # arxiv_id -> success
        self._download_total: int = 0  # Total papers in current batch

        # LLM summary state
        self._paper_summaries: dict[str, str] = {}  # arxiv_id -> summary (in-memory cache)
        self._summary_loading: set[str] = set()  # arxiv_ids with generation in progress
        self._summary_db_path: Path = get_summary_db_path()
        self._summary_mode_label: dict[str, str] = {}  # arxiv_id -> mode display name
        self._summary_command_hash: dict[str, str] = {}  # arxiv_id -> command+prompt hash

        # arXiv API mode state
        self._in_arxiv_api_mode: bool = False
        self._arxiv_search_state: ArxivSearchModeState | None = None
        self._local_browse_snapshot: LocalBrowseSnapshot | None = None
        self._arxiv_api_fetch_inflight: bool = False
        self._arxiv_api_loading: bool = False
        self._last_arxiv_api_request_at: float = 0.0
        self._arxiv_api_request_token: int = 0

        # Shared HTTP client for connection pooling (created in on_mount)
        self._http_client: httpx.AsyncClient | None = None

        # LLM provider (resolved from config)
        self._llm_provider: CLIProvider | None = resolve_provider(self._config)

        # Accessibility: allow ASCII-only indicators for terminals/fonts
        # that do not render emoji or box symbols well.
        _widget_listing.set_ascii_icons(ascii_icons)
        _widget_details.set_ascii_glyphs(ascii_icons)

        # Semantic Scholar enrichment state
        self._s2_active: bool = False  # Runtime toggle (set from config in on_mount)
        self._s2_cache: dict[str, SemanticScholarPaper] = {}  # In-memory cache
        self._s2_loading: set[str] = set()  # In-flight dedup
        self._s2_db_path: Path = get_s2_db_path()

        # HuggingFace trending state
        self._hf_active: bool = False  # Runtime toggle (set from config in on_mount)
        self._hf_cache: dict[str, HuggingFacePaper] = {}  # In-memory cache
        self._hf_loading: bool = False  # Single bool (bulk fetch)
        self._hf_db_path: Path = get_hf_db_path()

        # Version tracking state (ephemeral per-session)
        self._version_updates: dict[str, tuple[int, int]] = {}  # arxiv_id -> (old, new)
        self._version_checking: bool = False
        self._version_progress: tuple[int, int] | None = None  # (batch, total_batches)

        # Relevance scoring state
        self._relevance_scores: dict[str, tuple[int, str]] = {}  # arxiv_id -> (score, reason)
        self._relevance_scoring_active: bool = False
        self._scoring_progress: tuple[int, int] | None = None  # (current, total)
        self._relevance_db_path: Path = get_relevance_db_path()

        # Auto-tagging state
        self._auto_tag_active: bool = False
        self._auto_tag_progress: tuple[int, int] | None = None  # (current, total)

        # TF-IDF similarity index state
        self._tfidf_index: TfidfIndex | None = None
        self._tfidf_corpus_key: str | None = None
        self._tfidf_build_task: asyncio.Task[None] | None = None
        self._pending_similarity_paper_id: str | None = None

        # Internal UI boundaries (cached refs + refresh orchestration)
        self._ui_refs = UiRefs()
        self._ui_refresh = UiRefreshCoordinator(
            refresh_list_view=self._refresh_list_view,
            update_list_header=self._update_list_header,
            update_status_bar=self._update_status_bar,
            update_filter_pills=self._update_filter_pills,
            refresh_detail_pane=self._refresh_detail_pane,
            refresh_current_list_item=self._refresh_current_list_item,
        )

    def _get_services(self) -> AppServices:
        """Return app service interfaces, lazily creating defaults for test doubles."""
        services = getattr(self, "_services", None)
        if services is None:
            services = build_default_app_services()
            self._services = services
        return services

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-pane"):
                yield Label(f" Papers ({len(self.all_papers)} total)", id="list-header")
                yield DateNavigator(self._history_files, self._current_date_index)
                yield BookmarkTabBar(self._config.bookmarks, self._active_bookmark_index)
                yield FilterPillBar()
                with Vertical(id="search-container"):
                    yield Input(
                        placeholder=(
                            " Filter: text, author:, title:, cat:, tag:, unread, starred, AND/OR/NOT"
                        ),
                        id="search-input",
                    )
                yield OptionList(id="paper-list")
                yield Label("", id="status-bar")
            with Vertical(id="right-pane"):
                yield Label(" Paper Details", id="details-header")
                with VerticalScroll(id="details-scroll"):
                    yield PaperDetails()
        yield ContextFooter()

    def on_mount(self) -> None:
        """Called when app is mounted. Restores session state if enabled."""
        # Create shared HTTP client for connection pooling
        self._http_client = httpx.AsyncClient()

        # Warn if config was corrupt and defaults were used
        if self._config.config_defaulted:
            self.notify(
                "Config file was corrupt and has been backed up. Using defaults.",
                severity="warning",
                timeout=8,
            )

        # Initialize S2 runtime state from config
        self._s2_active = self._config.s2_enabled

        # Initialize HF runtime state from config
        self._hf_active = self._config.hf_enabled
        if self._hf_active:
            self._track_task(self._fetch_hf_daily())

        # Initialize date navigator if in history mode
        if self._is_history_mode() and len(self._history_files) > 1:
            self.call_after_refresh(self._refresh_date_navigator)

        # Set subtitle with date info if in history mode
        current_date = self._get_current_date()
        if current_date:
            self.sub_title = (
                f"{len(self.all_papers)} papers · {current_date.strftime(HISTORY_DATE_FORMAT)}"
            )
        else:
            self.sub_title = f"{len(self.all_papers)} papers loaded"

        # Restore session state if enabled
        if self._restore_session and self._config.session:
            session = self._config.session
            self._sort_index = session.sort_index
            self.selected_ids = set(session.selected_ids)

            # Apply saved filter if any
            if session.current_filter:
                search_input = self._get_search_input_widget()
                search_input.value = session.current_filter
                self._apply_filter(session.current_filter)  # calls _refresh_list_view
            else:
                self._refresh_list_view()  # populate without filter

            # Restore scroll position
            option_list = self._get_paper_list_widget()
            if option_list.option_count > 0:
                # Clamp index to valid range
                index = min(session.scroll_index, option_list.option_count - 1)
                option_list.highlighted = max(0, index)
        else:
            # Populate list (deferred from compose for faster first paint)
            self._refresh_list_view()
            # Default: select first item if available
            option_list = self._get_paper_list_widget()
            if option_list.option_count > 0:
                option_list.highlighted = 0
        self._update_status_bar()

        self._notify_watch_list_matches()

        logger.debug(
            "App mounted: %d papers, history_mode=%s, s2=%s, hf=%s",
            len(self.all_papers),
            self._is_history_mode(),
            self._s2_active,
            self._hf_active,
        )

        self._prime_ui_refs()

        # Focus the paper list so key bindings work
        try:
            self._get_paper_list_widget().focus()
        except NoMatches:
            pass

    async def on_unmount(self) -> None:
        """Called when app is unmounted. Saves session state and cleans up timers.

        Uses atomic swap pattern to avoid race conditions with timer callbacks.
        """
        # Save session state before exit
        self._save_session_state()

        # Clean up timers
        timer = self._search_timer
        self._search_timer = None
        if timer is not None:
            timer.stop()
        detail_timer = self._detail_timer
        self._detail_timer = None
        if detail_timer is not None:
            detail_timer.stop()
        badge_timer = self._badge_timer
        self._badge_timer = None
        if badge_timer is not None:
            badge_timer.stop()

        # Cancel tracked background tasks to avoid leaks during teardown.
        background_tasks = getattr(self, "_background_tasks", set())
        pending = [task for task in background_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=0.5)
            for task in still_pending:
                logger.debug("Background task did not cancel before shutdown: %r", task)
        if hasattr(self, "_background_tasks"):
            self._background_tasks.clear()
        self._tfidf_build_task = None

        # Close shared HTTP client
        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception as e:
                logger.debug(
                    "Failed to close shared HTTP client during shutdown: %s", e, exc_info=True
                )
        ui_refs = getattr(self, "_ui_refs", None)
        if ui_refs is not None:
            ui_refs.reset()

    @staticmethod
    def _is_live_widget(widget: Any) -> bool:
        """Return True for mounted/attached widgets safe to reuse."""
        return bool(widget is not None and getattr(widget, "is_attached", False))

    def _get_cached_widget(self, ref_name: str, resolver: Callable[[], Any]) -> Any:
        """Resolve and cache a widget reference by UiRefs attribute name."""
        ui_refs = getattr(self, "_ui_refs", None)
        if ui_refs is None:
            return resolver()
        widget = getattr(ui_refs, ref_name)
        if self._is_live_widget(widget):
            return widget
        widget = resolver()
        setattr(ui_refs, ref_name, widget)
        return widget

    def _get_search_input_widget(self) -> Input:
        return self._get_cached_widget(
            "search_input", lambda: self.query_one("#search-input", Input)
        )

    def _get_search_container_widget(self) -> Any:
        return self._get_cached_widget(
            "search_container", lambda: self.query_one("#search-container")
        )

    def _get_paper_list_widget(self) -> OptionList:
        return self._get_cached_widget(
            "paper_list", lambda: self.query_one("#paper-list", OptionList)
        )

    def _get_list_header_widget(self) -> Label:
        return self._get_cached_widget("list_header", lambda: self.query_one("#list-header", Label))

    def _get_status_bar_widget(self) -> Label:
        return self._get_cached_widget("status_bar", lambda: self.query_one("#status-bar", Label))

    def _get_footer_widget(self) -> ContextFooter:
        return self._get_cached_widget("footer", lambda: self.query_one(ContextFooter))

    def _get_date_navigator_widget(self) -> DateNavigator:
        return self._get_cached_widget("date_navigator", lambda: self.query_one(DateNavigator))

    def _get_filter_pill_bar_widget(self) -> FilterPillBar:
        return self._get_cached_widget("filter_pill_bar", lambda: self.query_one(FilterPillBar))

    def _get_bookmark_bar_widget(self) -> BookmarkTabBar:
        return self._get_cached_widget("bookmark_bar", lambda: self.query_one(BookmarkTabBar))

    def _get_paper_details_widget(self) -> PaperDetails:
        return self._get_cached_widget("paper_details", lambda: self.query_one(PaperDetails))

    def _prime_ui_refs(self) -> None:
        """Warm caches for frequently queried widgets once the DOM is mounted."""
        for ref_name, getter in (
            ("search_input", self._get_search_input_widget),
            ("search_container", self._get_search_container_widget),
            ("paper_list", self._get_paper_list_widget),
            ("list_header", self._get_list_header_widget),
            ("status_bar", self._get_status_bar_widget),
            ("footer", self._get_footer_widget),
            ("date_navigator", self._get_date_navigator_widget),
            ("filter_pill_bar", self._get_filter_pill_bar_widget),
            ("bookmark_bar", self._get_bookmark_bar_widget),
            ("paper_details", self._get_paper_details_widget),
        ):
            try:
                getattr(self._ui_refs, ref_name)
                getter()
            except NoMatches:
                continue

    def _get_ui_refresh_coordinator(self) -> UiRefreshCoordinator:
        """Return the UI refresh coordinator, lazily creating it for __new__ tests."""
        coordinator = getattr(self, "_ui_refresh", None)
        if coordinator is None:
            coordinator = UiRefreshCoordinator(
                refresh_list_view=self._refresh_list_view,
                update_list_header=self._update_list_header,
                update_status_bar=self._update_status_bar,
                update_filter_pills=self._update_filter_pills,
                refresh_detail_pane=self._refresh_detail_pane,
                refresh_current_list_item=self._refresh_current_list_item,
            )
            self._ui_refresh = coordinator
        return coordinator

    def _refresh_date_navigator(self) -> None:
        """Refresh date navigator labels after DOM updates settle."""
        try:
            date_nav = self._get_date_navigator_widget()
        except NoMatches:
            return
        self._track_task(date_nav.update_dates(self._history_files, self._current_date_index))

    def _track_task(self, coro: Any) -> asyncio.Task[None]:
        """Create an asyncio task and track it to prevent garbage collection."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._on_task_done)
        return task

    @staticmethod
    def _on_task_done(task: asyncio.Task[None]) -> None:
        """Log unhandled exceptions from background tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Unhandled exception in background task: %s", exc, exc_info=exc)

    def _apply_category_overrides(self) -> None:
        """Apply category color overrides from config.

        Layers: default → per-theme → user overrides.
        """
        CATEGORY_COLORS.clear()
        CATEGORY_COLORS.update(DEFAULT_CATEGORY_COLORS)
        theme_cats = THEME_CATEGORY_COLORS.get(self._config.theme_name)
        if theme_cats:
            CATEGORY_COLORS.update(theme_cats)
        CATEGORY_COLORS.update(self._config.category_colors)
        format_categories.cache_clear()

    def _apply_theme_overrides(self) -> None:
        """Apply theme overrides from config to both Rich markup and CSS variables.

        Layers: named base theme → per-key overrides from config.
        Also updates TAG_NAMESPACE_COLORS for per-theme tag styling.
        """
        base = THEMES.get(self._config.theme_name, DEFAULT_THEME)
        THEME_COLORS.clear()
        THEME_COLORS.update(base)
        THEME_COLORS.update(self._config.theme)
        # Rebuild and activate Textual theme for CSS variable resolution
        if self._config.theme:
            merged = dict(base)
            merged.update(self._config.theme)
            try:
                self.register_theme(_build_textual_theme(self._config.theme_name, merged))
            except Exception as e:
                logger.debug("Skipping theme registration in current context: %s", e, exc_info=True)
        try:
            self.theme = self._config.theme_name
        except Exception as e:
            logger.debug("Skipping theme activation in current context: %s", e, exc_info=True)
        # Apply per-theme tag namespace colors
        TAG_NAMESPACE_COLORS.clear()
        TAG_NAMESPACE_COLORS.update(
            {
                "topic": "#66d9ef",
                "status": "#a6e22e",
                "project": "#fd971f",
                "method": "#ae81ff",
                "priority": "#f92672",
            }
        )
        theme_ns = THEME_TAG_NAMESPACE_COLORS.get(self._config.theme_name)
        if theme_ns:
            TAG_NAMESPACE_COLORS.update(theme_ns)

    def _schedule_abstract_load(self, paper: Paper) -> None:
        """Schedule an abstract load with concurrency limits."""
        if paper.arxiv_id in self._abstract_loading or paper.arxiv_id in self._abstract_pending_ids:
            return
        if len(self._abstract_loading) < MAX_ABSTRACT_LOADS:
            self._abstract_loading.add(paper.arxiv_id)
            self._track_task(self._load_abstract_async(paper))
            return
        self._abstract_queue.append(paper)
        self._abstract_pending_ids.add(paper.arxiv_id)

    def _drain_abstract_queue(self) -> None:
        """Start queued abstract loads while capacity is available."""
        while self._abstract_queue and len(self._abstract_loading) < MAX_ABSTRACT_LOADS:
            paper = self._abstract_queue.popleft()
            self._abstract_pending_ids.discard(paper.arxiv_id)
            if paper.arxiv_id in self._abstract_loading:
                continue
            if paper.arxiv_id in self._abstract_cache:
                continue
            self._abstract_loading.add(paper.arxiv_id)
            self._track_task(self._load_abstract_async(paper))

    def _get_abstract_text(self, paper: Paper, allow_async: bool) -> str | None:
        """Return cached abstract text, scheduling async load if needed."""
        cached = self._abstract_cache.get(paper.arxiv_id)
        if cached is not None:
            return cached
        if paper.abstract is not None:
            self._abstract_cache[paper.arxiv_id] = paper.abstract
            return paper.abstract
        if not paper.abstract_raw:
            self._abstract_cache[paper.arxiv_id] = ""
            paper.abstract = ""
            return ""
        if not allow_async:
            cleaned = clean_latex(paper.abstract_raw)
            self._abstract_cache[paper.arxiv_id] = cleaned
            paper.abstract = cleaned
            return cleaned
        self._schedule_abstract_load(paper)
        return None

    async def _load_abstract_async(self, paper: Paper) -> None:
        try:
            cleaned = await asyncio.to_thread(clean_latex, paper.abstract_raw)
            self._abstract_cache[paper.arxiv_id] = cleaned
            # Only update if not already set (idempotent to avoid race conditions)
            if paper.abstract is None:
                paper.abstract = cleaned
            self._update_abstract_display(paper.arxiv_id)
        except Exception:
            logger.warning("Abstract load failed for %s", paper.arxiv_id, exc_info=True)
        finally:
            self._abstract_loading.discard(paper.arxiv_id)
            self._drain_abstract_queue()

    def _tags_for(self, arxiv_id: str) -> list[str] | None:
        """Return tags for a paper, or None if none set."""
        meta = self._config.paper_metadata.get(arxiv_id)
        return meta.tags if meta and meta.tags else None

    def _detail_kwargs(self, arxiv_id: str) -> dict:
        """Build keyword arguments for details.update_paper()."""
        s2_data, s2_loading = self._s2_state_for(arxiv_id)
        return {
            "summary": self._paper_summaries.get(arxiv_id),
            "summary_loading": arxiv_id in self._summary_loading,
            "highlight_terms": self._highlight_terms.get("abstract"),
            "s2_data": s2_data,
            "s2_loading": s2_loading,
            "hf_data": self._hf_state_for(arxiv_id),
            "version_update": self._version_update_for(arxiv_id),
            "summary_mode": self._summary_mode_label.get(arxiv_id, ""),
            "tags": self._tags_for(arxiv_id),
            "relevance": self._relevance_scores.get(arxiv_id),
            "collapsed_sections": self._config.collapsed_sections,
        }

    def _update_abstract_display(self, arxiv_id: str) -> None:
        try:
            details = self._get_paper_details_widget()
            if details.paper and details.paper.arxiv_id == arxiv_id:
                abstract_text = self._abstract_cache.get(arxiv_id, "")
                details.update_paper(details.paper, abstract_text, **self._detail_kwargs(arxiv_id))
        except NoMatches:
            pass
        # Update list option if showing preview
        if self._show_abstract_preview:
            self._update_option_for_paper(arxiv_id)

    def _save_config_or_warn(self, context: str) -> bool:
        """Save config and notify the user on failure.

        Returns True on success, False on failure.
        """
        if not save_config(self._config):
            self.notify(f"Failed to save {context}.", severity="warning")
            return False
        return True

    def _save_session_state(self) -> None:
        """Save current session state to config.

        Handles the case where DOM widgets may already be destroyed during unmount.
        """
        # API mode is intentionally session-ephemeral; persist the underlying local state.
        snapshot = self._local_browse_snapshot if self._in_arxiv_api_mode else None

        # Get current date for history mode
        current_date = self._get_current_date()
        current_date_str = current_date.strftime(HISTORY_DATE_FORMAT) if current_date else None

        if snapshot is not None:
            self._config.session = SessionState(
                scroll_index=snapshot.list_index,
                current_filter=snapshot.search_query.strip(),
                sort_index=snapshot.sort_index,
                selected_ids=list(snapshot.selected_ids),
                current_date=current_date_str,
            )
            if not save_config(self._config):
                logger.warning("Failed to save session state to config file")
                self.notify(
                    "Failed to save session — changes may be lost",
                    title="Save Error",
                    severity="error",
                    timeout=8,
                )
            return

        try:
            list_view = self._get_paper_list_widget()
            search_input = self._get_search_input_widget()

            self._config.session = SessionState(
                scroll_index=list_view.highlighted if list_view.highlighted is not None else 0,
                current_filter=search_input.value.strip(),
                sort_index=self._sort_index,
                selected_ids=list(self.selected_ids),
                current_date=current_date_str,
            )
        except (NoMatches, ScreenStackError):
            # DOM already torn down during shutdown, save with defaults
            self._config.session = SessionState(
                scroll_index=0,
                current_filter="",
                sort_index=self._sort_index,
                selected_ids=list(self.selected_ids),
                current_date=current_date_str,
            )

        if not save_config(self._config):
            logger.warning("Failed to save session state to config file")
            self.notify(
                "Failed to save session — changes may be lost",
                title="Save Error",
                severity="error",
                timeout=8,
            )

    @on(OptionList.OptionSelected, "#paper-list")
    def on_paper_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle paper selection (Enter key)."""
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            paper = self.filtered_papers[idx]
            details = self._get_paper_details_widget()
            aid = paper.arxiv_id
            abstract_text = self._get_abstract_text(paper, allow_async=True)
            details.update_paper(paper, abstract_text, **self._detail_kwargs(aid))

    @on(OptionList.OptionHighlighted, "#paper-list")
    def on_paper_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Handle paper highlight (keyboard navigation) with debouncing."""
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            self._pending_detail_paper = self.filtered_papers[idx]
            self._pending_detail_started_at = (
                time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
            )
            # Atomic swap pattern (same as search debounce)
            old_timer = self._detail_timer
            self._detail_timer = None
            if old_timer is not None:
                old_timer.stop()
            self._detail_timer = self.set_timer(
                DETAIL_PANE_DEBOUNCE_DELAY,
                self._debounced_detail_update,
            )

    def _debounced_detail_update(self) -> None:
        """Apply detail pane update after debounce delay."""
        self._detail_timer = None
        paper = self._pending_detail_paper
        self._pending_detail_paper = None
        started_at = self._pending_detail_started_at
        self._pending_detail_started_at = None
        if paper is None:
            return
        current = self._get_current_paper()
        if current is None or current.arxiv_id != paper.arxiv_id:
            return
        try:
            details = self._get_paper_details_widget()
        except NoMatches:
            return  # Widget tree torn down during shutdown
        aid = current.arxiv_id
        abstract_text = self._get_abstract_text(current, allow_async=True)
        details.update_paper(current, abstract_text, **self._detail_kwargs(aid))
        if started_at is not None:
            logger.debug(
                "Selection->detail latency: %.2fms (paper=%s)",
                (time.perf_counter() - started_at) * 1000.0,
                aid,
            )

    def _cancel_pending_detail_update(self) -> None:
        """Cancel any pending debounced detail-pane update."""
        timer = self._detail_timer
        self._detail_timer = None
        if timer is not None:
            timer.stop()
        self._pending_detail_paper = None
        self._pending_detail_started_at = None

    def action_toggle_search(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions.action_toggle_search(self)

    def action_cancel_search(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions.action_cancel_search(self)

    def _capture_local_browse_snapshot(self) -> LocalBrowseSnapshot | None:
        """Capture local browsing state before entering API search mode."""
        try:
            list_view = self._get_paper_list_widget()
            search_input = self._get_search_input_widget()
        except NoMatches:
            return None

        return LocalBrowseSnapshot(
            all_papers=self.all_papers,
            papers_by_id=self._papers_by_id,
            selected_ids=set(self.selected_ids),
            sort_index=self._sort_index,
            search_query=search_input.value.strip(),
            pending_query=self._pending_query,
            watch_filter_active=self._watch_filter_active,
            active_bookmark_index=self._active_bookmark_index,
            list_index=list_view.highlighted if list_view.highlighted is not None else 0,
            sub_title=self.sub_title,
            highlight_terms={key: terms.copy() for key, terms in self._highlight_terms.items()},
            match_scores=dict(self._match_scores),
        )

    def _restore_local_browse_snapshot(self) -> None:
        """Restore the browsing state saved before API search mode."""
        snapshot = self._local_browse_snapshot
        if snapshot is None:
            return

        self.all_papers = snapshot.all_papers
        self._papers_by_id = snapshot.papers_by_id
        self.selected_ids = set(snapshot.selected_ids)
        self._sort_index = snapshot.sort_index
        self._pending_query = snapshot.pending_query
        self._watch_filter_active = snapshot.watch_filter_active
        self._active_bookmark_index = snapshot.active_bookmark_index
        self._highlight_terms = {
            key: terms.copy() for key, terms in snapshot.highlight_terms.items()
        }
        self._match_scores = dict(snapshot.match_scores)
        self.sub_title = snapshot.sub_title

        # Recompute watch matches for the restored local dataset
        self._compute_watched_papers()

        try:
            self._get_search_input_widget().value = snapshot.search_query
        except NoMatches:
            pass

        self._apply_filter(snapshot.search_query)

        try:
            option_list = self._get_paper_list_widget()
            if option_list.option_count > 0:
                max_index = max(0, option_list.option_count - 1)
                option_list.highlighted = min(max(0, snapshot.list_index), max_index)
                option_list.focus()
        except NoMatches:
            pass

        if self._config.bookmarks:
            self._track_task(self._update_bookmark_bar())

    def action_ctrl_e_dispatch(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_ctrl_e_dispatch(self)

    def action_toggle_s2(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_toggle_s2(self)

    async def action_fetch_s2(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions.action_fetch_s2(self)

    async def _fetch_s2_paper_async(self, arxiv_id: str) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions._fetch_s2_paper_async(self, arxiv_id)

    def _s2_state_for(self, arxiv_id: str) -> tuple[SemanticScholarPaper | None, bool]:
        """Return (s2_data, s2_loading) for a paper, respecting the active toggle."""
        if not self._s2_active:
            return None, False
        return self._s2_cache.get(arxiv_id), arxiv_id in self._s2_loading

    # ========================================================================
    # HuggingFace trending
    # ========================================================================

    async def action_toggle_hf(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions.action_toggle_hf(self)

    async def _fetch_hf_daily(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions._fetch_hf_daily(self)

    async def _fetch_hf_daily_async(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions._fetch_hf_daily_async(self)

    def _hf_state_for(self, arxiv_id: str) -> HuggingFacePaper | None:
        """Return HF data for a paper if HF is active, else None."""
        if not self._hf_active:
            return None
        return self._hf_cache.get(arxiv_id)

    def _mark_badges_dirty(
        self,
        *badge_types: str,
        immediate: bool = False,
    ) -> None:
        """Schedule a coalesced badge refresh for the given types.

        Use immediate=True for toggle-off cases where UX needs instant feedback.
        """
        self._badges_dirty.update(badge_types)
        if immediate:
            old = self._badge_timer
            self._badge_timer = None
            if old is not None:
                old.stop()
            self._flush_badge_refresh()
            return
        # Atomic swap timer pattern (same as search/detail debounce)
        old = self._badge_timer
        self._badge_timer = None
        if old is not None:
            old.stop()
        self._badge_timer = self.set_timer(BADGE_COALESCE_DELAY, self._flush_badge_refresh)

    def _badge_refresh_indices(self, dirty: set[str]) -> list[int]:
        """Return visible list indices requiring badge redraw for dirty badge types."""
        if not self.filtered_papers:
            return []

        refresh_all = False
        dirty_ids: set[str] = set()

        if "s2" in dirty:
            if self._s2_active:
                dirty_ids.update(self._s2_cache.keys())
            else:
                refresh_all = True

        if "hf" in dirty:
            if self._hf_active:
                dirty_ids.update(self._hf_cache.keys())
            else:
                refresh_all = True

        if "version" in dirty:
            dirty_ids.update(self._version_updates.keys())

        # Unknown badge type, or explicit full redraw request: fall back to full repaint.
        if not dirty or refresh_all or any(kind not in {"s2", "hf", "version"} for kind in dirty):
            return list(range(len(self.filtered_papers)))

        if not dirty_ids:
            return list(range(len(self.filtered_papers)))

        visible_index_by_id = {
            paper.arxiv_id: idx for idx, paper in enumerate(self.filtered_papers)
        }
        return sorted(
            visible_index_by_id[paper_id]
            for paper_id in dirty_ids
            if paper_id in visible_index_by_id
        )

    def _flush_badge_refresh(self) -> None:
        """Coalesced badge refresh for only affected visible papers."""
        self._badge_timer = None
        dirty = self._badges_dirty.copy()
        self._badges_dirty.clear()
        if not dirty:
            return
        indices = self._badge_refresh_indices(dirty)
        for i in indices:
            self._update_option_at_index(i)
        logger.debug(
            "Badge refresh: dirty=%s updated=%d/%d",
            sorted(dirty),
            len(indices),
            len(self.filtered_papers),
        )

    # ========================================================================
    # Version tracking
    # ========================================================================

    VERSION_CHECK_BATCH_SIZE = 40  # IDs per API request (URL length safe)

    async def action_check_versions(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions.action_check_versions(self)

    async def _check_versions_async(self, arxiv_ids: set[str]) -> None:
        """Background task: check starred papers for newer arXiv versions."""
        try:
            client = self._http_client
            if client is None:
                return

            # Batch IDs into groups
            id_list = sorted(arxiv_ids)
            version_map: dict[str, int] = {}
            total_batches = max(1, -(-len(id_list) // self.VERSION_CHECK_BATCH_SIZE))

            for i in range(0, len(id_list), self.VERSION_CHECK_BATCH_SIZE):
                batch_num = i // self.VERSION_CHECK_BATCH_SIZE + 1
                self._version_progress = (batch_num, total_batches)
                self._update_footer()
                batch = id_list[i : i + self.VERSION_CHECK_BATCH_SIZE]
                await self._apply_arxiv_rate_limit()

                try:
                    response = await client.get(
                        ARXIV_API_URL,
                        params={
                            "id_list": ",".join(batch),
                            "max_results": len(batch) + 10,
                        },
                        headers={"User-Agent": "arxiv-subscription-viewer/1.0"},
                        timeout=ARXIV_API_TIMEOUT,
                    )
                    response.raise_for_status()
                    batch_map = parse_arxiv_version_map(response.text)
                    version_map.update(batch_map)
                except (httpx.HTTPError, ValueError, OSError):
                    logger.warning(
                        "Version check batch failed (IDs %d-%d)",
                        i,
                        i + len(batch),
                        exc_info=True,
                    )

            # Compare with stored versions
            updates_found = apply_version_updates(
                version_map,
                self._config.paper_metadata,
                self._version_updates,
            )

            # Persist updated metadata
            self._save_config_or_warn("version tracking data")

            # Refresh UI
            self._mark_badges_dirty("version")
            self._get_ui_refresh_coordinator().refresh_detail_pane()

            if updates_found > 0:
                self.notify(
                    f"{updates_found} paper(s) have new versions",
                    title="Versions",
                )
            else:
                self.notify("All starred papers are up to date", title="Versions")
        except Exception:
            logger.warning("Version check failed", exc_info=True)
            self.notify(
                build_actionable_error(
                    "check paper versions",
                    why="an API or network error occurred",
                    next_step="retry with V after a short delay",
                ),
                title="Versions",
                severity="error",
            )
        finally:
            self._version_checking = False
            self._version_progress = None
            self._update_status_bar()

    def _version_update_for(self, arxiv_id: str) -> tuple[int, int] | None:
        """Return version update tuple if paper has an update, else None."""
        return self._version_updates.get(arxiv_id)

    def _refresh_detail_pane(self) -> None:
        """Re-render the detail pane for the currently highlighted paper."""
        paper = self._get_current_paper()
        if not paper:
            return
        try:
            details = self._get_paper_details_widget()
        except NoMatches:
            return
        aid = paper.arxiv_id
        abstract_text = self._get_abstract_text(paper, allow_async=False)
        details.update_paper(paper, abstract_text, **self._detail_kwargs(aid))

    def _refresh_current_list_item(self) -> None:
        """Update the current list item's option display."""
        idx = self._get_current_index()
        if idx is not None:
            self._update_option_at_index(idx)

    def action_exit_arxiv_search_mode(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions.action_exit_arxiv_search_mode(self)

    def action_arxiv_search(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions.action_arxiv_search(self)

    def _format_arxiv_search_label(self, request: ArxivSearchRequest) -> str:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions._format_arxiv_search_label(self, request)

    async def _apply_arxiv_rate_limit(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return await _actions._apply_arxiv_rate_limit(self)

    async def _fetch_arxiv_api_page(
        self,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
    ) -> list[Paper]:
        from arxiv_browser.actions import search_api_actions as _actions

        return await _actions._fetch_arxiv_api_page(self, request, start, max_results)

    def _apply_arxiv_search_results(
        self,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
        papers: list[Paper],
    ) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions._apply_arxiv_search_results(self, request, start, max_results, papers)

    async def _run_arxiv_search(self, request: ArxivSearchRequest, start: int) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return await _actions._run_arxiv_search(self, request, start)

    async def _change_arxiv_page(self, direction: int) -> None:
        """Move to the previous or next arXiv API results page."""
        state = self._arxiv_search_state
        if not self._in_arxiv_api_mode or state is None:
            return
        if self._arxiv_api_fetch_inflight:
            self.notify("Search already in progress", title="arXiv Search")
            return

        if direction < 0 and state.start <= 0:
            self.notify("Already at first API page", title="arXiv Search")
            return

        target_start = max(0, state.start + (direction * state.max_results))
        await self._run_arxiv_search(state.request, start=target_start)

    def action_cursor_down(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_cursor_down(self)

    def action_cursor_up(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_cursor_up(self)

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        self._apply_filter(event.value)
        # Hide search after submission
        self._get_search_container_widget().remove_class("visible")
        # Focus the list
        self._get_paper_list_widget().focus()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Handle search input change with debouncing.

        Uses atomic swap pattern to avoid race conditions with timer callbacks.
        """
        self._pending_query = event.value
        # Atomic swap pattern: capture and clear before stopping
        old_timer = self._search_timer
        self._search_timer = None
        if old_timer is not None:
            old_timer.stop()
        # Set new timer for debounced filter
        self._search_timer = self.set_timer(
            SEARCH_DEBOUNCE_DELAY,
            self._debounced_filter,
        )

    def _debounced_filter(self) -> None:
        """Apply filter after debounce delay."""
        self._search_timer = None
        self._apply_filter(self._pending_query)

    @on(DateNavigator.JumpToDate)
    def on_date_jump(self, event: DateNavigator.JumpToDate) -> None:
        """Handle click on a date in the navigator."""
        if 0 <= event.index < len(self._history_files):
            self._set_history_index(event.index)

    @on(DateNavigator.NavigateDate)
    def on_date_navigate(self, event: DateNavigator.NavigateDate) -> None:
        """Handle click on prev/next arrows in the date navigator."""
        if event.direction > 0:
            self.action_prev_date()
        else:
            self.action_next_date()

    def _get_active_query(self) -> str:
        """Get the active search query, preferring the current input value."""
        try:
            return self._get_search_input_widget().value.strip()
        except NoMatches:
            return self._pending_query.strip()

    def _format_header_text(self, query: str = "") -> str:
        """Format the header text with paper count, date info, and filter indicator."""
        # Paper count: filtered/total when searching, total otherwise
        if query:
            count = f"{len(self.filtered_papers)}/{len(self.all_papers)}"
        else:
            count = f"{len(self.all_papers)} total"

        # Context suffix: API mode info or history date
        suffix = ""
        if self._in_arxiv_api_mode and self._arxiv_search_state is not None:
            state = self._arxiv_search_state
            page = (state.start // state.max_results) + 1
            mode_query = self._format_arxiv_search_label(state.request)
            mode_query = truncate_text(mode_query, 28)
            suffix = (
                f" · [{THEME_COLORS['orange']}]API[/]"
                f" [dim]({escape_rich_text(mode_query)} · page {page})[/]"
            )
        elif self._is_history_mode():
            current_date = self._get_current_date()
            if current_date:
                pos = self._current_date_index + 1
                total = len(self._history_files)
                suffix = f" · [{THEME_COLORS['accent']}]{current_date.strftime(HISTORY_DATE_FORMAT)}[/] [dim]({pos}/{total})[/]"

        # Selection badge
        if self.selected_ids:
            n = len(self.selected_ids)
            suffix += f" · [{THEME_COLORS['green']}]{n} selected[/]"

        return f" [bold]Papers[/] ({count}){suffix}"

    def _matches_advanced_query(self, paper: Paper, rpn: list[QueryToken]) -> bool:
        metadata = self._config.paper_metadata.get(paper.arxiv_id)
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return matches_advanced_query(paper, rpn, metadata, abstract_text)

    def _match_query_term(self, paper: Paper, token: QueryToken) -> bool:
        metadata = self._config.paper_metadata.get(paper.arxiv_id)
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return match_query_term(paper, token, metadata, abstract_text)

    def _fuzzy_search(self, query: str, papers: list[Paper] | None = None) -> list[Paper]:
        """Perform fuzzy search on title and authors.

        Populates self._match_scores with relevance scores.
        """
        query_lower = query.lower()
        scored_papers = []
        search_space = papers if papers is not None else self.all_papers

        for paper in search_space:
            # Combine title and authors for matching
            text = f"{paper.title} {paper.authors}"
            score = fuzz.WRatio(query_lower, text.lower())
            if score >= FUZZY_SCORE_CUTOFF:
                scored_papers.append((paper, score))

        # Sort by score descending
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        top_papers = scored_papers[:FUZZY_LIMIT]

        # Store scores for display (optional enhancement)
        self._match_scores = {p.arxiv_id: s for p, s in top_papers}

        return [p for p, _ in top_papers]

    def _apply_filter(self, query: str) -> None:
        """Filter papers by various criteria.

        Supported filters:
        - cat:<category>  - Filter by category (e.g., cat:cs.AI)
        - tag:<tag>       - Filter by tag (e.g., tag:to-read)
        - unread          - Show only unread papers
        - starred         - Show only starred papers
        - <text>          - Fuzzy search on title/author
        """
        perf_start = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
        query = query.strip()
        # Keep status/empty-state context synchronized with the applied filter.
        self._pending_query = query

        # Clear match scores by default (only fuzzy search populates them)
        self._match_scores.clear()
        _HIGHLIGHT_PATTERN_CACHE.clear()

        self.filtered_papers, self._highlight_terms = execute_query_filter(
            query,
            self.all_papers,
            fuzzy_search=self._fuzzy_search,
            advanced_match=self._matches_advanced_query,
        )

        # Apply watch filter if active (intersects with other filters)
        self.filtered_papers = apply_watch_filter(
            self.filtered_papers, self._watched_paper_ids, self._watch_filter_active
        )

        # Apply current sort order and refresh UI
        self._sort_papers()
        self._get_ui_refresh_coordinator().apply_filter_refresh(query)

        logger.debug(
            "Filter applied: query=%r, matched=%d/%d papers",
            query,
            len(self.filtered_papers),
            len(self.all_papers),
        )
        if perf_start is not None:
            logger.debug(
                "Search->list refresh latency: %.2fms (query=%r, matched=%d)",
                (time.perf_counter() - perf_start) * 1000.0,
                query,
                len(self.filtered_papers),
            )

    def _update_filter_pills(self, query: str) -> None:
        """Update the filter pill bar with current active filters."""
        if self._in_arxiv_api_mode:
            try:
                self._get_filter_pill_bar_widget().remove_class("visible")
            except NoMatches:
                pass
            return
        tokens = get_query_tokens(query)
        try:
            pill_bar = self._get_filter_pill_bar_widget()
            self._track_task(pill_bar.update_pills(tokens, self._watch_filter_active))
        except NoMatches:
            pass

    @on(FilterPillBar.RemoveFilter)
    def on_remove_filter(self, event: FilterPillBar.RemoveFilter) -> None:
        """Handle removal of a filter pill by reconstructing the query."""
        search_input = self._get_search_input_widget()
        search_input.value = remove_query_token(search_input.value, event.token_index)

    @on(FilterPillBar.RemoveWatchFilter)
    def on_remove_watch_filter(self, event: FilterPillBar.RemoveWatchFilter) -> None:
        """Handle removal of the watch filter pill."""
        self._watch_filter_active = False
        self._apply_filter(self._pending_query)

    def action_toggle_select(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_toggle_select(self)

    def action_select_all(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_select_all(self)

    def action_clear_selection(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_clear_selection(self)

    def _sort_papers(self) -> None:
        """Sort filtered_papers according to current sort order."""
        sort_key = SORT_OPTIONS[self._sort_index]
        self.filtered_papers = sort_papers(
            self.filtered_papers,
            sort_key,
            s2_cache=self._s2_cache,
            hf_cache=self._hf_cache,
            relevance_cache=self._relevance_scores,
        )

    def _refresh_list_view(self) -> None:
        """Refresh the list view with current filtered papers.

        Uses OptionList for virtual rendering — only visible lines are drawn.
        """
        self._cancel_pending_detail_update()
        option_list = self._get_paper_list_widget()
        option_list.clear_options()

        if self.filtered_papers:
            options = [
                Option(self._render_option(paper), id=paper.arxiv_id)
                for paper in self.filtered_papers
            ]
            option_list.add_options(options)
            option_list.highlighted = 0
        else:
            empty_msg = build_list_empty_message(
                query=self._get_active_query(),
                in_arxiv_api_mode=self._in_arxiv_api_mode,
                watch_filter_active=self._watch_filter_active,
                history_mode=self._is_history_mode(),
            )
            option_list.add_option(Option(empty_msg, disabled=True))
            try:
                details = self._get_paper_details_widget()
                details.update_paper(None)
            except NoMatches:
                pass

    def _render_option(self, paper: Paper) -> str:
        """Render a single paper as Rich markup for OptionList."""
        aid = paper.arxiv_id
        return render_paper_option(
            paper,
            selected=aid in self.selected_ids,
            metadata=self._config.paper_metadata.get(aid),
            watched=aid in self._watched_paper_ids,
            show_preview=self._show_abstract_preview,
            abstract_text=self._get_abstract_text(paper, allow_async=self._show_abstract_preview),
            highlight_terms=self._highlight_terms,
            s2_data=self._s2_cache.get(aid) if self._s2_active else None,
            hf_data=self._hf_cache.get(aid) if self._hf_active else None,
            version_update=self._version_updates.get(aid),
            relevance_score=self._relevance_scores.get(aid),
        )

    def _update_option_at_index(self, index: int) -> None:
        """Re-render a single option at the given index."""
        if index < 0 or index >= len(self.filtered_papers):
            return
        paper = self.filtered_papers[index]
        markup = self._render_option(paper)
        try:
            option_list = self._get_paper_list_widget()
            option_list.replace_option_prompt_at_index(index, markup)
        except (NoMatches, OptionDoesNotExist):
            pass

    def action_cycle_sort(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_cycle_sort(self)

    # ========================================================================
    # Phase 2: Read/Star Status and Notes/Tags
    # ========================================================================

    def _get_or_create_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Get or create metadata for a paper."""
        if arxiv_id not in self._config.paper_metadata:
            self._config.paper_metadata[arxiv_id] = PaperMetadata(arxiv_id=arxiv_id)
        return self._config.paper_metadata[arxiv_id]

    def _get_current_paper(self) -> Paper | None:
        """Get the currently highlighted paper."""
        try:
            option_list = self._get_paper_list_widget()
        except NoMatches:
            return None
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            return self.filtered_papers[idx]
        return None

    def _get_current_index(self) -> int | None:
        """Get the index of the currently highlighted paper."""
        try:
            option_list = self._get_paper_list_widget()
        except NoMatches:
            return None
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            return idx
        return None

    def _apply_to_selected(
        self,
        fn: Callable[[str], None],
        target_ids: set[str] | None = None,
    ) -> None:
        """Apply fn(arxiv_id) to all selected papers, refreshing visible list items.

        Uses target_ids if provided, otherwise self.selected_ids.
        """
        ids = target_ids if target_ids is not None else self.selected_ids
        visible_ids: set[str] = set()
        for i, paper in enumerate(self.filtered_papers):
            if paper.arxiv_id in ids:
                fn(paper.arxiv_id)
                self._update_option_at_index(i)
                visible_ids.add(paper.arxiv_id)
        for aid in ids - visible_ids:
            fn(aid)

    def _bulk_toggle_bool(
        self,
        attr: str,
        true_label: str,
        false_label: str,
        title: str,
    ) -> None:
        """Toggle a boolean metadata attribute for all selected papers.

        If any selected paper has the attribute False, sets all to True;
        otherwise sets all to False.
        """
        target = any(
            not getattr(
                self._config.paper_metadata.get(aid, PaperMetadata(arxiv_id=aid)),
                attr,
            )
            for aid in self.selected_ids
        )
        self._apply_to_selected(
            lambda aid: setattr(self._get_or_create_metadata(aid), attr, target)
        )
        status = true_label if target else false_label
        self.notify(f"{len(self.selected_ids)} papers {status}", title=title)

    def action_toggle_read(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_toggle_read(self)

    def action_toggle_star(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_toggle_star(self)

    def action_edit_notes(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_edit_notes(self)

    def action_edit_tags(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_edit_tags(self)

    def _collect_all_tags(self) -> list[str]:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._collect_all_tags(self)

    def _bulk_edit_tags(self) -> None:
        """Open tags editor for bulk-tagging all selected papers.

        Shows tags common to ALL selected papers as current tags.
        New tags are added to all; removed tags are removed from all.
        """
        n = len(self.selected_ids)
        # Find tags common to ALL selected papers (empty tags → empty intersection)
        tag_sets = [
            set(self._config.paper_metadata.get(aid, PaperMetadata(arxiv_id=aid)).tags)
            for aid in self.selected_ids
        ]
        common_tags = sorted(set.intersection(*tag_sets)) if tag_sets else []
        all_tags = self._collect_all_tags()
        target_ids = set(self.selected_ids)  # snapshot

        def on_bulk_tags_saved(tags: list[str] | None) -> None:
            if tags is None:
                return
            new_tag_set = set(tags)
            old_common = set(common_tags)
            added = new_tag_set - old_common
            removed = old_common - new_tag_set

            self._apply_to_selected(
                lambda aid: self._apply_tag_diff(aid, added, removed),
                target_ids=target_ids,
            )

            parts = []
            if added:
                parts.append(f"Added {', '.join(sorted(added))}")
            if removed:
                parts.append(f"Removed {', '.join(sorted(removed))}")
            msg = " / ".join(parts) if parts else "Tags unchanged"
            self.notify(f"{msg} on {len(target_ids)} papers", title="Bulk Tags")

        self.push_screen(
            TagsModal(f"bulk:{n}", common_tags, all_tags=all_tags),
            on_bulk_tags_saved,
        )

    def _apply_tag_diff(self, arxiv_id: str, added: set[str], removed: set[str]) -> None:
        """Apply tag additions and removals to a single paper's metadata."""
        meta = self._get_or_create_metadata(arxiv_id)
        tag_set = set(meta.tags)
        tag_set |= added
        tag_set -= removed
        meta.tags = sorted(tag_set)

    # ========================================================================
    # Phase 3: Watch List
    # ========================================================================

    def _compute_watched_papers(self) -> None:
        """Pre-compute which papers match watch list patterns.

        This runs once at startup and when watch list is modified,
        enabling O(1) lookup during display.
        """
        self._watched_paper_ids.clear()

        if not self._config.watch_list:
            return

        for paper in self.all_papers:
            for entry in self._config.watch_list:
                if paper_matches_watch_entry(paper, entry):
                    self._watched_paper_ids.add(paper.arxiv_id)
                    break  # Paper already matched, no need to check more entries

    def _notify_watch_list_matches(self) -> None:
        """Show a notification if any papers match the watch list."""
        if not self._watched_paper_ids:
            return
        n = len(self._watched_paper_ids)
        self.notify(
            f"{n} paper{'s' if n != 1 else ''} match your watch list",
            title="Watch List",
        )

    def _show_daily_digest(self) -> None:
        """Show a brief digest notification summarizing the day's papers."""
        if not self.all_papers:
            return
        digest = build_daily_digest(
            self.all_papers,
            watched_ids=self._watched_paper_ids,
            metadata=self._config.paper_metadata,
        )
        self.notify(digest, title="Daily Digest", timeout=8)

    def is_paper_watched(self, arxiv_id: str) -> bool:
        """Check if a paper is on the watch list. O(1) lookup."""
        return arxiv_id in self._watched_paper_ids

    def action_toggle_watch_filter(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_toggle_watch_filter(self)

    def action_manage_watch_list(self) -> None:
        from arxiv_browser.actions import library_actions as _actions

        return _actions.action_manage_watch_list(self)

    # ========================================================================
    # Phase 4: Bookmarked Search Tabs
    # ========================================================================

    async def _update_bookmark_bar(self) -> None:
        """Update the bookmark tab bar display."""
        bookmark_bar = self._get_bookmark_bar_widget()
        await bookmark_bar.update_bookmarks(self._config.bookmarks, self._active_bookmark_index)

    async def action_goto_bookmark(self, index: int) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return await _actions.action_goto_bookmark(self, index)

    async def action_add_bookmark(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return await _actions.action_add_bookmark(self)

    async def action_remove_bookmark(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return await _actions.action_remove_bookmark(self)

    # ========================================================================
    # Phase 5: Abstract Preview
    # ========================================================================

    def action_toggle_preview(self) -> None:
        """Toggle abstract preview in list items."""
        self._show_abstract_preview = not self._show_abstract_preview
        self._config.show_abstract_preview = self._show_abstract_preview

        status = "on" if self._show_abstract_preview else "off"
        self.notify(f"Abstract preview {status}", title="Preview")

        # Refresh list to show/hide previews
        self._refresh_list_view()
        self._update_status_bar()

    # ========================================================================
    # Phase 7: Vim-style Marks
    # ========================================================================

    def action_start_mark(self) -> None:
        """Start mark-set mode. Next letter key will set a mark."""
        self._pending_mark_action = "set"
        self.notify("Press a-z to set mark", title="Mark")

    def action_start_goto_mark(self) -> None:
        """Start goto-mark mode. Next letter key will jump to that mark."""
        self._pending_mark_action = "goto"
        self.notify("Press a-z to jump to mark", title="Mark")

    def on_key(self, event: Key) -> None:
        """Handle key events for vim-style mark capture."""
        if self._pending_mark_action is None:
            return

        key = event.key
        # Only accept single lowercase letters
        if len(key) == 1 and key.isalpha() and key.islower():
            if self._pending_mark_action == "set":
                self._set_mark(key)
            elif self._pending_mark_action == "goto":
                self._goto_mark(key)
            event.prevent_default()
            event.stop()

        # Cancel mark mode on any other key
        self._pending_mark_action = None

    def _set_mark(self, letter: str) -> None:
        """Set a mark at the current paper."""
        paper = self._get_current_paper()
        if not paper:
            self.notify("No paper selected", title="Mark", severity="warning")
            return

        self._config.marks[letter] = paper.arxiv_id
        self.notify(f"Mark '{letter}' set on {paper.arxiv_id}", title="Mark")

    def _goto_mark(self, letter: str) -> None:
        """Jump to a marked paper."""
        if letter not in self._config.marks:
            self.notify(f"Mark '{letter}' not set", title="Mark", severity="warning")
            return

        arxiv_id = self._config.marks[letter]
        paper = self._get_paper_by_id(arxiv_id)
        if not paper:
            self.notify(f"Paper {arxiv_id} not found", title="Mark", severity="warning")
            return

        # Find and scroll to the paper in the current list
        option_list = self._get_paper_list_widget()
        for i, p in enumerate(self.filtered_papers):
            if p.arxiv_id == arxiv_id:
                option_list.highlighted = i
                self.notify(f"Jumped to mark '{letter}'", title="Mark")
                return

        # Paper not in current filtered list
        self.notify(
            "Paper not in current view (try clearing filter)",
            title="Mark",
            severity="warning",
        )

    # ========================================================================
    # Phase 8: Export Features
    # ========================================================================

    def action_copy_bibtex(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_copy_bibtex(self)

    def action_export_bibtex_file(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_export_bibtex_file(self)

    def _format_paper_as_markdown(self, paper: Paper) -> str:
        """Format a paper as Markdown."""
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return format_paper_as_markdown(paper, abstract_text)

    def action_export_markdown(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_export_markdown(self)

    def action_export_menu(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_export_menu(self)

    def _do_export(self, fmt: str, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._do_export(self, fmt, papers)

    def _get_export_dir(self) -> Path:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._get_export_dir(self)

    def _export_to_file(self, content: str, extension: str, format_name: str) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._export_to_file(self, content, extension, format_name)

    def _export_clipboard_ris(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._export_clipboard_ris(self, papers)

    def _export_clipboard_csv(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._export_clipboard_csv(self, papers)

    def _export_clipboard_mdtable(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._export_clipboard_mdtable(self, papers)

    def _export_file_ris(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._export_file_ris(self, papers)

    def _export_file_csv(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._export_file_csv(self, papers)

    def action_export_metadata(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_export_metadata(self)

    def action_import_metadata(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_import_metadata(self)

    def _get_target_papers(self) -> list[Paper]:
        """Get papers to export (selected or current)."""
        details = self._get_paper_details_widget()
        return resolve_target_papers(
            filtered_papers=self.filtered_papers,
            selected_ids=self.selected_ids,
            papers_by_id=self._papers_by_id,
            current_paper=details.paper,
        )

    # ========================================================================
    # Phase 9: Paper Similarity
    # ========================================================================

    def action_show_similar(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_show_similar(self)

    def _show_recommendations(self, paper: Paper, source: str | None) -> None:
        """Dispatcher for local or S2 recommendations."""
        if not source:  # User cancelled the source modal
            return
        if source == "s2":
            self._track_task(self._show_s2_recommendations(paper))
        else:
            self._show_local_recommendations(paper)

    def _show_local_recommendations(self, paper: Paper) -> None:
        """Show TF-IDF + metadata local recommendations."""
        corpus_key = build_similarity_corpus_key(self.all_papers)
        tfidf_index = getattr(self, "_tfidf_index", None)
        tfidf_corpus_key = getattr(self, "_tfidf_corpus_key", None)
        if tfidf_index is None or tfidf_corpus_key != corpus_key:
            self._pending_similarity_paper_id = paper.arxiv_id
            build_task = getattr(self, "_tfidf_build_task", None)
            if build_task is not None and not build_task.done():
                self.notify("Similarity indexing in progress...", title="Similar")
                return
            self.notify("Indexing papers for similarity...", title="Similar")
            self._tfidf_build_task = self._track_task(self._build_tfidf_index_async(corpus_key))
            return

        similar_papers = find_similar_papers(
            paper,
            self.all_papers,
            metadata=self._config.paper_metadata,
            abstract_lookup=lambda _paper: "",
            tfidf_index=tfidf_index,
        )
        if not similar_papers:
            self.notify(
                build_actionable_warning(
                    "No similar papers were found",
                    next_step="try another paper, or broaden your search with /",
                ),
                title="Similar",
                severity="warning",
            )
            return
        self.push_screen(
            RecommendationsScreen(paper, similar_papers),
            self._on_recommendation_selected,
        )

    @staticmethod
    def _build_tfidf_index_for_similarity(papers: list[Paper]) -> TfidfIndex:
        """Build a TF-IDF index using cleaned abstract text."""

        abstract_cache: dict[str, str] = {}

        def _text_for(paper: Paper) -> str:
            abstract = paper.abstract
            if abstract is None:
                abstract = abstract_cache.get(paper.arxiv_id)
                if abstract is None:
                    abstract = clean_latex(paper.abstract_raw) if paper.abstract_raw else ""
                    abstract_cache[paper.arxiv_id] = abstract
            return f"{paper.title} {abstract}"

        return TfidfIndex.build(papers, text_fn=_text_for)

    async def _build_tfidf_index_async(self, corpus_key: str) -> None:
        """Build the TF-IDF index off the UI thread and publish it when fresh."""
        papers_snapshot = list(self.all_papers)
        try:
            index = await asyncio.to_thread(self._build_tfidf_index_for_similarity, papers_snapshot)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Failed to build similarity index", exc_info=True)
            self.notify(
                build_actionable_error(
                    "build the similarity index",
                    why="an indexing error occurred",
                    next_step="retry with R after changing paper or filter scope",
                ),
                title="Similar",
                severity="error",
            )
            return
        finally:
            self._tfidf_build_task = None

        if corpus_key != build_similarity_corpus_key(self.all_papers):
            logger.debug("Discarded stale similarity index for corpus key %s", corpus_key)
            return

        self._tfidf_index = index
        self._tfidf_corpus_key = corpus_key

        pending_id = self._pending_similarity_paper_id
        self._pending_similarity_paper_id = None
        if pending_id is None:
            self.notify("Similarity index ready", title="Similar")
            return

        current_paper = self._get_current_paper()
        if current_paper is None or current_paper.arxiv_id != pending_id:
            self.notify("Similarity index ready", title="Similar")
            return
        self._show_local_recommendations(current_paper)

    async def _show_s2_recommendations(self, paper: Paper) -> None:
        """Fetch S2 recommendations and show them in the modal."""
        try:
            self.notify("Fetching S2 recommendations...", title="S2")
            recs = await self._fetch_s2_recommendations_async(paper.arxiv_id)
            if not recs:
                self.notify(
                    build_actionable_warning(
                        "No Semantic Scholar recommendations were found",
                        next_step="press R and choose local recommendations, or retry later",
                    ),
                    title="S2",
                    severity="warning",
                )
                return
            similar = self._s2_recs_to_paper_tuples(recs)
            self.push_screen(
                RecommendationsScreen(paper, similar),
                self._on_recommendation_selected,
            )
        except Exception:
            logger.warning(
                "Failed to show S2 recommendations for %s",
                paper.arxiv_id,
                exc_info=True,
            )
            self.notify(
                build_actionable_error(
                    "fetch Semantic Scholar recommendations",
                    why="an API or network error occurred",
                    next_step="retry with R, or switch to local recommendations",
                ),
                title="S2",
                severity="error",
            )

    def _on_recommendation_selected(self, arxiv_id: str | None) -> None:
        """Handle selection from the recommendations modal."""
        if not arxiv_id:
            return
        option_list = self._get_paper_list_widget()
        for i, p in enumerate(self.filtered_papers):
            if p.arxiv_id == arxiv_id:
                option_list.highlighted = i
                return
        self.notify(
            build_actionable_warning(
                "That paper is not in the current filtered view",
                next_step="clear or adjust the filter with /, then try again",
            ),
            title="Similar",
            severity="warning",
        )

    async def _fetch_s2_recommendations_async(self, arxiv_id: str) -> list[SemanticScholarPaper]:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions._fetch_s2_recommendations_async(self, arxiv_id)

    @staticmethod
    def _s2_recs_to_paper_tuples(
        recs: list[SemanticScholarPaper],
    ) -> list[tuple[Paper, float]]:
        """Convert S2 recommendations to (Paper, score) tuples for RecommendationsScreen."""
        max_cites = max((r.citation_count for r in recs), default=1) or 1
        results = []
        for r in recs:
            paper = Paper(
                arxiv_id=r.arxiv_id or r.s2_paper_id,
                date="",
                title=r.title or "Unknown Title",
                authors="",
                categories="",
                comments=None,
                abstract=r.abstract or r.tldr or None,
                url=r.url or (f"https://arxiv.org/abs/{r.arxiv_id}" if r.arxiv_id else ""),
                source="s2",
            )
            score = r.citation_count / max_cites
            results.append((paper, score))
        return results

    # ========================================================================
    # Citation Graph
    # ========================================================================

    def action_citation_graph(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_citation_graph(self)

    async def _show_citation_graph(self, paper_id: str, title: str) -> None:
        """Fetch citation graph data and push the CitationGraphScreen."""
        try:
            refs, cites = await self._fetch_citation_graph(paper_id)
            if not refs and not cites:
                self.notify(
                    build_actionable_warning(
                        "No citation graph data was found",
                        next_step="press G again later, or press Ctrl+e to toggle S2",
                    ),
                    title="Citations",
                    severity="warning",
                )
                return
            local_ids = frozenset(self._papers_by_id.keys())
            self.push_screen(
                CitationGraphScreen(
                    root_title=title,
                    root_paper_id=paper_id,
                    references=refs,
                    citations=cites,
                    fetch_callback=self._fetch_citation_graph,
                    local_arxiv_ids=local_ids,
                ),
                self._on_citation_graph_selected,
            )
        except Exception:
            logger.warning("Failed to show citation graph for %s", paper_id, exc_info=True)
            self.notify(
                build_actionable_error(
                    "load the citation graph",
                    why="an API or network error occurred",
                    next_step="retry with G after a moment",
                ),
                title="Citations",
                severity="error",
            )

    async def _fetch_citation_graph(
        self, paper_id: str
    ) -> tuple[list[CitationEntry], list[CitationEntry]]:
        from arxiv_browser.actions import ui_actions as _actions

        return await _actions._fetch_citation_graph(self, paper_id)

    def _on_citation_graph_selected(self, arxiv_id: str | None) -> None:
        """Handle selection from the citation graph modal (jump to local paper)."""
        self._on_recommendation_selected(arxiv_id)

    def action_cycle_theme(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_cycle_theme(self)

    def action_toggle_sections(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_toggle_sections(self)

    def _build_help_sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Build help sections from the runtime key binding table."""
        return build_help_sections(self.BINDINGS)

    def action_show_help(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_show_help(self)

    def action_command_palette(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_command_palette(self)

    # ========================================================================
    # Paper Collections
    # ========================================================================

    def action_collections(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_collections(self)

    def action_add_to_collection(self) -> None:
        from arxiv_browser.actions import ui_actions as _actions

        return _actions.action_add_to_collection(self)

    # ========================================================================
    # LLM Summary Generation
    # ========================================================================

    @staticmethod
    def _trust_hash(command_template: str) -> str:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._trust_hash(command_template)

    def _remember_trusted_hash(
        self,
        command_template: str,
        trusted_hashes: list[str],
        title: str,
    ) -> bool:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._remember_trusted_hash(self, command_template, trusted_hashes, title)

    def _is_llm_command_trusted(self, command_template: str) -> bool:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._is_llm_command_trusted(self, command_template)

    def _is_pdf_viewer_trusted(self, viewer_cmd: str) -> bool:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._is_pdf_viewer_trusted(self, viewer_cmd)

    def _ensure_command_trusted(
        self,
        *,
        command_template: str,
        title: str,
        prompt_heading: str,
        trust_button_label: str,
        cancel_message: str,
        trusted_hashes: list[str],
        on_trusted: Callable[[], None],
    ) -> bool:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._ensure_command_trusted(
            self,
            command_template=command_template,
            title=title,
            prompt_heading=prompt_heading,
            trust_button_label=trust_button_label,
            cancel_message=cancel_message,
            trusted_hashes=trusted_hashes,
            on_trusted=on_trusted,
        )

    def _ensure_llm_command_trusted(
        self,
        command_template: str,
        on_trusted: Callable[[], None],
    ) -> bool:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._ensure_llm_command_trusted(self, command_template, on_trusted)

    def _ensure_pdf_viewer_trusted(
        self,
        viewer_cmd: str,
        on_trusted: Callable[[], None],
    ) -> bool:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._ensure_pdf_viewer_trusted(self, viewer_cmd, on_trusted)

    def _require_llm_command(self) -> str | None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._require_llm_command(self)

    def action_generate_summary(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions.action_generate_summary(self)

    def _start_summary_flow(self, command_template: str) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._start_summary_flow(self, command_template)

    def _on_summary_mode_selected(
        self, mode: str | None, paper: Paper, command_template: str
    ) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._on_summary_mode_selected(self, mode, paper, command_template)

    async def _generate_summary_async(
        self,
        paper: Paper,
        prompt_template: str,
        cmd_hash: str,
        mode_label: str = "",
        use_full_paper_content: bool = True,
    ) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return await _actions._generate_summary_async(
            self, paper, prompt_template, cmd_hash, mode_label, use_full_paper_content
        )

    # ========================================================================
    # Chat with Paper
    # ========================================================================

    def action_chat_with_paper(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions.action_chat_with_paper(self)

    def _start_chat_with_paper(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._start_chat_with_paper(self)

    async def _open_chat_screen(self, paper: Paper, provider: CLIProvider) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return await _actions._open_chat_screen(self, paper, provider)

    # ========================================================================
    # Relevance Scoring
    # ========================================================================

    def action_score_relevance(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions.action_score_relevance(self)

    def _start_score_relevance_flow(self, command_template: str) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._start_score_relevance_flow(self, command_template)

    def _on_interests_saved_then_score(self, interests: str | None, command_template: str) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._on_interests_saved_then_score(self, interests, command_template)

    def _start_relevance_scoring(self, command_template: str, interests: str) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._start_relevance_scoring(self, command_template, interests)

    def action_edit_interests(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions.action_edit_interests(self)

    def _on_interests_edited(self, interests: str | None) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._on_interests_edited(self, interests)

    async def _score_relevance_batch_async(
        self,
        papers: list[Paper],
        command_template: str,
        interests: str,
    ) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return await _actions._score_relevance_batch_async(
            self, papers, command_template, interests
        )

    def _update_option_for_paper(self, arxiv_id: str) -> None:
        """Update the list option display for a specific paper by arXiv ID."""
        for i, paper in enumerate(self.filtered_papers):
            if paper.arxiv_id == arxiv_id:
                self._update_option_at_index(i)
                break

    def _update_relevance_badge(self, arxiv_id: str) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._update_relevance_badge(self, arxiv_id)

    # ========================================================================
    # Auto-Tagging
    # ========================================================================

    def action_auto_tag(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions.action_auto_tag(self)

    def _start_auto_tag_flow(self) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return _actions._start_auto_tag_flow(self)

    async def _auto_tag_single_async(
        self,
        paper: Paper,
        taxonomy: list[str],
        current_tags: list[str],
    ) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return await _actions._auto_tag_single_async(self, paper, taxonomy, current_tags)

    async def _auto_tag_batch_async(
        self,
        papers: list[Paper],
        taxonomy: list[str],
    ) -> None:
        from arxiv_browser.actions import llm_actions as _actions

        return await _actions._auto_tag_batch_async(self, papers, taxonomy)

    async def _call_auto_tag_llm(self, paper: Paper, taxonomy: list[str]) -> list[str] | None:
        """Call the LLM to get tag suggestions for a paper. Returns tags or None on failure."""
        if self._llm_provider is None:
            logger.warning("LLM provider unexpectedly None in _call_auto_tag_llm")
            return None

        try:
            tags = await self._get_services().llm.suggest_tags_once(
                paper=paper,
                taxonomy=taxonomy,
                provider=self._llm_provider,
                timeout_seconds=AUTO_TAG_TIMEOUT,
            )
        except _LLMExecutionError as exc:
            logger.warning("Auto-tag failed for %s: %s", paper.arxiv_id, str(exc)[:200])
            return None
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Auto-tag runtime failure for %s: %s", paper.arxiv_id, exc, exc_info=True
            )
            return None
        except Exception as exc:
            logger.warning(
                "Unexpected auto-tag failure for %s: %s", paper.arxiv_id, exc, exc_info=True
            )
            return None

        if tags is None:
            logger.warning("Failed to parse auto-tag response for %s", paper.arxiv_id)
            self.notify("Could not parse LLM response", title="Auto-Tag", severity="warning")
            return None

        return tags

    def _on_auto_tag_accepted(self, tags: list[str] | None, arxiv_id: str) -> None:
        """Callback when user accepts auto-tag suggestions."""
        if tags is None:
            return
        meta = self._get_or_create_metadata(arxiv_id)
        meta.tags = tags
        self._save_config_or_warn("tag changes")

        self._update_option_for_paper(arxiv_id)
        self._refresh_detail_pane()
        self.notify(f"Tags updated: {', '.join(tags)}", title="Auto-Tag")

    # ========================================================================
    # History Mode: Date Navigation
    # ========================================================================

    def _is_history_mode(self) -> bool:
        """Check if we're in history mode (multiple date files available)."""
        return len(self._history_files) > 0

    def _get_current_date(self) -> date | None:
        """Get the currently loaded date, or None if not in history mode."""
        if not self._is_history_mode():
            return None
        return self._history_files[self._current_date_index][0]

    def _load_current_date(self) -> bool:
        """Load papers from the current date file and refresh UI."""
        if not self._is_history_mode():
            return False

        current_date, path = self._history_files[self._current_date_index]
        try:
            self.all_papers = parse_arxiv_file(path)
        except OSError as e:
            self.notify(
                f"Failed to load {path.name}: {e}",
                title="Load Error",
                severity="error",
            )
            return False
        self._papers_by_id = {p.arxiv_id: p for p in self.all_papers}
        self.filtered_papers = self.all_papers.copy()

        self._abstract_cache.clear()
        self._abstract_loading.clear()
        self._abstract_queue.clear()
        self._abstract_pending_ids.clear()
        try:
            self._get_paper_details_widget().clear_cache()
        except NoMatches:
            pass
        self._paper_summaries.clear()
        self._summary_loading.clear()
        self._summary_mode_label.clear()
        self._summary_command_hash.clear()
        self._s2_cache.clear()
        self._s2_loading.clear()
        self._hf_cache.clear()
        self._hf_loading = False
        self._version_updates.clear()
        self._relevance_scores.clear()
        self._relevance_scoring_active = False
        self._tfidf_index = None
        self._tfidf_corpus_key = None
        self._pending_similarity_paper_id = None
        if self._tfidf_build_task is not None and not self._tfidf_build_task.done():
            self._tfidf_build_task.cancel()
        self._tfidf_build_task = None

        # Clear selection when switching dates
        self.selected_ids.clear()

        # Recompute watched papers for new paper set
        self._compute_watched_papers()

        self._notify_watch_list_matches()
        self._show_daily_digest()

        # Apply current filter and sort
        query = self._get_search_input_widget().value.strip()
        self._apply_filter(query)

        # Re-fetch HF data if active (since HF data is date-specific)
        if self._hf_active:
            self._track_task(self._fetch_hf_daily())

        # Update subtitle
        self.sub_title = (
            f"{len(self.all_papers)} papers · {current_date.strftime(HISTORY_DATE_FORMAT)}"
        )

        # Update date navigator
        self.call_after_refresh(self._refresh_date_navigator)
        return True

    def _set_history_index(self, target_index: int) -> bool:
        """Set and load a history index, rolling back on load failure."""
        if not (0 <= target_index < len(self._history_files)):
            return False
        old_index = self._current_date_index
        if target_index == old_index:
            return True
        self._current_date_index = target_index
        if self._load_current_date():
            return True
        self._current_date_index = old_index
        return False

    def action_prev_date(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions.action_prev_date(self)

    def action_next_date(self) -> None:
        from arxiv_browser.actions import search_api_actions as _actions

        return _actions.action_next_date(self)

    def _update_list_header(self, query: str) -> None:
        """Update the list header text for the current query/context."""
        try:
            self._get_list_header_widget().update(self._format_header_text(query))
        except NoMatches:
            pass

    def _update_header(self) -> None:
        """Update header with selection count and sort info."""
        query = self._get_active_query()
        self._update_list_header(query)
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Update the status bar with semantic, context-aware information."""
        try:
            status = self._get_status_bar_widget()
        except NoMatches:
            return

        total = len(self.all_papers)
        filtered = len(self.filtered_papers)
        query = self._get_active_query()
        api_page: int | None = None
        if self._in_arxiv_api_mode and self._arxiv_search_state is not None:
            api_page = (self._arxiv_search_state.start // self._arxiv_search_state.max_results) + 1
        hf_match_count = count_hf_matches(self._hf_cache, self._papers_by_id)
        size = getattr(self, "size", None)

        status.update(
            _widget_chrome.build_status_bar_text(
                total=total,
                filtered=filtered,
                query=query,
                watch_filter_active=self._watch_filter_active,
                selected_count=len(self.selected_ids),
                sort_label=SORT_OPTIONS[self._sort_index],
                in_arxiv_api_mode=self._in_arxiv_api_mode,
                api_page=api_page,
                arxiv_api_loading=self._arxiv_api_loading,
                show_abstract_preview=self._show_abstract_preview,
                s2_active=self._s2_active,
                s2_loading=bool(self._s2_loading),
                s2_count=len(self._s2_cache),
                hf_active=self._hf_active,
                hf_loading=self._hf_loading,
                hf_match_count=hf_match_count,
                version_checking=self._version_checking,
                version_update_count=len(self._version_updates),
                max_width=getattr(size, "width", None),
            )
        )
        self._update_footer()

    def _get_footer_bindings(self) -> list[tuple[str, str]]:
        """Return context-sensitive binding hints for the footer."""
        # Progress operations take highest priority (visual progress bar)
        if self._scoring_progress is not None:
            current, total = self._scoring_progress
            bar = render_progress_bar(current, total)
            return [("", f"Scoring {bar} {current}/{total}"), ("?", "help")]
        if self._relevance_scoring_active:
            return [("", "Scoring papers…"), ("?", "help")]
        if self._version_progress is not None:
            batch, total = self._version_progress
            bar = render_progress_bar(batch, total)
            return [("", f"Versions {bar} {batch}/{total}"), ("?", "help")]
        if self._version_checking:
            return [("", "Checking versions…"), ("?", "help")]
        if self._is_download_batch_active():
            completed = len(self._download_results)
            total = self._download_total
            bar = render_progress_bar(completed, total)
            return [("", f"Downloading {bar} {completed}/{total}"), ("?", "help")]
        if self._auto_tag_progress is not None:
            current, total = self._auto_tag_progress
            bar = render_progress_bar(current, total)
            return [("", f"Auto-tagging {bar} {current}/{total}"), ("?", "help")]
        if self._auto_tag_active:
            return [("", "Auto-tagging…"), ("?", "help")]

        # Search mode — search container visible
        try:
            container = self._get_search_container_widget()
            if container.has_class("visible"):
                return _widget_chrome.build_search_footer_bindings()
        except NoMatches:
            pass

        # arXiv API search mode
        if self._in_arxiv_api_mode:
            return _widget_chrome.build_api_footer_bindings()

        # Selection mode — papers selected
        if self.selected_ids:
            return _widget_chrome.build_selection_footer_bindings(len(self.selected_ids))

        # Default browsing — dynamically show contextual hints
        has_starred = any(m.starred for m in self._config.paper_metadata.values())
        llm_configured = bool(_resolve_llm_command(self._config))
        return _widget_chrome.build_browse_footer_bindings(
            s2_active=self._s2_active,
            has_starred=has_starred,
            llm_configured=llm_configured,
            has_history_navigation=bool(self._history_files and len(self._history_files) > 1),
        )

    def _get_footer_mode_badge(self) -> str:
        """Return a Rich-markup mode badge string for the current state."""
        search_visible = False
        try:
            container = self._get_search_container_widget()
            if container.has_class("visible"):
                search_visible = True
        except NoMatches:
            pass
        return _widget_chrome.build_footer_mode_badge(
            relevance_scoring_active=self._relevance_scoring_active,
            version_checking=self._version_checking,
            search_visible=search_visible,
            in_arxiv_api_mode=self._in_arxiv_api_mode,
            selected_count=len(self.selected_ids),
        )

    def _update_footer(self) -> None:
        """Update the context-sensitive footer based on current state."""
        try:
            footer = self._get_footer_widget()
        except (NoMatches, AttributeError):
            # AttributeError: app not fully composed (e.g. __new__ mock tests)
            return
        footer.render_bindings(self._get_footer_bindings(), self._get_footer_mode_badge())

    def _get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Look up a paper by its arXiv ID. O(1) dict lookup."""
        return self._papers_by_id.get(arxiv_id)

    async def _download_pdf_async(
        self, paper: Paper, client: httpx.AsyncClient | None = None
    ) -> bool:
        """Download a single PDF asynchronously.

        Args:
            paper: The paper to download.
            client: Optional shared HTTP client. Creates a temporary one if None.

        Returns:
            True if download succeeded, False otherwise.
        """
        success = await self._get_services().download.download_pdf(
            paper=paper,
            config=self._config,
            client=client,
            timeout_seconds=PDF_DOWNLOAD_TIMEOUT,
        )
        if success:
            logger.debug("Downloaded PDF for %s", paper.arxiv_id)
        else:
            logger.warning("Download failed for %s", paper.arxiv_id)
        return success

    def _start_downloads(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._start_downloads(self)

    def _is_download_batch_active(self) -> bool:
        """Return True when a download batch is active or pending."""
        return bool(self._download_queue or self._downloading or self._download_total)

    async def _process_single_download(self, paper: Paper) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return await _actions._process_single_download(self, paper)

    def _update_download_progress(self, completed: int, total: int) -> None:
        """Update status bar and footer with download progress."""
        try:
            status_bar = self._get_status_bar_widget()
            status_bar.update(f"Downloading: {completed}/{total} complete")
        except NoMatches:
            pass
        self._update_footer()

    def _finish_download_batch(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._finish_download_batch(self)

    def _safe_browser_open(self, url: str) -> bool:
        """Open a URL in the browser with error handling. Returns True on success."""
        try:
            webbrowser.open(url)
            return True
        except (webbrowser.Error, OSError) as e:
            logger.warning("Failed to open browser for %s: %s", url, e)
            self.notify(
                build_actionable_error(
                    "open your browser",
                    why="the system browser command failed",
                    next_step="copy the URL with c or export it with E",
                ),
                title="Browser",
                severity="error",
                timeout=8,
            )
            return False

    def action_open_url(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_open_url(self)

    def _do_open_urls(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._do_open_urls(self, papers)

    def action_open_pdf(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_open_pdf(self)

    def _do_open_pdfs(self, papers: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._do_open_pdfs(self, papers)

    def _open_with_viewer(self, viewer_cmd: str, url_or_path: str) -> bool:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._open_with_viewer(self, viewer_cmd, url_or_path)

    def action_download_pdf(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_download_pdf(self)

    def _do_start_downloads(self, to_download: list[Paper]) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._do_start_downloads(self, to_download)

    def _format_paper_for_clipboard(self, paper: Paper) -> str:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._format_paper_for_clipboard(self, paper)

    def _copy_to_clipboard(self, text: str) -> bool:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions._copy_to_clipboard(self, text)

    def action_copy_selected(self) -> None:
        from arxiv_browser.actions import external_io_actions as _actions

        return _actions.action_copy_selected(self)


def main() -> int:
    """Main entry point wrapper for CLI/bootstrap logic."""
    return _cli_main(
        load_config_fn=load_config,
        discover_history_files_fn=discover_history_files,
        resolve_papers_fn=_resolve_papers,
        configure_logging_fn=_configure_logging,
        configure_color_mode_fn=_configure_color_mode,
        validate_interactive_tty_fn=_validate_interactive_tty,
        app_factory=ArxivBrowser,
    )


if __name__ == "__main__":
    sys.exit(main())
