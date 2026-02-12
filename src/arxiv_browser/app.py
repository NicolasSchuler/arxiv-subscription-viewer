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
    Ctrl+e  - Toggle S2 / Exit arXiv API mode
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

import argparse
import asyncio
import hashlib
import logging
import logging.handlers
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import webbrowser
from collections import deque
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_config_dir
from rapidfuzz import fuzz
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option, OptionDoesNotExist

from arxiv_browser.config import *  # noqa: F403
from arxiv_browser.config import (  # noqa: F401
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
from arxiv_browser.huggingface import (
    HuggingFacePaper,
    fetch_hf_daily_papers,
    get_hf_db_path,
    load_hf_daily_cache,
    save_hf_daily_cache,
)
from arxiv_browser.llm import *  # noqa: F403
from arxiv_browser.llm import (  # noqa: F401
    AUTO_TAG_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT,
    DEFAULT_LLM_PROMPT,
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
    build_highlight_terms,
    escape_rich_text,
    format_categories,
    format_summary_as_rich,
    highlight_text,
    insert_implicit_and,
    is_advanced_query,
    match_query_term,
    matches_advanced_query,
    paper_matches_watch_entry,
    pill_label_for_token,
    reconstruct_query,
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
    fetch_s2_paper,
    fetch_s2_recommendations,
    fetch_s2_references,
    get_s2_db_path,
    has_s2_citation_graph_cache,
    load_s2_citation_graph,
    load_s2_paper,
    load_s2_recommendations,
    save_s2_citation_graph,
    save_s2_paper,
    save_s2_recommendations,
)
from arxiv_browser.similarity import *  # noqa: F403
from arxiv_browser.similarity import (  # noqa: F401
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
    "CollectionViewModal",
    "CollectionsModal",
    "CommandPaletteModal",
    "FilterPillBar",
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

# Context-sensitive footer keybinding hints (max ~8 per context)
FOOTER_CONTEXTS: dict[str, list[tuple[str, str]]] = {
    "default": [
        ("/", "search"),
        ("o", "open"),
        ("s", "sort"),
        ("r", "read"),
        ("x", "star"),
        ("n", "notes"),
        ("t", "tags"),
        ("?", "help"),
    ],
    "selection": [
        ("o", "open"),
        ("r", "read"),
        ("x", "star"),
        ("t", "tags"),
        ("E", "export"),
        ("d", "download"),
        ("u", "clear"),
        ("?", "help"),
    ],
    "search": [
        ("type to filter", ""),
        ("Esc", "close"),
        ("↑↓", "navigate"),
        ("?", "help"),
    ],
    "api": [
        ("[/]", "pages"),
        ("Esc", "exit API"),
        ("o", "open"),
        ("s", "sort"),
        ("?", "help"),
    ],
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
    ("Toggle S2", "Enable/disable Semantic Scholar enrichment", "Ctrl+e", "toggle_s2"),
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
RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal
PREVIEW_ABSTRACT_MAX_LEN = 150  # Max abstract preview length in list items
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
LLM_COMMAND_TIMEOUT = 120  # Seconds to wait for LLM CLI response
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
# Maximum number of cached detail pane renderings (FIFO eviction)
DETAIL_CACHE_MAX = 100


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


def _render_title_line(
    paper: Paper,
    selected: bool,
    metadata: PaperMetadata | None,
    watched: bool,
    ht: dict[str, list[str]],
) -> str:
    """Build the title line with selection/watch/star/read indicators."""
    prefix_parts: list[str] = []
    if selected:
        prefix_parts.append(f"[{THEME_COLORS['green']}]●[/]")
    if watched:
        prefix_parts.append(f"[{THEME_COLORS['orange']}]👁[/]")
    if metadata and metadata.starred:
        prefix_parts.append(f"[{THEME_COLORS['yellow']}]⭐[/]")
    if metadata and metadata.is_read:
        prefix_parts.append(f"[{THEME_COLORS['muted']}]✓[/]")
    prefix = " ".join(prefix_parts)

    title_text = highlight_text(paper.title, ht.get("title", []), THEME_COLORS["accent"])
    if metadata and metadata.is_read:
        title_text = f"[dim]{title_text}[/]"
    return f"{prefix} {title_text}" if prefix else title_text


def _render_meta_badges(
    paper: Paper,
    metadata: PaperMetadata | None,
    s2_data: "SemanticScholarPaper | None",
    hf_data: "HuggingFacePaper | None",
    version_update: tuple[int, int] | None,
    relevance_score: tuple[int, str] | None,
) -> str:
    """Build the meta line with arxiv_id, categories, and badges."""
    parts: list[str] = []
    if paper.source == "api":
        parts.append(f"[{THEME_COLORS['orange']}]API[/]")
    parts.extend([f"[dim]{paper.arxiv_id}[/]", format_categories(paper.categories)])
    if metadata and metadata.tags:
        tag_str = " ".join(
            f"[{get_tag_color(tag)}]#{escape_rich_text(tag)}[/]" for tag in metadata.tags
        )
        parts.append(tag_str)
    if s2_data is not None:
        parts.append(f"[{THEME_COLORS['green']}]C{s2_data.citation_count}[/]")
    if hf_data is not None:
        parts.append(f"[{THEME_COLORS['orange']}]\u2191{hf_data.upvotes}[/]")
    if version_update is not None:
        old_v, new_v = version_update
        parts.append(f"[{THEME_COLORS['pink']}]v{old_v}\u2192v{new_v}[/]")
    if relevance_score is not None:
        score, _ = relevance_score
        if score >= 8:
            color = THEME_COLORS["green"]
        elif score >= 5:
            color = THEME_COLORS["yellow"]
        else:
            color = THEME_COLORS["muted"]
        parts.append(f"[{color}]{score}/10[/]")
    return "  ".join(parts)


def _render_abstract_preview(abstract_text: str | None, ht: dict[str, list[str]]) -> str:
    """Build the abstract preview line for the paper list."""
    if abstract_text is None:
        return "[dim italic]Loading abstract...[/]"
    if not abstract_text:
        return "[dim italic]No abstract available[/]"
    if len(abstract_text) <= PREVIEW_ABSTRACT_MAX_LEN:
        highlighted = highlight_text(abstract_text, ht.get("abstract", []), THEME_COLORS["accent"])
        return f"[dim italic]{highlighted}[/]"
    truncated = abstract_text[:PREVIEW_ABSTRACT_MAX_LEN].rsplit(" ", 1)[0]
    highlighted = highlight_text(truncated, ht.get("abstract", []), THEME_COLORS["accent"])
    return f"[dim italic]{highlighted}...[/]"


def render_paper_option(
    paper: Paper,
    *,
    selected: bool = False,
    metadata: PaperMetadata | None = None,
    watched: bool = False,
    show_preview: bool = False,
    abstract_text: str | None = None,
    highlight_terms: dict[str, list[str]] | None = None,
    s2_data: "SemanticScholarPaper | None" = None,
    hf_data: "HuggingFacePaper | None" = None,
    version_update: tuple[int, int] | None = None,
    relevance_score: tuple[int, str] | None = None,
) -> str:
    """Render a paper as Rich markup for OptionList display."""
    ht = highlight_terms or {"title": [], "author": [], "abstract": []}

    lines = [
        _render_title_line(paper, selected, metadata, watched, ht),
        highlight_text(paper.authors, ht.get("author", []), THEME_COLORS["accent"]),
        _render_meta_badges(paper, metadata, s2_data, hf_data, version_update, relevance_score),
    ]

    if show_preview:
        lines.append(_render_abstract_preview(abstract_text, ht))

    return "\n".join(lines)


class PaperListItem(ListItem):
    """A list item displaying a paper title and URL."""

    def __init__(
        self,
        paper: Paper,
        selected: bool = False,
        metadata: PaperMetadata | None = None,
        watched: bool = False,
        show_preview: bool = False,
        abstract_text: str | None = None,
        highlight_terms: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.paper = paper
        self._selected = selected
        self._metadata = metadata
        self._watched = watched
        self._show_preview = show_preview
        self._abstract_text = abstract_text
        self._highlight_terms = highlight_terms or {
            "title": [],
            "author": [],
            "abstract": [],
        }
        self._s2_data: SemanticScholarPaper | None = None
        self._hf_data: HuggingFacePaper | None = None
        self._version_update: tuple[int, int] | None = None
        self._relevance_score: tuple[int, str] | None = None
        if selected:
            self.add_class("selected")

    @property
    def is_selected(self) -> bool:
        return self._selected

    @property
    def metadata(self) -> PaperMetadata | None:
        return self._metadata

    def set_metadata(self, metadata: PaperMetadata | None) -> None:
        """Update metadata and refresh display."""
        self._metadata = metadata
        self._update_display()

    def set_abstract_text(self, text: str | None) -> None:
        """Update abstract text for preview and refresh display."""
        self._abstract_text = text
        if self._show_preview:
            self._update_display()

    def update_s2_data(self, s2_data: SemanticScholarPaper | None) -> None:
        """Update Semantic Scholar data and refresh display."""
        self._s2_data = s2_data
        self._update_display()

    def update_hf_data(self, hf_data: HuggingFacePaper | None) -> None:
        """Update HuggingFace data and refresh display."""
        self._hf_data = hf_data
        self._update_display()

    def update_version_data(self, version_update: tuple[int, int] | None) -> None:
        """Update version tracking data and refresh display."""
        self._version_update = version_update
        self._update_display()

    def update_relevance_data(self, relevance: tuple[int, str] | None) -> None:
        """Update relevance score data and refresh display."""
        self._relevance_score = relevance
        self._update_display()

    def _get_title_text(self) -> str:
        """Get the formatted title text based on selection and metadata state."""
        prefix_parts = []

        # Selection indicator
        if self._selected:
            prefix_parts.append(f"[{THEME_COLORS['green']}]●[/]")

        # Watched indicator
        if self._watched:
            prefix_parts.append(f"[{THEME_COLORS['orange']}]👁[/]")

        # Starred indicator
        if self._metadata and self._metadata.starred:
            prefix_parts.append(f"[{THEME_COLORS['yellow']}]⭐[/]")

        # Read indicator
        if self._metadata and self._metadata.is_read:
            prefix_parts.append(f"[{THEME_COLORS['muted']}]✓[/]")

        prefix = " ".join(prefix_parts)
        title_text = highlight_text(
            self.paper.title,
            self._highlight_terms.get("title", []),
            THEME_COLORS["accent"],
        )
        # Dim title for read papers — unread titles stay bold/bright
        is_read = self._metadata and self._metadata.is_read
        if is_read:
            title_text = f"[dim]{title_text}[/]"
        if prefix:
            return f"{prefix} {title_text}"
        return title_text

    def _get_authors_text(self) -> str:
        """Get the formatted author text."""
        return highlight_text(
            self.paper.authors,
            self._highlight_terms.get("author", []),
            THEME_COLORS["accent"],
        )

    def _get_meta_text(self) -> str:
        """Get the formatted metadata text."""
        parts = []
        if self.paper.source == "api":
            parts.append(f"[{THEME_COLORS['orange']}]API[/]")
        parts.extend(
            [
                f"[dim]{self.paper.arxiv_id}[/]",
                format_categories(self.paper.categories),
            ]
        )

        # Show tags if present (namespace-colored)
        if self._metadata and self._metadata.tags:
            tag_str = " ".join(
                f"[{get_tag_color(tag)}]#{escape_rich_text(tag)}[/]" for tag in self._metadata.tags
            )
            parts.append(tag_str)

        # S2 citation badge
        if self._s2_data is not None:
            parts.append(f"[{THEME_COLORS['green']}]C{self._s2_data.citation_count}[/]")

        # HF trending badge
        if self._hf_data is not None:
            parts.append(f"[{THEME_COLORS['orange']}]\u2191{self._hf_data.upvotes}[/]")

        # Version update badge
        if self._version_update is not None:
            old_v, new_v = self._version_update
            parts.append(f"[{THEME_COLORS['pink']}]v{old_v}\u2192v{new_v}[/]")

        # Relevance score badge
        if self._relevance_score is not None:
            score, _ = self._relevance_score
            if score >= 8:
                color = THEME_COLORS["green"]
            elif score >= 5:
                color = THEME_COLORS["yellow"]
            else:
                color = THEME_COLORS["muted"]
            parts.append(f"[{color}]{score}/10[/]")

        return "  ".join(parts)

    def _get_preview_text(self) -> str:
        """Get truncated abstract preview text.

        Returns formatted Rich markup for the abstract preview.
        Handles empty abstracts and truncates at word boundaries.
        """
        abstract = self._abstract_text
        if abstract is None:
            return "[dim italic]Loading abstract...[/]"
        if not abstract:
            return "[dim italic]No abstract available[/]"
        if len(abstract) <= PREVIEW_ABSTRACT_MAX_LEN:
            highlighted = highlight_text(
                abstract,
                self._highlight_terms.get("abstract", []),
                THEME_COLORS["accent"],
            )
            return f"[dim italic]{highlighted}[/]"
        # Truncate at word boundary for cleaner display
        truncated = abstract[:PREVIEW_ABSTRACT_MAX_LEN].rsplit(" ", 1)[0]
        highlighted = highlight_text(
            truncated,
            self._highlight_terms.get("abstract", []),
            THEME_COLORS["accent"],
        )
        return f"[dim italic]{highlighted}...[/]"

    def _update_selection_class(self) -> None:
        """Update the CSS class based on selection state."""
        if self._selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")

    def toggle_selected(self) -> bool:
        """Toggle selection state and return new state."""
        self._selected = not self._selected
        self._update_selection_class()
        self._update_display()
        return self._selected

    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self._selected = selected
        self._update_selection_class()
        self._update_display()

    def _update_display(self) -> None:
        """Update the visual display based on selection state."""
        try:
            title_widget = self.query_one(".paper-title", Static)
            authors_widget = self.query_one(".paper-authors", Static)
            meta_widget = self.query_one(".paper-meta", Static)
            title_widget.update(self._get_title_text())
            authors_widget.update(self._get_authors_text())
            meta_widget.update(self._get_meta_text())
            if self._show_preview:
                preview_widget = self.query_one(".paper-preview", Static)
                preview_widget.update(self._get_preview_text())
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        yield Static(self._get_title_text(), classes="paper-title")
        yield Static(self._get_authors_text(), classes="paper-authors")
        yield Static(self._get_meta_text(), classes="paper-meta")
        if self._show_preview:
            yield Static(self._get_preview_text(), classes="paper-preview")


def _detail_cache_key(
    paper: Paper,
    abstract_text: str | None,
    abstract_loading: bool = False,
    summary: str | None = None,
    summary_loading: bool = False,
    highlight_terms: list[str] | None = None,
    s2_data: "SemanticScholarPaper | None" = None,
    s2_loading: bool = False,
    hf_data: "HuggingFacePaper | None" = None,
    version_update: tuple[int, int] | None = None,
    summary_mode: str = "",
    tags: list[str] | None = None,
    relevance: tuple[int, str] | None = None,
    collapsed_sections: list[str] | None = None,
) -> tuple:
    """Compute a hashable cache key for detail pane rendering."""
    summary_digest = (
        hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16] if summary is not None else None
    )
    abstract_digest = (
        hashlib.sha256(abstract_text.encode("utf-8")).hexdigest()[:16]
        if abstract_text is not None
        else None
    )
    s2_key = (
        (
            s2_data.citation_count,
            s2_data.influential_citation_count,
            s2_data.tldr,
            s2_data.fields_of_study,
            s2_data.year,
            s2_data.url,
            s2_data.title,
            s2_data.abstract,
        )
        if s2_data
        else None
    )
    hf_key = (
        (
            hf_data.upvotes,
            hf_data.num_comments,
            hf_data.ai_summary,
            hf_data.ai_keywords,
            hf_data.github_repo,
            hf_data.github_stars,
        )
        if hf_data
        else None
    )
    return (
        paper.arxiv_id,
        paper.title,
        paper.authors,
        paper.date,
        paper.categories,
        paper.comments,
        paper.url,
        abstract_digest,
        abstract_loading,
        summary_digest,
        summary_loading,
        tuple(highlight_terms) if highlight_terms else (),
        s2_key,
        s2_loading,
        hf_key,
        version_update,
        summary_mode,
        tuple(tags) if tags else (),
        relevance,
        tuple(collapsed_sections) if collapsed_sections else (),
        tuple(sorted(THEME_COLORS.items())),
    )


class PaperDetails(Static):
    """Widget to display full paper details."""

    def __init__(self) -> None:
        super().__init__()
        self._paper: Paper | None = None
        self._detail_cache: dict[tuple, str] = {}
        self._detail_cache_order: list[tuple] = []

    def update_paper(
        self,
        paper: Paper | None,
        abstract_text: str | None = None,
        summary: str | None = None,
        summary_loading: bool = False,
        highlight_terms: list[str] | None = None,
        s2_data: SemanticScholarPaper | None = None,
        s2_loading: bool = False,
        hf_data: HuggingFacePaper | None = None,
        version_update: tuple[int, int] | None = None,
        summary_mode: str = "",
        tags: list[str] | None = None,
        relevance: tuple[int, str] | None = None,
        collapsed_sections: list[str] | None = None,
    ) -> None:
        """Update the displayed paper details."""
        self._paper = paper
        if paper is None:
            self.update("[dim italic]Select a paper to view details[/]")
            return

        loading = abstract_text is None and paper.abstract is None
        if abstract_text is None:
            abstract_text = paper.abstract or ""

        # Check detail cache before rebuilding markup
        cache_key = _detail_cache_key(
            paper,
            abstract_text,
            abstract_loading=loading,
            summary=summary,
            summary_loading=summary_loading,
            highlight_terms=highlight_terms,
            s2_data=s2_data,
            s2_loading=s2_loading,
            hf_data=hf_data,
            version_update=version_update,
            summary_mode=summary_mode,
            tags=tags,
            relevance=relevance,
            collapsed_sections=collapsed_sections,
        )
        cached = self._detail_cache.get(cache_key)
        if cached is not None:
            self.update(cached)
            return

        collapsed = set(collapsed_sections) if collapsed_sections else set()

        sections = [
            self._render_title(paper),
            self._render_metadata(paper),
            self._render_abstract(abstract_text, loading, highlight_terms, "abstract" in collapsed),
            self._render_authors(paper, "authors" in collapsed),
            self._render_tags(tags, "tags" in collapsed),
            self._render_relevance(relevance, "relevance" in collapsed),
            self._render_summary(summary, summary_loading, summary_mode, "summary" in collapsed),
            self._render_s2(s2_data, s2_loading, "s2" in collapsed),
            self._render_hf(hf_data, "hf" in collapsed),
            self._render_version(paper, version_update, "version" in collapsed),
            self._render_url(paper),
        ]
        markup = "\n".join(s for s in sections if s)

        # Store in cache with FIFO eviction
        if len(self._detail_cache) >= DETAIL_CACHE_MAX:
            oldest = self._detail_cache_order.pop(0)
            self._detail_cache.pop(oldest, None)
        self._detail_cache[cache_key] = markup
        self._detail_cache_order.append(cache_key)

        self.update(markup)

    def _render_title(self, paper: Paper) -> str:
        safe_title = escape_rich_text(paper.title)
        return f"[bold {THEME_COLORS['text']}]{safe_title}[/]"

    def _render_metadata(self, paper: Paper) -> str:
        safe_date = escape_rich_text(paper.date)
        safe_comments = escape_rich_text(paper.comments or "")
        lines = [
            f"  [bold {THEME_COLORS['accent']}]arXiv:[/] [{THEME_COLORS['purple']}]{paper.arxiv_id}[/]",
            f"  [bold {THEME_COLORS['accent']}]Date:[/] {safe_date}",
            f"  [bold {THEME_COLORS['accent']}]Categories:[/] {format_categories(paper.categories)}",
        ]
        if paper.comments:
            lines.append(f"  [bold {THEME_COLORS['accent']}]Comments:[/] [dim]{safe_comments}[/]")
        return "\n".join(lines)

    def _render_abstract(
        self,
        abstract_text: str,
        loading: bool,
        highlight_terms: list[str] | None,
        is_collapsed: bool,
    ) -> str:
        if is_collapsed:
            return "[dim]▸ Abstract[/]"
        if highlight_terms:
            safe_abstract = highlight_text(abstract_text, highlight_terms, THEME_COLORS["accent"])
        else:
            safe_abstract = escape_rich_text(abstract_text)
        lines = [f"[bold {THEME_COLORS['orange']}]▾ Abstract[/]"]
        if loading:
            lines.append("  [dim italic]Loading abstract...[/]")
        elif abstract_text:
            lines.append(f"  [{THEME_COLORS['text']}]{safe_abstract}[/]")
        else:
            lines.append("  [dim italic]No abstract available[/]")
        return "\n".join(lines)

    def _render_authors(self, paper: Paper, is_collapsed: bool) -> str:
        if is_collapsed:
            return "[dim]▸ Authors[/]"
        safe_authors = escape_rich_text(paper.authors)
        return (
            f"[bold {THEME_COLORS['green']}]▾ Authors[/]\n"
            f"  [{THEME_COLORS['text']}]{safe_authors}[/]"
        )

    def _render_tags(self, tags: list[str] | None, is_collapsed: bool) -> str:
        if not tags:
            return ""
        if is_collapsed:
            return f"[dim]▸ Tags ({len(tags)})[/]"
        lines = [f"[bold {THEME_COLORS['accent']}]▾ Tags[/]"]
        namespaced: dict[str, list[str]] = {}
        unnamespaced: list[str] = []
        for tag in tags:
            ns, val = parse_tag_namespace(tag)
            if ns:
                namespaced.setdefault(ns, []).append(val)
            else:
                unnamespaced.append(val)
        for ns in sorted(namespaced):
            color = get_tag_color(f"{ns}:")
            safe_ns = escape_rich_text(ns)
            vals = ", ".join(escape_rich_text(v) for v in namespaced[ns])
            lines.append(f"  [{color}]{safe_ns}:[/] {vals}")
        if unnamespaced:
            color = get_tag_color("")
            safe_unnamespaced = ", ".join(escape_rich_text(v) for v in unnamespaced)
            lines.append(f"  [{color}]{safe_unnamespaced}[/]")
        return "\n".join(lines)

    def _render_relevance(self, relevance: tuple[int, str] | None, is_collapsed: bool) -> str:
        if relevance is None:
            return ""
        rel_score, rel_reason = relevance
        if is_collapsed:
            return f"[dim]▸ Relevance ({rel_score}/10)[/]"
        if rel_score >= 8:
            score_color = THEME_COLORS["green"]
        elif rel_score >= 5:
            score_color = THEME_COLORS["yellow"]
        else:
            score_color = THEME_COLORS["muted"]
        lines = [
            f"[bold {THEME_COLORS['accent']}]▾ Relevance[/]",
            f"  [bold {THEME_COLORS['accent']}]Score:[/] [{score_color}]{rel_score}/10[/]",
        ]
        if rel_reason:
            safe_reason = escape_rich_text(rel_reason)
            lines.append(f"  [{THEME_COLORS['text']}]{safe_reason}[/]")
        return "\n".join(lines)

    def _render_summary(
        self,
        summary: str | None,
        summary_loading: bool,
        summary_mode: str,
        is_collapsed: bool,
    ) -> str:
        summary_header = "AI Summary"
        if summary_mode:
            summary_header += f" ({summary_mode})"
        if not summary_loading and not summary:
            return ""
        if is_collapsed:
            hint = " (loaded)" if summary else ""
            return f"[dim]▸ {summary_header}{hint}[/]"
        if summary_loading:
            return (
                f"[bold {THEME_COLORS['purple']}]▾ 🤖 {summary_header}[/]\n"
                "  [dim italic]⏳ Generating summary...[/]"
            )
        if summary:
            rendered_summary = format_summary_as_rich(summary)
            return f"[bold {THEME_COLORS['purple']}]▾ 🤖 {summary_header}[/]\n{rendered_summary}"
        return ""

    def _render_s2(
        self,
        s2_data: SemanticScholarPaper | None,
        s2_loading: bool,
        is_collapsed: bool,
    ) -> str:
        if not s2_loading and not s2_data:
            return ""
        if is_collapsed:
            hint = ""
            if s2_data:
                hint = f" ({s2_data.citation_count} cites)"
            return f"[dim]▸ Semantic Scholar{hint}[/]"
        if s2_loading:
            return (
                f"[bold {THEME_COLORS['green']}]▾ Semantic Scholar[/]\n"
                "  [dim italic]Fetching data...[/]"
            )
        if s2_data:
            lines = [
                f"[bold {THEME_COLORS['green']}]▾ Semantic Scholar[/]",
                (
                    f"  [bold {THEME_COLORS['accent']}]Citations:[/] {s2_data.citation_count}"
                    f"  [bold {THEME_COLORS['accent']}]Influential:[/] {s2_data.influential_citation_count}"
                ),
            ]
            if s2_data.fields_of_study:
                fos = ", ".join(escape_rich_text(field) for field in s2_data.fields_of_study)
                lines.append(f"  [bold {THEME_COLORS['accent']}]Fields:[/] {fos}")
            if s2_data.tldr:
                safe_tldr = escape_rich_text(s2_data.tldr)
                lines.append(
                    f"  [bold {THEME_COLORS['accent']}]TLDR:[/] [{THEME_COLORS['text']}]{safe_tldr}[/]"
                )
            return "\n".join(lines)
        return ""

    def _render_hf(self, hf_data: HuggingFacePaper | None, is_collapsed: bool) -> str:
        if not hf_data:
            return ""
        if is_collapsed:
            return f"[dim]\u25b8 HuggingFace (\u2191{hf_data.upvotes})[/]"
        lines = [f"[bold {THEME_COLORS['orange']}]\u25be HuggingFace[/]"]
        hf_parts = [f"  [bold {THEME_COLORS['accent']}]Upvotes:[/] {hf_data.upvotes}"]
        if hf_data.num_comments > 0:
            hf_parts.append(f"  [bold {THEME_COLORS['accent']}]Comments:[/] {hf_data.num_comments}")
        lines.append("".join(hf_parts))
        if hf_data.github_repo:
            stars_str = f" ({hf_data.github_stars} stars)" if hf_data.github_stars else ""
            safe_repo = escape_rich_text(hf_data.github_repo)
            lines.append(f"  [bold {THEME_COLORS['accent']}]GitHub:[/] {safe_repo}{stars_str}")
        if hf_data.ai_keywords:
            kw = ", ".join(escape_rich_text(keyword) for keyword in hf_data.ai_keywords)
            lines.append(f"  [bold {THEME_COLORS['accent']}]Keywords:[/] {kw}")
        if hf_data.ai_summary:
            safe_summary = escape_rich_text(hf_data.ai_summary)
            lines.append(
                f"  [bold {THEME_COLORS['accent']}]AI Summary:[/] [{THEME_COLORS['text']}]{safe_summary}[/]"
            )
        return "\n".join(lines)

    def _render_version(
        self,
        paper: Paper,
        version_update: tuple[int, int] | None,
        is_collapsed: bool,
    ) -> str:
        if version_update is None:
            return ""
        old_v, new_v = version_update
        if is_collapsed:
            return f"[dim]\u25b8 Version Update (v{old_v}\u2192v{new_v})[/]"
        return (
            f"[bold {THEME_COLORS['pink']}]\u25be Version Update[/]\n"
            f"  [bold {THEME_COLORS['accent']}]Updated:[/] [{THEME_COLORS['pink']}]v{old_v} \u2192 v{new_v}[/]\n"
            f"  [bold {THEME_COLORS['accent']}]View diff:[/] [{THEME_COLORS['accent']}]https://arxivdiff.org/abs/{paper.arxiv_id}[/]"
        )

    def _render_url(self, paper: Paper) -> str:
        safe_url = escape_rich_text(paper.url)
        return f"[bold {THEME_COLORS['pink']}]URL[/]\n  [{THEME_COLORS['accent']}]{safe_url}[/]"

    def clear_cache(self) -> None:
        """Clear the rendered markup cache."""
        self._detail_cache.clear()
        self._detail_cache_order.clear()

    @property
    def paper(self) -> Paper | None:
        return self._paper


# ============================================================================
# Help Overlay
# ============================================================================


class HelpScreen(ModalScreen[None]):
    """Full-screen help overlay showing all keyboard shortcuts by category."""

    BINDINGS = [
        Binding("question_mark", "dismiss", "Close", show=False),
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close", show=False),
    ]

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
        overflow-y: auto;
    }

    #help-title {
        text-style: bold;
        color: $th-accent-alt;
        text-align: center;
        margin-bottom: 1;
    }

    .help-section {
        margin-bottom: 1;
    }

    .help-section-title {
        text-style: bold;
        margin-bottom: 0;
    }

    .help-keys {
        padding-left: 2;
        color: $th-text;
    }

    #help-footer {
        text-align: center;
        color: $th-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-dialog"):
            yield Label("Keyboard Shortcuts", id="help-title")

            # Navigation
            yield Label(f"[{THEME_COLORS['accent']}]Navigation[/]", classes="help-section-title")
            yield Static(
                f"  [{THEME_COLORS['green']}]j / k[/]        Navigate down / up\n"
                f"  [{THEME_COLORS['green']}]bracketleft / bracketright[/]  Previous / next date  [dim](history mode)[/]\n"
                f"  [{THEME_COLORS['green']}]1-9[/]          Jump to bookmark\n"
                f"  [{THEME_COLORS['green']}]m[/] [dim]then[/] [{THEME_COLORS['green']}]a-z[/]  Set mark\n"
                f"  [{THEME_COLORS['green']}]'[/] [dim]then[/] [{THEME_COLORS['green']}]a-z[/]  Jump to mark",
                classes="help-keys",
            )

            # Search
            yield Label(
                f"[{THEME_COLORS['accent']}]Search & Filter[/]",
                classes="help-section-title",
            )
            yield Static(
                f"  [{THEME_COLORS['green']}]/[/]            Toggle search\n"
                f"  [{THEME_COLORS['green']}]Escape[/]       Clear search\n"
                f"  [{THEME_COLORS['green']}]A[/]            Search all arXiv (API mode)\n"
                f"  [{THEME_COLORS['green']}]Ctrl+e[/]       Exit API mode\n"
                f"  [{THEME_COLORS['green']}]bracketleft / bracketright[/]  API page - / + [dim](API mode)[/]\n"
                f"  [{THEME_COLORS['green']}]w[/]            Toggle watch list filter\n"
                f"  [{THEME_COLORS['green']}]s[/]            Cycle sort order  [dim](title / date / id / cites / trending / relevance)[/]\n"
                f"  [{THEME_COLORS['green']}]Ctrl+b[/]       Save search as bookmark\n"
                f"  [dim]Filters:[/] cat: author: title: abstract: tag: unread starred",
                classes="help-keys",
            )

            # Selection
            yield Label(f"[{THEME_COLORS['accent']}]Selection[/]", classes="help-section-title")
            yield Static(
                f"  [{THEME_COLORS['green']}]Space[/]        Toggle selection\n"
                f"  [{THEME_COLORS['green']}]a[/]            Select all visible\n"
                f"  [{THEME_COLORS['green']}]u[/]            Clear selection",
                classes="help-keys",
            )

            # Paper actions
            yield Label(
                f"[{THEME_COLORS['accent']}]Paper Actions[/]",
                classes="help-section-title",
            )
            yield Static(
                f"  [{THEME_COLORS['green']}]o[/]            Open in browser\n"
                f"  [{THEME_COLORS['green']}]P[/]            Open as PDF\n"
                f"  [{THEME_COLORS['green']}]r[/]            Toggle read\n"
                f"  [{THEME_COLORS['green']}]x[/]            Toggle star\n"
                f"  [{THEME_COLORS['green']}]n[/]            Edit notes\n"
                f"  [{THEME_COLORS['green']}]t[/]            Edit tags  [dim](namespace:value supported)[/]\n"
                f"  [{THEME_COLORS['green']}]R[/]            Show similar papers\n"
                f"  [{THEME_COLORS['green']}]G[/]            Citation graph  [dim](S2-powered, drill-down)[/]\n"
                f"  [{THEME_COLORS['green']}]W[/]            Manage watch list\n"
                f"  [{THEME_COLORS['green']}]Ctrl+s[/]       Generate AI summary  [dim](mode selector)[/]\n"
                f"  [{THEME_COLORS['green']}]L[/]            Score papers by relevance  [dim](LLM-powered)[/]\n"
                f"  [{THEME_COLORS['green']}]Ctrl+l[/]       Edit research interests",
                classes="help-keys",
            )

            # Export
            yield Label(
                f"[{THEME_COLORS['accent']}]Export & Copy[/]",
                classes="help-section-title",
            )
            yield Static(
                f"  [{THEME_COLORS['green']}]c[/]            Copy to clipboard\n"
                f"  [{THEME_COLORS['green']}]E[/]            Export menu  [dim](BibTeX, Markdown, RIS, CSV + more)[/]\n"
                f"  [{THEME_COLORS['green']}]d[/]            Download PDF",
                classes="help-keys",
            )

            # View
            yield Label(f"[{THEME_COLORS['accent']}]View[/]", classes="help-section-title")
            yield Static(
                f"  [{THEME_COLORS['green']}]p[/]            Toggle abstract preview\n"
                f"  [{THEME_COLORS['green']}]?[/]            This help screen\n"
                f"  [{THEME_COLORS['green']}]q[/]            Quit",
                classes="help-keys",
            )

            yield Label("Press ? / Escape / q to close", id="help-footer")

    def action_dismiss(self) -> None:
        """Close the help screen."""
        self.dismiss(None)


# ============================================================================
# Modal Screens for Notes and Tags
# ============================================================================


class NotesModal(ModalScreen[str]):
    """Modal dialog for editing paper notes."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    NotesModal {
        align: center middle;
    }

    #notes-dialog {
        width: 60%;
        height: 60%;
        min-width: 50;
        min-height: 15;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #notes-title {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #notes-textarea {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #notes-textarea:focus {
        border-left: tall $th-accent;
    }

    #notes-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #notes-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, arxiv_id: str, current_notes: str = "") -> None:
        super().__init__()
        self._arxiv_id = arxiv_id
        self._current_notes = current_notes

    def compose(self) -> ComposeResult:
        with Vertical(id="notes-dialog"):
            yield Label(f"Notes for {self._arxiv_id}", id="notes-title")
            yield TextArea(self._current_notes, id="notes-textarea")
            with Horizontal(id="notes-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")

    def on_mount(self) -> None:
        self.query_one("#notes-textarea", TextArea).focus()

    def action_save(self) -> None:
        text = self.query_one("#notes-textarea", TextArea).text
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss(self._current_notes)

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


class TagsModal(ModalScreen[list[str]]):
    """Modal dialog for editing paper tags."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    TagsModal {
        align: center middle;
    }

    #tags-dialog {
        width: 50%;
        height: auto;
        min-width: 40;
        background: $th-background;
        border: tall $th-green;
        padding: 0 2;
    }

    #tags-title {
        text-style: bold;
        color: $th-green;
        margin-bottom: 1;
    }

    #tags-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #tags-input {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #tags-input:focus {
        border-left: tall $th-green;
    }

    #tags-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #tags-buttons Button {
        margin-left: 1;
    }

    #tags-suggestions {
        color: $th-muted;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        arxiv_id: str,
        current_tags: list[str] | None = None,
        all_tags: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._arxiv_id = arxiv_id
        self._current_tags = current_tags or []
        self._all_tags = all_tags or []

    def _build_suggestions_markup(self) -> str:
        """Build Rich markup for tag suggestions grouped by namespace."""
        if not self._all_tags:
            return ""
        # Group unique tags by namespace
        namespaced: dict[str, list[str]] = {}
        unnamespaced: list[str] = []
        for tag in sorted(set(self._all_tags)):
            ns, val = parse_tag_namespace(tag)
            if ns:
                namespaced.setdefault(ns, []).append(val)
            else:
                unnamespaced.append(val)
        parts = []
        for ns in sorted(namespaced):
            color = get_tag_color(f"{ns}:")
            vals = ", ".join(namespaced[ns])
            parts.append(f"[{color}]{ns}:[/] {vals}")
        if unnamespaced:
            parts.append(", ".join(unnamespaced))
        return " | ".join(parts)

    def compose(self) -> ComposeResult:
        with Vertical(id="tags-dialog"):
            yield Label(f"Tags for {self._arxiv_id}", id="tags-title")
            yield Label(
                "Use namespace:tag format (e.g., topic:ml, status:to-read)",
                id="tags-help",
            )
            suggestions = self._build_suggestions_markup()
            if suggestions:
                yield Label(suggestions, id="tags-suggestions")
            yield Input(
                value=", ".join(self._current_tags),
                placeholder="Enter tags...",
                id="tags-input",
            )
            with Horizontal(id="tags-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")

    def on_mount(self) -> None:
        self.query_one("#tags-input", Input).focus()

    def _parse_tags(self, text: str) -> list[str]:
        """Parse comma-separated tags, stripping whitespace."""
        return [tag.strip() for tag in text.split(",") if tag.strip()]

    def action_save(self) -> None:
        text = self.query_one("#tags-input", Input).value
        self.dismiss(self._parse_tags(text))

    def action_cancel(self) -> None:
        self.dismiss(self._current_tags)

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    @on(Input.Submitted, "#tags-input")
    def on_input_submitted(self) -> None:
        self.action_save()


class WatchListItem(ListItem):
    """List item for watch list entries."""

    def __init__(self, entry: WatchListEntry, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.entry = entry


class WatchListModal(ModalScreen[list[WatchListEntry] | None]):
    """Modal dialog for managing watch list entries."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    WatchListModal {
        align: center middle;
    }

    #watch-dialog {
        width: 80%;
        height: 70%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #watch-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #watch-body {
        height: 1fr;
    }

    #watch-list {
        width: 2fr;
        height: 1fr;
        background: $th-panel;
        border: none;
        margin-right: 2;
    }

    #watch-form {
        width: 1fr;
        height: 1fr;
    }

    #watch-form Label {
        color: $th-muted;
        margin-top: 1;
    }

    #watch-pattern,
    #watch-type {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #watch-pattern:focus,
    #watch-type:focus {
        border-left: tall $th-accent;
    }

    #watch-case {
        margin-top: 1;
    }

    #watch-actions {
        height: auto;
        margin-top: 1;
        align: left middle;
    }

    #watch-actions Button {
        margin-right: 1;
    }

    #watch-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #watch-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, entries: list[WatchListEntry]) -> None:
        super().__init__()
        self._entries = [
            WatchListEntry(
                pattern=entry.pattern,
                match_type=entry.match_type,
                case_sensitive=entry.case_sensitive,
            )
            for entry in entries
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="watch-dialog"):
            yield Label("Watch List Manager", id="watch-title")
            with Horizontal(id="watch-body"):
                yield ListView(id="watch-list")
                with Vertical(id="watch-form"):
                    yield Label("Pattern")
                    yield Input(placeholder="e.g., diffusion", id="watch-pattern")
                    yield Label("Match Type")
                    yield Select(
                        [(value, value) for value in WATCH_MATCH_TYPES],
                        id="watch-type",
                    )
                    yield Checkbox("Case sensitive", id="watch-case")
                    with Horizontal(id="watch-actions"):
                        yield Button("Add", variant="primary", id="watch-add")
                        yield Button("Update", variant="default", id="watch-update")
                        yield Button("Delete", variant="default", id="watch-delete")
            with Horizontal(id="watch-buttons"):
                yield Button("Cancel", variant="default", id="watch-cancel")
                yield Button("Save (Ctrl+S)", variant="primary", id="watch-save")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#watch-pattern", Input).focus()

    def _refresh_list(self) -> None:
        list_view = self.query_one("#watch-list", ListView)
        list_view.clear()
        for entry in self._entries:
            label = f"{entry.match_type}: {entry.pattern}"
            if entry.case_sensitive:
                label = f"{label} (Aa)"
            list_view.mount(WatchListItem(entry, Label(label)))
        if list_view.children:
            list_view.index = 0
            self._populate_form(list_view.highlighted_child)

    def _populate_form(self, item: ListItem | None) -> None:
        if not isinstance(item, WatchListItem):
            return
        self.query_one("#watch-pattern", Input).value = item.entry.pattern
        self.query_one("#watch-type", Select).value = item.entry.match_type
        self.query_one("#watch-case", Checkbox).value = item.entry.case_sensitive

    def _build_entry_from_form(self) -> WatchListEntry | None:
        pattern = self.query_one("#watch-pattern", Input).value.strip()
        match_value = self.query_one("#watch-type", Select).value
        match_type = match_value if isinstance(match_value, str) else "author"
        case_sensitive = self.query_one("#watch-case", Checkbox).value
        if not pattern:
            self.notify("Pattern cannot be empty", title="Watch", severity="warning")
            return None
        if match_type not in WATCH_MATCH_TYPES:
            match_type = "author"
        return WatchListEntry(
            pattern=pattern,
            match_type=match_type,
            case_sensitive=case_sensitive,
        )

    def action_save(self) -> None:
        self.dismiss(self._entries)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(ListView.Highlighted, "#watch-list")
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        self._populate_form(event.item)

    @on(Button.Pressed, "#watch-add")
    def on_add_pressed(self) -> None:
        entry = self._build_entry_from_form()
        if not entry:
            return
        self._entries.append(entry)
        self._refresh_list()

    @on(Button.Pressed, "#watch-update")
    def on_update_pressed(self) -> None:
        list_view = self.query_one("#watch-list", ListView)
        if not isinstance(list_view.highlighted_child, WatchListItem):
            self.notify("Select a watch entry to update", title="Watch")
            return
        entry = self._build_entry_from_form()
        if not entry:
            return
        index = list_view.index if list_view.index is not None else 0
        self._entries[index] = entry
        self._refresh_list()

    @on(Button.Pressed, "#watch-delete")
    def on_delete_pressed(self) -> None:
        list_view = self.query_one("#watch-list", ListView)
        if not isinstance(list_view.highlighted_child, WatchListItem):
            self.notify("Select a watch entry to delete", title="Watch")
            return
        index = list_view.index if list_view.index is not None else 0
        self._entries.pop(index)
        self._refresh_list()

    @on(Button.Pressed, "#watch-save")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#watch-cancel")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


class ArxivSearchModal(ModalScreen[ArxivSearchRequest | None]):
    """Modal dialog for searching the full arXiv API."""

    BINDINGS = [
        Binding("enter", "search", "Search"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ArxivSearchModal {
        align: center middle;
    }

    #arxiv-search-dialog {
        width: 70%;
        height: auto;
        min-width: 60;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #arxiv-search-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #arxiv-search-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #arxiv-search-query,
    #arxiv-search-field,
    #arxiv-search-category {
        width: 100%;
        background: $th-panel;
        border: none;
        margin-bottom: 1;
    }

    #arxiv-search-query:focus,
    #arxiv-search-field:focus,
    #arxiv-search-category:focus {
        border-left: tall $th-accent;
    }

    #arxiv-search-buttons {
        height: auto;
        align: right middle;
    }

    #arxiv-search-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        initial_query: str = "",
        initial_field: str = "all",
        initial_category: str = "",
    ) -> None:
        super().__init__()
        self._initial_query = initial_query
        self._initial_field = initial_field if initial_field in ARXIV_QUERY_FIELDS else "all"
        self._initial_category = initial_category

    def compose(self) -> ComposeResult:
        with Vertical(id="arxiv-search-dialog"):
            yield Label("Search All arXiv", id="arxiv-search-title")
            yield Label(
                "Query all arXiv by field, with optional category filter.",
                id="arxiv-search-help",
            )
            yield Input(
                value=self._initial_query,
                placeholder="Search query (e.g., diffusion transformers)",
                id="arxiv-search-query",
            )
            yield Select(
                [
                    ("All fields", "all"),
                    ("Title", "title"),
                    ("Author", "author"),
                    ("Abstract", "abstract"),
                ],
                id="arxiv-search-field",
            )
            yield Input(
                value=self._initial_category,
                placeholder="Optional category (e.g., cs.AI)",
                id="arxiv-search-category",
            )
            with Horizontal(id="arxiv-search-buttons"):
                yield Button("Cancel (Esc)", variant="default", id="arxiv-cancel")
                yield Button("Search (Enter)", variant="primary", id="arxiv-search")

    def on_mount(self) -> None:
        self.query_one("#arxiv-search-field", Select).value = self._initial_field
        self.query_one("#arxiv-search-query", Input).focus()

    def action_search(self) -> None:
        query = self.query_one("#arxiv-search-query", Input).value.strip()
        category = self.query_one("#arxiv-search-category", Input).value.strip()
        field_value = self.query_one("#arxiv-search-field", Select).value
        field = field_value if isinstance(field_value, str) else "all"

        try:
            build_arxiv_search_query(query, field, category)
        except ValueError as exc:
            self.notify(str(exc), title="arXiv Search", severity="warning")
            return

        self.dismiss(ArxivSearchRequest(query=query, field=field, category=category))

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#arxiv-search")
    def on_search_pressed(self) -> None:
        self.action_search()

    @on(Button.Pressed, "#arxiv-cancel")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    @on(Input.Submitted, "#arxiv-search-query")
    def on_query_submitted(self) -> None:
        self.action_search()

    @on(Input.Submitted, "#arxiv-search-category")
    def on_category_submitted(self) -> None:
        self.action_search()


class RecommendationSourceModal(ModalScreen[str]):
    """Simple choice dialog: local or Semantic Scholar recommendations."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("l", "local", "Local", show=False),
        Binding("s", "s2", "S2", show=False),
    ]

    CSS = """
    RecommendationSourceModal {
        align: center middle;
    }

    #rec-source-dialog {
        width: 50;
        height: auto;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #rec-source-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
    }

    #rec-source-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #rec-source-buttons Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="rec-source-dialog"):
            yield Label("Recommendation Source", id="rec-source-title")
            with Horizontal(id="rec-source-buttons"):
                yield Button("Local (TF-IDF)", variant="default", id="local-btn")
                yield Button("Semantic Scholar", variant="primary", id="s2-btn")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_local(self) -> None:
        self.dismiss("local")

    def action_s2(self) -> None:
        self.dismiss("s2")

    @on(Button.Pressed, "#local-btn")
    def on_local_pressed(self) -> None:
        self.action_local()

    @on(Button.Pressed, "#s2-btn")
    def on_s2_pressed(self) -> None:
        self.action_s2()


class RecommendationListItem(ListItem):
    """A list item for the recommendations screen that stores a paper reference."""

    def __init__(self, paper: Paper, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.paper = paper


class RecommendationsScreen(ModalScreen[str | None]):
    """Modal screen displaying similar papers."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("enter", "select", "Select"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    CSS = """
    RecommendationsScreen {
        align: center middle;
    }

    #recommendations-dialog {
        width: 80%;
        height: 80%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #recommendations-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
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
        background: $th-highlight;
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
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(self, target_paper: Paper, similar_papers: list[tuple[Paper, float]]) -> None:
        super().__init__()
        self._target_paper = target_paper
        self._similar_papers = similar_papers

    def compose(self) -> ComposeResult:
        with Vertical(id="recommendations-dialog"):
            truncated_title = truncate_text(self._target_paper.title, RECOMMENDATION_TITLE_MAX_LEN)
            yield Label(f"Similar to: {truncated_title}", id="recommendations-title")
            yield ListView(id="recommendations-list")
            with Horizontal(id="recommendations-buttons"):
                yield Button("Close (Esc)", variant="default", id="close-btn")
                yield Button("Go to Paper (Enter)", variant="primary", id="select-btn")

    def on_mount(self) -> None:
        list_view = self.query_one("#recommendations-list", ListView)
        for paper, score in self._similar_papers:
            safe_title = escape_rich_text(paper.title)
            safe_categories = escape_rich_text(paper.categories)
            item = RecommendationListItem(
                paper,
                Static(f"[bold]{safe_title}[/]", classes="rec-title"),
                Static(
                    f"[dim]{paper.arxiv_id}[/] | {safe_categories} | [{THEME_COLORS['green']}]{score:.0%}[/] match",
                    classes="rec-meta",
                ),
            )
            list_view.mount(item)
        if list_view.children:
            list_view.index = 0
        list_view.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        list_view = self.query_one("#recommendations-list", ListView)
        if isinstance(list_view.highlighted_child, RecommendationListItem):
            self.dismiss(list_view.highlighted_child.paper.arxiv_id)
        else:
            self.dismiss(None)

    def action_cursor_down(self) -> None:
        self.query_one("#recommendations-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#recommendations-list", ListView).action_cursor_up()

    @on(Button.Pressed, "#close-btn")
    def on_close_pressed(self) -> None:
        self.action_cancel()

    @on(Button.Pressed, "#select-btn")
    def on_select_pressed(self) -> None:
        self.action_select()

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, RecommendationListItem):
            self.dismiss(event.item.paper.arxiv_id)


# ============================================================================
# Citation Graph Modal
# ============================================================================


class CitationGraphListItem(ListItem):
    """A list item in the citation graph modal storing a CitationEntry."""

    def __init__(
        self,
        entry: "CitationEntry",
        *children,
        is_local: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*children, **kwargs)
        self.entry = entry
        self.is_local = is_local


class CitationGraphScreen(ModalScreen[str | None]):
    """Modal screen for exploring citation graphs with drill-down navigation.

    Returns the arxiv_id of a local paper to jump to, or None.
    """

    BINDINGS = [
        Binding("escape", "back_or_close", "Back / Close"),
        Binding("enter", "drill_down", "Drill down"),
        Binding("o", "open_url", "Open in browser", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("tab", "switch_panel", "Switch panel", show=False),
        Binding("g", "go_to_local", "Go to paper", show=False),
    ]

    CSS = """
    CitationGraphScreen {
        align: center middle;
    }

    #citation-graph-dialog {
        width: 85%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-purple;
        padding: 0 2;
    }

    #citation-graph-breadcrumb {
        text-style: bold;
        color: $th-purple;
        margin-bottom: 1;
        height: auto;
    }

    #citation-graph-panels {
        height: 1fr;
    }

    .citation-panel {
        width: 1fr;
        height: 1fr;
    }

    .citation-panel-title {
        text-style: bold;
        color: $th-accent;
        height: auto;
    }

    .citation-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    .citation-list.active-panel {
        border-left: tall $th-purple;
    }

    .citation-list > ListItem {
        padding: 0 1;
    }

    .citation-list > ListItem.--highlight {
        background: $th-highlight;
    }

    .cite-title {
        color: $th-text;
    }

    .cite-meta {
        color: $th-muted;
    }

    #citation-graph-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #citation-graph-status {
        color: $th-muted;
        height: auto;
    }
    """

    def __init__(
        self,
        root_title: str,
        root_paper_id: str,
        references: list["CitationEntry"],
        citations: list["CitationEntry"],
        fetch_callback: Callable,
        local_arxiv_ids: frozenset[str],
    ) -> None:
        super().__init__()
        self._root_title = root_title
        self._root_paper_id = root_paper_id
        self._fetch_callback = fetch_callback
        self._local_arxiv_ids = local_arxiv_ids
        # Stack: list of (paper_id, title, refs, cites)
        self._stack: list[tuple[str, str, list, list]] = []
        # Current state
        self._current_refs = references
        self._current_cites = citations
        self._current_title = root_title
        self._current_paper_id = root_paper_id
        self._active_panel: str = "refs"  # "refs" or "cites"
        self._loading = False

    def compose(self) -> ComposeResult:
        with Vertical(id="citation-graph-dialog"):
            yield Static("", id="citation-graph-breadcrumb")
            with Horizontal(id="citation-graph-panels"):
                with Vertical(classes="citation-panel"):
                    yield Static("", id="refs-title", classes="citation-panel-title")
                    yield ListView(id="refs-list", classes="citation-list active-panel")
                with Vertical(classes="citation-panel"):
                    yield Static("", id="cites-title", classes="citation-panel-title")
                    yield ListView(id="cites-list", classes="citation-list")
            yield Static("", id="citation-graph-status")
            with Horizontal(id="citation-graph-buttons"):
                yield Button("Close (Esc)", variant="default", id="cg-close-btn")
                yield Button("Drill Down (Enter)", variant="primary", id="cg-drill-btn")

    def on_mount(self) -> None:
        self._populate_lists()
        self._update_breadcrumb()
        refs_list = self.query_one("#refs-list", ListView)
        refs_list.focus()

    def _build_citation_item(self, entry: "CitationEntry") -> CitationGraphListItem:
        """Build a list item widget for a single citation graph entry."""
        is_local = entry.arxiv_id != "" and entry.arxiv_id in self._local_arxiv_ids
        safe_title = escape_rich_text(entry.title)
        local_badge = f" [{THEME_COLORS['green']}]\\[LOCAL][/]" if is_local else ""
        year_str = str(entry.year) if entry.year else "?"
        authors_short = truncate_text(entry.authors, 50) if entry.authors else ""
        return CitationGraphListItem(
            entry,
            Static(
                f"[bold]{safe_title}[/]{local_badge}",
                classes="cite-title",
            ),
            Static(
                f"[dim]{year_str}[/] · {escape_rich_text(authors_short)}"
                f" · [{THEME_COLORS['accent']}]{entry.citation_count} cites[/]",
                classes="cite-meta",
            ),
            is_local=is_local,
        )

    def _populate_lists(self) -> None:
        """Fill both list views with current references and citations."""
        refs_list = self.query_one("#refs-list", ListView)
        cites_list = self.query_one("#cites-list", ListView)
        refs_list.clear()
        cites_list.clear()

        for entry in self._current_refs:
            refs_list.mount(self._build_citation_item(entry))
        for entry in self._current_cites:
            cites_list.mount(self._build_citation_item(entry))

        if refs_list.children:
            refs_list.index = 0
        if cites_list.children:
            cites_list.index = 0

        # Update panel titles
        self.query_one("#refs-title", Static).update(f"References ({len(self._current_refs)})")
        self.query_one("#cites-title", Static).update(f"Cited By ({len(self._current_cites)})")

        # Update status
        self._update_status()

    def _update_breadcrumb(self) -> None:
        """Update the breadcrumb trail."""
        parts = [truncate_text(t, 40) for _, t, _, _ in self._stack]
        parts.append(truncate_text(self._current_title, 40))
        breadcrumb = " → ".join(
            f"[{THEME_COLORS['purple']}]{escape_rich_text(p)}[/]" for p in parts
        )
        self.query_one("#citation-graph-breadcrumb", Static).update(f"Citation Graph: {breadcrumb}")

    def _update_status(self) -> None:
        """Update status bar with navigation hints."""
        if self._loading:
            self.query_one("#citation-graph-status", Static).update(
                "[dim]Loading citation graph...[/]"
            )
            return
        active = self._active_panel
        panel_name = "references" if active == "refs" else "cited by"
        depth = len(self._stack)
        depth_str = f" [dim](depth {depth})[/]" if depth > 0 else ""
        self.query_one("#citation-graph-status", Static).update(
            f"[dim]Tab: switch panel · Enter: drill down · "
            f"o: open · g: go to local · Esc: back[/]"
            f"  Active: [{THEME_COLORS['purple']}]{panel_name}[/]{depth_str}"
        )

    def _get_active_list(self) -> ListView:
        """Return the currently active list view."""
        list_id = "#refs-list" if self._active_panel == "refs" else "#cites-list"
        return self.query_one(list_id, ListView)

    def _get_highlighted_entry(self) -> CitationGraphListItem | None:
        """Get the highlighted item from the active list."""
        lv = self._get_active_list()
        child = lv.highlighted_child
        if isinstance(child, CitationGraphListItem):
            return child
        return None

    def action_back_or_close(self) -> None:
        """Pop one level or close the modal."""
        if self._stack:
            paper_id, title, refs, cites = self._stack.pop()
            self._current_paper_id = paper_id
            self._current_title = title
            self._current_refs = refs
            self._current_cites = cites
            self._populate_lists()
            self._update_breadcrumb()
            self._get_active_list().focus()
        else:
            self.dismiss(None)

    async def action_drill_down(self) -> None:
        """Drill into the highlighted entry's citation graph."""
        if self._loading:
            return
        item = self._get_highlighted_entry()
        if not item:
            return
        entry = item.entry
        # Push current state
        self._stack.append(
            (
                self._current_paper_id,
                self._current_title,
                self._current_refs,
                self._current_cites,
            )
        )
        self._loading = True
        self._update_status()
        try:
            refs, cites = await self._fetch_callback(entry.s2_paper_id)
            self._current_paper_id = entry.s2_paper_id
            self._current_title = entry.title
            self._current_refs = refs
            self._current_cites = cites
            self._populate_lists()
            self._update_breadcrumb()
        except Exception:
            logger.warning(
                "Citation graph fetch failed for %s",
                entry.s2_paper_id,
                exc_info=True,
            )
            # Undo the push
            self._stack.pop()
        finally:
            self._loading = False
            self._update_status()
            self._get_active_list().focus()

    def action_switch_panel(self) -> None:
        """Toggle between references and citations panels."""
        refs_list = self.query_one("#refs-list", ListView)
        cites_list = self.query_one("#cites-list", ListView)
        if self._active_panel == "refs":
            self._active_panel = "cites"
            refs_list.remove_class("active-panel")
            cites_list.add_class("active-panel")
            cites_list.focus()
        else:
            self._active_panel = "refs"
            cites_list.remove_class("active-panel")
            refs_list.add_class("active-panel")
            refs_list.focus()
        self._update_status()

    def action_open_url(self) -> None:
        """Open the highlighted entry's URL in the browser."""
        item = self._get_highlighted_entry()
        if item and item.entry.url:
            webbrowser.open(item.entry.url)

    def action_go_to_local(self) -> None:
        """If highlighted entry is local, dismiss with its arxiv_id to jump to it."""
        item = self._get_highlighted_entry()
        if item and item.is_local and item.entry.arxiv_id:
            self.dismiss(item.entry.arxiv_id)

    def action_cursor_down(self) -> None:
        self._get_active_list().action_cursor_down()

    def action_cursor_up(self) -> None:
        self._get_active_list().action_cursor_up()

    @on(Button.Pressed, "#cg-close-btn")
    def on_close_pressed(self) -> None:
        self.action_back_or_close()

    @on(Button.Pressed, "#cg-drill-btn")
    def on_drill_pressed(self) -> None:
        # Button click needs to invoke the async action; use app's tracked task
        self.app._track_task(self.action_drill_down())  # type: ignore[attr-defined]


class ConfirmModal(ModalScreen[bool]):
    """Modal dialog for confirming batch operations."""

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmModal {
        align: center middle;
    }

    #confirm-dialog {
        width: 50%;
        min-width: 40;
        height: auto;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #confirm-message {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #confirm-buttons {
        height: auto;
        align: right middle;
    }

    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self._message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes (y)", variant="warning", id="confirm-yes")
                yield Button("No (n)", variant="default", id="confirm-no")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-yes")
    def on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def on_no(self) -> None:
        self.dismiss(False)


class ExportMenuModal(ModalScreen[str]):
    """Unified export menu offering all clipboard and file export formats."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("c", "do_clipboard_plain", "Plain text", show=False),
        Binding("b", "do_clipboard_bibtex", "BibTeX", show=False),
        Binding("m", "do_clipboard_markdown", "Markdown", show=False),
        Binding("r", "do_clipboard_ris", "RIS", show=False),
        Binding("v", "do_clipboard_csv", "CSV", show=False),
        Binding("t", "do_clipboard_mdtable", "Md table", show=False),
        Binding("B", "do_file_bibtex", "BibTeX file", show=False),
        Binding("R", "do_file_ris", "RIS file", show=False),
        Binding("C", "do_file_csv", "CSV file", show=False),
    ]

    CSS = """
    ExportMenuModal {
        align: center middle;
    }

    #export-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-orange;
        padding: 0 2;
    }

    #export-title {
        text-style: bold;
        color: $th-orange;
        margin-bottom: 1;
    }

    .export-section {
        color: $th-muted;
        margin-top: 1;
    }

    .export-keys {
        padding-left: 2;
        color: $th-text;
    }

    #export-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, paper_count: int) -> None:
        super().__init__()
        self._paper_count = paper_count

    def compose(self) -> ComposeResult:
        s = "s" if self._paper_count != 1 else ""
        with Vertical(id="export-dialog"):
            yield Label(
                f"Export Papers ({self._paper_count} selected{s})",
                id="export-title",
            )
            yield Label("[bold]Clipboard[/bold]", classes="export-section")
            g = THEME_COLORS["green"]
            yield Static(
                f"  [{g}]c[/]  Plain text     [{g}]b[/]  BibTeX\n"
                f"  [{g}]m[/]  Markdown       [{g}]r[/]  RIS\n"
                f"  [{g}]v[/]  CSV            [{g}]t[/]  Md table",
                classes="export-keys",
            )
            yield Label("[bold]File[/bold]", classes="export-section")
            yield Static(
                f"  [{g}]B[/]  BibTeX (.bib)  [{g}]R[/]  RIS (.ris)\n  [{g}]C[/]  CSV (.csv)",
                classes="export-keys",
            )
            yield Static("[dim]Esc to cancel[/dim]", id="export-footer")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_do_clipboard_plain(self) -> None:
        self.dismiss("clipboard-plain")

    def action_do_clipboard_bibtex(self) -> None:
        self.dismiss("clipboard-bibtex")

    def action_do_clipboard_markdown(self) -> None:
        self.dismiss("clipboard-markdown")

    def action_do_clipboard_ris(self) -> None:
        self.dismiss("clipboard-ris")

    def action_do_clipboard_csv(self) -> None:
        self.dismiss("clipboard-csv")

    def action_do_clipboard_mdtable(self) -> None:
        self.dismiss("clipboard-mdtable")

    def action_do_file_bibtex(self) -> None:
        self.dismiss("file-bibtex")

    def action_do_file_ris(self) -> None:
        self.dismiss("file-ris")

    def action_do_file_csv(self) -> None:
        self.dismiss("file-csv")


class SummaryModeModal(ModalScreen[str]):
    """Modal for selecting AI summary mode (TLDR, methods, results, etc.)."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("d", "mode_default", "Default", show=False),
        Binding("q", "mode_quick", "Quick", show=False),
        Binding("t", "mode_tldr", "TLDR", show=False),
        Binding("m", "mode_methods", "Methods", show=False),
        Binding("r", "mode_results", "Results", show=False),
        Binding("c", "mode_comparison", "Comparison", show=False),
    ]

    CSS = """
    SummaryModeModal {
        align: center middle;
    }

    #summary-mode-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-purple;
        padding: 0 2;
    }

    #summary-mode-title {
        text-style: bold;
        color: $th-purple;
        margin-bottom: 1;
    }

    .summary-mode-keys {
        padding-left: 2;
        color: $th-text;
    }

    #summary-mode-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="summary-mode-dialog"):
            yield Label("AI Summary Mode", id="summary-mode-title")
            g = THEME_COLORS["green"]
            yield Static(
                f"  [{g}]d[/]  Default  [dim]— Full summary (Problem / Approach / Results)[/]\n"
                f"  [{g}]q[/]  Quick    [dim]— Fast abstract-only summary[/]\n"
                f"  [{g}]t[/]  TLDR     [dim]— 1-2 sentence summary[/]\n"
                f"  [{g}]m[/]  Methods  [dim]— Technical methodology deep-dive[/]\n"
                f"  [{g}]r[/]  Results  [dim]— Key experimental results with numbers[/]\n"
                f"  [{g}]c[/]  Compare  [dim]— Comparison with related work[/]",
                classes="summary-mode-keys",
            )
            yield Static("[dim]Esc to cancel[/dim]", id="summary-mode-footer")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_mode_default(self) -> None:
        self.dismiss("default")

    def action_mode_quick(self) -> None:
        self.dismiss("quick")

    def action_mode_tldr(self) -> None:
        self.dismiss("tldr")

    def action_mode_methods(self) -> None:
        self.dismiss("methods")

    def action_mode_results(self) -> None:
        self.dismiss("results")

    def action_mode_comparison(self) -> None:
        self.dismiss("comparison")


class ResearchInterestsModal(ModalScreen[str]):
    """Modal dialog for editing research interests used for relevance scoring."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ResearchInterestsModal {
        align: center middle;
    }

    #interests-dialog {
        width: 60%;
        height: 60%;
        min-width: 50;
        min-height: 15;
        background: $th-background;
        border: tall $th-accent-alt;
        padding: 0 2;
    }

    #interests-title {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #interests-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #interests-textarea {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #interests-textarea:focus {
        border-left: tall $th-accent;
    }

    #interests-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #interests-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, current_interests: str = "") -> None:
        super().__init__()
        self._current_interests = current_interests

    def compose(self) -> ComposeResult:
        with Vertical(id="interests-dialog"):
            yield Label("Research Interests", id="interests-title")
            yield Static(
                "[dim]Describe your research focus. The LLM will score papers based on this.[/]",
                id="interests-help",
            )
            yield TextArea(self._current_interests, id="interests-textarea")
            with Horizontal(id="interests-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")

    def on_mount(self) -> None:
        self.query_one("#interests-textarea", TextArea).focus()

    def action_save(self) -> None:
        text = self.query_one("#interests-textarea", TextArea).text.strip()
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


class AutoTagSuggestModal(ModalScreen[list[str] | None]):
    """Modal showing LLM-suggested tags for user to accept or modify."""

    BINDINGS = [
        Binding("ctrl+s", "accept", "Accept"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    AutoTagSuggestModal {
        align: center middle;
    }

    #autotag-dialog {
        width: 55%;
        height: auto;
        min-width: 45;
        max-height: 80%;
        background: $th-background;
        border: tall $th-green;
        padding: 0 2;
    }

    #autotag-title {
        text-style: bold;
        color: $th-green;
        margin-bottom: 1;
    }

    #autotag-current {
        color: $th-muted;
        margin-bottom: 1;
    }

    #autotag-input {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #autotag-input:focus {
        border-left: tall $th-green;
    }

    #autotag-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #autotag-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        paper_title: str,
        suggested_tags: list[str],
        current_tags: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._paper_title = paper_title
        self._suggested = suggested_tags
        self._current = current_tags or []

    def compose(self) -> ComposeResult:
        with Vertical(id="autotag-dialog"):
            yield Static(f"Auto-Tag: {self._paper_title[:60]}", id="autotag-title")
            if self._current:
                yield Static(
                    f"Current: [bold]{', '.join(self._current)}[/bold]",
                    id="autotag-current",
                )
            # Merge current + suggested, dedup
            merged = list(dict.fromkeys(self._current + self._suggested))
            yield Input(
                value=", ".join(merged),
                placeholder="Edit tags (comma-separated)",
                id="autotag-input",
            )
            with Horizontal(id="autotag-buttons"):
                yield Button("Accept [Ctrl+s]", id="accept-btn", variant="success")
                yield Button("Cancel [Esc]", id="cancel-btn")

    def action_accept(self) -> None:
        text_input = self.query_one("#autotag-input", Input)
        raw = text_input.value
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        self.dismiss(tags)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#accept-btn")
    def on_accept_pressed(self) -> None:
        self.action_accept()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


class PaperChatScreen(ModalScreen[None]):
    """Interactive chat modal for asking questions about a paper."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    CSS = """
    PaperChatScreen {
        align: center middle;
    }

    #chat-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #chat-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
        height: auto;
    }

    #chat-messages {
        height: 1fr;
        background: $th-panel;
        padding: 1 1;
    }

    .chat-user {
        color: $th-green;
        margin-bottom: 1;
    }

    .chat-assistant {
        color: $th-foreground;
        margin-bottom: 1;
    }

    .chat-system {
        color: $th-muted;
        margin-bottom: 1;
    }

    #chat-input-row {
        height: auto;
        margin-top: 1;
    }

    #chat-input {
        width: 1fr;
        background: $th-panel;
        border: none;
    }

    #chat-input:focus {
        border-left: tall $th-accent;
    }

    #chat-status {
        height: auto;
        color: $th-muted;
    }
    """

    def __init__(
        self,
        paper: "Paper",
        command_template: str,
        paper_content: str = "",
    ) -> None:
        super().__init__()
        self._paper = paper
        self._command_template = command_template
        self._paper_content = paper_content
        self._history: list[tuple[str, str]] = []  # (role, text)
        self._waiting = False

    def compose(self) -> ComposeResult:
        title = self._paper.title[:70]
        with Vertical(id="chat-dialog"):
            yield Static(f"Chat: {title}", id="chat-title")
            yield VerticalScroll(id="chat-messages")
            yield Static("", id="chat-status")
            with Horizontal(id="chat-input-row"):
                yield Input(
                    placeholder="Ask a question about this paper... (Enter to send, Esc to close)",
                    id="chat-input",
                )

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        hint = (
            "Paper content loaded. Ask anything!"
            if self._paper_content
            else "Using abstract only (HTML not available). Ask anything!"
        )
        messages = self.query_one("#chat-messages", VerticalScroll)
        messages.mount(Static(f"[dim]{hint}[/]", classes="chat-system"))

    @on(Input.Submitted, "#chat-input")
    def on_question_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self._waiting:
            return
        event.input.value = ""
        self._add_message("user", question)
        self._waiting = True
        self.query_one("#chat-status", Static).update("[dim]Thinking...[/]")
        app: ArxivBrowser = self.app  # type: ignore[assignment]
        app._track_task(self._ask_llm(question))

    def _add_message(self, role: str, text: str, *, markup: bool = False) -> None:
        self._history.append((role, text))
        display = text if markup else escape_rich_text(text)
        messages = self.query_one("#chat-messages", VerticalScroll)
        if role == "user":
            messages.mount(Static(f"[bold green]You:[/] {display}", classes="chat-user"))
        else:
            messages.mount(Static(f"[bold cyan]AI:[/] {display}", classes="chat-assistant"))
        messages.scroll_end(animate=False)

    async def _ask_llm(self, question: str) -> None:
        try:
            # Build context with conversation history
            context = CHAT_SYSTEM_PROMPT.format(
                title=self._paper.title,
                authors=self._paper.authors,
                categories=self._paper.categories,
                paper_content=self._paper_content or self._paper.abstract or "",
            )
            # Append conversation history (exclude current question)
            if self._history[:-1]:
                history_lines = [
                    f"{'User' if role == 'user' else 'Assistant'}: {text}"
                    for role, text in self._history[:-1]
                ]
                context += "\n\nConversation so far:\n" + "\n".join(history_lines)
            context += f"\n\nUser: {question}\nAssistant:"

            shell_command = _build_llm_shell_command(self._command_template, context)
            proc = await asyncio.create_subprocess_shell(  # nosec B602
                shell_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=LLM_COMMAND_TIMEOUT
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                self._add_message(
                    "assistant", "[red]Timed out waiting for response.[/]", markup=True
                )
                return

            if proc.returncode != 0:
                err = escape_rich_text((stderr or b"").decode("utf-8", errors="replace").strip())
                self._add_message("assistant", f"[red]Error: {err[:200]}[/]", markup=True)
                return

            response = (stdout or b"").decode("utf-8", errors="replace").strip()
            if response:
                self._add_message("assistant", response)
            else:
                self._add_message("assistant", "[dim]Empty response from LLM.[/]", markup=True)
        except Exception as e:
            logger.warning("Chat LLM call failed: %s", e, exc_info=True)
            self._add_message(
                "assistant", f"[red]Error: {escape_rich_text(str(e))}[/]", markup=True
            )
        finally:
            self._waiting = False
            try:
                self.query_one("#chat-status", Static).update("")
            except NoMatches:
                pass

    def action_close(self) -> None:
        self.dismiss(None)


# Section toggle hotkeys: single key → section key
_SECTION_TOGGLE_KEYS: dict[str, str] = {
    "a": "authors",
    "b": "abstract",
    "t": "tags",
    "r": "relevance",
    "s": "summary",
    "e": "s2",
    "h": "hf",
    "v": "version",
}


class SectionToggleModal(ModalScreen[list[str] | None]):
    """Modal for toggling collapsible detail pane sections."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
        Binding("a", "toggle_a", "", show=False),
        Binding("b", "toggle_b", "", show=False),
        Binding("t", "toggle_t", "", show=False),
        Binding("r", "toggle_r", "", show=False),
        Binding("s", "toggle_s", "", show=False),
        Binding("e", "toggle_e", "", show=False),
        Binding("h", "toggle_h", "", show=False),
        Binding("v", "toggle_v", "", show=False),
    ]

    CSS = """
    SectionToggleModal {
        align: center middle;
    }

    #section-toggle-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #section-toggle-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    .section-toggle-list {
        padding-left: 2;
        color: $th-text;
    }

    #section-toggle-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, collapsed: list[str]) -> None:
        super().__init__()
        self._collapsed: set[str] = set(collapsed)

    def compose(self) -> ComposeResult:
        with Vertical(id="section-toggle-dialog"):
            yield Label("Detail Pane Sections", id="section-toggle-title")
            yield Static(
                self._render_list(), id="section-toggle-list", classes="section-toggle-list"
            )
            yield Static(
                "[dim]Press key to toggle · Enter to save · Esc to cancel[/]",
                id="section-toggle-footer",
            )

    def _render_list(self) -> str:
        g = THEME_COLORS["green"]
        lines = []
        for key, section in _SECTION_TOGGLE_KEYS.items():
            name = DETAIL_SECTION_NAMES[section]
            indicator = "\u25b8" if section in self._collapsed else "\u25be"
            state = "[dim]collapsed[/]" if section in self._collapsed else f"[{g}]expanded[/]"
            lines.append(f"  [{g}]{key}[/]  {indicator} {name:<18s} {state}")
        return "\n".join(lines)

    def _toggle(self, key: str) -> None:
        section = _SECTION_TOGGLE_KEYS.get(key)
        if section is None:
            return
        if section in self._collapsed:
            self._collapsed.discard(section)
        else:
            self._collapsed.add(section)
        try:
            self.query_one("#section-toggle-list", Static).update(self._render_list())
        except NoMatches:
            pass

    def action_toggle_a(self) -> None:
        self._toggle("a")

    def action_toggle_b(self) -> None:
        self._toggle("b")

    def action_toggle_t(self) -> None:
        self._toggle("t")

    def action_toggle_r(self) -> None:
        self._toggle("r")

    def action_toggle_s(self) -> None:
        self._toggle("s")

    def action_toggle_e(self) -> None:
        self._toggle("e")

    def action_toggle_h(self) -> None:
        self._toggle("h")

    def action_toggle_v(self) -> None:
        self._toggle("v")

    def action_save(self) -> None:
        self.dismiss(sorted(self._collapsed))

    def action_cancel(self) -> None:
        self.dismiss(None)


class CommandPaletteModal(ModalScreen[str]):
    """Fuzzy-searchable command palette for discovering and executing actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    DEFAULT_CSS = """
    CommandPaletteModal {
        align: center middle;
    }

    CommandPaletteModal > Vertical {
        width: 70;
        max-height: 28;
        background: $th-panel;
        border: thick $th-accent;
        padding: 1 2;
    }

    CommandPaletteModal #palette-search {
        margin-bottom: 1;
    }

    CommandPaletteModal #palette-results {
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._commands = COMMAND_PALETTE_COMMANDS
        self._filtered: list[tuple[str, str, str, str]] = list(self._commands)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"[bold {THEME_COLORS['accent']}]Command Palette[/]")
            yield Input(placeholder="Type to search commands...", id="palette-search")
            yield OptionList(id="palette-results")

    def on_mount(self) -> None:
        self._populate_results("")
        self.query_one("#palette-search", Input).focus()

    @on(Input.Changed, "#palette-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._populate_results(event.value.strip())

    def _populate_results(self, query: str) -> None:
        """Populate the results list, optionally filtered by fuzzy query."""
        option_list = self.query_one("#palette-results", OptionList)
        option_list.clear_options()

        if query:
            q = query.lower()
            scored: list[tuple[float, tuple[str, str, str, str]]] = []
            for cmd in self._commands:
                name, desc, _, _ = cmd
                score = max(
                    fuzz.partial_ratio(q, name.lower()),
                    fuzz.partial_ratio(q, desc.lower()),
                )
                if score >= 40:
                    scored.append((score, cmd))
            scored.sort(key=lambda x: x[0], reverse=True)
            self._filtered = [cmd for _, cmd in scored]
        else:
            self._filtered = list(self._commands)

        accent = THEME_COLORS["accent"]
        muted = THEME_COLORS["muted"]
        for name, desc, key_hint, action in self._filtered:
            safe_name = escape_rich_text(name)
            safe_desc = escape_rich_text(desc)
            safe_key = escape_rich_text(key_hint)
            markup = f"[bold]{safe_name}[/]  [{muted}]{safe_desc}[/]\n  [{accent}]{safe_key}[/]"
            option_list.add_option(Option(markup, id=action))

        if option_list.option_count > 0:
            option_list.highlighted = 0

    @on(OptionList.OptionSelected, "#palette-results")
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self.dismiss(str(event.option_id))

    def action_cancel(self) -> None:
        self.dismiss("")

    def key_enter(self) -> None:
        """Execute the currently highlighted command."""
        option_list = self.query_one("#palette-results", OptionList)
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self._filtered):
            _, _, _, action = self._filtered[idx]
            self.dismiss(action)


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


class CollectionsModal(ModalScreen[str | None]):
    """Modal dialog for managing paper collections (reading lists)."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    CollectionsModal {
        align: center middle;
    }

    #col-dialog {
        width: 80%;
        height: 70%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #col-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #col-body {
        height: 1fr;
    }

    #col-list {
        width: 2fr;
        height: 1fr;
        background: $th-panel;
        border: none;
        margin-right: 2;
    }

    #col-form {
        width: 1fr;
        height: 1fr;
    }

    #col-form Label {
        color: $th-muted;
        margin-top: 1;
    }

    #col-name,
    #col-desc {
        width: 100%;
        background: $th-panel;
        border: none;
    }

    #col-name:focus,
    #col-desc:focus {
        border-left: tall $th-accent;
    }

    #col-actions {
        height: auto;
        margin-top: 1;
        align: left middle;
    }

    #col-actions Button {
        margin-right: 1;
    }

    #col-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #col-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        collections: list[PaperCollection],
        papers_by_id: dict[str, Paper] | None = None,
    ) -> None:
        super().__init__()
        self._collections = [
            PaperCollection(
                name=c.name,
                description=c.description,
                paper_ids=list(c.paper_ids),
                created=c.created,
            )
            for c in collections
        ]
        self._papers_by_id = papers_by_id or {}

    def compose(self) -> ComposeResult:
        with Vertical(id="col-dialog"):
            yield Label("Collections Manager", id="col-title")
            with Horizontal(id="col-body"):
                yield ListView(id="col-list")
                with Vertical(id="col-form"):
                    yield Label("Name")
                    yield Input(placeholder="e.g., ML Reading List", id="col-name")
                    yield Label("Description")
                    yield Input(placeholder="Optional description", id="col-desc")
                    with Horizontal(id="col-actions"):
                        yield Button("Create", variant="primary", id="col-create")
                        yield Button("Rename", variant="default", id="col-rename")
                        yield Button("Delete", variant="default", id="col-delete")
                        yield Button("View", variant="default", id="col-view")
            with Horizontal(id="col-buttons"):
                yield Button("Close", variant="default", id="col-close")
                yield Button("Save", variant="primary", id="col-save")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#col-name", Input).focus()

    def _refresh_list(self) -> None:
        list_view = self.query_one("#col-list", ListView)
        list_view.clear()
        for col in self._collections:
            count = len(col.paper_ids)
            label = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0
            self._populate_form(0)

    def _populate_form(self, index: int) -> None:
        if index < 0 or index >= len(self._collections):
            return
        col = self._collections[index]
        self.query_one("#col-name", Input).value = col.name
        self.query_one("#col-desc", Input).value = col.description

    def _get_selected_index(self) -> int | None:
        list_view = self.query_one("#col-list", ListView)
        idx = list_view.index
        if idx is None or idx < 0 or idx >= len(self._collections):
            return None
        return idx

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(ListView.Highlighted, "#col-list")
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        list_view = self.query_one("#col-list", ListView)
        idx = list_view.index
        if idx is not None:
            self._populate_form(idx)

    @on(Button.Pressed, "#col-create")
    def on_create_pressed(self) -> None:
        name = self.query_one("#col-name", Input).value.strip()
        if not name:
            self.notify("Name cannot be empty", title="Collections", severity="warning")
            return
        if any(c.name == name for c in self._collections):
            self.notify(
                f"Collection '{name}' already exists", title="Collections", severity="warning"
            )
            return
        if len(self._collections) >= MAX_COLLECTIONS:
            self.notify(
                f"Maximum {MAX_COLLECTIONS} collections", title="Collections", severity="warning"
            )
            return
        desc = self.query_one("#col-desc", Input).value.strip()
        self._collections.append(
            PaperCollection(
                name=name,
                description=desc,
                created=datetime.now().isoformat(),
            )
        )
        self._refresh_list()

    @on(Button.Pressed, "#col-rename")
    def on_rename_pressed(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            self.notify("Select a collection", title="Collections")
            return
        name = self.query_one("#col-name", Input).value.strip()
        if not name:
            self.notify("Name cannot be empty", title="Collections", severity="warning")
            return
        desc = self.query_one("#col-desc", Input).value.strip()
        self._collections[idx].name = name
        self._collections[idx].description = desc
        self._refresh_list()

    @on(Button.Pressed, "#col-delete")
    def on_delete_pressed(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            self.notify("Select a collection", title="Collections")
            return
        self._collections.pop(idx)
        self._refresh_list()

    @on(Button.Pressed, "#col-view")
    def on_view_pressed(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            self.notify("Select a collection", title="Collections")
            return
        col = self._collections[idx]
        self.app.push_screen(
            CollectionViewModal(col, self._papers_by_id),
            callback=self._on_view_result,
        )

    def _on_view_result(self, result: PaperCollection | None) -> None:
        if result is not None:
            # Find and update the collection
            for i, c in enumerate(self._collections):
                if c.name == result.name:
                    self._collections[i] = result
                    break
            self._refresh_list()

    @on(Button.Pressed, "#col-save")
    def on_save_pressed(self) -> None:
        self.dismiss("save")

    @on(Button.Pressed, "#col-close")
    def on_close_pressed(self) -> None:
        self.dismiss(None)

    @property
    def collections(self) -> list[PaperCollection]:
        return self._collections


class CollectionViewModal(ModalScreen[PaperCollection | None]):
    """Modal for viewing and editing papers in a collection."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    CollectionViewModal {
        align: center middle;
    }

    #colview-dialog {
        width: 75%;
        height: 65%;
        min-width: 50;
        min-height: 15;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #colview-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #colview-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #colview-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #colview-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        collection: PaperCollection,
        papers_by_id: dict[str, Paper] | None = None,
    ) -> None:
        super().__init__()
        self._collection = PaperCollection(
            name=collection.name,
            description=collection.description,
            paper_ids=list(collection.paper_ids),
            created=collection.created,
        )
        self._papers_by_id = papers_by_id or {}

    def compose(self) -> ComposeResult:
        count = len(self._collection.paper_ids)
        title = f"{self._collection.name} ({count} paper{'s' if count != 1 else ''})"
        with Vertical(id="colview-dialog"):
            yield Label(title, id="colview-title")
            yield ListView(id="colview-list")
            with Horizontal(id="colview-buttons"):
                yield Button("Remove Selected", variant="default", id="colview-remove")
                yield Button("Done", variant="primary", id="colview-done")

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        list_view = self.query_one("#colview-list", ListView)
        list_view.clear()
        for pid in self._collection.paper_ids:
            paper = self._papers_by_id.get(pid)
            label = paper.title if paper else pid
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#colview-remove")
    def on_remove_pressed(self) -> None:
        list_view = self.query_one("#colview-list", ListView)
        idx = list_view.index
        if idx is None or idx < 0 or idx >= len(self._collection.paper_ids):
            self.notify("Select a paper to remove", title="Collection")
            return
        self._collection.paper_ids.pop(idx)
        self._refresh_list()
        # Update title
        count = len(self._collection.paper_ids)
        self.query_one("#colview-title", Label).update(
            f"{self._collection.name} ({count} paper{'s' if count != 1 else ''})"
        )

    @on(Button.Pressed, "#colview-done")
    def on_done_pressed(self) -> None:
        self.dismiss(self._collection)


class AddToCollectionModal(ModalScreen[str | None]):
    """Quick picker to select a collection to add papers to."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    AddToCollectionModal {
        align: center middle;
    }

    #addcol-dialog {
        width: 50%;
        height: 50%;
        min-width: 40;
        min-height: 12;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #addcol-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #addcol-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #addcol-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #addcol-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, collections: list[PaperCollection]) -> None:
        super().__init__()
        self._collections = collections

    def compose(self) -> ComposeResult:
        with Vertical(id="addcol-dialog"):
            yield Label("Add to Collection", id="addcol-title")
            yield ListView(id="addcol-list")
            with Horizontal(id="addcol-buttons"):
                yield Button("Cancel", variant="default", id="addcol-cancel")

    def on_mount(self) -> None:
        list_view = self.query_one("#addcol-list", ListView)
        for col in self._collections:
            count = len(col.paper_ids)
            label = f"{col.name} ({count} paper{'s' if count != 1 else ''})"
            list_view.mount(ListItem(Label(label)))
        if list_view.children:
            list_view.index = 0

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(ListView.Selected, "#addcol-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        list_view = self.query_one("#addcol-list", ListView)
        idx = list_view.index
        if idx is not None and 0 <= idx < len(self._collections):
            self.dismiss(self._collections[idx].name)

    @on(Button.Pressed, "#addcol-cancel")
    def on_cancel_pressed(self) -> None:
        self.dismiss(None)


DATE_NAV_WINDOW_SIZE = 5


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


class ArxivBrowser(App):
    """A TUI application to browse arXiv papers."""

    TITLE = "arXiv Paper Browser"

    # Theme-aware CSS — all colors reference $th-* custom variables from registered Textual themes
    CSS = """
    Screen {
        background: $th-background;
    }

    Header {
        background: $th-panel-alt;
        color: $th-text;
    }



    #main-container {
        height: 1fr;
    }

    #left-pane {
        width: 2fr;
        min-width: 50;
        max-width: 100;
        height: 100%;
        border: tall $th-highlight;
        background: $th-panel;
    }

    #left-pane:focus-within {
        border: tall $th-accent;
    }

    #right-pane {
        width: 3fr;
        height: 100%;
        border: tall $th-highlight;
        background: $th-panel;
    }

    #right-pane:focus-within {
        border: tall $th-accent;
    }

    #list-header {
        padding: 0 1;
        background: $th-panel;
        color: $th-accent;
        text-style: bold;
    }

    #details-header {
        padding: 0 1;
        background: $th-panel;
        color: $th-accent-alt;
        text-style: bold;
    }

    #paper-list {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #details-scroll {
        height: 1fr;
        padding: 0 1;
    }

    #search-container {
        height: auto;
        padding: 0 1;
        background: $th-panel;
        display: none;
    }

    #search-container.visible {
        display: block;
    }

    #search-input {
        width: 100%;
        border: tall $th-accent;
        background: $th-background;
    }

    #search-input:focus {
        border: tall $th-accent-alt;
    }

    #paper-list > .option-list--option-highlighted {
        background: $th-highlight;
    }

    #paper-list:focus > .option-list--option-highlighted {
        background: $th-highlight-focus;
    }

    #paper-list > .option-list--option-hover {
        background: $th-panel-alt;
    }

    PaperDetails {
        padding: 0;
    }

    VerticalScroll {
        scrollbar-background: $th-scrollbar-bg;
        scrollbar-color: $th-scrollbar-thumb;
        scrollbar-color-hover: $th-scrollbar-hover;
        scrollbar-color-active: $th-scrollbar-active;
    }

    #status-bar {
        padding: 0 1;
        color: $th-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=False),
        Binding("slash", "toggle_search", "Search", show=False),
        Binding("A", "arxiv_search", "arXiv Search", show=False),
        Binding("ctrl+e", "ctrl_e_dispatch", "S2 / Exit API", show=False),
        Binding("escape", "cancel_search", "Cancel", show=False),
        Binding("o", "open_url", "Open Selected", show=False),
        Binding("P", "open_pdf", "Open PDF", show=False),
        Binding("c", "copy_selected", "Copy", show=False),
        Binding("s", "cycle_sort", "Sort", show=False),
        Binding("space", "toggle_select", "Select", show=False),
        Binding("a", "select_all", "Select All", show=False),
        Binding("u", "clear_selection", "Clear Selection", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        # Phase 2: Read/Star status and Notes/Tags
        Binding("r", "toggle_read", "Read", show=False),
        Binding("x", "toggle_star", "Star", show=False),
        Binding("n", "edit_notes", "Notes", show=False),
        Binding("t", "edit_tags", "Tags", show=False),
        # Phase 3: Watch list
        Binding("w", "toggle_watch_filter", "Watch", show=False),
        Binding("W", "manage_watch_list", "Watch List", show=False),
        # Phase 4: Bookmarked search tabs
        Binding("1", "goto_bookmark(0)", "Bookmark 1", show=False),
        Binding("2", "goto_bookmark(1)", "Bookmark 2", show=False),
        Binding("3", "goto_bookmark(2)", "Bookmark 3", show=False),
        Binding("4", "goto_bookmark(3)", "Bookmark 4", show=False),
        Binding("5", "goto_bookmark(4)", "Bookmark 5", show=False),
        Binding("6", "goto_bookmark(5)", "Bookmark 6", show=False),
        Binding("7", "goto_bookmark(6)", "Bookmark 7", show=False),
        Binding("8", "goto_bookmark(7)", "Bookmark 8", show=False),
        Binding("9", "goto_bookmark(8)", "Bookmark 9", show=False),
        Binding("ctrl+b", "add_bookmark", "Add Bookmark", show=False),
        Binding("ctrl+shift+b", "remove_bookmark", "Del Bookmark", show=False),
        # Phase 5: Abstract preview
        Binding("p", "toggle_preview", "Preview", show=False),
        # Phase 7: Vim-style marks
        Binding("m", "start_mark", "Mark", show=False),
        Binding("apostrophe", "start_goto_mark", "Goto Mark", show=False),
        # Phase 8: Export features (b/B/M accessible via E → export menu)
        Binding("E", "export_menu", "Export...", show=False),
        Binding("d", "download_pdf", "Download", show=False),
        # Phase 9: Paper similarity
        Binding("R", "show_similar", "Similar", show=False),
        # LLM summary & chat
        Binding("ctrl+s", "generate_summary", "AI Summary", show=False),
        Binding("C", "chat_with_paper", "Chat", show=False),
        # Semantic Scholar enrichment
        Binding("e", "fetch_s2", "Enrich (S2)", show=False),
        # HuggingFace trending
        Binding("ctrl+h", "toggle_hf", "Toggle HF", show=False),
        # Version tracking
        Binding("V", "check_versions", "Check Versions", show=False),
        # Citation graph
        Binding("G", "citation_graph", "Citation Graph", show=False),
        # Relevance scoring
        Binding("L", "score_relevance", "Score Relevance", show=False),
        Binding("ctrl+l", "edit_interests", "Edit Interests", show=False),
        Binding("ctrl+g", "auto_tag", "Auto-Tag", show=False),
        # Theme cycling
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        # Collapsible sections
        Binding("ctrl+d", "toggle_sections", "Sections", show=False),
        # History mode: date navigation
        Binding("bracketleft", "prev_date", "Older", show=False),
        Binding("bracketright", "next_date", "Newer", show=False),
        # Help overlay
        Binding("question_mark", "show_help", "Help (?)", show=False),
        # Command palette
        Binding("ctrl+p", "command_palette", "Commands", show=False),
        # Collections
        Binding("ctrl+k", "collections", "Collections", show=False),
    ]

    def __init__(
        self,
        papers: list[Paper],
        config: UserConfig | None = None,
        restore_session: bool = True,
        history_files: list[tuple[date, Path]] | None = None,
        current_date_index: int = 0,
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
        self._badges_dirty: set[str] = set()
        self._badge_timer: Timer | None = None
        self._sort_index: int = 0  # Index into SORT_OPTIONS

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

        # TF-IDF index for similarity (lazy-built on first use)
        self._tfidf_index: TfidfIndex | None = None

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
                search_input = self.query_one("#search-input", Input)
                search_input.value = session.current_filter
                self._apply_filter(session.current_filter)  # calls _refresh_list_view
            else:
                self._refresh_list_view()  # populate without filter

            # Restore scroll position
            option_list = self.query_one("#paper-list", OptionList)
            if option_list.option_count > 0:
                # Clamp index to valid range
                index = min(session.scroll_index, option_list.option_count - 1)
                option_list.highlighted = max(0, index)
        else:
            # Populate list (deferred from compose for faster first paint)
            self._refresh_list_view()
            # Default: select first item if available
            option_list = self.query_one("#paper-list", OptionList)
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

        # Focus the paper list so key bindings work
        self.query_one("#paper-list", OptionList).focus()

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

        # Close shared HTTP client
        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass  # Best-effort cleanup during shutdown

    def _refresh_date_navigator(self) -> None:
        """Refresh date navigator labels after DOM updates settle."""
        try:
            date_nav = self.query_one(DateNavigator)
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
            except Exception:
                pass  # __new__() mock tests lack reactive infrastructure
        try:
            self.theme = self._config.theme_name
        except Exception:
            pass  # __new__() mock tests lack reactive infrastructure
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
            details = self.query_one(PaperDetails)
            if details.paper and details.paper.arxiv_id == arxiv_id:
                abstract_text = self._abstract_cache.get(arxiv_id, "")
                details.update_paper(details.paper, abstract_text, **self._detail_kwargs(arxiv_id))
        except NoMatches:
            pass
        # Update list option if showing preview
        if self._show_abstract_preview:
            self._update_option_for_paper(arxiv_id)

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
            list_view = self.query_one("#paper-list", OptionList)
            search_input = self.query_one("#search-input", Input)

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
            details = self.query_one(PaperDetails)
            aid = paper.arxiv_id
            abstract_text = self._get_abstract_text(paper, allow_async=True)
            details.update_paper(paper, abstract_text, **self._detail_kwargs(aid))

    @on(OptionList.OptionHighlighted, "#paper-list")
    def on_paper_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Handle paper highlight (keyboard navigation) with debouncing."""
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            self._pending_detail_paper = self.filtered_papers[idx]
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
        if paper is None:
            return
        current = self._get_current_paper()
        if current is None or current.arxiv_id != paper.arxiv_id:
            return
        try:
            details = self.query_one(PaperDetails)
        except NoMatches:
            return  # Widget tree torn down during shutdown
        aid = current.arxiv_id
        abstract_text = self._get_abstract_text(current, allow_async=True)
        details.update_paper(current, abstract_text, **self._detail_kwargs(aid))

    def _cancel_pending_detail_update(self) -> None:
        """Cancel any pending debounced detail-pane update."""
        timer = self._detail_timer
        self._detail_timer = None
        if timer is not None:
            timer.stop()
        self._pending_detail_paper = None

    def action_toggle_search(self) -> None:
        """Toggle search input visibility."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
        else:
            container.add_class("visible")
            self.query_one("#search-input", Input).focus()
        self._update_footer()

    def action_cancel_search(self) -> None:
        """Cancel search and hide input."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
            self._apply_filter("")
        if self._in_arxiv_api_mode:
            self.action_exit_arxiv_search_mode()

    def _capture_local_browse_snapshot(self) -> LocalBrowseSnapshot | None:
        """Capture local browsing state before entering API search mode."""
        try:
            list_view = self.query_one("#paper-list", OptionList)
            search_input = self.query_one("#search-input", Input)
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
            self.query_one("#search-input", Input).value = snapshot.search_query
        except NoMatches:
            pass

        self._apply_filter(snapshot.search_query)

        try:
            option_list = self.query_one("#paper-list", OptionList)
            if option_list.option_count > 0:
                max_index = max(0, option_list.option_count - 1)
                option_list.highlighted = min(max(0, snapshot.list_index), max_index)
                option_list.focus()
        except NoMatches:
            pass

        if self._config.bookmarks:
            self._track_task(self._update_bookmark_bar())

    def action_ctrl_e_dispatch(self) -> None:
        """Context-sensitive Ctrl+e: exit API mode if active, else toggle S2."""
        if self._in_arxiv_api_mode:
            self.action_exit_arxiv_search_mode()
        else:
            self.action_toggle_s2()

    def action_toggle_s2(self) -> None:
        """Toggle Semantic Scholar enrichment and persist the setting."""
        prev_state = self._s2_active
        self._s2_active = not self._s2_active
        self._config.s2_enabled = self._s2_active
        if not save_config(self._config):
            self._s2_active = prev_state
            self._config.s2_enabled = prev_state
            self.notify(
                "Failed to save Semantic Scholar setting",
                title="S2",
                severity="error",
            )
            return
        state = "enabled" if self._s2_active else "disabled"
        self.notify(f"Semantic Scholar {state}", title="S2")
        self._update_status_bar()
        self._refresh_detail_pane()
        self._mark_badges_dirty("s2", immediate=True)

    async def action_fetch_s2(self) -> None:
        """Fetch Semantic Scholar data for the currently highlighted paper."""
        if not self._s2_active:
            self.notify("S2 is disabled (Ctrl+e to enable)", title="S2", severity="warning")
            return
        paper = self._get_current_paper()
        if not paper:
            return
        aid = paper.arxiv_id
        if aid in self._s2_loading:
            return  # Already fetching
        if aid in self._s2_cache:
            self.notify("S2 data already loaded", title="S2")
            return
        self._s2_loading.add(aid)
        self._refresh_detail_pane()  # Show loading indicator immediately
        # Try SQLite cache first (off main thread)
        try:
            cached = await asyncio.to_thread(
                load_s2_paper, self._s2_db_path, aid, self._config.s2_cache_ttl_days
            )
        except Exception:
            self._s2_loading.discard(aid)
            logger.warning("S2 cache lookup failed for %s", aid, exc_info=True)
            self.notify("S2 fetch failed", title="S2", severity="error")
            return
        if cached:
            self._s2_cache[aid] = cached
            self._s2_loading.discard(aid)
            self._refresh_detail_pane()
            self._refresh_current_list_item()
            return
        # Fetch from API
        try:
            self._track_task(self._fetch_s2_paper_async(aid))
        except Exception:
            self._s2_loading.discard(aid)
            raise

    async def _fetch_s2_paper_async(self, arxiv_id: str) -> None:
        """Fetch S2 paper data and update UI on completion."""
        try:
            client = self._http_client
            if client is None:
                return
            result = await fetch_s2_paper(arxiv_id, client, api_key=self._config.s2_api_key)
            if result is None:
                self.notify("No S2 data found", title="S2", severity="warning")
                return
            # Cache in memory + SQLite
            self._s2_cache[arxiv_id] = result
            await asyncio.to_thread(save_s2_paper, self._s2_db_path, result)
            # Update UI if still relevant
            self._refresh_detail_pane()
            self._refresh_current_list_item()
        except Exception:
            logger.warning("S2 fetch failed for %s", arxiv_id, exc_info=True)
            self.notify("S2 fetch failed", title="S2", severity="error")
        finally:
            self._s2_loading.discard(arxiv_id)

    def _s2_state_for(self, arxiv_id: str) -> tuple[SemanticScholarPaper | None, bool]:
        """Return (s2_data, s2_loading) for a paper, respecting the active toggle."""
        if not self._s2_active:
            return None, False
        return self._s2_cache.get(arxiv_id), arxiv_id in self._s2_loading

    # ========================================================================
    # HuggingFace trending
    # ========================================================================

    async def action_toggle_hf(self) -> None:
        """Toggle HuggingFace trending on/off and persist the setting."""
        prev_state = self._hf_active
        self._hf_active = not self._hf_active
        self._config.hf_enabled = self._hf_active
        if not save_config(self._config):
            self._hf_active = prev_state
            self._config.hf_enabled = prev_state
            self.notify(
                "Failed to save HuggingFace setting",
                title="HF",
                severity="error",
            )
            return
        if self._hf_active:
            self.notify("HuggingFace trending enabled", title="HF")
            if not self._hf_cache:
                await self._fetch_hf_daily()
        else:
            self.notify("HuggingFace trending disabled", title="HF")
        self._update_status_bar()
        self._refresh_detail_pane()
        self._mark_badges_dirty("hf", immediate=True)

    async def _fetch_hf_daily(self) -> None:
        """Fetch HF daily papers list and update caches."""
        if self._hf_loading:
            return
        self._hf_loading = True
        self._update_status_bar()
        # Try SQLite cache first
        try:
            cached = await asyncio.to_thread(
                load_hf_daily_cache, self._hf_db_path, self._config.hf_cache_ttl_hours
            )
        except Exception:
            self._hf_loading = False
            self._update_status_bar()
            logger.warning("HF cache lookup failed", exc_info=True)
            self.notify("HF fetch failed", title="HF", severity="error")
            return
        if cached is not None:
            self._hf_cache = cached
            self._hf_loading = False
            self._refresh_detail_pane()
            self._mark_badges_dirty("hf")
            matched = sum(1 for aid in self._hf_cache if aid in self._papers_by_id)
            self.notify(f"HF: {matched} trending papers matched", title="HF")
            self._update_status_bar()
            return
        # Fetch from API
        try:
            self._track_task(self._fetch_hf_daily_async())
        except Exception:
            self._hf_loading = False
            self._update_status_bar()
            raise

    async def _fetch_hf_daily_async(self) -> None:
        """Background task: fetch HF daily papers and update UI."""
        try:
            client = self._http_client
            if client is None:
                return
            papers = await fetch_hf_daily_papers(client)
            if not papers:
                self.notify("No HF trending data found", title="HF", severity="warning")
                return
            self._hf_cache = {p.arxiv_id: p for p in papers}
            await asyncio.to_thread(save_hf_daily_cache, self._hf_db_path, papers)
            self._refresh_detail_pane()
            self._mark_badges_dirty("hf")
            matched = sum(1 for aid in self._hf_cache if aid in self._papers_by_id)
            self.notify(f"HF: {matched} trending papers matched", title="HF")
        except Exception:
            logger.warning("HF daily fetch failed", exc_info=True)
            self.notify("HF fetch failed", title="HF", severity="error")
        finally:
            self._hf_loading = False
            self._update_status_bar()

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

    def _flush_badge_refresh(self) -> None:
        """Single-pass badge refresh for all visible papers."""
        self._badge_timer = None
        dirty = self._badges_dirty.copy()
        self._badges_dirty.clear()
        if not dirty:
            return
        for i in range(len(self.filtered_papers)):
            self._update_option_at_index(i)

    # ========================================================================
    # Version tracking
    # ========================================================================

    VERSION_CHECK_BATCH_SIZE = 40  # IDs per API request (URL length safe)

    async def action_check_versions(self) -> None:
        """Check starred papers for newer arXiv versions."""
        if self._version_checking:
            self.notify("Version check already in progress", title="Versions")
            return

        starred_ids = {aid for aid, meta in self._config.paper_metadata.items() if meta.starred}
        if not starred_ids:
            self.notify("No starred papers to check", title="Versions")
            return

        self._version_checking = True
        self._update_status_bar()
        self.notify(
            f"Checking {len(starred_ids)} starred papers...",
            title="Versions",
        )
        self._track_task(self._check_versions_async(starred_ids))

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
                except Exception:
                    logger.warning(
                        "Version check batch failed (IDs %d-%d)",
                        i,
                        i + len(batch),
                        exc_info=True,
                    )

            # Compare with stored versions
            updates_found = 0
            for aid, new_ver in version_map.items():
                meta = self._config.paper_metadata.get(aid)
                if meta is None or not meta.starred:
                    continue
                old_ver = meta.last_checked_version
                if old_ver is not None and new_ver > old_ver:
                    self._version_updates[aid] = (old_ver, new_ver)
                    updates_found += 1
                meta.last_checked_version = new_ver

            # Persist updated metadata
            save_config(self._config)

            # Refresh UI
            self._mark_badges_dirty("version")
            self._refresh_detail_pane()

            if updates_found > 0:
                self.notify(
                    f"{updates_found} paper(s) have new versions",
                    title="Versions",
                )
            else:
                self.notify("All starred papers are up to date", title="Versions")
        except Exception:
            logger.warning("Version check failed", exc_info=True)
            self.notify("Version check failed", title="Versions", severity="error")
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
            details = self.query_one(PaperDetails)
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
        """Exit API search mode and restore local papers."""
        if not self._in_arxiv_api_mode:
            return

        # Invalidate in-flight responses from older requests.
        self._arxiv_api_request_token += 1

        self._in_arxiv_api_mode = False
        self._arxiv_search_state = None
        self._arxiv_api_fetch_inflight = False
        self._arxiv_api_loading = False
        self._restore_local_browse_snapshot()
        self._local_browse_snapshot = None
        self._update_header()
        self.notify("Exited arXiv API mode", title="arXiv Search")

    def action_arxiv_search(self) -> None:
        """Open modal to search all arXiv."""
        default_query = ""
        default_field = "all"
        default_category = ""
        if self._arxiv_search_state is not None:
            default_query = self._arxiv_search_state.request.query
            default_field = self._arxiv_search_state.request.field
            default_category = self._arxiv_search_state.request.category

        def on_search(request: ArxivSearchRequest | None) -> None:
            if request is None:
                return
            self._track_task(self._run_arxiv_search(request, start=0))

        self.push_screen(
            ArxivSearchModal(
                initial_query=default_query,
                initial_field=default_field,
                initial_category=default_category,
            ),
            on_search,
        )

    @staticmethod
    def _format_arxiv_search_label(request: ArxivSearchRequest) -> str:
        """Build a human-readable query label for API mode UI."""
        try:
            return build_arxiv_search_query(request.query, request.field, request.category)
        except ValueError:
            return request.query or f"cat:{request.category}"

    async def _apply_arxiv_rate_limit(self) -> None:
        """Sleep as needed to respect arXiv API rate limits."""
        now = asyncio.get_running_loop().time()
        elapsed = now - self._last_arxiv_api_request_at
        if self._last_arxiv_api_request_at > 0 and elapsed < ARXIV_API_MIN_INTERVAL_SECONDS:
            wait_seconds = ARXIV_API_MIN_INTERVAL_SECONDS - elapsed
            self.notify(
                f"Waiting {wait_seconds:.1f}s for arXiv API rate limit",
                title="arXiv Search",
            )
            await asyncio.sleep(wait_seconds)
        self._last_arxiv_api_request_at = asyncio.get_running_loop().time()

    async def _fetch_arxiv_api_page(
        self,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
    ) -> list[Paper]:
        """Fetch one page of results from arXiv API."""
        search_query = build_arxiv_search_query(request.query, request.field, request.category)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start,
            "max_results": max_results,
        }
        headers = {"User-Agent": "arxiv-subscription-viewer/1.0"}

        await self._apply_arxiv_rate_limit()

        if self._http_client is not None:
            response = await self._http_client.get(
                ARXIV_API_URL,
                params=params,
                headers=headers,
                timeout=ARXIV_API_TIMEOUT,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    ARXIV_API_URL,
                    params=params,
                    headers=headers,
                    timeout=ARXIV_API_TIMEOUT,
                )

        response.raise_for_status()
        return parse_arxiv_api_feed(response.text)

    def _apply_arxiv_search_results(
        self,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
        papers: list[Paper],
    ) -> None:
        """Switch UI to API mode and render fetched papers."""
        was_in_api_mode = self._in_arxiv_api_mode
        if not was_in_api_mode and self._local_browse_snapshot is None:
            self._local_browse_snapshot = self._capture_local_browse_snapshot()

        self._in_arxiv_api_mode = True
        self._arxiv_search_state = ArxivSearchModeState(
            request=request,
            start=start,
            max_results=max_results,
        )

        # API mode has its own paper set and selection state.
        self.all_papers = papers
        self.filtered_papers = papers.copy()
        self._papers_by_id = {paper.arxiv_id: paper for paper in papers}
        self.selected_ids.clear()
        if not was_in_api_mode:
            # First API entry starts unfiltered; subsequent pages preserve user choice.
            self._watch_filter_active = False
        self._pending_query = ""
        self._highlight_terms = {"title": [], "author": [], "abstract": []}
        self._match_scores.clear()
        try:
            self.query_one("#search-input", Input).value = ""
        except NoMatches:
            pass

        self._compute_watched_papers()
        if self._watch_filter_active:
            self.filtered_papers = [
                paper for paper in self.filtered_papers if paper.arxiv_id in self._watched_paper_ids
            ]
        self._sort_papers()
        self._refresh_list_view()
        self._update_header()

        query_label = self._format_arxiv_search_label(request)
        self.sub_title = f"API search · {truncate_text(query_label, 60)}"

        try:
            self.query_one("#paper-list", OptionList).focus()
        except NoMatches:
            pass

    async def _run_arxiv_search(self, request: ArxivSearchRequest, start: int) -> None:
        """Execute an arXiv API search and display one results page."""
        if self._arxiv_api_fetch_inflight:
            self.notify("Search already in progress", title="arXiv Search")
            return

        max_results = _coerce_arxiv_api_max_results(self._config.arxiv_api_max_results)
        self._config.arxiv_api_max_results = max_results
        start = max(0, start)

        self._arxiv_api_request_token += 1
        request_token = self._arxiv_api_request_token
        self._arxiv_api_fetch_inflight = True
        self._arxiv_api_loading = True
        self._update_status_bar()

        try:
            papers = await self._fetch_arxiv_api_page(request, start, max_results)
        except ValueError as exc:
            self.notify(str(exc), title="arXiv Search", severity="error")
            return
        except httpx.HTTPStatusError as exc:
            self.notify(
                f"arXiv API returned HTTP {exc.response.status_code}",
                title="arXiv Search",
                severity="error",
            )
            return
        except (httpx.HTTPError, OSError) as exc:
            self.notify(f"Search failed: {exc}", title="arXiv Search", severity="error")
            return
        finally:
            if request_token == self._arxiv_api_request_token:
                self._arxiv_api_fetch_inflight = False
                self._arxiv_api_loading = False
                self._update_status_bar()

        # Ignore stale responses after mode exits or newer requests.
        if request_token != self._arxiv_api_request_token:
            return

        if start > 0 and not papers:
            self.notify("No more results", title="arXiv Search")
            return

        self._apply_arxiv_search_results(request, start, max_results, papers)
        page_number = (start // max_results) + 1
        if papers:
            self.notify(
                f"Loaded {len(papers)} results (page {page_number})",
                title="arXiv Search",
            )
        else:
            self.notify("No results found", title="arXiv Search")

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
        """Move cursor down (vim-style j key)."""
        list_view = self.query_one("#paper-list", OptionList)
        list_view.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up (vim-style k key)."""
        list_view = self.query_one("#paper-list", OptionList)
        list_view.action_cursor_up()

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        self._apply_filter(event.value)
        # Hide search after submission
        self.query_one("#search-container").remove_class("visible")
        # Focus the list
        self.query_one("#paper-list", OptionList).focus()

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
            return self.query_one("#search-input", Input).value.strip()
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

    def _fuzzy_search(self, query: str) -> list[Paper]:
        """Perform fuzzy search on title and authors.

        Populates self._match_scores with relevance scores.
        """
        query_lower = query.lower()
        scored_papers = []

        for paper in self.all_papers:
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
        query = query.strip()
        # Keep status/empty-state context synchronized with the applied filter.
        self._pending_query = query

        # Clear match scores by default (only fuzzy search populates them)
        self._match_scores.clear()
        _HIGHLIGHT_PATTERN_CACHE.clear()

        if not query:
            self.filtered_papers = self.all_papers.copy()
            self._highlight_terms = {"title": [], "author": [], "abstract": []}
        else:
            tokens = tokenize_query(query)
            self._highlight_terms = build_highlight_terms(tokens)
            if is_advanced_query(tokens):
                tokens = insert_implicit_and(tokens)
                rpn = to_rpn(tokens)
                self.filtered_papers = [
                    paper for paper in self.all_papers if self._matches_advanced_query(paper, rpn)
                ]
            else:
                self.filtered_papers = self._fuzzy_search(query)

        # Apply watch filter if active (intersects with other filters)
        if self._watch_filter_active:
            self.filtered_papers = [
                p for p in self.filtered_papers if p.arxiv_id in self._watched_paper_ids
            ]

        # Apply current sort order and refresh the list view
        self._sort_papers()
        self._refresh_list_view()

        logger.debug(
            "Filter applied: query=%r, matched=%d/%d papers",
            query,
            len(self.filtered_papers),
            len(self.all_papers),
        )

        # Update header with current query context
        self.query_one("#list-header", Label).update(self._format_header_text(query))
        self._update_status_bar()
        self._update_filter_pills(query)

    def _update_filter_pills(self, query: str) -> None:
        """Update the filter pill bar with current active filters."""
        if self._in_arxiv_api_mode:
            try:
                self.query_one(FilterPillBar).remove_class("visible")
            except NoMatches:
                pass
            return
        tokens = tokenize_query(query) if query.strip() else []
        try:
            pill_bar = self.query_one(FilterPillBar)
            self._track_task(pill_bar.update_pills(tokens, self._watch_filter_active))
        except NoMatches:
            pass

    @on(FilterPillBar.RemoveFilter)
    def on_remove_filter(self, event: FilterPillBar.RemoveFilter) -> None:
        """Handle removal of a filter pill by reconstructing the query."""
        search_input = self.query_one("#search-input", Input)
        current_query = search_input.value.strip()
        tokens = tokenize_query(current_query) if current_query else []
        new_query = reconstruct_query(tokens, event.token_index)
        search_input.value = new_query

    @on(FilterPillBar.RemoveWatchFilter)
    def on_remove_watch_filter(self, event: FilterPillBar.RemoveWatchFilter) -> None:
        """Handle removal of the watch filter pill."""
        self._watch_filter_active = False
        self._apply_filter(self._pending_query)

    def action_toggle_select(self) -> None:
        """Toggle selection of the currently highlighted paper."""
        paper = self._get_current_paper()
        if not paper:
            return
        aid = paper.arxiv_id
        if aid in self.selected_ids:
            self.selected_ids.discard(aid)
        else:
            self.selected_ids.add(aid)
        idx = self._get_current_index()
        if idx is not None:
            self._update_option_at_index(idx)
        self._update_header()

    def action_select_all(self) -> None:
        """Select all currently visible papers."""
        for paper in self.filtered_papers:
            self.selected_ids.add(paper.arxiv_id)
        for i in range(len(self.filtered_papers)):
            self._update_option_at_index(i)
        self._update_header()

    def action_clear_selection(self) -> None:
        """Clear all selections."""
        self.selected_ids.clear()
        for i in range(len(self.filtered_papers)):
            self._update_option_at_index(i)
        self._update_header()

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
        option_list = self.query_one("#paper-list", OptionList)
        option_list.clear_options()

        if self.filtered_papers:
            options = [
                Option(self._render_option(paper), id=paper.arxiv_id)
                for paper in self.filtered_papers
            ]
            option_list.add_options(options)
            option_list.highlighted = 0
        else:
            query = self._get_active_query()
            if query:
                empty_msg = "[dim italic]No papers match your search.[/]\n[dim]Try a different query or press [bold]Escape[/bold] to clear.[/]"
            elif self._in_arxiv_api_mode:
                empty_msg = "[dim italic]No arXiv API results on this page.[/]\n[dim]Try [bold]][/bold] for next page or [bold]A[/bold] for a new query.[/]"
            elif self._watch_filter_active:
                empty_msg = "[dim italic]No watched papers found.[/]\n[dim]Press [bold]w[/bold] to show all papers or [bold]W[/bold] to manage watch list.[/]"
            else:
                empty_msg = "[dim italic]No papers available.[/]"
            option_list.add_option(Option(empty_msg, disabled=True))
            try:
                details = self.query_one(PaperDetails)
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
            option_list = self.query_one("#paper-list", OptionList)
            option_list.replace_option_prompt_at_index(index, markup)
        except (NoMatches, OptionDoesNotExist):
            pass

    def action_cycle_sort(self) -> None:
        """Cycle through sort options: title, date, arxiv_id."""
        self._sort_index = (self._sort_index + 1) % len(SORT_OPTIONS)
        sort_key = SORT_OPTIONS[self._sort_index]
        self.notify(f"Sorted by {sort_key}", title="Sort")

        # Re-sort and refresh the list
        self._sort_papers()
        self._refresh_list_view()
        self._update_header()

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
            option_list = self.query_one("#paper-list", OptionList)
        except NoMatches:
            return None
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            return self.filtered_papers[idx]
        return None

    def _get_current_index(self) -> int | None:
        """Get the index of the currently highlighted paper."""
        try:
            option_list = self.query_one("#paper-list", OptionList)
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
        """Toggle read status of highlighted paper, or bulk toggle for selected papers."""
        if self.selected_ids:
            self._bulk_toggle_bool("is_read", "marked read", "marked unread", "Read Status")
            return

        paper = self._get_current_paper()
        if not paper:
            return
        metadata = self._get_or_create_metadata(paper.arxiv_id)
        metadata.is_read = not metadata.is_read
        idx = self._get_current_index()
        if idx is not None:
            self._update_option_at_index(idx)
        status = "read" if metadata.is_read else "unread"
        self.notify(f"Marked as {status}", title="Read Status")

    def action_toggle_star(self) -> None:
        """Toggle star status of highlighted paper, or bulk toggle for selected papers."""
        if self.selected_ids:
            self._bulk_toggle_bool("starred", "starred", "unstarred", "Star")
            return

        paper = self._get_current_paper()
        if not paper:
            return
        metadata = self._get_or_create_metadata(paper.arxiv_id)
        metadata.starred = not metadata.starred
        idx = self._get_current_index()
        if idx is not None:
            self._update_option_at_index(idx)
        status = "starred" if metadata.starred else "unstarred"
        self.notify(f"Paper {status}", title="Star")

    def action_edit_notes(self) -> None:
        """Open notes editor for the currently highlighted paper."""
        paper = self._get_current_paper()
        if not paper:
            return

        arxiv_id = paper.arxiv_id
        current_notes = ""
        if arxiv_id in self._config.paper_metadata:
            current_notes = self._config.paper_metadata[arxiv_id].notes

        def on_notes_saved(notes: str | None) -> None:
            if notes is None:
                return
            metadata = self._get_or_create_metadata(arxiv_id)
            metadata.notes = notes
            # Update the option display if still on the same paper
            cur = self._get_current_paper()
            if cur and cur.arxiv_id == arxiv_id:
                idx = self._get_current_index()
                if idx is not None:
                    self._update_option_at_index(idx)
            self.notify("Notes saved", title="Notes")

        self.push_screen(NotesModal(arxiv_id, current_notes), on_notes_saved)

    def action_edit_tags(self) -> None:
        """Open tags editor for the current paper, or bulk-tag selected papers."""
        if self.selected_ids:
            self._bulk_edit_tags()
            return

        paper = self._get_current_paper()
        if not paper:
            return

        arxiv_id = paper.arxiv_id
        current_tags: list[str] = []
        if arxiv_id in self._config.paper_metadata:
            current_tags = self._config.paper_metadata[arxiv_id].tags.copy()

        # Collect all unique tags across all paper metadata for suggestions
        all_tags = self._collect_all_tags()

        def on_tags_saved(tags: list[str] | None) -> None:
            if tags is None:
                return
            metadata = self._get_or_create_metadata(arxiv_id)
            metadata.tags = tags
            # Update the option display if still on the same paper
            cur = self._get_current_paper()
            if cur and cur.arxiv_id == arxiv_id:
                idx = self._get_current_index()
                if idx is not None:
                    self._update_option_at_index(idx)
            self.notify(f"Tags: {', '.join(tags) if tags else 'none'}", title="Tags")

        self.push_screen(TagsModal(arxiv_id, current_tags, all_tags=all_tags), on_tags_saved)

    def _collect_all_tags(self) -> list[str]:
        """Collect all unique tags across all paper metadata."""
        return list(
            dict.fromkeys(tag for meta in self._config.paper_metadata.values() for tag in meta.tags)
        )

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
        """Toggle filtering to show only watched papers."""
        self._watch_filter_active = not self._watch_filter_active

        if self._watch_filter_active:
            if not self._watched_paper_ids:
                self.notify("Watch list is empty", title="Watch", severity="warning")
                self._watch_filter_active = False
                return
            self.notify("Showing watched papers", title="Watch")
        else:
            self.notify("Showing all papers", title="Watch")

        # Re-apply current filter with watch list consideration
        query = self.query_one("#search-input", Input).value.strip()
        self._apply_filter(query)

    def action_manage_watch_list(self) -> None:
        """Open the watch list manager."""

        def on_watch_list_updated(entries: list[WatchListEntry] | None) -> None:
            if entries is None:
                return
            old_entries = list(self._config.watch_list)
            self._config.watch_list = entries
            if not save_config(self._config):
                self._config.watch_list = old_entries
                self.notify(
                    "Failed to save watch list",
                    title="Watch",
                    severity="error",
                )
                return
            self._compute_watched_papers()
            if self._watch_filter_active and not self._watched_paper_ids:
                self._watch_filter_active = False
            query = self.query_one("#search-input", Input).value.strip()
            self._apply_filter(query)
            self.notify("Watch list updated", title="Watch")

        self.push_screen(WatchListModal(self._config.watch_list), on_watch_list_updated)

    # ========================================================================
    # Phase 4: Bookmarked Search Tabs
    # ========================================================================

    async def _update_bookmark_bar(self) -> None:
        """Update the bookmark tab bar display."""
        bookmark_bar = self.query_one(BookmarkTabBar)
        await bookmark_bar.update_bookmarks(self._config.bookmarks, self._active_bookmark_index)

    async def action_goto_bookmark(self, index: int) -> None:
        """Switch to a bookmarked search query."""
        if index < 0 or index >= len(self._config.bookmarks):
            return

        bookmark = self._config.bookmarks[index]
        self._active_bookmark_index = index

        # Update search input and apply filter
        search_input = self.query_one("#search-input", Input)
        search_input.value = bookmark.query
        self._apply_filter(bookmark.query)

        # Update bookmark bar to show active tab
        await self._update_bookmark_bar()
        self.notify(f"Bookmark: {bookmark.name}", title="Search")

    async def action_add_bookmark(self) -> None:
        """Add current search query as a bookmark."""
        query = self.query_one("#search-input", Input).value.strip()

        if not query:
            self.notify("Enter a search query first", title="Bookmark", severity="warning")
            return

        if len(self._config.bookmarks) >= 9:
            self.notify("Maximum 9 bookmarks allowed", title="Bookmark", severity="warning")
            return

        # Generate a short name from the query
        name = truncate_text(query, BOOKMARK_NAME_MAX_LEN)

        bookmark = SearchBookmark(name=name, query=query)
        self._config.bookmarks.append(bookmark)
        self._active_bookmark_index = len(self._config.bookmarks) - 1

        await self._update_bookmark_bar()
        self.notify(f"Added bookmark: {name}", title="Bookmark")

    async def action_remove_bookmark(self) -> None:
        """Remove the currently active bookmark."""
        if self._active_bookmark_index < 0 or self._active_bookmark_index >= len(
            self._config.bookmarks
        ):
            self.notify("No active bookmark to remove", title="Bookmark", severity="warning")
            return

        removed = self._config.bookmarks.pop(self._active_bookmark_index)
        self._active_bookmark_index = -1

        await self._update_bookmark_bar()
        self.notify(f"Removed bookmark: {removed.name}", title="Bookmark")

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
        option_list = self.query_one("#paper-list", OptionList)
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
        """Copy selected papers as BibTeX entries to clipboard."""
        papers = self._get_target_papers()
        if not papers:
            self.notify("No paper selected", title="BibTeX", severity="warning")
            return

        bibtex_entries = [format_paper_as_bibtex(p) for p in papers]
        bibtex_text = "\n\n".join(bibtex_entries)

        if self._copy_to_clipboard(bibtex_text):
            count = len(papers)
            self.notify(
                f"Copied {count} BibTeX entr{'ies' if count > 1 else 'y'}",
                title="BibTeX",
            )
        else:
            self.notify("Failed to copy to clipboard", title="BibTeX", severity="error")

    def action_export_bibtex_file(self) -> None:
        """Export selected papers to a BibTeX file for Zotero import."""
        papers = self._get_target_papers()
        if not papers:
            self.notify("No paper selected", title="Export", severity="warning")
            return

        bibtex_entries = [format_paper_as_bibtex(p) for p in papers]
        content = "\n\n".join(bibtex_entries)
        self._export_to_file(content, "bib", "BibTeX")

    def _format_paper_as_markdown(self, paper: Paper) -> str:
        """Format a paper as Markdown."""
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return format_paper_as_markdown(paper, abstract_text)

    def action_export_markdown(self) -> None:
        """Export selected papers as Markdown to clipboard."""
        papers = self._get_target_papers()
        if not papers:
            self.notify("No paper selected", title="Markdown", severity="warning")
            return

        # Create markdown document
        lines = ["# arXiv Papers Export", "", f"*Exported {len(papers)} paper(s)*", ""]
        for paper in papers:
            lines.append(self._format_paper_as_markdown(paper))
            lines.append("")
            lines.append("---")
            lines.append("")

        markdown_text = "\n".join(lines)

        if self._copy_to_clipboard(markdown_text):
            count = len(papers)
            self.notify(
                f"Copied {count} paper{'s' if count > 1 else ''} as Markdown",
                title="Markdown",
            )
        else:
            self.notify("Failed to copy to clipboard", title="Markdown", severity="error")

    def action_export_menu(self) -> None:
        """Open the unified export menu modal."""
        papers = self._get_target_papers()
        if not papers:
            self.notify("No paper selected", title="Export", severity="warning")
            return
        self.push_screen(
            ExportMenuModal(len(papers)),
            callback=lambda fmt: self._do_export(fmt, papers) if fmt else None,
        )

    def _do_export(self, fmt: str, papers: list[Paper]) -> None:
        """Dispatch export based on format string from ExportMenuModal."""
        dispatch: dict[str, Callable[..., None]] = {
            "clipboard-plain": lambda: self.action_copy_selected(),
            "clipboard-bibtex": lambda: self.action_copy_bibtex(),
            "clipboard-markdown": lambda: self.action_export_markdown(),
            "clipboard-ris": lambda: self._export_clipboard_ris(papers),
            "clipboard-csv": lambda: self._export_clipboard_csv(papers),
            "clipboard-mdtable": lambda: self._export_clipboard_mdtable(papers),
            "file-bibtex": lambda: self.action_export_bibtex_file(),
            "file-ris": lambda: self._export_file_ris(papers),
            "file-csv": lambda: self._export_file_csv(papers),
        }
        handler = dispatch.get(fmt)
        if handler:
            handler()

    def _get_export_dir(self) -> Path:
        """Return the configured export directory path."""
        return Path(self._config.bibtex_export_dir or Path.home() / DEFAULT_BIBTEX_EXPORT_DIR)

    def _export_to_file(self, content: str, extension: str, format_name: str) -> None:
        """Write content to a timestamped file using atomic write."""
        export_dir = self._get_export_dir()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"arxiv-{timestamp}.{extension}"
        filepath = export_dir / filename
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=export_dir, suffix=".tmp", prefix=f".{extension}-")
            closed = False
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                closed = True
                os.replace(tmp_path, filepath)
            except BaseException:
                if not closed:
                    os.close(fd)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            self.notify(
                f"Failed to export {format_name}: {exc}",
                title=f"{format_name} Export",
                severity="error",
            )
            return
        self.notify(
            f"Exported to {filepath.name}",
            title=f"{format_name} Export",
        )

    def _export_clipboard_ris(self, papers: list[Paper]) -> None:
        """Copy selected papers as RIS entries to clipboard."""
        entries = []
        for paper in papers:
            abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
            entries.append(format_paper_as_ris(paper, abstract_text))
        ris_text = "\n\n".join(entries)
        if self._copy_to_clipboard(ris_text):
            count = len(papers)
            self.notify(
                f"Copied {count} RIS entr{'ies' if count > 1 else 'y'}",
                title="RIS",
            )
        else:
            self.notify("Failed to copy to clipboard", title="RIS", severity="error")

    def _export_clipboard_csv(self, papers: list[Paper]) -> None:
        """Copy selected papers as CSV to clipboard."""
        csv_text = format_papers_as_csv(papers, self._config.paper_metadata)
        if self._copy_to_clipboard(csv_text):
            count = len(papers)
            self.notify(
                f"Copied {count} paper{'s' if count > 1 else ''} as CSV",
                title="CSV",
            )
        else:
            self.notify("Failed to copy to clipboard", title="CSV", severity="error")

    def _export_clipboard_mdtable(self, papers: list[Paper]) -> None:
        """Copy selected papers as a Markdown table to clipboard."""
        table_text = format_papers_as_markdown_table(papers)
        if self._copy_to_clipboard(table_text):
            count = len(papers)
            self.notify(
                f"Copied {count} paper{'s' if count > 1 else ''} as Markdown table",
                title="Markdown Table",
            )
        else:
            self.notify(
                "Failed to copy to clipboard",
                title="Markdown Table",
                severity="error",
            )

    def _export_file_ris(self, papers: list[Paper]) -> None:
        """Export selected papers to an RIS file."""
        entries = []
        for paper in papers:
            abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
            entries.append(format_paper_as_ris(paper, abstract_text))
        content = "\n\n".join(entries)
        self._export_to_file(content, "ris", "RIS")

    def _export_file_csv(self, papers: list[Paper]) -> None:
        """Export selected papers to a CSV file."""
        content = format_papers_as_csv(papers, self._config.paper_metadata)
        self._export_to_file(content, "csv", "CSV")

    def action_export_metadata(self) -> None:
        """Export all user metadata to a portable JSON file."""
        import json as _json

        data = export_metadata(self._config)
        content = _json.dumps(data, indent=2, ensure_ascii=False)
        self._export_to_file(content, "json", "Metadata")

    def action_import_metadata(self) -> None:
        """Import metadata from a JSON file in the export directory."""
        import json as _json

        export_dir = self._get_export_dir()
        json_files = sorted(export_dir.glob("arxiv-*.json"), reverse=True)
        if not json_files:
            self.notify(
                f"No metadata files found in {export_dir}",
                title="Import",
                severity="warning",
            )
            return
        filepath = json_files[0]
        try:
            raw = filepath.read_text(encoding="utf-8")
            data = _json.loads(raw)
            papers_n, watch_n, bk_n, col_n = import_metadata(data, self._config)
        except (OSError, ValueError) as exc:
            self.notify(f"Import failed: {exc}", title="Import", severity="error")
            return
        if not save_config(self._config):
            self.notify(
                "Import applied but failed to save to disk",
                title="Import",
                severity="warning",
            )
        self._compute_watched_papers()
        self._refresh_list_view()
        parts = []
        if papers_n:
            parts.append(f"{papers_n} papers")
        if watch_n:
            parts.append(f"{watch_n} watch entries")
        if bk_n:
            parts.append(f"{bk_n} bookmarks")
        if col_n:
            parts.append(f"{col_n} collections")
        summary = ", ".join(parts) or "nothing new"
        self.notify(f"Imported {summary} from {filepath.name}", title="Import")

    def _get_target_papers(self) -> list[Paper]:
        """Get papers to export (selected or current)."""
        if self.selected_ids:
            # Preserve list order for selected papers
            ordered: list[Paper] = []
            seen: set[str] = set()
            for paper in self.filtered_papers:
                if paper.arxiv_id in self.selected_ids:
                    ordered.append(paper)
                    seen.add(paper.arxiv_id)
            # Include selected papers not in current filter
            remaining_ids = sorted(aid for aid in self.selected_ids if aid not in seen)
            for arxiv_id in remaining_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper is not None:
                    ordered.append(paper)
            return ordered
        details = self.query_one(PaperDetails)
        if details.paper:
            return [details.paper]
        return []

    # ========================================================================
    # Phase 9: Paper Similarity
    # ========================================================================

    def action_show_similar(self) -> None:
        """Show papers similar to the currently highlighted paper."""
        paper = self._get_current_paper()
        if not paper:
            self.notify("No paper selected", title="Similar", severity="warning")
            return

        if self._s2_active:
            self.push_screen(
                RecommendationSourceModal(),
                callback=lambda source: self._show_recommendations(paper, source),
            )
        else:
            self._show_recommendations(paper, "local")

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
        if self._tfidf_index is None:
            self._tfidf_index = TfidfIndex.build(
                self.all_papers,
                text_fn=lambda p: (
                    f"{p.title} {self._get_abstract_text(p, allow_async=False) or ''}"
                ),
            )
        similar_papers = find_similar_papers(
            paper,
            self.all_papers,
            metadata=self._config.paper_metadata,
            abstract_lookup=lambda p: self._get_abstract_text(p, allow_async=False) or "",
            tfidf_index=self._tfidf_index,
        )
        if not similar_papers:
            self.notify("No similar papers found", title="Similar", severity="warning")
            return
        self.push_screen(
            RecommendationsScreen(paper, similar_papers),
            self._on_recommendation_selected,
        )

    async def _show_s2_recommendations(self, paper: Paper) -> None:
        """Fetch S2 recommendations and show them in the modal."""
        try:
            self.notify("Fetching S2 recommendations...", title="S2")
            recs = await self._fetch_s2_recommendations_async(paper.arxiv_id)
            if not recs:
                self.notify("No S2 recommendations found", title="S2", severity="warning")
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
            self.notify("S2 recommendations failed", title="S2", severity="error")

    def _on_recommendation_selected(self, arxiv_id: str | None) -> None:
        """Handle selection from the recommendations modal."""
        if not arxiv_id:
            return
        option_list = self.query_one("#paper-list", OptionList)
        for i, p in enumerate(self.filtered_papers):
            if p.arxiv_id == arxiv_id:
                option_list.highlighted = i
                return
        self.notify(
            "Paper not in current view (try clearing filter)",
            title="Similar",
            severity="warning",
        )

    async def _fetch_s2_recommendations_async(self, arxiv_id: str) -> list[SemanticScholarPaper]:
        """Fetch S2 recommendations with SQLite cache."""
        cached = await asyncio.to_thread(
            load_s2_recommendations,
            self._s2_db_path,
            arxiv_id,
            S2_REC_CACHE_TTL_DAYS,
        )
        if cached:
            return cached
        client = self._http_client
        if client is None:
            return []
        recs = await fetch_s2_recommendations(arxiv_id, client, api_key=self._config.s2_api_key)
        if recs:
            await asyncio.to_thread(save_s2_recommendations, self._s2_db_path, arxiv_id, recs)
        return recs

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
        """Open the citation graph modal for the current paper."""
        if not self._s2_active:
            self.notify("S2 is disabled (Ctrl+e to enable)", title="S2", severity="warning")
            return
        paper = self._get_current_paper()
        if not paper:
            return
        # Determine S2 paper ID: prefer cached S2 data, fallback to ARXIV:id
        s2_data = self._s2_cache.get(paper.arxiv_id)
        paper_id = s2_data.s2_paper_id if s2_data else f"ARXIV:{paper.arxiv_id}"
        self.notify("Fetching citation graph...", title="Citations")
        self._track_task(self._show_citation_graph(paper_id, paper.title))

    async def _show_citation_graph(self, paper_id: str, title: str) -> None:
        """Fetch citation graph data and push the CitationGraphScreen."""
        try:
            refs, cites = await self._fetch_citation_graph(paper_id)
            if not refs and not cites:
                self.notify("No citation data found", title="Citations", severity="warning")
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
            self.notify("Citation graph failed", title="Citations", severity="error")

    async def _fetch_citation_graph(
        self, paper_id: str
    ) -> tuple[list[CitationEntry], list[CitationEntry]]:
        """Fetch references + citations with SQLite cache."""
        cache_hit = await asyncio.to_thread(
            has_s2_citation_graph_cache,
            self._s2_db_path,
            paper_id,
            S2_CITATION_GRAPH_CACHE_TTL_DAYS,
        )
        if cache_hit:
            cached_refs = await asyncio.to_thread(
                load_s2_citation_graph,
                self._s2_db_path,
                paper_id,
                "references",
                S2_CITATION_GRAPH_CACHE_TTL_DAYS,
            )
            cached_cites = await asyncio.to_thread(
                load_s2_citation_graph,
                self._s2_db_path,
                paper_id,
                "citations",
                S2_CITATION_GRAPH_CACHE_TTL_DAYS,
            )
            return cached_refs, cached_cites

        # Fetch from API
        client = self._http_client
        if client is None:
            return [], []
        api_key = self._config.s2_api_key
        refs, refs_ok = await fetch_s2_references(
            paper_id,
            client,
            api_key=api_key,
            include_status=True,
        )
        cites, cites_ok = await fetch_s2_citations(
            paper_id,
            client,
            api_key=api_key,
            include_status=True,
        )

        # Cache only when both directions completed cleanly.
        if refs_ok and cites_ok:
            await asyncio.to_thread(
                save_s2_citation_graph,
                self._s2_db_path,
                paper_id,
                "references",
                refs,
            )
            await asyncio.to_thread(
                save_s2_citation_graph,
                self._s2_db_path,
                paper_id,
                "citations",
                cites,
            )
        else:
            logger.info(
                "Skipping citation graph cache write for %s due to fetch error "
                "(refs_ok=%s cites_ok=%s)",
                paper_id,
                refs_ok,
                cites_ok,
            )
        return refs, cites

    def _on_citation_graph_selected(self, arxiv_id: str | None) -> None:
        """Handle selection from the citation graph modal (jump to local paper)."""
        self._on_recommendation_selected(arxiv_id)

    def action_cycle_theme(self) -> None:
        """Cycle through available color themes."""
        current = self._config.theme_name
        try:
            idx = THEME_NAMES.index(current)
        except ValueError:
            idx = 0
        next_idx = (idx + 1) % len(THEME_NAMES)
        self._config.theme_name = THEME_NAMES[next_idx]
        self._apply_theme_overrides()
        self._apply_category_overrides()
        try:
            self.query_one(PaperDetails).clear_cache()
        except NoMatches:
            pass
        self._refresh_list_view()
        self._refresh_detail_pane()
        self._update_status_bar()
        save_config(self._config)
        self.notify(f"Theme: {THEME_NAMES[next_idx]}", title="Theme")

    def action_toggle_sections(self) -> None:
        """Open the section toggle modal to collapse/expand detail pane sections."""

        def _on_result(result: list[str] | None) -> None:
            if result is not None:
                self._config.collapsed_sections = result
                save_config(self._config)
                self._refresh_detail_pane()

        self.push_screen(SectionToggleModal(self._config.collapsed_sections), _on_result)

    def action_show_help(self) -> None:
        """Show the help overlay with all keyboard shortcuts."""
        self.push_screen(HelpScreen())

    def action_command_palette(self) -> None:
        """Open the fuzzy-searchable command palette."""

        def _on_command_selected(action_name: str | None) -> None:
            if not action_name:
                return
            method = getattr(self, f"action_{action_name}", None)
            if method is not None:
                try:
                    result = method()
                    if asyncio.iscoroutine(result):
                        self._track_task(result)
                except Exception:
                    logger.warning("Command palette action %s failed", action_name, exc_info=True)
                    self.notify(f"Command failed: {action_name}", title="Error", severity="error")
            else:
                logger.warning("Unknown command palette action: %s", action_name)

        self.push_screen(CommandPaletteModal(), _on_command_selected)

    # ========================================================================
    # Paper Collections
    # ========================================================================

    def action_collections(self) -> None:
        """Open the collections manager modal."""
        modal = CollectionsModal(self._config.collections, self._papers_by_id)

        def _save_collections(result: str | None) -> None:
            if result != "save":
                return
            self._config.collections = modal.collections
            save_config(self._config)
            count = len(self._config.collections)
            self.notify(
                f"Saved {count} collection{'s' if count != 1 else ''}",
                title="Collections",
            )

        self.push_screen(modal, _save_collections)

    def action_add_to_collection(self) -> None:
        """Add selected papers to a collection."""
        if not self._config.collections:
            self.notify(
                "No collections. Press Ctrl+k to create one.",
                title="Collections",
                severity="warning",
            )
            return
        papers = self._get_target_papers()
        if not papers:
            return
        paper_ids = [p.arxiv_id for p in papers]

        def _on_collection_selected(name: str | None) -> None:
            if not name:
                return
            for col in self._config.collections:
                if col.name == name:
                    existing = set(col.paper_ids)
                    added = 0
                    for pid in paper_ids:
                        if pid not in existing and len(col.paper_ids) < MAX_PAPERS_PER_COLLECTION:
                            col.paper_ids.append(pid)
                            existing.add(pid)
                            added += 1
                    save_config(self._config)
                    self.notify(
                        f"Added {added} paper{'s' if added != 1 else ''} to '{name}'",
                        title="Collections",
                    )
                    break

        self.push_screen(AddToCollectionModal(self._config.collections), _on_collection_selected)

    # ========================================================================
    # LLM Summary Generation
    # ========================================================================

    def _require_llm_command(self) -> str | None:
        """Resolve LLM command, showing a notification if not configured."""
        command_template = _resolve_llm_command(self._config)
        if not command_template:
            preset = self._config.llm_preset
            if preset and preset not in LLM_PRESETS:
                valid = ", ".join(sorted(LLM_PRESETS))
                msg = f"Unknown preset '{preset}'. Valid: {valid}"
            else:
                msg = f"Set llm_command or llm_preset in config.json ({get_config_path()})"
            self.notify(msg, title="LLM not configured", severity="warning", timeout=8)
            return None
        return command_template

    def action_generate_summary(self) -> None:
        """Generate an AI summary for the currently highlighted paper."""
        command_template = self._require_llm_command()
        if not command_template:
            return

        paper = self._get_current_paper()
        if not paper:
            self.notify("No paper selected", title="AI Summary", severity="warning")
            return

        if paper.arxiv_id in self._summary_loading:
            self.notify("Summary already generating...", title="AI Summary")
            return

        self.push_screen(
            SummaryModeModal(),
            lambda mode: self._on_summary_mode_selected(mode, paper, command_template),
        )

    def _on_summary_mode_selected(
        self, mode: str | None, paper: Paper, command_template: str
    ) -> None:
        """Handle the mode chosen from SummaryModeModal."""
        if not mode:
            return
        if mode not in SUMMARY_MODES:
            self.notify(f"Unknown summary mode: {mode}", title="AI Summary", severity="error")
            return

        arxiv_id = paper.arxiv_id
        if arxiv_id in self._summary_loading:
            return

        # Resolve prompt template for this mode
        if mode == "default" and self._config.llm_prompt_template:
            prompt_template = self._config.llm_prompt_template
        else:
            prompt_template = SUMMARY_MODES[mode][1]

        cmd_hash = _compute_command_hash(command_template, prompt_template)
        mode_label = mode.upper() if mode != "default" else ""
        use_full_paper_content = mode != "quick"

        # Check SQLite cache first
        cached = _load_summary(self._summary_db_path, arxiv_id, cmd_hash)
        if cached:
            self._paper_summaries[arxiv_id] = cached
            self._summary_mode_label[arxiv_id] = mode_label
            self._summary_command_hash[arxiv_id] = cmd_hash
            self._update_abstract_display(arxiv_id)
            self.notify("Summary loaded from cache", title="AI Summary")
            return

        # Avoid showing stale content under a newly selected mode.
        if self._summary_command_hash.get(arxiv_id) != cmd_hash:
            self._paper_summaries.pop(arxiv_id, None)
            self._summary_command_hash.pop(arxiv_id, None)

        # Start async generation
        self._summary_loading.add(arxiv_id)
        self._summary_mode_label[arxiv_id] = mode_label
        self._update_abstract_display(arxiv_id)
        self._track_task(
            self._generate_summary_async(
                paper,
                command_template,
                prompt_template,
                cmd_hash,
                mode_label=mode_label,
                use_full_paper_content=use_full_paper_content,
            )
        )

    async def _generate_summary_async(
        self,
        paper: Paper,
        command_template: str,
        prompt_template: str,
        cmd_hash: str,
        mode_label: str = "",
        use_full_paper_content: bool = True,
    ) -> None:
        """Run the LLM CLI tool asynchronously and update the UI."""
        arxiv_id = paper.arxiv_id
        generated_summary = False
        try:
            if use_full_paper_content:
                # Fetch full paper content from arXiv HTML (falls back to abstract).
                # Use a shorter timeout for summary flow to keep interactions snappy.
                self.notify("Fetching paper content...", title="AI Summary")
                paper_content = await _fetch_paper_content_async(
                    paper, self._http_client, timeout=SUMMARY_HTML_TIMEOUT
                )
            else:
                abstract = paper.abstract or paper.abstract_raw or ""
                paper_content = f"Abstract:\n{abstract}" if abstract else ""

            prompt = build_llm_prompt(paper, prompt_template, paper_content)
            shell_command = _build_llm_shell_command(command_template, prompt)
            logger.debug(f"Running LLM command for {arxiv_id}: {shell_command[:100]}...")

            proc = await asyncio.create_subprocess_shell(  # nosec B602
                shell_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=LLM_COMMAND_TIMEOUT
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                self.notify(
                    f"LLM timed out after {LLM_COMMAND_TIMEOUT}s",
                    title="AI Summary",
                    severity="error",
                )
                return

            if proc.returncode != 0:
                err_msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                self.notify(
                    f"LLM failed (exit {proc.returncode}): {err_msg[:200]}",
                    title="AI Summary",
                    severity="error",
                    timeout=8,
                )
                return

            summary = (stdout or b"").decode("utf-8", errors="replace").strip()
            if not summary:
                self.notify("LLM returned empty output", title="AI Summary", severity="warning")
                return

            # Cache in memory and persist to SQLite
            self._paper_summaries[arxiv_id] = summary
            self._summary_mode_label[arxiv_id] = mode_label
            self._summary_command_hash[arxiv_id] = cmd_hash
            await asyncio.to_thread(
                _save_summary, self._summary_db_path, arxiv_id, summary, cmd_hash
            )
            generated_summary = True
            self.notify("Summary generated", title="AI Summary")

        except ValueError as e:
            # Config/template errors — show the descriptive message directly
            logger.warning("Summary config error for %s: %s", arxiv_id, e)
            self.notify(str(e), title="AI Summary", severity="error", timeout=10)
        except Exception as e:
            logger.warning("Summary generation failed for %s", arxiv_id, exc_info=True)
            self.notify(f"Summary failed: {e}", title="AI Summary", severity="error")
        finally:
            self._summary_loading.discard(arxiv_id)
            if not generated_summary and arxiv_id not in self._paper_summaries:
                self._summary_mode_label.pop(arxiv_id, None)
                self._summary_command_hash.pop(arxiv_id, None)
            self._update_abstract_display(arxiv_id)

    # ========================================================================
    # Chat with Paper
    # ========================================================================

    def action_chat_with_paper(self) -> None:
        """Open an interactive chat session about the current paper."""
        command_template = self._require_llm_command()
        if not command_template:
            return
        paper = self._get_current_paper()
        if not paper:
            self.notify("No paper selected", title="Chat", severity="warning")
            return
        self.notify("Fetching paper content...", title="Chat")
        self._track_task(self._open_chat_screen(paper, command_template))

    async def _open_chat_screen(self, paper: Paper, command_template: str) -> None:
        """Fetch paper content and open the chat modal."""
        paper_content = await _fetch_paper_content_async(paper, self._http_client)
        self.push_screen(PaperChatScreen(paper, command_template, paper_content))

    # ========================================================================
    # Relevance Scoring
    # ========================================================================

    def action_score_relevance(self) -> None:
        """Score all loaded papers for relevance using the configured LLM."""
        command_template = self._require_llm_command()
        if not command_template:
            return

        if self._relevance_scoring_active:
            self.notify("Relevance scoring already in progress", title="Relevance")
            return

        interests = self._config.research_interests
        if not interests:
            self.push_screen(
                ResearchInterestsModal(),
                lambda text: self._on_interests_saved_then_score(text, command_template),
            )
            return

        self._start_relevance_scoring(command_template, interests)

    def _on_interests_saved_then_score(self, interests: str | None, command_template: str) -> None:
        """Callback after ResearchInterestsModal: save interests then start scoring."""
        if not interests:
            return
        if self._relevance_scoring_active:
            self.notify("Relevance scoring already in progress", title="Relevance")
            return
        self._config.research_interests = interests
        save_config(self._config)
        self.notify("Research interests saved", title="Relevance")
        self._start_relevance_scoring(command_template, interests)

    def _start_relevance_scoring(self, command_template: str, interests: str) -> None:
        """Begin batch relevance scoring for all loaded papers."""
        if self._relevance_scoring_active:
            self.notify("Relevance scoring already in progress", title="Relevance")
            return
        self._relevance_scoring_active = True
        self._update_footer()
        papers = list(self.all_papers)
        self._track_task(self._score_relevance_batch_async(papers, command_template, interests))

    def action_edit_interests(self) -> None:
        """Edit research interests and clear relevance cache."""
        self.push_screen(
            ResearchInterestsModal(self._config.research_interests),
            self._on_interests_edited,
        )

    def _on_interests_edited(self, interests: str | None) -> None:
        """Callback after editing interests: save and clear cache."""
        if not interests or interests == self._config.research_interests:
            return
        self._config.research_interests = interests
        save_config(self._config)
        self._relevance_scores.clear()
        self._mark_badges_dirty("relevance", immediate=True)
        self._refresh_detail_pane()
        if interests:
            self.notify("Research interests updated — press L to re-score", title="Relevance")
        else:
            self.notify("Research interests cleared", title="Relevance")

    async def _score_relevance_batch_async(
        self,
        papers: list[Paper],
        command_template: str,
        interests: str,
    ) -> None:
        """Background task: batch-score papers for relevance."""
        try:
            interests_hash = _compute_command_hash(command_template, interests)

            # Bulk-load existing scores from SQLite
            cached_scores = await asyncio.to_thread(
                _load_all_relevance_scores, self._relevance_db_path, interests_hash
            )

            # Populate in-memory cache with DB-cached scores
            for aid, score_data in cached_scores.items():
                self._relevance_scores[aid] = score_data

            # Refresh badges for cached papers
            self._mark_badges_dirty("relevance")
            self._refresh_detail_pane()

            # Filter to uncached papers
            uncached = [p for p in papers if p.arxiv_id not in cached_scores]

            if not uncached:
                self.notify(
                    f"All {len(papers)} papers already scored",
                    title="Relevance",
                )
                return

            total = len(uncached)
            scored = 0
            failed = 0

            for i, paper in enumerate(uncached):
                self._scoring_progress = (i + 1, total)
                self._update_footer()
                prompt = build_relevance_prompt(paper, interests)
                try:
                    shell_command = _build_llm_shell_command(command_template, prompt)
                except ValueError as e:
                    logger.warning("Relevance prompt error: %s", e)
                    self.notify(str(e), title="Relevance", severity="error", timeout=10)
                    return

                try:
                    proc = await asyncio.create_subprocess_shell(  # nosec B602
                        shell_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=RELEVANCE_SCORE_TIMEOUT
                        )
                    except TimeoutError:
                        proc.kill()
                        await proc.wait()
                        logger.warning("Relevance scoring timed out for %s", paper.arxiv_id)
                        failed += 1
                        continue

                    if proc.returncode != 0:
                        err_msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                        logger.warning(
                            "Relevance scoring failed for %s (exit %d): %s",
                            paper.arxiv_id,
                            proc.returncode,
                            err_msg[:200],
                        )
                        failed += 1
                        continue

                    output = (stdout or b"").decode("utf-8", errors="replace").strip()
                    result = _parse_relevance_response(output)
                    if result is None:
                        logger.warning(
                            "Failed to parse relevance response for %s: %s",
                            paper.arxiv_id,
                            output[:200],
                        )
                        failed += 1
                        continue

                    score, reason = result
                    self._relevance_scores[paper.arxiv_id] = (score, reason)

                    # Persist to SQLite
                    await asyncio.to_thread(
                        _save_relevance_score,
                        self._relevance_db_path,
                        paper.arxiv_id,
                        interests_hash,
                        score,
                        reason,
                    )

                    # Update list item badge
                    self._update_relevance_badge(paper.arxiv_id)
                    scored += 1

                except Exception:
                    logger.warning(
                        "Relevance scoring error for %s",
                        paper.arxiv_id,
                        exc_info=True,
                    )
                    failed += 1

                # Progress notification every 5 papers
                done = i + 1
                if done % 5 == 0:
                    self.notify(
                        f"Scoring relevance {done}/{total}...",
                        title="Relevance",
                    )

            # Final notification
            msg = f"Relevance scoring complete: {scored} scored"
            if failed:
                msg += f", {failed} failed"
            cached_count = len(papers) - total
            if cached_count:
                msg += f", {cached_count} cached"
            self.notify(msg, title="Relevance")

            # Refresh display
            self._mark_badges_dirty("relevance")
            self._refresh_detail_pane()

        except Exception:
            logger.warning("Relevance batch scoring failed", exc_info=True)
            self.notify("Relevance scoring failed", title="Relevance", severity="error")
        finally:
            self._relevance_scoring_active = False
            self._scoring_progress = None
            self._update_footer()

    def _update_option_for_paper(self, arxiv_id: str) -> None:
        """Update the list option display for a specific paper by arXiv ID."""
        for i, paper in enumerate(self.filtered_papers):
            if paper.arxiv_id == arxiv_id:
                self._update_option_at_index(i)
                break

    def _update_relevance_badge(self, arxiv_id: str) -> None:
        """Update a single list item's relevance badge."""
        self._update_option_for_paper(arxiv_id)

    # ========================================================================
    # Auto-Tagging
    # ========================================================================

    def action_auto_tag(self) -> None:
        """Auto-tag current or selected papers using the configured LLM."""
        command_template = self._require_llm_command()
        if not command_template:
            return

        if self._auto_tag_active:
            self.notify("Auto-tagging already in progress", title="Auto-Tag")
            return

        taxonomy = self._collect_all_tags()
        self._auto_tag_active = True

        if self.selected_ids:
            papers = [p for p in self.all_papers if p.arxiv_id in self.selected_ids]
            if not papers:
                self._auto_tag_active = False
                self.notify("No selected papers found", title="Auto-Tag", severity="warning")
                return
            self._update_footer()
            self._track_task(self._auto_tag_batch_async(papers, command_template, taxonomy))
        else:
            paper = self._get_current_paper()
            if not paper:
                self._auto_tag_active = False
                self.notify("No paper selected", title="Auto-Tag", severity="warning")
                return
            current_tags = (self._tags_for(paper.arxiv_id) or [])[:]
            self._track_task(
                self._auto_tag_single_async(paper, command_template, taxonomy, current_tags)
            )

    async def _auto_tag_single_async(
        self,
        paper: Paper,
        command_template: str,
        taxonomy: list[str],
        current_tags: list[str],
    ) -> None:
        """Auto-tag a single paper: call LLM, show suggestion modal."""
        try:
            suggested = await self._call_auto_tag_llm(paper, command_template, taxonomy)
            if suggested is None:
                self.notify("Auto-tagging failed", title="Auto-Tag", severity="warning")
                return

            # Show modal for user to accept/modify
            self.push_screen(
                AutoTagSuggestModal(paper.title, suggested, current_tags),
                lambda tags: self._on_auto_tag_accepted(tags, paper.arxiv_id),
            )
        except Exception:
            logger.warning("Auto-tag single failed for %s", paper.arxiv_id, exc_info=True)
            self.notify("Auto-tagging failed", title="Auto-Tag", severity="error")
        finally:
            self._auto_tag_active = False
            self._update_footer()

    async def _auto_tag_batch_async(
        self,
        papers: list[Paper],
        command_template: str,
        taxonomy: list[str],
    ) -> None:
        """Batch auto-tag: call LLM for each paper, apply directly."""
        try:
            total = len(papers)
            tagged = 0
            failed = 0

            for i, paper in enumerate(papers):
                self._auto_tag_progress = (i + 1, total)
                self._update_footer()

                suggested = await self._call_auto_tag_llm(paper, command_template, taxonomy)
                if suggested is None:
                    failed += 1
                    continue

                # Apply tags directly in batch mode (merge with existing)
                meta = self._get_or_create_metadata(paper.arxiv_id)
                merged = list(dict.fromkeys(meta.tags + suggested))
                meta.tags = merged
                tagged += 1

                # Update taxonomy for subsequent papers
                for tag in suggested:
                    if tag not in taxonomy:
                        taxonomy.append(tag)

            save_config(self._config)
            self._mark_badges_dirty("tags", immediate=True)
            self._refresh_detail_pane()

            msg = f"Auto-tagged {tagged} paper{'s' if tagged != 1 else ''}"
            if failed:
                msg += f" ({failed} failed)"
            self.notify(msg, title="Auto-Tag")

        except Exception:
            logger.error("Auto-tag batch failed after tagging %d papers", tagged, exc_info=True)
            if tagged > 0:
                save_config(self._config)
            self.notify(
                f"Auto-tagging failed ({tagged} tagged before error)"
                if tagged
                else "Auto-tagging failed",
                title="Auto-Tag",
                severity="error",
            )
        finally:
            self._auto_tag_active = False
            self._auto_tag_progress = None
            self._update_footer()

    async def _call_auto_tag_llm(
        self, paper: Paper, command_template: str, taxonomy: list[str]
    ) -> list[str] | None:
        """Call the LLM to get tag suggestions for a paper. Returns tags or None on failure."""
        prompt = build_auto_tag_prompt(paper, taxonomy)
        try:
            shell_command = _build_llm_shell_command(command_template, prompt)
        except ValueError as e:
            logger.warning("Auto-tag prompt error: %s", e)
            self.notify(str(e), title="Auto-Tag", severity="error", timeout=10)
            return None

        try:
            proc = await asyncio.create_subprocess_shell(  # nosec B602
                shell_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=AUTO_TAG_TIMEOUT
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("Auto-tag timed out for %s", paper.arxiv_id)
                return None

            if proc.returncode != 0:
                err_msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                logger.warning(
                    "Auto-tag failed for %s (exit %d): %s",
                    paper.arxiv_id,
                    proc.returncode,
                    err_msg[:200],
                )
                return None

            output = (stdout or b"").decode("utf-8", errors="replace").strip()
            if not output:
                logger.warning("Auto-tag returned empty output for %s", paper.arxiv_id)
                return None

            tags = _parse_auto_tag_response(output)
            if tags is None:
                logger.warning(
                    "Failed to parse auto-tag response for %s: %s", paper.arxiv_id, output[:200]
                )
                self.notify("Could not parse LLM response", title="Auto-Tag", severity="warning")
                return None

            return tags

        except Exception:
            logger.warning("Auto-tag LLM call failed for %s", paper.arxiv_id, exc_info=True)
            return None

    def _on_auto_tag_accepted(self, tags: list[str] | None, arxiv_id: str) -> None:
        """Callback when user accepts auto-tag suggestions."""
        if tags is None:
            return
        meta = self._get_or_create_metadata(arxiv_id)
        meta.tags = tags
        save_config(self._config)

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
            self.query_one(PaperDetails).clear_cache()
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

        # Clear selection when switching dates
        self.selected_ids.clear()

        # Recompute watched papers for new paper set
        self._compute_watched_papers()

        self._notify_watch_list_matches()
        self._show_daily_digest()

        # Apply current filter and sort
        query = self.query_one("#search-input", Input).value.strip()
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
        """Navigate to previous (older) date file."""
        if self._in_arxiv_api_mode:
            self._track_task(self._change_arxiv_page(-1))
            return

        if not self._is_history_mode():
            self.notify("Not in history mode", title="Navigate", severity="warning")
            return

        if self._current_date_index >= len(self._history_files) - 1:
            self.notify("Already at oldest", title="Navigate")
            return

        if not self._set_history_index(self._current_date_index + 1):
            return
        current_date = self._get_current_date()
        if current_date:
            self.notify(f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate")

    def action_next_date(self) -> None:
        """Navigate to next (newer) date file."""
        if self._in_arxiv_api_mode:
            self._track_task(self._change_arxiv_page(1))
            return

        if not self._is_history_mode():
            self.notify("Not in history mode", title="Navigate", severity="warning")
            return

        if self._current_date_index <= 0:
            self.notify("Already at newest", title="Navigate")
            return

        if not self._set_history_index(self._current_date_index - 1):
            return
        current_date = self._get_current_date()
        if current_date:
            self.notify(f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate")

    def _update_header(self) -> None:
        """Update header with selection count and sort info."""
        query = self.query_one("#search-input", Input).value.strip()
        self.query_one("#list-header", Label).update(self._format_header_text(query))
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Update the status bar with semantic, context-aware information."""
        try:
            status = self.query_one("#status-bar", Label)
        except NoMatches:
            return

        parts = []

        # Paper count — accent if filtered, dim otherwise
        total = len(self.all_papers)
        filtered = len(self.filtered_papers)
        query = self._get_active_query()
        if query:
            truncated_query = query if len(query) <= 30 else query[:27] + "..."
            safe_query = escape_rich_text(truncated_query)
            parts.append(
                f'[{THEME_COLORS["accent"]}]{filtered}[/][dim]/{total} matching [/][{THEME_COLORS["accent"]}]"{safe_query}"[/]'
            )
        elif self._watch_filter_active:
            parts.append(f"[{THEME_COLORS['orange']}]{filtered}[/][dim]/{total} watched[/]")
        else:
            parts.append(f"[dim]{total} papers[/]")

        # Selection count — green when > 0
        selected = len(self.selected_ids)
        if selected > 0:
            parts.append(f"[bold {THEME_COLORS['green']}]{selected} selected[/]")

        # Sort order
        parts.append(f"[dim]Sort: {SORT_OPTIONS[self._sort_index]}[/]")

        # Mode badges
        if self._in_arxiv_api_mode and self._arxiv_search_state is not None:
            page = (self._arxiv_search_state.start // self._arxiv_search_state.max_results) + 1
            parts.append(f"[{THEME_COLORS['orange']}]API[/]")
            parts.append(f"[dim]Page: {page}[/]")
            if self._arxiv_api_loading:
                parts.append(f"[{THEME_COLORS['orange']}]Loading...[/]")
        if self._show_abstract_preview:
            parts.append(f"[{THEME_COLORS['purple']}]Preview[/]")
        if self._s2_active:
            s2_count = len(self._s2_cache)
            if self._s2_loading:
                parts.append(f"[{THEME_COLORS['green']}]S2 loading...[/]")
            elif s2_count > 0:
                parts.append(f"[{THEME_COLORS['green']}]S2:{s2_count}[/]")
            else:
                parts.append(f"[{THEME_COLORS['green']}]S2[/]")
        if self._hf_active:
            if self._hf_loading:
                parts.append(f"[{THEME_COLORS['orange']}]HF loading...[/]")
            else:
                matched = sum(1 for aid in self._hf_cache if aid in self._papers_by_id)
                if matched > 0:
                    parts.append(f"[{THEME_COLORS['orange']}]HF:{matched}[/]")
                else:
                    parts.append(f"[{THEME_COLORS['orange']}]HF[/]")
        if self._version_checking:
            parts.append(f"[{THEME_COLORS['pink']}]Checking versions...[/]")
        elif self._version_updates:
            parts.append(f"[{THEME_COLORS['pink']}]{len(self._version_updates)} updated[/]")

        status.update(" [dim]│[/] ".join(parts))
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
            container = self.query_one("#search-container")
            if container.has_class("visible"):
                return FOOTER_CONTEXTS["search"]
        except NoMatches:
            pass

        # arXiv API search mode
        if self._in_arxiv_api_mode:
            return FOOTER_CONTEXTS["api"]

        # Selection mode — papers selected
        if self.selected_ids:
            n = len(self.selected_ids)
            bindings = list(FOOTER_CONTEXTS["selection"])
            bindings[0] = ("o", f"open({n})")
            return bindings

        # Default browsing — dynamically show contextual hints
        bindings: list[tuple[str, str]] = [("/", "search"), ("o", "open"), ("s", "sort")]
        if self._s2_active:
            bindings.extend([("e", "S2"), ("G", "graph")])
        else:
            bindings.extend([("r", "read"), ("x", "star")])
            bindings.extend([("n", "notes"), ("t", "tags")])

        # Feature discovery hints
        has_starred = any(m.starred for m in self._config.paper_metadata.values())
        llm_configured = bool(_resolve_llm_command(self._config))
        if has_starred:
            bindings.append(("V", "versions"))
        if llm_configured:
            bindings.append(("L", "relevance"))
        if self._history_files and len(self._history_files) > 1:
            bindings.append(("[/]", "dates"))
        bindings.append(("E", "export"))
        bindings.append(("^p", "commands"))
        bindings.append(("?", "help"))
        return bindings

    def _get_footer_mode_badge(self) -> str:
        """Return a Rich-markup mode badge string for the current state."""
        pink = THEME_COLORS["pink"]
        accent = THEME_COLORS["accent"]
        orange = THEME_COLORS["orange"]
        green = THEME_COLORS["green"]
        panel_alt = THEME_COLORS["panel_alt"]
        if self._relevance_scoring_active:
            return f"[bold {pink} on {panel_alt}] SCORING [/]"
        if self._version_checking:
            return f"[bold {pink} on {panel_alt}] VERSIONS [/]"
        try:
            container = self.query_one("#search-container")
            if container.has_class("visible"):
                return f"[bold {accent} on {panel_alt}] SEARCH [/]"
        except NoMatches:
            pass
        if self._in_arxiv_api_mode:
            return f"[bold {orange} on {panel_alt}] API [/]"
        if self.selected_ids:
            n = len(self.selected_ids)
            return f"[bold {green} on {panel_alt}] {n} SEL [/]"
        return ""

    def _update_footer(self) -> None:
        """Update the context-sensitive footer based on current state."""
        try:
            footer = self.query_one(ContextFooter)
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
        url = get_pdf_url(paper)
        path = get_pdf_download_path(paper, self._config)

        try:
            # Create directory if needed (inside try for permission/disk errors)
            path.parent.mkdir(parents=True, exist_ok=True)
            if client is not None:
                response = await client.get(
                    url, timeout=PDF_DOWNLOAD_TIMEOUT, follow_redirects=True
                )
            else:
                async with httpx.AsyncClient() as tmp_client:
                    response = await tmp_client.get(
                        url, timeout=PDF_DOWNLOAD_TIMEOUT, follow_redirects=True
                    )
            response.raise_for_status()
            path.write_bytes(response.content)
            logger.debug("Downloaded PDF for %s to %s", paper.arxiv_id, path)
            return True
        except (httpx.HTTPError, OSError) as e:
            logger.debug("Download failed for %s: %s", paper.arxiv_id, e)
            return False

    def _start_downloads(self) -> None:
        """Start download tasks up to the concurrency limit."""
        while self._download_queue and len(self._downloading) < MAX_CONCURRENT_DOWNLOADS:
            paper = self._download_queue.popleft()
            if paper.arxiv_id in self._downloading:
                continue
            self._downloading.add(paper.arxiv_id)
            self._track_task(self._process_single_download(paper))

    def _is_download_batch_active(self) -> bool:
        """Return True when a download batch is active or pending."""
        return bool(self._download_queue or self._downloading or self._download_total)

    async def _process_single_download(self, paper: Paper) -> None:
        """Process a single download and update state."""
        try:
            success = await self._download_pdf_async(paper, self._http_client)
            self._download_results[paper.arxiv_id] = success
        except Exception:
            logger.warning("Download failed for %s", paper.arxiv_id, exc_info=True)
            self._download_results[paper.arxiv_id] = False
        finally:
            self._downloading.discard(paper.arxiv_id)

            # Update progress
            completed = len(self._download_results)
            total = self._download_total
            self._update_download_progress(completed, total)

            # Start more downloads if queue has items
            self._start_downloads()

            # Check if batch is complete
            if completed == total:
                self._finish_download_batch()

    def _update_download_progress(self, completed: int, total: int) -> None:
        """Update status bar and footer with download progress."""
        try:
            status_bar = self.query_one("#status-bar", Label)
            status_bar.update(f"Downloading: {completed}/{total} complete")
        except NoMatches:
            pass
        self._update_footer()

    def _finish_download_batch(self) -> None:
        """Handle completion of a download batch."""
        if self._download_total <= 0:
            return

        successes = sum(1 for v in self._download_results.values() if v)
        failures = len(self._download_results) - successes

        # Get download directory for notification
        download_dir = self._config.pdf_download_dir or f"~/{DEFAULT_PDF_DOWNLOAD_DIR}"

        if failures == 0:
            self.notify(
                f"Downloaded {successes} PDF{'s' if successes != 1 else ''} to {download_dir}",
                title="Download Complete",
            )
        else:
            self.notify(
                f"Downloaded {successes}/{self._download_total} PDFs ({failures} failed)",
                title="Download Complete",
                severity="warning",
            )

        # Reset state
        self._download_results.clear()
        self._download_total = 0
        self._update_status_bar()

    def _safe_browser_open(self, url: str) -> bool:
        """Open a URL in the browser with error handling. Returns True on success."""
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.debug("Failed to open browser for %s: %s", url, e)
            self.notify("Failed to open browser", title="Error", severity="error")
            return False

    def action_open_url(self) -> None:
        """Open selected papers' URLs in the default browser."""
        papers = self._get_target_papers()
        if not papers:
            return
        if len(papers) > BATCH_CONFIRM_THRESHOLD:
            self.push_screen(
                ConfirmModal(f"Open {len(papers)} papers in browser?"),
                lambda confirmed: self._do_open_urls(papers) if confirmed else None,
            )
        else:
            self._do_open_urls(papers)

    def _do_open_urls(self, papers: list[Paper]) -> None:
        """Open the given papers' URLs in the browser."""
        for paper in papers:
            self._safe_browser_open(get_paper_url(paper, prefer_pdf=self._config.prefer_pdf_url))
        count = len(papers)
        self.notify(f"Opening {count} paper{'s' if count > 1 else ''}", title="Browser")

    def action_open_pdf(self) -> None:
        """Open selected papers' PDF URLs in the default browser."""
        papers = self._get_target_papers()
        if not papers:
            return
        if len(papers) > BATCH_CONFIRM_THRESHOLD:
            self.push_screen(
                ConfirmModal(f"Open {len(papers)} PDFs in browser?"),
                lambda confirmed: self._do_open_pdfs(papers) if confirmed else None,
            )
        else:
            self._do_open_pdfs(papers)

    def _do_open_pdfs(self, papers: list[Paper]) -> None:
        """Open the given papers' PDF URLs in the browser or configured viewer."""
        viewer = self._config.pdf_viewer.strip()
        for paper in papers:
            url = get_pdf_url(paper)
            if viewer:
                self._open_with_viewer(viewer, url)
            else:
                self._safe_browser_open(url)
        count = len(papers)
        self.notify(f"Opening {count} PDF{'s' if count > 1 else ''}", title="PDF")

    def _open_with_viewer(self, viewer_cmd: str, url_or_path: str) -> bool:
        """Open a URL/path with a configured external viewer command.

        The command template can use {url} or {path} as placeholders.
        If no placeholder is found, the URL is appended as an argument.
        The URL/path is always shell-quoted to prevent injection.
        """
        try:
            quoted = shlex.quote(url_or_path)
            if "{url}" in viewer_cmd or "{path}" in viewer_cmd:
                cmd = viewer_cmd.replace("{url}", quoted).replace("{path}", quoted)
            else:
                cmd = f"{viewer_cmd} {quoted}"
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as e:
            logger.warning("Failed to open with viewer %r: %s", viewer_cmd, e)
            self.notify("Failed to open PDF viewer", title="Error", severity="error")
            return False

    def action_download_pdf(self) -> None:
        """Download PDFs for selected papers (or current paper)."""
        if self._is_download_batch_active():
            self.notify("Download already in progress", title="Download", severity="warning")
            return

        papers_to_download = self._get_target_papers()
        if not papers_to_download:
            self.notify("No papers to download", title="Download", severity="warning")
            return

        # Filter out already downloaded
        to_download: list[Paper] = []
        for paper in papers_to_download:
            path = get_pdf_download_path(paper, self._config)
            if path.exists() and path.stat().st_size > 0:
                logger.debug("Skipping %s: already downloaded", paper.arxiv_id)
            else:
                to_download.append(paper)

        if not to_download:
            self.notify("All PDFs already downloaded", title="Download")
            return

        if len(to_download) > BATCH_CONFIRM_THRESHOLD:
            self.push_screen(
                ConfirmModal(f"Download {len(to_download)} PDFs?"),
                lambda confirmed: self._do_start_downloads(to_download) if confirmed else None,
            )
        else:
            self._do_start_downloads(to_download)

    def _do_start_downloads(self, to_download: list[Paper]) -> None:
        """Initialize and start batch PDF downloads."""
        if self._is_download_batch_active():
            self.notify("Download already in progress", title="Download", severity="warning")
            return

        # Initialize download batch
        self._download_queue.extend(to_download)
        self._download_total = len(to_download)
        self._download_results.clear()

        # Notify and start downloads
        self.notify(
            f"Downloading {len(to_download)} PDF{'s' if len(to_download) != 1 else ''}...",
            title="Download",
        )
        self._start_downloads()

    def _format_paper_for_clipboard(self, paper: Paper) -> str:
        """Format a paper's metadata for clipboard export."""
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return format_paper_for_clipboard(paper, abstract_text)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard. Returns True on success.

        Uses platform-specific clipboard tools with timeout protection.
        Logs failures at debug level for troubleshooting.
        """
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(  # nosec B603 B607
                    ["pbcopy"],
                    input=text.encode("utf-8"),
                    check=True,
                    shell=False,
                    timeout=SUBPROCESS_TIMEOUT,
                )
            elif system == "Linux":
                # Try xclip first, then xsel
                try:
                    subprocess.run(  # nosec B603 B607
                        ["xclip", "-selection", "clipboard"],
                        input=text.encode("utf-8"),
                        check=True,
                        shell=False,
                        timeout=SUBPROCESS_TIMEOUT,
                    )
                except (FileNotFoundError, subprocess.CalledProcessError):
                    subprocess.run(  # nosec B603 B607
                        ["xsel", "--clipboard", "--input"],
                        input=text.encode("utf-8"),
                        check=True,
                        shell=False,
                        timeout=SUBPROCESS_TIMEOUT,
                    )
            elif system == "Windows":
                subprocess.run(  # nosec B603 B607
                    ["clip"],
                    input=text.encode("utf-16"),
                    check=True,
                    shell=False,
                    timeout=SUBPROCESS_TIMEOUT,
                )
            else:
                logger.debug("Clipboard copy failed: unsupported platform %s", system)
                return False
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
            OSError,
        ) as e:
            logger.debug("Clipboard copy failed: %s", e)
            return False

    def action_copy_selected(self) -> None:
        """Copy selected papers' metadata to clipboard."""
        papers_to_copy = self._get_target_papers()
        if not papers_to_copy:
            self.notify("No papers to copy", title="Copy", severity="warning")
            return

        # Format papers with separator between them
        separator = f"\n\n{CLIPBOARD_SEPARATOR}\n\n"
        formatted = separator.join(self._format_paper_for_clipboard(p) for p in papers_to_copy)

        # Copy to clipboard
        if self._copy_to_clipboard(formatted):
            count = len(papers_to_copy)
            self.notify(
                f"Copied {count} paper{'s' if count > 1 else ''} to clipboard",
                title="Copy",
            )
        else:
            self.notify(
                "Failed to copy to clipboard",
                title="Copy",
                severity="error",
            )


def _resolve_input_file(input_path: Path) -> list[Paper] | int:
    """Validate and parse an explicit input file. Returns papers or exit code."""
    arxiv_file = input_path.resolve()
    if not arxiv_file.exists():
        print(f"Error: {arxiv_file} not found", file=sys.stderr)
        return 1
    if arxiv_file.is_dir():
        print(f"Error: {arxiv_file} is a directory, not a file", file=sys.stderr)
        return 1
    if not os.access(arxiv_file, os.R_OK):
        print(f"Error: {arxiv_file} is not readable (permission denied)", file=sys.stderr)
        return 1
    try:
        return parse_arxiv_file(arxiv_file)
    except OSError as e:
        print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
        return 1


def _resolve_history_date(history_files: list[tuple[date, Path]], date_str: str) -> int | None:
    """Find the index for --date argument. Returns index or None on error."""
    try:
        target_date = datetime.strptime(date_str, HISTORY_DATE_FORMAT).date()
    except ValueError:
        print(
            f"Error: Invalid date format '{date_str}', expected YYYY-MM-DD",
            file=sys.stderr,
        )
        return None
    for i, (d, _) in enumerate(history_files):
        if d == target_date:
            return i
    print(f"Error: No file found for date {date_str}", file=sys.stderr)
    return None


def _resolve_legacy_fallback(base_dir: Path) -> list[Paper] | int:
    """Find and parse arxiv.txt in the current directory. Returns papers or exit code."""
    arxiv_file = base_dir / "arxiv.txt"
    if not arxiv_file.exists():
        print("Error: No papers found. Either:", file=sys.stderr)
        print("  - Create history/YYYY-MM-DD.txt files, or", file=sys.stderr)
        print("  - Create arxiv.txt in the current directory, or", file=sys.stderr)
        print("  - Use -i to specify an input file", file=sys.stderr)
        return 1
    if not os.access(arxiv_file, os.R_OK):
        print(f"Error: {arxiv_file} is not readable (permission denied)", file=sys.stderr)
        return 1
    try:
        return parse_arxiv_file(arxiv_file)
    except OSError as e:
        print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
        return 1


def _resolve_papers(
    args: argparse.Namespace,
    base_dir: Path,
    config: UserConfig,
    history_files: list[tuple[date, Path]],
) -> tuple[list[Paper], list[tuple[date, Path]], int] | int:
    """Resolve which papers to load based on CLI args. Returns (papers, history_files, date_index) or exit code."""
    if args.input is not None:
        result = _resolve_input_file(args.input)
        if isinstance(result, int):
            return result
        return (result, [], 0)

    if history_files:
        current_date_index = 0
        if args.date:
            idx = _resolve_history_date(history_files, args.date)
            if idx is None:
                return 1
            current_date_index = idx
        elif not args.no_restore and config.session.current_date:
            try:
                saved_date = datetime.strptime(
                    config.session.current_date, HISTORY_DATE_FORMAT
                ).date()
                newest_date = history_files[0][0]
                if saved_date >= newest_date:
                    for i, (d, _) in enumerate(history_files):
                        if d == saved_date:
                            current_date_index = i
                            break
            except ValueError:
                pass
        _, arxiv_file = history_files[current_date_index]
        try:
            papers = parse_arxiv_file(arxiv_file)
        except OSError as e:
            print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
            return 1
        return (papers, history_files, current_date_index)

    result = _resolve_legacy_fallback(base_dir)
    if isinstance(result, int):
        return result
    return (result, [], 0)


def _configure_logging(debug: bool) -> None:
    """Configure logging. When debug=True, logs to file at DEBUG level."""
    if not debug:
        # Default: suppress all logging (TUI captures stderr)
        logging.disable(logging.CRITICAL)
        return

    log_dir = Path(user_config_dir(CONFIG_APP_NAME))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "debug.log"

    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = argparse.ArgumentParser(description="Browse arXiv papers from a text file in a TUI")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input file containing arXiv metadata (overrides history mode)",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Start with fresh session (ignore saved scroll position, filters, etc.)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Open specific date in YYYY-MM-DD format (history mode only)",
    )
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates in history/ and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to file (~/.config/arxiv-browser/debug.log)",
    )
    args = parser.parse_args()

    _configure_logging(args.debug)
    logger.debug("arxiv-viewer starting, cwd=%s", Path.cwd())

    base_dir = Path.cwd()

    # Load user config early (needed for session restore)
    config = load_config()

    # Discover history files
    history_files = discover_history_files(base_dir)

    # Handle --list-dates
    if args.list_dates:
        if not history_files:
            print("No history files found in history/", file=sys.stderr)
            return 1
        print("Available dates:")
        for d, path in history_files:
            print(f"  {d.strftime(HISTORY_DATE_FORMAT)}  ({path.name})")
        return 0

    # Determine which file(s) to load
    result = _resolve_papers(args, base_dir, config, history_files)
    if isinstance(result, int):
        return result
    papers, history_files, current_date_index = result

    if not papers:
        print("No papers found in the file", file=sys.stderr)
        return 1

    # Sort papers alphabetically by title
    papers.sort(key=lambda p: p.title.lower())

    app = ArxivBrowser(
        papers,
        config=config,
        restore_session=not args.no_restore,
        history_files=history_files,
        current_date_index=current_date_index,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
