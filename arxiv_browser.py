#!/usr/bin/env python3
"""arXiv Paper Browser TUI - Browse arXiv papers from a text file.

Usage:
    python arxiv_browser.py                    # Use default arxiv.txt
    python arxiv_browser.py -i papers.txt      # Use custom file
    python arxiv_browser.py --no-restore       # Start fresh session

Key bindings:
    /       - Toggle search (fuzzy matching)
    o       - Open selected paper(s) in browser
    P       - Open selected paper(s) as PDF
    c       - Copy selected paper(s) to clipboard
    b       - Copy as BibTeX
    B       - Export BibTeX to file (for Zotero import)
    d       - Download PDF(s) to local folder
    M       - Copy as Markdown
    space   - Toggle selection
    a       - Select all visible
    u       - Clear selection
    s       - Cycle sort order (title/date/arxiv_id)
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
    [       - Go to previous (older) date (history mode)
    ]       - Go to next (newer) date (history mode)
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
import functools
import json
import logging
import os
import platform
import re
import subprocess
import sys
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from platformdirs import user_config_dir
import httpx
from rapidfuzz import fuzz
from rich.markup import escape as escape_markup
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TextArea,
)

# Public API for this module
__all__ = [
    # Core data models
    "Paper",
    "PaperMetadata",
    "UserConfig",
    "SessionState",
    "SearchBookmark",
    "WatchListEntry",
    # Parsing functions
    "parse_arxiv_file",
    "clean_latex",
    "parse_arxiv_date",
    # Configuration
    "load_config",
    "save_config",
    "get_config_path",
    # Main application
    "ArxivBrowser",
    "main",
    # Utility functions
    "format_categories",
    "find_similar_papers",
    "discover_history_files",
    "get_pdf_download_path",
]

# Module logger for debugging
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Placeholder using control characters (cannot appear in academic text)
_ESCAPED_DOLLAR = "\x00ESCAPED_DOLLAR\x00"

# UI Layout constants
MIN_LIST_WIDTH = 50
MAX_LIST_WIDTH = 100
CLIPBOARD_SEPARATOR = "=" * 80

# Sort order options
SORT_OPTIONS = ["title", "date", "arxiv_id"]

# Default color for unknown categories (Monokai gray)
DEFAULT_CATEGORY_COLOR = "#888888"

# Subprocess timeout in seconds
SUBPROCESS_TIMEOUT = 5

# Fuzzy search settings
FUZZY_SCORE_CUTOFF = 60  # Minimum score (0-100) to include in results
FUZZY_LIMIT = 100  # Maximum number of results to return

# Paper similarity settings
SIMILARITY_TOP_N = 10  # Number of similar papers to show
SIMILARITY_RECENCY_WEIGHT = 0.08
SIMILARITY_STARRED_BOOST = 0.05
SIMILARITY_UNREAD_BOOST = 0.03
SIMILARITY_READ_PENALTY = 0.02
SIMILARITY_RECENCY_DAYS = 365

# UI truncation limits
RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal
PREVIEW_ABSTRACT_MAX_LEN = 150  # Max abstract preview length in list items
BOOKMARK_NAME_MAX_LEN = 15  # Max bookmark name display length
MAX_ABSTRACT_LOADS = 32  # Maximum concurrent abstract loads

# BibTeX export settings
DEFAULT_BIBTEX_EXPORT_DIR = "arxiv-exports"  # Default subdirectory in home folder

# History file discovery limit
MAX_HISTORY_FILES = 365  # Limit to ~1 year of history to prevent memory issues

# PDF download settings
DEFAULT_PDF_DOWNLOAD_DIR = "arxiv-pdfs"  # Relative to home directory
PDF_DOWNLOAD_TIMEOUT = 60  # Seconds per download
MAX_CONCURRENT_DOWNLOADS = 3  # Limit parallel downloads

# Search debounce delay in seconds
SEARCH_DEBOUNCE_DELAY = 0.3
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "then",
        "once",
        "again",
        "further",
        "still",
        "already",
        "always",
        "never",
        "using",
        "based",
        "via",
        "novel",
        "new",
        "approach",
        "method",
        "methods",
        "paper",
        "propose",
        "proposed",
        "show",
        "results",
        "model",
        "models",
    }
)

WATCH_MATCH_TYPES = ("author", "title", "keyword")

# Date format used in arXiv emails (e.g., "Mon, 15 Jan 2024")
ARXIV_DATE_FORMAT = "%a, %d %b %Y"
# Extract the date prefix when time/zone info is present
_ARXIV_DATE_PREFIX_PATTERN = re.compile(
    r"([A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})"
)


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated.

    Args:
        text: The text to truncate.
        max_len: Maximum length before truncation (not including suffix).
        suffix: String to append when truncated.

    Returns:
        Original text if within limit, otherwise truncated with suffix.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix


def escape_rich_text(text: str) -> str:
    """Escape text for safe Rich markup rendering."""
    return escape_markup(text) if text else ""


def highlight_text(text: str, terms: list[str], color: str) -> str:
    """Highlight terms inside text using Rich markup."""
    if not text:
        return text
    escaped_text = escape_rich_text(text)
    if not terms:
        return escaped_text
    normalized = []
    seen = set()
    for term in terms:
        cleaned = term.strip()
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    if not normalized:
        return escaped_text
    normalized.sort(key=len, reverse=True)
    escaped_terms = [escape_rich_text(term) for term in normalized]
    pattern = re.compile(
        "|".join(re.escape(term) for term in escaped_terms), re.IGNORECASE
    )
    return pattern.sub(lambda match: f"[bold {color}]{match.group(0)}[/]", escaped_text)


@dataclass(slots=True)
class Paper:
    """Represents an arXiv paper entry."""

    arxiv_id: str
    date: str
    title: str
    authors: str
    categories: str
    comments: str | None
    abstract: str | None
    url: str
    abstract_raw: str = ""


# ============================================================================
# User Configuration Data Structures
# ============================================================================


@dataclass(slots=True)
class PaperMetadata:
    """User annotations for a paper (notes, tags, read status)."""

    arxiv_id: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    is_read: bool = False
    starred: bool = False


@dataclass(slots=True)
class WatchListEntry:
    """Author or keyword to watch for highlighting papers."""

    pattern: str
    match_type: str = "author"  # "author" | "keyword" | "title"
    case_sensitive: bool = False


@dataclass(slots=True)
class SearchBookmark:
    """Saved search query for quick access."""

    name: str
    query: str


@dataclass(slots=True)
class SessionState:
    """State to restore on next run (scroll position, filters, etc.)."""

    scroll_index: int = 0
    current_filter: str = ""
    sort_index: int = 0
    selected_ids: list[str] = field(default_factory=list)
    current_date: str | None = None  # YYYY-MM-DD format, None for non-history mode


@dataclass(slots=True)
class QueryToken:
    """Token used for advanced query parsing."""

    kind: str  # "term" or "op"
    value: str
    field: str | None = None
    phrase: bool = False


@dataclass(slots=True)
class UserConfig:
    """Complete user configuration including session state and preferences."""

    paper_metadata: dict[str, PaperMetadata] = field(default_factory=dict)
    watch_list: list[WatchListEntry] = field(default_factory=list)
    bookmarks: list[SearchBookmark] = field(default_factory=list)
    marks: dict[str, str] = field(default_factory=dict)  # letter -> arxiv_id
    session: SessionState = field(default_factory=SessionState)
    show_abstract_preview: bool = False
    bibtex_export_dir: str = ""  # Empty = use ~/arxiv-exports/
    pdf_download_dir: str = ""  # Empty = use ~/arxiv-pdfs/
    prefer_pdf_url: bool = False
    category_colors: dict[str, str] = field(default_factory=dict)
    theme: dict[str, str] = field(default_factory=dict)
    version: int = 1


# ============================================================================
# Configuration Persistence
# ============================================================================

CONFIG_APP_NAME = "arxiv-browser"
CONFIG_FILENAME = "config.json"


def get_config_path() -> Path:
    """Get the path to the configuration file.

    Uses platformdirs for cross-platform config directory:
    - Linux: ~/.config/arxiv-browser/config.json
    - macOS: ~/Library/Application Support/arxiv-browser/config.json
    - Windows: %APPDATA%/arxiv-browser/config.json
    """
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / CONFIG_FILENAME


def _config_to_dict(config: UserConfig) -> dict[str, Any]:
    """Serialize UserConfig to a JSON-compatible dictionary."""
    return {
        "version": config.version,
        "show_abstract_preview": config.show_abstract_preview,
        "bibtex_export_dir": config.bibtex_export_dir,
        "pdf_download_dir": config.pdf_download_dir,
        "prefer_pdf_url": config.prefer_pdf_url,
        "category_colors": config.category_colors,
        "theme": config.theme,
        "session": {
            "scroll_index": config.session.scroll_index,
            "current_filter": config.session.current_filter,
            "sort_index": config.session.sort_index,
            "selected_ids": config.session.selected_ids,
            "current_date": config.session.current_date,
        },
        "paper_metadata": {
            arxiv_id: {
                "notes": meta.notes,
                "tags": meta.tags,
                "is_read": meta.is_read,
                "starred": meta.starred,
            }
            for arxiv_id, meta in config.paper_metadata.items()
        },
        "watch_list": [
            {
                "pattern": entry.pattern,
                "match_type": entry.match_type,
                "case_sensitive": entry.case_sensitive,
            }
            for entry in config.watch_list
        ],
        "bookmarks": [{"name": b.name, "query": b.query} for b in config.bookmarks],
        "marks": config.marks,
    }


def _safe_get(data: dict, key: str, default: Any, expected_type: type) -> Any:
    """Safely get a value from dict with type validation.

    Returns the default if key is missing or value has wrong type.
    """
    value = data.get(key, default)
    if not isinstance(value, expected_type):
        return default
    return value


def _dict_to_config(data: dict[str, Any]) -> UserConfig:
    """Deserialize a dictionary to UserConfig with type validation."""
    # Parse session state with type validation
    session_data = data.get("session", {})
    if not isinstance(session_data, dict):
        session_data = {}

    # Handle current_date which can be str or None
    current_date_raw = session_data.get("current_date")
    current_date = current_date_raw if isinstance(current_date_raw, str) else None

    sort_index = _safe_get(session_data, "sort_index", 0, int)
    if sort_index < 0 or sort_index >= len(SORT_OPTIONS):
        sort_index = 0

    session = SessionState(
        scroll_index=_safe_get(session_data, "scroll_index", 0, int),
        current_filter=_safe_get(session_data, "current_filter", "", str),
        sort_index=sort_index,
        selected_ids=_safe_get(session_data, "selected_ids", [], list),
        current_date=current_date,
    )

    # Parse paper metadata with type validation
    paper_metadata = {}
    raw_metadata = data.get("paper_metadata", {})
    if isinstance(raw_metadata, dict):
        for arxiv_id, meta_data in raw_metadata.items():
            if not isinstance(meta_data, dict):
                continue
            paper_metadata[arxiv_id] = PaperMetadata(
                arxiv_id=arxiv_id,
                notes=_safe_get(meta_data, "notes", "", str),
                tags=_safe_get(meta_data, "tags", [], list),
                is_read=_safe_get(meta_data, "is_read", False, bool),
                starred=_safe_get(meta_data, "starred", False, bool),
            )

    # Parse watch list with type validation
    watch_list = []
    raw_watch_list = data.get("watch_list", [])
    if isinstance(raw_watch_list, list):
        for entry in raw_watch_list:
            if not isinstance(entry, dict):
                continue
            watch_list.append(
                WatchListEntry(
                    pattern=_safe_get(entry, "pattern", "", str),
                    match_type=_safe_get(entry, "match_type", "author", str),
                    case_sensitive=_safe_get(entry, "case_sensitive", False, bool),
                )
            )

    # Parse bookmarks with type validation
    bookmarks = []
    raw_bookmarks = data.get("bookmarks", [])
    if isinstance(raw_bookmarks, list):
        for b in raw_bookmarks:
            if not isinstance(b, dict):
                continue
            bookmarks.append(
                SearchBookmark(
                    name=_safe_get(b, "name", "", str),
                    query=_safe_get(b, "query", "", str),
                )
            )

    # Parse marks with type validation
    marks = data.get("marks", {})
    if not isinstance(marks, dict):
        marks = {}

    category_colors = _safe_get(data, "category_colors", {}, dict)
    if not isinstance(category_colors, dict):
        category_colors = {}
    safe_category_colors = {
        str(key): str(value)
        for key, value in category_colors.items()
        if isinstance(key, str) and isinstance(value, str)
    }

    theme = _safe_get(data, "theme", {}, dict)
    if not isinstance(theme, dict):
        theme = {}
    safe_theme = {
        str(key): str(value)
        for key, value in theme.items()
        if isinstance(key, str) and isinstance(value, str)
    }

    return UserConfig(
        paper_metadata=paper_metadata,
        watch_list=watch_list,
        bookmarks=bookmarks,
        marks=marks,
        session=session,
        show_abstract_preview=_safe_get(data, "show_abstract_preview", False, bool),
        bibtex_export_dir=_safe_get(data, "bibtex_export_dir", "", str),
        pdf_download_dir=_safe_get(data, "pdf_download_dir", "", str),
        prefer_pdf_url=_safe_get(data, "prefer_pdf_url", False, bool),
        category_colors=safe_category_colors,
        theme=safe_theme,
        version=_safe_get(data, "version", 1, int),
    )


def load_config() -> UserConfig:
    """Load configuration from disk.

    Returns default config if file doesn't exist or is corrupted.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return UserConfig()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return _dict_to_config(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        # Return default config on any parsing error
        return UserConfig()


def save_config(config: UserConfig) -> bool:
    """Save configuration to disk.

    Creates the config directory if it doesn't exist.
    Returns True on success, False on failure.
    """
    config_path = get_config_path()

    try:
        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config with pretty formatting
        data = _config_to_dict(config)
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


# Pre-compiled regex patterns for LaTeX cleaning (performance optimization)
# Each tuple is (pattern, replacement) applied in order
_LATEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Text formatting commands: \textbf{text} -> text, \emph{text} -> text
    (re.compile(r"\\text(?:tt|bf|it|rm|sf)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\emph\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:bf|it|tt|rm|sf)\{([^}]*)\}"), r"\1"),
    # Escaped dollar signs: \$ -> placeholder (restored later)
    (re.compile(r"\\\$"), _ESCAPED_DOLLAR),
    # Math mode: $x^2$ -> x^2 (extracts content, non-greedy)
    (re.compile(r"\$([^$]*)\$"), r"\1"),
    # Restore escaped dollar signs: placeholder -> $
    (re.compile(re.escape(_ESCAPED_DOLLAR)), "$"),
    # Accented characters: \'e -> é, \"a -> ä, \c{c} -> ç, etc.
    (re.compile(r"\\c\{c\}"), "ç"),
    (re.compile(r"\\c\{C\}"), "Ç"),
    (re.compile(r"\\'e"), "é"),
    (re.compile(r"\\'a"), "á"),
    (re.compile(r"\\'o"), "ó"),
    (re.compile(r"\\'i"), "í"),
    (re.compile(r"\\'u"), "ú"),
    (re.compile(r'\\"\{a\}'), "ä"),
    (re.compile(r'\\"\{o\}'), "ö"),
    (re.compile(r'\\"\{u\}'), "ü"),
    (re.compile(r"\\~n"), "ñ"),
    (re.compile(r"\\&"), "&"),
    # Generic command with braces: \foo{content} -> content
    (re.compile(r"\\[a-zA-Z]+\{([^}]*)\}"), r"\1"),
    # Standalone commands: \foo -> (removed)
    (re.compile(r"\\[a-zA-Z]+(?:\s|$)"), " "),
]

# Pre-compiled regex patterns for parsing arXiv entries
# Matches: "arXiv:2301.12345" -> captures "2301.12345"
_ARXIV_ID_PATTERN = re.compile(r"arXiv:(\S+)")
# Matches: "Date: Mon, 15 Jan 2024 (v1)" -> captures "Mon, 15 Jan 2024"
_DATE_PATTERN = re.compile(r"Date:\s*(.+?)(?:\s*\(|$)", re.MULTILINE)
# Matches multi-line title up to "Authors:" label
_TITLE_PATTERN = re.compile(r"Title:\s*(.+?)(?=\nAuthors:)", re.DOTALL)
# Matches multi-line authors up to "Categories:" label
_AUTHORS_PATTERN = re.compile(r"Authors:\s*(.+?)(?=\nCategories:)", re.DOTALL)
# Matches: "Categories: cs.AI cs.LG" -> captures "cs.AI cs.LG"
_CATEGORIES_PATTERN = re.compile(r"Categories:\s*(.+?)$", re.MULTILINE)
# Matches: "Comments: 10 pages, 5 figures" -> captures "10 pages, 5 figures"
_COMMENTS_PATTERN = re.compile(r"Comments:\s*(.+?)$", re.MULTILINE)
# Matches abstract text between \\ markers after Categories/Comments line
# Uses .*? to handle multi-line Comments fields (continuation lines)
_ABSTRACT_PATTERN = re.compile(
    r"(?:Categories|Comments):.*?\n\\\\\n(.+?)\n\\\\", re.DOTALL
)
# Matches: "( https://arxiv.org/abs/2301.12345" -> captures the URL
_URL_PATTERN = re.compile(r"\(\s*(https://arxiv\.org/abs/\S+)")
# Matches 70+ dashes used as entry separator
_ENTRY_SEPARATOR = re.compile(r"-{70,}")
# Matches 4-digit years (2000-2099) for BibTeX export
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def clean_latex(text: str) -> str:
    """Remove or convert common LaTeX commands to plain text.

    Handles nested braces by iteratively applying patterns until no more
    changes occur (e.g., \\textbf{$O(n^{2})$} -> O(n^{2})).

    Args:
        text: Input text potentially containing LaTeX commands.

    Returns:
        Plain text with LaTeX commands removed or converted.
    """
    # Short-circuit: skip regex processing for text without LaTeX markers
    if "\\" not in text and "$" not in text:
        return " ".join(text.split())

    # Iteratively apply patterns to handle nested structures
    prev_text = None
    while prev_text != text:
        prev_text = text
        for pattern, replacement in _LATEX_PATTERNS:
            text = pattern.sub(replacement, text)

    # Clean up extra whitespace
    return " ".join(text.split())


def parse_arxiv_file(filepath: Path) -> list[Paper]:
    """Parse arxiv.txt and return a list of Paper objects.

    Duplicate arXiv IDs are skipped (first occurrence is kept).
    """
    # Use errors="replace" to handle any non-UTF-8 characters gracefully
    content = filepath.read_text(encoding="utf-8", errors="replace")
    papers = []
    seen_ids: set[str] = set()  # Track seen IDs to skip duplicates

    # Split by paper separator using pre-compiled pattern
    entries = _ENTRY_SEPARATOR.split(content)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Extract arXiv ID
        arxiv_match = _ARXIV_ID_PATTERN.search(entry)
        if not arxiv_match:
            continue
        arxiv_id = arxiv_match.group(1)

        # Skip duplicate papers
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        # Extract date
        date_match = _DATE_PATTERN.search(entry)
        date = date_match.group(1).strip() if date_match else ""

        # Extract title (may span multiple lines)
        title_match = _TITLE_PATTERN.search(entry)
        if title_match:
            title = " ".join(title_match.group(1).split())
        else:
            title = ""

        # Extract authors (may span multiple lines)
        authors_match = _AUTHORS_PATTERN.search(entry)
        if authors_match:
            authors = " ".join(authors_match.group(1).split())
        else:
            authors = ""

        # Extract categories
        categories_match = _CATEGORIES_PATTERN.search(entry)
        categories = categories_match.group(1).strip() if categories_match else ""

        # Extract comments (optional)
        comments_match = _COMMENTS_PATTERN.search(entry)
        comments = comments_match.group(1).strip() if comments_match else None

        # Extract abstract (text between \\ markers)
        abstract_match = _ABSTRACT_PATTERN.search(entry)
        if abstract_match:
            abstract_raw = " ".join(abstract_match.group(1).split())
        else:
            abstract_raw = ""

        # Extract URL
        url_match = _URL_PATTERN.search(entry)
        url = url_match.group(1) if url_match else f"https://arxiv.org/abs/{arxiv_id}"

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                date=date,
                title=clean_latex(title),
                authors=clean_latex(authors),
                categories=categories,
                comments=clean_latex(comments) if comments else None,
                abstract=None,
                abstract_raw=abstract_raw,
                url=url,
            )
        )

    return papers


# History file date format (ISO 8601)
HISTORY_DATE_FORMAT = "%Y-%m-%d"


def discover_history_files(
    base_dir: Path,
    limit: int = MAX_HISTORY_FILES,
) -> list[tuple[date, Path]]:
    """Discover and sort history files by date.

    Looks for files matching YYYY-MM-DD.txt pattern in the history/ subdirectory.

    Args:
        base_dir: Base directory containing the history/ folder.
        limit: Maximum number of history files to return (default: MAX_HISTORY_FILES).
               Prevents memory issues with very large history directories.

    Returns:
        List of (date, path) tuples, sorted newest first, limited to `limit` entries.
        Empty list if history/ doesn't exist or has no matching files.
    """
    history_dir = base_dir / "history"
    if not history_dir.is_dir():
        return []

    files: list[tuple[date, Path]] = []
    for path in history_dir.glob("*.txt"):
        try:
            d = datetime.strptime(path.stem, HISTORY_DATE_FORMAT).date()
            files.append((d, path))
        except ValueError:
            continue  # Skip files that don't match YYYY-MM-DD pattern

    # Sort newest first and limit to prevent memory issues
    return sorted(files, key=lambda x: x[0], reverse=True)[:limit]


def get_pdf_download_path(paper: Paper, config: UserConfig) -> Path:
    """Get the local file path for a downloaded PDF.

    Args:
        paper: The paper to get the download path for.
        config: User configuration with optional custom download directory.

    Returns:
        Path where the PDF should be saved.
    """
    if config.pdf_download_dir:
        base_dir = Path(config.pdf_download_dir)
    else:
        base_dir = Path.home() / DEFAULT_PDF_DOWNLOAD_DIR
    return base_dir / f"{paper.arxiv_id}.pdf"


# Category color mapping (Monokai-inspired palette)
DEFAULT_CATEGORY_COLORS = {
    "cs.AI": "#f92672",  # Monokai pink
    "cs.CL": "#66d9ef",  # Monokai blue
    "cs.LG": "#a6e22e",  # Monokai green
    "cs.CV": "#e6db74",  # Monokai yellow
    "cs.SE": "#ae81ff",  # Monokai purple
    "cs.HC": "#fd971f",  # Monokai orange
    "cs.RO": "#66d9ef",  # Monokai blue
    "cs.NE": "#f92672",  # Monokai pink
    "cs.IR": "#ae81ff",  # Monokai purple
    "cs.CR": "#fd971f",  # Monokai orange
}

CATEGORY_COLORS = DEFAULT_CATEGORY_COLORS.copy()

DEFAULT_THEME = {
    "background": "#272822",
    "panel": "#1e1e1e",
    "panel_alt": "#3e3d32",
    "border": "#75715e",
    "text": "#f8f8f2",
    "muted": "#75715e",
    "accent": "#66d9ef",
    "accent_alt": "#e6db74",
    "green": "#a6e22e",
    "yellow": "#e6db74",
    "orange": "#fd971f",
    "pink": "#f92672",
    "purple": "#ae81ff",
    "highlight": "#49483e",
    "highlight_focus": "#5a5950",
    "selection": "#3d4a32",
    "selection_highlight": "#4d5a42",
    "scrollbar_background": "#3e3d32",
    "scrollbar_background_hover": "#49483e",
    "scrollbar_background_active": "#5a5950",
    "scrollbar": "#75715e",
    "scrollbar_active": "#66d9ef",
    "scrollbar_hover": "#a6e22e",
    "scrollbar_corner_color": "#3e3d32",
}

THEME_COLORS = DEFAULT_THEME.copy()


def parse_arxiv_date(date_str: str) -> datetime:
    """Parse arXiv date string to datetime for proper sorting.

    Args:
        date_str: Date string like "Mon, 15 Jan 2024"

    Returns:
        Parsed datetime object, or datetime.min for malformed dates.
    """
    cleaned = date_str.strip()
    if not cleaned:
        return datetime.min

    try:
        return datetime.strptime(cleaned, ARXIV_DATE_FORMAT)
    except ValueError:
        pass

    match = _ARXIV_DATE_PREFIX_PATTERN.search(cleaned)
    if match:
        try:
            return datetime.strptime(match.group(1), ARXIV_DATE_FORMAT)
        except ValueError:
            pass

    return datetime.min  # Fallback for malformed dates


@functools.lru_cache(maxsize=256)
def format_categories(categories: str) -> str:
    """Format categories with colors. Results are automatically cached via lru_cache."""
    parts = []
    for cat in categories.split():
        color = CATEGORY_COLORS.get(cat, DEFAULT_CATEGORY_COLOR)
        parts.append(f"[{color}]{cat}[/]")
    return " ".join(parts)


# ============================================================================
# Paper Similarity Functions
# ============================================================================


def _extract_keywords(text: str | None, min_length: int = 4) -> set[str]:
    """Extract significant keywords from text, filtering stopwords."""
    if not text:
        return set()
    words = set()
    for word in text.lower().split():
        # Remove non-alphanumeric characters
        clean = "".join(c for c in word if c.isalnum())
        if len(clean) >= min_length and clean not in STOPWORDS:
            words.add(clean)
    return words


def _extract_author_lastnames(authors: str) -> set[str]:
    """Extract last names from author string."""
    lastnames = set()
    # Split by common separators
    for author in re.split(r",|(?:\s+and\s+)", authors):
        parts = author.strip().split()
        if parts:
            # Last word is typically the last name
            lastname = parts[-1].lower()
            # Remove non-alphanumeric
            lastname = "".join(c for c in lastname if c.isalnum())
            if lastname:
                lastnames.add(lastname)
    return lastnames


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_paper_similarity(
    paper_a: Paper,
    paper_b: Paper,
    abstract_a: str | None = None,
    abstract_b: str | None = None,
) -> float:
    """Compute weighted similarity score between two papers.

    Weights:
    - Category overlap: 40%
    - Author overlap: 30%
    - Title keywords: 20%
    - Abstract keywords: 10%

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if paper_a.arxiv_id == paper_b.arxiv_id:
        return 1.0

    # Category similarity (40%)
    cats_a = set(paper_a.categories.split())
    cats_b = set(paper_b.categories.split())
    cat_sim = _jaccard_similarity(cats_a, cats_b)

    # Author similarity (30%)
    authors_a = _extract_author_lastnames(paper_a.authors)
    authors_b = _extract_author_lastnames(paper_b.authors)
    author_sim = _jaccard_similarity(authors_a, authors_b)

    # Title keyword similarity (20%)
    title_kw_a = _extract_keywords(paper_a.title)
    title_kw_b = _extract_keywords(paper_b.title)
    title_sim = _jaccard_similarity(title_kw_a, title_kw_b)

    # Abstract keyword similarity (10%)
    if abstract_a is None:
        abstract_a = paper_a.abstract or ""
    if abstract_b is None:
        abstract_b = paper_b.abstract or ""
    abstract_kw_a = _extract_keywords(abstract_a)
    abstract_kw_b = _extract_keywords(abstract_b)
    abstract_sim = _jaccard_similarity(abstract_kw_a, abstract_kw_b)

    # Weighted sum
    return 0.4 * cat_sim + 0.3 * author_sim + 0.2 * title_sim + 0.1 * abstract_sim


def find_similar_papers(
    target: Paper,
    all_papers: list[Paper],
    top_n: int = SIMILARITY_TOP_N,
    metadata: dict[str, PaperMetadata] | None = None,
    abstract_lookup: Callable[[Paper], str] | None = None,
) -> list[tuple[Paper, float]]:
    """Find the top N most similar papers to the target.

    Args:
        target: The paper to find similarities for
        all_papers: List of all papers to search
        top_n: Number of similar papers to return

    Returns:
        List of (paper, score) tuples, sorted by score descending
    """
    scored = []
    if abstract_lookup is None:
        abstract_lookup = lambda paper: paper.abstract or ""

    newest_date = datetime.min
    for paper in all_papers:
        paper_date = parse_arxiv_date(paper.date)
        if paper_date > newest_date:
            newest_date = paper_date

    def metadata_boost(arxiv_id: str) -> float:
        if not metadata:
            return 0.0
        entry = metadata.get(arxiv_id)
        if not entry:
            return 0.0
        boost = 0.0
        if entry.starred:
            boost += SIMILARITY_STARRED_BOOST
        if entry.is_read:
            boost -= SIMILARITY_READ_PENALTY
        else:
            boost += SIMILARITY_UNREAD_BOOST
        return boost

    def recency_score(paper: Paper) -> float:
        if newest_date == datetime.min:
            return 0.0
        paper_date = parse_arxiv_date(paper.date)
        if paper_date == datetime.min:
            return 0.0
        age_days = max(0, (newest_date - paper_date).days)
        return max(0.0, 1.0 - (age_days / SIMILARITY_RECENCY_DAYS))

    for paper in all_papers:
        if paper.arxiv_id == target.arxiv_id:
            continue
        score = compute_paper_similarity(
            target,
            paper,
            abstract_lookup(target),
            abstract_lookup(paper),
        )
        score += SIMILARITY_RECENCY_WEIGHT * recency_score(paper)
        score += metadata_boost(paper.arxiv_id)
        score = max(0.0, min(1.0, score))
        if score > 0:
            scored.append((paper, score))

    # Sort by score descending and take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


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
        parts = [
            f"[dim]{self.paper.arxiv_id}[/]",
            format_categories(self.paper.categories),
        ]

        # Show tags if present
        if self._metadata and self._metadata.tags:
            tag_str = " ".join(
                f"[{THEME_COLORS['purple']}]#{escape_rich_text(tag)}[/]"
                for tag in self._metadata.tags
            )
            parts.append(tag_str)

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
        title_widget = self.query_one(".paper-title", Static)
        authors_widget = self.query_one(".paper-authors", Static)
        meta_widget = self.query_one(".paper-meta", Static)
        title_widget.update(self._get_title_text())
        authors_widget.update(self._get_authors_text())
        meta_widget.update(self._get_meta_text())
        if self._show_preview:
            preview_widget = self.query_one(".paper-preview", Static)
            preview_widget.update(self._get_preview_text())

    def compose(self) -> ComposeResult:
        yield Static(self._get_title_text(), classes="paper-title")
        yield Static(self._get_authors_text(), classes="paper-authors")
        yield Static(self._get_meta_text(), classes="paper-meta")
        if self._show_preview:
            yield Static(self._get_preview_text(), classes="paper-preview")


class PaperDetails(Static):
    """Widget to display full paper details."""

    def __init__(self) -> None:
        super().__init__()
        self._paper: Paper | None = None

    def update_paper(
        self, paper: Paper | None, abstract_text: str | None = None
    ) -> None:
        """Update the displayed paper details."""
        self._paper = paper
        if paper is None:
            self.update("[dim italic]Select a paper to view details[/]")
            return

        loading = abstract_text is None and paper.abstract is None
        if abstract_text is None:
            abstract_text = paper.abstract or ""

        lines = []
        safe_title = escape_rich_text(paper.title)
        safe_authors = escape_rich_text(paper.authors)
        safe_date = escape_rich_text(paper.date)
        safe_comments = escape_rich_text(paper.comments or "")
        safe_abstract = escape_rich_text(abstract_text)
        safe_url = escape_rich_text(paper.url)

        # Title section (Monokai foreground)
        lines.append(f"[bold {THEME_COLORS['text']}]{safe_title}[/]")
        lines.append("")

        # Metadata section (Monokai blue for labels, purple for values)
        lines.append(
            f"[bold {THEME_COLORS['accent']}]arXiv:[/] [{THEME_COLORS['purple']}]{paper.arxiv_id}[/]"
        )
        lines.append(f"[bold {THEME_COLORS['accent']}]Date:[/] {safe_date}")
        lines.append(
            f"[bold {THEME_COLORS['accent']}]Categories:[/] {format_categories(paper.categories)}"
        )
        if paper.comments:
            lines.append(
                f"[bold {THEME_COLORS['accent']}]Comments:[/] [dim]{safe_comments}[/]"
            )
        lines.append("")

        # Authors section (Monokai green)
        lines.append(f"[bold {THEME_COLORS['green']}]Authors[/]")
        lines.append(f"[{THEME_COLORS['text']}]{safe_authors}[/]")
        lines.append("")

        # Abstract section (Monokai orange)
        lines.append(f"[bold {THEME_COLORS['orange']}]Abstract[/]")
        if loading:
            lines.append("[dim italic]Loading abstract...[/]")
        elif abstract_text:
            lines.append(f"[{THEME_COLORS['text']}]{safe_abstract}[/]")
        else:
            lines.append("[dim italic]No abstract available[/]")
        lines.append("")

        # URL section (Monokai pink/red for label, blue for URL)
        lines.append(
            f"[bold {THEME_COLORS['pink']}]URL:[/] [{THEME_COLORS['accent']}]{safe_url}[/]"
        )

        self.update("\n".join(lines))

    @property
    def paper(self) -> Paper | None:
        return self._paper


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
        background: #272822;
        border: round #66d9ef;
        padding: 1 2;
    }

    #notes-title {
        text-style: bold;
        color: #e6db74;
        margin-bottom: 1;
    }

    #notes-textarea {
        height: 1fr;
        background: #1e1e1e;
        border: round #75715e;
    }

    #notes-textarea:focus {
        border: round #66d9ef;
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
        background: #272822;
        border: round #a6e22e;
        padding: 1 2;
    }

    #tags-title {
        text-style: bold;
        color: #a6e22e;
        margin-bottom: 1;
    }

    #tags-help {
        color: #75715e;
        margin-bottom: 1;
    }

    #tags-input {
        width: 100%;
        background: #1e1e1e;
        border: round #75715e;
    }

    #tags-input:focus {
        border: round #a6e22e;
    }

    #tags-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #tags-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, arxiv_id: str, current_tags: list[str] | None = None) -> None:
        super().__init__()
        self._arxiv_id = arxiv_id
        self._current_tags = current_tags or []

    def compose(self) -> ComposeResult:
        with Vertical(id="tags-dialog"):
            yield Label(f"Tags for {self._arxiv_id}", id="tags-title")
            yield Label(
                "Separate tags with commas (e.g., to-read, llm, important)",
                id="tags-help",
            )
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
        background: #272822;
        border: round #66d9ef;
        padding: 1 2;
    }

    #watch-title {
        text-style: bold;
        color: #66d9ef;
        margin-bottom: 1;
    }

    #watch-body {
        height: 1fr;
    }

    #watch-list {
        width: 2fr;
        height: 1fr;
        background: #1e1e1e;
        border: round #75715e;
        margin-right: 2;
    }

    #watch-form {
        width: 1fr;
        height: 1fr;
    }

    #watch-form Label {
        color: #75715e;
        margin-top: 1;
    }

    #watch-pattern,
    #watch-type {
        width: 100%;
        background: #1e1e1e;
        border: round #75715e;
    }

    #watch-pattern:focus,
    #watch-type:focus {
        border: round #66d9ef;
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
        index = list_view.index or 0
        self._entries[index] = entry
        self._refresh_list()

    @on(Button.Pressed, "#watch-delete")
    def on_delete_pressed(self) -> None:
        list_view = self.query_one("#watch-list", ListView)
        if not isinstance(list_view.highlighted_child, WatchListItem):
            self.notify("Select a watch entry to delete", title="Watch")
            return
        index = list_view.index or 0
        self._entries.pop(index)
        self._refresh_list()

    @on(Button.Pressed, "#watch-save")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#watch-cancel")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


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
        background: #272822;
        border: round #fd971f;
        padding: 1 2;
    }

    #recommendations-title {
        text-style: bold;
        color: #fd971f;
        margin-bottom: 1;
    }

    #recommendations-list {
        height: 1fr;
        background: #1e1e1e;
        border: round #75715e;
    }

    #recommendations-list > ListItem {
        padding: 1;
        border-bottom: dashed #3e3d32;
    }

    #recommendations-list > ListItem.--highlight {
        background: #49483e;
    }

    .rec-title {
        color: #f8f8f2;
    }

    .rec-meta {
        color: #75715e;
    }

    .rec-score {
        color: #a6e22e;
        text-style: bold;
    }

    #recommendations-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(
        self, target_paper: Paper, similar_papers: list[tuple[Paper, float]]
    ) -> None:
        super().__init__()
        self._target_paper = target_paper
        self._similar_papers = similar_papers

    def compose(self) -> ComposeResult:
        with Vertical(id="recommendations-dialog"):
            truncated_title = truncate_text(
                self._target_paper.title, RECOMMENDATION_TITLE_MAX_LEN
            )
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


class BookmarkTabBar(Horizontal):
    """Horizontal bar displaying search bookmarks as numbered tabs."""

    DEFAULT_CSS = """
    BookmarkTabBar {
        height: auto;
        padding: 0 1;
        background: #3e3d32;
        border-bottom: solid #75715e;
    }

    BookmarkTabBar .bookmark-tab {
        padding: 0 2;
        margin-right: 1;
        color: #75715e;
    }

    BookmarkTabBar .bookmark-tab:hover {
        color: #f8f8f2;
    }

    BookmarkTabBar .bookmark-tab.active {
        color: #e6db74;
        text-style: bold;
    }

    BookmarkTabBar .bookmark-add {
        color: #75715e;
        padding: 0 1;
    }

    BookmarkTabBar .bookmark-add:hover {
        color: #a6e22e;
    }
    """

    def __init__(self, bookmarks: list[SearchBookmark], active_index: int = -1) -> None:
        super().__init__()
        self._bookmarks = bookmarks
        self._active_index = active_index

    def compose(self) -> ComposeResult:
        for i, bookmark in enumerate(self._bookmarks[:9]):  # Max 9 bookmarks
            classes = (
                "bookmark-tab active" if i == self._active_index else "bookmark-tab"
            )
            yield Label(
                f"{i + 1}: {bookmark.name}", classes=classes, id=f"bookmark-{i}"
            )
        yield Label("[+]", classes="bookmark-add", id="bookmark-add")

    async def update_bookmarks(
        self, bookmarks: list[SearchBookmark], active_index: int = -1
    ) -> None:
        """Update the displayed bookmarks."""
        self._bookmarks = bookmarks
        self._active_index = active_index
        await self.remove_children()
        for i, bookmark in enumerate(bookmarks[:9]):
            classes = (
                "bookmark-tab active" if i == self._active_index else "bookmark-tab"
            )
            self.mount(
                Label(f"{i + 1}: {bookmark.name}", classes=classes, id=f"bookmark-{i}")
            )
        self.mount(Label("[+]", classes="bookmark-add", id="bookmark-add"))


class ArxivBrowser(App):
    """A TUI application to browse arXiv papers."""

    TITLE = "arXiv Paper Browser"

    # Monokai color theme - using hardcoded colors to avoid conflicts with Textual's theme system
    # Colors: background=#272822, panel=#1e1e1e, panel-alt=#3e3d32, border=#75715e,
    #         text=#f8f8f2, muted=#75715e, accent=#66d9ef, accent-alt=#e6db74,
    #         highlight=#49483e, highlight-focus=#5a5950, selection=#3d4a32, selection-highlight=#4d5a42
    CSS = """
    Screen {
        background: #272822;
    }

    Header {
        background: #3e3d32;
        color: #f8f8f2;
    }

    Footer {
        background: #3e3d32;
    }

    #main-container {
        height: 1fr;
    }

    #left-pane {
        width: 2fr;
        min-width: 50;
        max-width: 100;
        height: 100%;
        border: round #75715e;
        background: #1e1e1e;
    }

    #right-pane {
        width: 3fr;
        height: 100%;
        border: round #75715e;
        background: #1e1e1e;
    }

    #list-header {
        padding: 1 2;
        background: #3e3d32;
        color: #66d9ef;
        text-style: bold;
        border-bottom: solid #75715e;
    }

    #details-header {
        padding: 1 2;
        background: #3e3d32;
        color: #e6db74;
        text-style: bold;
        border-bottom: solid #75715e;
    }

    #paper-list {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #details-scroll {
        height: 1fr;
        padding: 1 2;
    }

    #search-container {
        height: auto;
        padding: 1;
        background: #3e3d32;
        display: none;
    }

    #search-container.visible {
        display: block;
    }

    #search-input {
        width: 100%;
        border: round #66d9ef;
        background: #272822;
    }

    #search-input:focus {
        border: round #e6db74;
    }

    PaperListItem {
        padding: 1 1;
        height: auto;
        border-bottom: dashed #3e3d32;
    }

    PaperListItem:hover {
        background: #3e3d32;
    }

    PaperListItem.-highlight {
        background: #49483e;
    }

    ListView > ListItem.--highlight {
        background: #49483e;
    }

    ListView:focus > ListItem.--highlight {
        background: #5a5950;
    }

    .paper-title {
        color: #f8f8f2;
        text-style: bold;
    }

    .paper-authors {
        color: #75715e;
    }

    .paper-meta {
        color: #75715e;
        margin-top: 0;
    }

    .paper-preview {
        color: #75715e;
        margin-top: 0;
        padding-left: 2;
    }

    PaperListItem.selected {
        background: #3d4a32;
    }

    PaperListItem.selected.--highlight {
        background: #4d5a42;
    }

    PaperDetails {
        padding: 0;
    }

    VerticalScroll {
        scrollbar-background: #3e3d32;
        scrollbar-color: #75715e;
        scrollbar-color-hover: #75715e;
        scrollbar-color-active: #66d9ef;
    }

    #status-bar {
        padding: 0 2 1 2;
        color: #75715e;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "toggle_search", "Search"),
        Binding("escape", "cancel_search", "Cancel", show=False),
        Binding("o", "open_url", "Open Selected"),
        Binding("P", "open_pdf", "Open PDF", show=False),
        Binding("c", "copy_selected", "Copy"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("space", "toggle_select", "Select", show=False),
        Binding("a", "select_all", "Select All"),
        Binding("u", "clear_selection", "Clear Selection"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        # Phase 2: Read/Star status and Notes/Tags
        Binding("r", "toggle_read", "Read"),
        Binding("x", "toggle_star", "Star"),
        Binding("n", "edit_notes", "Notes"),
        Binding("t", "edit_tags", "Tags"),
        # Phase 3: Watch list
        Binding("w", "toggle_watch_filter", "Watch"),
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
        Binding("ctrl+b", "add_bookmark", "Add Bookmark"),
        Binding("ctrl+shift+b", "remove_bookmark", "Del Bookmark", show=False),
        # Phase 5: Abstract preview
        Binding("p", "toggle_preview", "Preview"),
        # Phase 7: Vim-style marks
        Binding("m", "start_mark", "Mark", show=False),
        Binding("apostrophe", "start_goto_mark", "Goto Mark", show=False),
        # Phase 8: Export features
        Binding("b", "copy_bibtex", "BibTeX"),
        Binding("B", "export_bibtex_file", "Export BibTeX", show=False),
        Binding("M", "export_markdown", "Markdown", show=False),
        Binding("d", "download_pdf", "Download"),
        # Phase 9: Paper similarity
        Binding("R", "show_similar", "Similar"),
        # History mode: date navigation
        Binding("bracketleft", "prev_date", "Older", show=False),
        Binding("bracketright", "next_date", "Newer", show=False),
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
        self.all_papers = papers
        self.filtered_papers = papers.copy()
        # Build O(1) lookup dict for papers by arxiv_id
        self._papers_by_id: dict[str, Paper] = {p.arxiv_id: p for p in papers}
        self.selected_ids: set[str] = set()  # Track selected arxiv_ids
        self._search_timer: Timer | None = None
        self._pending_query: str = ""
        self._sort_index: int = 0  # Index into SORT_OPTIONS

        # Configuration and persistence
        self._config = config or UserConfig()
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

        # PDF download state
        self._download_queue: deque[Paper] = deque()
        self._downloading: set[str] = set()  # arxiv_ids currently downloading
        self._download_results: dict[str, bool] = {}  # arxiv_id -> success
        self._download_total: int = 0  # Total papers in current batch

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-pane"):
                yield Label(f" Papers ({len(self.all_papers)} total)", id="list-header")
                yield BookmarkTabBar(
                    self._config.bookmarks, self._active_bookmark_index
                )
                with Vertical(id="search-container"):
                    yield Input(
                        placeholder=(
                            " Filter: text, author:, title:, cat:, tag:, unread, starred, AND/OR/NOT"
                        ),
                        id="search-input",
                    )
                yield ListView(
                    *[
                        PaperListItem(
                            p,
                            abstract_text=self._get_abstract_text(p, allow_async=True),
                            highlight_terms=self._highlight_terms,
                        )
                        for p in self.filtered_papers
                    ],
                    id="paper-list",
                )
                yield Label("", id="status-bar")
            with Vertical(id="right-pane"):
                yield Label(" Paper Details", id="details-header")
                with VerticalScroll(id="details-scroll"):
                    yield PaperDetails()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted. Restores session state if enabled."""
        # Set subtitle with date info if in history mode
        current_date = self._get_current_date()
        if current_date:
            self.sub_title = f"{len(self.all_papers)} papers · {current_date.strftime(HISTORY_DATE_FORMAT)}"
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
                self._apply_filter(session.current_filter)

            # Restore scroll position
            list_view = self.query_one("#paper-list", ListView)
            if list_view.children:
                # Clamp index to valid range
                index = min(session.scroll_index, len(list_view.children) - 1)
                list_view.index = max(0, index)
        else:
            # Default: select first item if available
            list_view = self.query_one("#paper-list", ListView)
            if list_view.children:
                list_view.index = 0
        self._update_status_bar()
        # Focus the paper list so key bindings work
        self.query_one("#paper-list", ListView).focus()

    def on_unmount(self) -> None:
        """Called when app is unmounted. Saves session state and cleans up timers.

        Uses atomic swap pattern to avoid race conditions with timer callbacks.
        """
        # Save session state before exit
        self._save_session_state()

        # Clean up timer
        timer = self._search_timer
        self._search_timer = None
        if timer is not None:
            timer.stop()

    def _apply_category_overrides(self) -> None:
        """Apply category color overrides from config."""
        CATEGORY_COLORS.clear()
        CATEGORY_COLORS.update(DEFAULT_CATEGORY_COLORS)
        CATEGORY_COLORS.update(self._config.category_colors)

    def _apply_theme_overrides(self) -> None:
        """Apply theme overrides from config to markup colors (THEME_COLORS dict).

        Note: CSS variables are handled by Textual's built-in theme system.
        THEME_COLORS is only used for Rich markup styling in the UI.
        """
        THEME_COLORS.clear()
        THEME_COLORS.update(DEFAULT_THEME)
        THEME_COLORS.update(self._config.theme)

    def _schedule_abstract_load(self, paper: Paper) -> None:
        """Schedule an abstract load with concurrency limits."""
        if (
            paper.arxiv_id in self._abstract_loading
            or paper.arxiv_id in self._abstract_pending_ids
        ):
            return
        if len(self._abstract_loading) < MAX_ABSTRACT_LOADS:
            self._abstract_loading.add(paper.arxiv_id)
            asyncio.create_task(self._load_abstract_async(paper))
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
            asyncio.create_task(self._load_abstract_async(paper))

    def _get_abstract_text(self, paper: Paper, allow_async: bool) -> str | None:
        """Return cached abstract text, scheduling async load if needed."""
        cached = self._abstract_cache.get(paper.arxiv_id)
        if cached is not None:
            return cached
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
        cleaned = await asyncio.to_thread(clean_latex, paper.abstract_raw)
        self._abstract_cache[paper.arxiv_id] = cleaned
        # Only update if not already set (idempotent to avoid race conditions)
        if paper.abstract is None:
            paper.abstract = cleaned
        self._abstract_loading.discard(paper.arxiv_id)
        self._drain_abstract_queue()
        self._update_abstract_display(paper.arxiv_id)

    def _update_abstract_display(self, arxiv_id: str) -> None:
        try:
            details = self.query_one(PaperDetails)
            if details.paper and details.paper.arxiv_id == arxiv_id:
                abstract_text = self._abstract_cache.get(arxiv_id, "")
                details.update_paper(details.paper, abstract_text)
            list_view = self.query_one("#paper-list", ListView)
            for item in list_view.children:
                if isinstance(item, PaperListItem) and item.paper.arxiv_id == arxiv_id:
                    item.set_abstract_text(self._abstract_cache.get(arxiv_id))
        except NoMatches:
            return

    def _save_session_state(self) -> None:
        """Save current session state to config.

        Handles the case where DOM widgets may already be destroyed during unmount.
        """
        # Get current date for history mode
        current_date = self._get_current_date()
        current_date_str = (
            current_date.strftime(HISTORY_DATE_FORMAT) if current_date else None
        )

        try:
            list_view = self.query_one("#paper-list", ListView)
            search_input = self.query_one("#search-input", Input)

            self._config.session = SessionState(
                scroll_index=list_view.index or 0,
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

        save_config(self._config)

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle paper selection."""
        if isinstance(event.item, PaperListItem):
            details = self.query_one(PaperDetails)
            abstract_text = self._get_abstract_text(event.item.paper, allow_async=True)
            details.update_paper(event.item.paper, abstract_text)

    @on(ListView.Highlighted)
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle paper highlight (keyboard navigation)."""
        if isinstance(event.item, PaperListItem):
            details = self.query_one(PaperDetails)
            abstract_text = self._get_abstract_text(event.item.paper, allow_async=True)
            details.update_paper(event.item.paper, abstract_text)

    def action_toggle_search(self) -> None:
        """Toggle search input visibility."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
        else:
            container.add_class("visible")
            self.query_one("#search-input", Input).focus()

    def action_cancel_search(self) -> None:
        """Cancel search and hide input."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
            self._apply_filter("")

    def action_cursor_down(self) -> None:
        """Move cursor down (vim-style j key)."""
        list_view = self.query_one("#paper-list", ListView)
        list_view.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up (vim-style k key)."""
        list_view = self.query_one("#paper-list", ListView)
        list_view.action_cursor_up()

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        self._apply_filter(event.value)
        # Hide search after submission
        self.query_one("#search-container").remove_class("visible")
        # Focus the list
        self.query_one("#paper-list", ListView).focus()

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

    def _format_header_text(self, query: str = "") -> str:
        """Format the header text with paper count, date info, and selection info."""
        selection_info = (
            f" [{len(self.selected_ids)} selected]" if self.selected_ids else ""
        )
        sort_info = f" [dim]sorted by {SORT_OPTIONS[self._sort_index]}[/]"

        # Date info for history mode
        date_info = ""
        if self._is_history_mode():
            current_date = self._get_current_date()
            if current_date:
                pos = self._current_date_index + 1
                total = len(self._history_files)
                date_info = f" · [#66d9ef]{current_date.strftime(HISTORY_DATE_FORMAT)}[/] [{pos}/{total}]"

        if query:
            return f" Papers ({len(self.filtered_papers)}/{len(self.all_papers)}){date_info}{selection_info}{sort_info}"
        return f" Papers ({len(self.all_papers)} total){date_info}{selection_info}{sort_info}"

    def _tokenize_query(self, query: str) -> list[QueryToken]:
        """Tokenize a query string into terms and operators."""
        tokens: list[QueryToken] = []
        i = 0
        query_len = len(query)
        while i < query_len:
            if query[i].isspace():
                i += 1
                continue
            if query[i] == '"':
                i += 1
                start = i
                while i < query_len and query[i] != '"':
                    i += 1
                value = query[start:i]
                tokens.append(QueryToken(kind="term", value=value, phrase=True))
                i += 1
                continue
            start = i
            while i < query_len and not query[i].isspace() and query[i] != ":":
                i += 1
            if i < query_len and query[i] == ":":
                field = query[start:i].lower()
                if field in {"title", "author", "abstract", "cat", "tag"}:
                    i += 1
                    if i < query_len and query[i] == '"':
                        i += 1
                        value_start = i
                        while i < query_len and query[i] != '"':
                            i += 1
                        value = query[value_start:i]
                        tokens.append(
                            QueryToken(
                                kind="term", value=value, field=field, phrase=True
                            )
                        )
                        i += 1
                    else:
                        value_start = i
                        while i < query_len and not query[i].isspace():
                            i += 1
                        value = query[value_start:i]
                        tokens.append(QueryToken(kind="term", value=value, field=field))
                    continue
            while i < query_len and not query[i].isspace():
                i += 1
            raw = query[start:i]
            upper = raw.upper()
            if upper in {"AND", "OR", "NOT"}:
                tokens.append(QueryToken(kind="op", value=upper))
            else:
                tokens.append(QueryToken(kind="term", value=raw))
        return tokens

    def _is_advanced_query(self, tokens: list[QueryToken]) -> bool:
        return any(
            tok.kind == "op"
            or tok.field
            or tok.phrase
            or tok.value.lower() in {"unread", "starred"}
            for tok in tokens
        )

    def _build_highlight_terms(self, tokens: list[QueryToken]) -> None:
        highlight = {"title": [], "author": [], "abstract": []}
        for token in tokens:
            if token.kind != "term":
                continue
            if token.value.lower() in {"unread", "starred"}:
                continue
            if token.field == "title":
                highlight["title"].append(token.value)
            elif token.field == "author":
                highlight["author"].append(token.value)
            elif token.field == "abstract":
                highlight["abstract"].append(token.value)
            elif token.field is None:
                highlight["title"].append(token.value)
                highlight["author"].append(token.value)
        self._highlight_terms = highlight

    def _insert_implicit_and(self, tokens: list[QueryToken]) -> list[QueryToken]:
        result: list[QueryToken] = []
        prev_was_term = False
        for token in tokens:
            token_is_term_start = token.kind == "term" or token.value == "NOT"
            if prev_was_term and token_is_term_start:
                result.append(QueryToken(kind="op", value="AND"))
            result.append(token)
            prev_was_term = token.kind == "term"
        return result

    def _to_rpn(self, tokens: list[QueryToken]) -> list[QueryToken]:
        output: list[QueryToken] = []
        ops: list[QueryToken] = []
        precedence = {"OR": 1, "AND": 2, "NOT": 3}
        for token in tokens:
            if token.kind == "term":
                output.append(token)
                continue
            while ops and precedence[ops[-1].value] >= precedence[token.value]:
                output.append(ops.pop())
            ops.append(token)
        while ops:
            output.append(ops.pop())
        return output

    def _matches_advanced_query(self, paper: Paper, rpn: list[QueryToken]) -> bool:
        if not rpn:
            return True
        stack: list[bool] = []
        for token in rpn:
            if token.kind == "term":
                stack.append(self._match_query_term(paper, token))
                continue
            if token.value == "NOT":
                value = stack.pop() if stack else False
                stack.append(not value)
            else:
                right = stack.pop() if stack else False
                left = stack.pop() if stack else False
                if token.value == "AND":
                    stack.append(left and right)
                else:
                    stack.append(left or right)
        return stack[-1] if stack else True

    def _match_query_term(self, paper: Paper, token: QueryToken) -> bool:
        value = token.value.strip()
        if not value:
            return True
        value_lower = value.lower()
        if token.field == "cat":
            return value_lower in paper.categories.lower()
        if token.field == "tag":
            metadata = self._config.paper_metadata.get(paper.arxiv_id)
            if not metadata:
                return False
            return any(value_lower in tag.lower() for tag in metadata.tags)
        if token.field == "title":
            return value_lower in paper.title.lower()
        if token.field == "author":
            return value_lower in paper.authors.lower()
        if token.field == "abstract":
            abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
            return value_lower in abstract_text.lower()
        if value_lower == "unread":
            metadata = self._config.paper_metadata.get(paper.arxiv_id)
            return not metadata or not metadata.is_read
        if value_lower == "starred":
            metadata = self._config.paper_metadata.get(paper.arxiv_id)
            return bool(metadata and metadata.starred)
        haystack = f"{paper.title} {paper.authors}".lower()
        return value_lower in haystack

    def _filter_by_category(self, category: str) -> list[Paper]:
        """Filter papers by category substring match."""
        category_lower = category.lower()
        return [p for p in self.all_papers if category_lower in p.categories.lower()]

    def _filter_by_tag(self, tag: str) -> list[Paper]:
        """Filter papers that have the specified tag."""
        tag_lower = tag.lower()
        return [
            p
            for p in self.all_papers
            if p.arxiv_id in self._config.paper_metadata
            and tag_lower
            in [t.lower() for t in self._config.paper_metadata[p.arxiv_id].tags]
        ]

    def _filter_unread(self) -> list[Paper]:
        """Filter to show only unread papers."""
        return [
            p
            for p in self.all_papers
            if p.arxiv_id not in self._config.paper_metadata
            or not self._config.paper_metadata[p.arxiv_id].is_read
        ]

    def _filter_starred(self) -> list[Paper]:
        """Filter to show only starred papers."""
        return [
            p
            for p in self.all_papers
            if p.arxiv_id in self._config.paper_metadata
            and self._config.paper_metadata[p.arxiv_id].starred
        ]

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

        # Clear match scores by default (only fuzzy search populates them)
        self._match_scores.clear()

        if not query:
            self.filtered_papers = self.all_papers.copy()
            self._highlight_terms = {"title": [], "author": [], "abstract": []}
        else:
            tokens = self._tokenize_query(query)
            self._build_highlight_terms(tokens)
            if self._is_advanced_query(tokens):
                tokens = self._insert_implicit_and(tokens)
                rpn = self._to_rpn(tokens)
                self.filtered_papers = [
                    paper
                    for paper in self.all_papers
                    if self._matches_advanced_query(paper, rpn)
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

        # Update header with current query context
        self.query_one("#list-header", Label).update(self._format_header_text(query))
        self._update_status_bar()

    def action_toggle_select(self) -> None:
        """Toggle selection of the currently highlighted paper."""
        list_view = self.query_one("#paper-list", ListView)
        if list_view.highlighted_child is None:
            return

        item = list_view.highlighted_child
        if isinstance(item, PaperListItem):
            new_state = item.toggle_selected()
            if new_state:
                self.selected_ids.add(item.paper.arxiv_id)
            else:
                self.selected_ids.discard(item.paper.arxiv_id)
            self._update_header()

    def action_select_all(self) -> None:
        """Select all currently visible papers."""
        list_view = self.query_one("#paper-list", ListView)
        for item in list_view.children:
            if isinstance(item, PaperListItem):
                item.set_selected(True)
                self.selected_ids.add(item.paper.arxiv_id)
        self._update_header()

    def action_clear_selection(self) -> None:
        """Clear all selections."""
        list_view = self.query_one("#paper-list", ListView)
        for item in list_view.children:
            if isinstance(item, PaperListItem):
                item.set_selected(False)
        self.selected_ids.clear()
        self._update_header()

    def _sort_papers(self) -> None:
        """Sort filtered_papers according to current sort order."""
        sort_key = SORT_OPTIONS[self._sort_index]
        if sort_key == "title":
            self.filtered_papers.sort(key=lambda p: p.title.lower())
        elif sort_key == "date":
            # Sort by date descending (newest first) using proper datetime parsing
            self.filtered_papers.sort(
                key=lambda p: parse_arxiv_date(p.date), reverse=True
            )
        elif sort_key == "arxiv_id":
            # Sort by arxiv_id descending (newest first)
            self.filtered_papers.sort(key=lambda p: p.arxiv_id, reverse=True)

    def _refresh_list_view(self) -> None:
        """Refresh the list view with current filtered papers.

        This method handles clearing, repopulating, and selecting the first item.
        Selection state is preserved based on selected_ids.
        Paper metadata (read/star/tags), watch status, and preview are passed to each item.
        """
        list_view = self.query_one("#paper-list", ListView)
        list_view.clear()

        # Create all items and mount in batch for better performance
        items = [
            PaperListItem(
                paper,
                selected=paper.arxiv_id in self.selected_ids,
                metadata=self._config.paper_metadata.get(paper.arxiv_id),
                watched=paper.arxiv_id in self._watched_paper_ids,
                show_preview=self._show_abstract_preview,
                abstract_text=self._get_abstract_text(
                    paper, allow_async=self._show_abstract_preview
                ),
                highlight_terms=self._highlight_terms,
            )
            for paper in self.filtered_papers
        ]
        if items:
            list_view.mount(*items)
            list_view.index = 0
        else:
            try:
                details = self.query_one(PaperDetails)
                details.update_paper(None)
            except NoMatches:
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

    def _get_current_paper_item(self) -> PaperListItem | None:
        """Get the currently highlighted paper list item."""
        list_view = self.query_one("#paper-list", ListView)
        if list_view.highlighted_child and isinstance(
            list_view.highlighted_child, PaperListItem
        ):
            return list_view.highlighted_child
        return None

    def action_toggle_read(self) -> None:
        """Toggle read status of the currently highlighted paper."""
        item = self._get_current_paper_item()
        if not item:
            return

        metadata = self._get_or_create_metadata(item.paper.arxiv_id)
        metadata.is_read = not metadata.is_read
        item.set_metadata(metadata)

        status = "read" if metadata.is_read else "unread"
        self.notify(f"Marked as {status}", title="Read Status")

    def action_toggle_star(self) -> None:
        """Toggle star status of the currently highlighted paper."""
        item = self._get_current_paper_item()
        if not item:
            return

        metadata = self._get_or_create_metadata(item.paper.arxiv_id)
        metadata.starred = not metadata.starred
        item.set_metadata(metadata)

        status = "starred" if metadata.starred else "unstarred"
        self.notify(f"Paper {status}", title="Star")

    def action_edit_notes(self) -> None:
        """Open notes editor for the currently highlighted paper."""
        item = self._get_current_paper_item()
        if not item:
            return

        arxiv_id = item.paper.arxiv_id
        current_notes = ""
        if arxiv_id in self._config.paper_metadata:
            current_notes = self._config.paper_metadata[arxiv_id].notes

        def on_notes_saved(notes: str | None) -> None:
            if notes is None:
                return
            metadata = self._get_or_create_metadata(arxiv_id)
            metadata.notes = notes
            # Update the item's display
            current_item = self._get_current_paper_item()
            if current_item and current_item.paper.arxiv_id == arxiv_id:
                current_item.set_metadata(metadata)
            self.notify("Notes saved", title="Notes")

        self.push_screen(NotesModal(arxiv_id, current_notes), on_notes_saved)

    def action_edit_tags(self) -> None:
        """Open tags editor for the currently highlighted paper."""
        item = self._get_current_paper_item()
        if not item:
            return

        arxiv_id = item.paper.arxiv_id
        current_tags: list[str] = []
        if arxiv_id in self._config.paper_metadata:
            current_tags = self._config.paper_metadata[arxiv_id].tags.copy()

        def on_tags_saved(tags: list[str] | None) -> None:
            if tags is None:
                return
            metadata = self._get_or_create_metadata(arxiv_id)
            metadata.tags = tags
            # Update the item's display
            current_item = self._get_current_paper_item()
            if current_item and current_item.paper.arxiv_id == arxiv_id:
                current_item.set_metadata(metadata)
            self.notify(f"Tags: {', '.join(tags) if tags else 'none'}", title="Tags")

        self.push_screen(TagsModal(arxiv_id, current_tags), on_tags_saved)

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
                if self._paper_matches_watch_entry(paper, entry):
                    self._watched_paper_ids.add(paper.arxiv_id)
                    break  # Paper already matched, no need to check more entries

    def _paper_matches_watch_entry(self, paper: Paper, entry: WatchListEntry) -> bool:
        """Check if a paper matches a watch list entry."""
        pattern = entry.pattern if entry.case_sensitive else entry.pattern.lower()

        if entry.match_type == "author":
            text = paper.authors if entry.case_sensitive else paper.authors.lower()
            return pattern in text
        elif entry.match_type == "title":
            text = paper.title if entry.case_sensitive else paper.title.lower()
            return pattern in text
        elif entry.match_type == "keyword":
            # Match against title and abstract
            if entry.case_sensitive:
                return pattern in paper.title or pattern in paper.abstract_raw
            else:
                return (
                    pattern in paper.title.lower()
                    or pattern in paper.abstract_raw.lower()
                )
        return False

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
            self._config.watch_list = entries
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
        await bookmark_bar.update_bookmarks(
            self._config.bookmarks, self._active_bookmark_index
        )

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
            self.notify(
                "Enter a search query first", title="Bookmark", severity="warning"
            )
            return

        if len(self._config.bookmarks) >= 9:
            self.notify(
                "Maximum 9 bookmarks allowed", title="Bookmark", severity="warning"
            )
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
            self.notify(
                "No active bookmark to remove", title="Bookmark", severity="warning"
            )
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
        item = self._get_current_paper_item()
        if not item:
            self.notify("No paper selected", title="Mark", severity="warning")
            return

        self._config.marks[letter] = item.paper.arxiv_id
        self.notify(f"Mark '{letter}' set on {item.paper.arxiv_id}", title="Mark")

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
        list_view = self.query_one("#paper-list", ListView)
        for i, item in enumerate(list_view.children):
            if isinstance(item, PaperListItem) and item.paper.arxiv_id == arxiv_id:
                list_view.index = i
                self.notify(f"Jumped to mark '{letter}'", title="Mark")
                return

        # Paper not in current filtered list
        self.notify(
            f"Paper not in current view (try clearing filter)",
            title="Mark",
            severity="warning",
        )

    # ========================================================================
    # Phase 8: Export Features
    # ========================================================================

    def _escape_bibtex(self, text: str) -> str:
        """Escape special characters for BibTeX."""
        replacements = [
            ("&", r"\&"),
            ("%", r"\%"),
            ("_", r"\_"),
            ("#", r"\#"),
            ("{", r"\{"),
            ("}", r"\}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def _format_authors_bibtex(self, authors: str) -> str:
        """Format authors for BibTeX (Last, First and Last, First format)."""
        # Simple heuristic: split by comma or "and", keep as-is
        # arXiv authors are typically in "First Last" format
        return self._escape_bibtex(authors)

    def _extract_year(self, date_str: str) -> str:
        """Extract year from date string, with fallback to current year.

        Args:
            date_str: Date string like "Mon, 15 Jan 2024".

        Returns:
            4-digit year string, or current year if not found.
        """
        current_year = str(datetime.now().year)

        if not date_str or not date_str.strip():
            return current_year

        # Try to find a 4-digit year (2000-2099) using pre-compiled pattern
        year_match = _YEAR_PATTERN.search(date_str)
        if year_match:
            return year_match.group(1)

        return current_year

    def _generate_citation_key(self, paper: Paper) -> str:
        """Generate a BibTeX citation key like 'smith2024attention'."""
        # Extract first author's last name
        authors = paper.authors.split(",")[0].strip()
        parts = authors.split()
        last_name = parts[-1].lower() if parts else "unknown"
        # Remove non-alphanumeric characters
        last_name = "".join(c for c in last_name if c.isalnum())

        # Extract year from date
        year = self._extract_year(paper.date)

        # First significant word from title (uses module-level STOPWORDS)
        title_words = paper.title.lower().split()
        first_word = "paper"
        for word in title_words:
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word and clean_word not in STOPWORDS:
                first_word = clean_word
                break

        return f"{last_name}{year}{first_word}"

    def _format_paper_as_bibtex(self, paper: Paper) -> str:
        """Format a paper as a BibTeX @misc entry."""
        key = self._generate_citation_key(paper)
        year = self._extract_year(paper.date)
        # Safely extract primary category (handles empty/whitespace-only strings)
        categories_list = paper.categories.split()
        primary_class = categories_list[0] if categories_list else "cs.AI"
        lines = [
            f"@misc{{{key},",
            f"  title = {{{self._escape_bibtex(paper.title)}}},",
            f"  author = {{{self._format_authors_bibtex(paper.authors)}}},",
            f"  year = {{{year}}},",
            f"  eprint = {{{paper.arxiv_id}}},",
            f"  archivePrefix = {{arXiv}},",
            f"  primaryClass = {{{primary_class}}},",
            f"  url = {{{paper.url}}},",
            "}",
        ]
        return "\n".join(lines)

    def action_copy_bibtex(self) -> None:
        """Copy selected papers as BibTeX entries to clipboard."""
        papers = self._get_papers_to_export()
        if not papers:
            self.notify("No paper selected", title="BibTeX", severity="warning")
            return

        bibtex_entries = [self._format_paper_as_bibtex(p) for p in papers]
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
        papers = self._get_papers_to_export()
        if not papers:
            self.notify("No paper selected", title="Export", severity="warning")
            return

        # Determine export directory
        export_dir = Path(
            self._config.bibtex_export_dir or Path.home() / DEFAULT_BIBTEX_EXPORT_DIR
        )
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"arxiv-{timestamp}.bib"
        filepath = export_dir / filename

        # Format and write BibTeX
        bibtex_entries = [self._format_paper_as_bibtex(p) for p in papers]
        content = "\n\n".join(bibtex_entries)

        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
        except OSError as exc:
            self.notify(
                f"Failed to export BibTeX: {exc}",
                title="BibTeX Export",
                severity="error",
            )
            return

        self.notify(
            f"Exported {len(papers)} paper(s) to {filepath.name}",
            title="BibTeX Export",
        )

    def _format_paper_as_markdown(self, paper: Paper) -> str:
        """Format a paper as Markdown."""
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        lines = [
            f"## {paper.title}",
            "",
            f"**arXiv:** [{paper.arxiv_id}]({paper.url})",
            f"**Date:** {paper.date}",
            f"**Categories:** {paper.categories}",
            f"**Authors:** {paper.authors}",
        ]
        if paper.comments:
            lines.append(f"**Comments:** {paper.comments}")
        lines.extend(
            [
                "",
                "### Abstract",
                "",
                abstract_text,
            ]
        )
        return "\n".join(lines)

    def action_export_markdown(self) -> None:
        """Export selected papers as Markdown to clipboard."""
        papers = self._get_papers_to_export()
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
            self.notify(
                "Failed to copy to clipboard", title="Markdown", severity="error"
            )

    def _get_papers_to_export(self) -> list[Paper]:
        """Get papers to export (selected or current)."""
        if self.selected_ids:
            list_view = self.query_one("#paper-list", ListView)
            ordered: list[Paper] = []
            seen: set[str] = set()
            for item in list_view.children:
                if (
                    isinstance(item, PaperListItem)
                    and item.paper.arxiv_id in self.selected_ids
                ):
                    ordered.append(item.paper)
                    seen.add(item.paper.arxiv_id)
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
        item = self._get_current_paper_item()
        if not item:
            self.notify("No paper selected", title="Similar", severity="warning")
            return

        target_paper = item.paper
        similar_papers = find_similar_papers(
            target_paper,
            self.all_papers,
            metadata=self._config.paper_metadata,
            abstract_lookup=lambda paper: self._get_abstract_text(
                paper, allow_async=False
            )
            or "",
        )

        if not similar_papers:
            self.notify("No similar papers found", title="Similar", severity="warning")
            return

        def on_paper_selected(arxiv_id: str | None) -> None:
            if arxiv_id:
                # Find and scroll to the selected paper
                list_view = self.query_one("#paper-list", ListView)
                for i, list_item in enumerate(list_view.children):
                    if (
                        isinstance(list_item, PaperListItem)
                        and list_item.paper.arxiv_id == arxiv_id
                    ):
                        list_view.index = i
                        return
                # Paper not in current view
                self.notify(
                    "Paper not in current view (try clearing filter)",
                    title="Similar",
                    severity="warning",
                )

        self.push_screen(
            RecommendationsScreen(target_paper, similar_papers), on_paper_selected
        )

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

    def _load_current_date(self) -> None:
        """Load papers from the current date file and refresh UI."""
        if not self._is_history_mode():
            return

        current_date, path = self._history_files[self._current_date_index]
        self.all_papers = parse_arxiv_file(path)
        self._papers_by_id = {p.arxiv_id: p for p in self.all_papers}
        self.filtered_papers = self.all_papers.copy()

        self._abstract_cache.clear()
        self._abstract_loading.clear()
        self._abstract_queue.clear()
        self._abstract_pending_ids.clear()

        # Clear selection when switching dates
        self.selected_ids.clear()

        # Recompute watched papers for new paper set
        self._compute_watched_papers()

        # Apply current filter and sort
        query = self.query_one("#search-input", Input).value.strip()
        self._apply_filter(query)

        # Update subtitle
        self.sub_title = f"{len(self.all_papers)} papers · {current_date.strftime(HISTORY_DATE_FORMAT)}"

    def action_prev_date(self) -> None:
        """Navigate to previous (older) date file."""
        if not self._is_history_mode():
            self.notify("Not in history mode", title="Navigate", severity="warning")
            return

        if self._current_date_index >= len(self._history_files) - 1:
            self.notify("Already at oldest", title="Navigate")
            return

        self._current_date_index += 1
        self._load_current_date()
        current_date = self._get_current_date()
        if current_date:
            self.notify(
                f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate"
            )

    def action_next_date(self) -> None:
        """Navigate to next (newer) date file."""
        if not self._is_history_mode():
            self.notify("Not in history mode", title="Navigate", severity="warning")
            return

        if self._current_date_index <= 0:
            self.notify("Already at newest", title="Navigate")
            return

        self._current_date_index -= 1
        self._load_current_date()
        current_date = self._get_current_date()
        if current_date:
            self.notify(
                f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate"
            )

    def _update_header(self) -> None:
        """Update header with selection count and sort info."""
        query = self.query_one("#search-input", Input).value.strip()
        self.query_one("#list-header", Label).update(self._format_header_text(query))
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Update the status bar with total, filtered, and selected counts."""
        try:
            status = self.query_one("#status-bar", Label)
        except NoMatches:
            return
        status.update(
            f"[dim]Total: {len(self.all_papers)}  Filtered: {len(self.filtered_papers)}  Selected: {len(self.selected_ids)}[/]"
        )

    def _get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Look up a paper by its arXiv ID. O(1) dict lookup."""
        return self._papers_by_id.get(arxiv_id)

    def _get_pdf_url(self, paper: Paper) -> str:
        """Get the PDF URL for a paper."""
        if "arxiv.org/pdf/" in paper.url:
            return paper.url if paper.url.endswith(".pdf") else f"{paper.url}.pdf"
        return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

    def _get_paper_url(self, paper: Paper) -> str:
        """Get the preferred URL for a paper (abs or PDF)."""
        if self._config.prefer_pdf_url:
            return self._get_pdf_url(paper)
        return paper.url

    async def _download_pdf_async(self, paper: Paper) -> bool:
        """Download a single PDF asynchronously.

        Args:
            paper: The paper to download.

        Returns:
            True if download succeeded, False otherwise.
        """
        url = self._get_pdf_url(paper)
        path = get_pdf_download_path(paper, self._config)

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    timeout=PDF_DOWNLOAD_TIMEOUT,
                    follow_redirects=True,
                )
                response.raise_for_status()
                path.write_bytes(response.content)
            logger.debug(f"Downloaded PDF for {paper.arxiv_id} to {path}")
            return True
        except (httpx.HTTPError, OSError) as e:
            logger.debug(f"Download failed for {paper.arxiv_id}: {e}")
            return False

    def _start_downloads(self) -> None:
        """Start download tasks up to the concurrency limit."""
        while self._download_queue and len(self._downloading) < MAX_CONCURRENT_DOWNLOADS:
            paper = self._download_queue.popleft()
            if paper.arxiv_id in self._downloading:
                continue
            self._downloading.add(paper.arxiv_id)
            asyncio.create_task(self._process_single_download(paper))

    async def _process_single_download(self, paper: Paper) -> None:
        """Process a single download and update state."""
        success = await self._download_pdf_async(paper)
        self._download_results[paper.arxiv_id] = success
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
        """Update status bar with download progress."""
        try:
            status_bar = self.query_one("#status-bar", Label)
            status_bar.update(f"Downloading: {completed}/{total} complete")
        except NoMatches:
            pass

    def _finish_download_batch(self) -> None:
        """Handle completion of a download batch."""
        successes = sum(1 for v in self._download_results.values() if v)
        failures = len(self._download_results) - successes

        # Get download directory for notification
        if self._config.pdf_download_dir:
            download_dir = self._config.pdf_download_dir
        else:
            download_dir = f"~/{DEFAULT_PDF_DOWNLOAD_DIR}"

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

    def action_open_url(self) -> None:
        """Open selected papers' URLs in the default browser."""
        # If papers are selected, open all of them
        if self.selected_ids:
            for arxiv_id in self.selected_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper:
                    webbrowser.open(self._get_paper_url(paper))
            self.notify(f"Opening {len(self.selected_ids)} papers", title="Browser")
        else:
            # Otherwise, open the currently highlighted paper
            details = self.query_one(PaperDetails)
            if details.paper:
                webbrowser.open(self._get_paper_url(details.paper))
                self.notify(f"Opening {details.paper.arxiv_id}", title="Browser")

    def action_open_pdf(self) -> None:
        """Open selected papers' PDF URLs in the default browser."""
        if self.selected_ids:
            for arxiv_id in self.selected_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper:
                    webbrowser.open(self._get_pdf_url(paper))
            self.notify(f"Opening {len(self.selected_ids)} PDFs", title="PDF")
        else:
            details = self.query_one(PaperDetails)
            if details.paper:
                webbrowser.open(self._get_pdf_url(details.paper))
                self.notify(f"Opening PDF for {details.paper.arxiv_id}", title="PDF")

    def action_download_pdf(self) -> None:
        """Download PDFs for selected papers (or current paper)."""
        # Collect papers to download
        papers_to_download: list[Paper] = []

        if self.selected_ids:
            for arxiv_id in self.selected_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper:
                    papers_to_download.append(paper)
        else:
            details = self.query_one(PaperDetails)
            if details.paper:
                papers_to_download.append(details.paper)

        if not papers_to_download:
            self.notify("No papers to download", title="Download", severity="warning")
            return

        # Filter out already downloaded
        to_download: list[Paper] = []
        for paper in papers_to_download:
            path = get_pdf_download_path(paper, self._config)
            if path.exists() and path.stat().st_size > 0:
                logger.debug(f"Skipping {paper.arxiv_id}: already downloaded")
            else:
                to_download.append(paper)

        if not to_download:
            self.notify("All PDFs already downloaded", title="Download")
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
        lines = [
            f"Title: {paper.title}",
            f"Authors: {paper.authors}",
            f"arXiv: {paper.arxiv_id}",
            f"Date: {paper.date}",
            f"Categories: {paper.categories}",
        ]
        if paper.comments:
            lines.append(f"Comments: {paper.comments}")
        lines.append(f"URL: {paper.url}")
        lines.append("")
        lines.append(f"Abstract: {abstract_text}")
        return "\n".join(lines)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard. Returns True on success.

        Uses platform-specific clipboard tools with timeout protection.
        Logs failures at debug level for troubleshooting.
        """
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(
                    ["pbcopy"],
                    input=text.encode("utf-8"),
                    check=True,
                    shell=False,
                    timeout=SUBPROCESS_TIMEOUT,
                )
            elif system == "Linux":
                # Try xclip first, then xsel
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=text.encode("utf-8"),
                        check=True,
                        shell=False,
                        timeout=SUBPROCESS_TIMEOUT,
                    )
                except (FileNotFoundError, subprocess.CalledProcessError):
                    subprocess.run(
                        ["xsel", "--clipboard", "--input"],
                        input=text.encode("utf-8"),
                        check=True,
                        shell=False,
                        timeout=SUBPROCESS_TIMEOUT,
                    )
            elif system == "Windows":
                subprocess.run(
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
        # Get papers to copy
        if self.selected_ids:
            papers_to_copy = [
                self._get_paper_by_id(arxiv_id) for arxiv_id in self.selected_ids
            ]
            papers_to_copy = [p for p in papers_to_copy if p is not None]
        else:
            # Copy currently highlighted paper if none selected
            details = self.query_one(PaperDetails)
            if details.paper:
                papers_to_copy = [details.paper]
            else:
                self.notify("No paper selected", title="Copy", severity="warning")
                return

        if not papers_to_copy:
            self.notify("No papers to copy", title="Copy", severity="warning")
            return

        # Format papers with separator between them
        separator = f"\n\n{CLIPBOARD_SEPARATOR}\n\n"
        formatted = separator.join(
            self._format_paper_for_clipboard(p) for p in papers_to_copy
        )

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


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Browse arXiv papers from a text file in a TUI"
    )
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
    args = parser.parse_args()

    base_dir = Path(__file__).parent

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
    current_date_index = 0

    if args.input is not None:
        # Explicit -i flag: use that file, disable history mode
        # Resolve symlinks for security and normalize path
        arxiv_file = args.input.resolve()
        history_files = []  # Disable history mode

        if not arxiv_file.exists():
            print(f"Error: {arxiv_file} not found", file=sys.stderr)
            return 1
        if arxiv_file.is_dir():
            print(f"Error: {arxiv_file} is a directory, not a file", file=sys.stderr)
            return 1
        if not os.access(arxiv_file, os.R_OK):
            print(
                f"Error: {arxiv_file} is not readable (permission denied)",
                file=sys.stderr,
            )
            return 1

        papers = parse_arxiv_file(arxiv_file)

    elif history_files:
        # History mode: use history directory
        if args.date:
            # Find specific date
            try:
                target_date = datetime.strptime(args.date, HISTORY_DATE_FORMAT).date()
            except ValueError:
                print(
                    f"Error: Invalid date format '{args.date}', expected YYYY-MM-DD",
                    file=sys.stderr,
                )
                return 1

            found = False
            for i, (d, _) in enumerate(history_files):
                if d == target_date:
                    current_date_index = i
                    found = True
                    break
            if not found:
                print(f"Error: No file found for date {args.date}", file=sys.stderr)
                return 1
        elif not args.no_restore and config.session.current_date:
            # Try to restore saved date
            try:
                saved_date = datetime.strptime(
                    config.session.current_date, HISTORY_DATE_FORMAT
                ).date()
                for i, (d, _) in enumerate(history_files):
                    if d == saved_date:
                        current_date_index = i
                        break
            except ValueError:
                pass  # Invalid saved date, use newest

        _, arxiv_file = history_files[current_date_index]
        papers = parse_arxiv_file(arxiv_file)

    else:
        # Legacy fallback: use arxiv.txt in current directory
        arxiv_file = base_dir / "arxiv.txt"

        if not arxiv_file.exists():
            print("Error: No papers found. Either:", file=sys.stderr)
            print("  - Create history/YYYY-MM-DD.txt files, or", file=sys.stderr)
            print("  - Create arxiv.txt in the current directory, or", file=sys.stderr)
            print("  - Use -i to specify an input file", file=sys.stderr)
            return 1
        if not os.access(arxiv_file, os.R_OK):
            print(
                f"Error: {arxiv_file} is not readable (permission denied)",
                file=sys.stderr,
            )
            return 1

        papers = parse_arxiv_file(arxiv_file)

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
