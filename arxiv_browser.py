#!/usr/bin/env python3
"""arXiv Paper Browser TUI - Browse arXiv papers from a text file.

Usage:
    python arxiv_browser.py                    # Use default arxiv.txt
    python arxiv_browser.py -i papers.txt      # Use custom file
    python arxiv_browser.py --no-restore       # Start fresh session

Key bindings:
    /       - Toggle search (fuzzy matching)
    o       - Open selected paper(s) in browser
    c       - Copy selected paper(s) to clipboard
    b       - Copy as BibTeX
    B       - Export BibTeX to file (for Zotero import)
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
    unread          - Show only unread papers
    starred         - Show only starred papers
    <text>          - Filter by title/author
"""

import argparse
import functools
import json
import logging
import os
import platform
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir
from rapidfuzz import fuzz
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static, TextArea

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

# UI truncation limits
RECOMMENDATION_TITLE_MAX_LEN = 60  # Max title length in recommendations modal
PREVIEW_ABSTRACT_MAX_LEN = 150  # Max abstract preview length in list items
BOOKMARK_NAME_MAX_LEN = 15  # Max bookmark name display length

# BibTeX export settings
DEFAULT_BIBTEX_EXPORT_DIR = "arxiv-exports"  # Default subdirectory in home folder

# History file discovery limit
MAX_HISTORY_FILES = 365  # Limit to ~1 year of history to prevent memory issues
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "now", "here", "there", "then",
    "once", "again", "further", "still", "already", "always", "never",
    "using", "based", "via", "novel", "new", "approach", "method", "methods",
    "paper", "propose", "proposed", "show", "results", "model", "models",
})

# Date format used in arXiv emails (e.g., "Mon, 15 Jan 2024")
ARXIV_DATE_FORMAT = "%a, %d %b %Y"


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


@dataclass(slots=True)
class Paper:
    """Represents an arXiv paper entry."""
    arxiv_id: str
    date: str
    title: str
    authors: str
    categories: str
    comments: str | None
    abstract: str
    url: str


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


@dataclass
class UserConfig:
    """Complete user configuration including session state and preferences."""
    paper_metadata: dict[str, PaperMetadata] = field(default_factory=dict)
    watch_list: list[WatchListEntry] = field(default_factory=list)
    bookmarks: list[SearchBookmark] = field(default_factory=list)
    marks: dict[str, str] = field(default_factory=dict)  # letter -> arxiv_id
    session: SessionState = field(default_factory=SessionState)
    show_abstract_preview: bool = False
    bibtex_export_dir: str = ""  # Empty = use ~/arxiv-exports/
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
        "bookmarks": [
            {"name": b.name, "query": b.query}
            for b in config.bookmarks
        ],
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

    session = SessionState(
        scroll_index=_safe_get(session_data, "scroll_index", 0, int),
        current_filter=_safe_get(session_data, "current_filter", "", str),
        sort_index=_safe_get(session_data, "sort_index", 0, int),
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
            watch_list.append(WatchListEntry(
                pattern=_safe_get(entry, "pattern", "", str),
                match_type=_safe_get(entry, "match_type", "author", str),
                case_sensitive=_safe_get(entry, "case_sensitive", False, bool),
            ))

    # Parse bookmarks with type validation
    bookmarks = []
    raw_bookmarks = data.get("bookmarks", [])
    if isinstance(raw_bookmarks, list):
        for b in raw_bookmarks:
            if not isinstance(b, dict):
                continue
            bookmarks.append(SearchBookmark(
                name=_safe_get(b, "name", "", str),
                query=_safe_get(b, "query", "", str),
            ))

    # Parse marks with type validation
    marks = data.get("marks", {})
    if not isinstance(marks, dict):
        marks = {}

    return UserConfig(
        paper_metadata=paper_metadata,
        watch_list=watch_list,
        bookmarks=bookmarks,
        marks=marks,
        session=session,
        show_abstract_preview=_safe_get(data, "show_abstract_preview", False, bool),
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
_ABSTRACT_PATTERN = re.compile(r"(?:Categories|Comments):[^\n]*\n\\\\\n(.+?)\n\\\\", re.DOTALL)
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
        if not entry or not entry.startswith("\\"):
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
            abstract = " ".join(abstract_match.group(1).split())
        else:
            abstract = ""

        # Extract URL
        url_match = _URL_PATTERN.search(entry)
        url = url_match.group(1) if url_match else f"https://arxiv.org/abs/{arxiv_id}"

        papers.append(Paper(
            arxiv_id=arxiv_id,
            date=date,
            title=clean_latex(title),
            authors=clean_latex(authors),
            categories=categories,
            comments=clean_latex(comments) if comments else None,
            abstract=clean_latex(abstract),
            url=url,
        ))

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


# Category color mapping (Monokai-inspired palette)
CATEGORY_COLORS = {
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

def parse_arxiv_date(date_str: str) -> datetime:
    """Parse arXiv date string to datetime for proper sorting.

    Args:
        date_str: Date string like "Mon, 15 Jan 2024"

    Returns:
        Parsed datetime object, or datetime.min for malformed dates.
    """
    try:
        return datetime.strptime(date_str.strip(), ARXIV_DATE_FORMAT)
    except ValueError:
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

def _extract_keywords(text: str, min_length: int = 4) -> set[str]:
    """Extract significant keywords from text, filtering stopwords."""
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


def compute_paper_similarity(paper_a: Paper, paper_b: Paper) -> float:
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
    abstract_kw_a = _extract_keywords(paper_a.abstract)
    abstract_kw_b = _extract_keywords(paper_b.abstract)
    abstract_sim = _jaccard_similarity(abstract_kw_a, abstract_kw_b)

    # Weighted sum
    return 0.4 * cat_sim + 0.3 * author_sim + 0.2 * title_sim + 0.1 * abstract_sim


def find_similar_papers(
    target: Paper,
    all_papers: list[Paper],
    top_n: int = SIMILARITY_TOP_N,
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
    for paper in all_papers:
        if paper.arxiv_id == target.arxiv_id:
            continue
        score = compute_paper_similarity(target, paper)
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
    ) -> None:
        super().__init__()
        self.paper = paper
        self._selected = selected
        self._metadata = metadata
        self._watched = watched
        self._show_preview = show_preview
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

    def _get_title_text(self) -> str:
        """Get the formatted title text based on selection and metadata state."""
        prefix_parts = []

        # Selection indicator
        if self._selected:
            prefix_parts.append("[#a6e22e]●[/]")  # Monokai green

        # Watched indicator
        if self._watched:
            prefix_parts.append("[#fd971f]👁[/]")  # Monokai orange

        # Starred indicator
        if self._metadata and self._metadata.starred:
            prefix_parts.append("[#e6db74]⭐[/]")  # Monokai yellow

        # Read indicator
        if self._metadata and self._metadata.is_read:
            prefix_parts.append("[#75715e]✓[/]")  # Monokai gray

        prefix = " ".join(prefix_parts)
        if prefix:
            return f"{prefix} {self.paper.title}"
        return self.paper.title

    def _get_meta_text(self) -> str:
        """Get the formatted metadata text."""
        parts = [f"[dim]{self.paper.arxiv_id}[/]", format_categories(self.paper.categories)]

        # Show tags if present
        if self._metadata and self._metadata.tags:
            tag_str = " ".join(f"[#ae81ff]#{tag}[/]" for tag in self._metadata.tags)
            parts.append(tag_str)

        return "  ".join(parts)

    def _get_preview_text(self) -> str:
        """Get truncated abstract preview text.

        Returns formatted Rich markup for the abstract preview.
        Handles empty abstracts and truncates at word boundaries.
        """
        abstract = self.paper.abstract
        if not abstract:
            return "[dim italic]No abstract available[/]"
        if len(abstract) <= PREVIEW_ABSTRACT_MAX_LEN:
            return f"[dim italic]{abstract}[/]"
        # Truncate at word boundary for cleaner display
        truncated = abstract[:PREVIEW_ABSTRACT_MAX_LEN].rsplit(" ", 1)[0]
        return f"[dim italic]{truncated}...[/]"

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
        meta_widget = self.query_one(".paper-meta", Static)
        title_widget.update(self._get_title_text())
        meta_widget.update(self._get_meta_text())

    def compose(self) -> ComposeResult:
        yield Static(self._get_title_text(), classes="paper-title")
        yield Static(self._get_meta_text(), classes="paper-meta")
        if self._show_preview:
            yield Static(self._get_preview_text(), classes="paper-preview")


class PaperDetails(Static):
    """Widget to display full paper details."""

    def __init__(self) -> None:
        super().__init__()
        self._paper: Paper | None = None

    def update_paper(self, paper: Paper | None) -> None:
        """Update the displayed paper details."""
        self._paper = paper
        if paper is None:
            self.update("[dim italic]Select a paper to view details[/]")
            return

        lines = []

        # Title section (Monokai foreground)
        lines.append(f"[bold #f8f8f2]{paper.title}[/]")
        lines.append("")

        # Metadata section (Monokai blue for labels, purple for values)
        lines.append(f"[bold #66d9ef]arXiv:[/] [#ae81ff]{paper.arxiv_id}[/]")
        lines.append(f"[bold #66d9ef]Date:[/] {paper.date}")
        lines.append(f"[bold #66d9ef]Categories:[/] {format_categories(paper.categories)}")
        if paper.comments:
            lines.append(f"[bold #66d9ef]Comments:[/] [dim]{paper.comments}[/]")
        lines.append("")

        # Authors section (Monokai green)
        lines.append("[bold #a6e22e]Authors[/]")
        lines.append(f"[#f8f8f2]{paper.authors}[/]")
        lines.append("")

        # Abstract section (Monokai orange)
        lines.append("[bold #fd971f]Abstract[/]")
        lines.append(f"[#f8f8f2]{paper.abstract}[/]")
        lines.append("")

        # URL section (Monokai pink/red for label, blue for URL)
        lines.append(f"[bold #f92672]URL:[/] [#66d9ef]{paper.url}[/]")

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
            yield Label("Separate tags with commas (e.g., to-read, llm, important)", id="tags-help")
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
            item = RecommendationListItem(
                paper,
                Static(f"[bold]{paper.title}[/]", classes="rec-title"),
                Static(
                    f"[dim]{paper.arxiv_id}[/] | {paper.categories} | [#a6e22e]{score:.0%}[/] match",
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
            classes = "bookmark-tab active" if i == self._active_index else "bookmark-tab"
            yield Label(f"{i + 1}: {bookmark.name}", classes=classes, id=f"bookmark-{i}")
        yield Label("[+]", classes="bookmark-add", id="bookmark-add")

    async def update_bookmarks(self, bookmarks: list[SearchBookmark], active_index: int = -1) -> None:
        """Update the displayed bookmarks."""
        self._bookmarks = bookmarks
        self._active_index = active_index
        await self.remove_children()
        for i, bookmark in enumerate(bookmarks[:9]):
            classes = "bookmark-tab active" if i == self._active_index else "bookmark-tab"
            self.mount(Label(f"{i + 1}: {bookmark.name}", classes=classes, id=f"bookmark-{i}"))
        self.mount(Label("[+]", classes="bookmark-add", id="bookmark-add"))


class ArxivBrowser(App):
    """A TUI application to browse arXiv papers."""

    TITLE = "arXiv Paper Browser"

    # Monokai color theme
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
        scrollbar-color-hover: #a6a68a;
        scrollbar-color-active: #66d9ef;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "toggle_search", "Search"),
        Binding("escape", "cancel_search", "Cancel", show=False),
        Binding("o", "open_url", "Open Selected"),
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
        # Phase 9: Paper similarity
        Binding("R", "show_similar", "Similar"),
        # History mode: date navigation
        Binding("bracketleft", "prev_date", "Older", show=False),
        Binding("bracketright", "next_date", "Newer", show=False),
    ]

    DEBOUNCE_DELAY = 0.3  # seconds

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

        # History mode: multiple date files
        self._history_files: list[tuple[date, Path]] = history_files or []
        self._current_date_index: int = current_date_index

        # Watch list: pre-compute matching papers for O(1) lookup
        self._watched_paper_ids: set[str] = set()
        self._watch_filter_active: bool = False
        self._compute_watched_papers()

        # Abstract preview toggle
        self._show_abstract_preview: bool = self._config.show_abstract_preview

        # Bookmark state
        self._active_bookmark_index: int = -1  # -1 means no active bookmark

        # Fuzzy search match scores
        self._match_scores: dict[str, float] = {}

        # Vim-style marks state
        self._pending_mark_action: str | None = None  # "set" or "goto"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-pane"):
                yield Label(f" Papers ({len(self.all_papers)} total)", id="list-header")
                yield BookmarkTabBar(self._config.bookmarks, self._active_bookmark_index)
                with Vertical(id="search-container"):
                    yield Input(placeholder=" Filter: text, cat:cs.AI, tag:name, unread, starred", id="search-input")
                yield ListView(
                    *[PaperListItem(p) for p in self.filtered_papers],
                    id="paper-list"
                )
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

    def _save_session_state(self) -> None:
        """Save current session state to config.

        Handles the case where DOM widgets may already be destroyed during unmount.
        """
        # Get current date for history mode
        current_date = self._get_current_date()
        current_date_str = current_date.strftime(HISTORY_DATE_FORMAT) if current_date else None

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
        except NoMatches:
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
            details.update_paper(event.item.paper)

    @on(ListView.Highlighted)
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle paper highlight (keyboard navigation)."""
        if isinstance(event.item, PaperListItem):
            details = self.query_one(PaperDetails)
            details.update_paper(event.item.paper)

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
            self.DEBOUNCE_DELAY,
            self._debounced_filter,
        )

    def _debounced_filter(self) -> None:
        """Apply filter after debounce delay."""
        self._search_timer = None
        self._apply_filter(self._pending_query)

    def _format_header_text(self, query: str = "") -> str:
        """Format the header text with paper count, date info, and selection info."""
        selection_info = f" [{len(self.selected_ids)} selected]" if self.selected_ids else ""
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

    def _filter_by_category(self, category: str) -> list[Paper]:
        """Filter papers by category substring match."""
        category_lower = category.lower()
        return [p for p in self.all_papers if category_lower in p.categories.lower()]

    def _filter_by_tag(self, tag: str) -> list[Paper]:
        """Filter papers that have the specified tag."""
        tag_lower = tag.lower()
        return [
            p for p in self.all_papers
            if p.arxiv_id in self._config.paper_metadata
            and tag_lower in [t.lower() for t in self._config.paper_metadata[p.arxiv_id].tags]
        ]

    def _filter_unread(self) -> list[Paper]:
        """Filter to show only unread papers."""
        return [
            p for p in self.all_papers
            if p.arxiv_id not in self._config.paper_metadata
            or not self._config.paper_metadata[p.arxiv_id].is_read
        ]

    def _filter_starred(self) -> list[Paper]:
        """Filter to show only starred papers."""
        return [
            p for p in self.all_papers
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
        query_lower = query.lower()

        # Clear match scores by default (only fuzzy search populates them)
        self._match_scores.clear()

        # Apply the appropriate filter based on query prefix
        if not query:
            self.filtered_papers = self.all_papers.copy()
        elif query_lower.startswith("cat:"):
            self.filtered_papers = self._filter_by_category(query[4:].strip())
        elif query_lower.startswith("tag:"):
            self.filtered_papers = self._filter_by_tag(query[4:].strip())
        elif query_lower == "unread":
            self.filtered_papers = self._filter_unread()
        elif query_lower == "starred":
            self.filtered_papers = self._filter_starred()
        else:
            self.filtered_papers = self._fuzzy_search(query)

        # Apply watch filter if active (intersects with other filters)
        if self._watch_filter_active:
            self.filtered_papers = [
                p for p in self.filtered_papers
                if p.arxiv_id in self._watched_paper_ids
            ]

        # Apply current sort order and refresh the list view
        self._sort_papers()
        self._refresh_list_view()

        # Update header with current query context
        self.query_one("#list-header", Label).update(self._format_header_text(query))

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
            self.filtered_papers.sort(key=lambda p: parse_arxiv_date(p.date), reverse=True)
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
            )
            for paper in self.filtered_papers
        ]
        if items:
            list_view.mount(*items)
            list_view.index = 0

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
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, PaperListItem):
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

        def on_notes_saved(notes: str) -> None:
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

        def on_tags_saved(tags: list[str]) -> None:
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
                return pattern in paper.title or pattern in paper.abstract
            else:
                return pattern in paper.title.lower() or pattern in paper.abstract.lower()
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
        if self._active_bookmark_index < 0 or self._active_bookmark_index >= len(self._config.bookmarks):
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
        self.notify(f"Paper not in current view (try clearing filter)", title="Mark", severity="warning")

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
            self.notify(f"Copied {count} BibTeX entr{'ies' if count > 1 else 'y'}", title="BibTeX")
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
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"arxiv-{timestamp}.bib"
        filepath = export_dir / filename

        # Format and write BibTeX
        bibtex_entries = [self._format_paper_as_bibtex(p) for p in papers]
        content = "\n\n".join(bibtex_entries)

        filepath.write_text(content, encoding="utf-8")

        self.notify(
            f"Exported {len(papers)} paper(s) to {filepath.name}",
            title="BibTeX Export",
        )

    def _format_paper_as_markdown(self, paper: Paper) -> str:
        """Format a paper as Markdown."""
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
        lines.extend([
            "",
            "### Abstract",
            "",
            paper.abstract,
        ])
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
            self.notify(f"Copied {count} paper{'s' if count > 1 else ''} as Markdown", title="Markdown")
        else:
            self.notify("Failed to copy to clipboard", title="Markdown", severity="error")

    def _get_papers_to_export(self) -> list[Paper]:
        """Get papers to export (selected or current)."""
        if self.selected_ids:
            papers = [self._get_paper_by_id(aid) for aid in self.selected_ids]
            return [p for p in papers if p is not None]
        else:
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
        similar_papers = find_similar_papers(target_paper, self.all_papers)

        if not similar_papers:
            self.notify("No similar papers found", title="Similar", severity="warning")
            return

        def on_paper_selected(arxiv_id: str | None) -> None:
            if arxiv_id:
                # Find and scroll to the selected paper
                list_view = self.query_one("#paper-list", ListView)
                for i, list_item in enumerate(list_view.children):
                    if isinstance(list_item, PaperListItem) and list_item.paper.arxiv_id == arxiv_id:
                        list_view.index = i
                        return
                # Paper not in current view
                self.notify("Paper not in current view (try clearing filter)", title="Similar", severity="warning")

        self.push_screen(RecommendationsScreen(target_paper, similar_papers), on_paper_selected)

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
            self.notify(f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate")

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
            self.notify(f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate")

    def _update_header(self) -> None:
        """Update header with selection count and sort info."""
        query = self.query_one("#search-input", Input).value.strip()
        self.query_one("#list-header", Label).update(self._format_header_text(query))

    def _get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Look up a paper by its arXiv ID. O(1) dict lookup."""
        return self._papers_by_id.get(arxiv_id)

    def action_open_url(self) -> None:
        """Open selected papers' URLs in the default browser."""
        # If papers are selected, open all of them
        if self.selected_ids:
            for arxiv_id in self.selected_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper:
                    webbrowser.open(paper.url)
            self.notify(f"Opening {len(self.selected_ids)} papers", title="Browser")
        else:
            # Otherwise, open the currently highlighted paper
            details = self.query_one(PaperDetails)
            if details.paper:
                webbrowser.open(details.paper.url)
                self.notify(f"Opening {details.paper.arxiv_id}", title="Browser")

    def _format_paper_for_clipboard(self, paper: Paper) -> str:
        """Format a paper's metadata for clipboard export."""
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
        lines.append(f"Abstract: {paper.abstract}")
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
                except FileNotFoundError:
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
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired, OSError) as e:
            logger.debug("Clipboard copy failed: %s", e)
            return False

    def action_copy_selected(self) -> None:
        """Copy selected papers' metadata to clipboard."""
        # Get papers to copy
        if self.selected_ids:
            papers_to_copy = [
                self._get_paper_by_id(arxiv_id)
                for arxiv_id in self.selected_ids
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
        "-i", "--input",
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
            print(f"Error: {arxiv_file} is not readable (permission denied)", file=sys.stderr)
            return 1

        papers = parse_arxiv_file(arxiv_file)

    elif history_files:
        # History mode: use history directory
        if args.date:
            # Find specific date
            try:
                target_date = datetime.strptime(args.date, HISTORY_DATE_FORMAT).date()
            except ValueError:
                print(f"Error: Invalid date format '{args.date}', expected YYYY-MM-DD", file=sys.stderr)
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
                saved_date = datetime.strptime(config.session.current_date, HISTORY_DATE_FORMAT).date()
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
            print(f"Error: {arxiv_file} is not readable (permission denied)", file=sys.stderr)
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
