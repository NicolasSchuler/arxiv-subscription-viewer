"""Data models and constants for the arXiv Browser application."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from datetime import datetime

# Application identity — single source of truth for platformdirs config paths
CONFIG_APP_NAME = "arxiv-browser"

# Sort order options
SORT_OPTIONS = [
    "title",
    "date",
    "arxiv_id",
    "citations",
    "trending",
    "relevance",
    "queue",
    "triage",
]

# Watch list entry types
WATCH_MATCH_TYPES = ("author", "title", "keyword")

# arXiv API constants
ARXIV_API_DEFAULT_MAX_RESULTS = 50
ARXIV_API_MAX_RESULTS_LIMIT = 200

# Collection limits
MAX_COLLECTIONS = 20
MAX_PAPERS_PER_COLLECTION = 500

# Collapsible detail pane section definitions
DETAIL_SECTION_KEYS: list[str] = [
    "authors",
    "abstract",
    "tags",
    "relevance",
    "deadlines",
    "summary",
    "s2",
    "hf",
    "version",
]
DETAIL_SECTION_NAMES: dict[str, str] = {
    "authors": "Authors",
    "abstract": "Abstract",
    "tags": "Tags",
    "relevance": "Relevance",
    "deadlines": "Submission Targets",
    "summary": "AI Summary",
    "s2": "Semantic Scholar",
    "hf": "HuggingFace",
    "version": "Version Update",
}
DETAIL_MODES = ("scan", "full")
DEFAULT_COLLAPSED_SECTIONS: list[str] = ["tags", "relevance", "summary", "s2", "hf", "version"]
PANE_SPLIT_MIN = 1
PANE_SPLIT_DEFAULT = 2
PANE_SPLIT_MAX = 4
PANE_SPLIT_TOTAL = 5


def coerce_pane_split(value: object) -> int:
    """Validate and clamp the persisted list/detail pane split."""
    if not isinstance(value, int) or isinstance(value, bool):
        return PANE_SPLIT_DEFAULT
    return max(PANE_SPLIT_MIN, min(value, PANE_SPLIT_MAX))


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
    source: str = "local"  # "local" | "api"
    provider: str = "arxiv"  # "arxiv" | future preprint provider IDs


@dataclass(slots=True)
class LineAnnotation:
    """Inline annotation anchored to a visible detail-pane line."""

    line: int
    text: str


@dataclass(slots=True)
class PaperMetadata:
    """User annotations for a paper (notes, tags, read status)."""

    arxiv_id: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    is_read: bool = False
    starred: bool = False
    last_checked_version: int | None = None
    next_review_date: str | None = None
    review_stage: int | None = None
    line_annotations: list[LineAnnotation] = field(default_factory=list)


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
class PaperCollection:
    """A named, ordered collection of papers (reading list)."""

    name: str
    description: str = ""
    paper_ids: list[str] = field(default_factory=list)
    created: str = ""  # ISO 8601 timestamp, set on creation


@dataclass(slots=True)
class DigestInboxContext:
    """Ephemeral display metadata for a digest-backed TUI inbox."""

    source_label: str
    section_labels_by_id: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class SessionState:
    """State to restore on next run (scroll position, filters, etc.)."""

    scroll_index: int = 0
    current_filter: str = ""
    sort_index: int = 0
    selected_ids: list[str] = field(default_factory=list)
    current_date: str | None = None  # YYYY-MM-DD format, None for non-history mode

    def __post_init__(self) -> None:
        """Clamp sort_index to valid SORT_OPTIONS range (defense-in-depth)."""
        if self.sort_index < 0 or self.sort_index >= len(SORT_OPTIONS):
            self.sort_index = 0


@dataclass(slots=True)
class QueryToken:
    """Token used for advanced query parsing."""

    kind: str  # "term" or "op"
    value: str
    field: str | None = None
    phrase: bool = False


@dataclass(slots=True)
class UserConfig:
    """Complete persisted user configuration.

    This dataclass mixes a few categories of state:
    persistent library metadata (notes/tags/collections), UI preferences,
    optional enrichment/configuration knobs, trust decisions for shell-backed
    commands, and lightweight session restore state for the next launch.
    """

    paper_metadata: dict[str, PaperMetadata] = field(default_factory=dict)
    watch_list: list[WatchListEntry] = field(default_factory=list)
    tracked_authors: list[str] = field(default_factory=list)
    bookmarks: list[SearchBookmark] = field(default_factory=list)
    collections: list[PaperCollection] = field(default_factory=list)
    marks: dict[str, str] = field(default_factory=dict)  # letter -> arxiv_id
    session: SessionState = field(default_factory=SessionState)
    show_abstract_preview: bool = False
    compact_list: bool = False
    detail_mode: str = "scan"
    pane_split: int = PANE_SPLIT_DEFAULT  # Relative list/detail pane-size preset
    bibtex_export_dir: str = ""  # Empty = use ~/arxiv-exports/
    pdf_download_dir: str = ""  # Empty = use ~/arxiv-pdfs/
    prefer_pdf_url: bool = False
    category_colors: dict[str, str] = field(default_factory=dict)
    theme: dict[str, str] = field(default_factory=dict)
    theme_name: str = "monokai"
    custom_themes: dict[str, dict[str, str]] = field(default_factory=dict)
    llm_command: str = ""  # Shell command template, e.g. 'claude -p {prompt}'
    llm_prompt_template: str = ""  # Empty = use DEFAULT_LLM_PROMPT
    llm_phd_explainer_field: str = "physics"  # Target field for cross-field PhD summaries
    llm_preset: str = ""  # "claude" | "codex" | "llm" | "" (custom)
    allow_llm_shell_fallback: bool = True  # False = reject shell-only command templates
    llm_max_retries: int = 1  # Retries for transient LLM failures (timeout, non-zero exit)
    llm_timeout: int = 120  # Seconds to wait for LLM CLI response
    llm_streaming_enabled: bool = False  # Opt-in incremental LLM output updates
    llm_provider_type: str = "cli"  # "cli" | "http" — selects LLM execution backend
    llm_api_base_url: str = ""  # Base URL for HTTP providers (e.g. https://api.openai.com)
    llm_api_key: str = ""  # API key for HTTP providers
    llm_api_model: str = ""  # Model name for HTTP providers (e.g. "gpt-4o")
    paper_content_cache_ttl_days: int = 7  # Days to cache extracted full-paper text
    paper_content_pdf_fallback: bool = True  # Use PDF text when arXiv HTML is unavailable
    pdf_preview_max_pages: int = 3  # Pages to render for terminal PDF preview
    arxiv_api_max_results: int = ARXIV_API_DEFAULT_MAX_RESULTS
    s2_enabled: bool = False  # Semantic Scholar enrichment (opt-in)
    s2_api_key: str = ""  # Optional S2 API key for higher rate limits
    s2_cache_ttl_days: int = 7  # Days to cache S2 data
    hf_enabled: bool = False  # HuggingFace trending (opt-in)
    hf_cache_ttl_hours: int = 6  # Hours to cache HF daily data
    semantic_search_backend: str = (
        "auto"  # "auto" | "fastembed" | "sentence-transformers" | "http" | "off"
    )
    semantic_search_model: str = "BAAI/bge-small-en-v1.5"
    semantic_search_api_base_url: str = ""
    semantic_search_api_key: str = ""
    semantic_search_top_k: int = 100
    semantic_search_min_score: int = 15
    conference_deadlines_enabled: bool = False  # Third-party conference deadline import
    conference_deadlines_source_url: str = (
        "https://raw.githubusercontent.com/paperswithcode/ai-deadlines/gh-pages/"
        "_data/conferences.yml"
    )
    conference_deadlines_cache_ttl_hours: int = 24
    research_interests: str = ""  # Free-text research interest description for relevance scoring
    collapsed_sections: list[str] = field(default_factory=lambda: list(DEFAULT_COLLAPSED_SECTIONS))
    pdf_viewer: str = (
        ""  # External PDF viewer command, e.g. "zathura {path}" or "open -a Skim {path}"
    )
    trusted_llm_command_hashes: list[str] = field(default_factory=list)
    trusted_pdf_viewer_hashes: list[str] = field(default_factory=list)
    version: int = 1
    onboarding_seen: bool = False  # True after user has dismissed the first-run help overlay
    shortcuts_hint_seen: bool = False  # True after the one-time "Press ? for shortcuts" nudge
    badge_legend_hint_seen: bool = False  # True after the one-time "badges → ? legend" nudge
    last_seen_whats_new: str = ""  # Tag of the last What's New notes the user dismissed
    config_defaulted: bool = False  # True when config was corrupt and defaults were used

    @staticmethod
    def parse_custom_themes(raw: object) -> dict[str, dict[str, str]]:
        """Return a sanitized custom theme registry from untrusted config data."""
        if not isinstance(raw, dict):
            return {}
        themes: dict[str, dict[str, str]] = {}
        for name, palette in raw.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(palette, dict):
                continue
            colors = {
                color_key: color_value
                for color_key, color_value in palette.items()
                if isinstance(color_key, str) and isinstance(color_value, str)
            }
            if colors:
                themes[name] = colors
        return themes


@dataclass(slots=True)
class ArxivSearchRequest:
    """User-entered parameters for an arXiv API search."""

    query: str
    field: str = "all"
    category: str = ""


@dataclass(slots=True)
class ArxivSearchModeState:
    """Current state for arXiv API search mode."""

    request: ArxivSearchRequest
    start: int = 0
    max_results: int = ARXIV_API_DEFAULT_MAX_RESULTS


@dataclass(slots=True)
class LocalBrowseSnapshot:
    """Snapshot of the pre-API local browsing state.

    The browser stores one of these before switching into live arXiv API mode
    so it can later restore the previous local dataset, filter/sort context,
    selection, highlights, and visible-list position as one coherent transition.
    """

    all_papers: list[Paper]
    papers_by_id: dict[str, Paper]
    selected_ids: set[str]
    sort_index: int
    search_query: str
    pending_query: str
    applied_query: str
    watch_filter_active: bool
    active_bookmark_index: int
    list_index: int
    sub_title: str
    highlight_terms: dict[str, list[str]]
    match_scores: dict[str, float]


# ============================================================================
# Date Parsing
# ============================================================================

# Date format used in arXiv emails (e.g., "Mon, 15 Jan 2024")
ARXIV_DATE_FORMAT = "%a, %d %b %Y"
_ARXIV_DATE_PREFIX_PATTERN = re.compile(r"([A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})")


@functools.lru_cache(maxsize=512)
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

    return datetime.min


__all__ = [
    "ARXIV_API_DEFAULT_MAX_RESULTS",
    "ARXIV_API_MAX_RESULTS_LIMIT",
    "ARXIV_DATE_FORMAT",
    "CONFIG_APP_NAME",
    "DEFAULT_COLLAPSED_SECTIONS",
    "DETAIL_MODES",
    "DETAIL_SECTION_KEYS",
    "DETAIL_SECTION_NAMES",
    "MAX_COLLECTIONS",
    "MAX_PAPERS_PER_COLLECTION",
    "PANE_SPLIT_DEFAULT",
    "PANE_SPLIT_MAX",
    "PANE_SPLIT_MIN",
    "PANE_SPLIT_TOTAL",
    "SORT_OPTIONS",
    "STOPWORDS",
    "WATCH_MATCH_TYPES",
    "ArxivSearchModeState",
    "ArxivSearchRequest",
    "LineAnnotation",
    "LocalBrowseSnapshot",
    "Paper",
    "PaperCollection",
    "PaperMetadata",
    "QueryToken",
    "SearchBookmark",
    "SessionState",
    "UserConfig",
    "WatchListEntry",
    "coerce_pane_split",
    "parse_arxiv_date",
]
