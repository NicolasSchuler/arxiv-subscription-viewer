"""Data models and constants for the arXiv Browser application."""

from __future__ import annotations

from dataclasses import dataclass, field

# Application identity — single source of truth for platformdirs config paths
CONFIG_APP_NAME = "arxiv-browser"

# Sort order options
SORT_OPTIONS = ["title", "date", "arxiv_id", "citations", "trending", "relevance"]

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
    "summary": "AI Summary",
    "s2": "Semantic Scholar",
    "hf": "HuggingFace",
    "version": "Version Update",
}
DEFAULT_COLLAPSED_SECTIONS: list[str] = ["tags", "relevance", "summary", "s2", "hf", "version"]

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


@dataclass(slots=True)
class PaperMetadata:
    """User annotations for a paper (notes, tags, read status)."""

    arxiv_id: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    is_read: bool = False
    starred: bool = False
    last_checked_version: int | None = None


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
    """Complete user configuration including session state and preferences."""

    paper_metadata: dict[str, PaperMetadata] = field(default_factory=dict)
    watch_list: list[WatchListEntry] = field(default_factory=list)
    bookmarks: list[SearchBookmark] = field(default_factory=list)
    collections: list[PaperCollection] = field(default_factory=list)
    marks: dict[str, str] = field(default_factory=dict)  # letter -> arxiv_id
    session: SessionState = field(default_factory=SessionState)
    show_abstract_preview: bool = False
    bibtex_export_dir: str = ""  # Empty = use ~/arxiv-exports/
    pdf_download_dir: str = ""  # Empty = use ~/arxiv-pdfs/
    prefer_pdf_url: bool = False
    category_colors: dict[str, str] = field(default_factory=dict)
    theme: dict[str, str] = field(default_factory=dict)
    theme_name: str = "monokai"
    llm_command: str = ""  # Shell command template, e.g. 'claude -p {prompt}'
    llm_prompt_template: str = ""  # Empty = use DEFAULT_LLM_PROMPT
    llm_preset: str = ""  # "claude" | "codex" | "llm" | "" (custom)
    arxiv_api_max_results: int = ARXIV_API_DEFAULT_MAX_RESULTS
    s2_enabled: bool = False  # Semantic Scholar enrichment (opt-in)
    s2_api_key: str = ""  # Optional S2 API key for higher rate limits
    s2_cache_ttl_days: int = 7  # Days to cache S2 data
    hf_enabled: bool = False  # HuggingFace trending (opt-in)
    hf_cache_ttl_hours: int = 6  # Hours to cache HF daily data
    research_interests: str = ""  # Free-text research interest description for relevance scoring
    collapsed_sections: list[str] = field(default_factory=lambda: list(DEFAULT_COLLAPSED_SECTIONS))
    pdf_viewer: str = (
        ""  # External PDF viewer command, e.g. "zathura {path}" or "open -a Skim {path}"
    )
    version: int = 1


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
    """Snapshot of local browsing state to restore after API search mode."""

    all_papers: list[Paper]
    papers_by_id: dict[str, Paper]
    selected_ids: set[str]
    sort_index: int
    search_query: str
    pending_query: str
    watch_filter_active: bool
    active_bookmark_index: int
    list_index: int
    sub_title: str
    highlight_terms: dict[str, list[str]]
    match_scores: dict[str, float]


__all__ = [
    "ARXIV_API_DEFAULT_MAX_RESULTS",
    "ARXIV_API_MAX_RESULTS_LIMIT",
    "CONFIG_APP_NAME",
    "DEFAULT_COLLAPSED_SECTIONS",
    "DETAIL_SECTION_KEYS",
    "DETAIL_SECTION_NAMES",
    "MAX_COLLECTIONS",
    "MAX_PAPERS_PER_COLLECTION",
    "SORT_OPTIONS",
    "STOPWORDS",
    "WATCH_MATCH_TYPES",
    "ArxivSearchModeState",
    "ArxivSearchRequest",
    "LocalBrowseSnapshot",
    "Paper",
    "PaperCollection",
    "PaperMetadata",
    "QueryToken",
    "SearchBookmark",
    "SessionState",
    "UserConfig",
    "WatchListEntry",
]
