"""Shared data models and response parsers for the Semantic Scholar client.

This module keeps the public schema and parsing layer separate from API + cache
orchestration to keep file size in source modules under the 1000-line policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
S2_PAPER_FIELDS = "paperId,citationCount,influentialCitationCount,tldr,fieldsOfStudy,year,url"
S2_REC_FIELDS = (
    "paperId,externalIds,title,citationCount,influentialCitationCount,tldr,year,url,abstract"
)
S2_DB_FILENAME = "semantic_scholar.db"
S2_DEFAULT_CACHE_TTL_DAYS = 7
S2_REC_CACHE_TTL_DAYS = 3
S2_CITATION_FIELDS = "paperId,externalIds,title,authors,year,citationCount,url"
S2_CITATION_GRAPH_CACHE_TTL_DAYS = 3
S2_MAX_REFERENCES = 100
S2_MAX_CITATIONS = 50
S2_CITATIONS_PAGE_SIZE = 100
S2_CITATIONS_SCAN_CAP = 1000
S2_REQUEST_TIMEOUT = 20  # seconds
S2_MAX_RETRIES = 3
S2_INITIAL_BACKOFF = 1.0  # seconds, doubles each retry


@dataclass(slots=True)
class SemanticScholarPaper:
    """Semantic Scholar paper metadata."""

    arxiv_id: str
    s2_paper_id: str
    citation_count: int
    influential_citation_count: int
    tldr: str  # Empty string if unavailable
    fields_of_study: tuple[str, ...]  # Immutable for caching
    year: int | None
    url: str
    # Optional fields for recommendations
    title: str = ""
    abstract: str = ""


@dataclass(slots=True)
class CitationEntry:
    """A paper in a citation graph (reference or citation)."""

    s2_paper_id: str
    arxiv_id: str  # Empty string if no arXiv ID
    title: str
    authors: str  # Comma-joined author names
    year: int | None
    citation_count: int
    url: str  # arXiv URL if arxiv_id present, else S2 URL


@dataclass(slots=True, frozen=True)
class S2PaperCacheSnapshot:
    """Resolved cache state for one S2 paper lookup."""

    status: Literal["miss", "not_found", "found"]
    paper: SemanticScholarPaper | None


@dataclass(slots=True, frozen=True)
class S2RecommendationsCacheSnapshot:
    """Resolved cache state for one S2 recommendations lookup."""

    status: Literal["miss", "empty", "found"]
    papers: list[SemanticScholarPaper]


@dataclass(slots=True)
class S2Request:
    """One Semantic Scholar GET request plus retry/logging metadata."""

    url: str
    params: dict[str, str]
    api_key: str = ""
    timeout: int = S2_REQUEST_TIMEOUT
    label: str = ""


def parse_s2_paper_response(data: dict, arxiv_id: str = "") -> SemanticScholarPaper | None:
    """Parse S2 API JSON into dataclass. Returns None if essential fields missing."""
    paper_id = data.get("paperId")
    if not paper_id:
        return None

    if not arxiv_id:
        arxiv_id = _external_arxiv_id(data)

    return SemanticScholarPaper(
        arxiv_id=arxiv_id,
        s2_paper_id=paper_id,
        citation_count=data.get("citationCount") or 0,
        influential_citation_count=data.get("influentialCitationCount") or 0,
        tldr=_parse_s2_tldr(data.get("tldr")),
        fields_of_study=_parse_string_sequence(data.get("fieldsOfStudy")),
        year=data.get("year"),
        url=data.get("url") or "",
        title=data.get("title") or "",
        abstract=data.get("abstract") or "",
    )


def parse_citation_entry(data: dict[str, Any]) -> CitationEntry | None:
    """Parse an S2 citation/reference wrapper into CitationEntry.

    The ``data`` dict is the inner paper object (e.g. ``citedPaper`` or
    ``citingPaper``) from the S2 references/citations endpoint.
    """
    paper_id = data.get("paperId")
    if not paper_id:
        return None
    arxiv_id = _external_arxiv_id(data)
    return CitationEntry(
        s2_paper_id=paper_id,
        arxiv_id=arxiv_id,
        title=data.get("title") or "Unknown Title",
        authors=_parse_author_names(data.get("authors")),
        year=data.get("year"),
        citation_count=data.get("citationCount") or 0,
        url=_citation_url(arxiv_id, data.get("url")),
    )


def _external_arxiv_id(data: dict[str, Any]) -> str:
    """Return an arXiv ID from an S2 externalIds payload when present."""
    external_ids = data.get("externalIds")
    if not isinstance(external_ids, dict):
        return ""
    arxiv_id = external_ids.get("ArXiv")
    return arxiv_id if isinstance(arxiv_id, str) else ""


def _parse_s2_tldr(tldr_obj: Any) -> str:
    """Return TLDR text from S2's optional TLDR object."""
    if not isinstance(tldr_obj, dict):
        return ""
    text = tldr_obj.get("text")
    return text if isinstance(text, str) else ""


def _parse_string_sequence(value: Any) -> tuple[str, ...]:
    """Return only string items from a JSON sequence."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _parse_author_names(value: Any) -> str:
    """Return comma-joined author names from an S2 authors payload."""
    if not isinstance(value, list):
        return ""
    return ", ".join(_author_name(author) for author in value if _author_name(author))


def _author_name(author: Any) -> str:
    """Return one author name, ignoring malformed author entries."""
    if not isinstance(author, dict):
        return ""
    name = author.get("name")
    return name if isinstance(name, str) else ""


def _citation_url(arxiv_id: str, s2_url: Any) -> str:
    """Prefer arXiv URLs for citation entries when an arXiv ID is available."""
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return s2_url if isinstance(s2_url, str) else ""
