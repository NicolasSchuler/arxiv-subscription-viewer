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

    # Extract arXiv ID from externalIds if not provided
    if not arxiv_id:
        external_ids = data.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")

    # Parse TLDR — it's an object with a "text" field
    tldr_obj = data.get("tldr")
    tldr = ""
    if isinstance(tldr_obj, dict):
        tldr = tldr_obj.get("text", "")

    # Parse fields of study
    fos_raw = data.get("fieldsOfStudy") or []
    fields_of_study = tuple(f for f in fos_raw if isinstance(f, str))

    return SemanticScholarPaper(
        arxiv_id=arxiv_id,
        s2_paper_id=paper_id,
        citation_count=data.get("citationCount") or 0,
        influential_citation_count=data.get("influentialCitationCount") or 0,
        tldr=tldr,
        fields_of_study=fields_of_study,
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
    external_ids = data.get("externalIds")
    if not isinstance(external_ids, dict):
        external_ids = {}
    arxiv_id = external_ids.get("ArXiv", "")
    authors_raw = data.get("authors") or []
    authors = ", ".join(
        a.get("name", "")
        for a in authors_raw
        if isinstance(a, dict) and isinstance(a.get("name"), str) and a.get("name")
    )
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else data.get("url") or ""
    return CitationEntry(
        s2_paper_id=paper_id,
        arxiv_id=arxiv_id,
        title=data.get("title") or "Unknown Title",
        authors=authors,
        year=data.get("year"),
        citation_count=data.get("citationCount") or 0,
        url=url,
    )
