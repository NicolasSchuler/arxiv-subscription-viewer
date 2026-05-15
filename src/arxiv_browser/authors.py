"""Author parsing, matching, and profile helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arxiv_browser.models import Paper, parse_arxiv_date

if TYPE_CHECKING:
    from arxiv_browser.semantic_scholar import SemanticScholarPaper


_AUTHOR_SPLIT_RE = re.compile(r"\s*,\s*|\s+\band\b\s+", re.IGNORECASE)
_AUTHOR_KEY_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(slots=True, frozen=True)
class AuthorPaperRecord:
    """One paper row in an author profile."""

    paper: Paper
    citation_count: int | None = None


@dataclass(slots=True, frozen=True)
class AuthorCount:
    """Counted author or co-author entry."""

    name: str
    count: int


@dataclass(slots=True, frozen=True)
class AuthorProfile:
    """Derived local-library profile for one author."""

    author: str
    papers: tuple[AuthorPaperRecord, ...]
    coauthors: tuple[AuthorCount, ...]
    total_cached_citations: int
    citation_coverage: int


def normalize_author_name(name: str) -> str:
    """Return a stable key for exact author matching."""
    collapsed = _AUTHOR_KEY_RE.sub(" ", name.casefold())
    return " ".join(collapsed.split())


def split_author_names(authors: str) -> list[str]:
    """Split a paper's author string into display names."""
    cleaned = " ".join(authors.replace("\n", " ").split())
    if not cleaned:
        return []
    names = [part.strip(" ;") for part in _AUTHOR_SPLIT_RE.split(cleaned)]
    return [name for name in names if name]


def author_matches_exact(authors: str, query: str) -> bool:
    """Return whether *query* exactly matches one normalized author name."""
    query_key = normalize_author_name(query)
    if not query_key:
        return True
    return any(normalize_author_name(author) == query_key for author in split_author_names(authors))


def paper_matches_tracked_author(paper: Paper, tracked_authors: Sequence[str]) -> bool:
    """Return whether a paper matches any exact tracked-author name."""
    return any(author_matches_exact(paper.authors, author) for author in tracked_authors)


def dedupe_author_names(authors: Sequence[str]) -> list[str]:
    """Return author names deduplicated by normalized exact-match key."""
    result: list[str] = []
    seen: set[str] = set()
    for author in authors:
        cleaned = " ".join(author.split())
        key = normalize_author_name(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def build_author_profile(
    author: str,
    papers: Sequence[Paper],
    s2_cache: Mapping[str, SemanticScholarPaper] | None = None,
) -> AuthorProfile:
    """Build a cache-only author profile from loaded library papers."""
    author_key = normalize_author_name(author)
    matched: list[AuthorPaperRecord] = []
    coauthor_counts: Counter[str] = Counter()
    coauthor_display: dict[str, str] = {}
    total_citations = 0
    citation_coverage = 0

    for paper in papers:
        names = split_author_names(paper.authors)
        if not any(normalize_author_name(name) == author_key for name in names):
            continue
        s2 = s2_cache.get(paper.arxiv_id) if s2_cache else None
        citation_count = s2.citation_count if s2 is not None else None
        if citation_count is not None:
            total_citations += citation_count
            citation_coverage += 1
        matched.append(AuthorPaperRecord(paper=paper, citation_count=citation_count))
        for name in names:
            key = normalize_author_name(name)
            if not key or key == author_key:
                continue
            coauthor_counts[key] += 1
            coauthor_display.setdefault(key, name)

    matched.sort(key=lambda record: parse_arxiv_date(record.paper.date), reverse=True)
    coauthors = tuple(
        AuthorCount(coauthor_display[key], count)
        for key, count in sorted(coauthor_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return AuthorProfile(
        author=author,
        papers=tuple(matched),
        coauthors=coauthors,
        total_cached_citations=total_citations,
        citation_coverage=citation_coverage,
    )


__all__ = [
    "AuthorCount",
    "AuthorPaperRecord",
    "AuthorProfile",
    "author_matches_exact",
    "build_author_profile",
    "dedupe_author_names",
    "normalize_author_name",
    "paper_matches_tracked_author",
    "split_author_names",
]
