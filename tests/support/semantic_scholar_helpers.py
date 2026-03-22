"""Shared Semantic Scholar test helpers."""

from __future__ import annotations

from arxiv_browser.semantic_scholar import CitationEntry, SemanticScholarPaper


def _make_paper(**kwargs) -> SemanticScholarPaper:
    """Create a SemanticScholarPaper with sensible defaults for testing."""
    defaults = {
        "arxiv_id": "2401.12345",
        "s2_paper_id": "abc123",
        "citation_count": 42,
        "influential_citation_count": 5,
        "tldr": "Does X.",
        "fields_of_study": ("CS",),
        "year": 2024,
        "url": "https://example.com",
        "title": "Test Paper",
        "abstract": "Abstract text.",
    }
    defaults.update(kwargs)
    return SemanticScholarPaper(**defaults)


def _make_citation_entry(**kwargs) -> CitationEntry:
    """Create a CitationEntry with sensible defaults for testing."""
    defaults = {
        "s2_paper_id": "s2id001",
        "arxiv_id": "2401.12345",
        "title": "Test Paper",
        "authors": "Alice, Bob",
        "year": 2024,
        "citation_count": 42,
        "url": "https://arxiv.org/abs/2401.12345",
    }
    defaults.update(kwargs)
    return CitationEntry(**defaults)
