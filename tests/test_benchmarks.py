"""Performance benchmarks — run with `just bench` or `pytest -m slow -v -s`.

Each test verifies that a core operation completes within a generous time budget.
These are NOT micro-benchmarks — they guard against O(n²) regressions and other
algorithmic performance cliffs when operating at realistic scale (200-500 papers).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from rapidfuzz import fuzz

from arxiv_browser.export import format_papers_as_csv
from arxiv_browser.models import Paper, WatchListEntry
from arxiv_browser.parsing import clean_latex, parse_arxiv_file
from arxiv_browser.query import paper_matches_watch_entry, sort_papers
from arxiv_browser.similarity import TfidfIndex, find_similar_papers

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILE = FIXTURE_DIR / "2026-01-26.txt"


def _assert_within(fn: object, max_seconds: float, label: str = "") -> float:
    """Run fn() and assert it completes within max_seconds. Returns elapsed time."""
    t0 = time.perf_counter()
    fn()  # type: ignore[operator]
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.4f}s (limit: {max_seconds}s)")
    assert elapsed < max_seconds, f"{label} took {elapsed:.4f}s, limit {max_seconds}s"
    return elapsed


def _parse_fixture() -> list[Paper]:
    """Parse the fixture file."""
    return parse_arxiv_file(FIXTURE_FILE)


def _generate_papers(n: int, base_papers: list[Paper]) -> list[Paper]:
    """Generate n papers by cycling through base_papers with unique IDs."""
    papers: list[Paper] = []
    for i in range(n):
        base = base_papers[i % len(base_papers)]
        papers.append(
            Paper(
                arxiv_id=f"2401.{10000 + i:05d}",
                date=base.date,
                title=f"{base.title} variant-{i}",
                authors=f"{base.authors}, Author-{i}",
                categories=base.categories,
                comments=base.comments,
                abstract=f"{base.abstract_raw} Extended discussion variant {i}.",
                url=f"https://arxiv.org/abs/2401.{10000 + i:05d}",
                abstract_raw=f"{base.abstract_raw} Extended discussion variant {i}.",
            )
        )
    return papers


@pytest.fixture(scope="module")
def base_papers() -> list[Paper]:
    """Parse the fixture file once for the module."""
    return _parse_fixture()


@pytest.fixture(scope="module")
def papers_200(base_papers: list[Paper]) -> list[Paper]:
    return _generate_papers(200, base_papers)


@pytest.fixture(scope="module")
def papers_500(base_papers: list[Paper]) -> list[Paper]:
    return _generate_papers(500, base_papers)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBenchmarks:
    """Performance regression tests — excluded from default test runs."""

    def test_parse_arxiv_file(self) -> None:
        """Parsing the largest fixture file should complete quickly."""
        _assert_within(lambda: parse_arxiv_file(FIXTURE_FILE), 0.5, "parse_arxiv_file")

    def test_fuzzy_search_500(self, papers_500: list[Paper]) -> None:
        """Fuzzy matching 500 papers against a query should be fast."""
        query = "neural network optimization"

        def run() -> None:
            for p in papers_500:
                fuzz.WRatio(query, p.title)

        _assert_within(run, 0.5, "fuzzy_search_500")

    def test_sort_all_keys(self, papers_500: list[Paper]) -> None:
        """Sorting 500 papers by each sort key should be fast."""
        sort_keys = ["title", "date", "arxiv_id", "citations", "trending", "relevance"]
        for key in sort_keys:
            _assert_within(
                lambda k=key: sort_papers(papers_500[:], k),
                0.05,
                f"sort_{key}",
            )

    def test_tfidf_build(self, papers_200: list[Paper]) -> None:
        """Building a TF-IDF index over 200 papers."""
        _assert_within(
            lambda: TfidfIndex.build(papers_200, lambda p: f"{p.title} {p.abstract_raw}"),
            0.5,
            "tfidf_build_200",
        )

    def test_find_similar(self, papers_200: list[Paper]) -> None:
        """Finding similar papers in a 200-paper corpus."""
        index = TfidfIndex.build(papers_200, lambda p: f"{p.title} {p.abstract_raw}")

        def run() -> None:
            find_similar_papers(
                papers_200[0],
                papers_200,
                tfidf_index=index,
                metadata={},
                top_n=10,
            )

        _assert_within(run, 0.3, "find_similar_200")

    def test_watched_papers(self, papers_500: list[Paper]) -> None:
        """Matching 500 papers against 10 watch entries."""
        entries = [
            WatchListEntry(pattern=f"pattern-{i}", match_type=mt, case_sensitive=False)
            for i, mt in enumerate(["author", "title", "keyword"] * 4)
        ][:10]

        def run() -> None:
            for paper in papers_500:
                for entry in entries:
                    paper_matches_watch_entry(paper, entry)

        _assert_within(run, 0.1, "watch_match_500x10")

    def test_clean_latex_bulk(self) -> None:
        """Cleaning 1000 LaTeX strings."""
        samples = [
            r"This is a $\mathcal{O}(n \log n)$ algorithm with $\alpha$-divergence",
            r"We propose \textbf{BERT} for \emph{language understanding} tasks",
            r"The $\beta$-VAE with $\gamma = 0.1$ achieves $\sim$95\% accuracy",
            r"Using {\sc Transformer} with \textit{attention} mechanism",
            r"Results: $f(x) = \frac{1}{n}\sum_{i=1}^{n} x_i$",
        ]

        def run() -> None:
            for i in range(1000):
                clean_latex(samples[i % len(samples)])

        _assert_within(run, 0.2, "clean_latex_1000")

    def test_csv_export(self, papers_500: list[Paper]) -> None:
        """Exporting 500 papers to CSV."""
        _assert_within(
            lambda: format_papers_as_csv(papers_500),
            0.2,
            "csv_export_500",
        )
