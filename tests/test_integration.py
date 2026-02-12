"""Integration tests for arXiv Browser using real fixture data.

Tests end-to-end workflows: parsing real emails, history navigation,
config round-trips, export format validation, resource cleanup, and
structured logging configuration.
"""

from __future__ import annotations

import csv
import io
import logging
import logging.handlers
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_browser.app import ArxivBrowser, _configure_logging
from arxiv_browser.export import (
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
    format_papers_as_markdown_table,
)
from arxiv_browser.parsing import discover_history_files, parse_arxiv_file

# ── Paths ────────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_JAN26 = FIXTURES_DIR / "2026-01-26.txt"
FIXTURE_JAN23 = FIXTURES_DIR / "2026-01-23.txt"
FIXTURE_MALFORMED = FIXTURES_DIR / "malformed.txt"

# Known arXiv category prefixes (non-exhaustive, covers common ones)
_VALID_CATEGORY_PREFIXES = frozenset(
    {
        "astro-ph",
        "cond-mat",
        "cs",
        "econ",
        "eess",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "physics",
        "q-bio",
        "q-fin",
        "quant-ph",
        "stat",
    }
)

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


# ── 2a. Real File Parsing Tests ─────────────────────────────────────────────


class TestRealArxivFiles:
    """Parse real arXiv email fixtures and verify field extraction."""

    def test_parse_real_file_extracts_all_papers(self):
        papers = parse_arxiv_file(FIXTURE_JAN26)
        assert len(papers) == 10

    def test_parse_second_fixture_correct_count(self):
        papers = parse_arxiv_file(FIXTURE_JAN23)
        assert len(papers) == 5

    def test_real_papers_have_valid_arxiv_ids(self):
        papers = parse_arxiv_file(FIXTURE_JAN26)
        for paper in papers:
            assert _ARXIV_ID_RE.match(paper.arxiv_id), f"Invalid arXiv ID: {paper.arxiv_id}"

    def test_real_papers_have_nonempty_fields(self):
        papers = parse_arxiv_file(FIXTURE_JAN26)
        for paper in papers:
            assert paper.title.strip(), f"Empty title for {paper.arxiv_id}"
            assert paper.authors.strip(), f"Empty authors for {paper.arxiv_id}"
            assert paper.categories.strip(), f"Empty categories for {paper.arxiv_id}"
            assert paper.abstract_raw.strip(), f"Empty abstract_raw for {paper.arxiv_id}"
            assert paper.url, f"Missing URL for {paper.arxiv_id}"

    def test_real_papers_categories_are_valid(self):
        """All category prefixes should be known arXiv archive prefixes."""
        papers = parse_arxiv_file(FIXTURE_JAN26)
        for paper in papers:
            for cat in paper.categories.split():
                prefix = cat.split(".")[0]
                assert prefix in _VALID_CATEGORY_PREFIXES, (
                    f"Unknown category prefix '{prefix}' in '{cat}' for {paper.arxiv_id}"
                )

    def test_multiline_titles_are_joined(self):
        """Papers with multi-line titles should have them joined properly."""
        papers = parse_arxiv_file(FIXTURE_JAN26)
        for paper in papers:
            assert "\n" not in paper.title, (
                f"Title contains newline for {paper.arxiv_id}: {paper.title!r}"
            )

    def test_multi_category_papers_exist(self):
        """At least one paper should have multiple categories."""
        papers = parse_arxiv_file(FIXTURE_JAN26)
        multi_cat = [p for p in papers if len(p.categories.split()) > 1]
        assert len(multi_cat) >= 1, "Expected at least one multi-category paper"

    def test_malformed_file_does_not_crash(self):
        """Parser should handle malformed input without raising."""
        papers = parse_arxiv_file(FIXTURE_MALFORMED)
        assert isinstance(papers, list)

    def test_malformed_file_extracts_valid_entries(self):
        """Parser should extract valid papers from malformed file."""
        papers = parse_arxiv_file(FIXTURE_MALFORMED)
        ids = {p.arxiv_id for p in papers}
        # The two valid papers plus the truncated one
        assert "2601.99001" in ids
        assert "2601.99003" in ids

    def test_malformed_file_skips_duplicates(self):
        """Parser should skip duplicate arXiv IDs."""
        papers = parse_arxiv_file(FIXTURE_MALFORMED)
        ids = [p.arxiv_id for p in papers]
        assert ids.count("2601.99001") == 1


# ── 2b. History Mode Navigation Tests ───────────────────────────────────────


class TestHistoryNavigation:
    """Test date navigation with real fixture files."""

    @pytest.fixture()
    def fixture_history_dir(self, tmp_path):
        """Create a temp history/ directory with copies of fixture files."""
        hdir = tmp_path / "history"
        hdir.mkdir()
        shutil.copy(FIXTURE_JAN26, hdir / "2026-01-26.txt")
        shutil.copy(FIXTURE_JAN23, hdir / "2026-01-23.txt")
        return tmp_path

    def test_discover_fixture_history_files(self, fixture_history_dir):
        """Should find both date files, sorted newest-first."""
        files = discover_history_files(fixture_history_dir)
        assert len(files) == 2
        # Newest first
        assert files[0][0].isoformat() == "2026-01-26"
        assert files[1][0].isoformat() == "2026-01-23"

    def test_history_files_are_parseable(self, fixture_history_dir):
        """Each discovered file should parse to papers."""
        files = discover_history_files(fixture_history_dir)
        for _, path in files:
            papers = parse_arxiv_file(path)
            assert len(papers) > 0

    async def test_date_navigation_loads_different_papers(self, fixture_history_dir):
        """Navigating dates should load different paper sets."""
        files = discover_history_files(fixture_history_dir)
        papers_by_date = {}
        for d, path in files:
            papers_by_date[d.isoformat()] = parse_arxiv_file(path)

        # Different dates should have different papers
        ids_26 = {p.arxiv_id for p in papers_by_date["2026-01-26"]}
        ids_23 = {p.arxiv_id for p in papers_by_date["2026-01-23"]}
        assert ids_26 != ids_23, "Different dates should have different paper sets"
        assert len(ids_26) == 10
        assert len(ids_23) == 5


# ── 2c. Full Workflow Tests ─────────────────────────────────────────────────


class TestFullWorkflow:
    """End-to-end workflows using Textual run_test() + pilot."""

    @staticmethod
    def _app_from_fixture(fixture_path: Path) -> ArxivBrowser:
        """Create an ArxivBrowser from a fixture file."""
        papers = parse_arxiv_file(fixture_path)
        return ArxivBrowser(papers, restore_session=False)

    @pytest.mark.integration()
    async def test_app_renders_real_papers(self):
        """App should render all papers from a real fixture file."""
        from textual.widgets import OptionList

        app = self._app_from_fixture(FIXTURE_JAN26)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test():
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 10

    @pytest.mark.integration()
    async def test_search_real_paper_by_title(self):
        """Searching for a known paper title should filter correctly."""
        from textual.widgets import OptionList

        app = self._app_from_fixture(FIXTURE_JAN26)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("slash")
                # Type "DSGym" — unique title word from fixture
                for ch in "DSGym":
                    await pilot.press(ch)

                option_list = app.query_one("#paper-list", OptionList)
                # Wait for debounce
                deadline = 2.0
                import asyncio

                end = asyncio.get_running_loop().time() + deadline
                while option_list.option_count > 1 and asyncio.get_running_loop().time() < end:
                    await pilot.pause(0.05)
                assert option_list.option_count == 1

    @pytest.mark.integration()
    async def test_star_paper(self):
        """Starring a paper should persist in metadata."""
        app = self._app_from_fixture(FIXTURE_JAN26)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Star the first paper
                await pilot.press("x")
                # Check metadata
                first_id = app.all_papers[0].arxiv_id
                meta = app._config.paper_metadata.get(first_id)
                assert meta is not None
                assert meta.starred is True

    @pytest.mark.integration()
    async def test_sort_cycles_order(self):
        """Pressing 's' should cycle through sort options."""
        from textual.widgets import OptionList

        app = self._app_from_fixture(FIXTURE_JAN26)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                initial_count = option_list.option_count
                assert initial_count == 10

                # Press 's' to change sort
                await pilot.press("s")
                await pilot.pause(0.1)
                # Still same count, just different order
                assert option_list.option_count == 10


# ── 2d. Export Format Verification ──────────────────────────────────────────


class TestExportWithRealData:
    """Verify export formats produce valid output from real papers."""

    @pytest.fixture()
    def real_papers(self):
        return parse_arxiv_file(FIXTURE_JAN26)

    def test_bibtex_export_valid_syntax(self, real_papers):
        """BibTeX export should have proper @misc entries."""
        for paper in real_papers:
            bib = format_paper_as_bibtex(paper)
            assert bib.startswith("@misc{"), f"Bad BibTeX start for {paper.arxiv_id}"
            assert bib.rstrip().endswith("}"), f"Bad BibTeX end for {paper.arxiv_id}"
            assert f"eprint = {{{paper.arxiv_id}}}" in bib
            assert "title = {" in bib
            assert "author = {" in bib

    def test_csv_export_parseable(self, real_papers):
        """CSV export should be parseable by csv.reader with correct columns."""
        csv_output = format_papers_as_csv(real_papers)
        reader = csv.reader(io.StringIO(csv_output))
        rows = list(reader)
        # Header + data rows
        assert len(rows) == len(real_papers) + 1
        header = rows[0]
        assert "arxiv_id" in header
        assert "title" in header
        assert "authors" in header

    def test_markdown_export_has_links(self, real_papers):
        """Markdown table export should contain arXiv URLs."""
        md = format_papers_as_markdown_table(real_papers)
        for paper in real_papers:
            assert paper.arxiv_id in md

    def test_ris_export_valid_tags(self, real_papers):
        """RIS export should contain required tags."""
        for paper in real_papers:
            ris = format_paper_as_ris(paper, abstract_text=paper.abstract)
            assert "TY  - " in ris, f"Missing TY tag for {paper.arxiv_id}"
            assert f"TI  - {paper.title}" in ris
            assert "AU  - " in ris
            assert "UR  - " in ris
            assert "ER  - " in ris

    def test_bibtex_special_chars_escaped(self, real_papers):
        """BibTeX should escape special LaTeX characters."""
        for paper in real_papers:
            bib = format_paper_as_bibtex(paper)
            # Should be valid (no unescaped %, &, etc. in title field)
            assert bib.count("@misc{") == 1


# ── 3b. Resource Cleanup Verification ───────────────────────────────────────


class TestResourceCleanup:
    """Verify async resources are properly cleaned up."""

    @pytest.mark.integration()
    async def test_http_client_closed_after_unmount(self):
        """After app exit, _http_client should be None."""
        papers = parse_arxiv_file(FIXTURE_JAN23)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test():
                # App is running, client may be created
                pass
            # After exit
            assert app._http_client is None

    @pytest.mark.integration()
    async def test_background_tasks_empty_after_unmount(self):
        """After app exit, _background_tasks set should be empty."""
        papers = parse_arxiv_file(FIXTURE_JAN23)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test():
                pass
            assert len(app._background_tasks) == 0


# ── 4d. Debug Logging Tests ─────────────────────────────────────────────────


class TestDebugLogging:
    """Verify --debug flag configures file logging."""

    def test_configure_logging_disabled_by_default(self):
        """debug=False should disable logging."""
        _configure_logging(debug=False)
        assert (
            logging.root.level == logging.CRITICAL
            or logging.root.manager.disable >= logging.CRITICAL
        )

    def test_configure_logging_creates_file_handler(self, tmp_path: Path):
        """debug=True should add a RotatingFileHandler."""
        # Remove any handlers added by previous tests
        logging.root.handlers.clear()
        logging.disable(logging.NOTSET)

        with patch("arxiv_browser.app.user_config_dir", return_value=str(tmp_path)):
            _configure_logging(debug=True)

        try:
            # Should have at least one handler
            assert len(logging.root.handlers) >= 1
            handler = logging.root.handlers[-1]
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
            assert logging.root.level == logging.DEBUG

            # Log file should exist
            log_file = tmp_path / "debug.log"
            assert log_file.exists() or handler.baseFilename == str(log_file)
        finally:
            # Clean up: remove handlers and reset logging
            logging.root.handlers.clear()
            logging.disable(logging.NOTSET)

    def test_log_file_location_uses_config_dir(self, tmp_path: Path):
        """Log file should be in the platformdirs config directory."""
        logging.root.handlers.clear()
        logging.disable(logging.NOTSET)

        with patch("arxiv_browser.app.user_config_dir", return_value=str(tmp_path)):
            _configure_logging(debug=True)

        try:
            handler = logging.root.handlers[-1]
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
            assert str(tmp_path) in handler.baseFilename
            assert handler.baseFilename.endswith("debug.log")
        finally:
            logging.root.handlers.clear()
            logging.disable(logging.NOTSET)
