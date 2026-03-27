#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.themes import THEME_NAMES, THEMES
from tests.support.canonical_exports import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    ARXIV_DATE_FORMAT,
    DEFAULT_CATEGORY_COLOR,
    DEFAULT_LLM_PROMPT,
    LLM_PRESETS,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    SUBPROCESS_TIMEOUT,
    SUMMARY_MODES,
    TAG_NAMESPACE_COLORS,
    Paper,
    PaperCollection,
    PaperMetadata,
    QueryToken,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
    build_arxiv_search_query,
    build_llm_prompt,
    clean_latex,
    escape_bibtex,
    export_metadata,
    extract_text_from_html,
    extract_year,
    format_categories,
    format_collection_as_markdown,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    format_summary_as_rich,
    generate_citation_key,
    get_pdf_download_path,
    get_summary_db_path,
    get_tag_color,
    import_metadata,
    insert_implicit_and,
    load_config,
    normalize_arxiv_id,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
    parse_arxiv_version_map,
    parse_tag_namespace,
    pill_label_for_token,
    reconstruct_query,
    save_config,
    to_rpn,
    tokenize_query,
)

# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestRISFormat:
    """Tests for format_paper_as_ris function."""

    def test_basic_output(self, make_paper):
        paper = make_paper()
        ris = format_paper_as_ris(paper)
        assert ris.startswith("TY  - ELEC")
        assert ris.endswith("ER  - ")
        assert "TI  - Test Paper" in ris
        assert "AU  - Test Author" in ris
        assert f"UR  - {paper.url}" in ris
        assert f"N1  - arXiv:{paper.arxiv_id}" in ris

    def test_multiple_authors(self, make_paper):
        paper = make_paper(authors="Alice Smith, Bob Jones, Carol Lee")
        ris = format_paper_as_ris(paper)
        assert ris.count("AU  - ") == 3
        assert "AU  - Alice Smith" in ris
        assert "AU  - Bob Jones" in ris
        assert "AU  - Carol Lee" in ris

    def test_multiple_categories(self, make_paper):
        paper = make_paper(categories="cs.AI cs.LG stat.ML")
        ris = format_paper_as_ris(paper)
        assert ris.count("KW  - ") == 3
        assert "KW  - cs.AI" in ris
        assert "KW  - cs.LG" in ris
        assert "KW  - stat.ML" in ris

    def test_with_comments(self, make_paper):
        paper = make_paper(comments="10 pages, 5 figures")
        ris = format_paper_as_ris(paper)
        assert "N2  - 10 pages, 5 figures" in ris

    def test_without_comments(self, make_paper):
        paper = make_paper(comments=None)
        ris = format_paper_as_ris(paper)
        assert "N2  - " not in ris

    def test_with_abstract(self, make_paper):
        paper = make_paper()
        ris = format_paper_as_ris(paper, abstract_text="A detailed abstract.")
        assert "AB  - A detailed abstract." in ris

    def test_without_abstract(self, make_paper):
        paper = make_paper()
        ris = format_paper_as_ris(paper, abstract_text="")
        assert "AB  - " not in ris

    def test_year_extraction(self, make_paper):
        paper = make_paper(date="Mon, 15 Jan 2024")
        ris = format_paper_as_ris(paper)
        assert "PY  - 2024" in ris


class TestCSVFormat:
    """Tests for format_papers_as_csv function."""

    def test_header_without_metadata(self, make_paper):
        papers = [make_paper()]
        csv_text = format_papers_as_csv(papers)
        header = csv_text.split("\n")[0]
        assert "arxiv_id" in header
        assert "title" in header
        assert "starred" not in header
        assert "read" not in header

    def test_header_with_metadata(self, make_paper):
        papers = [make_paper()]
        csv_text = format_papers_as_csv(papers, metadata={})
        header = csv_text.split("\n")[0]
        assert "starred" in header
        assert "read" in header
        assert "tags" in header
        assert "notes" in header

    def test_single_paper(self, make_paper):
        papers = [make_paper(title="My Paper", arxiv_id="2401.00001")]
        csv_text = format_papers_as_csv(papers)
        lines = csv_text.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "2401.00001" in lines[1]
        assert "My Paper" in lines[1]

    def test_multiple_papers(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00001", title="Paper One"),
            make_paper(arxiv_id="2401.00002", title="Paper Two"),
        ]
        csv_text = format_papers_as_csv(papers)
        lines = csv_text.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_quoting_with_commas(self, make_paper):
        papers = [make_paper(title='Paper with, commas "and" quotes')]
        csv_text = format_papers_as_csv(papers)
        # csv.writer should properly quote the field
        import csv as csv_mod
        import io

        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert rows[1][1] == 'Paper with, commas "and" quotes'

    def test_metadata_present(self, make_paper):
        papers = [make_paper(arxiv_id="2401.00001")]
        meta = {
            "2401.00001": PaperMetadata(
                arxiv_id="2401.00001",
                starred=True,
                is_read=False,
                tags=["to-read", "important"],
                notes="Check later",
            )
        }
        csv_text = format_papers_as_csv(papers, metadata=meta)
        import csv as csv_mod
        import io

        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        data = rows[1]
        # starred column
        assert data[7] == "true"
        # read column
        assert data[8] == "false"
        # tags column (semicolon-joined)
        assert data[9] == "to-read;important"
        # notes column
        assert data[10] == "Check later"

    def test_metadata_missing_paper(self, make_paper):
        papers = [make_paper(arxiv_id="2401.99999")]
        csv_text = format_papers_as_csv(papers, metadata={})
        import csv as csv_mod
        import io

        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        data = rows[1]
        assert data[7] == "false"
        assert data[8] == "false"
        assert data[9] == ""
        assert data[10] == ""

    def test_empty_list(self):
        csv_text = format_papers_as_csv([])
        lines = csv_text.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_comments_none(self, make_paper):
        papers = [make_paper(comments=None)]
        csv_text = format_papers_as_csv(papers)
        import csv as csv_mod
        import io

        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        # comments column (index 6)
        assert rows[1][6] == ""

    def test_formula_like_cells_are_sanitized(self, make_paper):
        papers = [make_paper(title="=SUM(1,2)", comments="@A1")]
        meta = {
            papers[0].arxiv_id: PaperMetadata(
                arxiv_id=papers[0].arxiv_id,
                notes="-cmd|' /C calc'!A0",
                tags=["+danger"],
            )
        }
        csv_text = format_papers_as_csv(papers, metadata=meta)

        import csv as csv_mod
        import io

        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        data = rows[1]

        assert data[1] == "'=SUM(1,2)"
        assert data[6] == "'@A1"
        assert data[9] == "'+danger"
        assert data[10] == "'-cmd|' /C calc'!A0"


class TestCSVExportMethods:
    """Verify ArxivBrowser CSV export methods use self._config.paper_metadata."""

    def _make_mock_app(self, papers, metadata=None):
        from unittest.mock import MagicMock

        from tests.support.canonical_exports import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = type(
            "Config",
            (),
            {
                "paper_metadata": metadata or {},
                "bibtex_export_dir": None,
            },
        )()
        app.selected_ids = set()
        app.notify = MagicMock()
        app._copy_to_clipboard = MagicMock(return_value=True)
        return app

    def test_export_clipboard_csv_uses_config_metadata(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001", title="Test")
        meta = {
            "2401.00001": PaperMetadata(
                arxiv_id="2401.00001",
                starred=True,
                is_read=True,
                tags=["ml"],
                notes="good paper",
            )
        }
        app = self._make_mock_app([paper], metadata=meta)
        app._export_clipboard_csv([paper])
        app._copy_to_clipboard.assert_called_once()
        csv_text = app._copy_to_clipboard.call_args[0][0]
        assert "starred" in csv_text
        assert "2401.00001" in csv_text
        assert "true" in csv_text

    def test_export_file_csv_uses_config_metadata(self, make_paper, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", title="Test")
        meta = {
            "2401.00001": PaperMetadata(
                arxiv_id="2401.00001",
                starred=True,
                is_read=False,
            )
        }
        app = self._make_mock_app([paper], metadata=meta)
        app._config.bibtex_export_dir = str(tmp_path / "exports")
        app._export_file_csv([paper])
        csv_files = list((tmp_path / "exports").glob("*.csv"))
        assert len(csv_files) == 1
        content = csv_files[0].read_text()
        assert "starred" in content
        assert "2401.00001" in content


class TestMarkdownTable:
    """Tests for format_papers_as_markdown_table function."""

    def test_header_and_separator(self, make_paper):
        papers = [make_paper()]
        table = format_papers_as_markdown_table(papers)
        lines = table.split("\n")
        assert lines[0].startswith("| arXiv ID |")
        assert lines[1].startswith("|-------")
        assert len(lines) == 3  # header + separator + 1 data row

    def test_arxiv_link(self, make_paper):
        paper = make_paper(arxiv_id="2401.12345")
        table = format_papers_as_markdown_table([paper])
        assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in table

    def test_pipe_escaping(self, make_paper):
        paper = make_paper(title="A | B", categories="cs.AI | stat.ML")
        table = format_papers_as_markdown_table([paper])
        data_line = table.split("\n")[2]
        assert "A \\| B" in data_line
        assert "cs.AI \\| stat.ML" in data_line

    def test_author_truncation_over_three(self, make_paper):
        paper = make_paper(authors="Alice, Bob, Carol, Dave")
        table = format_papers_as_markdown_table([paper])
        data_line = table.split("\n")[2]
        assert "Alice et al." in data_line
        assert "Bob" not in data_line

    def test_author_no_truncation_three_or_fewer(self, make_paper):
        paper = make_paper(authors="Alice, Bob, Carol")
        table = format_papers_as_markdown_table([paper])
        data_line = table.split("\n")[2]
        assert "Alice" in data_line
        assert "Bob" in data_line
        assert "Carol" in data_line
        assert "et al." not in data_line

    def test_empty_list(self):
        table = format_papers_as_markdown_table([])
        lines = table.split("\n")
        assert len(lines) == 2  # header + separator, no data rows

    def test_multiple_papers(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00001", title="Paper One"),
            make_paper(arxiv_id="2401.00002", title="Paper Two"),
        ]
        table = format_papers_as_markdown_table(papers)
        lines = table.split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows
        assert "Paper One" in lines[2]
        assert "Paper Two" in lines[3]


class TestSummaryModes:
    """Tests for SUMMARY_MODES constant and mode templates."""

    def test_summary_modes_has_expected_keys(self):
        expected = {"default", "quick", "tldr", "methods", "results", "comparison"}
        assert set(SUMMARY_MODES.keys()) == expected

    def test_each_mode_has_description_and_template(self):
        for mode_name, (desc, template) in SUMMARY_MODES.items():
            assert isinstance(desc, str) and len(desc) > 0, f"{mode_name} missing description"
            assert isinstance(template, str) and len(template) > 0, f"{mode_name} missing template"

    def test_all_templates_contain_paper_content_placeholder(self):
        for mode_name, (_, template) in SUMMARY_MODES.items():
            assert "{paper_content}" in template, (
                f"Mode '{mode_name}' template missing {{paper_content}} placeholder"
            )

    def test_all_templates_contain_title_placeholder(self):
        for mode_name, (_, template) in SUMMARY_MODES.items():
            assert "{title}" in template, (
                f"Mode '{mode_name}' template missing {{title}} placeholder"
            )

    def test_default_mode_uses_default_llm_prompt(self):
        _, template = SUMMARY_MODES["default"]
        assert template == DEFAULT_LLM_PROMPT


class TestSummaryModeModal:
    """Tests for the SummaryModeModal dismiss values."""

    def test_modal_returns_mode_names(self):
        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        # Each action should produce the correct mode name
        assert hasattr(modal, "action_mode_default")
        assert hasattr(modal, "action_mode_quick")
        assert hasattr(modal, "action_mode_tldr")
        assert hasattr(modal, "action_mode_methods")
        assert hasattr(modal, "action_mode_results")
        assert hasattr(modal, "action_mode_comparison")
        assert hasattr(modal, "action_cancel")

    def test_modal_bindings(self):
        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        binding_keys = {b.key for b in modal.BINDINGS}
        assert {"d", "q", "t", "m", "r", "c", "escape"} <= binding_keys


class TestSummaryDbMigration:
    """Tests for the DB schema migration from single to composite PK."""

    def test_creates_new_db_with_composite_pk(self, tmp_path):
        import sqlite3

        from tests.support.canonical_exports import _init_summary_db

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries'"
            ).fetchone()
            assert "PRIMARY KEY (arxiv_id, command_hash)" in row[0]

    def test_migrates_old_single_pk_schema(self, tmp_path):
        import sqlite3

        from tests.support.canonical_exports import _init_summary_db

        db_path = tmp_path / "summaries.db"
        # Create old schema
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "CREATE TABLE summaries ("
                "  arxiv_id TEXT PRIMARY KEY,"
                "  summary TEXT NOT NULL,"
                "  command_hash TEXT NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "INSERT INTO summaries VALUES ('2401.00001', 'old summary', 'abc', '2024-01-01')"
            )

        # Migrate
        _init_summary_db(db_path)

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries'"
            ).fetchone()
            assert "PRIMARY KEY (arxiv_id, command_hash)" in row[0]
            # Old data is gone (cache table, safe to drop)
            count = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            assert count == 0

    def test_composite_pk_allows_multiple_modes(self, tmp_path):
        from tests.support.canonical_exports import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)

        _save_summary(db_path, "2401.00001", "default summary", "hash_default")
        _save_summary(db_path, "2401.00001", "tldr summary", "hash_tldr")

        assert _load_summary(db_path, "2401.00001", "hash_default") == "default summary"
        assert _load_summary(db_path, "2401.00001", "hash_tldr") == "tldr summary"


class TestSummaryModeDisplay:
    """Tests for mode label display in AI Summary header."""

    def test_summary_header_includes_mode_label(self, make_paper):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper,
            abstract_text="test abstract",
            summary="This is a summary",
            summary_mode="TLDR",
        )
        content = details.content
        assert "TLDR" in content
        assert "AI Summary" in content

    def test_summary_header_no_mode_for_default(self, make_paper):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper,
            abstract_text="test abstract",
            summary="This is a summary",
            summary_mode="",
        )
        content = details.content
        assert "AI Summary" in content
        # No mode label in parentheses when empty
        assert "()" not in content

    def test_summary_loading_header_includes_mode(self, make_paper):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper,
            abstract_text="test abstract",
            summary_loading=True,
            summary_mode="METHODS",
        )
        content = details.content
        assert "METHODS" in content
        assert "Generating summary" in content


class TestSummaryModePromptResolution:
    """Tests for prompt template resolution per mode."""

    def test_compute_command_hash_varies_by_template(self):
        from tests.support.canonical_exports import _compute_command_hash

        hash1 = _compute_command_hash("cmd", SUMMARY_MODES["default"][1])
        hash2 = _compute_command_hash("cmd", SUMMARY_MODES["tldr"][1])
        assert hash1 != hash2

    def test_each_mode_produces_unique_hash(self):
        from tests.support.canonical_exports import _compute_command_hash

        hashes = set()
        for mode_name, (_, template) in SUMMARY_MODES.items():
            h = _compute_command_hash("test_cmd", template)
            assert h not in hashes, f"Mode '{mode_name}' has duplicate hash"
            hashes.add(h)
