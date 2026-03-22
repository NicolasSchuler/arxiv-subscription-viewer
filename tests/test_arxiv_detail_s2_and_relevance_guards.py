#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.app import (
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
from arxiv_browser.themes import THEME_NAMES, THEMES

# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestMatchTypeValidation:
    """Verify WatchListEntry.match_type is validated during deserialization."""

    def test_valid_match_types_preserved(self):
        from arxiv_browser.app import WATCH_MATCH_TYPES, _dict_to_config

        for mt in WATCH_MATCH_TYPES:
            data = {"watch_list": [{"pattern": "test", "match_type": mt}]}
            config = _dict_to_config(data)
            assert config.watch_list[0].match_type == mt

    def test_invalid_match_type_defaults_to_author(self, caplog):
        import logging

        from arxiv_browser.app import _dict_to_config

        data = {"watch_list": [{"pattern": "test", "match_type": "category"}]}
        with caplog.at_level(logging.WARNING, logger="arxiv_browser"):
            config = _dict_to_config(data)
        assert config.watch_list[0].match_type == "author"
        assert "Invalid watch list match_type" in caplog.text


class TestAtomicBibtexExport:
    """Verify BibTeX file export uses atomic writes."""

    def test_export_creates_valid_file(self, make_paper, tmp_path):
        """BibTeX export should produce a valid file with no leftover temp files."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        paper = make_paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="cs.AI",
        )
        from arxiv_browser.app import UserConfig

        config = UserConfig(bibtex_export_dir=str(tmp_path / "exports"))
        app = ArxivBrowser([paper], config=config, restore_session=False)

        with patch("arxiv_browser.app.save_config", return_value=True):

            async def run_test():
                async with app.run_test() as pilot:
                    await pilot.pause(0.2)  # Let detail pane debounce fire
                    app.action_export_bibtex_file()

                    export_dir = tmp_path / "exports"
                    # Check that the export directory was created
                    assert export_dir.exists()
                    # Check that exactly one .bib file was created
                    bib_files = list(export_dir.glob("*.bib"))
                    assert len(bib_files) == 1
                    # Check no leftover temp files
                    tmp_files = list(export_dir.glob(".bibtex-*"))
                    assert len(tmp_files) == 0
                    # Check content is valid BibTeX
                    content = bib_files[0].read_text()
                    assert "@misc{" in content
                    assert "Smith" in content

            import asyncio

            asyncio.run(run_test())


class TestDetailStateBuilder:
    """Verify _build_detail_state aggregates all detail pane state."""

    def test_build_detail_state_returns_expected_fields(self, make_paper):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.themes import build_theme_runtime

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._paper_summaries = {"2401.00001": "A summary"}
        app._summary_loading = set()
        app._highlight_terms = {"abstract": ["test"]}
        app._s2_active = False
        app._s2_cache = {}
        app._s2_loading = set()
        app._hf_active = False
        app._hf_cache = {}
        app._version_updates = {}
        app._summary_mode_label = {"2401.00001": "tldr"}
        app._config = type(
            "Config",
            (),
            {
                "paper_metadata": {},
                "collapsed_sections": ["tags", "relevance", "summary", "s2", "hf", "version"],
                "theme_name": "monokai",
                "theme": {},
                "category_colors": {},
            },
        )()
        app._relevance_scores = {"2401.00001": (8, "relevant")}
        app._resolved_theme_runtime = MagicMock(return_value=build_theme_runtime("monokai"))
        paper = make_paper(arxiv_id="2401.00001")

        state = app._build_detail_state("2401.00001", paper)
        assert state.summary == "A summary"
        assert state.summary_loading is False
        assert state.highlight_terms == ("test",)
        assert state.s2_data is None
        assert state.s2_loading is False
        assert state.hf_data is None
        assert state.version_update is None
        assert state.summary_mode == "tldr"
        assert state.tags == ()
        assert state.relevance == (8, "relevant")
        assert state.collapsed_sections == (
            "tags",
            "relevance",
            "summary",
            "s2",
            "hf",
            "version",
        )
        assert state.detail_mode == "scan"


class TestArxivBrowserConstructorCompatibility:
    """Verify public constructor compatibility is preserved."""

    def test_legacy_positional_constructor_arguments_still_work(self, make_paper):
        from datetime import date as dt_date

        from arxiv_browser.app import ArxivBrowser

        config = UserConfig(show_abstract_preview=True)
        history_files = [(dt_date(2026, 3, 22), Path("history/2026-03-22.txt"))]
        services = object()

        app = ArxivBrowser(
            [make_paper(arxiv_id="2401.00001")],
            config,
            False,
            history_files,
            0,
            True,
            services,
        )

        assert app._config is config
        assert app._restore_session is False
        assert app._history_files == history_files
        assert app._current_date_index == 0
        assert app._show_abstract_preview is True
        assert app._services is services


class TestDetailPaneHighlighting:
    """Verify search terms are highlighted in the detail pane abstract."""

    def test_highlight_terms_applied(self, make_paper):
        from arxiv_browser.app import THEME_COLORS, PaperDetails

        details = PaperDetails()
        paper = make_paper(abstract="Deep learning is transforming AI research.")

        details.update_paper(
            paper,
            abstract_text="Deep learning is transforming AI research.",
            highlight_terms=["learning"],
        )

        rendered = str(details.content)
        assert f"[bold {THEME_COLORS['accent']}]learning[/]" in rendered

    def test_no_highlight_without_terms(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(abstract="Deep learning is transforming AI research.")

        details.update_paper(
            paper,
            abstract_text="Deep learning is transforming AI research.",
        )

        rendered = str(details.content)
        # Without highlight terms, "learning" should NOT have bold markup around it
        assert "learning[/]" not in rendered


class TestHighlightPatternCache:
    """Verify highlight_text caches compiled regex patterns."""

    def test_cache_populated(self):
        from arxiv_browser.app import _HIGHLIGHT_PATTERN_CACHE, highlight_text

        _HIGHLIGHT_PATTERN_CACHE.clear()
        highlight_text("Hello world", ["world"], "#ff0000")
        assert ("world",) in _HIGHLIGHT_PATTERN_CACHE

    def test_cache_reused(self):
        from arxiv_browser.app import _HIGHLIGHT_PATTERN_CACHE, highlight_text

        _HIGHLIGHT_PATTERN_CACHE.clear()
        highlight_text("Hello world", ["world"], "#ff0000")
        pattern1 = _HIGHLIGHT_PATTERN_CACHE[("world",)]
        highlight_text("Goodbye world", ["world"], "#00ff00")
        pattern2 = _HIGHLIGHT_PATTERN_CACHE[("world",)]
        assert pattern1 is pattern2

    def test_cache_different_terms(self):
        from arxiv_browser.app import _HIGHLIGHT_PATTERN_CACHE, highlight_text

        _HIGHLIGHT_PATTERN_CACHE.clear()
        highlight_text("Hello world", ["world"], "#ff0000")
        highlight_text("Hello earth", ["earth"], "#ff0000")
        assert len(_HIGHLIGHT_PATTERN_CACHE) == 2

    def test_cache_has_bounded_size(self):
        from arxiv_browser.app import _HIGHLIGHT_PATTERN_CACHE, highlight_text
        from arxiv_browser.query import _HIGHLIGHT_PATTERN_CACHE_MAX

        _HIGHLIGHT_PATTERN_CACHE.clear()
        for i in range(_HIGHLIGHT_PATTERN_CACHE_MAX + 20):
            highlight_text("value", [f"term{i}"], "#ff0000")

        assert len(_HIGHLIGHT_PATTERN_CACHE) == _HIGHLIGHT_PATTERN_CACHE_MAX
        assert ("term0",) not in _HIGHLIGHT_PATTERN_CACHE
        assert (f"term{_HIGHLIGHT_PATTERN_CACHE_MAX + 19}",) in _HIGHLIGHT_PATTERN_CACHE


class TestBatchConfirmationThreshold:
    """Verify batch confirmation modal behavior."""

    def test_threshold_constant_exists(self):
        from arxiv_browser.app import BATCH_CONFIRM_THRESHOLD

        assert BATCH_CONFIRM_THRESHOLD == 10

    def test_confirm_modal_class_exists(self):
        from arxiv_browser.modals import ConfirmModal

        modal = ConfirmModal("Test message?")
        assert modal._message == "Test message?"

    def test_confirm_modal_compose_copy(self):
        import inspect

        from arxiv_browser.modals import ConfirmModal

        source = inspect.getsource(ConfirmModal.compose)
        assert "Confirm (y)" in source
        assert "Cancel (Esc)" in source
        assert "Confirm: y  Cancel: n / Esc" in source


@pytest.mark.integration
class TestBatchConfirmationIntegration:
    """Integration test for batch confirmation modal."""

    async def test_no_modal_below_threshold(self, make_paper):
        """Opening few papers should not show confirmation modal."""
        from unittest.mock import patch

        from arxiv_browser.app import BATCH_CONFIRM_THRESHOLD, ArxivBrowser

        papers = [make_paper(arxiv_id=f"2401.{i:05d}") for i in range(BATCH_CONFIRM_THRESHOLD)]
        app = ArxivBrowser(papers, restore_session=False)

        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch("arxiv_browser.app.webbrowser.open"),
        ):
            async with app.run_test() as pilot:
                # Select all papers
                await pilot.press("a")
                await pilot.pause(0.1)
                # Open URLs — should NOT show modal (at threshold, not above)
                await pilot.press("o")
                await pilot.pause(0.1)
                # No modal should be on screen stack
                assert len(app.screen_stack) == 1

    async def test_modal_shown_above_threshold(self, make_paper):
        """Opening many papers should show confirmation modal."""
        from unittest.mock import patch

        from arxiv_browser.app import BATCH_CONFIRM_THRESHOLD, ArxivBrowser
        from arxiv_browser.modals import ConfirmModal

        papers = [make_paper(arxiv_id=f"2401.{i:05d}") for i in range(BATCH_CONFIRM_THRESHOLD + 1)]
        app = ArxivBrowser(papers, restore_session=False)

        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch("arxiv_browser.app.webbrowser.open"),
        ):
            async with app.run_test() as pilot:
                # Select all papers
                await pilot.press("a")
                await pilot.pause(0.1)
                # Open URLs — should show modal
                await pilot.press("o")
                await pilot.pause(0.1)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], ConfirmModal)

                # Dismiss with 'n' (cancel)
                await pilot.press("n")
                await pilot.pause(0.1)
                assert len(app.screen_stack) == 1


class TestS2ConfigSerialization:
    """Tests for S2 config fields round-trip."""

    def test_s2_fields_default(self):
        from arxiv_browser.app import UserConfig

        config = UserConfig()
        assert config.s2_enabled is False
        assert config.s2_api_key == ""
        assert config.s2_cache_ttl_days == 7

    def test_s2_fields_serialize_roundtrip(self):
        from arxiv_browser.app import UserConfig, _config_to_dict, _dict_to_config

        config = UserConfig(s2_enabled=True, s2_api_key="test-key", s2_cache_ttl_days=14)
        data = _config_to_dict(config)
        assert data["s2_enabled"] is True
        assert data["s2_api_key"] == "test-key"
        assert data["s2_cache_ttl_days"] == 14

        restored = _dict_to_config(data)
        assert restored.s2_enabled is True
        assert restored.s2_api_key == "test-key"
        assert restored.s2_cache_ttl_days == 14

    def test_s2_fields_missing_in_data(self):
        from arxiv_browser.app import _dict_to_config

        data = {"version": 1}
        config = _dict_to_config(data)
        assert config.s2_enabled is False
        assert config.s2_api_key == ""
        assert config.s2_cache_ttl_days == 7

    def test_s2_fields_wrong_type(self):
        from arxiv_browser.app import _dict_to_config

        data = {
            "s2_enabled": "not-a-bool",
            "s2_api_key": 12345,
            "s2_cache_ttl_days": "seven",
        }
        config = _dict_to_config(data)
        assert config.s2_enabled is False
        assert config.s2_api_key == ""
        assert config.s2_cache_ttl_days == 7


class TestS2SortPapers:
    """Tests for citation sort."""

    def test_citation_sort_orders_by_count(self, make_paper):
        from arxiv_browser.app import sort_papers
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        papers = [
            make_paper(arxiv_id="a", title="A"),
            make_paper(arxiv_id="b", title="B"),
            make_paper(arxiv_id="c", title="C"),
        ]
        s2_cache = {
            "a": SemanticScholarPaper(
                arxiv_id="a",
                s2_paper_id="x",
                citation_count=10,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=None,
                url="",
            ),
            "b": SemanticScholarPaper(
                arxiv_id="b",
                s2_paper_id="y",
                citation_count=100,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=None,
                url="",
            ),
            "c": SemanticScholarPaper(
                arxiv_id="c",
                s2_paper_id="z",
                citation_count=50,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=None,
                url="",
            ),
        }
        result = sort_papers(papers, "citations", s2_cache=s2_cache)
        assert [p.arxiv_id for p in result] == ["b", "c", "a"]

    def test_citation_sort_papers_without_s2_last(self, make_paper):
        from arxiv_browser.app import sort_papers
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        papers = [
            make_paper(arxiv_id="a", title="A"),
            make_paper(arxiv_id="b", title="B"),
        ]
        s2_cache = {
            "a": SemanticScholarPaper(
                arxiv_id="a",
                s2_paper_id="x",
                citation_count=10,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=None,
                url="",
            ),
        }
        result = sort_papers(papers, "citations", s2_cache=s2_cache)
        assert result[0].arxiv_id == "a"
        assert result[1].arxiv_id == "b"

    def test_citation_sort_no_cache(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(arxiv_id="a", title="A"),
            make_paper(arxiv_id="b", title="B"),
        ]
        result = sort_papers(papers, "citations", s2_cache=None)
        # All papers have no S2 data, order is stable
        assert len(result) == 2


class TestS2DetailPane:
    """Tests for S2 section rendering in PaperDetails."""

    def test_s2_section_rendered(self, make_paper):
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        details = PaperDetails()
        paper = make_paper()
        s2 = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=42,
            influential_citation_count=5,
            tldr="Does cool things.",
            fields_of_study=("Computer Science",),
            year=2024,
            url="https://example.com",
        )
        details.update_paper(paper, "Abstract text", s2_data=s2)
        content = details.content
        assert "Semantic Scholar" in content
        assert "42" in content
        assert "5" in content
        assert "Does cool things." in content
        assert "Computer Science" in content

    def test_s2_loading_indicator(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(paper, "Abstract text", s2_loading=True)
        content = details.content
        assert "Semantic Scholar" in content
        assert "Fetching data" in content

    def test_s2_section_hidden_when_no_data(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(paper, "Abstract text")
        content = details.content
        assert "Semantic Scholar" not in content

    def test_s2_section_with_empty_tldr(self, make_paper):
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        details = PaperDetails()
        paper = make_paper()
        s2 = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=10,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=None,
            url="",
        )
        details.update_paper(paper, "Abstract text", s2_data=s2)
        content = details.content
        assert "Semantic Scholar" in content
        assert "TLDR" not in content  # Should not show empty TLDR

    def test_s2_section_with_empty_fields(self, make_paper):
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        details = PaperDetails()
        paper = make_paper()
        s2 = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=10,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=None,
            url="",
        )
        details.update_paper(paper, "Abstract text", s2_data=s2)
        content = details.content
        assert "Fields" not in content  # Should not show empty fields


class TestS2PaperListItem:
    """Tests for S2 citation badge in PaperListItem."""

    def test_citation_badge_shown(self, make_paper):
        from arxiv_browser.app import PaperListItem
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        paper = make_paper()
        item = PaperListItem(paper)
        s2 = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=42,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=None,
            url="",
        )
        item.update_s2_data(s2)
        meta = item._get_meta_text()
        assert "C42" in meta

    def test_no_badge_without_s2(self, make_paper):
        from arxiv_browser.app import PaperListItem

        paper = make_paper()
        item = PaperListItem(paper)
        meta = item._get_meta_text()
        assert "C0" not in meta


class TestS2RecsConversion:
    """Tests for _s2_recs_to_paper_tuples static method."""

    def test_converts_recs_to_paper_tuples(self):
        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        recs = [
            SemanticScholarPaper(
                arxiv_id="2401.00001",
                s2_paper_id="r1",
                citation_count=100,
                influential_citation_count=10,
                tldr="Great paper.",
                fields_of_study=("CS",),
                year=2024,
                url="https://example.com",
                title="Rec Paper 1",
                abstract="Abstract 1",
            ),
            SemanticScholarPaper(
                arxiv_id="2401.00002",
                s2_paper_id="r2",
                citation_count=50,
                influential_citation_count=5,
                tldr="OK paper.",
                fields_of_study=("Math",),
                year=2024,
                url="https://example.com/2",
                title="Rec Paper 2",
                abstract="Abstract 2",
            ),
        ]
        tuples = ArxivBrowser._s2_recs_to_paper_tuples(recs)
        assert len(tuples) == 2
        # First rec has 100/100 = 1.0 score
        assert tuples[0][0].title == "Rec Paper 1"
        assert tuples[0][1] == 1.0
        # Second rec has 50/100 = 0.5 score
        assert tuples[1][0].title == "Rec Paper 2"
        assert tuples[1][1] == 0.5
        # Papers should have source="s2"
        assert tuples[0][0].source == "s2"

    def test_zero_citations_handled(self):
        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        recs = [
            SemanticScholarPaper(
                arxiv_id="2401.00001",
                s2_paper_id="r1",
                citation_count=0,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=None,
                url="",
                title="Zero Cites",
                abstract="",
            ),
        ]
        tuples = ArxivBrowser._s2_recs_to_paper_tuples(recs)
        assert len(tuples) == 1
        # Score should be 0/1 = 0.0 (max_cites fallback to 1)
        assert tuples[0][1] == 0.0

    def test_url_fallback_with_empty_arxiv_id_and_url(self):
        """When both arxiv_id and url are empty, URL should be empty string."""
        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        recs = [
            SemanticScholarPaper(
                arxiv_id="",
                s2_paper_id="s2only",
                citation_count=10,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=2024,
                url="",
                title="S2 Only Paper",
                abstract="",
            ),
        ]
        tuples = ArxivBrowser._s2_recs_to_paper_tuples(recs)
        assert len(tuples) == 1
        # arxiv_id is empty so paper.arxiv_id should fall back to s2_paper_id
        assert tuples[0][0].arxiv_id == "s2only"
        # URL should be empty (not r.url which is also empty)
        assert tuples[0][0].url == ""

    def test_url_uses_arxiv_id_when_url_empty(self):
        """When url is empty but arxiv_id is present, URL should be constructed."""
        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        recs = [
            SemanticScholarPaper(
                arxiv_id="2401.99999",
                s2_paper_id="r1",
                citation_count=10,
                influential_citation_count=0,
                tldr="",
                fields_of_study=(),
                year=2024,
                url="",
                title="Has ArXiv ID",
                abstract="",
            ),
        ]
        tuples = ArxivBrowser._s2_recs_to_paper_tuples(recs)
        assert tuples[0][0].url == "https://arxiv.org/abs/2401.99999"


class TestS2AppActions:
    """Tests for S2 app action methods using mock self."""

    def _make_mock_app(self):
        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._s2_active = False
        app._s2_cache = {}
        app._s2_loading = set()
        app._s2_db_path = None
        app._badges_dirty = set()
        app._badge_timer = None
        app._config = type(
            "Config",
            (),
            {
                "s2_api_key": "",
                "s2_cache_ttl_days": 7,
                "s2_enabled": False,
            },
        )()
        return app

    def test_toggle_s2_flips_state(self):
        from unittest.mock import MagicMock, patch

        app = self._make_mock_app()
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()

        assert app._s2_active is False
        with patch("arxiv_browser.app.save_config", return_value=True) as save_mock:
            app.action_toggle_s2()
        assert app._s2_active is True
        assert app._config.s2_enabled is True
        save_mock.assert_called_once_with(app._config)
        app.notify.assert_called_once()
        assert "enabled" in app.notify.call_args[0][0]
        assert "press e on a paper" in app.notify.call_args[0][0]
        app._mark_badges_dirty.assert_called_once_with("s2", immediate=True)

        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_toggle_s2()
        assert app._s2_active is False
        assert app._config.s2_enabled is False
        assert "disabled" in app.notify.call_args[0][0]
        assert app._mark_badges_dirty.call_count == 2

    def test_toggle_s2_reverts_when_save_fails(self):
        from unittest.mock import MagicMock, patch

        app = self._make_mock_app()
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()

        with patch("arxiv_browser.app.save_config", return_value=False):
            app.action_toggle_s2()

        assert app._s2_active is False
        assert app._config.s2_enabled is False
        app._mark_badges_dirty.assert_not_called()
        assert "Failed to save Semantic Scholar setting" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_action_fetch_s2_dedupes_concurrent_calls(self, make_paper):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        paper = make_paper(arxiv_id="2401.42424")
        app = ArxivBrowser([paper], restore_session=False)

        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                app._s2_active = True
                track_calls = 0

                def fake_track_task(coro):
                    nonlocal track_calls
                    track_calls += 1
                    coro.close()
                    return MagicMock()

                app._fetch_s2_paper_async = AsyncMock(return_value=None)
                app._track_task = fake_track_task
                await asyncio.gather(app.action_fetch_s2(), app.action_fetch_s2())
                await pilot.pause(0)

                assert track_calls == 1
                assert app._fetch_s2_paper_async.call_count == 1

    def test_ctrl_e_dispatch_to_exit_api_mode(self):
        from unittest.mock import MagicMock

        app = self._make_mock_app()
        app._in_arxiv_api_mode = True
        app.action_exit_arxiv_search_mode = MagicMock()
        app.action_toggle_s2 = MagicMock()

        app.action_ctrl_e_dispatch()
        app.action_exit_arxiv_search_mode.assert_called_once()
        app.action_toggle_s2.assert_not_called()

    def test_ctrl_e_dispatch_to_toggle_s2(self):
        from unittest.mock import MagicMock

        app = self._make_mock_app()
        app._in_arxiv_api_mode = False
        app.action_exit_arxiv_search_mode = MagicMock()
        app.action_toggle_s2 = MagicMock()

        app.action_ctrl_e_dispatch()
        app.action_toggle_s2.assert_called_once()
        app.action_exit_arxiv_search_mode.assert_not_called()


class TestDownloadBatchGuards:
    """Tests for batch-overlap guards in PDF download flows."""

    def test_action_download_pdf_rejects_when_batch_active(self):
        from collections import deque
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._download_queue = deque([object()])
        app._downloading = set()
        app._download_total = 1
        app.notify = MagicMock()
        app._get_target_papers = MagicMock(return_value=[])

        app.action_download_pdf()

        app._get_target_papers.assert_not_called()
        app.notify.assert_called_once()
        assert "already in progress" in app.notify.call_args[0][0]

    def test_do_start_downloads_rejects_when_batch_active(self):
        from collections import deque
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._download_queue = deque()
        app._downloading = {"2401.12345"}
        app._download_total = 1
        app._download_results = {}
        app.notify = MagicMock()
        app._start_downloads = MagicMock()

        app._do_start_downloads([])

        app._start_downloads.assert_not_called()
        app.notify.assert_called_once()
        assert "already in progress" in app.notify.call_args[0][0]

    def test_reset_dataset_view_state_keeps_download_batch(self):
        from collections import deque
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.app import ArxivBrowser

        download_item = object()
        app = ArxivBrowser.__new__(ArxivBrowser)
        app._cancel_pending_detail_update = MagicMock()
        app._badge_timer = None
        app._sort_refresh_timer = None
        app._badges_dirty = set()
        app._sort_refresh_dirty = set()
        app._abstract_cache = {}
        app._abstract_loading = set()
        app._abstract_queue = deque()
        app._abstract_pending_ids = set()
        app._get_paper_details_widget = MagicMock(side_effect=NoMatches())
        app._paper_summaries = {}
        app._summary_loading = set()
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._s2_cache = {}
        app._s2_loading = set()
        app._s2_api_error = False
        app._hf_cache = {}
        app._hf_loading = False
        app._hf_api_error = False
        app._version_updates = {}
        app._version_checking = False
        app._version_progress = None
        app._relevance_scores = {}
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._download_queue = deque([download_item])
        app._downloading = {"2401.12345"}
        app._download_results = {"2401.12345": True}
        app._download_total = 1
        app._tfidf_index = object()
        app._tfidf_corpus_key = "corpus"
        app._pending_similarity_paper_id = "paper"
        app._tfidf_build_task = None

        app._reset_dataset_view_state()

        assert list(app._download_queue) == [download_item]
        assert app._downloading == {"2401.12345"}
        assert app._download_results == {"2401.12345": True}
        assert app._download_total == 1


class TestRelevanceScoringGuards:
    """Tests for relevance-scoring race prevention."""

    def test_start_relevance_sets_flag_before_task(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._relevance_scoring_active = False
        app.all_papers = []
        app.notify = MagicMock()

        def _track(coro):
            assert app._relevance_scoring_active is True
            coro.close()

        app._track_task = MagicMock(side_effect=_track)

        app._start_relevance_scoring("cmd {prompt}", "transformers")

        assert app._relevance_scoring_active is True
        assert app._track_task.call_count == 1

    def test_start_relevance_rejects_when_already_active(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._relevance_scoring_active = True
        app.notify = MagicMock()
        app._track_task = MagicMock()

        app._start_relevance_scoring("cmd {prompt}", "transformers")

        app._track_task.assert_not_called()
        app.notify.assert_called_once()
        assert "already in progress" in app.notify.call_args[0][0]

    def test_modal_callback_rejects_when_active(self):
        from unittest.mock import MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._relevance_scoring_active = True
        app.notify = MagicMock()
        app._start_relevance_scoring = MagicMock()
        app._config = UserConfig()

        with patch("arxiv_browser.app.save_config") as save:
            app._on_interests_saved_then_score("NLP", "cmd {prompt}")

        save.assert_not_called()
        app._start_relevance_scoring.assert_not_called()
        app.notify.assert_called_once()
        assert "already in progress" in app.notify.call_args[0][0]


class TestCitationGraphCaching:
    """Tests for citation graph empty-cache handling."""

    @pytest.mark.asyncio
    async def test_fetch_uses_cache_marker_even_for_empty_results(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._s2_db_path = tmp_path / "s2.db"
        app._http_client = AsyncMock()
        app._config = type("Config", (), {"s2_api_key": ""})()

        with (
            patch("arxiv_browser.app.has_s2_citation_graph_cache", return_value=True),
            patch("arxiv_browser.app.load_s2_citation_graph", side_effect=[[], []]) as load_cache,
            patch("arxiv_browser.app.fetch_s2_references", new_callable=AsyncMock) as fetch_refs,
            patch("arxiv_browser.app.fetch_s2_citations", new_callable=AsyncMock) as fetch_cites,
        ):
            refs, cites = await app._fetch_citation_graph("paper1")

        assert refs == []
        assert cites == []
        assert load_cache.call_count == 2
        fetch_refs.assert_not_called()
        fetch_cites.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_does_not_cache_when_fetch_incomplete(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._s2_db_path = tmp_path / "s2.db"
        app._http_client = AsyncMock()
        app._config = type("Config", (), {"s2_api_key": ""})()

        with (
            patch("arxiv_browser.app.has_s2_citation_graph_cache", return_value=False),
            patch("arxiv_browser.app.fetch_s2_references", new_callable=AsyncMock) as fetch_refs,
            patch("arxiv_browser.app.fetch_s2_citations", new_callable=AsyncMock) as fetch_cites,
            patch("arxiv_browser.app.save_s2_citation_graph") as save_graph,
        ):
            fetch_refs.return_value = ([], False)
            fetch_cites.return_value = ([], True)
            refs, cites = await app._fetch_citation_graph("paper1")

        assert refs == []
        assert cites == []
        save_graph.assert_not_called()
