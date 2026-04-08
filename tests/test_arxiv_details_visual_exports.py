#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.browser.core import SUBPROCESS_TIMEOUT
from arxiv_browser.config import (
    export_metadata,
    import_metadata,
    load_config,
    save_config,
)
from arxiv_browser.export import (
    escape_bibtex,
    extract_year,
    format_collection_as_markdown,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    generate_citation_key,
    get_pdf_download_path,
)
from arxiv_browser.llm import (
    DEFAULT_LLM_PROMPT,
    LLM_PRESETS,
    SUMMARY_MODES,
    build_llm_prompt,
    get_summary_db_path,
)
from arxiv_browser.models import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    Paper,
    PaperCollection,
    PaperMetadata,
    QueryToken,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
)
from arxiv_browser.parsing import (
    ARXIV_DATE_FORMAT,
    build_arxiv_search_query,
    clean_latex,
    extract_text_from_html,
    normalize_arxiv_id,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
    parse_arxiv_version_map,
)
from arxiv_browser.query import (
    format_categories,
    format_summary_as_rich,
    insert_implicit_and,
    pill_label_for_token,
    reconstruct_query,
    to_rpn,
    tokenize_query,
)
from arxiv_browser.themes import (
    DEFAULT_CATEGORY_COLOR,
    TAG_NAMESPACE_COLORS,
    THEME_NAMES,
    THEMES,
    get_tag_color,
    parse_tag_namespace,
)

# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestPaperDetailsCacheIntegration:
    """Tests for PaperDetails.update_paper cache behavior."""

    def test_cache_hit_returns_same_markup(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001", title="Cache Test")

        details.update_paper(paper, "abstract text")
        first_content = str(details.content)
        cache_size_after_first = len(details._detail_cache)

        details.update_paper(paper, "abstract text")
        second_content = str(details.content)
        cache_size_after_second = len(details._detail_cache)

        assert first_content == second_content
        assert cache_size_after_first == cache_size_after_second == 1

    def test_cache_miss_on_different_abstract(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract version 1")
        details.update_paper(paper, "abstract version 2")

        assert len(details._detail_cache) == 2

    def test_cache_miss_on_different_tags(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", tags=["ml"])
        details.update_paper(paper, "abstract", tags=["ml", "cv"])

        assert len(details._detail_cache) == 2

    def test_cache_miss_on_summary_loading_change(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", summary_loading=False)
        details.update_paper(paper, "abstract", summary_loading=True)

        assert len(details._detail_cache) == 2

    def test_cache_miss_on_collapsed_sections_change(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", collapsed_sections=[])
        details.update_paper(paper, "abstract", collapsed_sections=["authors"])

        assert len(details._detail_cache) == 2

    def test_cache_stores_in_order_list(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()

        for i in range(3):
            paper = make_paper(arxiv_id=f"2401.{i:05d}")
            details.update_paper(paper, f"abstract {i}")

        assert len(details._detail_cache_order) == 3
        assert len(details._detail_cache) == 3

    def test_none_paper_does_not_cache(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        details.update_paper(None)

        assert len(details._detail_cache) == 0
        assert details.paper is None

    def test_cache_eviction_removes_oldest(self, make_paper):
        from arxiv_browser.widgets.details import (
            DETAIL_CACHE_MAX,
            PaperDetails,
        )

        details = PaperDetails()

        for i in range(DETAIL_CACHE_MAX):
            paper = make_paper(arxiv_id=f"2401.{i:05d}")
            details.update_paper(paper, f"abstract {i}")

        first_key = details._detail_cache_order[0]
        assert first_key in details._detail_cache

        paper = make_paper(arxiv_id="2401.99999")
        details.update_paper(paper, "new abstract")

        assert len(details._detail_cache) == DETAIL_CACHE_MAX
        assert first_key not in details._detail_cache
        assert first_key not in details._detail_cache_order

    def test_cache_hit_does_not_reorder(self, make_paper):
        """A cache hit should not change the order list."""
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()

        paper_a = make_paper(arxiv_id="2401.00001")
        paper_b = make_paper(arxiv_id="2401.00002")

        details.update_paper(paper_a, "abstract a")
        details.update_paper(paper_b, "abstract b")

        order_before = list(details._detail_cache_order)

        details.update_paper(paper_a, "abstract a")

        order_after = list(details._detail_cache_order)
        assert order_before == order_after

    def test_cache_with_relevance_data(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", relevance=(8, "Good match"))
        details.update_paper(paper, "abstract", relevance=(3, "Poor match"))

        assert len(details._detail_cache) == 2

    def test_cache_with_version_update(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", version_update=None)
        details.update_paper(paper, "abstract", version_update=(1, 3))

        assert len(details._detail_cache) == 2


class TestTextualThemes:
    """Tests for the Textual theme system with custom CSS variables."""

    def test_build_textual_theme_maps_all_keys(self):
        from arxiv_browser.themes import (
            DEFAULT_THEME,
            _build_textual_theme,
        )

        theme = _build_textual_theme("test", DEFAULT_THEME)
        assert theme.name == "test"
        expected_vars = {
            "th-background",
            "th-panel",
            "th-panel-alt",
            "th-highlight",
            "th-highlight-focus",
            "th-accent",
            "th-accent-alt",
            "th-muted",
            "th-text",
            "th-green",
            "th-orange",
            "th-purple",
            "th-scrollbar-bg",
            "th-scrollbar-thumb",
            "th-scrollbar-active",
            "th-scrollbar-hover",
        }
        assert set(theme.variables.keys()) == expected_vars

    def test_textual_themes_key_parity(self):
        from arxiv_browser.themes import TEXTUAL_THEMES

        theme_names = list(TEXTUAL_THEMES.keys())
        assert len(theme_names) == 4
        keys_0 = set(TEXTUAL_THEMES[theme_names[0]].variables.keys())
        for name in theme_names[1:]:
            assert set(TEXTUAL_THEMES[name].variables.keys()) == keys_0

    def test_textual_theme_values_are_hex_colors(self):
        from arxiv_browser.themes import TEXTUAL_THEMES

        for name, theme in TEXTUAL_THEMES.items():
            for var_name, value in theme.variables.items():
                assert value.startswith("#"), (
                    f"Theme '{name}' variable '{var_name}' has non-hex value: {value}"
                )


class TestFooterContrast:
    """Tests for footer rendering with accent-colored keys."""

    def test_footer_contrast_uses_accent_for_keys(self):
        from arxiv_browser.themes import THEME_COLORS
        from arxiv_browser.widgets.chrome import ContextFooter

        footer = ContextFooter()
        footer.render_bindings([("o", "open"), ("s", "sort")])
        rendered = str(footer.content)
        assert THEME_COLORS["accent"] in rendered

    def test_footer_mode_badge_renders(self):
        from arxiv_browser.themes import THEME_COLORS
        from arxiv_browser.widgets.chrome import ContextFooter

        footer = ContextFooter()
        badge = f"[bold {THEME_COLORS['accent']}] SEARCH [/]"
        footer.render_bindings([("Esc", "close")], mode_badge=badge)
        rendered = str(footer.content)
        assert "SEARCH" in rendered

    def test_footer_mode_badge_default_empty(self):
        """In default browsing state, mode badge should be empty string."""
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._relevance_scoring_active = False
        app._version_checking = False
        app._version_progress = None
        app._in_arxiv_api_mode = False
        app.selected_ids = set()
        # Mock query_one to raise NoMatches for search container
        app.query_one = MagicMock(side_effect=NoMatches())
        badge = app._get_footer_mode_badge()
        assert badge == ""

    def test_footer_mode_badge_api(self):
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._relevance_scoring_active = False
        app._version_checking = False
        app._version_progress = None
        app._in_arxiv_api_mode = True
        app.selected_ids = set()
        app.query_one = MagicMock(side_effect=NoMatches())
        badge = app._get_footer_mode_badge()
        assert "API" in badge

    def test_footer_mode_badge_selection(self):
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._relevance_scoring_active = False
        app._version_checking = False
        app._version_progress = None
        app._in_arxiv_api_mode = False
        app.selected_ids = {"2401.00001", "2401.00002", "2401.00003"}
        app.query_one = MagicMock(side_effect=NoMatches())
        badge = app._get_footer_mode_badge()
        assert "3 SEL" in badge


class TestDetailPaneOrdering:
    """Tests for the abstract-before-authors ordering in the detail pane."""

    def test_abstract_before_authors_in_detail(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(authors="Alice, Bob, Charlie")
        details.update_paper(paper, "This is the abstract text")
        rendered = str(details.content)
        abstract_pos = rendered.find("Abstract")
        authors_pos = rendered.find("Authors")
        assert abstract_pos > 0 and authors_pos > 0
        assert abstract_pos < authors_pos, "Abstract should appear before Authors"

    def test_url_always_visible_without_collapse_option(self):
        """URL should not appear in DETAIL_SECTION_KEYS (not collapsible)."""
        from arxiv_browser.models import (
            DETAIL_SECTION_KEYS,
            DETAIL_SECTION_NAMES,
        )

        assert "url" not in DETAIL_SECTION_KEYS
        assert "url" not in DETAIL_SECTION_NAMES


class TestFooterDiscoverability:
    """Tests for default browsing footer discoverability."""

    @staticmethod
    def _make_footer_app(config):
        """Helper to create a minimal ArxivBrowser mock for footer tests."""
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._version_checking = False
        app._version_progress = None
        app._in_arxiv_api_mode = False
        app.selected_ids = set()
        app._s2_active = False
        app._config = config
        app._history_files = []
        app._watch_filter_active = False
        app._download_queue = []
        app._downloading = set()
        app._download_total = 0
        app._download_results = {}
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app.query_one = MagicMock(side_effect=NoMatches())
        return app

    def test_footer_shows_default_browse_order(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig()
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        assert bindings == [
            ("/", "search"),
            ("Space", "select"),
            ("o", "open"),
            ("s", "sort"),
            ("r", "read"),
            ("x", "star"),
            ("E", "export"),
            ("Ctrl+p", "commands"),
            ("?", "help"),
        ]

    def test_footer_shows_export_hint(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig()
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        keys = [k for k, _ in bindings]
        assert "E" in keys
        assert "Space" in keys

    def test_footer_is_capped_and_keeps_help_palette(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_preset="claude")
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        keys = [k for k, _ in bindings]
        assert len(bindings) <= 10
        assert "Ctrl+p" in keys
        assert "?" in keys

    def test_footer_keeps_core_actions_when_s2_active(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig()
        app = self._make_footer_app(config)
        app._s2_active = True
        bindings = app._get_footer_bindings()
        assert bindings == [
            ("/", "search"),
            ("Space", "select"),
            ("o", "open"),
            ("s", "sort"),
            ("r", "read"),
            ("x", "star"),
            ("E", "export"),
            ("Ctrl+p", "commands"),
            ("?", "help"),
        ]

    def test_footer_uses_palette_and_history_labels(self):
        from datetime import date as dt_date

        from arxiv_browser.models import UserConfig

        config = UserConfig()
        app = self._make_footer_app(config)
        app._history_files = [
            (dt_date(2026, 1, 2), Path("history/2026-01-02.txt")),
            (dt_date(2026, 1, 1), Path("history/2026-01-01.txt")),
        ]
        bindings = app._get_footer_bindings()
        assert len(bindings) == 9
        assert ("Ctrl+p", "commands") in bindings
        assert ("[/]", "dates") in bindings
        assert ("x", "star") not in bindings

    def test_search_and_api_footer_copy(self):
        from arxiv_browser.browser.contracts import FOOTER_CONTEXTS

        assert ("type to search", "") in FOOTER_CONTEXTS["search"]
        assert ("Enter", "apply") in FOOTER_CONTEXTS["search"]
        assert ("Esc", "close") in FOOTER_CONTEXTS["search"]
        assert ("↑↓", "move") in FOOTER_CONTEXTS["search"]
        assert ("[/]", "page") in FOOTER_CONTEXTS["api"]
        assert ("Esc/Ctrl+e", "exit") in FOOTER_CONTEXTS["api"]
        assert ("A", "new query") in FOOTER_CONTEXTS["api"]

    def test_footer_context_alias_matches_chrome_builders(self):
        from arxiv_browser.browser.contracts import FOOTER_CONTEXTS
        from arxiv_browser.widgets.chrome import (
            build_api_footer_bindings,
            build_browse_footer_bindings,
            build_search_footer_bindings,
            build_selection_footer_base_bindings,
        )

        assert FOOTER_CONTEXTS["default"] == build_browse_footer_bindings(
            s2_active=False,
            has_starred=False,
            llm_configured=False,
            has_history_navigation=False,
        )
        assert FOOTER_CONTEXTS["selection"] == build_selection_footer_base_bindings()
        assert FOOTER_CONTEXTS["search"] == build_search_footer_bindings()
        assert FOOTER_CONTEXTS["api"] == build_api_footer_bindings()


class TestExportMetadata:
    """Tests for export_metadata()."""

    def test_empty_config_exports_structure(self):
        config = UserConfig()
        result = export_metadata(config)
        assert result["format"] == "arxiv-browser-metadata"
        assert result["version"] == 1
        assert "exported_at" in result
        assert result["paper_metadata"] == {}
        assert result["watch_list"] == []
        assert result["bookmarks"] == []
        assert result["research_interests"] == ""

    def test_only_annotated_papers_exported(self, make_paper):
        config = UserConfig()
        config.paper_metadata["2401.0001"] = PaperMetadata(
            arxiv_id="2401.0001", notes="important", tags=["topic:ml"], is_read=True, starred=True
        )
        # Paper with no annotations (default state) — should be excluded
        config.paper_metadata["2401.0002"] = PaperMetadata(arxiv_id="2401.0002")
        result = export_metadata(config)
        assert "2401.0001" in result["paper_metadata"]
        assert "2401.0002" not in result["paper_metadata"]

    def test_watch_list_and_bookmarks_exported(self):
        config = UserConfig()
        config.watch_list = [WatchListEntry(pattern="transformer", match_type="keyword")]
        config.bookmarks = [SearchBookmark(name="ML", query="cat:cs.LG")]
        config.research_interests = "LLM inference"
        result = export_metadata(config)
        assert len(result["watch_list"]) == 1
        assert result["watch_list"][0]["pattern"] == "transformer"
        assert len(result["bookmarks"]) == 1
        assert result["bookmarks"][0]["query"] == "cat:cs.LG"
        assert result["research_interests"] == "LLM inference"

    def test_roundtrip_preserves_data(self):
        config = UserConfig()
        config.paper_metadata["2401.0001"] = PaperMetadata(
            arxiv_id="2401.0001",
            notes="my notes",
            tags=["topic:ml", "status:read"],
            is_read=True,
            starred=True,
            last_checked_version=3,
        )
        config.watch_list = [
            WatchListEntry(pattern="GPT", match_type="keyword", case_sensitive=True)
        ]
        config.bookmarks = [SearchBookmark(name="AI", query="cat:cs.AI")]
        config.research_interests = "quantization"

        exported = export_metadata(config)

        # Import into a fresh config
        fresh = UserConfig()
        papers_n, watch_n, bk_n, _ = import_metadata(exported, fresh)
        assert papers_n == 1
        assert watch_n == 1
        assert bk_n == 1
        meta = fresh.paper_metadata["2401.0001"]
        assert meta.notes == "my notes"
        assert meta.tags == ["topic:ml", "status:read"]
        assert meta.is_read is True
        assert meta.starred is True
        assert meta.last_checked_version == 3
        assert fresh.watch_list[0].pattern == "GPT"
        assert fresh.watch_list[0].case_sensitive is True
        assert fresh.bookmarks[0].query == "cat:cs.AI"
        assert fresh.research_interests == "quantization"


class TestImportMetadata:
    """Tests for import_metadata()."""

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Not a valid"):
            import_metadata({"format": "wrong"}, UserConfig())

    def test_merge_preserves_existing_notes(self):
        config = UserConfig()
        config.paper_metadata["2401.0001"] = PaperMetadata(
            arxiv_id="2401.0001", notes="existing notes"
        )
        data = {
            "format": "arxiv-browser-metadata",
            "paper_metadata": {
                "2401.0001": {"notes": "new notes", "tags": ["topic:new"], "is_read": True}
            },
        }
        papers_n, _, _, _ = import_metadata(data, config)
        assert papers_n == 1
        assert config.paper_metadata["2401.0001"].notes == "existing notes"
        assert "topic:new" in config.paper_metadata["2401.0001"].tags
        assert config.paper_metadata["2401.0001"].is_read is True

    def test_merge_fills_empty_notes(self):
        config = UserConfig()
        config.paper_metadata["2401.0001"] = PaperMetadata(arxiv_id="2401.0001")
        data = {
            "format": "arxiv-browser-metadata",
            "paper_metadata": {"2401.0001": {"notes": "imported notes"}},
        }
        import_metadata(data, config)
        assert config.paper_metadata["2401.0001"].notes == "imported notes"

    def test_merge_deduplicates_tags(self):
        config = UserConfig()
        config.paper_metadata["2401.0001"] = PaperMetadata(
            arxiv_id="2401.0001", tags=["topic:ml", "status:read"]
        )
        data = {
            "format": "arxiv-browser-metadata",
            "paper_metadata": {"2401.0001": {"tags": ["topic:ml", "topic:new"]}},
        }
        import_metadata(data, config)
        tags = config.paper_metadata["2401.0001"].tags
        assert tags.count("topic:ml") == 1
        assert "topic:new" in tags

    def test_replace_mode_overwrites(self):
        config = UserConfig()
        config.paper_metadata["2401.0001"] = PaperMetadata(
            arxiv_id="2401.0001", notes="old", starred=True
        )
        data = {
            "format": "arxiv-browser-metadata",
            "paper_metadata": {"2401.0001": {"notes": "new", "starred": False}},
        }
        import_metadata(data, config, merge=False)
        assert config.paper_metadata["2401.0001"].notes == "new"
        assert config.paper_metadata["2401.0001"].starred is False

    def test_watch_list_deduplication(self):
        config = UserConfig()
        config.watch_list = [WatchListEntry(pattern="GPT", match_type="keyword")]
        data = {
            "format": "arxiv-browser-metadata",
            "watch_list": [
                {"pattern": "GPT", "match_type": "keyword"},  # Duplicate
                {"pattern": "BERT", "match_type": "keyword"},  # New
            ],
        }
        _, watch_n, _, _ = import_metadata(data, config)
        assert watch_n == 1
        assert len(config.watch_list) == 2

    def test_bookmarks_capped_at_9(self):
        config = UserConfig()
        config.bookmarks = [SearchBookmark(name=f"B{i}", query=f"q{i}") for i in range(8)]
        data = {
            "format": "arxiv-browser-metadata",
            "bookmarks": [
                {"name": "New1", "query": "new1"},
                {"name": "New2", "query": "new2"},
                {"name": "New3", "query": "new3"},
            ],
        }
        _, _, bk_n, _ = import_metadata(data, config)
        assert bk_n == 1
        assert len(config.bookmarks) == 9

    def test_research_interests_imported_only_if_empty(self):
        config = UserConfig()
        config.research_interests = "existing interests"
        data = {
            "format": "arxiv-browser-metadata",
            "research_interests": "new interests",
        }
        import_metadata(data, config)
        assert config.research_interests == "existing interests"

        # But imports into empty
        config2 = UserConfig()
        import_metadata(data, config2)
        assert config2.research_interests == "new interests"

    def test_malformed_entries_skipped(self):
        config = UserConfig()
        data = {
            "format": "arxiv-browser-metadata",
            "paper_metadata": {
                "good": {"notes": "ok"},
                "bad": "not a dict",
            },
            "watch_list": [
                {"pattern": "ok", "match_type": "keyword"},
                "not a dict",
            ],
        }
        papers_n, watch_n, _, _ = import_metadata(data, config)
        assert papers_n == 1
        assert watch_n == 1
