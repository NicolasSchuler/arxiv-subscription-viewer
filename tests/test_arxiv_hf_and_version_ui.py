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


class TestHfConfigSerialization:
    """Tests for HuggingFace config fields in _config_to_dict / _dict_to_config."""

    def test_roundtrip_with_hf_fields(self):
        """HF config fields should survive serialization round-trip."""
        from arxiv_browser.app import _config_to_dict, _dict_to_config

        original = UserConfig(hf_enabled=True, hf_cache_ttl_hours=12)
        data = _config_to_dict(original)
        restored = _dict_to_config(data)
        assert restored.hf_enabled is True
        assert restored.hf_cache_ttl_hours == 12

    def test_defaults_when_absent(self):
        """HF config fields should use defaults when not present in data."""
        from arxiv_browser.app import _dict_to_config

        config = _dict_to_config({})
        assert config.hf_enabled is False
        assert config.hf_cache_ttl_hours == 6

    def test_wrong_type_uses_default(self):
        """Non-bool hf_enabled and non-int hf_cache_ttl_hours should use defaults."""
        from arxiv_browser.app import _dict_to_config

        config = _dict_to_config({"hf_enabled": "yes", "hf_cache_ttl_hours": "six"})
        assert config.hf_enabled is False
        assert config.hf_cache_ttl_hours == 6


class TestHfSortPapers:
    """Tests for trending sort via sort_papers()."""

    def test_sort_by_trending(self, make_paper):
        from arxiv_browser.app import sort_papers
        from arxiv_browser.huggingface import HuggingFacePaper

        papers = [
            make_paper(arxiv_id="a", title="A"),
            make_paper(arxiv_id="b", title="B"),
            make_paper(arxiv_id="c", title="C"),
        ]
        hf_cache = {
            "a": HuggingFacePaper("a", "A", 10, 0, "", (), "", 0),
            "c": HuggingFacePaper("c", "C", 50, 0, "", (), "", 0),
        }
        result = sort_papers(papers, "trending", hf_cache=hf_cache)
        # c (50 upvotes) should come first, then a (10), then b (no HF data)
        assert [p.arxiv_id for p in result] == ["c", "a", "b"]

    def test_sort_trending_without_cache(self, make_paper):
        """Trending sort with no hf_cache should just keep original order."""
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(arxiv_id="a", title="A"),
            make_paper(arxiv_id="b", title="B"),
        ]
        result = sort_papers(papers, "trending")
        # All papers have no HF data, so they should be stable-sorted
        assert [p.arxiv_id for p in result] == ["a", "b"]

    def test_papers_without_hf_sort_last(self, make_paper):
        """Papers without HF data should sort after papers with HF data."""
        from arxiv_browser.app import sort_papers
        from arxiv_browser.huggingface import HuggingFacePaper

        papers = [
            make_paper(arxiv_id="x"),
            make_paper(arxiv_id="y"),
        ]
        hf_cache = {
            "y": HuggingFacePaper("y", "Y", 5, 0, "", (), "", 0),
        }
        result = sort_papers(papers, "trending", hf_cache=hf_cache)
        assert result[0].arxiv_id == "y"
        assert result[1].arxiv_id == "x"


class TestHfDetailPane:
    """Tests for HuggingFace section in PaperDetails.update_paper()."""

    def test_hf_section_shown_when_data_present(self, make_paper):
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.huggingface import HuggingFacePaper

        details = PaperDetails()
        paper = make_paper()
        hf = HuggingFacePaper(
            "2401.12345", "Test", 42, 5, "A summary.", ("ML",), "https://github.com/test/repo", 100
        )
        details.update_paper(paper, "abstract text", hf_data=hf)
        content = details.content
        assert "HuggingFace" in content
        assert "42" in content  # upvotes
        assert "5" in content  # comments
        assert "A summary." in content
        assert "ML" in content  # keyword
        assert "github.com/test/repo" in content
        assert "100 stars" in content

    def test_hf_section_hidden_when_no_data(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(paper, "abstract text")
        content = details.content
        assert "HuggingFace" not in content

    def test_hf_section_optional_fields(self, make_paper):
        """HF section omits optional fields when empty."""
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.huggingface import HuggingFacePaper

        details = PaperDetails()
        paper = make_paper()
        hf = HuggingFacePaper("2401.12345", "Test", 10, 0, "", (), "", 0)
        details.update_paper(paper, "abstract text", hf_data=hf)
        content = details.content
        assert "HuggingFace" in content
        assert "10" in content  # upvotes
        assert "GitHub" not in content  # No github repo
        assert "Keywords" not in content  # No keywords
        assert "AI Summary" not in content  # No ai_summary


class TestHfPaperListItem:
    """Tests for HuggingFace badge in PaperListItem."""

    def test_hf_badge_present(self, make_paper):
        from arxiv_browser.app import PaperListItem
        from arxiv_browser.huggingface import HuggingFacePaper

        paper = make_paper()
        item = PaperListItem(paper)
        hf = HuggingFacePaper("2401.12345", "Test", 42, 0, "", (), "", 0)
        item.update_hf_data(hf)
        meta = item._get_meta_text()
        assert "\u219142" in meta

    def test_hf_badge_absent_when_no_data(self, make_paper):
        from arxiv_browser.app import PaperListItem

        paper = make_paper()
        item = PaperListItem(paper)
        meta = item._get_meta_text()
        assert "\u2191" not in meta


class TestHfAppState:
    """Tests for HF app state and helpers."""

    def _make_mock_app(self):
        """Create a minimal ArxivBrowser without running the full TUI."""
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._hf_active = False
        app._hf_cache = {}
        app._hf_loading = False
        app._http_client = None
        app._config = type("Config", (), {"hf_enabled": False, "hf_cache_ttl_hours": 6})()
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        return app

    def test_hf_state_for_inactive(self):
        app = self._make_mock_app()
        assert app._hf_state_for("2401.12345") is None

    def test_hf_state_for_active_with_data(self):
        from arxiv_browser.huggingface import HuggingFacePaper

        app = self._make_mock_app()
        app._hf_active = True
        hf = HuggingFacePaper("2401.12345", "Test", 42, 0, "", (), "", 0)
        app._hf_cache["2401.12345"] = hf
        assert app._hf_state_for("2401.12345") is hf

    def test_hf_state_for_active_no_data(self):
        app = self._make_mock_app()
        app._hf_active = True
        assert app._hf_state_for("unknown") is None

    @pytest.mark.asyncio
    async def test_action_toggle_hf_persists_state(self):
        from unittest.mock import AsyncMock, patch

        app = self._make_mock_app()
        app._fetch_hf_daily = AsyncMock()

        with patch("arxiv_browser.app.save_config", return_value=True):
            await app.action_toggle_hf()

        assert app._hf_active is True
        assert app._config.hf_enabled is True
        app._fetch_hf_daily.assert_called_once()
        assert "populate automatically" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_action_toggle_hf_reverts_when_save_fails(self):
        from unittest.mock import AsyncMock, patch

        app = self._make_mock_app()
        app._fetch_hf_daily = AsyncMock()

        with patch("arxiv_browser.app.save_config", return_value=False):
            await app.action_toggle_hf()

        assert app._hf_active is False
        assert app._config.hf_enabled is False
        app._fetch_hf_daily.assert_not_called()
        assert "Failed to save HuggingFace setting" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_dedupes_concurrent_calls(self, make_paper):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        paper = make_paper(arxiv_id="2401.51515")
        app = ArxivBrowser([paper], restore_session=False)

        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                track_calls = 0

                def fake_track_task(coro):
                    nonlocal track_calls
                    track_calls += 1
                    coro.close()
                    return MagicMock()

                app._fetch_hf_daily_async = AsyncMock(return_value=None)
                app._track_task = fake_track_task
                await asyncio.gather(app._fetch_hf_daily(), app._fetch_hf_daily())
                await pilot.pause(0)

                assert track_calls == 1
                assert app._fetch_hf_daily_async.call_count == 1


class TestBadgeCoalescing:
    """Tests for the P5 badge refresh coalescing system."""

    def _make_badge_app(self):
        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._badges_dirty = set()
        app._badge_timer = None
        app._s2_active = True
        app._s2_cache = {}
        app._hf_active = True
        app._hf_cache = {}
        app._version_updates = {}
        app._relevance_scores = {}
        return app

    def test_mark_badges_dirty_coalesces(self):
        """Multiple dirty calls accumulate badge types with a single timer."""
        from unittest.mock import MagicMock

        app = self._make_badge_app()
        mock_timer = MagicMock()
        app.set_timer = MagicMock(return_value=mock_timer)

        app._mark_badges_dirty("hf")
        assert "hf" in app._badges_dirty
        first_timer_call_count = app.set_timer.call_count

        app._mark_badges_dirty("s2")
        assert "hf" in app._badges_dirty
        assert "s2" in app._badges_dirty
        # Timer was swapped (old stopped, new created)
        assert app.set_timer.call_count == first_timer_call_count + 1

    def test_flush_badge_refresh_clears_dirty(self, make_paper):
        from unittest.mock import MagicMock

        app = self._make_badge_app()
        app._badges_dirty = {"hf", "s2"}
        paper = make_paper(arxiv_id="2401.00001")
        app.filtered_papers = [paper]
        app._update_option_at_index = MagicMock()

        app._flush_badge_refresh()

        assert app._badges_dirty == set()
        app._update_option_at_index.assert_called_once_with(0)

    def test_mark_badges_dirty_immediate_flushes(self, make_paper):
        """immediate=True flushes synchronously without setting a timer."""
        from unittest.mock import MagicMock

        app = self._make_badge_app()
        app.set_timer = MagicMock()
        paper = make_paper(arxiv_id="2401.00001")
        app.filtered_papers = [paper]
        app._update_option_at_index = MagicMock()

        app._mark_badges_dirty("hf", immediate=True)

        # No timer was set
        app.set_timer.assert_not_called()
        # Badge was flushed
        assert app._badges_dirty == set()
        app._update_option_at_index.assert_called_once_with(0)

    def test_flush_badge_refresh_skips_empty_dirty(self):
        """Empty dirty set skips iteration entirely."""
        from unittest.mock import MagicMock

        app = self._make_badge_app()
        app._badges_dirty = set()
        app.filtered_papers = []
        app._update_option_at_index = MagicMock()

        app._flush_badge_refresh()

        app._update_option_at_index.assert_not_called()

    async def test_badge_timer_cleanup_on_unmount(self):
        """Verify _badge_timer is stopped during on_unmount."""
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = self._make_badge_app()
        app._search_timer = None
        app._detail_timer = None
        mock_timer = MagicMock()
        app._badge_timer = mock_timer
        app._save_session_state = MagicMock()
        app._http_client = None

        await app.on_unmount()

        mock_timer.stop.assert_called_once()
        assert app._badge_timer is None

    def test_flush_updates_multiple_papers(self, make_paper):
        """Flush iterates all filtered papers."""
        from unittest.mock import MagicMock

        app = self._make_badge_app()
        app._badges_dirty = {"hf"}
        app._badge_dirty_all = True
        app.filtered_papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00002"),
            make_paper(arxiv_id="2401.00003"),
        ]
        app._update_option_at_index = MagicMock()

        app._flush_badge_refresh()

        assert app._update_option_at_index.call_count == 3
        app._update_option_at_index.assert_any_call(0)
        app._update_option_at_index.assert_any_call(1)
        app._update_option_at_index.assert_any_call(2)


class TestDetailCacheKey:
    """Tests for the _detail_cache_key pure function."""

    def test_deterministic(self, make_paper):
        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        key1 = _detail_cache_key(paper, "abstract text", tags=["ml", "cv"])
        key2 = _detail_cache_key(paper, "abstract text", tags=["ml", "cv"])
        assert key1 == key2

    def test_varies_on_summary(self, make_paper):
        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        key1 = _detail_cache_key(paper, "abstract")
        key2 = _detail_cache_key(paper, "abstract", summary="This paper does X")
        assert key1 != key2

    def test_varies_on_tags(self, make_paper):
        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        key1 = _detail_cache_key(paper, "abstract", tags=["ml"])
        key2 = _detail_cache_key(paper, "abstract", tags=["ml", "cv"])
        assert key1 != key2

    def test_varies_on_highlight(self, make_paper):
        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        key1 = _detail_cache_key(paper, "abstract", highlight_terms=["attention"])
        key2 = _detail_cache_key(paper, "abstract", highlight_terms=["transformer"])
        assert key1 != key2

    def test_varies_on_collapsed(self, make_paper):
        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        key1 = _detail_cache_key(paper, "abstract", collapsed_sections=["authors"])
        key2 = _detail_cache_key(paper, "abstract", collapsed_sections=["abstract"])
        assert key1 != key2

    def test_varies_on_abstract_tail_beyond_prefix(self, make_paper):
        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        prefix = "A" * 80
        key1 = _detail_cache_key(paper, f"{prefix} tail-one")
        key2 = _detail_cache_key(paper, f"{prefix} tail-two")
        assert key1 != key2

    def test_varies_on_s2_citation_count(self, make_paper):
        from unittest.mock import MagicMock

        from arxiv_browser.app import _detail_cache_key

        paper = make_paper(arxiv_id="2401.00001")
        s2_a = MagicMock()
        s2_a.citation_count = 10
        s2_b = MagicMock()
        s2_b.citation_count = 20
        key1 = _detail_cache_key(paper, "abstract", s2_data=s2_a)
        key2 = _detail_cache_key(paper, "abstract", s2_data=s2_b)
        assert key1 != key2

    def test_varies_on_s2_non_count_fields(self, make_paper):
        from arxiv_browser.app import _detail_cache_key
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        paper = make_paper(arxiv_id="2401.00001")
        s2_a = SemanticScholarPaper(
            arxiv_id=paper.arxiv_id,
            s2_paper_id="s2-id",
            citation_count=10,
            influential_citation_count=2,
            tldr="first",
            fields_of_study=("CS",),
            year=2024,
            url="https://example.com",
            title="title",
            abstract="abs",
        )
        s2_b = SemanticScholarPaper(
            arxiv_id=paper.arxiv_id,
            s2_paper_id="s2-id",
            citation_count=10,
            influential_citation_count=2,
            tldr="second",
            fields_of_study=("CS",),
            year=2024,
            url="https://example.com",
            title="title",
            abstract="abs",
        )
        key1 = _detail_cache_key(paper, "abstract", s2_data=s2_a)
        key2 = _detail_cache_key(paper, "abstract", s2_data=s2_b)
        assert key1 != key2

    def test_varies_on_hf_non_upvote_fields(self, make_paper):
        from arxiv_browser.app import _detail_cache_key
        from arxiv_browser.huggingface import HuggingFacePaper

        paper = make_paper(arxiv_id="2401.00001")
        hf_a = HuggingFacePaper(
            arxiv_id=paper.arxiv_id,
            title="HF",
            upvotes=7,
            num_comments=1,
            ai_summary="one",
            ai_keywords=("k",),
            github_repo="owner/repo",
            github_stars=3,
        )
        hf_b = HuggingFacePaper(
            arxiv_id=paper.arxiv_id,
            title="HF",
            upvotes=7,
            num_comments=1,
            ai_summary="two",
            ai_keywords=("k",),
            github_repo="owner/repo",
            github_stars=3,
        )
        key1 = _detail_cache_key(paper, "abstract", hf_data=hf_a)
        key2 = _detail_cache_key(paper, "abstract", hf_data=hf_b)
        assert key1 != key2


class TestDetailPaneCache:
    """Tests for the PaperDetails caching behavior."""

    def test_cache_hit_skips_rebuild(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001", title="Test Paper")

        details.update_paper(paper, "abstract text")
        content1 = details.content

        # Second call with same args should hit cache — content identical
        details.update_paper(paper, "abstract text")
        content2 = details.content

        assert content1 == content2
        assert len(details._detail_cache) == 1

    def test_cache_miss_on_new_data(self, make_paper):
        from unittest.mock import MagicMock

        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001", title="Test Paper")

        details.update_paper(paper, "abstract text")
        assert len(details._detail_cache) == 1

        s2 = MagicMock()
        s2.citation_count = 42
        s2.influential_citation_count = 5
        s2.fields_of_study = ("CS",)
        s2.tldr = "A paper"
        details.update_paper(paper, "abstract text", s2_data=s2)
        assert len(details._detail_cache) == 2

    def test_cache_eviction(self, make_paper):
        from arxiv_browser.app import DETAIL_CACHE_MAX, PaperDetails

        details = PaperDetails()
        for i in range(DETAIL_CACHE_MAX + 5):
            paper = make_paper(arxiv_id=f"2401.{i:05d}", title=f"Paper {i}")
            details.update_paper(paper, f"abstract {i}")

        assert len(details._detail_cache) == DETAIL_CACHE_MAX

    def test_clear_cache(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001", title="Test")
        details.update_paper(paper, "abstract")
        assert len(details._detail_cache) == 1

        details.clear_cache()
        assert len(details._detail_cache) == 0
        assert len(details._detail_cache_order) == 0


class TestParseArxivVersionMap:
    """Tests for arXiv Atom feed version extraction."""

    ATOM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
{entries}
</feed>"""

    ENTRY_TEMPLATE = """<entry>
  <id>http://arxiv.org/abs/{id_with_version}</id>
  <title>Test Paper</title>
</entry>"""

    def _make_feed(self, entries: list[str]) -> str:
        entry_xml = "\n".join(self.ENTRY_TEMPLATE.format(id_with_version=e) for e in entries)
        return self.ATOM_TEMPLATE.format(entries=entry_xml)

    def test_single_entry_with_version(self):
        xml = self._make_feed(["2401.12345v3"])
        result = parse_arxiv_version_map(xml)
        assert result == {"2401.12345": 3}

    def test_multiple_entries(self):
        xml = self._make_feed(["2401.12345v2", "2401.67890v5", "hep-th/9901001v1"])
        result = parse_arxiv_version_map(xml)
        assert result == {
            "2401.12345": 2,
            "2401.67890": 5,
            "hep-th/9901001": 1,
        }

    def test_missing_version_suffix_defaults_to_1(self):
        xml = self._make_feed(["2401.12345"])
        result = parse_arxiv_version_map(xml)
        assert result == {"2401.12345": 1}

    def test_empty_feed(self):
        xml = self.ATOM_TEMPLATE.format(entries="")
        result = parse_arxiv_version_map(xml)
        assert result == {}

    def test_empty_string(self):
        assert parse_arxiv_version_map("") == {}
        assert parse_arxiv_version_map("   ") == {}

    def test_invalid_xml(self):
        result = parse_arxiv_version_map("<not valid xml")
        assert result == {}

    def test_unsafe_entities_are_rejected(self):
        xml = """<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2401.12345v1</id></entry>
</feed>"""
        result = parse_arxiv_version_map(xml)
        assert result == {}

    def test_duplicate_ids_last_wins(self):
        """If the same ID appears twice, the last entry wins."""
        entries = """
<entry><id>http://arxiv.org/abs/2401.12345v2</id><title>A</title></entry>
<entry><id>http://arxiv.org/abs/2401.12345v3</id><title>B</title></entry>
"""
        xml = self.ATOM_TEMPLATE.format(entries=entries)
        result = parse_arxiv_version_map(xml)
        assert result == {"2401.12345": 3}

    def test_high_version_number(self):
        xml = self._make_feed(["2401.12345v42"])
        result = parse_arxiv_version_map(xml)
        assert result == {"2401.12345": 42}


class TestVersionMetadataSerialization:
    """Tests for last_checked_version config round-trip."""

    def test_round_trip_with_version(self, tmp_path, monkeypatch):
        """last_checked_version survives save/load cycle."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config = UserConfig()
        config.paper_metadata["2401.12345"] = PaperMetadata(
            arxiv_id="2401.12345",
            starred=True,
            last_checked_version=3,
        )
        assert save_config(config) is True
        loaded = load_config()
        meta = loaded.paper_metadata["2401.12345"]
        assert meta.last_checked_version == 3
        assert meta.starred is True

    def test_defaults_when_absent(self, tmp_path, monkeypatch):
        """Old configs without last_checked_version default to None."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config = UserConfig()
        config.paper_metadata["2401.12345"] = PaperMetadata(
            arxiv_id="2401.12345",
        )
        assert save_config(config) is True
        loaded = load_config()
        meta = loaded.paper_metadata["2401.12345"]
        assert meta.last_checked_version is None

    def test_type_validation_rejects_string(self, tmp_path, monkeypatch):
        """Non-int values for last_checked_version are treated as None."""
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config_data = {
            "version": 1,
            "paper_metadata": {
                "2401.12345": {
                    "notes": "",
                    "tags": [],
                    "is_read": False,
                    "starred": True,
                    "last_checked_version": "not_an_int",
                }
            },
        }
        config_file.write_text(json.dumps(config_data))
        loaded = load_config()
        assert loaded.paper_metadata["2401.12345"].last_checked_version is None


class TestVersionDetailPane:
    """Tests for version update rendering in PaperDetails."""

    def _make_paper(self):
        return Paper(
            arxiv_id="2401.12345",
            date="2024-01-01",
            title="Test Paper",
            authors="Author",
            categories="cs.AI",
            comments=None,
            abstract="Abstract",
            url="https://arxiv.org/abs/2401.12345",
            abstract_raw="Abstract",
        )

    def test_version_section_shown_with_update(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper, version_update=(1, 3))
        content = details.content
        assert "v1" in content
        assert "v3" in content
        assert "arxivdiff.org" in content
        assert "2401.12345" in content

    def test_version_section_hidden_without_update(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper, version_update=None)
        content = details.content
        assert "Version Update" not in content
        assert "arxivdiff" not in content

    def test_version_section_absent_by_default(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper)
        content = details.content
        assert "Version Update" not in content


class TestVersionListBadge:
    """Tests for version update badge in PaperListItem."""

    def _make_paper(self):
        return Paper(
            arxiv_id="2401.12345",
            date="2024-01-01",
            title="Test Paper",
            authors="Author",
            categories="cs.AI",
            comments=None,
            abstract="Abstract",
            url="https://arxiv.org/abs/2401.12345",
            abstract_raw="Abstract",
        )

    def test_badge_appears_with_update(self):
        from arxiv_browser.app import PaperListItem

        item = PaperListItem(self._make_paper())
        item._version_update = (1, 3)
        meta_text = item._get_meta_text()
        assert "v1" in meta_text
        assert "v3" in meta_text

    def test_badge_absent_without_update(self):
        from arxiv_browser.app import PaperListItem

        item = PaperListItem(self._make_paper())
        meta_text = item._get_meta_text()
        # Should not contain version arrow
        assert "\u2192" not in meta_text


class TestVersionAppState:
    """Tests for version tracking app state helpers."""

    def _make_mock_app(self):
        """Create a minimal ArxivBrowser without running the full TUI."""
        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._version_updates = {}
        app._version_checking = False
        app._config = UserConfig()
        return app

    def test_version_update_for_returns_tuple(self):
        app = self._make_mock_app()
        app._version_updates["2401.12345"] = (1, 3)
        assert app._version_update_for("2401.12345") == (1, 3)

    def test_version_update_for_returns_none(self):
        app = self._make_mock_app()
        assert app._version_update_for("2401.12345") is None

    def test_version_update_for_unknown_id(self):
        app = self._make_mock_app()
        app._version_updates["2401.99999"] = (2, 5)
        assert app._version_update_for("2401.12345") is None
