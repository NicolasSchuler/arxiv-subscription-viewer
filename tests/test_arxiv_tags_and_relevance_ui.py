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


class TestParseTagNamespace:
    """Tests for parse_tag_namespace function."""

    def test_simple_namespace(self):
        assert parse_tag_namespace("topic:transformers") == ("topic", "transformers")

    def test_no_namespace(self):
        assert parse_tag_namespace("important") == ("", "important")

    def test_multiple_colons(self):
        ns, val = parse_tag_namespace("topic:sub:detail")
        assert ns == "topic"
        assert val == "sub:detail"

    def test_empty_value_after_colon(self):
        assert parse_tag_namespace("topic:") == ("topic", "")

    def test_empty_string(self):
        assert parse_tag_namespace("") == ("", "")

    def test_colon_at_start(self):
        assert parse_tag_namespace(":value") == ("", "value")

    def test_status_namespace(self):
        assert parse_tag_namespace("status:to-read") == ("status", "to-read")

    def test_project_namespace(self):
        assert parse_tag_namespace("project:my-project") == ("project", "my-project")


class TestGetTagColor:
    """Tests for get_tag_color function."""

    def test_known_namespace_topic(self):
        assert get_tag_color("topic:ml") == TAG_NAMESPACE_COLORS["topic"]

    def test_known_namespace_status(self):
        assert get_tag_color("status:to-read") == TAG_NAMESPACE_COLORS["status"]

    def test_known_namespace_project(self):
        assert get_tag_color("project:foo") == TAG_NAMESPACE_COLORS["project"]

    def test_known_namespace_method(self):
        assert get_tag_color("method:cnn") == TAG_NAMESPACE_COLORS["method"]

    def test_known_namespace_priority(self):
        assert get_tag_color("priority:high") == TAG_NAMESPACE_COLORS["priority"]

    def test_unnamespaced_tag_gets_default_purple(self):
        assert get_tag_color("important") == "#ae81ff"

    def test_unknown_namespace_gets_deterministic_color(self):
        color1 = get_tag_color("custom:foo")
        color2 = get_tag_color("custom:bar")
        # Same namespace → same color
        assert color1 == color2

    def test_different_unknown_namespaces_may_differ(self):
        # Different namespaces get deterministic but potentially different colors
        color1 = get_tag_color("ns1:foo")
        color2 = get_tag_color("ns2:foo")
        # Both should be valid hex colors
        assert color1.startswith("#")
        assert color2.startswith("#")

    def test_unknown_namespace_uses_stable_hash_algorithm(self):
        import hashlib

        from arxiv_browser.themes import _TAG_FALLBACK_COLORS

        ns = "custom"
        digest = hashlib.sha256(ns.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % len(_TAG_FALLBACK_COLORS)
        expected = _TAG_FALLBACK_COLORS[idx]
        assert get_tag_color(f"{ns}:foo") == expected

    def test_empty_string_gets_default(self):
        assert get_tag_color("") == "#ae81ff"


class TestTagNamespaceDisplay:
    """Tests for namespace-colored tag display."""

    def test_tags_section_in_detail_pane(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper,
            abstract_text="test abstract",
            tags=["topic:ml", "status:to-read", "important"],
        )
        content = details.content
        assert "Tags" in content
        assert "topic" in content
        assert "status" in content
        assert "important" in content

    def test_no_tags_section_when_empty(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(paper, abstract_text="test abstract", tags=None)
        content = details.content
        # No Tags section header — bold heading should be absent
        assert "Tags[/]" not in content

    def test_tags_section_groups_by_namespace(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper,
            abstract_text="test abstract",
            tags=["topic:ml", "topic:nlp", "status:done"],
        )
        content = details.content
        # Should contain grouped namespace labels
        assert "topic:" in content
        assert "status:" in content


class TestTagSuggestionGrouping:
    """Tests for the TagsModal suggestion label."""

    def test_build_suggestions_empty(self):
        from arxiv_browser.modals import TagsModal

        modal = TagsModal("2401.00001", all_tags=[])
        assert modal._build_suggestions_markup() == ""

    def test_build_suggestions_groups_by_namespace(self):
        from arxiv_browser.modals import TagsModal

        modal = TagsModal(
            "2401.00001",
            all_tags=["topic:ml", "topic:nlp", "status:to-read", "important"],
        )
        markup = modal._build_suggestions_markup()
        assert "status:" in markup
        assert "topic:" in markup
        assert "important" in markup

    def test_build_suggestions_deduplicates(self):
        from arxiv_browser.modals import TagsModal

        modal = TagsModal(
            "2401.00001",
            all_tags=["topic:ml", "topic:ml", "topic:ml"],
        )
        markup = modal._build_suggestions_markup()
        # "ml" should appear only once
        assert markup.count("ml") == 1

    def test_modal_accepts_all_tags_param(self):
        from arxiv_browser.modals import TagsModal

        modal = TagsModal(
            "2401.00001",
            current_tags=["existing"],
            all_tags=["topic:ml", "status:done"],
        )
        assert modal._all_tags == ["topic:ml", "status:done"]


class TestTagNamespaceConstants:
    """Tests for TAG_NAMESPACE_COLORS constant."""

    def test_has_expected_namespaces(self):
        expected = {"topic", "status", "project", "method", "priority"}
        assert set(TAG_NAMESPACE_COLORS.keys()) == expected

    def test_all_colors_are_valid_hex(self):
        for ns, color in TAG_NAMESPACE_COLORS.items():
            assert color.startswith("#"), f"{ns} color {color} is not hex"
            assert len(color) == 7, f"{ns} color {color} is not 7 chars"


class TestRelevancePrompt:
    """Tests for RELEVANCE_PROMPT_TEMPLATE and build_relevance_prompt()."""

    def test_template_has_required_placeholders(self):
        from arxiv_browser.app import RELEVANCE_PROMPT_TEMPLATE

        for field in ("title", "authors", "categories", "abstract", "interests"):
            assert f"{{{field}}}" in RELEVANCE_PROMPT_TEMPLATE

    def test_build_relevance_prompt_substitution(self, make_paper):
        from arxiv_browser.app import build_relevance_prompt

        paper = make_paper(
            title="Efficient LLM Inference",
            authors="Alice, Bob",
            categories="cs.AI cs.CL",
            abstract="We propose a new method.",
        )
        result = build_relevance_prompt(paper, "quantization and distillation")
        assert "Efficient LLM Inference" in result
        assert "Alice, Bob" in result
        assert "cs.AI cs.CL" in result
        assert "We propose a new method." in result
        assert "quantization and distillation" in result

    def test_build_relevance_prompt_missing_abstract(self):
        from arxiv_browser.app import build_relevance_prompt

        paper = Paper(
            arxiv_id="2401.12345",
            date="2024-01-01",
            title="Test",
            authors="Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2401.12345",
            abstract_raw=None,
        )
        result = build_relevance_prompt(paper, "test interests")
        assert "(no abstract)" in result

    def test_template_requests_json_output(self):
        from arxiv_browser.app import RELEVANCE_PROMPT_TEMPLATE

        assert "JSON" in RELEVANCE_PROMPT_TEMPLATE
        assert '"score"' in RELEVANCE_PROMPT_TEMPLATE
        assert '"reason"' in RELEVANCE_PROMPT_TEMPLATE


class TestParseRelevanceResponse:
    """Tests for _parse_relevance_response()."""

    def test_valid_json(self):
        from arxiv_browser.app import _parse_relevance_response

        result = _parse_relevance_response('{"score": 8, "reason": "Highly relevant"}')
        assert result == (8, "Highly relevant")

    def test_markdown_wrapped_json(self):
        from arxiv_browser.app import _parse_relevance_response

        text = '```json\n{"score": 7, "reason": "Good match"}\n```'
        result = _parse_relevance_response(text)
        assert result == (7, "Good match")

    def test_markdown_fence_without_json_label(self):
        from arxiv_browser.app import _parse_relevance_response

        text = '```\n{"score": 5, "reason": "Moderate"}\n```'
        result = _parse_relevance_response(text)
        assert result == (5, "Moderate")

    def test_regex_fallback(self):
        from arxiv_browser.app import _parse_relevance_response

        text = 'Here is the result: "score": 9, "reason": "Very relevant paper"'
        result = _parse_relevance_response(text)
        assert result is not None
        assert result[0] == 9
        assert result[1] == "Very relevant paper"

    def test_regex_fallback_score_only(self):
        from arxiv_browser.app import _parse_relevance_response

        text = 'The "score": 6 for this paper'
        result = _parse_relevance_response(text)
        assert result is not None
        assert result[0] == 6
        assert result[1] == ""

    def test_invalid_input_returns_none(self):
        from arxiv_browser.app import _parse_relevance_response

        assert _parse_relevance_response("not valid at all") is None
        assert _parse_relevance_response("") is None
        assert _parse_relevance_response("just some text") is None

    def test_score_clamped_high(self):
        from arxiv_browser.app import _parse_relevance_response

        result = _parse_relevance_response('{"score": 15, "reason": "Off scale"}')
        assert result == (10, "Off scale")

    def test_score_clamped_low(self):
        from arxiv_browser.app import _parse_relevance_response

        result = _parse_relevance_response('{"score": 0, "reason": "Below range"}')
        assert result == (1, "Below range")

    def test_score_clamped_negative(self):
        from arxiv_browser.app import _parse_relevance_response

        result = _parse_relevance_response('{"score": -3, "reason": "Negative"}')
        assert result == (1, "Negative")

    def test_json_with_extra_whitespace(self):
        from arxiv_browser.app import _parse_relevance_response

        text = '  \n  {"score": 4, "reason": "Some reason"}  \n  '
        result = _parse_relevance_response(text)
        assert result == (4, "Some reason")

    def test_missing_reason_in_json(self):
        from arxiv_browser.app import _parse_relevance_response

        result = _parse_relevance_response('{"score": 5}')
        assert result == (5, "")


class TestRelevanceDb:
    """Tests for relevance score SQLite persistence."""

    def test_init_creates_table(self, tmp_path):
        import sqlite3

        from arxiv_browser.app import _init_relevance_db

        db_path = tmp_path / "relevance.db"
        _init_relevance_db(db_path)
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='relevance_scores'"
            ).fetchone()
            assert row is not None
            assert "PRIMARY KEY (arxiv_id, interests_hash)" in row[0]

    def test_save_and_load_score(self, tmp_path):
        from arxiv_browser.app import (
            _init_relevance_db,
            _load_relevance_score,
            _save_relevance_score,
        )

        db_path = tmp_path / "relevance.db"
        _init_relevance_db(db_path)
        _save_relevance_score(db_path, "2401.12345", "hash123", 8, "Good match")
        result = _load_relevance_score(db_path, "2401.12345", "hash123")
        assert result == (8, "Good match")

    def test_load_missing_returns_none(self, tmp_path):
        from arxiv_browser.app import _init_relevance_db, _load_relevance_score

        db_path = tmp_path / "relevance.db"
        _init_relevance_db(db_path)
        result = _load_relevance_score(db_path, "nonexistent", "hash123")
        assert result is None

    def test_load_from_nonexistent_db(self, tmp_path):
        from arxiv_browser.app import _load_relevance_score

        db_path = tmp_path / "does_not_exist.db"
        result = _load_relevance_score(db_path, "2401.12345", "hash123")
        assert result is None

    def test_bulk_load(self, tmp_path):
        from arxiv_browser.app import (
            _init_relevance_db,
            _load_all_relevance_scores,
            _save_relevance_score,
        )

        db_path = tmp_path / "relevance.db"
        _init_relevance_db(db_path)
        _save_relevance_score(db_path, "2401.00001", "hash_a", 9, "Great")
        _save_relevance_score(db_path, "2401.00002", "hash_a", 3, "Not relevant")
        _save_relevance_score(db_path, "2401.00003", "hash_b", 7, "Different hash")

        result = _load_all_relevance_scores(db_path, "hash_a")
        assert len(result) == 2
        assert result["2401.00001"] == (9, "Great")
        assert result["2401.00002"] == (3, "Not relevant")
        assert "2401.00003" not in result

    def test_composite_pk_different_interests(self, tmp_path):
        from arxiv_browser.app import (
            _init_relevance_db,
            _load_relevance_score,
            _save_relevance_score,
        )

        db_path = tmp_path / "relevance.db"
        _init_relevance_db(db_path)
        _save_relevance_score(db_path, "2401.12345", "hash_x", 9, "Very relevant")
        _save_relevance_score(db_path, "2401.12345", "hash_y", 2, "Not relevant")
        assert _load_relevance_score(db_path, "2401.12345", "hash_x") == (9, "Very relevant")
        assert _load_relevance_score(db_path, "2401.12345", "hash_y") == (2, "Not relevant")

    def test_save_replaces_existing(self, tmp_path):
        from arxiv_browser.app import (
            _init_relevance_db,
            _load_relevance_score,
            _save_relevance_score,
        )

        db_path = tmp_path / "relevance.db"
        _init_relevance_db(db_path)
        _save_relevance_score(db_path, "2401.12345", "hash_a", 5, "Old reason")
        _save_relevance_score(db_path, "2401.12345", "hash_a", 8, "New reason")
        result = _load_relevance_score(db_path, "2401.12345", "hash_a")
        assert result == (8, "New reason")

    def test_bulk_load_nonexistent_db(self, tmp_path):
        from arxiv_browser.app import _load_all_relevance_scores

        db_path = tmp_path / "does_not_exist.db"
        result = _load_all_relevance_scores(db_path, "any_hash")
        assert result == {}


class TestRelevanceConfigSerialization:
    """Tests for research_interests config round-trip."""

    def test_round_trip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config = UserConfig(research_interests="efficient LLM inference, quantization")
        assert save_config(config) is True
        loaded = load_config()
        assert loaded.research_interests == "efficient LLM inference, quantization"

    def test_defaults_when_absent(self, tmp_path, monkeypatch):
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config_file.write_text(json.dumps({"version": 1}))
        loaded = load_config()
        assert loaded.research_interests == ""

    def test_type_validation_rejects_non_string(self, tmp_path, monkeypatch):
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config_file.write_text(json.dumps({"version": 1, "research_interests": 42}))
        loaded = load_config()
        assert loaded.research_interests == ""


class TestRelevanceSortPapers:
    """Tests for 'relevance' sort key in sort_papers()."""

    def test_sort_by_relevance(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(arxiv_id="low", title="Low"),
            make_paper(arxiv_id="high", title="High"),
            make_paper(arxiv_id="mid", title="Mid"),
        ]
        cache = {
            "low": (2, "Low relevance"),
            "high": (9, "High relevance"),
            "mid": (5, "Mid relevance"),
        }
        result = sort_papers(papers, "relevance", relevance_cache=cache)
        assert [p.arxiv_id for p in result] == ["high", "mid", "low"]

    def test_sort_relevance_unscored_last(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(arxiv_id="unscored", title="Unscored"),
            make_paper(arxiv_id="scored", title="Scored"),
        ]
        cache = {"scored": (7, "Scored")}
        result = sort_papers(papers, "relevance", relevance_cache=cache)
        assert result[0].arxiv_id == "scored"
        assert result[1].arxiv_id == "unscored"

    def test_sort_relevance_empty_cache(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [make_paper(arxiv_id="a"), make_paper(arxiv_id="b")]
        result = sort_papers(papers, "relevance", relevance_cache={})
        assert len(result) == 2

    def test_sort_relevance_none_cache(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [make_paper(arxiv_id="a"), make_paper(arxiv_id="b")]
        result = sort_papers(papers, "relevance", relevance_cache=None)
        assert len(result) == 2


class TestRelevanceDetailPane:
    """Tests for relevance section in PaperDetails."""

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

    def test_relevance_section_shown_with_score(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper, relevance=(9, "Highly relevant paper"))
        content = details.content
        assert "Relevance" in content
        assert "9/10" in content
        assert "Highly relevant paper" in content

    def test_relevance_section_hidden_without_score(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper, relevance=None)
        content = details.content
        assert "Relevance" not in content

    def test_relevance_section_absent_by_default(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper)
        content = details.content
        assert "Relevance" not in content

    def test_relevance_low_score_shown(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper, relevance=(2, "Not very relevant"))
        content = details.content
        assert "2/10" in content
        assert "Not very relevant" in content

    def test_relevance_empty_reason(self):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = self._make_paper()
        details.update_paper(paper, relevance=(6, ""))
        content = details.content
        assert "6/10" in content


class TestRelevanceListBadge:
    """Tests for relevance score badge in PaperListItem."""

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

    def test_badge_appears_with_high_score(self):
        from arxiv_browser.app import THEME_COLORS, PaperListItem

        item = PaperListItem(self._make_paper())
        item._relevance_score = (9, "Great match")
        meta_text = item._get_meta_text()
        assert "9/10" in meta_text
        assert THEME_COLORS["green"] in meta_text

    def test_badge_appears_with_mid_score(self):
        from arxiv_browser.app import THEME_COLORS, PaperListItem

        item = PaperListItem(self._make_paper())
        item._relevance_score = (6, "Moderate")
        meta_text = item._get_meta_text()
        assert "6/10" in meta_text
        assert THEME_COLORS["yellow"] in meta_text

    def test_badge_appears_with_low_score(self):
        from arxiv_browser.app import THEME_COLORS, PaperListItem

        item = PaperListItem(self._make_paper())
        item._relevance_score = (2, "Low relevance")
        meta_text = item._get_meta_text()
        assert "2/10" in meta_text
        assert THEME_COLORS["muted"] in meta_text

    def test_badge_absent_without_score(self):
        from arxiv_browser.app import PaperListItem

        item = PaperListItem(self._make_paper())
        meta_text = item._get_meta_text()
        assert "/10" not in meta_text

    def test_update_relevance_data_sets_score(self):
        from arxiv_browser.app import PaperListItem

        item = PaperListItem(self._make_paper())
        assert item._relevance_score is None
        item.update_relevance_data((8, "Relevant"))
        assert item._relevance_score == (8, "Relevant")


class TestResearchInterestsModal:
    """Tests for ResearchInterestsModal structure."""

    def test_modal_exists(self):
        from arxiv_browser.modals import ResearchInterestsModal

        modal = ResearchInterestsModal("test interests")
        assert modal._current_interests == "test interests"

    def test_modal_empty_default(self):
        from arxiv_browser.modals import ResearchInterestsModal

        modal = ResearchInterestsModal()
        assert modal._current_interests == ""

    def test_modal_has_bindings(self):
        from arxiv_browser.modals import ResearchInterestsModal

        binding_keys = {b.key for b in ResearchInterestsModal.BINDINGS}
        assert "ctrl+s" in binding_keys
        assert "escape" in binding_keys


class TestRelevanceDbPath:
    """Tests for get_relevance_db_path()."""

    def test_returns_path_object(self):
        from arxiv_browser.app import get_relevance_db_path

        result = get_relevance_db_path()
        assert isinstance(result, Path)
        assert result.name == "relevance.db"

    def test_in_config_dir(self):
        from arxiv_browser.app import get_relevance_db_path

        rel_path = get_relevance_db_path()
        sum_path = get_summary_db_path()
        assert rel_path.parent == sum_path.parent


class TestRelevanceSortOptions:
    """Tests for relevance in SORT_OPTIONS."""

    def test_relevance_in_sort_options(self):
        assert "relevance" in SORT_OPTIONS

    def test_sort_options_count(self):
        assert len(SORT_OPTIONS) == 6
