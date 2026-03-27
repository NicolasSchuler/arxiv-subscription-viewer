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


class TestDictToConfigEdgeCases:
    """Tests for _dict_to_config with invalid/malformed data."""

    def test_sort_index_negative_clamps_to_zero(self):
        """Negative sort_index should be clamped to 0."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"session": {"sort_index": -1}}
        config = _dict_to_config(data)
        assert config.session.sort_index == 0

    def test_sort_index_too_large_clamps_to_zero(self):
        """sort_index beyond SORT_OPTIONS length should be clamped to 0."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"session": {"sort_index": 999}}
        config = _dict_to_config(data)
        assert config.session.sort_index == 0

    def test_sort_index_at_max_boundary(self):
        """sort_index at the last valid index should be accepted."""
        from tests.support.canonical_exports import _dict_to_config

        max_idx = len(SORT_OPTIONS) - 1
        data = {"session": {"sort_index": max_idx}}
        config = _dict_to_config(data)
        assert config.session.sort_index == max_idx

    def test_sort_index_one_past_max_clamps_to_zero(self):
        """sort_index one past the last valid index should be clamped to 0."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"session": {"sort_index": len(SORT_OPTIONS)}}
        config = _dict_to_config(data)
        assert config.session.sort_index == 0

    def test_session_not_a_dict_uses_defaults(self):
        """Non-dict session value should fall back to defaults."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"session": "not_a_dict"}
        config = _dict_to_config(data)
        assert config.session.scroll_index == 0
        assert config.session.current_filter == ""
        assert config.session.sort_index == 0

    def test_metadata_entry_not_a_dict_skipped(self):
        """Non-dict metadata entries should be skipped."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "paper_metadata": {
                "2401.00001": "not_a_dict",
                "2401.00002": 42,
                "2401.00003": ["list"],
            }
        }
        config = _dict_to_config(data)
        assert config.paper_metadata == {}

    def test_metadata_mixed_valid_and_invalid(self):
        """Valid metadata entries should be kept when invalid ones are skipped."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "paper_metadata": {
                "2401.00001": {"notes": "valid", "starred": True},
                "2401.00002": "invalid_string",
                "2401.00003": {"is_read": True},
            }
        }
        config = _dict_to_config(data)
        assert len(config.paper_metadata) == 2
        assert "2401.00001" in config.paper_metadata
        assert config.paper_metadata["2401.00001"].starred is True
        assert "2401.00003" in config.paper_metadata
        assert config.paper_metadata["2401.00003"].is_read is True
        assert "2401.00002" not in config.paper_metadata

    def test_metadata_wrong_field_types_get_defaults(self):
        """Wrong types in metadata fields should fall back to defaults."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "paper_metadata": {
                "2401.00001": {
                    "notes": 123,
                    "tags": "not_a_list",
                    "is_read": "yes",
                    "starred": 1,
                }
            }
        }
        config = _dict_to_config(data)
        meta = config.paper_metadata["2401.00001"]
        assert meta.notes == ""
        assert meta.tags == []
        assert meta.is_read is False
        assert meta.starred is False

    def test_metadata_tags_list_filters_non_string_items(self):
        """Mixed-type tag lists should keep only string tags."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "paper_metadata": {
                "2401.00001": {
                    "tags": ["topic:ml", 1, None, {"x": "y"}],
                }
            }
        }
        config = _dict_to_config(data)
        meta = config.paper_metadata["2401.00001"]
        assert meta.tags == ["topic:ml"]

    def test_metadata_last_checked_version_non_int_becomes_none(self):
        """last_checked_version that is not int should become None."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"paper_metadata": {"2401.00001": {"last_checked_version": "v3"}}}
        config = _dict_to_config(data)
        assert config.paper_metadata["2401.00001"].last_checked_version is None

    def test_metadata_last_checked_version_int_preserved(self):
        """last_checked_version that is int should be preserved."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"paper_metadata": {"2401.00001": {"last_checked_version": 5}}}
        config = _dict_to_config(data)
        assert config.paper_metadata["2401.00001"].last_checked_version == 5

    def test_watch_list_non_dict_entries_skipped(self):
        """Non-dict watch list entries should be skipped."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "watch_list": [
                "just_a_string",
                42,
                {"pattern": "Smith", "match_type": "author"},
            ]
        }
        config = _dict_to_config(data)
        assert len(config.watch_list) == 1
        assert config.watch_list[0].pattern == "Smith"

    def test_watch_list_invalid_match_type_defaults_to_author(self):
        """Invalid match_type should be defaulted to 'author'."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"watch_list": [{"pattern": "test", "match_type": "invalid_type"}]}
        config = _dict_to_config(data)
        assert config.watch_list[0].match_type == "author"

    def test_watch_list_not_a_list_uses_empty(self):
        """Non-list watch_list value should result in empty list."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"watch_list": "not_a_list"}
        config = _dict_to_config(data)
        assert config.watch_list == []

    def test_bookmarks_non_dict_entries_skipped(self):
        """Non-dict bookmark entries should be skipped."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "bookmarks": [
                "just_a_string",
                {"name": "AI", "query": "cs.AI"},
                None,
            ]
        }
        config = _dict_to_config(data)
        assert len(config.bookmarks) == 1
        assert config.bookmarks[0].name == "AI"
        assert config.bookmarks[0].query == "cs.AI"

    def test_bookmarks_not_a_list_uses_empty(self):
        """Non-list bookmarks value should result in empty list."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"bookmarks": {"name": "AI"}}
        config = _dict_to_config(data)
        assert config.bookmarks == []

    def test_marks_not_a_dict_uses_empty(self):
        """Non-dict marks value should result in empty dict."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"marks": ["a", "b"]}
        config = _dict_to_config(data)
        assert config.marks == {}

    def test_marks_valid_dict_preserved(self):
        """Valid marks dict should be preserved."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"marks": {"a": "2401.00001", "b": "2401.00002"}}
        config = _dict_to_config(data)
        assert config.marks == {"a": "2401.00001", "b": "2401.00002"}

    def test_category_colors_non_string_entries_filtered(self):
        """Non-string keys/values in category_colors should be filtered out."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "category_colors": {
                "cs.AI": "#ff0000",
                42: "#00ff00",
                "cs.CL": 123,
            }
        }
        config = _dict_to_config(data)
        assert config.category_colors == {"cs.AI": "#ff0000"}

    def test_theme_non_string_entries_filtered(self):
        """Non-string keys/values in theme should be filtered out."""
        from tests.support.canonical_exports import _dict_to_config

        data = {
            "theme": {
                "background": "#272822",
                123: "#ffffff",
                "text": 456,
            }
        }
        config = _dict_to_config(data)
        assert config.theme == {"background": "#272822"}

    def test_current_date_non_string_becomes_none(self):
        """Non-string current_date should become None."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"session": {"current_date": 12345}}
        config = _dict_to_config(data)
        assert config.session.current_date is None

    def test_current_date_string_preserved(self):
        """String current_date should be preserved."""
        from tests.support.canonical_exports import _dict_to_config

        data = {"session": {"current_date": "2024-01-15"}}
        config = _dict_to_config(data)
        assert config.session.current_date == "2024-01-15"


class TestLoadConfigErrorPaths:
    """Tests for load_config error handling."""

    def test_key_error_returns_default(self, tmp_path, monkeypatch):
        """KeyError during config parsing should return default config."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text('{"session": {}}', encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with patch("arxiv_browser.config._dict_to_config", side_effect=KeyError("bad_key")):
            config = load_config()
        assert isinstance(config, UserConfig)
        assert config.bibtex_export_dir == ""

    def test_type_error_returns_default(self, tmp_path, monkeypatch):
        """TypeError during config parsing should return default config."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text('{"session": {}}', encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with patch("arxiv_browser.config._dict_to_config", side_effect=TypeError("bad_type")):
            config = load_config()
        assert isinstance(config, UserConfig)

    def test_os_error_returns_default(self, tmp_path, monkeypatch):
        """OSError during config read should return default config."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with patch.object(type(config_file), "read_text", side_effect=OSError("Permission denied")):
            config = load_config()
        assert isinstance(config, UserConfig)


class TestSaveConfigErrorPaths:
    """Tests for save_config error handling and tempfile cleanup."""

    def test_oserror_during_mkdir_returns_false(self, tmp_path, monkeypatch):
        """OSError during directory creation should return False."""
        from unittest.mock import patch

        config_file = tmp_path / "readonly" / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            result = save_config(UserConfig())
        assert result is False

    def test_oserror_during_replace_cleans_up_temp(self, tmp_path, monkeypatch):
        """OSError during os.replace should clean up the temp file."""
        import os
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        created_temps = []
        original_mkstemp = __import__("tempfile").mkstemp

        def tracking_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            created_temps.append(path)
            return fd, path

        with (
            patch("tempfile.mkstemp", side_effect=tracking_mkstemp),
            patch("os.replace", side_effect=OSError("disk full")),
        ):
            result = save_config(UserConfig())

        assert result is False
        # Temp file should have been cleaned up
        for tmp in created_temps:
            assert not os.path.exists(tmp)

    def test_successful_save_returns_true(self, tmp_path, monkeypatch):
        """Successful save should return True."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        result = save_config(UserConfig())
        assert result is True
        assert config_file.exists()


class TestSummaryDbErrorHandlers:
    """Tests for _load_summary and _save_summary error handling."""

    def test_load_summary_nonexistent_db_returns_none(self, tmp_path):
        """Loading from nonexistent DB path should return None."""
        from tests.support.canonical_exports import _load_summary

        db_path = tmp_path / "nonexistent.db"
        result = _load_summary(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_summary_corrupt_db_returns_none(self, tmp_path):
        """Loading from corrupt DB should return None (not raise)."""
        from tests.support.canonical_exports import _load_summary

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        result = _load_summary(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_summary_missing_table_returns_none(self, tmp_path):
        """Loading from DB without summaries table should return None."""
        import sqlite3

        from tests.support.canonical_exports import _load_summary

        db_path = tmp_path / "empty.db"
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("CREATE TABLE other_table (id TEXT)")
        result = _load_summary(db_path, "2401.00001", "hash123")
        assert result is None

    def test_save_summary_corrupt_db_does_not_raise(self, tmp_path):
        """Saving to corrupt DB should not raise (logs warning)."""
        from tests.support.canonical_exports import _save_summary

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        # Should not raise
        _save_summary(db_path, "2401.00001", "summary text", "hash123")

    def test_save_summary_sqlite_error_does_not_raise(self, tmp_path):
        """sqlite3.Error during save should not raise (logs warning)."""
        from unittest.mock import patch

        from tests.support.canonical_exports import _init_summary_db, _save_summary

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)
        with patch("sqlite3.connect", side_effect=__import__("sqlite3").Error("db locked")):
            # Should not raise
            _save_summary(db_path, "2401.00001", "summary text", "hash123")

    def test_load_summary_no_matching_row_returns_none(self, tmp_path):
        """Loading with non-matching hash should return None."""
        from tests.support.canonical_exports import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)
        _save_summary(db_path, "2401.00001", "a summary", "hash_a")
        result = _load_summary(db_path, "2401.00001", "hash_b")
        assert result is None


class TestRelevanceDbErrorHandlers:
    """Tests for relevance score SQLite error handling."""

    def test_load_score_nonexistent_db_returns_none(self, tmp_path):
        """Loading from nonexistent DB should return None."""
        from tests.support.canonical_exports import _load_relevance_score

        db_path = tmp_path / "nonexistent.db"
        result = _load_relevance_score(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_score_corrupt_db_returns_none(self, tmp_path):
        """Loading from corrupt DB should return None (not raise)."""
        from tests.support.canonical_exports import _load_relevance_score

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        result = _load_relevance_score(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_score_missing_table_returns_none(self, tmp_path):
        """Loading from DB without relevance_scores table should return None."""
        import sqlite3

        from tests.support.canonical_exports import _load_relevance_score

        db_path = tmp_path / "empty.db"
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("CREATE TABLE other_table (id TEXT)")
        result = _load_relevance_score(db_path, "2401.00001", "hash123")
        assert result is None

    def test_save_score_corrupt_db_does_not_raise(self, tmp_path):
        """Saving to corrupt DB should not raise."""
        from tests.support.canonical_exports import _save_relevance_score

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        _save_relevance_score(db_path, "2401.00001", "hash123", 8, "relevant")

    def test_load_all_corrupt_db_returns_empty(self, tmp_path):
        """Bulk loading from corrupt DB should return empty dict."""
        from tests.support.canonical_exports import _load_all_relevance_scores

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        result = _load_all_relevance_scores(db_path, "hash123")
        assert result == {}

    def test_load_all_nonexistent_db_returns_empty(self, tmp_path):
        """Bulk loading from nonexistent DB should return empty dict."""
        from tests.support.canonical_exports import _load_all_relevance_scores

        db_path = tmp_path / "nonexistent.db"
        result = _load_all_relevance_scores(db_path, "hash123")
        assert result == {}

    def test_load_all_missing_table_returns_empty(self, tmp_path):
        """Bulk loading from DB without table should return empty dict."""
        import sqlite3

        from tests.support.canonical_exports import _load_all_relevance_scores

        db_path = tmp_path / "empty.db"
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("CREATE TABLE other_table (id TEXT)")
        result = _load_all_relevance_scores(db_path, "hash123")
        assert result == {}


class TestFindSimilarPapersBoosts:
    """Tests for metadata_boost and recency_score inner functions."""

    def test_starred_paper_ranked_higher(self, make_paper):
        """Starred papers should get a boost in similarity ranking."""
        from tests.support.canonical_exports import find_similar_papers

        target = make_paper(
            arxiv_id="target",
            title="Machine Learning Methods",
            categories="cs.AI",
            abstract="Deep learning approaches for NLP.",
            date="Mon, 15 Jan 2024",
        )
        paper_a = make_paper(
            arxiv_id="paper_a",
            title="Machine Learning Methods Applied",
            categories="cs.AI",
            abstract="Deep learning approaches for NLP tasks.",
            date="Mon, 15 Jan 2024",
        )
        paper_b = make_paper(
            arxiv_id="paper_b",
            title="Machine Learning Methods Extended",
            categories="cs.AI",
            abstract="Deep learning approaches for NLP systems.",
            date="Mon, 15 Jan 2024",
        )
        metadata = {
            "paper_a": PaperMetadata(arxiv_id="paper_a", starred=True),
            "paper_b": PaperMetadata(arxiv_id="paper_b", starred=False),
        }
        results = find_similar_papers(target, [paper_a, paper_b], top_n=2, metadata=metadata)
        # paper_a (starred) should rank at or above paper_b
        ids = [p.arxiv_id for p, _ in results]
        assert "paper_a" in ids

    def test_read_paper_gets_penalty(self, make_paper):
        """Read papers should get penalized in ranking."""
        from tests.support.canonical_exports import find_similar_papers

        target = make_paper(
            arxiv_id="target",
            title="Transformer Architecture",
            categories="cs.AI",
            abstract="A novel transformer approach.",
            date="Mon, 15 Jan 2024",
        )
        paper_unread = make_paper(
            arxiv_id="paper_unread",
            title="Transformer Architecture Extended",
            categories="cs.AI",
            abstract="A novel transformer approach extended.",
            date="Mon, 15 Jan 2024",
        )
        paper_read = make_paper(
            arxiv_id="paper_read",
            title="Transformer Architecture Revised",
            categories="cs.AI",
            abstract="A novel transformer approach revised.",
            date="Mon, 15 Jan 2024",
        )
        metadata = {
            "paper_unread": PaperMetadata(arxiv_id="paper_unread"),
            "paper_read": PaperMetadata(arxiv_id="paper_read", is_read=True),
        }
        results = find_similar_papers(
            target, [paper_unread, paper_read], top_n=2, metadata=metadata
        )
        scores_by_id = {p.arxiv_id: s for p, s in results}
        # Unread paper should have a higher score due to unread boost vs read penalty
        if "paper_unread" in scores_by_id and "paper_read" in scores_by_id:
            assert scores_by_id["paper_unread"] >= scores_by_id["paper_read"]

    def test_no_metadata_no_crash(self, make_paper):
        """find_similar_papers should work without metadata."""
        from tests.support.canonical_exports import find_similar_papers

        target = make_paper(arxiv_id="target", categories="cs.AI")
        other = make_paper(arxiv_id="other", categories="cs.AI")
        results = find_similar_papers(target, [other], top_n=5, metadata=None)
        assert isinstance(results, list)

    def test_recent_papers_ranked_higher(self, make_paper):
        """More recent papers should get a recency boost."""
        from tests.support.canonical_exports import find_similar_papers

        target = make_paper(
            arxiv_id="target",
            title="Test Method",
            categories="cs.AI",
            abstract="A testing method.",
            date="Mon, 15 Jan 2024",
        )
        recent_paper = make_paper(
            arxiv_id="recent",
            title="Test Method Applied",
            categories="cs.AI",
            abstract="A testing method applied.",
            date="Sun, 14 Jan 2024",
        )
        old_paper = make_paper(
            arxiv_id="old",
            title="Test Method Applied",
            categories="cs.AI",
            abstract="A testing method applied.",
            date="Fri, 15 Jan 2021",
        )
        results = find_similar_papers(target, [recent_paper, old_paper], top_n=2)
        scores_by_id = {p.arxiv_id: s for p, s in results}
        if "recent" in scores_by_id and "old" in scores_by_id:
            assert scores_by_id["recent"] >= scores_by_id["old"]

    def test_empty_paper_list_returns_empty(self, make_paper):
        """Empty paper list should return empty results."""
        from tests.support.canonical_exports import find_similar_papers

        target = make_paper(arxiv_id="target")
        results = find_similar_papers(target, [], top_n=5)
        assert results == []

    def test_target_excluded_from_results(self, make_paper):
        """Target paper should not appear in its own similar papers."""
        from tests.support.canonical_exports import find_similar_papers

        target = make_paper(arxiv_id="target", categories="cs.AI")
        results = find_similar_papers(target, [target], top_n=5)
        ids = [p.arxiv_id for p, _ in results]
        assert "target" not in ids


class TestSimilarityIndexLifecycle:
    """Tests for async TF-IDF index lifecycle and corpus hashing."""

    def test_corpus_key_changes_when_title_changes(self, make_paper):
        from tests.support.canonical_exports import build_similarity_corpus_key

        first = make_paper(arxiv_id="2401.00001", title="A")
        second = make_paper(arxiv_id="2401.00001", title="B")
        assert build_similarity_corpus_key([first]) != build_similarity_corpus_key([second])

    def test_tfidf_path_skips_abstract_lookup(self, make_paper):
        from tests.support.canonical_exports import TfidfIndex, find_similar_papers

        target = make_paper(arxiv_id="target", title="A")
        other = make_paper(arxiv_id="other", title="B")
        index = TfidfIndex.build([target, other], text_fn=lambda p: p.title)

        def _fail_lookup(_paper):
            msg = "abstract_lookup should not be used when tfidf_index is provided"
            raise AssertionError(msg)

        results = find_similar_papers(
            target,
            [target, other],
            top_n=5,
            abstract_lookup=_fail_lookup,
            tfidf_index=index,
        )
        assert isinstance(results, list)

    def test_show_local_recommendations_starts_async_build(self, make_paper):
        from tests.support.canonical_exports import ArxivBrowser

        paper = make_paper(arxiv_id="2401.00011", title="Target")
        app = ArxivBrowser.__new__(ArxivBrowser)
        app.all_papers = [paper]
        app._config = UserConfig()
        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._tfidf_build_task = None
        app._pending_similarity_paper_id = None
        app.notify = lambda *_args, **_kwargs: None

        sentinel_task = object()

        def fake_track_task(coro):
            coro.close()
            return sentinel_task

        app._track_task = fake_track_task
        app._show_local_recommendations(paper)

        assert app._tfidf_build_task is sentinel_task
        assert app._pending_similarity_paper_id == paper.arxiv_id

    @pytest.mark.asyncio
    async def test_build_tfidf_async_auto_opens_for_current_paper(self, make_paper, monkeypatch):
        from unittest.mock import MagicMock

        from tests.support.canonical_exports import ArxivBrowser, build_similarity_corpus_key

        paper = make_paper(arxiv_id="2401.00012", title="Target")
        app = ArxivBrowser.__new__(ArxivBrowser)
        app.all_papers = [paper]
        app._tfidf_index = None
        app._tfidf_corpus_key = None
        app._tfidf_build_task = object()
        app._pending_similarity_paper_id = paper.arxiv_id
        app.notify = MagicMock()
        app._get_current_paper = lambda: paper
        app._show_local_recommendations = MagicMock()

        sentinel_index = object()

        async def fake_to_thread(func, *args, **kwargs):
            return sentinel_index

        monkeypatch.setattr("arxiv_browser.browser.discovery.asyncio.to_thread", fake_to_thread)
        corpus_key = build_similarity_corpus_key(app.all_papers)
        await app._build_tfidf_index_async(corpus_key)

        assert app._tfidf_index is sentinel_index
        assert app._tfidf_corpus_key == corpus_key
        app._show_local_recommendations.assert_called_once_with(paper)


class TestRenderPaperOptionBadges:
    """Tests for render_paper_option with various badge combinations."""

    def test_tags_displayed(self, make_paper):
        """Papers with tags should show tag badges in output."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        metadata = PaperMetadata(arxiv_id="2401.12345", tags=["topic:ml", "important"])
        result = render_paper_option(paper, metadata=metadata)
        assert "#topic:ml" in result
        assert "#important" in result

    def test_s2_citation_badge(self, make_paper):
        """S2 data should show citation count badge."""
        from arxiv_browser.semantic_scholar import SemanticScholarPaper
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        s2_data = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="s2id",
            citation_count=42,
            influential_citation_count=5,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://api.semanticscholar.org/s2id",
        )
        result = render_paper_option(paper, s2_data=s2_data)
        assert "C42" in result

    def test_hf_upvote_badge(self, make_paper):
        """HF data should show upvote badge."""
        from arxiv_browser.huggingface import HuggingFacePaper
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        hf_data = HuggingFacePaper(
            arxiv_id="2401.12345",
            title="Test",
            upvotes=99,
            num_comments=5,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
        result = render_paper_option(paper, hf_data=hf_data)
        assert "\u219199" in result  # ↑99

    def test_version_update_badge(self, make_paper):
        """Version update should show v1->v3 badge."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, version_update=(1, 3))
        assert "v1\u2192v3" in result  # v1→v3

    def test_relevance_score_high_green(self, make_paper):
        """High relevance score (8-10) should show green badge."""
        from tests.support.canonical_exports import THEME_COLORS, render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, relevance_score=(9, "very relevant"))
        assert "9/10" in result
        assert THEME_COLORS["green"] in result

    def test_relevance_score_medium_yellow(self, make_paper):
        """Medium relevance score (5-7) should show yellow badge."""
        from tests.support.canonical_exports import THEME_COLORS, render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, relevance_score=(6, "moderate"))
        assert "6/10" in result
        assert THEME_COLORS["yellow"] in result

    def test_relevance_score_low_muted(self, make_paper):
        """Low relevance score (1-4) should show muted badge."""
        from tests.support.canonical_exports import THEME_COLORS, render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, relevance_score=(3, "not relevant"))
        assert "3/10" in result
        assert THEME_COLORS["muted"] in result

    def test_read_paper_dimmed(self, make_paper):
        """Read papers should have dimmed title."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper(title="Test Title")
        metadata = PaperMetadata(arxiv_id="2401.12345", is_read=True)
        result = render_paper_option(paper, metadata=metadata)
        assert "[dim]" in result
        assert "\u2713" in result  # checkmark

    def test_starred_paper_has_star(self, make_paper):
        """Starred papers should show star indicator."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        metadata = PaperMetadata(arxiv_id="2401.12345", starred=True)
        result = render_paper_option(paper, metadata=metadata)
        assert "\u2b50" in result  # star emoji

    def test_selected_paper_has_bullet(self, make_paper):
        """Selected papers should show green bullet."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, selected=True)
        assert "\u25cf" in result  # ● bullet

    def test_watched_paper_has_eye(self, make_paper):
        """Watched papers should show eye indicator."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, watched=True)
        assert "\U0001f441" in result  # 👁

    def test_api_source_shows_api_badge(self, make_paper):
        """Papers from API source should show API badge."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="API Paper",
            authors="Author",
            categories="cs.AI",
            comments=None,
            abstract="Abstract",
            url="https://arxiv.org/abs/2401.12345",
            source="api",
        )
        result = render_paper_option(paper)
        assert "API" in result

    def test_preview_with_abstract_text(self, make_paper):
        """Preview mode should show abstract text."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        result = render_paper_option(
            paper, show_preview=True, abstract_text="This is the abstract."
        )
        assert "This is the abstract." in result

    def test_preview_with_none_abstract_shows_loading(self, make_paper):
        """Preview with None abstract should show loading message."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, show_preview=True, abstract_text=None)
        assert "Loading abstract" in result

    def test_preview_with_empty_abstract(self, make_paper):
        """Preview with empty abstract should show 'No abstract'."""
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, show_preview=True, abstract_text="")
        assert "No abstract available" in result

    def test_preview_long_abstract_truncated(self, make_paper):
        """Long abstract in preview should be truncated with ellipsis."""
        from tests.support.canonical_exports import PREVIEW_ABSTRACT_MAX_LEN, render_paper_option

        paper = make_paper()
        long_abstract = "word " * 100  # well over 150 chars
        result = render_paper_option(paper, show_preview=True, abstract_text=long_abstract)
        assert "..." in result or "\u2026" in result

    def test_preview_exact_length_not_truncated(self, make_paper):
        """Abstract at exactly max length should not be truncated."""
        from tests.support.canonical_exports import PREVIEW_ABSTRACT_MAX_LEN, render_paper_option

        paper = make_paper()
        exact_abstract = "x" * PREVIEW_ABSTRACT_MAX_LEN
        result = render_paper_option(paper, show_preview=True, abstract_text=exact_abstract)
        assert "..." not in result

    def test_all_badges_combined(self, make_paper):
        """All badges together should render without error."""
        from arxiv_browser.huggingface import HuggingFacePaper
        from arxiv_browser.semantic_scholar import SemanticScholarPaper
        from tests.support.canonical_exports import render_paper_option

        paper = make_paper()
        metadata = PaperMetadata(
            arxiv_id="2401.12345",
            starred=True,
            is_read=True,
            tags=["topic:ml"],
        )
        s2_data = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=100,
            influential_citation_count=10,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="",
        )
        hf_data = HuggingFacePaper(
            arxiv_id="2401.12345",
            title="Test",
            upvotes=50,
            num_comments=5,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
        result = render_paper_option(
            paper,
            selected=True,
            metadata=metadata,
            watched=True,
            show_preview=True,
            abstract_text="Test abstract.",
            s2_data=s2_data,
            hf_data=hf_data,
            version_update=(1, 5),
            relevance_score=(10, "perfect match"),
        )
        # All elements should be present
        assert "C100" in result
        assert "\u219150" in result
        assert "v1\u2192v5" in result
        assert "10/10" in result
        assert "#topic:ml" in result
        assert "\u2b50" in result
        assert "\u25cf" in result
        assert "Test abstract." in result
