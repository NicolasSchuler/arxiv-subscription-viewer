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


class TestPaperSimilarity:
    """Tests for paper similarity algorithm."""

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            Paper(
                arxiv_id="2401.00001",
                date="Mon, 15 Jan 2024",
                title="Deep Learning for Natural Language Processing",
                authors="John Smith, Jane Doe",
                categories="cs.CL cs.AI",
                comments=None,
                abstract="We propose a deep learning approach for NLP tasks.",
                url="https://arxiv.org/abs/2401.00001",
            ),
            Paper(
                arxiv_id="2401.00002",
                date="Tue, 16 Jan 2024",
                title="Neural Networks for Text Classification",
                authors="John Smith, Bob Wilson",
                categories="cs.CL cs.LG",
                comments=None,
                abstract="Neural network methods for classifying text documents.",
                url="https://arxiv.org/abs/2401.00002",
            ),
            Paper(
                arxiv_id="2401.00003",
                date="Wed, 17 Jan 2024",
                title="Quantum Computing Algorithms",
                authors="Alice Brown",
                categories="quant-ph cs.DS",
                comments=None,
                abstract="Novel quantum algorithms for optimization problems.",
                url="https://arxiv.org/abs/2401.00003",
            ),
        ]

    def test_jaccard_similarity_identical_sets(self):
        """Identical sets should have similarity of 1.0."""
        from arxiv_browser.similarity import _jaccard_similarity

        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_jaccard_similarity_disjoint_sets(self):
        """Disjoint sets should have similarity of 0.0."""
        from arxiv_browser.similarity import _jaccard_similarity

        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        """Partial overlap should return correct similarity."""
        from arxiv_browser.similarity import _jaccard_similarity

        result = _jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 0.01

    def test_jaccard_similarity_empty_sets(self):
        """Empty sets should return 0.0."""
        from arxiv_browser.similarity import _jaccard_similarity

        assert _jaccard_similarity(set(), set()) == 0.0

    def test_extract_keywords_filters_stopwords(self):
        """Keywords extraction should filter stopwords."""
        from arxiv_browser.similarity import _extract_keywords

        keywords = _extract_keywords("the quick brown fox and the lazy dog")
        assert "the" not in keywords
        assert "and" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords

    def test_extract_keywords_min_length(self):
        """Keywords shorter than min_length should be excluded."""
        from arxiv_browser.similarity import _extract_keywords

        keywords = _extract_keywords("a big cat sat on the mat")
        assert "a" not in keywords
        assert "cat" not in keywords  # len("cat") = 3 < 4

    def test_extract_author_lastnames(self):
        """Author lastname extraction should work correctly."""
        from arxiv_browser.similarity import _extract_author_lastnames

        lastnames = _extract_author_lastnames("John Smith, Jane Doe and Bob Wilson")
        assert "smith" in lastnames
        assert "doe" in lastnames
        assert "wilson" in lastnames

    def test_similar_papers_have_higher_score(self, sample_papers):
        """Similar papers should have higher similarity than dissimilar ones."""
        from arxiv_browser.similarity import compute_paper_similarity

        nlp_paper = sample_papers[0]
        text_paper = sample_papers[1]
        quantum_paper = sample_papers[2]

        nlp_text_sim = compute_paper_similarity(nlp_paper, text_paper)
        nlp_quantum_sim = compute_paper_similarity(nlp_paper, quantum_paper)

        assert nlp_text_sim > nlp_quantum_sim

    def test_find_similar_papers_excludes_self(self, sample_papers):
        """find_similar_papers should not include the target paper."""
        from arxiv_browser.similarity import find_similar_papers

        target = sample_papers[0]
        similar = find_similar_papers(target, sample_papers)

        arxiv_ids = [p.arxiv_id for p, _ in similar]
        assert target.arxiv_id not in arxiv_ids

    def test_find_similar_papers_respects_top_n(self, sample_papers):
        """find_similar_papers should return at most top_n results."""
        from arxiv_browser.similarity import find_similar_papers

        target = sample_papers[0]
        similar = find_similar_papers(target, sample_papers, top_n=1)

        assert len(similar) <= 1


class TestBibTeXExport:
    """Tests for BibTeX formatting functions."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("A & B", r"A \& B"),
            ("100%", r"100\%"),
            ("a_b", r"a\_b"),
            ("item #1", r"item \#1"),
            ("{braces}", r"\{braces\}"),
            (r"A & B: 100% of {items}", r"A \& B: 100\% of \{items\}"),
            ("plain text", "plain text"),
        ],
        ids=["ampersand", "percent", "underscore", "hash", "braces", "multiple", "none"],
    )
    def test_escape_bibtex_special_chars(self, text, expected):
        """Special characters should be escaped for BibTeX."""
        assert escape_bibtex(text) == expected

    @pytest.mark.parametrize(
        "date_str, expected_year",
        [
            ("Mon, 15 Jan 1999", "1999"),
            ("Mon, 15 Jan 2024", "2024"),
            ("Tue, 3 Feb 2026", "2026"),
            ("", None),  # None = current year
            ("   ", None),
            ("no date here", None),
        ],
        ids=["1999", "2024", "2026", "empty", "whitespace", "no-year"],
    )
    def test_extract_year(self, date_str, expected_year):
        """extract_year should find 4-digit years or fall back to current year."""
        result = extract_year(date_str)
        if expected_year is None:
            assert result == str(datetime.now().year)
        else:
            assert result == expected_year

    def test_generate_citation_key(self):
        """generate_citation_key should create valid BibTeX keys."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Attention Is All You Need",
            authors="John Smith, Jane Doe",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2401.12345",
        )
        key = generate_citation_key(paper)
        assert key == "smith2024attention"

    def test_generate_citation_key_unknown_author(self):
        """generate_citation_key should handle empty authors."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Some Paper",
            authors="",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2401.12345",
        )
        key = generate_citation_key(paper)
        assert key.startswith("unknown")

    def test_format_paper_as_bibtex(self):
        """format_paper_as_bibtex should produce valid BibTeX."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="cs.AI cs.LG",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2401.12345",
        )
        bibtex = format_paper_as_bibtex(paper)
        assert bibtex.startswith("@misc{")
        assert "title = {Test Paper}" in bibtex
        assert "author = {John Smith}" in bibtex
        assert "year = {2024}" in bibtex
        assert "eprint = {2401.12345}" in bibtex
        assert "primaryClass = {cs.AI}" in bibtex

    def test_format_paper_as_bibtex_empty_categories(self):
        """format_paper_as_bibtex should use 'misc' for empty categories."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2401.12345",
        )
        bibtex = format_paper_as_bibtex(paper)
        assert "primaryClass = {misc}" in bibtex


class TestQueryParser:
    """Tests for query tokenizer and parser functions."""

    def test_tokenize_simple_term(self):
        """Single word should produce one term token."""
        tokens = tokenize_query("attention")
        assert len(tokens) == 1
        assert tokens[0].kind == "term"
        assert tokens[0].value == "attention"

    def test_tokenize_multiple_terms(self):
        """Multiple words should produce multiple term tokens."""
        tokens = tokenize_query("attention mechanism")
        assert len(tokens) == 2
        assert all(t.kind == "term" for t in tokens)

    def test_tokenize_quoted_phrase(self):
        """Quoted text should produce a single phrase token."""
        tokens = tokenize_query('"attention mechanism"')
        assert len(tokens) == 1
        assert tokens[0].kind == "term"
        assert tokens[0].value == "attention mechanism"
        assert tokens[0].phrase is True

    def test_tokenize_operators(self):
        """AND, OR, NOT should be recognized as operators."""
        tokens = tokenize_query("foo AND bar")
        assert len(tokens) == 3
        assert tokens[1].kind == "op"
        assert tokens[1].value == "AND"

    @pytest.mark.parametrize("op", ["and", "And", "AND"], ids=["lower", "mixed", "upper"])
    def test_tokenize_operators_case_insensitive(self, op):
        """Operators should be case-insensitive."""
        tokens = tokenize_query(f"foo {op} bar")
        assert tokens[1].kind == "op"
        assert tokens[1].value == "AND"

    def test_tokenize_field_prefix(self):
        """Field:value should set the field attribute."""
        tokens = tokenize_query("cat:cs.AI")
        assert len(tokens) == 1
        assert tokens[0].field == "cat"
        assert tokens[0].value == "cs.AI"

    def test_tokenize_field_with_quoted_value(self):
        """Field:"quoted value" should work."""
        tokens = tokenize_query('title:"deep learning"')
        assert len(tokens) == 1
        assert tokens[0].field == "title"
        assert tokens[0].value == "deep learning"
        assert tokens[0].phrase is True

    def test_tokenize_empty_query(self):
        """Empty string should produce no tokens."""
        assert tokenize_query("") == []
        assert tokenize_query("   ") == []

    def test_tokenize_not_operator(self):
        """NOT should be recognized as a unary operator."""
        tokens = tokenize_query("NOT starred")
        assert len(tokens) == 2
        assert tokens[0].kind == "op"
        assert tokens[0].value == "NOT"

    def test_insert_implicit_and(self):
        """Adjacent terms should get AND inserted between them."""
        tokens = tokenize_query("foo bar")
        result = insert_implicit_and(tokens)
        assert len(result) == 3
        assert result[1].kind == "op"
        assert result[1].value == "AND"

    def test_insert_implicit_and_with_explicit_or(self):
        """Explicit OR should not get extra AND inserted."""
        tokens = tokenize_query("foo OR bar")
        result = insert_implicit_and(tokens)
        assert len(result) == 3
        assert result[1].value == "OR"

    def test_to_rpn_simple(self):
        """Simple AND expression should convert to RPN."""
        tokens = [
            QueryToken(kind="term", value="a"),
            QueryToken(kind="op", value="AND"),
            QueryToken(kind="term", value="b"),
        ]
        rpn = to_rpn(tokens)
        assert len(rpn) == 3
        assert rpn[0].value == "a"
        assert rpn[1].value == "b"
        assert rpn[2].value == "AND"

    def test_to_rpn_precedence(self):
        """AND should bind tighter than OR."""
        tokens = [
            QueryToken(kind="term", value="a"),
            QueryToken(kind="op", value="OR"),
            QueryToken(kind="term", value="b"),
            QueryToken(kind="op", value="AND"),
            QueryToken(kind="term", value="c"),
        ]
        rpn = to_rpn(tokens)
        assert [t.value for t in rpn] == ["a", "b", "c", "AND", "OR"]

    def test_to_rpn_not_highest_precedence(self):
        """NOT should have highest precedence."""
        tokens = [
            QueryToken(kind="op", value="NOT"),
            QueryToken(kind="term", value="a"),
            QueryToken(kind="op", value="AND"),
            QueryToken(kind="term", value="b"),
        ]
        rpn = to_rpn(tokens)
        assert [t.value for t in rpn] == ["a", "NOT", "b", "AND"]

    @pytest.mark.parametrize("field", ["title", "author", "abstract", "cat", "tag"])
    def test_all_field_types(self, field):
        """All supported field prefixes should be recognized."""
        tokens = tokenize_query(f"{field}:test")
        assert tokens[0].field == field, f"Field {field} not recognized"


class TestConfigIO:
    """Tests for configuration loading and saving."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Config should survive save/load roundtrip."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config = UserConfig(bibtex_export_dir="/custom/path")
        assert save_config(config) is True

        loaded = load_config()
        assert loaded.bibtex_export_dir == "/custom/path"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        """Missing config file should return default UserConfig."""
        config_file = tmp_path / "nonexistent" / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config = load_config()
        assert config.bibtex_export_dir == ""  # default

    def test_load_corrupted_json(self, tmp_path, monkeypatch):
        """Corrupted JSON should return default UserConfig."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json {{{{", encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        config = load_config()
        assert isinstance(config, UserConfig)

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        """save_config should create parent directories."""
        config_file = tmp_path / "deep" / "nested" / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        assert save_config(UserConfig()) is True
        assert config_file.exists()

    def test_atomic_write_produces_valid_json(self, tmp_path, monkeypatch):
        """Atomic write should produce valid JSON that can be parsed."""
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        save_config(UserConfig(pdf_download_dir="/test/path"))
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["pdf_download_dir"] == "/test/path"


class TestPdfDownloadPathValidation:
    """Tests for path traversal validation in get_pdf_download_path."""

    def test_valid_arxiv_id_produces_correct_path(self, tmp_path):
        """Normal arXiv ID should produce correct path."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="",
            title="Test",
            authors="Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2401.12345",
        )
        config = UserConfig(pdf_download_dir=str(tmp_path))
        path = get_pdf_download_path(paper, config)
        assert path.name == "2401.12345.pdf"
        assert str(path).startswith(str(tmp_path))

    def test_path_traversal_rejected(self, tmp_path):
        """arXiv ID with path traversal should raise ValueError."""
        paper = Paper(
            arxiv_id="../../etc/passwd",
            date="",
            title="Test",
            authors="Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/test",
        )
        config = UserConfig(pdf_download_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Invalid arXiv ID"):
            get_pdf_download_path(paper, config)


class TestTruncateText:
    """Tests for text truncation utility."""

    def test_short_text_unchanged(self):
        """Text shorter than max_len should be returned unchanged."""
        from arxiv_browser.query import truncate_text

        assert truncate_text("Hello", 10) == "Hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_len should be returned unchanged."""
        from arxiv_browser.query import truncate_text

        assert truncate_text("Hello", 5) == "Hello"

    def test_long_text_truncated(self):
        """Text longer than max_len should be truncated with suffix."""
        from arxiv_browser.query import truncate_text

        assert truncate_text("Hello World", 5) == "Hello..."

    def test_custom_suffix(self):
        """Custom suffix should be used when provided."""
        from arxiv_browser.query import truncate_text

        assert truncate_text("Hello World", 5, suffix=">>>") == "Hello>>>"

    def test_empty_string(self):
        """Empty string should be handled correctly."""
        from arxiv_browser.query import truncate_text

        assert truncate_text("", 10) == ""


class TestSafeGetAndTypeValidation:
    """Tests for type-safe configuration parsing."""

    def test_safe_get_correct_type(self):
        """_safe_get should return value when type matches."""
        from arxiv_browser.config import _safe_get

        data = {"key": 42}
        assert _safe_get(data, "key", 0, int) == 42

    def test_safe_get_wrong_type(self):
        """_safe_get should return default when type doesn't match."""
        from arxiv_browser.config import _safe_get

        data = {"key": "not_an_int"}
        assert _safe_get(data, "key", 0, int) == 0

    def test_safe_get_rejects_bool_for_int(self):
        """_safe_get should reject bool values for integer fields."""
        from arxiv_browser.config import _safe_get

        data = {"key": True}
        assert _safe_get(data, "key", 50, int) == 50

    def test_safe_get_missing_key(self):
        """_safe_get should return default for missing key."""
        from arxiv_browser.config import _safe_get

        data = {}
        assert _safe_get(data, "key", "default", str) == "default"

    def test_dict_to_config_handles_invalid_types(self):
        """_dict_to_config should handle invalid types gracefully."""
        from arxiv_browser.config import _dict_to_config

        invalid_data = {
            "session": {
                "scroll_index": "not_an_int",
                "current_filter": 123,
                "sort_index": None,
            },
            "paper_metadata": "not_a_dict",
            "arxiv_api_max_results": "invalid",
            "version": "1",
        }

        config = _dict_to_config(invalid_data)

        assert config.session.scroll_index == 0
        assert config.session.current_filter == ""
        assert config.session.sort_index == 0
        assert config.paper_metadata == {}
        assert config.arxiv_api_max_results == ARXIV_API_DEFAULT_MAX_RESULTS
        assert config.version == 1

    def test_dict_to_config_rejects_bool_for_int_fields(self):
        """_dict_to_config should treat bool as invalid for integer configuration fields."""
        from arxiv_browser.config import _dict_to_config

        config = _dict_to_config({"arxiv_api_max_results": True})
        assert config.arxiv_api_max_results == ARXIV_API_DEFAULT_MAX_RESULTS


class TestPaperDeduplication:
    """Tests for paper deduplication in parser."""

    def test_duplicate_arxiv_ids_skipped(self, tmp_path):
        """Parser should skip papers with duplicate arXiv IDs."""
        content = """\\
arXiv:2401.00001
Date: Mon, 15 Jan 2024
Title: First Paper
Authors: Author One
Categories: cs.AI
\\\\
This is the first paper abstract.
\\\\
( https://arxiv.org/abs/2401.00001 ,
------------------------------------------------------------------------------
\\
arXiv:2401.00001
Date: Tue, 16 Jan 2024
Title: Duplicate Paper - Should Be Skipped
Authors: Author Two
Categories: cs.LG
\\\\
This is a duplicate entry that should be skipped.
\\\\
( https://arxiv.org/abs/2401.00001 ,
------------------------------------------------------------------------------
\\
arXiv:2401.00002
Date: Wed, 17 Jan 2024
Title: Second Paper
Authors: Author Three
Categories: cs.CV
\\\\
This is the second unique paper abstract.
\\\\
( https://arxiv.org/abs/2401.00002 ,
"""
        file_path = tmp_path / "duplicates.txt"
        file_path.write_text(content)

        papers = parse_arxiv_file(file_path)

        assert len(papers) == 2
        arxiv_ids = [p.arxiv_id for p in papers]
        assert "2401.00001" in arxiv_ids
        assert "2401.00002" in arxiv_ids

        first_paper = next(p for p in papers if p.arxiv_id == "2401.00001")
        assert first_paper.title == "First Paper"

    def test_versioned_duplicates_keep_highest_version(self, tmp_path):
        """Parser should keep the highest version when duplicate bare IDs exist."""
        content = """\\
arXiv:2401.11111v1
Date: Mon, 15 Jan 2024
Title: Lower Version
Authors: Author One
Categories: cs.AI
\\\\
First abstract.
\\\\
( https://arxiv.org/abs/2401.11111v1 ,
------------------------------------------------------------------------------
\\
arXiv:2401.11111v3
Date: Tue, 16 Jan 2024
Title: Higher Version
Authors: Author Two
Categories: cs.LG
\\\\
Second abstract.
\\\\
( https://arxiv.org/abs/2401.11111v3 ,
------------------------------------------------------------------------------
"""
        file_path = tmp_path / "versioned_duplicates.txt"
        file_path.write_text(content)

        papers = parse_arxiv_file(file_path)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.11111"
        assert papers[0].title == "Higher Version"

    def test_mixed_bare_and_versioned_duplicates_keep_highest_version(self, tmp_path):
        """Bare IDs are treated as v1 and replaced by higher explicit versions."""
        content = """\\
arXiv:2401.22222
Date: Mon, 15 Jan 2024
Title: Bare Version
Authors: Author One
Categories: cs.AI
\\\\
First abstract.
\\\\
( https://arxiv.org/abs/2401.22222 ,
------------------------------------------------------------------------------
\\
arXiv:2401.22222v2
Date: Tue, 16 Jan 2024
Title: Explicit Higher Version
Authors: Author Two
Categories: cs.LG
\\\\
Second abstract.
\\\\
( https://arxiv.org/abs/2401.22222v2 ,
------------------------------------------------------------------------------
"""
        file_path = tmp_path / "mixed_duplicates.txt"
        file_path.write_text(content)

        papers = parse_arxiv_file(file_path)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.22222"
        assert papers[0].title == "Explicit Higher Version"

    def test_legacy_arxiv_id_is_parsed_and_normalized(self, tmp_path):
        """Legacy arXiv IDs should be parsed and normalized by stripping version suffix."""
        content = """\\
arXiv:hep-th/9901001v2
Date: Mon, 15 Jan 2024
Title: Legacy Format Paper
Authors: Legacy Author
Categories: hep-th
\\\\
Legacy abstract text.
\\\\
( https://arxiv.org/abs/hep-th/9901001v2 ,
------------------------------------------------------------------------------
"""
        file_path = tmp_path / "legacy_id.txt"
        file_path.write_text(content)

        papers = parse_arxiv_file(file_path)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "hep-th/9901001"


class TestHistoryFileDiscovery:
    """Tests for history file discovery functionality."""

    def test_discover_history_files_empty_dir(self, tmp_path):
        """discover_history_files should return empty list for empty history dir."""
        from arxiv_browser.parsing import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        assert discover_history_files(tmp_path) == []

    def test_discover_history_files_no_history_dir(self, tmp_path):
        """discover_history_files should return empty list when history/ doesn't exist."""
        from arxiv_browser.parsing import discover_history_files

        assert discover_history_files(tmp_path) == []

    def test_discover_history_files_respects_limit(self, tmp_path):
        """discover_history_files should respect the limit parameter."""
        from arxiv_browser.parsing import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        for i in range(10):
            (history_dir / f"2024-01-{i + 10:02d}.txt").write_text("test")

        result = discover_history_files(tmp_path, limit=5)
        assert len(result) == 5

        dates = [d for d, _ in result]
        assert dates == sorted(dates, reverse=True)

    def test_discover_history_files_default_is_unbounded(self, tmp_path):
        """discover_history_files should return all files when limit is omitted."""
        from arxiv_browser.parsing import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        for i in range(10):
            (history_dir / f"2024-01-{i + 10:02d}.txt").write_text("test")

        result = discover_history_files(tmp_path)
        assert len(result) == 10

    def test_max_history_files_constant_is_positive(self):
        """MAX_HISTORY_FILES constant should be positive."""
        from arxiv_browser.browser.core import MAX_HISTORY_FILES

        assert MAX_HISTORY_FILES > 0

    def test_discover_history_files_skips_invalid_names(self, tmp_path):
        """discover_history_files should skip files that don't match YYYY-MM-DD pattern."""
        from arxiv_browser.parsing import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        (history_dir / "2024-01-15.txt").write_text("valid")
        (history_dir / "invalid.txt").write_text("invalid")
        (history_dir / "2024-13-01.txt").write_text("invalid month")
        (history_dir / "notes.txt").write_text("notes")

        result = discover_history_files(tmp_path)
        assert len(result) == 1
        assert result[0][0].isoformat() == "2024-01-15"

    def test_discover_history_files_handles_glob_oserror(self, tmp_path, monkeypatch, caplog):
        """discover_history_files should return [] when history directory can't be read."""
        import logging

        from arxiv_browser.parsing import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        original_glob = Path.glob

        def fake_glob(path_obj: Path, pattern: str):
            if path_obj == history_dir and pattern == "*.txt":
                msg = "permission denied"
                raise OSError(msg)
            return original_glob(path_obj, pattern)

        monkeypatch.setattr(Path, "glob", fake_glob)
        with caplog.at_level(logging.WARNING):
            result = discover_history_files(tmp_path)
        assert result == []
        assert "Failed to enumerate history files" in caplog.text


class TestYearExtractionEdgeCases:
    """Tests for year extraction in BibTeX export."""

    def test_extract_year_whitespace_only(self):
        """Year extraction should handle whitespace-only date."""
        result = extract_year("   ")
        assert len(result) == 4
        assert result.isdigit()

    def test_extract_year_empty_string(self):
        """Year extraction should handle empty string."""
        result = extract_year("")
        assert len(result) == 4
        assert result.isdigit()


class TestBibTeXFormattingEdgeCases:
    """Tests for BibTeX formatting edge cases."""

    def test_format_bibtex_empty_categories(self):
        """BibTeX formatting should handle empty categories."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="",
            comments=None,
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )

        bibtex = format_paper_as_bibtex(paper)
        assert "primaryClass = {misc}" in bibtex

    def test_format_bibtex_whitespace_categories(self):
        """BibTeX formatting should handle whitespace-only categories."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="   ",
            comments=None,
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )

        bibtex = format_paper_as_bibtex(paper)
        assert "primaryClass = {misc}" in bibtex


class TestPdfDownloadConfig:
    """Tests for PDF download configuration."""

    def test_pdf_download_dir_default_empty(self):
        """Default pdf_download_dir should be empty string."""
        config = UserConfig()
        assert config.pdf_download_dir == ""

    def test_pdf_download_dir_serialization_roundtrip(self):
        """pdf_download_dir should survive config serialization."""
        from arxiv_browser.config import (
            _config_to_dict,
            _dict_to_config,
        )

        config = UserConfig(pdf_download_dir="/custom/path")
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.pdf_download_dir == "/custom/path"

    def test_get_pdf_download_path_default(self, tmp_path, monkeypatch):
        """Default path should be ~/arxiv-pdfs/{arxiv_id}.pdf."""
        from arxiv_browser.export import DEFAULT_PDF_DOWNLOAD_DIR

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        paper = Paper(
            arxiv_id="2301.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="Test Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2301.12345",
        )
        config = UserConfig()

        path = get_pdf_download_path(paper, config)
        assert path == tmp_path / DEFAULT_PDF_DOWNLOAD_DIR / "2301.12345.pdf"

    def test_get_pdf_download_path_custom_dir(self, tmp_path):
        """Custom dir should be used when configured."""
        paper = Paper(
            arxiv_id="2301.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="Test Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2301.12345",
        )
        config = UserConfig(pdf_download_dir=str(tmp_path / "my-pdfs"))

        path = get_pdf_download_path(paper, config)
        assert path == tmp_path / "my-pdfs" / "2301.12345.pdf"
