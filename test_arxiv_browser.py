#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser import (
    ARXIV_DATE_FORMAT,
    DEFAULT_CATEGORY_COLOR,
    Paper,
    PaperMetadata,
    QueryToken,
    SORT_OPTIONS,
    SUBPROCESS_TIMEOUT,
    UserConfig,
    clean_latex,
    escape_bibtex,
    extract_year,
    format_categories,
    format_paper_as_bibtex,
    generate_citation_key,
    get_pdf_download_path,
    insert_implicit_and,
    load_config,
    parse_arxiv_date,
    parse_arxiv_file,
    save_config,
    to_rpn,
    tokenize_query,
)


# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestCleanLatex:
    """Tests for LaTeX cleaning functionality."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Hello World", "Hello World"),
            ("Hello    World", "Hello World"),
            ("  leading and trailing  ", "leading and trailing"),
            ("No LaTeX here at all", "No LaTeX here at all"),
        ],
        ids=["plain", "extra-whitespace", "leading-trailing", "short-circuit"],
    )
    def test_whitespace_and_plain_text(self, text, expected):
        """Plain text and whitespace normalization."""
        assert clean_latex(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            (r"\textbf{bold text}", "bold text"),
            (r"\textit{italic text}", "italic text"),
            (r"\emph{emphasized}", "emphasized"),
            (r"\unknown{content}", "content"),
        ],
        ids=["textbf", "textit", "emph", "unknown-command"],
    )
    def test_command_removal(self, text, expected):
        """LaTeX commands with braces should keep content."""
        assert clean_latex(text) == expected

    def test_standalone_command_removed(self):
        """Standalone commands without braces should be removed."""
        result = clean_latex(r"\noindent Some text")
        assert "noindent" not in result
        assert "Some text" in result

    @pytest.mark.parametrize(
        "text, expected",
        [
            (r"$x^2$", "x^2"),
            (r"The formula $E=mc^2$ is famous", "The formula E=mc^2 is famous"),
            (r"Price is \$100", "Price is $100"),
            (r"A \& B", "A & B"),
            (r"caf\'e", "café"),
            (r'M\"{u}ller', "Müller"),
            (r"\c{c}", "ç"),
        ],
        ids=["math", "inline-math", "escaped-dollar", "ampersand", "acute", "umlaut", "cedilla"],
    )
    def test_special_chars_and_math(self, text, expected):
        """Math mode, escaped chars, and accent commands."""
        assert clean_latex(text) == expected

    def test_complex_latex_mixed(self):
        """Complex text with multiple LaTeX commands should be cleaned."""
        text = r"\textbf{Bold} and $x^2$ with \emph{emphasis}"
        result = clean_latex(text)
        assert "Bold" in result
        assert "x^2" in result
        assert "emphasis" in result
        assert "\\" not in result
        assert "$" not in result

    def test_nested_latex_commands(self):
        """Nested LaTeX commands should be handled via iterative cleaning."""
        text = r"\textbf{$O(n)$}"
        result = clean_latex(text)
        assert "O(n)" in result
        assert "textbf" not in result
        assert "$" not in result

    def test_nested_braces_limitation(self):
        """Document known limitation: nested braces inside commands."""
        text = r"\textbf{$O(n^{2})$}"
        result = clean_latex(text)
        assert "O(n" in result
        assert "textbf" not in result.lower()

    def test_deeply_nested_commands(self):
        """Deeply nested LaTeX commands should be fully cleaned."""
        text = r"\textbf{\emph{nested \textit{content}}}"
        result = clean_latex(text)
        assert "nested" in result
        assert "content" in result
        assert "\\" not in result

    def test_math_with_nested_braces(self):
        """Math mode with nested braces should preserve structure."""
        text = r"$\sum_{i=1}^{n}$"
        result = clean_latex(text)
        assert "$" not in result


# ============================================================================
# Tests for parse_arxiv_date function
# ============================================================================


class TestParseArxivDate:
    """Tests for date parsing functionality."""

    @pytest.mark.parametrize(
        "date_str, year, month, day",
        [
            ("Mon, 15 Jan 2024", 2024, 1, 15),
            ("Mon, 15 Jan 2024 00:00:00 GMT", 2024, 1, 15),
            ("Tue, 25 Dec 2023", 2023, 12, 25),
            ("  Mon, 15 Jan 2024  ", 2024, 1, 15),
        ],
        ids=["basic", "with-timezone", "different-date", "whitespace-trimmed"],
    )
    def test_valid_dates(self, date_str, year, month, day):
        """Valid arXiv date strings should be parsed correctly."""
        result = parse_arxiv_date(date_str)
        assert (result.year, result.month, result.day) == (year, month, day)

    def test_malformed_date_returns_min(self):
        """Malformed dates should return datetime.min."""
        result = parse_arxiv_date("invalid date")
        assert result == datetime.min

    def test_empty_string_returns_min(self):
        """Empty string should return datetime.min."""
        result = parse_arxiv_date("")
        assert result == datetime.min

    def test_date_sorting_order(self):
        """Dates should sort in correct chronological order."""
        dates = [
            "Mon, 9 Jan 2024",
            "Mon, 15 Jan 2024",
            "Tue, 2 Jan 2024",
        ]
        parsed = sorted(dates, key=parse_arxiv_date)
        assert parse_arxiv_date(parsed[0]).day == 2
        assert parse_arxiv_date(parsed[1]).day == 9
        assert parse_arxiv_date(parsed[2]).day == 15

    def test_date_sorting_with_string_would_fail(self):
        """Demonstrate that string sorting produces wrong results."""
        dates = ["Mon, 9 Jan 2024", "Mon, 15 Jan 2024"]
        string_sorted = sorted(dates)
        assert string_sorted[0] == "Mon, 15 Jan 2024"
        date_sorted = sorted(dates, key=parse_arxiv_date)
        assert parse_arxiv_date(date_sorted[0]).day == 9


# ============================================================================
# Tests for format_categories function
# ============================================================================


class TestFormatCategories:
    """Tests for category formatting with colors."""

    def test_single_known_category(self):
        """Known category should get its assigned color."""
        result = format_categories("cs.AI")
        assert "cs.AI" in result
        assert "#f92672" in result  # Monokai pink for cs.AI

    def test_single_unknown_category(self):
        """Unknown category should get default gray color."""
        result = format_categories("unknown.cat")
        assert "unknown.cat" in result
        assert DEFAULT_CATEGORY_COLOR in result

    def test_multiple_categories(self):
        """Multiple categories should all be formatted."""
        result = format_categories("cs.AI cs.LG cs.CL")
        assert "cs.AI" in result
        assert "cs.LG" in result
        assert "cs.CL" in result

    def test_caching(self):
        """Same input should return cached result."""
        result1 = format_categories("cs.AI cs.LG")
        result2 = format_categories("cs.AI cs.LG")
        assert result1 == result2


# ============================================================================
# Tests for parse_arxiv_file function
# ============================================================================


class TestParseArxivFile:
    """Tests for arXiv file parsing."""

    @pytest.fixture
    def sample_arxiv_content(self):
        """Sample arXiv email content for testing."""
        return """------------------------------------------------------------------------------
\\\\
arXiv:2401.12345
Date: Mon, 15 Jan 2024 00:00:00 GMT   (100kb)

Title: A Sample Paper About Machine Learning
Authors: John Doe, Jane Smith
Categories: cs.AI cs.LG
Comments: 10 pages, 5 figures
\\\\
  This is the abstract of the paper. It describes the key contributions
  and findings of the research.
\\\\
( https://arxiv.org/abs/2401.12345 ,  100kb)
------------------------------------------------------------------------------
\\\\
arXiv:2401.67890
Date: Tue, 16 Jan 2024 00:00:00 GMT   (50kb)

Title: Another Paper on Natural Language Processing
Authors: Alice Johnson
Categories: cs.CL
\\\\
  Abstract of the second paper about NLP research.
\\\\
( https://arxiv.org/abs/2401.67890 ,  50kb)
------------------------------------------------------------------------------
"""

    @pytest.fixture
    def temp_arxiv_file(self, sample_arxiv_content, tmp_path):
        """Create a temporary file with sample content using pytest tmp_path."""
        file_path = tmp_path / "test_arxiv.txt"
        file_path.write_text(sample_arxiv_content)
        return file_path

    def test_parse_returns_list_of_papers(self, temp_arxiv_file):
        """Parser should return a list of Paper objects."""
        papers = parse_arxiv_file(temp_arxiv_file)
        assert isinstance(papers, list)
        assert all(isinstance(p, Paper) for p in papers)

    def test_parse_extracts_correct_count(self, temp_arxiv_file):
        """Parser should extract the correct number of papers."""
        papers = parse_arxiv_file(temp_arxiv_file)
        assert len(papers) == 2

    def test_parse_extracts_arxiv_id(self, temp_arxiv_file):
        """Parser should correctly extract arXiv IDs."""
        papers = parse_arxiv_file(temp_arxiv_file)
        arxiv_ids = [p.arxiv_id for p in papers]
        assert "2401.12345" in arxiv_ids
        assert "2401.67890" in arxiv_ids

    def test_parse_extracts_title(self, temp_arxiv_file):
        """Parser should correctly extract titles."""
        papers = parse_arxiv_file(temp_arxiv_file)
        titles = [p.title for p in papers]
        assert any("Sample Paper About Machine Learning" in t for t in titles)

    def test_parse_extracts_authors(self, temp_arxiv_file):
        """Parser should correctly extract authors."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.12345")
        assert "John Doe" in paper.authors
        assert "Jane Smith" in paper.authors

    def test_parse_extracts_categories(self, temp_arxiv_file):
        """Parser should correctly extract categories."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.12345")
        assert "cs.AI" in paper.categories
        assert "cs.LG" in paper.categories

    def test_parse_extracts_comments(self, temp_arxiv_file):
        """Parser should correctly extract comments when present."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.12345")
        assert paper.comments is not None
        assert "10 pages" in paper.comments

    def test_parse_handles_missing_comments(self, temp_arxiv_file):
        """Parser should set comments to None when not present."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.67890")
        assert paper.comments is None

    def test_parse_extracts_url(self, temp_arxiv_file):
        """Parser should correctly extract URLs."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.12345")
        assert paper.url == "https://arxiv.org/abs/2401.12345"

    def test_parse_empty_file(self, tmp_path):
        """Parser should return empty list for empty file."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")
        papers = parse_arxiv_file(file_path)
        assert papers == []

    def test_parse_malformed_entries_skipped(self, tmp_path):
        """Parser should skip malformed entries without crashing."""
        content = """Some random text that is not a valid entry
------------------------------------------------------------------------------
This is also not valid
------------------------------------------------------------------------------
"""
        file_path = tmp_path / "malformed.txt"
        file_path.write_text(content)
        papers = parse_arxiv_file(file_path)
        assert papers == []


# ============================================================================
# Tests for Paper dataclass
# ============================================================================


class TestPaperDataclass:
    """Tests for Paper dataclass."""

    def test_paper_creation(self):
        """Paper should be created with all required fields."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="Test Author",
            categories="cs.AI",
            comments="5 pages",
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )
        assert paper.arxiv_id == "2401.12345"
        assert paper.title == "Test Paper"

    def test_paper_comments_optional(self):
        """Paper should accept None for comments."""
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="Test Author",
            categories="cs.AI",
            comments=None,
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )
        assert paper.comments is None

    def test_paper_uses_slots(self):
        """Paper should use __slots__ for memory efficiency."""
        assert hasattr(Paper, "__slots__")


# ============================================================================
# Tests for constants (consolidated from TestConstants, TestFuzzySearchConstants,
# TestUIConstants — only keeping tests that enforce real invariants)
# ============================================================================


class TestConstants:
    """Tests for module constants with real invariants."""

    def test_sort_options_defined(self):
        """SORT_OPTIONS should contain the three expected sort keys."""
        assert "title" in SORT_OPTIONS
        assert "date" in SORT_OPTIONS
        assert "arxiv_id" in SORT_OPTIONS

    def test_default_category_color_is_hex(self):
        """DEFAULT_CATEGORY_COLOR should be a valid #RRGGBB hex color."""
        assert DEFAULT_CATEGORY_COLOR.startswith("#")
        assert len(DEFAULT_CATEGORY_COLOR) == 7

    def test_arxiv_date_format_valid(self):
        """ARXIV_DATE_FORMAT should be a valid strptime format."""
        sample_date = "Mon, 15 Jan 2024"
        parsed = datetime.strptime(sample_date, ARXIV_DATE_FORMAT)
        assert parsed.year == 2024

    def test_fuzzy_score_cutoff_valid_range(self):
        """FUZZY_SCORE_CUTOFF should be in 0-100 range."""
        from arxiv_browser import FUZZY_SCORE_CUTOFF

        assert 0 <= FUZZY_SCORE_CUTOFF <= 100

    def test_stopwords_contains_common_words(self):
        """STOPWORDS should contain common English stopwords."""
        from arxiv_browser import STOPWORDS

        assert "the" in STOPWORDS
        assert "and" in STOPWORDS
        assert "or" in STOPWORDS
        assert "is" in STOPWORDS


# ============================================================================
# Tests for new dataclasses
# ============================================================================


class TestNewDataclasses:
    """Tests for Phase 1 dataclasses."""

    def test_paper_metadata_defaults(self):
        """PaperMetadata should have correct defaults."""
        meta = PaperMetadata(arxiv_id="2401.12345")
        assert meta.arxiv_id == "2401.12345"
        assert meta.notes == ""
        assert meta.tags == []
        assert meta.is_read is False
        assert meta.starred is False

    def test_watch_list_entry_defaults(self):
        """WatchListEntry should have correct defaults."""
        from arxiv_browser import WatchListEntry

        entry = WatchListEntry(pattern="test")
        assert entry.pattern == "test"
        assert entry.match_type == "author"
        assert entry.case_sensitive is False

    def test_search_bookmark_creation(self):
        """SearchBookmark should store name and query."""
        from arxiv_browser import SearchBookmark

        bookmark = SearchBookmark(name="AI Papers", query="cat:cs.AI")
        assert bookmark.name == "AI Papers"
        assert bookmark.query == "cat:cs.AI"

    def test_session_state_defaults(self):
        """SessionState should have correct defaults."""
        from arxiv_browser import SessionState

        session = SessionState()
        assert session.scroll_index == 0
        assert session.current_filter == ""
        assert session.sort_index == 0
        assert session.selected_ids == []

    def test_user_config_defaults(self):
        """UserConfig should have correct defaults."""
        config = UserConfig()
        assert config.paper_metadata == {}
        assert config.watch_list == []
        assert config.bookmarks == []
        assert config.marks == {}
        assert config.show_abstract_preview is False
        assert config.version == 1


# ============================================================================
# Tests for config persistence
# ============================================================================


class TestConfigPersistence:
    """Tests for configuration save/load functions."""

    def test_config_to_dict_roundtrip(self):
        """Config should serialize and deserialize correctly."""
        from arxiv_browser import (
            WatchListEntry,
            SearchBookmark,
            SessionState,
            _config_to_dict,
            _dict_to_config,
        )

        original = UserConfig(
            paper_metadata={
                "2401.12345": PaperMetadata(
                    arxiv_id="2401.12345",
                    notes="Test note",
                    tags=["important", "to-read"],
                    is_read=True,
                    starred=True,
                )
            },
            watch_list=[WatchListEntry(pattern="Smith", match_type="author")],
            bookmarks=[SearchBookmark(name="AI", query="cat:cs.AI")],
            marks={"a": "2401.12345"},
            session=SessionState(scroll_index=5, current_filter="test"),
            show_abstract_preview=True,
            bibtex_export_dir="custom-exports",
        )

        # Serialize and deserialize
        data = _config_to_dict(original)
        restored = _dict_to_config(data)

        assert restored.show_abstract_preview is True
        assert "2401.12345" in restored.paper_metadata
        assert restored.paper_metadata["2401.12345"].notes == "Test note"
        assert restored.paper_metadata["2401.12345"].tags == ["important", "to-read"]
        assert len(restored.watch_list) == 1
        assert restored.watch_list[0].pattern == "Smith"
        assert len(restored.bookmarks) == 1
        assert restored.marks["a"] == "2401.12345"
        assert restored.session.scroll_index == 5
        assert restored.bibtex_export_dir == "custom-exports"

    def test_dict_to_config_handles_empty(self):
        """Loading empty dict should return default config."""
        from arxiv_browser import _dict_to_config

        config = _dict_to_config({})
        default = UserConfig()
        assert config.show_abstract_preview == default.show_abstract_preview
        assert config.paper_metadata == default.paper_metadata


# ============================================================================
# Tests for paper similarity
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
        from arxiv_browser import _jaccard_similarity

        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_jaccard_similarity_disjoint_sets(self):
        """Disjoint sets should have similarity of 0.0."""
        from arxiv_browser import _jaccard_similarity

        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        """Partial overlap should return correct similarity."""
        from arxiv_browser import _jaccard_similarity

        result = _jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 0.01

    def test_jaccard_similarity_empty_sets(self):
        """Empty sets should return 0.0."""
        from arxiv_browser import _jaccard_similarity

        assert _jaccard_similarity(set(), set()) == 0.0

    def test_extract_keywords_filters_stopwords(self):
        """Keywords extraction should filter stopwords."""
        from arxiv_browser import _extract_keywords

        keywords = _extract_keywords("the quick brown fox and the lazy dog")
        assert "the" not in keywords
        assert "and" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords

    def test_extract_keywords_min_length(self):
        """Keywords shorter than min_length should be excluded."""
        from arxiv_browser import _extract_keywords

        keywords = _extract_keywords("a big cat sat on the mat")
        assert "a" not in keywords
        assert "cat" not in keywords  # len("cat") = 3 < 4

    def test_extract_author_lastnames(self):
        """Author lastname extraction should work correctly."""
        from arxiv_browser import _extract_author_lastnames

        lastnames = _extract_author_lastnames("John Smith, Jane Doe and Bob Wilson")
        assert "smith" in lastnames
        assert "doe" in lastnames
        assert "wilson" in lastnames

    def test_similar_papers_have_higher_score(self, sample_papers):
        """Similar papers should have higher similarity than dissimilar ones."""
        from arxiv_browser import compute_paper_similarity

        nlp_paper = sample_papers[0]
        text_paper = sample_papers[1]
        quantum_paper = sample_papers[2]

        nlp_text_sim = compute_paper_similarity(nlp_paper, text_paper)
        nlp_quantum_sim = compute_paper_similarity(nlp_paper, quantum_paper)

        assert nlp_text_sim > nlp_quantum_sim

    def test_find_similar_papers_excludes_self(self, sample_papers):
        """find_similar_papers should not include the target paper."""
        from arxiv_browser import find_similar_papers

        target = sample_papers[0]
        similar = find_similar_papers(target, sample_papers)

        arxiv_ids = [p.arxiv_id for p, _ in similar]
        assert target.arxiv_id not in arxiv_ids

    def test_find_similar_papers_respects_top_n(self, sample_papers):
        """find_similar_papers should return at most top_n results."""
        from arxiv_browser import find_similar_papers

        target = sample_papers[0]
        similar = find_similar_papers(target, sample_papers, top_n=1)

        assert len(similar) <= 1


# ============================================================================
# Tests for BibTeX export
# ============================================================================


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
            ("Mon, 15 Jan 2024", "2024"),
            ("Tue, 3 Feb 2026", "2026"),
            ("", None),  # None = current year
            ("   ", None),
            ("no date here", None),
        ],
        ids=["2024", "2026", "empty", "whitespace", "no-year"],
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


# ============================================================================
# Tests for query parser
# ============================================================================


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


# ============================================================================
# Tests for config I/O
# ============================================================================


class TestConfigIO:
    """Tests for configuration loading and saving."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Config should survive save/load roundtrip."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.get_config_path", lambda: config_file)

        config = UserConfig(bibtex_export_dir="/custom/path")
        assert save_config(config) is True

        loaded = load_config()
        assert loaded.bibtex_export_dir == "/custom/path"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        """Missing config file should return default UserConfig."""
        config_file = tmp_path / "nonexistent" / "config.json"
        monkeypatch.setattr("arxiv_browser.get_config_path", lambda: config_file)

        config = load_config()
        assert config.bibtex_export_dir == ""  # default

    def test_load_corrupted_json(self, tmp_path, monkeypatch):
        """Corrupted JSON should return default UserConfig."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json {{{{", encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.get_config_path", lambda: config_file)

        config = load_config()
        assert isinstance(config, UserConfig)

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        """save_config should create parent directories."""
        config_file = tmp_path / "deep" / "nested" / "config.json"
        monkeypatch.setattr("arxiv_browser.get_config_path", lambda: config_file)

        assert save_config(UserConfig()) is True
        assert config_file.exists()

    def test_atomic_write_produces_valid_json(self, tmp_path, monkeypatch):
        """Atomic write should produce valid JSON that can be parsed."""
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.get_config_path", lambda: config_file)

        save_config(UserConfig(pdf_download_dir="/test/path"))
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["pdf_download_dir"] == "/test/path"


# ============================================================================
# Tests for PDF download path validation
# ============================================================================


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


# ============================================================================
# Tests for truncate_text function
# ============================================================================


class TestTruncateText:
    """Tests for text truncation utility."""

    def test_short_text_unchanged(self):
        """Text shorter than max_len should be returned unchanged."""
        from arxiv_browser import truncate_text

        assert truncate_text("Hello", 10) == "Hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_len should be returned unchanged."""
        from arxiv_browser import truncate_text

        assert truncate_text("Hello", 5) == "Hello"

    def test_long_text_truncated(self):
        """Text longer than max_len should be truncated with suffix."""
        from arxiv_browser import truncate_text

        assert truncate_text("Hello World", 5) == "Hello..."

    def test_custom_suffix(self):
        """Custom suffix should be used when provided."""
        from arxiv_browser import truncate_text

        assert truncate_text("Hello World", 5, suffix=">>>") == "Hello>>>"

    def test_empty_string(self):
        """Empty string should be handled correctly."""
        from arxiv_browser import truncate_text

        assert truncate_text("", 10) == ""


# ============================================================================
# Tests for safe_get function and type validation
# ============================================================================


class TestSafeGetAndTypeValidation:
    """Tests for type-safe configuration parsing."""

    def test_safe_get_correct_type(self):
        """_safe_get should return value when type matches."""
        from arxiv_browser import _safe_get

        data = {"key": 42}
        assert _safe_get(data, "key", 0, int) == 42

    def test_safe_get_wrong_type(self):
        """_safe_get should return default when type doesn't match."""
        from arxiv_browser import _safe_get

        data = {"key": "not_an_int"}
        assert _safe_get(data, "key", 0, int) == 0

    def test_safe_get_missing_key(self):
        """_safe_get should return default for missing key."""
        from arxiv_browser import _safe_get

        data = {}
        assert _safe_get(data, "key", "default", str) == "default"

    def test_dict_to_config_handles_invalid_types(self):
        """_dict_to_config should handle invalid types gracefully."""
        from arxiv_browser import _dict_to_config

        invalid_data = {
            "session": {
                "scroll_index": "not_an_int",
                "current_filter": 123,
                "sort_index": None,
            },
            "paper_metadata": "not_a_dict",
            "version": "1",
        }

        config = _dict_to_config(invalid_data)

        assert config.session.scroll_index == 0
        assert config.session.current_filter == ""
        assert config.session.sort_index == 0
        assert config.paper_metadata == {}
        assert config.version == 1


# ============================================================================
# Tests for paper deduplication
# ============================================================================


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


# ============================================================================
# Tests for history file discovery
# ============================================================================


class TestHistoryFileDiscovery:
    """Tests for history file discovery functionality."""

    def test_discover_history_files_empty_dir(self, tmp_path):
        """discover_history_files should return empty list for empty history dir."""
        from arxiv_browser import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        assert discover_history_files(tmp_path) == []

    def test_discover_history_files_no_history_dir(self, tmp_path):
        """discover_history_files should return empty list when history/ doesn't exist."""
        from arxiv_browser import discover_history_files

        assert discover_history_files(tmp_path) == []

    def test_discover_history_files_respects_limit(self, tmp_path):
        """discover_history_files should respect the limit parameter."""
        from arxiv_browser import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        for i in range(10):
            (history_dir / f"2024-01-{i + 10:02d}.txt").write_text("test")

        result = discover_history_files(tmp_path, limit=5)
        assert len(result) == 5

        dates = [d for d, _ in result]
        assert dates == sorted(dates, reverse=True)

    def test_max_history_files_constant_is_positive(self):
        """MAX_HISTORY_FILES constant should be positive."""
        from arxiv_browser import MAX_HISTORY_FILES

        assert MAX_HISTORY_FILES > 0

    def test_discover_history_files_skips_invalid_names(self, tmp_path):
        """discover_history_files should skip files that don't match YYYY-MM-DD pattern."""
        from arxiv_browser import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        (history_dir / "2024-01-15.txt").write_text("valid")
        (history_dir / "invalid.txt").write_text("invalid")
        (history_dir / "2024-13-01.txt").write_text("invalid month")
        (history_dir / "notes.txt").write_text("notes")

        result = discover_history_files(tmp_path)
        assert len(result) == 1
        assert result[0][0].isoformat() == "2024-01-15"


# ============================================================================
# Tests for year extraction edge cases
# ============================================================================


class TestYearExtractionEdgeCases:
    """Tests for year extraction in BibTeX export."""

    def test_extract_year_whitespace_only(self):
        """Year extraction should handle whitespace-only date."""
        from arxiv_browser import ArxivBrowser

        paper = Paper(
            arxiv_id="test",
            date="   ",
            title="Test",
            authors="Test",
            categories="cs.AI",
            comments=None,
            abstract="Test",
            url="http://test",
        )
        app = ArxivBrowser([paper])

        result = app._extract_year("   ")
        assert len(result) == 4
        assert result.isdigit()

    def test_extract_year_empty_string(self):
        """Year extraction should handle empty string."""
        from arxiv_browser import ArxivBrowser

        paper = Paper(
            arxiv_id="test",
            date="",
            title="Test",
            authors="Test",
            categories="cs.AI",
            comments=None,
            abstract="Test",
            url="http://test",
        )
        app = ArxivBrowser([paper])

        result = app._extract_year("")
        assert len(result) == 4
        assert result.isdigit()


# ============================================================================
# Tests for BibTeX formatting edge cases
# ============================================================================


class TestBibTeXFormattingEdgeCases:
    """Tests for BibTeX formatting edge cases."""

    def test_format_bibtex_empty_categories(self):
        """BibTeX formatting should handle empty categories."""
        from arxiv_browser import ArxivBrowser

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
        app = ArxivBrowser([paper])

        bibtex = app._format_paper_as_bibtex(paper)
        assert "primaryClass = {misc}" in bibtex

    def test_format_bibtex_whitespace_categories(self):
        """BibTeX formatting should handle whitespace-only categories."""
        from arxiv_browser import ArxivBrowser

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
        app = ArxivBrowser([paper])

        bibtex = app._format_paper_as_bibtex(paper)
        assert "primaryClass = {misc}" in bibtex


# ============================================================================
# Tests for PDF download configuration
# ============================================================================


class TestPdfDownloadConfig:
    """Tests for PDF download configuration."""

    def test_pdf_download_dir_default_empty(self):
        """Default pdf_download_dir should be empty string."""
        config = UserConfig()
        assert config.pdf_download_dir == ""

    def test_pdf_download_dir_serialization_roundtrip(self):
        """pdf_download_dir should survive config serialization."""
        from arxiv_browser import _config_to_dict, _dict_to_config

        config = UserConfig(pdf_download_dir="/custom/path")
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.pdf_download_dir == "/custom/path"

    def test_get_pdf_download_path_default(self, tmp_path, monkeypatch):
        """Default path should be ~/arxiv-pdfs/{arxiv_id}.pdf."""
        from arxiv_browser import DEFAULT_PDF_DOWNLOAD_DIR

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


# ============================================================================
# Regression tests for status/filter/help UX behavior
# ============================================================================


class TestStatusFilterRegressions:
    """Regression tests for status bar and filter query handling.

    The three fragile tests that relied on _Static__content and monkey-patched
    query_one have been migrated to TestStatusFilterIntegration (Phase 7).
    Only the source-inspection test remains here.
    """

    def test_help_screen_mentions_actual_history_keys(self):
        """Help text should reference the actual history key names."""
        import inspect

        from arxiv_browser import HelpScreen

        source = inspect.getsource(HelpScreen.compose)
        assert "bracketleft / bracketright" in source


# ============================================================================
# Tests for __all__ exports
# ============================================================================


class TestModuleExports:
    """Tests for module public API."""

    def test_all_exports_are_importable(self):
        """All items in __all__ should be importable."""
        from arxiv_browser import __all__
        import arxiv_browser

        for name in __all__:
            assert hasattr(arxiv_browser, name), f"{name} not found in module"

    def test_main_exports_exist(self):
        """Key exports should be available."""
        from arxiv_browser import (
            SessionState,
            ArxivBrowser,
            main,
        )

        assert Paper is not None
        assert ArxivBrowser is not None


# ============================================================================
# Tests for extracted pure functions
# ============================================================================


class TestHighlightText:
    """Tests for highlight_text()."""

    def test_empty_text_returns_empty(self):
        from arxiv_browser import highlight_text

        assert highlight_text("", ["foo"], "#ff0000") == ""

    def test_empty_terms_returns_escaped(self):
        from arxiv_browser import highlight_text

        # Rich's escape only escapes recognized markup-like brackets
        result = highlight_text("Hello [bold]text[/bold]", [], "#ff0000")
        assert r"\[bold]" in result

    def test_short_terms_filtered(self):
        from arxiv_browser import highlight_text

        result = highlight_text("a b c", ["a"], "#ff0000")
        assert "[bold" not in result  # "a" is too short (< 2 chars)

    def test_dedup_terms(self):
        from arxiv_browser import highlight_text

        result = highlight_text("hello world", ["hello", "HELLO"], "#ff0000")
        assert result.count("[bold") == 1  # Deduped

    def test_case_insensitive_highlight(self):
        from arxiv_browser import highlight_text

        result = highlight_text("Deep Learning", ["deep"], "#ff0000")
        assert "[bold #ff0000]Deep[/]" in result

    def test_rich_escaping_preserved(self):
        from arxiv_browser import highlight_text

        result = highlight_text("[bold]text[/bold]", ["text"], "#ff0000")
        assert r"\[bold]" in result


class TestEscapeRichText:
    """Tests for escape_rich_text()."""

    def test_empty_string(self):
        from arxiv_browser import escape_rich_text

        assert escape_rich_text("") == ""

    def test_normal_text(self):
        from arxiv_browser import escape_rich_text

        assert escape_rich_text("Hello World") == "Hello World"

    def test_brackets_escaped(self):
        from arxiv_browser import escape_rich_text

        assert escape_rich_text("[bold]text[/bold]") == r"\[bold]text\[/bold]"


class TestFormatAuthorsBibtex:
    """Tests for format_authors_bibtex()."""

    def test_single_author(self):
        from arxiv_browser import format_authors_bibtex

        assert format_authors_bibtex("John Smith") == "John Smith"

    def test_special_chars_escaped(self):
        from arxiv_browser import format_authors_bibtex

        assert format_authors_bibtex("A & B") == r"A \& B"


class TestGetConfigPath:
    """Tests for get_config_path()."""

    def test_returns_path_with_config_json(self):
        from arxiv_browser import get_config_path

        path = get_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.json"


class TestComputePaperSimilarity:
    """Tests for compute_paper_similarity()."""

    def test_identity_similarity(self, make_paper):
        from arxiv_browser import compute_paper_similarity

        paper = make_paper()
        assert compute_paper_similarity(paper, paper) == 1.0

    def test_different_papers_less_than_one(self, make_paper):
        from arxiv_browser import compute_paper_similarity

        p1 = make_paper(arxiv_id="001", categories="cs.AI", authors="Smith")
        p2 = make_paper(arxiv_id="002", categories="quant-ph", authors="Jones")
        assert compute_paper_similarity(p1, p2) < 1.0

    def test_category_weight_dominates(self, make_paper):
        from arxiv_browser import compute_paper_similarity

        # Same categories, different authors
        p1 = make_paper(arxiv_id="001", categories="cs.AI cs.LG", authors="Smith")
        p2 = make_paper(arxiv_id="002", categories="cs.AI cs.LG", authors="Jones")
        # Different categories, same authors
        p3 = make_paper(arxiv_id="003", categories="quant-ph", authors="Smith")

        sim_same_cat = compute_paper_similarity(p1, p2)
        sim_diff_cat = compute_paper_similarity(p1, p3)
        assert sim_same_cat > sim_diff_cat


class TestIsAdvancedQuery:
    """Tests for is_advanced_query()."""

    def test_plain_terms_not_advanced(self):
        from arxiv_browser import is_advanced_query

        tokens = [QueryToken(kind="term", value="attention")]
        assert is_advanced_query(tokens) is False

    def test_operator_is_advanced(self):
        from arxiv_browser import is_advanced_query

        tokens = [
            QueryToken(kind="term", value="a"),
            QueryToken(kind="op", value="AND"),
            QueryToken(kind="term", value="b"),
        ]
        assert is_advanced_query(tokens) is True

    def test_field_prefix_is_advanced(self):
        from arxiv_browser import is_advanced_query

        tokens = [QueryToken(kind="term", value="cs.AI", field="cat")]
        assert is_advanced_query(tokens) is True

    def test_quoted_phrase_is_advanced(self):
        from arxiv_browser import is_advanced_query

        tokens = [QueryToken(kind="term", value="deep learning", phrase=True)]
        assert is_advanced_query(tokens) is True

    def test_unread_virtual_term_is_advanced(self):
        from arxiv_browser import is_advanced_query

        tokens = [QueryToken(kind="term", value="unread")]
        assert is_advanced_query(tokens) is True

    def test_starred_virtual_term_is_advanced(self):
        from arxiv_browser import is_advanced_query

        tokens = [QueryToken(kind="term", value="starred")]
        assert is_advanced_query(tokens) is True


class TestMatchQueryTerm:
    """Tests for match_query_term()."""

    def test_empty_value_matches_all(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="   ")
        assert match_query_term(paper, token, None) is True

    def test_cat_field_matches(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper(categories="cs.AI cs.LG")
        token = QueryToken(kind="term", value="cs.AI", field="cat")
        assert match_query_term(paper, token, None) is True

    def test_cat_field_no_match(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper(categories="cs.AI")
        token = QueryToken(kind="term", value="cs.CV", field="cat")
        assert match_query_term(paper, token, None) is False

    def test_tag_field_matches(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper()
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["important", "to-read"])
        token = QueryToken(kind="term", value="important", field="tag")
        assert match_query_term(paper, token, meta) is True

    def test_tag_field_no_metadata(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="important", field="tag")
        assert match_query_term(paper, token, None) is False

    def test_title_field_matches(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper(title="Deep Learning for NLP")
        token = QueryToken(kind="term", value="Deep", field="title")
        assert match_query_term(paper, token, None) is True

    def test_author_field_matches(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper(authors="John Smith")
        token = QueryToken(kind="term", value="smith", field="author")
        assert match_query_term(paper, token, None) is True

    def test_abstract_field_matches(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="test abstract", field="abstract")
        assert match_query_term(paper, token, None, abstract_text="Test abstract content.") is True

    def test_unread_virtual_term(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="unread")
        # No metadata = unread
        assert match_query_term(paper, token, None) is True
        # Read = not unread
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, is_read=True)
        assert match_query_term(paper, token, meta) is False

    def test_starred_virtual_term(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="starred")
        # No metadata = not starred
        assert match_query_term(paper, token, None) is False
        # Starred = starred
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, starred=True)
        assert match_query_term(paper, token, meta) is True

    def test_fallback_search_title_and_authors(self, make_paper):
        from arxiv_browser import match_query_term

        paper = make_paper(title="Attention Mechanism", authors="Jane Doe")
        token = QueryToken(kind="term", value="attention")
        assert match_query_term(paper, token, None) is True


class TestMatchesAdvancedQuery:
    """Tests for matches_advanced_query()."""

    def test_empty_rpn_matches_all(self, make_paper):
        from arxiv_browser import matches_advanced_query

        paper = make_paper()
        assert matches_advanced_query(paper, [], None) is True

    def test_single_term(self, make_paper):
        from arxiv_browser import matches_advanced_query

        paper = make_paper(categories="cs.AI")
        rpn = [QueryToken(kind="term", value="cs.AI", field="cat")]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_and_query(self, make_paper):
        from arxiv_browser import matches_advanced_query

        paper = make_paper(title="Deep Learning for NLP")
        rpn = [
            QueryToken(kind="term", value="deep"),
            QueryToken(kind="term", value="nlp"),
            QueryToken(kind="op", value="AND"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_or_query(self, make_paper):
        from arxiv_browser import matches_advanced_query

        paper = make_paper(title="Deep Learning")
        rpn = [
            QueryToken(kind="term", value="quantum"),
            QueryToken(kind="term", value="deep"),
            QueryToken(kind="op", value="OR"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_not_query(self, make_paper):
        from arxiv_browser import matches_advanced_query

        paper = make_paper(title="Deep Learning")
        rpn = [
            QueryToken(kind="term", value="quantum"),
            QueryToken(kind="op", value="NOT"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True


class TestPaperMatchesWatchEntry:
    """Tests for paper_matches_watch_entry()."""

    def test_author_match(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper(authors="John Smith, Jane Doe")
        entry = WatchListEntry(pattern="Smith", match_type="author")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_author_no_match(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper(authors="John Smith")
        entry = WatchListEntry(pattern="Wilson", match_type="author")
        assert paper_matches_watch_entry(paper, entry) is False

    def test_title_match(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper(title="Deep Learning for NLP")
        entry = WatchListEntry(pattern="Deep Learning", match_type="title")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_keyword_match_in_title(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper(title="Transformer Architecture", abstract_raw="Some abstract")
        entry = WatchListEntry(pattern="transformer", match_type="keyword")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_keyword_match_in_abstract(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper(title="Some Title", abstract_raw="attention mechanism")
        entry = WatchListEntry(pattern="attention", match_type="keyword")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_case_sensitive(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper(authors="john smith")
        entry = WatchListEntry(pattern="John", match_type="author", case_sensitive=True)
        assert paper_matches_watch_entry(paper, entry) is False

    def test_unknown_match_type(self, make_paper):
        from arxiv_browser import paper_matches_watch_entry, WatchListEntry

        paper = make_paper()
        entry = WatchListEntry(pattern="test", match_type="unknown")
        assert paper_matches_watch_entry(paper, entry) is False


class TestSortPapers:
    """Tests for sort_papers()."""

    def test_sort_by_title(self, make_paper):
        from arxiv_browser import sort_papers

        papers = [
            make_paper(title="Zebra"),
            make_paper(title="Apple"),
            make_paper(title="Mango"),
        ]
        result = sort_papers(papers, "title")
        assert [p.title for p in result] == ["Apple", "Mango", "Zebra"]

    def test_sort_by_date_descending(self, make_paper):
        from arxiv_browser import sort_papers

        papers = [
            make_paper(date="Mon, 1 Jan 2024"),
            make_paper(date="Wed, 15 Jan 2024"),
            make_paper(date="Tue, 10 Jan 2024"),
        ]
        result = sort_papers(papers, "date")
        assert result[0].date == "Wed, 15 Jan 2024"
        assert result[-1].date == "Mon, 1 Jan 2024"

    def test_sort_by_arxiv_id_descending(self, make_paper):
        from arxiv_browser import sort_papers

        papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00002"),
        ]
        result = sort_papers(papers, "arxiv_id")
        assert [p.arxiv_id for p in result] == ["2401.00003", "2401.00002", "2401.00001"]

    def test_sort_does_not_mutate_original(self, make_paper):
        from arxiv_browser import sort_papers

        papers = [make_paper(title="B"), make_paper(title="A")]
        original_order = [p.title for p in papers]
        sort_papers(papers, "title")
        assert [p.title for p in papers] == original_order


class TestFormatPaperForClipboard:
    """Tests for format_paper_for_clipboard()."""

    def test_basic_format(self, make_paper):
        from arxiv_browser import format_paper_for_clipboard

        paper = make_paper(title="Test Paper", authors="Author", arxiv_id="2401.12345")
        result = format_paper_for_clipboard(paper, abstract_text="Some abstract")
        assert "Title: Test Paper" in result
        assert "Authors: Author" in result
        assert "Abstract: Some abstract" in result

    def test_includes_comments(self, make_paper):
        from arxiv_browser import format_paper_for_clipboard

        paper = make_paper(comments="10 pages, 5 figures")
        result = format_paper_for_clipboard(paper)
        assert "Comments: 10 pages, 5 figures" in result

    def test_omits_none_comments(self, make_paper):
        from arxiv_browser import format_paper_for_clipboard

        paper = make_paper(comments=None)
        result = format_paper_for_clipboard(paper)
        assert "Comments:" not in result


class TestFormatPaperAsMarkdown:
    """Tests for format_paper_as_markdown()."""

    def test_headers_and_sections(self, make_paper):
        from arxiv_browser import format_paper_as_markdown

        paper = make_paper(title="Test Paper", authors="Author")
        result = format_paper_as_markdown(paper, abstract_text="Some abstract")
        assert "## Test Paper" in result
        assert "### Abstract" in result
        assert "**Authors:** Author" in result

    def test_arxiv_link_format(self, make_paper):
        from arxiv_browser import format_paper_as_markdown

        paper = make_paper(arxiv_id="2401.12345")
        result = format_paper_as_markdown(paper)
        assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in result


class TestGetPdfUrl:
    """Tests for get_pdf_url()."""

    def test_standard_abs_url(self, make_paper):
        from arxiv_browser import get_pdf_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345", arxiv_id="2401.12345")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_already_pdf_url(self, make_paper):
        from arxiv_browser import get_pdf_url

        paper = make_paper(url="https://arxiv.org/pdf/2401.12345.pdf")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_pdf_url_without_extension(self, make_paper):
        from arxiv_browser import get_pdf_url

        paper = make_paper(url="https://arxiv.org/pdf/2401.12345")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"


class TestGetPaperUrl:
    """Tests for get_paper_url()."""

    def test_default_abs_url(self, make_paper):
        from arxiv_browser import get_paper_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345")
        assert get_paper_url(paper) == "https://arxiv.org/abs/2401.12345"

    def test_prefer_pdf(self, make_paper):
        from arxiv_browser import get_paper_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345", arxiv_id="2401.12345")
        result = get_paper_url(paper, prefer_pdf=True)
        assert "pdf" in result


class TestBuildHighlightTerms:
    """Tests for build_highlight_terms()."""

    def test_title_field(self):
        from arxiv_browser import build_highlight_terms

        tokens = [QueryToken(kind="term", value="deep", field="title")]
        result = build_highlight_terms(tokens)
        assert "deep" in result["title"]
        assert result["author"] == []

    def test_unfielded_goes_to_title_and_author(self):
        from arxiv_browser import build_highlight_terms

        tokens = [QueryToken(kind="term", value="smith")]
        result = build_highlight_terms(tokens)
        assert "smith" in result["title"]
        assert "smith" in result["author"]

    def test_operators_skipped(self):
        from arxiv_browser import build_highlight_terms

        tokens = [QueryToken(kind="op", value="AND")]
        result = build_highlight_terms(tokens)
        assert all(v == [] for v in result.values())

    def test_virtual_terms_skipped(self):
        from arxiv_browser import build_highlight_terms

        tokens = [QueryToken(kind="term", value="unread")]
        result = build_highlight_terms(tokens)
        assert all(v == [] for v in result.values())


# ============================================================================
# Tests for CLI main() function
# ============================================================================


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    def test_list_dates_with_files(self, tmp_path, monkeypatch, capsys):
        """--list-dates should print dates and return 0."""
        from datetime import date as datemod
        from arxiv_browser import main

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        (history_dir / "2024-01-15.txt").write_text("test content")

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--list-dates"])
        monkeypatch.setattr(
            "arxiv_browser.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), history_dir / "2024-01-15.txt")],
        )
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "2024-01-15" in captured.out

    def test_list_dates_empty_history(self, monkeypatch, capsys):
        """--list-dates with no files should return 1."""
        from arxiv_browser import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--list-dates"])
        monkeypatch.setattr("arxiv_browser.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "No history files" in captured.err

    def test_input_file_not_found(self, tmp_path, monkeypatch, capsys):
        """-i nonexistent.txt should return 1."""
        from arxiv_browser import main

        nonexistent = str(tmp_path / "nonexistent.txt")
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", nonexistent])
        monkeypatch.setattr("arxiv_browser.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "not found" in captured.err

    def test_input_file_is_directory(self, tmp_path, monkeypatch, capsys):
        """-i /some/dir should return 1."""
        from arxiv_browser import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", str(tmp_path)])
        monkeypatch.setattr("arxiv_browser.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "directory" in captured.err

    def test_no_papers_exits_with_error(self, tmp_path, monkeypatch, capsys):
        """Empty file should return 1 with 'No papers'."""
        from arxiv_browser import main

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", str(empty_file)])
        monkeypatch.setattr("arxiv_browser.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "No papers" in captured.err

    def test_invalid_date_format(self, monkeypatch, capsys):
        """--date Jan-15-2024 should return 1 with 'Invalid date'."""
        from datetime import date as datemod
        from arxiv_browser import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "Jan-15-2024"])
        monkeypatch.setattr(
            "arxiv_browser.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), Path("/fake/2024-01-15.txt"))],
        )
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Invalid date" in captured.err

    def test_date_not_found(self, monkeypatch, capsys):
        """--date 2099-01-01 should return 1 with 'No file found'."""
        from datetime import date as datemod
        from arxiv_browser import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "2099-01-01"])
        monkeypatch.setattr(
            "arxiv_browser.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), Path("/fake/2024-01-15.txt"))],
        )
        monkeypatch.setattr("arxiv_browser.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "No file found" in captured.err


# ============================================================================
# Phase 6: Textual Integration Tests
# ============================================================================


@pytest.mark.integration
class TestTextualIntegration:
    """Integration tests using Textual's run_test() / Pilot API.

    These tests run the full ArxivBrowser app in headless mode and simulate
    keyboard interactions to verify end-to-end behavior.
    """

    @staticmethod
    def _make_papers(make_paper, count: int = 3) -> list:
        """Create a list of distinct test papers."""
        papers = []
        for i in range(count):
            papers.append(
                make_paper(
                    arxiv_id=f"2401.{10000 + i}",
                    title=f"Paper Title {chr(65 + i)}",
                    authors=f"Author {chr(65 + i)}",
                    categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                    abstract=f"Abstract content for paper {chr(65 + i)}.",
                )
            )
        return papers

    async def test_app_renders_paper_list(self, make_paper):
        """App should mount and render all papers in the list view."""
        from unittest.mock import patch

        from textual.widgets import ListView

        from arxiv_browser import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                list_view = app.query_one("#paper-list", ListView)
                assert len(list_view.children) == 5

    async def test_search_filters_papers(self, make_paper):
        """Typing in search should filter the paper list after debounce."""
        from unittest.mock import patch

        from textual.widgets import ListView

        from arxiv_browser import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        # Make one paper with a unique title for easy filtering
        papers[0] = make_paper(
            arxiv_id="2401.99999",
            title="Quantum Computing Breakthrough",
            authors="Alice Wonderland",
            categories="quant-ph",
            abstract="Quantum abstract.",
        )
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search
                await pilot.press("slash")
                # Type a query that matches only the quantum paper
                await pilot.press("Q", "u", "a", "n", "t", "u", "m")
                # Wait for debounce (0.3s) + margin for DOM update
                await pilot.pause(0.7)
                list_view = app.query_one("#paper-list", ListView)
                assert len(list_view.children) == 1

    async def test_search_clear_restores_all(self, make_paper):
        """Pressing escape on search should restore all papers."""
        from unittest.mock import patch

        from textual.widgets import ListView

        from arxiv_browser import ArxivBrowser, PaperListItem

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search and type an advanced query that matches nothing
                # Use "cat:" prefix to trigger exact (non-fuzzy) matching
                await pilot.press("slash")
                for ch in "cat:nonexistent":
                    await pilot.press(ch)
                await pilot.pause(0.7)
                list_view = app.query_one("#paper-list", ListView)
                # Empty state shows a placeholder ListItem, not PaperListItems
                paper_items = [
                    c for c in list_view.children if isinstance(c, PaperListItem)
                ]
                assert len(paper_items) == 0

                # Cancel search with escape
                await pilot.press("escape")
                await pilot.pause(0.4)
                paper_items = [
                    c for c in list_view.children if isinstance(c, PaperListItem)
                ]
                assert len(paper_items) == 3

    async def test_sort_cycling(self, make_paper):
        """Pressing 's' should cycle through sort options."""
        from unittest.mock import patch

        from arxiv_browser import ArxivBrowser, SORT_OPTIONS

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert app._sort_index == 0
                await pilot.press("s")
                assert app._sort_index == 1
                await pilot.press("s")
                assert app._sort_index == 2
                # Should cycle back to 0
                await pilot.press("s")
                assert app._sort_index == 0

    async def test_toggle_read_status(self, make_paper):
        """Pressing 'r' should toggle read status of the current paper."""
        from unittest.mock import patch

        from arxiv_browser import ArxivBrowser, PaperListItem

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # First paper should be highlighted by default
                await pilot.press("r")
                # Check that metadata was created and is_read is True
                first_id = papers[0].arxiv_id
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].is_read is True

                # Toggle again
                await pilot.press("r")
                assert app._config.paper_metadata[first_id].is_read is False

    async def test_toggle_star(self, make_paper):
        """Pressing 'x' should toggle star status of the current paper."""
        from unittest.mock import patch

        from arxiv_browser import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("x")
                first_id = papers[0].arxiv_id
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].starred is True

                await pilot.press("x")
                assert app._config.paper_metadata[first_id].starred is False

    async def test_help_screen_opens_and_closes(self, make_paper):
        """Pressing '?' should open help, 'escape' should close it."""
        from unittest.mock import patch

        from arxiv_browser import ArxivBrowser, HelpScreen

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Initially no help screen
                assert len(app.screen_stack) == 1

                # Open help
                await pilot.press("question_mark")
                await pilot.pause(0.1)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], HelpScreen)

                # Close help
                await pilot.press("escape")
                await pilot.pause(0.1)
                assert len(app.screen_stack) == 1

    async def test_vim_navigation(self, make_paper):
        """Pressing 'j' moves cursor down, 'k' moves cursor up."""
        from unittest.mock import patch

        from textual.widgets import ListView

        from arxiv_browser import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                list_view = app.query_one("#paper-list", ListView)
                # Should start at index 0
                assert list_view.index == 0

                # Move down
                await pilot.press("j")
                assert list_view.index == 1
                await pilot.press("j")
                assert list_view.index == 2

                # Move back up
                await pilot.press("k")
                assert list_view.index == 1


# ============================================================================
# Phase 7: Migrate Fragile Regressions to Integration Tests
# ============================================================================


@pytest.mark.integration
class TestStatusFilterIntegration:
    """Migrated regression tests using Textual Pilot instead of fragile internals.

    Replaces the tests from TestStatusFilterRegressions that used private
    attributes (_Static__content) and monkey-patched query_one.
    """

    @staticmethod
    def _make_app(make_paper):
        from arxiv_browser import ArxivBrowser

        papers = [
            make_paper(
                arxiv_id="2401.12345",
                title="Attention Is All You Need",
                authors="Jane Smith",
                categories="cs.AI",
                abstract="Test abstract content.",
            )
        ]
        return ArxivBrowser(papers, restore_session=False)

    async def test_status_bar_escapes_markup_in_query(self, make_paper):
        """Status bar should not crash when query contains Rich markup tokens."""
        from unittest.mock import patch

        from textual.widgets import Input, Label

        app = self._make_app(make_paper)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search and inject Rich markup directly into input
                # (bracket keys are intercepted by app bindings, so set value directly)
                await pilot.press("slash")
                search_input = app.query_one("#search-input", Input)
                search_input.value = "[/]"
                await pilot.pause(0.7)
                # Verify the app didn't crash — status bar has content
                status_bar = app.query_one("#status-bar", Label)
                assert status_bar.content != ""
                # Verify the app is still responsive
                await pilot.press("escape")

    async def test_apply_filter_syncs_pending_query(self, make_paper):
        """Applying a filter should sync _pending_query and update UI."""
        from unittest.mock import patch

        from textual.widgets import Label, ListView

        app = self._make_app(make_paper)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search and type a query that won't match our paper
                await pilot.press("slash")
                await pilot.press(
                    "t", "r", "a", "n", "s", "f", "o", "r", "m", "e", "r"
                )
                await pilot.pause(0.7)
                # Verify internal state synced
                assert app._pending_query == "transformer"
                # Verify UI reflects the filter — header shows filtered/total count
                header = app.query_one("#list-header", Label)
                header_text = str(header.content)
                # When filtered, header shows "(filtered/total)" instead of "(N total)"
                assert "/1)" in header_text  # e.g. "(0/1)" or "(1/1)"

    async def test_stale_query_does_not_persist(self, make_paper):
        """Cancelling search should clear the query state and restore UI."""
        from unittest.mock import patch

        from textual.widgets import Input, ListView

        from arxiv_browser import PaperListItem

        app = self._make_app(make_paper)
        with patch("arxiv_browser.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Search for something
                await pilot.press("slash")
                await pilot.press("t", "e", "s", "t")
                await pilot.pause(0.7)
                assert app._pending_query == "test"

                # Cancel search — should clear query and restore all papers
                await pilot.press("escape")
                await pilot.pause(0.4)
                assert app._pending_query == ""
                # Verify search input was cleared
                search_input = app.query_one("#search-input", Input)
                assert search_input.value == ""
                # Verify all papers are restored in the list
                list_view = app.query_one("#paper-list", ListView)
                paper_items = [
                    c for c in list_view.children if isinstance(c, PaperListItem)
                ]
                assert len(paper_items) == 1  # App was created with 1 paper


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
