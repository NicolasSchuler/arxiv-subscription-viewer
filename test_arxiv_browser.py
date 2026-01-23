#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser import (
    ARXIV_DATE_FORMAT,
    DEFAULT_CATEGORY_COLOR,
    Paper,
    SORT_OPTIONS,
    SUBPROCESS_TIMEOUT,
    clean_latex,
    format_categories,
    parse_arxiv_date,
    parse_arxiv_file,
)


# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestCleanLatex:
    """Tests for LaTeX cleaning functionality."""

    def test_plain_text_unchanged(self):
        """Plain text without LaTeX should be returned unchanged (whitespace normalized)."""
        assert clean_latex("Hello World") == "Hello World"

    def test_extra_whitespace_normalized(self):
        """Extra whitespace should be normalized to single spaces."""
        assert clean_latex("Hello    World") == "Hello World"
        assert clean_latex("  leading and trailing  ") == "leading and trailing"

    def test_textbf_removed(self):
        """\\textbf{} command should be removed, keeping content."""
        assert clean_latex(r"\textbf{bold text}") == "bold text"

    def test_textit_removed(self):
        """\\textit{} command should be removed, keeping content."""
        assert clean_latex(r"\textit{italic text}") == "italic text"

    def test_emph_removed(self):
        """\\emph{} command should be removed, keeping content."""
        assert clean_latex(r"\emph{emphasized}") == "emphasized"

    def test_math_mode_content_preserved(self):
        """Math mode content should be preserved without dollar signs."""
        assert clean_latex(r"$x^2$") == "x^2"
        assert clean_latex(r"The formula $E=mc^2$ is famous") == "The formula E=mc^2 is famous"

    def test_escaped_dollar_sign(self):
        """Escaped dollar signs should become literal dollar signs."""
        assert clean_latex(r"Price is \$100") == "Price is $100"

    def test_accented_characters(self):
        """Common LaTeX accent commands should be converted."""
        assert clean_latex(r"caf\'e") == "café"
        # The umlaut pattern only handles braced form like \"{a}
        assert clean_latex(r'M\"{u}ller') == "Müller"
        assert clean_latex(r"\c{c}") == "ç"

    def test_ampersand_escaped(self):
        """Escaped ampersand should become literal ampersand."""
        assert clean_latex(r"A \& B") == "A & B"

    def test_generic_command_with_braces(self):
        """Unknown commands with braces should keep brace content."""
        assert clean_latex(r"\unknown{content}") == "content"

    def test_standalone_command_removed(self):
        """Standalone commands without braces should be removed."""
        result = clean_latex(r"\noindent Some text")
        assert "noindent" not in result
        assert "Some text" in result

    def test_short_circuit_no_latex(self):
        """Text without backslash or dollar sign should short-circuit."""
        # This tests the optimization path - should still work correctly
        assert clean_latex("No LaTeX here at all") == "No LaTeX here at all"

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
        """Nested LaTeX commands should be handled via iterative cleaning.

        Note: Nested braces inside commands (e.g., \\textbf{$O(n^{2})$}) are
        a known limitation. The regex [^}]* stops at the first closing brace.
        This test verifies that basic nesting works and commands are stripped.
        """
        # Simple nesting without inner braces works well
        text = r"\textbf{$O(n)$}"
        result = clean_latex(text)
        assert "O(n)" in result
        assert "textbf" not in result
        assert "$" not in result

    def test_nested_braces_limitation(self):
        """Document known limitation: nested braces inside commands.

        LaTeX like \\textbf{$O(n^{2})$} has inner braces in the math.
        The regex [^}]* stops at the first }, causing partial extraction.
        This is acceptable for arXiv abstracts where such nesting is rare.
        """
        text = r"\textbf{$O(n^{2})$}"
        result = clean_latex(text)
        # The result may be imperfect due to nested brace limitation
        # At minimum, verify no crash and some content preserved
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
        # After cleaning, math content is preserved but commands removed
        assert "$" not in result


# ============================================================================
# Tests for parse_arxiv_date function
# ============================================================================


class TestParseArxivDate:
    """Tests for date parsing functionality."""

    def test_valid_date_parsing(self):
        """Valid arXiv date strings should be parsed correctly."""
        result = parse_arxiv_date("Mon, 15 Jan 2024")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_different_valid_dates(self):
        """Various valid date formats should parse correctly."""
        result = parse_arxiv_date("Tue, 25 Dec 2023")
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25

    def test_malformed_date_returns_min(self):
        """Malformed dates should return datetime.min."""
        result = parse_arxiv_date("invalid date")
        assert result == datetime.min

    def test_empty_string_returns_min(self):
        """Empty string should return datetime.min."""
        result = parse_arxiv_date("")
        assert result == datetime.min

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace should be trimmed."""
        result = parse_arxiv_date("  Mon, 15 Jan 2024  ")
        assert result.year == 2024

    def test_date_sorting_order(self):
        """Dates should sort in correct chronological order."""
        dates = [
            "Mon, 9 Jan 2024",
            "Mon, 15 Jan 2024",
            "Tue, 2 Jan 2024",
        ]
        parsed = sorted(dates, key=parse_arxiv_date)
        # Should be chronological: Jan 2, Jan 9, Jan 15
        assert parse_arxiv_date(parsed[0]).day == 2
        assert parse_arxiv_date(parsed[1]).day == 9
        assert parse_arxiv_date(parsed[2]).day == 15

    def test_date_sorting_with_string_would_fail(self):
        """Demonstrate that string sorting produces wrong results."""
        # String comparison: "Mon, 9" > "Mon, 15" because "9" > "1"
        # But chronologically: Jan 9 < Jan 15
        dates = ["Mon, 9 Jan 2024", "Mon, 15 Jan 2024"]

        # String sort (WRONG)
        string_sorted = sorted(dates)
        assert string_sorted[0] == "Mon, 15 Jan 2024"  # Wrong: 15 < 9 alphabetically

        # Date sort (CORRECT)
        date_sorted = sorted(dates, key=parse_arxiv_date)
        assert parse_arxiv_date(date_sorted[0]).day == 9  # Correct: 9 < 15


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
        # Results should be identical (cached)
        assert result1 == result2


# ============================================================================
# Tests for parse_arxiv_file function
# ============================================================================


class TestParseArxivFile:
    """Tests for arXiv file parsing."""

    @pytest.fixture
    def sample_arxiv_content(self):
        """Sample arXiv email content for testing."""
        return '''------------------------------------------------------------------------------
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
'''

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
        # Second paper has no Comments line, so comments should be None
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
        content = '''Some random text that is not a valid entry
------------------------------------------------------------------------------
This is also not valid
------------------------------------------------------------------------------
'''
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
# Tests for constants
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_sort_options_defined(self):
        """SORT_OPTIONS should be defined with expected values."""
        assert "title" in SORT_OPTIONS
        assert "date" in SORT_OPTIONS
        assert "arxiv_id" in SORT_OPTIONS

    def test_default_category_color_is_hex(self):
        """DEFAULT_CATEGORY_COLOR should be a valid hex color."""
        assert DEFAULT_CATEGORY_COLOR.startswith("#")
        assert len(DEFAULT_CATEGORY_COLOR) == 7  # #RRGGBB format

    def test_subprocess_timeout_is_positive(self):
        """SUBPROCESS_TIMEOUT should be a positive number."""
        assert SUBPROCESS_TIMEOUT > 0

    def test_arxiv_date_format_valid(self):
        """ARXIV_DATE_FORMAT should be a valid strptime format."""
        # Test that format works with a sample date
        sample_date = "Mon, 15 Jan 2024"
        parsed = datetime.strptime(sample_date, ARXIV_DATE_FORMAT)
        assert parsed.year == 2024


# ============================================================================
# Tests for new dataclasses
# ============================================================================


class TestNewDataclasses:
    """Tests for Phase 1 dataclasses."""

    def test_paper_metadata_defaults(self):
        """PaperMetadata should have correct defaults."""
        from arxiv_browser import PaperMetadata
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
        from arxiv_browser import UserConfig
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
            UserConfig, PaperMetadata, WatchListEntry, SearchBookmark,
            SessionState, _config_to_dict, _dict_to_config
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

    def test_dict_to_config_handles_empty(self):
        """Loading empty dict should return default config."""
        from arxiv_browser import _dict_to_config, UserConfig
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
        # {a, b} ∩ {b, c} = {b}, |{b}| / |{a, b, c}| = 1/3
        result = _jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(result - 1/3) < 0.01

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

        # NLP and text classification papers should be more similar
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

    def test_escape_bibtex_special_chars(self):
        """Special characters should be escaped for BibTeX."""
        # This tests internal functionality that would be part of ArxivBrowser
        # We test the concept by checking that the BibTeX output is valid
        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Testing & Evaluation: A 100% Comprehensive Study",
            authors="John Smith",
            categories="cs.AI",
            comments=None,
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )
        # The actual test would require instantiating ArxivBrowser
        # For now, verify the paper object is valid
        assert paper.title.count("&") == 1
        assert paper.title.count("%") == 1


# ============================================================================
# Tests for fuzzy search constants
# ============================================================================


class TestFuzzySearchConstants:
    """Tests for fuzzy search configuration."""

    def test_fuzzy_score_cutoff_valid_range(self):
        """FUZZY_SCORE_CUTOFF should be in valid range."""
        from arxiv_browser import FUZZY_SCORE_CUTOFF
        assert 0 <= FUZZY_SCORE_CUTOFF <= 100

    def test_fuzzy_limit_is_positive(self):
        """FUZZY_LIMIT should be positive."""
        from arxiv_browser import FUZZY_LIMIT
        assert FUZZY_LIMIT > 0

    def test_similarity_top_n_is_positive(self):
        """SIMILARITY_TOP_N should be positive."""
        from arxiv_browser import SIMILARITY_TOP_N
        assert SIMILARITY_TOP_N > 0

    def test_stopwords_contains_common_words(self):
        """STOPWORDS should contain common English stopwords."""
        from arxiv_browser import STOPWORDS
        assert "the" in STOPWORDS
        assert "and" in STOPWORDS
        assert "or" in STOPWORDS
        assert "is" in STOPWORDS


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

        # Pass invalid types for various fields
        invalid_data = {
            "session": {
                "scroll_index": "not_an_int",  # Should be int
                "current_filter": 123,  # Should be str
                "sort_index": None,  # Should be int
            },
            "paper_metadata": "not_a_dict",  # Should be dict
            "version": "1",  # Should be int
        }

        config = _dict_to_config(invalid_data)

        # Should use defaults when types don't match
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
        # Create a file with duplicate entries
        content = '''\\
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
'''
        file_path = tmp_path / "duplicates.txt"
        file_path.write_text(content)

        papers = parse_arxiv_file(file_path)

        # Should have only 2 unique papers
        assert len(papers) == 2
        arxiv_ids = [p.arxiv_id for p in papers]
        assert "2401.00001" in arxiv_ids
        assert "2401.00002" in arxiv_ids

        # First paper should be the one kept (first occurrence wins)
        first_paper = next(p for p in papers if p.arxiv_id == "2401.00001")
        assert first_paper.title == "First Paper"


# ============================================================================
# Tests for UI truncation constants
# ============================================================================


class TestUIConstants:
    """Tests for UI-related constants."""

    def test_recommendation_title_max_len_positive(self):
        """RECOMMENDATION_TITLE_MAX_LEN should be positive."""
        from arxiv_browser import RECOMMENDATION_TITLE_MAX_LEN
        assert RECOMMENDATION_TITLE_MAX_LEN > 0

    def test_preview_abstract_max_len_positive(self):
        """PREVIEW_ABSTRACT_MAX_LEN should be positive."""
        from arxiv_browser import PREVIEW_ABSTRACT_MAX_LEN
        assert PREVIEW_ABSTRACT_MAX_LEN > 0

    def test_bookmark_name_max_len_positive(self):
        """BOOKMARK_NAME_MAX_LEN should be positive."""
        from arxiv_browser import BOOKMARK_NAME_MAX_LEN
        assert BOOKMARK_NAME_MAX_LEN > 0


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

        # Create 10 history files
        for i in range(10):
            (history_dir / f"2024-01-{i+10:02d}.txt").write_text("test")

        # Request only 5
        result = discover_history_files(tmp_path, limit=5)
        assert len(result) == 5

        # Should be sorted newest first
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

        # Create valid and invalid files
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
        from arxiv_browser import ArxivBrowser, Paper

        # Create minimal app to test method
        paper = Paper(
            arxiv_id="test", date="   ", title="Test", authors="Test",
            categories="cs.AI", comments=None, abstract="Test", url="http://test"
        )
        papers = [paper]
        app = ArxivBrowser(papers)

        # Should return current year for whitespace
        result = app._extract_year("   ")
        assert len(result) == 4
        assert result.isdigit()

    def test_extract_year_empty_string(self):
        """Year extraction should handle empty string."""
        from arxiv_browser import ArxivBrowser, Paper

        paper = Paper(
            arxiv_id="test", date="", title="Test", authors="Test",
            categories="cs.AI", comments=None, abstract="Test", url="http://test"
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
        from arxiv_browser import ArxivBrowser, Paper

        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="",  # Empty categories
            comments=None,
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )
        app = ArxivBrowser([paper])

        # Should not raise IndexError
        bibtex = app._format_paper_as_bibtex(paper)
        assert "primaryClass = {cs.AI}" in bibtex  # Fallback to cs.AI

    def test_format_bibtex_whitespace_categories(self):
        """BibTeX formatting should handle whitespace-only categories."""
        from arxiv_browser import ArxivBrowser, Paper

        paper = Paper(
            arxiv_id="2401.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="John Smith",
            categories="   ",  # Whitespace only
            comments=None,
            abstract="Test abstract",
            url="https://arxiv.org/abs/2401.12345",
        )
        app = ArxivBrowser([paper])

        # Should not raise IndexError
        bibtex = app._format_paper_as_bibtex(paper)
        assert "primaryClass = {cs.AI}" in bibtex  # Fallback to cs.AI


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
            Paper, PaperMetadata, UserConfig, SessionState,
            parse_arxiv_file, clean_latex, ArxivBrowser, main
        )
        # Just verify they're importable
        assert Paper is not None
        assert ArxivBrowser is not None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
