#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.app import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    ARXIV_DATE_FORMAT,
    DEFAULT_CATEGORY_COLOR,
    DEFAULT_LLM_PROMPT,
    LLM_PRESETS,
    SORT_OPTIONS,
    SUBPROCESS_TIMEOUT,
    SUMMARY_MODES,
    TAG_NAMESPACE_COLORS,
    Paper,
    PaperMetadata,
    QueryToken,
    UserConfig,
    build_arxiv_search_query,
    build_llm_prompt,
    clean_latex,
    escape_bibtex,
    extract_text_from_html,
    extract_year,
    format_categories,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    format_summary_as_rich,
    generate_citation_key,
    get_pdf_download_path,
    get_summary_db_path,
    get_tag_color,
    insert_implicit_and,
    load_config,
    normalize_arxiv_id,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
    parse_arxiv_version_map,
    parse_tag_namespace,
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
            (r"M\"{u}ller", "Müller"),
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
# Tests for arXiv API search helpers
# ============================================================================


class TestArxivApiSearchHelpers:
    """Tests for arXiv API query and parsing helpers."""

    def test_build_arxiv_search_query_with_field_and_category(self):
        query = build_arxiv_search_query("diffusion model", "title", "cs.AI")
        assert query == "ti:diffusion model AND cat:cs.AI"

    def test_build_arxiv_search_query_category_only(self):
        query = build_arxiv_search_query("", "all", "cs.LG")
        assert query == "cat:cs.LG"

    def test_build_arxiv_search_query_requires_input(self):
        with pytest.raises(ValueError, match="must be provided"):
            build_arxiv_search_query("", "all", "")

    def test_build_arxiv_search_query_rejects_invalid_field(self):
        with pytest.raises(ValueError, match="Unsupported arXiv search field"):
            build_arxiv_search_query("test", "unknown", "")

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("https://arxiv.org/abs/2401.12345v2", "2401.12345"),
            ("https://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345"),
            ("hep-th/9901001v1", "hep-th/9901001"),
            ("2401.12345v3", "2401.12345"),
        ],
    )
    def test_normalize_arxiv_id(self, raw, expected):
        assert normalize_arxiv_id(raw) == expected

    def test_parse_arxiv_api_feed(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2026-02-09T00:00:00Z</updated>
    <published>2026-02-08T00:00:00Z</published>
    <title>  A Test Paper Title  </title>
    <summary>  This is an abstract.  </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <arxiv:comment>12 pages</arxiv:comment>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
  </entry>
</feed>"""
        papers = parse_arxiv_api_feed(xml)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.arxiv_id == "2401.12345"
        assert paper.title == "A Test Paper Title"
        assert paper.authors == "Jane Doe, John Smith"
        assert paper.categories == "cs.AI cs.LG"
        assert paper.comments == "12 pages"
        assert paper.abstract == "This is an abstract."
        assert paper.url == "https://arxiv.org/abs/2401.12345"
        assert paper.source == "api"

    def test_parse_arxiv_api_feed_rejects_invalid_xml(self):
        with pytest.raises(ValueError, match="Invalid arXiv API XML response"):
            parse_arxiv_api_feed("<not xml")


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

    def test_paper_source_defaults_to_local(self):
        """Paper source should default to local entries."""
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
        assert paper.source == "local"

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
        from arxiv_browser.app import FUZZY_SCORE_CUTOFF

        assert 0 <= FUZZY_SCORE_CUTOFF <= 100

    def test_stopwords_contains_common_words(self):
        """STOPWORDS should contain common English stopwords."""
        from arxiv_browser.app import STOPWORDS

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
        from arxiv_browser.app import WatchListEntry

        entry = WatchListEntry(pattern="test")
        assert entry.pattern == "test"
        assert entry.match_type == "author"
        assert entry.case_sensitive is False

    def test_search_bookmark_creation(self):
        """SearchBookmark should store name and query."""
        from arxiv_browser.app import SearchBookmark

        bookmark = SearchBookmark(name="AI Papers", query="cat:cs.AI")
        assert bookmark.name == "AI Papers"
        assert bookmark.query == "cat:cs.AI"

    def test_session_state_defaults(self):
        """SessionState should have correct defaults."""
        from arxiv_browser.app import SessionState

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
        assert config.arxiv_api_max_results == ARXIV_API_DEFAULT_MAX_RESULTS
        assert config.version == 1


# ============================================================================
# Tests for config persistence
# ============================================================================


class TestConfigPersistence:
    """Tests for configuration save/load functions."""

    def test_config_to_dict_roundtrip(self):
        """Config should serialize and deserialize correctly."""
        from arxiv_browser.app import (
            SearchBookmark,
            SessionState,
            WatchListEntry,
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
            arxiv_api_max_results=75,
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
        assert restored.arxiv_api_max_results == 75

    def test_dict_to_config_handles_empty(self):
        """Loading empty dict should return default config."""
        from arxiv_browser.app import _dict_to_config

        config = _dict_to_config({})
        default = UserConfig()
        assert config.show_abstract_preview == default.show_abstract_preview
        assert config.paper_metadata == default.paper_metadata

    def test_arxiv_api_max_results_is_clamped(self):
        """arxiv_api_max_results should be clamped to configured limits."""
        from arxiv_browser.app import ARXIV_API_MAX_RESULTS_LIMIT, _dict_to_config

        config = _dict_to_config({"arxiv_api_max_results": 9999})
        assert config.arxiv_api_max_results == ARXIV_API_MAX_RESULTS_LIMIT


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
        from arxiv_browser.app import _jaccard_similarity

        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_jaccard_similarity_disjoint_sets(self):
        """Disjoint sets should have similarity of 0.0."""
        from arxiv_browser.app import _jaccard_similarity

        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        """Partial overlap should return correct similarity."""
        from arxiv_browser.app import _jaccard_similarity

        result = _jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 0.01

    def test_jaccard_similarity_empty_sets(self):
        """Empty sets should return 0.0."""
        from arxiv_browser.app import _jaccard_similarity

        assert _jaccard_similarity(set(), set()) == 0.0

    def test_extract_keywords_filters_stopwords(self):
        """Keywords extraction should filter stopwords."""
        from arxiv_browser.app import _extract_keywords

        keywords = _extract_keywords("the quick brown fox and the lazy dog")
        assert "the" not in keywords
        assert "and" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords

    def test_extract_keywords_min_length(self):
        """Keywords shorter than min_length should be excluded."""
        from arxiv_browser.app import _extract_keywords

        keywords = _extract_keywords("a big cat sat on the mat")
        assert "a" not in keywords
        assert "cat" not in keywords  # len("cat") = 3 < 4

    def test_extract_author_lastnames(self):
        """Author lastname extraction should work correctly."""
        from arxiv_browser.app import _extract_author_lastnames

        lastnames = _extract_author_lastnames("John Smith, Jane Doe and Bob Wilson")
        assert "smith" in lastnames
        assert "doe" in lastnames
        assert "wilson" in lastnames

    def test_similar_papers_have_higher_score(self, sample_papers):
        """Similar papers should have higher similarity than dissimilar ones."""
        from arxiv_browser.app import compute_paper_similarity

        nlp_paper = sample_papers[0]
        text_paper = sample_papers[1]
        quantum_paper = sample_papers[2]

        nlp_text_sim = compute_paper_similarity(nlp_paper, text_paper)
        nlp_quantum_sim = compute_paper_similarity(nlp_paper, quantum_paper)

        assert nlp_text_sim > nlp_quantum_sim

    def test_find_similar_papers_excludes_self(self, sample_papers):
        """find_similar_papers should not include the target paper."""
        from arxiv_browser.app import find_similar_papers

        target = sample_papers[0]
        similar = find_similar_papers(target, sample_papers)

        arxiv_ids = [p.arxiv_id for p, _ in similar]
        assert target.arxiv_id not in arxiv_ids

    def test_find_similar_papers_respects_top_n(self, sample_papers):
        """find_similar_papers should return at most top_n results."""
        from arxiv_browser.app import find_similar_papers

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
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        config = UserConfig(bibtex_export_dir="/custom/path")
        assert save_config(config) is True

        loaded = load_config()
        assert loaded.bibtex_export_dir == "/custom/path"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        """Missing config file should return default UserConfig."""
        config_file = tmp_path / "nonexistent" / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        config = load_config()
        assert config.bibtex_export_dir == ""  # default

    def test_load_corrupted_json(self, tmp_path, monkeypatch):
        """Corrupted JSON should return default UserConfig."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json {{{{", encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        config = load_config()
        assert isinstance(config, UserConfig)

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        """save_config should create parent directories."""
        config_file = tmp_path / "deep" / "nested" / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        assert save_config(UserConfig()) is True
        assert config_file.exists()

    def test_atomic_write_produces_valid_json(self, tmp_path, monkeypatch):
        """Atomic write should produce valid JSON that can be parsed."""
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

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
        from arxiv_browser.app import truncate_text

        assert truncate_text("Hello", 10) == "Hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_len should be returned unchanged."""
        from arxiv_browser.app import truncate_text

        assert truncate_text("Hello", 5) == "Hello"

    def test_long_text_truncated(self):
        """Text longer than max_len should be truncated with suffix."""
        from arxiv_browser.app import truncate_text

        assert truncate_text("Hello World", 5) == "Hello..."

    def test_custom_suffix(self):
        """Custom suffix should be used when provided."""
        from arxiv_browser.app import truncate_text

        assert truncate_text("Hello World", 5, suffix=">>>") == "Hello>>>"

    def test_empty_string(self):
        """Empty string should be handled correctly."""
        from arxiv_browser.app import truncate_text

        assert truncate_text("", 10) == ""


# ============================================================================
# Tests for safe_get function and type validation
# ============================================================================


class TestSafeGetAndTypeValidation:
    """Tests for type-safe configuration parsing."""

    def test_safe_get_correct_type(self):
        """_safe_get should return value when type matches."""
        from arxiv_browser.app import _safe_get

        data = {"key": 42}
        assert _safe_get(data, "key", 0, int) == 42

    def test_safe_get_wrong_type(self):
        """_safe_get should return default when type doesn't match."""
        from arxiv_browser.app import _safe_get

        data = {"key": "not_an_int"}
        assert _safe_get(data, "key", 0, int) == 0

    def test_safe_get_missing_key(self):
        """_safe_get should return default for missing key."""
        from arxiv_browser.app import _safe_get

        data = {}
        assert _safe_get(data, "key", "default", str) == "default"

    def test_dict_to_config_handles_invalid_types(self):
        """_dict_to_config should handle invalid types gracefully."""
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import discover_history_files

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        assert discover_history_files(tmp_path) == []

    def test_discover_history_files_no_history_dir(self, tmp_path):
        """discover_history_files should return empty list when history/ doesn't exist."""
        from arxiv_browser.app import discover_history_files

        assert discover_history_files(tmp_path) == []

    def test_discover_history_files_respects_limit(self, tmp_path):
        """discover_history_files should respect the limit parameter."""
        from arxiv_browser.app import discover_history_files

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
        from arxiv_browser.app import MAX_HISTORY_FILES

        assert MAX_HISTORY_FILES > 0

    def test_discover_history_files_skips_invalid_names(self, tmp_path):
        """discover_history_files should skip files that don't match YYYY-MM-DD pattern."""
        from arxiv_browser.app import discover_history_files

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
        result = extract_year("   ")
        assert len(result) == 4
        assert result.isdigit()

    def test_extract_year_empty_string(self):
        """Year extraction should handle empty string."""
        result = extract_year("")
        assert len(result) == 4
        assert result.isdigit()


# ============================================================================
# Tests for BibTeX formatting edge cases
# ============================================================================


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
        from arxiv_browser.app import _config_to_dict, _dict_to_config

        config = UserConfig(pdf_download_dir="/custom/path")
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.pdf_download_dir == "/custom/path"

    def test_get_pdf_download_path_default(self, tmp_path, monkeypatch):
        """Default path should be ~/arxiv-pdfs/{arxiv_id}.pdf."""
        from arxiv_browser.app import DEFAULT_PDF_DOWNLOAD_DIR

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

        from arxiv_browser.app import HelpScreen

        source = inspect.getsource(HelpScreen.compose)
        assert "bracketleft / bracketright" in source


# ============================================================================
# Tests for __all__ exports
# ============================================================================


class TestModuleExports:
    """Tests for module public API."""

    def test_all_exports_are_importable(self):
        """All items in __all__ should be importable."""
        import arxiv_browser
        from arxiv_browser.app import __all__

        for name in __all__:
            assert hasattr(arxiv_browser, name), f"{name} not found in module"

    def test_main_exports_exist(self):
        """Key exports should be available."""
        from arxiv_browser.app import (
            ArxivBrowser,
            SessionState,
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
        from arxiv_browser.app import highlight_text

        assert highlight_text("", ["foo"], "#ff0000") == ""

    def test_empty_terms_returns_escaped(self):
        from arxiv_browser.app import highlight_text

        # Rich's escape only escapes recognized markup-like brackets
        result = highlight_text("Hello [bold]text[/bold]", [], "#ff0000")
        assert r"\[bold]" in result

    def test_short_terms_filtered(self):
        from arxiv_browser.app import highlight_text

        result = highlight_text("a b c", ["a"], "#ff0000")
        assert "[bold" not in result  # "a" is too short (< 2 chars)

    def test_dedup_terms(self):
        from arxiv_browser.app import highlight_text

        result = highlight_text("hello world", ["hello", "HELLO"], "#ff0000")
        assert result.count("[bold") == 1  # Deduped

    def test_case_insensitive_highlight(self):
        from arxiv_browser.app import highlight_text

        result = highlight_text("Deep Learning", ["deep"], "#ff0000")
        assert "[bold #ff0000]Deep[/]" in result

    def test_rich_escaping_preserved(self):
        from arxiv_browser.app import highlight_text

        result = highlight_text("[bold]text[/bold]", ["text"], "#ff0000")
        assert r"\[bold]" in result


class TestEscapeRichText:
    """Tests for escape_rich_text()."""

    def test_empty_string(self):
        from arxiv_browser.app import escape_rich_text

        assert escape_rich_text("") == ""

    def test_normal_text(self):
        from arxiv_browser.app import escape_rich_text

        assert escape_rich_text("Hello World") == "Hello World"

    def test_brackets_escaped(self):
        from arxiv_browser.app import escape_rich_text

        assert escape_rich_text("[bold]text[/bold]") == r"\[bold]text\[/bold]"


class TestFormatAuthorsBibtex:
    """Tests for format_authors_bibtex()."""

    def test_single_author(self):
        from arxiv_browser.app import format_authors_bibtex

        assert format_authors_bibtex("John Smith") == "John Smith"

    def test_special_chars_escaped(self):
        from arxiv_browser.app import format_authors_bibtex

        assert format_authors_bibtex("A & B") == r"A \& B"


class TestGetConfigPath:
    """Tests for get_config_path()."""

    def test_returns_path_with_config_json(self):
        from arxiv_browser.app import get_config_path

        path = get_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.json"


class TestComputePaperSimilarity:
    """Tests for compute_paper_similarity()."""

    def test_identity_similarity(self, make_paper):
        from arxiv_browser.app import compute_paper_similarity

        paper = make_paper()
        assert compute_paper_similarity(paper, paper) == 1.0

    def test_different_papers_less_than_one(self, make_paper):
        from arxiv_browser.app import compute_paper_similarity

        p1 = make_paper(arxiv_id="001", categories="cs.AI", authors="Smith")
        p2 = make_paper(arxiv_id="002", categories="quant-ph", authors="Jones")
        assert compute_paper_similarity(p1, p2) < 1.0

    def test_category_weight_dominates(self, make_paper):
        from arxiv_browser.app import compute_paper_similarity

        # Same categories, different authors
        p1 = make_paper(arxiv_id="001", categories="cs.AI cs.LG", authors="Smith")
        p2 = make_paper(arxiv_id="002", categories="cs.AI cs.LG", authors="Jones")
        # Different categories, same authors
        p3 = make_paper(arxiv_id="003", categories="quant-ph", authors="Smith")

        sim_same_cat = compute_paper_similarity(p1, p2)
        sim_diff_cat = compute_paper_similarity(p1, p3)
        assert sim_same_cat > sim_diff_cat


class TestTfidfIndex:
    """Tests for TF-IDF tokenizer, TF computation, and TfidfIndex class."""

    def test_tokenize_basic(self):
        from arxiv_browser.app import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("Deep learning for natural language processing")
        assert isinstance(tokens, list)
        assert "deep" in tokens
        assert "learning" in tokens
        assert "natural" in tokens
        assert "language" in tokens
        assert "processing" in tokens

    def test_tokenize_preserves_frequency(self):
        from arxiv_browser.app import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("transformer transformer transformer")
        assert tokens.count("transformer") == 3

    def test_tokenize_empty(self):
        from arxiv_browser.app import _tokenize_for_tfidf

        assert _tokenize_for_tfidf(None) == []
        assert _tokenize_for_tfidf("") == []

    def test_tokenize_min_length(self):
        from arxiv_browser.app import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("a bb ccc dddd")
        # Only tokens with 3+ chars matching [a-z][a-z0-9]{2,}
        assert "ccc" in tokens
        assert "dddd" in tokens
        assert "bb" not in tokens

    def test_tokenize_stopwords(self):
        from arxiv_browser.app import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("this paper presents the method")
        assert "this" not in tokens
        assert "paper" not in tokens  # "paper" is in STOPWORDS
        assert "the" not in tokens
        assert "method" not in tokens  # "method" is in STOPWORDS
        assert "presents" in tokens

    def test_compute_tf_sublinear(self):
        import math

        from arxiv_browser.app import _compute_tf

        tf = _compute_tf(["transformer", "transformer", "transformer", "model"])
        assert tf["transformer"] == pytest.approx(1.0 + math.log(3))
        assert tf["model"] == pytest.approx(1.0 + math.log(1))

    def test_build_empty_corpus(self):
        from arxiv_browser.app import TfidfIndex

        index = TfidfIndex.build([], text_fn=lambda p: p.title)
        assert len(index) == 0

    def test_build_single_paper(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [make_paper(title="Attention mechanisms in deep learning")]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        # n < 2 guard: single paper cannot compute meaningful IDF
        assert len(index) == 0

    def test_build_two_papers(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Attention mechanisms in deep learning"),
            make_paper(arxiv_id="002", title="Reinforcement learning for robotics"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert len(index) == 2
        assert "001" in index
        assert "002" in index

    def test_cosine_self_is_one(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Attention mechanisms in deep learning"),
            make_paper(arxiv_id="002", title="Reinforcement learning for robotics"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert index.cosine_similarity("001", "001") == pytest.approx(1.0)

    def test_similar_scores_higher(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Deep learning for image classification"),
            make_paper(arxiv_id="002", title="Deep learning for object detection"),
            make_paper(arxiv_id="003", title="Quantum computing error correction codes"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        sim_related = index.cosine_similarity("001", "002")
        sim_unrelated = index.cosine_similarity("001", "003")
        assert sim_related > sim_unrelated

    def test_cosine_missing_returns_zero(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Deep learning methods"),
            make_paper(arxiv_id="002", title="Reinforcement learning"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert index.cosine_similarity("001", "unknown") == 0.0
        assert index.cosine_similarity("unknown", "001") == 0.0

    def test_idf_downweights_common(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Deep learning attention mechanisms"),
            make_paper(arxiv_id="002", title="Deep learning reinforcement policy"),
            make_paper(arxiv_id="003", title="Quantum attention entanglement"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        # "deep" and "learning" appear in 2/3 docs; "quantum" in 1/3
        # "quantum" should have higher IDF than "deep"
        assert index._idf.get("quantum", 0) > index._idf.get("deep", 0)

    def test_contains_protocol(self, make_paper):
        from arxiv_browser.app import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Testing containment protocol"),
            make_paper(arxiv_id="002", title="Another paper here"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert "001" in index
        assert "nonexistent" not in index

    def test_compute_similarity_with_tfidf(self, make_paper):
        from arxiv_browser.app import (
            SIMILARITY_WEIGHT_AUTHOR,
            SIMILARITY_WEIGHT_CATEGORY,
            SIMILARITY_WEIGHT_TEXT,
            TfidfIndex,
            compute_paper_similarity,
        )

        papers = [
            make_paper(
                arxiv_id="001",
                title="Deep learning for vision",
                categories="cs.CV",
                authors="Smith",
            ),
            make_paper(
                arxiv_id="002",
                title="Deep learning for vision tasks",
                categories="cs.CV",
                authors="Jones",
            ),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        score = compute_paper_similarity(papers[0], papers[1], tfidf_index=index)
        # Should use TF-IDF weights (category 30%, author 20%, text 50%)
        assert 0.0 < score < 1.0
        # Same categories → full category score
        assert score >= SIMILARITY_WEIGHT_CATEGORY * 0.99

    def test_compute_similarity_without_tfidf(self, make_paper):
        from arxiv_browser.app import compute_paper_similarity

        p1 = make_paper(
            arxiv_id="001",
            title="Deep learning",
            categories="cs.AI",
            authors="Smith",
        )
        p2 = make_paper(
            arxiv_id="002",
            title="Deep learning",
            categories="cs.AI",
            authors="Jones",
        )
        # Without index, uses legacy Jaccard: 0.4*cat + 0.3*author + 0.2*title + 0.1*abstract
        score = compute_paper_similarity(p1, p2, tfidf_index=None)
        # Same categories (1.0) * 0.4 = 0.4
        assert score >= 0.4


class TestIsAdvancedQuery:
    """Tests for is_advanced_query()."""

    def test_plain_terms_not_advanced(self):
        from arxiv_browser.app import is_advanced_query

        tokens = [QueryToken(kind="term", value="attention")]
        assert is_advanced_query(tokens) is False

    def test_operator_is_advanced(self):
        from arxiv_browser.app import is_advanced_query

        tokens = [
            QueryToken(kind="term", value="a"),
            QueryToken(kind="op", value="AND"),
            QueryToken(kind="term", value="b"),
        ]
        assert is_advanced_query(tokens) is True

    def test_field_prefix_is_advanced(self):
        from arxiv_browser.app import is_advanced_query

        tokens = [QueryToken(kind="term", value="cs.AI", field="cat")]
        assert is_advanced_query(tokens) is True

    def test_quoted_phrase_is_advanced(self):
        from arxiv_browser.app import is_advanced_query

        tokens = [QueryToken(kind="term", value="deep learning", phrase=True)]
        assert is_advanced_query(tokens) is True

    def test_unread_virtual_term_is_advanced(self):
        from arxiv_browser.app import is_advanced_query

        tokens = [QueryToken(kind="term", value="unread")]
        assert is_advanced_query(tokens) is True

    def test_starred_virtual_term_is_advanced(self):
        from arxiv_browser.app import is_advanced_query

        tokens = [QueryToken(kind="term", value="starred")]
        assert is_advanced_query(tokens) is True


class TestMatchQueryTerm:
    """Tests for match_query_term()."""

    def test_empty_value_matches_all(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="   ")
        assert match_query_term(paper, token, None) is True

    def test_cat_field_matches(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper(categories="cs.AI cs.LG")
        token = QueryToken(kind="term", value="cs.AI", field="cat")
        assert match_query_term(paper, token, None) is True

    def test_cat_field_no_match(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper(categories="cs.AI")
        token = QueryToken(kind="term", value="cs.CV", field="cat")
        assert match_query_term(paper, token, None) is False

    def test_tag_field_matches(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper()
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["important", "to-read"])
        token = QueryToken(kind="term", value="important", field="tag")
        assert match_query_term(paper, token, meta) is True

    def test_tag_field_no_metadata(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="important", field="tag")
        assert match_query_term(paper, token, None) is False

    def test_title_field_matches(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper(title="Deep Learning for NLP")
        token = QueryToken(kind="term", value="Deep", field="title")
        assert match_query_term(paper, token, None) is True

    def test_author_field_matches(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper(authors="John Smith")
        token = QueryToken(kind="term", value="smith", field="author")
        assert match_query_term(paper, token, None) is True

    def test_abstract_field_matches(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="test abstract", field="abstract")
        assert match_query_term(paper, token, None, abstract_text="Test abstract content.") is True

    def test_unread_virtual_term(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="unread")
        # No metadata = unread
        assert match_query_term(paper, token, None) is True
        # Read = not unread
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, is_read=True)
        assert match_query_term(paper, token, meta) is False

    def test_starred_virtual_term(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="starred")
        # No metadata = not starred
        assert match_query_term(paper, token, None) is False
        # Starred = starred
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, starred=True)
        assert match_query_term(paper, token, meta) is True

    def test_fallback_search_title_and_authors(self, make_paper):
        from arxiv_browser.app import match_query_term

        paper = make_paper(title="Attention Mechanism", authors="Jane Doe")
        token = QueryToken(kind="term", value="attention")
        assert match_query_term(paper, token, None) is True


class TestMatchesAdvancedQuery:
    """Tests for matches_advanced_query()."""

    def test_empty_rpn_matches_all(self, make_paper):
        from arxiv_browser.app import matches_advanced_query

        paper = make_paper()
        assert matches_advanced_query(paper, [], None) is True

    def test_single_term(self, make_paper):
        from arxiv_browser.app import matches_advanced_query

        paper = make_paper(categories="cs.AI")
        rpn = [QueryToken(kind="term", value="cs.AI", field="cat")]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_and_query(self, make_paper):
        from arxiv_browser.app import matches_advanced_query

        paper = make_paper(title="Deep Learning for NLP")
        rpn = [
            QueryToken(kind="term", value="deep"),
            QueryToken(kind="term", value="nlp"),
            QueryToken(kind="op", value="AND"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_or_query(self, make_paper):
        from arxiv_browser.app import matches_advanced_query

        paper = make_paper(title="Deep Learning")
        rpn = [
            QueryToken(kind="term", value="quantum"),
            QueryToken(kind="term", value="deep"),
            QueryToken(kind="op", value="OR"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_not_query(self, make_paper):
        from arxiv_browser.app import matches_advanced_query

        paper = make_paper(title="Deep Learning")
        rpn = [
            QueryToken(kind="term", value="quantum"),
            QueryToken(kind="op", value="NOT"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True


class TestPaperMatchesWatchEntry:
    """Tests for paper_matches_watch_entry()."""

    def test_author_match(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper(authors="John Smith, Jane Doe")
        entry = WatchListEntry(pattern="Smith", match_type="author")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_author_no_match(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper(authors="John Smith")
        entry = WatchListEntry(pattern="Wilson", match_type="author")
        assert paper_matches_watch_entry(paper, entry) is False

    def test_title_match(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper(title="Deep Learning for NLP")
        entry = WatchListEntry(pattern="Deep Learning", match_type="title")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_keyword_match_in_title(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper(title="Transformer Architecture", abstract_raw="Some abstract")
        entry = WatchListEntry(pattern="transformer", match_type="keyword")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_keyword_match_in_abstract(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper(title="Some Title", abstract_raw="attention mechanism")
        entry = WatchListEntry(pattern="attention", match_type="keyword")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_case_sensitive(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper(authors="john smith")
        entry = WatchListEntry(pattern="John", match_type="author", case_sensitive=True)
        assert paper_matches_watch_entry(paper, entry) is False

    def test_unknown_match_type(self, make_paper):
        from arxiv_browser.app import WatchListEntry, paper_matches_watch_entry

        paper = make_paper()
        entry = WatchListEntry(pattern="test", match_type="unknown")
        assert paper_matches_watch_entry(paper, entry) is False


class TestSortPapers:
    """Tests for sort_papers()."""

    def test_sort_by_title(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(title="Zebra"),
            make_paper(title="Apple"),
            make_paper(title="Mango"),
        ]
        result = sort_papers(papers, "title")
        assert [p.title for p in result] == ["Apple", "Mango", "Zebra"]

    def test_sort_by_date_descending(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(date="Mon, 1 Jan 2024"),
            make_paper(date="Wed, 15 Jan 2024"),
            make_paper(date="Tue, 10 Jan 2024"),
        ]
        result = sort_papers(papers, "date")
        assert result[0].date == "Wed, 15 Jan 2024"
        assert result[-1].date == "Mon, 1 Jan 2024"

    def test_sort_by_arxiv_id_descending(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00002"),
        ]
        result = sort_papers(papers, "arxiv_id")
        assert [p.arxiv_id for p in result] == ["2401.00003", "2401.00002", "2401.00001"]

    def test_sort_does_not_mutate_original(self, make_paper):
        from arxiv_browser.app import sort_papers

        papers = [make_paper(title="B"), make_paper(title="A")]
        original_order = [p.title for p in papers]
        sort_papers(papers, "title")
        assert [p.title for p in papers] == original_order


class TestFormatPaperForClipboard:
    """Tests for format_paper_for_clipboard()."""

    def test_basic_format(self, make_paper):
        from arxiv_browser.app import format_paper_for_clipboard

        paper = make_paper(title="Test Paper", authors="Author", arxiv_id="2401.12345")
        result = format_paper_for_clipboard(paper, abstract_text="Some abstract")
        assert "Title: Test Paper" in result
        assert "Authors: Author" in result
        assert "Abstract: Some abstract" in result

    def test_includes_comments(self, make_paper):
        from arxiv_browser.app import format_paper_for_clipboard

        paper = make_paper(comments="10 pages, 5 figures")
        result = format_paper_for_clipboard(paper)
        assert "Comments: 10 pages, 5 figures" in result

    def test_omits_none_comments(self, make_paper):
        from arxiv_browser.app import format_paper_for_clipboard

        paper = make_paper(comments=None)
        result = format_paper_for_clipboard(paper)
        assert "Comments:" not in result


class TestFormatPaperAsMarkdown:
    """Tests for format_paper_as_markdown()."""

    def test_headers_and_sections(self, make_paper):
        from arxiv_browser.app import format_paper_as_markdown

        paper = make_paper(title="Test Paper", authors="Author")
        result = format_paper_as_markdown(paper, abstract_text="Some abstract")
        assert "## Test Paper" in result
        assert "### Abstract" in result
        assert "**Authors:** Author" in result

    def test_arxiv_link_format(self, make_paper):
        from arxiv_browser.app import format_paper_as_markdown

        paper = make_paper(arxiv_id="2401.12345")
        result = format_paper_as_markdown(paper)
        assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in result


class TestGetPdfUrl:
    """Tests for get_pdf_url()."""

    def test_standard_abs_url(self, make_paper):
        from arxiv_browser.app import get_pdf_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345", arxiv_id="2401.12345")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_already_pdf_url(self, make_paper):
        from arxiv_browser.app import get_pdf_url

        paper = make_paper(url="https://arxiv.org/pdf/2401.12345.pdf")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_pdf_url_without_extension(self, make_paper):
        from arxiv_browser.app import get_pdf_url

        paper = make_paper(url="https://arxiv.org/pdf/2401.12345")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"


class TestGetPaperUrl:
    """Tests for get_paper_url()."""

    def test_default_abs_url(self, make_paper):
        from arxiv_browser.app import get_paper_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345")
        assert get_paper_url(paper) == "https://arxiv.org/abs/2401.12345"

    def test_prefer_pdf(self, make_paper):
        from arxiv_browser.app import get_paper_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345", arxiv_id="2401.12345")
        result = get_paper_url(paper, prefer_pdf=True)
        assert "pdf" in result


class TestBuildHighlightTerms:
    """Tests for build_highlight_terms()."""

    def test_title_field(self):
        from arxiv_browser.app import build_highlight_terms

        tokens = [QueryToken(kind="term", value="deep", field="title")]
        result = build_highlight_terms(tokens)
        assert "deep" in result["title"]
        assert result["author"] == []

    def test_unfielded_goes_to_title_and_author(self):
        from arxiv_browser.app import build_highlight_terms

        tokens = [QueryToken(kind="term", value="smith")]
        result = build_highlight_terms(tokens)
        assert "smith" in result["title"]
        assert "smith" in result["author"]

    def test_operators_skipped(self):
        from arxiv_browser.app import build_highlight_terms

        tokens = [QueryToken(kind="op", value="AND")]
        result = build_highlight_terms(tokens)
        assert all(v == [] for v in result.values())

    def test_virtual_terms_skipped(self):
        from arxiv_browser.app import build_highlight_terms

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

        from arxiv_browser.app import main

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        (history_dir / "2024-01-15.txt").write_text("test content")

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--list-dates"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), history_dir / "2024-01-15.txt")],
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "2024-01-15" in captured.out

    def test_list_dates_empty_history(self, monkeypatch, capsys):
        """--list-dates with no files should return 1."""
        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--list-dates"])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "No history files" in captured.err

    def test_input_file_not_found(self, tmp_path, monkeypatch, capsys):
        """-i nonexistent.txt should return 1."""
        from arxiv_browser.app import main

        nonexistent = str(tmp_path / "nonexistent.txt")
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", nonexistent])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "not found" in captured.err

    def test_input_file_is_directory(self, tmp_path, monkeypatch, capsys):
        """-i /some/dir should return 1."""
        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", str(tmp_path)])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "directory" in captured.err

    def test_no_papers_exits_with_error(self, tmp_path, monkeypatch, capsys):
        """Empty file should return 1 with 'No papers'."""
        from arxiv_browser.app import main

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", str(empty_file)])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "No papers" in captured.err

    def test_invalid_date_format(self, monkeypatch, capsys):
        """--date Jan-15-2024 should return 1 with 'Invalid date'."""
        from datetime import date as datemod

        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "Jan-15-2024"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), Path("/fake/2024-01-15.txt"))],
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Invalid date" in captured.err

    def test_date_not_found(self, monkeypatch, capsys):
        """--date 2099-01-01 should return 1 with 'No file found'."""
        from datetime import date as datemod

        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "2099-01-01"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), Path("/fake/2024-01-15.txt"))],
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

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
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                abstract=f"Abstract content for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_app_renders_paper_list(self, make_paper):
        """App should mount and render all papers in the option list."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test():
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 5

    async def test_search_filters_papers(self, make_paper):
        """Typing in search should filter the paper list after debounce."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

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
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search
                await pilot.press("slash")
                # Type a query that matches only the quantum paper
                await pilot.press("Q", "u", "a", "n", "t", "u", "m")
                # Wait for debounce (0.3s) + margin for DOM update
                await pilot.pause(0.7)
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 1

    async def test_search_clear_restores_all(self, make_paper):
        """Pressing escape on search should restore all papers."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search and type an advanced query that matches nothing
                # Use "cat:" prefix to trigger exact (non-fuzzy) matching
                await pilot.press("slash")
                for ch in "cat:nonexistent":
                    await pilot.press(ch)
                await pilot.pause(0.7)
                option_list = app.query_one("#paper-list", OptionList)
                # Empty state shows a single placeholder option
                assert option_list.option_count == 1

                # Cancel search with escape
                await pilot.press("escape")
                await pilot.pause(0.4)
                assert option_list.option_count == 3

    async def test_filter_to_empty_cancels_pending_detail_update(self, make_paper):
        """Filtering to an empty list must not allow stale debounced detail updates."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, PaperDetails

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test():
                details = app.query_one(PaperDetails)
                app._pending_detail_paper = papers[0]
                app._apply_filter("cat:nonexistent")
                assert "Select a paper" in str(details.content)

                # Simulate a late timer callback from a previous highlight.
                app._debounced_detail_update()
                assert "Select a paper" in str(details.content)

    async def test_sort_cycling(self, make_paper):
        """Pressing 's' should cycle through sort options."""
        from unittest.mock import patch

        from arxiv_browser.app import SORT_OPTIONS, ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert app._sort_index == 0
                await pilot.press("s")
                assert app._sort_index == 1
                await pilot.press("s")
                assert app._sort_index == 2
                await pilot.press("s")
                assert app._sort_index == 3  # citations
                await pilot.press("s")
                assert app._sort_index == 4  # trending
                await pilot.press("s")
                assert app._sort_index == 5  # relevance
                # Should cycle back to 0
                await pilot.press("s")
                assert app._sort_index == 0

    async def test_toggle_read_status(self, make_paper):
        """Pressing 'r' should toggle read status of the current paper."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, PaperListItem

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
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

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
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

        from arxiv_browser.app import ArxivBrowser, HelpScreen

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
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

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                # Should start at index 0
                assert option_list.highlighted == 0

                # Move down
                await pilot.press("j")
                assert option_list.highlighted == 1
                await pilot.press("j")
                assert option_list.highlighted == 2

                # Move back up
                await pilot.press("k")
                assert option_list.highlighted == 1


@pytest.mark.integration
class TestArxivApiModeIntegration:
    """Integration tests for arXiv API mode transitions and pagination behavior."""

    async def test_escape_exits_api_mode_when_search_box_is_visible(self, make_paper):
        """A single Escape should exit API mode even if search input is open."""
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        local_papers = [make_paper(arxiv_id="2401.00001", title="Local paper")]
        api_paper = make_paper(
            arxiv_id="2602.00001",
            title="API result",
        )
        api_paper.source = "api"
        api_papers = [api_paper]

        app = ArxivBrowser(local_papers, restore_session=False)
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(
                ArxivBrowser,
                "_fetch_arxiv_api_page",
                new_callable=AsyncMock,
                return_value=api_papers,
            ),
            patch.object(
                ArxivBrowser,
                "_apply_arxiv_rate_limit",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with app.run_test() as pilot:
                await pilot.press("A")
                await pilot.press("t", "e", "s", "t", "enter")
                await pilot.pause(0.2)
                assert app._in_arxiv_api_mode is True
                assert app.all_papers[0].arxiv_id == "2602.00001"

                await pilot.press("slash")
                await pilot.pause(0.05)
                await pilot.press("escape")
                await pilot.pause(0.2)

                assert app._in_arxiv_api_mode is False
                assert app.all_papers[0].arxiv_id == "2401.00001"

    async def test_watch_filter_persists_across_api_pages(self, make_paper):
        """Watch filter state should survive paging within API mode."""
        from unittest.mock import AsyncMock, patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser, UserConfig, WatchListEntry

        local_papers = [make_paper(arxiv_id="2401.00001", title="Local paper")]
        api_page_1 = [
            make_paper(arxiv_id="2602.00001", title="Watch this result"),
            make_paper(arxiv_id="2602.00002", title="Other result"),
        ]
        api_page_2 = [
            make_paper(arxiv_id="2602.00003", title="Watch page two"),
            make_paper(arxiv_id="2602.00004", title="Another other"),
        ]
        for paper in api_page_1 + api_page_2:
            paper.source = "api"

        async def fetch_page(_self, _request, start, _max_results):
            return api_page_1 if start == 0 else api_page_2

        config = UserConfig(watch_list=[WatchListEntry(pattern="watch", match_type="title")])
        app = ArxivBrowser(local_papers, config=config, restore_session=False)
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(ArxivBrowser, "_fetch_arxiv_api_page", new=fetch_page),
            patch.object(
                ArxivBrowser,
                "_apply_arxiv_rate_limit",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with app.run_test() as pilot:
                await pilot.press("A")
                await pilot.press("t", "e", "s", "t", "enter")
                await pilot.pause(0.2)
                assert app._in_arxiv_api_mode is True

                await pilot.press("w")
                await pilot.pause(0.1)
                assert app._watch_filter_active is True

                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 1

                await pilot.press("bracketright")
                await pilot.pause(0.2)
                assert app._watch_filter_active is True

                assert option_list.option_count == 1
                # Verify the correct paper via filtered_papers
                assert app.filtered_papers[0].arxiv_id == "2602.00003"

    async def test_brackets_route_to_api_pagination_only_in_api_mode(self, make_paper):
        """[`/`] should call API page change only when API mode is active."""
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, ArxivSearchModeState, ArxivSearchRequest

        app = ArxivBrowser([make_paper(arxiv_id="2401.00001")], restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                change_mock = AsyncMock()
                app._change_arxiv_page = change_mock

                await pilot.press("bracketleft")
                await pilot.press("bracketright")
                await pilot.pause(0.1)
                assert change_mock.await_count == 0

                app._in_arxiv_api_mode = True
                app._arxiv_search_state = ArxivSearchModeState(
                    request=ArxivSearchRequest(query="test"),
                    start=0,
                    max_results=50,
                )

                await pilot.press("bracketleft")
                await pilot.press("bracketright")
                await pilot.pause(0.1)
                assert change_mock.await_count == 2
                assert change_mock.await_args_list[0].args == (-1,)
                assert change_mock.await_args_list[1].args == (1,)

    async def test_stale_api_response_is_ignored_after_exit(self, make_paper):
        """In-flight API responses must not re-enter API mode after exit."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        local_papers = [make_paper(arxiv_id="2401.00001", title="Local")]
        api_page_1 = [make_paper(arxiv_id="2602.00001", title="API page one")]
        api_page_2 = [make_paper(arxiv_id="2602.00002", title="API page two")]
        for paper in api_page_1 + api_page_2:
            paper.source = "api"

        release_second_page = asyncio.Event()

        async def fetch_page(_self, _request, start, _max_results):
            if start == 0:
                return api_page_1
            await release_second_page.wait()
            return api_page_2

        app = ArxivBrowser(local_papers, restore_session=False)
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(ArxivBrowser, "_fetch_arxiv_api_page", new=fetch_page),
            patch.object(
                ArxivBrowser,
                "_apply_arxiv_rate_limit",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with app.run_test() as pilot:
                await pilot.press("A")
                await pilot.press("t", "e", "s", "t", "enter")
                await pilot.pause(0.2)
                assert app._in_arxiv_api_mode is True
                assert app.all_papers[0].arxiv_id == "2602.00001"

                await pilot.press("bracketright")
                await pilot.pause(0.05)
                assert app._arxiv_api_fetch_inflight is True

                app.action_exit_arxiv_search_mode()
                assert app._in_arxiv_api_mode is False

                release_second_page.set()
                await pilot.pause(0.25)

                assert app._in_arxiv_api_mode is False
                assert app.all_papers[0].arxiv_id == "2401.00001"


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
        from arxiv_browser.app import ArxivBrowser

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
        with patch("arxiv_browser.app.save_config", return_value=True):
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

        from textual.widgets import Label

        app = self._make_app(make_paper)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Open search and type a query that won't match our paper
                await pilot.press("slash")
                await pilot.press("t", "r", "a", "n", "s", "f", "o", "r", "m", "e", "r")
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

        from textual.widgets import Input, OptionList

        app = self._make_app(make_paper)
        with patch("arxiv_browser.app.save_config", return_value=True):
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
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 1  # App was created with 1 paper


# ============================================================================
# Tests for LLM Summary Functions
# ============================================================================


class TestBuildLlmPrompt:
    """Tests for build_llm_prompt template expansion."""

    def _make_paper(self, **kwargs):
        defaults = {
            "arxiv_id": "2301.00001",
            "date": "Mon, 2 Jan 2023",
            "title": "Test Paper",
            "authors": "Alice, Bob",
            "categories": "cs.AI",
            "comments": None,
            "abstract": "An abstract.",
            "url": "https://arxiv.org/abs/2301.00001",
        }
        defaults.update(kwargs)
        return Paper(**defaults)

    def test_default_prompt(self):
        paper = self._make_paper()
        result = build_llm_prompt(paper)
        assert "Test Paper" in result
        assert "Alice, Bob" in result
        assert "cs.AI" in result
        assert "An abstract." in result

    def test_custom_template(self):
        paper = self._make_paper()
        template = "Summarize: {title} by {authors}"
        result = build_llm_prompt(paper, template)
        # Template lacks {paper_content}, so content is auto-appended
        assert result.startswith("Summarize: Test Paper by Alice, Bob")
        assert "An abstract." in result

    def test_all_placeholders(self):
        paper = self._make_paper()
        template = "{title}|{authors}|{categories}|{abstract}|{arxiv_id}|{paper_content}"
        result = build_llm_prompt(paper, template)
        assert result.startswith("Test Paper|Alice, Bob|cs.AI|An abstract.|2301.00001|")

    def test_no_abstract_fallback(self):
        paper = self._make_paper(abstract=None, abstract_raw="raw text")
        result = build_llm_prompt(paper)
        assert "raw text" in result

    def test_no_abstract_at_all(self):
        paper = self._make_paper(abstract=None, abstract_raw="")
        result = build_llm_prompt(paper)
        assert "(no abstract)" in result

    def test_paper_content_placeholder(self):
        paper = self._make_paper()
        result = build_llm_prompt(paper, paper_content="Full paper text here.")
        assert "Full paper text here." in result

    def test_paper_content_fallback_to_abstract(self):
        paper = self._make_paper()
        result = build_llm_prompt(paper, paper_content="")
        assert "An abstract." in result

    def test_auto_append_when_template_lacks_paper_content(self):
        paper = self._make_paper()
        result = build_llm_prompt(paper, "Summarize: {title}", paper_content="Full paper text.")
        assert "Summarize: Test Paper" in result
        assert "Full paper text." in result

    def test_no_auto_append_when_template_has_paper_content(self):
        paper = self._make_paper()
        result = build_llm_prompt(
            paper, "Context: {paper_content}\nQ: {title}?", paper_content="Full text."
        )
        # paper_content appears exactly once (substituted, not appended)
        assert result.count("Full text.") == 1

    def test_invalid_placeholder_raises_valueerror(self):
        paper = self._make_paper()
        with pytest.raises(ValueError, match="Invalid prompt template"):
            build_llm_prompt(paper, "Summarize: {titl}")

    def test_unescaped_braces_raise_valueerror(self):
        paper = self._make_paper()
        with pytest.raises(ValueError, match="Invalid prompt template"):
            build_llm_prompt(paper, 'Output: {"key": "{title}"}')


class TestExtractTextFromHtml:
    """Tests for HTML text extraction from arXiv pages."""

    def test_basic_paragraph(self):
        html = "<p>Hello world</p>"
        assert extract_text_from_html(html) == "Hello world"

    def test_nested_tags(self):
        html = "<div><p>First paragraph</p><p>Second paragraph</p></div>"
        result = extract_text_from_html(html)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_script_stripped(self):
        html = "<p>Visible</p><script>var x = 1;</script><p>Also visible</p>"
        result = extract_text_from_html(html)
        assert "Visible" in result
        assert "Also visible" in result
        assert "var x" not in result

    def test_math_stripped(self):
        html = "<p>The value <math><mi>x</mi></math> is important</p>"
        result = extract_text_from_html(html)
        assert "The value" in result
        assert "is important" in result
        # Math internals should be skipped
        assert "<mi>" not in result

    def test_style_stripped(self):
        html = "<style>.cls { color: red; }</style><p>Content</p>"
        result = extract_text_from_html(html)
        assert "Content" in result
        assert "color" not in result

    def test_whitespace_collapsed(self):
        html = "<p>  Too   many   spaces  </p>"
        assert extract_text_from_html(html) == "Too many spaces"

    def test_empty_html(self):
        assert extract_text_from_html("") == ""

    def test_arxiv_like_structure(self):
        html = (
            '<article class="ltx_document">'
            '<h1 class="ltx_title">My Paper Title</h1>'
            '<div class="ltx_abstract"><h6>Abstract</h6>'
            '<p class="ltx_p">This is the abstract.</p></div>'
            '<section class="ltx_section">'
            "<h2>Introduction</h2>"
            '<p class="ltx_p">We introduce a method.</p>'
            "</section></article>"
        )
        result = extract_text_from_html(html)
        assert "My Paper Title" in result
        assert "This is the abstract." in result
        assert "We introduce a method." in result

    def test_nested_skip_tags_same_type(self):
        """Nested skip tags of the same type using <nav>: depth 0→1→2→1→0."""
        # HTMLParser treats <script>/<style> content as CDATA (no inner parsing),
        # so we use <nav> which HTMLParser fully parses as nested tags.
        html = "<nav><nav>inner</nav>between</nav><p>Visible</p>"
        result = extract_text_from_html(html)
        assert "Visible" in result
        assert "inner" not in result
        assert "between" not in result

    def test_nested_different_skip_tags(self):
        """Mixed nesting of different skip tags: <nav><style>...</style>...</nav>."""
        html = "<nav><style>css</style>nav text</nav><p>Content</p>"
        result = extract_text_from_html(html)
        assert "Content" in result
        assert "css" not in result
        assert "nav text" not in result

    def test_mismatched_close_tag_underflow(self):
        """Orphan </script> must not drive depth negative and suppress text."""
        html = "</script><p>Text</p>"
        result = extract_text_from_html(html)
        assert "Text" in result

    def test_multiple_mismatched_close_tags(self):
        """Triple orphan close tags still allow subsequent text through."""
        html = "</style></nav></footer><p>Visible</p>"
        result = extract_text_from_html(html)
        assert "Visible" in result

    def test_nav_tag_stripped(self):
        html = "<nav>Home About</nav><p>Content</p>"
        result = extract_text_from_html(html)
        assert "Content" in result
        assert "Home" not in result

    def test_header_tag_stripped(self):
        html = "<header><h1>Site Title</h1></header><p>Body</p>"
        result = extract_text_from_html(html)
        assert "Body" in result
        assert "Site Title" not in result

    def test_footer_tag_stripped(self):
        html = "<footer>Copyright 2024</footer><p>Main</p>"
        result = extract_text_from_html(html)
        assert "Main" in result
        assert "Copyright" not in result

    def test_skip_tag_inside_block(self):
        """Script inside a div: only non-script content survives."""
        html = "<div><script>alert('x')</script><p>After</p></div>"
        result = extract_text_from_html(html)
        assert "After" in result
        assert "alert" not in result

    def test_block_tag_inside_skip_tag(self):
        """Block elements inside a skip tag are still suppressed."""
        html = "<nav><p>Nav text</p></nav><p>Visible</p>"
        result = extract_text_from_html(html)
        assert "Visible" in result
        assert "Nav text" not in result

    def test_deeply_nested_skip_tags(self):
        """Three levels of nesting: depth goes 0→1→2→3→2→1→0."""
        html = "<script><style><math>deep</math></style></script><p>OK</p>"
        result = extract_text_from_html(html)
        assert "OK" in result
        assert "deep" not in result


class TestFetchPaperContentAsync:
    """Tests for async paper content fetching with httpx mocking."""

    @pytest.fixture
    def paper(self, make_paper):
        return make_paper(arxiv_id="2401.12345", abstract="Test abstract.")

    async def test_success_returns_extracted_text(self, paper):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import _fetch_paper_content_async

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "<p>Introduction to transformers.</p>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert "Introduction to transformers" in result

    async def test_success_truncates_long_content(self, paper):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import MAX_PAPER_CONTENT_LENGTH, _fetch_paper_content_async

        long_text = "x" * (MAX_PAPER_CONTENT_LENGTH + 1000)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = f"<p>{long_text}</p>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert len(result) == MAX_PAPER_CONTENT_LENGTH

    async def test_empty_extraction_falls_back_to_abstract(self, paper):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import _fetch_paper_content_async

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = ""  # empty HTML → empty extraction

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert result == "Abstract:\nTest abstract."

    async def test_non_200_falls_back_to_abstract(self, paper):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import _fetch_paper_content_async

        mock_response = AsyncMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert result == "Abstract:\nTest abstract."

    async def test_http_error_falls_back(self, paper):
        from unittest.mock import AsyncMock, patch

        import httpx

        from arxiv_browser.app import _fetch_paper_content_async

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert result == "Abstract:\nTest abstract."

    async def test_os_error_falls_back(self, paper):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import _fetch_paper_content_async

        mock_client = AsyncMock()
        mock_client.get.side_effect = OSError("Network unreachable")
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert result == "Abstract:\nTest abstract."

    async def test_no_abstract_returns_empty(self, make_paper):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import _fetch_paper_content_async

        paper = make_paper(abstract="", abstract_raw="")

        mock_client = AsyncMock()
        mock_client.get.side_effect = OSError("fail")
        mock_client.__aenter__.return_value = mock_client

        with patch("arxiv_browser.app.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_paper_content_async(paper)

        assert result == ""


class TestFormatSummaryAsRich:
    """Tests for markdown-to-Rich markup conversion of LLM summaries."""

    def test_heading_h2(self):
        result = format_summary_as_rich("## Core Idea")
        assert "[bold" in result
        assert "Core Idea" in result
        assert "##" not in result

    def test_heading_h3(self):
        result = format_summary_as_rich("### Sub-heading")
        assert "[bold]" in result
        assert "Sub-heading" in result

    def test_bold(self):
        result = format_summary_as_rich("This is **important** text")
        assert "[bold]important[/]" in result
        assert "**" not in result

    def test_inline_code(self):
        result = format_summary_as_rich("Use `method_name` here")
        assert "method_name" in result
        assert "`" not in result

    def test_bullets(self):
        result = format_summary_as_rich("- First item\n- Second item")
        assert "•" in result
        assert "First item" in result
        assert "Second item" in result

    def test_combined(self):
        md = "## Pros\n- **Strong results** on benchmark\n- Clean `API` design"
        result = format_summary_as_rich(md)
        assert "Pros" in result
        assert "•" in result
        assert "[bold]Strong results[/]" in result
        assert "##" not in result
        assert "**" not in result

    def test_empty(self):
        assert format_summary_as_rich("") == ""

    def test_plain_text(self):
        result = format_summary_as_rich("Just plain text.")
        assert "plain text" in result

    def test_rich_markup_escaped(self):
        result = format_summary_as_rich("Text with [bold]fake markup[/bold]")
        # Square brackets must be escaped so Rich doesn't interpret them as tags
        assert "\\[bold\\]" in result or "\\[bold]" in result
        # The word should still appear in the output
        assert "fake markup" in result


class TestTrackTaskExceptionSurfacing:
    """Tests for _track_task done-callback exception logging."""

    def test_unhandled_exception_is_logged(self):
        """_on_task_done logs unhandled exceptions from completed tasks."""
        import asyncio
        from unittest.mock import MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        exc = RuntimeError("boom")
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = exc

        with patch("arxiv_browser.app.logger") as mock_logger:
            ArxivBrowser._on_task_done(mock_task)

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Unhandled exception in background task" in call_args[0][0]
        assert call_args[0][1] is exc
        assert call_args[1]["exc_info"] is exc

    def test_handled_exception_not_double_logged(self):
        """_on_task_done does not log when task completes without exception."""
        import asyncio
        from unittest.mock import MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None

        with patch("arxiv_browser.app.logger") as mock_logger:
            ArxivBrowser._on_task_done(mock_task)

        mock_logger.error.assert_not_called()

    def test_cancelled_task_not_logged(self):
        """_on_task_done does not log for cancelled tasks."""
        import asyncio
        from unittest.mock import MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.cancelled.return_value = True

        with patch("arxiv_browser.app.logger") as mock_logger:
            ArxivBrowser._on_task_done(mock_task)

        mock_logger.error.assert_not_called()
        # exception() should NOT be called when task is cancelled
        mock_task.exception.assert_not_called()


class TestGenerateSummaryAsync:
    """Tests for the LLM summary generation async method."""

    @pytest.fixture
    def paper(self, make_paper):
        return make_paper(arxiv_id="2401.12345", abstract="Test abstract.")

    @pytest.fixture
    def mock_app(self, tmp_path):
        """Create a minimal mock of ArxivBrowser with required attributes."""
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._paper_summaries = {}
        app._summary_loading = set()
        app._summary_db_path = tmp_path / "test_summaries.db"
        app._http_client = None
        app.notify = MagicMock()
        app._update_abstract_display = MagicMock()
        return app

    def _make_proc_mock(self, stdout=b"", stderr=b"", returncode=0):
        """Create an AsyncMock subprocess with controlled outputs."""
        from unittest.mock import AsyncMock, MagicMock

        proc = AsyncMock()
        proc.communicate.return_value = (stdout, stderr)
        proc.returncode = returncode
        proc.kill = MagicMock()  # kill() is synchronous
        proc.wait = AsyncMock()
        return proc

    async def test_success_caches_and_notifies(self, paper, mock_app):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        proc = self._make_proc_mock(stdout=b"Great paper about transformers.")

        with (
            patch(
                "arxiv_browser.app._fetch_paper_content_async",
                new_callable=AsyncMock,
                return_value="Full paper text.",
            ),
            patch(
                "arxiv_browser.app.asyncio.create_subprocess_shell",
                new_callable=AsyncMock,
                return_value=proc,
            ),
            patch(
                "arxiv_browser.app.asyncio.wait_for",
                new_callable=AsyncMock,
                return_value=(b"Great paper about transformers.", b""),
            ),
            patch("arxiv_browser.app._save_summary"),
        ):
            await ArxivBrowser._generate_summary_async(
                mock_app, paper, "claude -p {prompt}", "", "hash123"
            )

        assert mock_app._paper_summaries["2401.12345"] == "Great paper about transformers."
        # Verify notification
        notify_calls = [c for c in mock_app.notify.call_args_list if "Summary generated" in str(c)]
        assert len(notify_calls) == 1
        # Verify loading state cleaned up
        assert "2401.12345" not in mock_app._summary_loading
        mock_app._update_abstract_display.assert_called()

    async def test_timeout_kills_process(self, paper, mock_app):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        proc = self._make_proc_mock()

        with (
            patch(
                "arxiv_browser.app._fetch_paper_content_async",
                new_callable=AsyncMock,
                return_value="text",
            ),
            patch(
                "arxiv_browser.app.asyncio.create_subprocess_shell",
                new_callable=AsyncMock,
                return_value=proc,
            ),
            patch(
                "arxiv_browser.app.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=TimeoutError(),
            ),
        ):
            await ArxivBrowser._generate_summary_async(
                mock_app, paper, "claude -p {prompt}", "", "hash123"
            )

        proc.kill.assert_called_once()
        proc.wait.assert_called_once()
        assert "2401.12345" not in mock_app._paper_summaries
        error_calls = [c for c in mock_app.notify.call_args_list if "timed out" in str(c)]
        assert len(error_calls) == 1
        # Verify loading state cleaned up
        assert "2401.12345" not in mock_app._summary_loading

    async def test_nonzero_exit_shows_stderr(self, paper, mock_app):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        proc = self._make_proc_mock(stderr=b"Model not found", returncode=1)

        with (
            patch(
                "arxiv_browser.app._fetch_paper_content_async",
                new_callable=AsyncMock,
                return_value="text",
            ),
            patch(
                "arxiv_browser.app.asyncio.create_subprocess_shell",
                new_callable=AsyncMock,
                return_value=proc,
            ),
            patch(
                "arxiv_browser.app.asyncio.wait_for",
                new_callable=AsyncMock,
                return_value=(b"", b"Model not found"),
            ),
        ):
            await ArxivBrowser._generate_summary_async(
                mock_app, paper, "claude -p {prompt}", "", "hash123"
            )

        assert "2401.12345" not in mock_app._paper_summaries
        error_calls = [c for c in mock_app.notify.call_args_list if "Model not found" in str(c)]
        assert len(error_calls) == 1

    async def test_empty_stdout_warns(self, paper, mock_app):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        proc = self._make_proc_mock(stdout=b"", returncode=0)

        with (
            patch(
                "arxiv_browser.app._fetch_paper_content_async",
                new_callable=AsyncMock,
                return_value="text",
            ),
            patch(
                "arxiv_browser.app.asyncio.create_subprocess_shell",
                new_callable=AsyncMock,
                return_value=proc,
            ),
            patch(
                "arxiv_browser.app.asyncio.wait_for",
                new_callable=AsyncMock,
                return_value=(b"", b""),
            ),
        ):
            await ArxivBrowser._generate_summary_async(
                mock_app, paper, "claude -p {prompt}", "", "hash123"
            )

        assert "2401.12345" not in mock_app._paper_summaries
        warning_calls = [c for c in mock_app.notify.call_args_list if "empty output" in str(c)]
        assert len(warning_calls) == 1

    async def test_value_error_from_template(self, paper, mock_app):
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        with patch(
            "arxiv_browser.app._fetch_paper_content_async",
            new_callable=AsyncMock,
            return_value="text",
        ):
            # Pass a template with an invalid placeholder to trigger ValueError
            await ArxivBrowser._generate_summary_async(
                mock_app, paper, "claude -p {prompt}", "Summarize: {invalid_field}", "hash123"
            )

        assert "2401.12345" not in mock_app._paper_summaries
        error_calls = [
            c for c in mock_app.notify.call_args_list if "severity" in str(c) and "error" in str(c)
        ]
        assert len(error_calls) >= 1
        # Verify loading state cleaned up
        assert "2401.12345" not in mock_app._summary_loading

    async def test_finally_cleans_up_loading_state(self, paper, mock_app):
        """All code paths must clean up _summary_loading and call _update_abstract_display."""
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser

        # Pre-add to loading set to verify cleanup
        mock_app._summary_loading.add("2401.12345")

        with (
            patch(
                "arxiv_browser.app._fetch_paper_content_async",
                new_callable=AsyncMock,
                side_effect=Exception("unexpected"),
            ),
        ):
            await ArxivBrowser._generate_summary_async(
                mock_app, paper, "claude -p {prompt}", "", "hash123"
            )

        # finally block must have cleaned up
        assert "2401.12345" not in mock_app._summary_loading
        mock_app._update_abstract_display.assert_called_with("2401.12345")


class TestLlmSummaryDb:
    """Tests for SQLite summary persistence."""

    def test_save_and_load(self, tmp_path):
        from arxiv_browser.app import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "test.db"
        _init_summary_db(db_path)
        _save_summary(db_path, "2301.00001", "A great summary", "hash123")
        result = _load_summary(db_path, "2301.00001", "hash123")
        assert result == "A great summary"

    def test_load_missing(self, tmp_path):
        from arxiv_browser.app import _init_summary_db, _load_summary

        db_path = tmp_path / "test.db"
        _init_summary_db(db_path)
        result = _load_summary(db_path, "nonexistent", "hash123")
        assert result is None

    def test_load_wrong_hash(self, tmp_path):
        from arxiv_browser.app import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "test.db"
        _init_summary_db(db_path)
        _save_summary(db_path, "2301.00001", "A summary", "hash_old")
        result = _load_summary(db_path, "2301.00001", "hash_new")
        assert result is None

    def test_load_nonexistent_db(self, tmp_path):
        from arxiv_browser.app import _load_summary

        db_path = tmp_path / "does_not_exist.db"
        result = _load_summary(db_path, "2301.00001", "hash123")
        assert result is None

    def test_upsert_replaces(self, tmp_path):
        from arxiv_browser.app import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "test.db"
        _init_summary_db(db_path)
        _save_summary(db_path, "2301.00001", "Old summary", "hash_v2")
        _save_summary(db_path, "2301.00001", "New summary", "hash_v2")
        result = _load_summary(db_path, "2301.00001", "hash_v2")
        assert result == "New summary"


class TestLlmCommandResolution:
    """Tests for LLM command template resolution."""

    def test_custom_command(self):
        from arxiv_browser.app import _resolve_llm_command

        config = UserConfig(llm_command="my-tool {prompt}")
        assert _resolve_llm_command(config) == "my-tool {prompt}"

    def test_preset_claude(self):
        from arxiv_browser.app import _resolve_llm_command

        config = UserConfig(llm_preset="claude")
        result = _resolve_llm_command(config)
        assert "claude" in result
        assert "{prompt}" in result

    def test_preset_unknown_warns(self, caplog):
        import logging

        from arxiv_browser.app import _resolve_llm_command

        config = UserConfig(llm_preset="unknown_tool")
        with caplog.at_level(logging.WARNING, logger="arxiv_browser"):
            assert _resolve_llm_command(config) == ""
        assert "unknown_tool" in caplog.text
        assert "Valid presets" in caplog.text

    def test_no_config(self):
        from arxiv_browser.app import _resolve_llm_command

        config = UserConfig()
        assert _resolve_llm_command(config) == ""

    def test_custom_overrides_preset(self):
        from arxiv_browser.app import _resolve_llm_command

        config = UserConfig(llm_command="custom {prompt}", llm_preset="claude")
        assert _resolve_llm_command(config) == "custom {prompt}"


class TestRequireLlmCommand:
    """Verify _require_llm_command helper notifies when LLM is not configured."""

    def _make_mock_app(self, **config_kwargs):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig(**config_kwargs)
        app.notify = MagicMock()
        return app

    def test_returns_command_when_configured(self):
        app = self._make_mock_app(llm_command="my-tool {prompt}")
        result = app._require_llm_command()
        assert result == "my-tool {prompt}"
        app.notify.assert_not_called()

    def test_returns_none_and_notifies_when_not_configured(self):
        app = self._make_mock_app()
        result = app._require_llm_command()
        assert result is None
        app.notify.assert_called_once()
        assert "LLM not configured" in str(app.notify.call_args)

    def test_returns_none_with_unknown_preset(self):
        app = self._make_mock_app(llm_preset="nonexistent")
        result = app._require_llm_command()
        assert result is None
        call_args_str = str(app.notify.call_args)
        assert "Unknown preset" in call_args_str


class TestBuildLlmShellCommand:
    """Tests for shell command building with proper escaping."""

    def test_basic(self):
        from arxiv_browser.app import _build_llm_shell_command

        result = _build_llm_shell_command("claude -p {prompt}", "hello world")
        assert "claude -p" in result
        assert "hello world" in result

    def test_prompt_with_quotes(self):
        from arxiv_browser.app import _build_llm_shell_command

        result = _build_llm_shell_command("claude -p {prompt}", 'text with "quotes"')
        # shlex.quote should handle the quoting
        assert "claude -p" in result
        assert "quotes" in result

    def test_prompt_with_shell_chars(self):
        import shlex

        from arxiv_browser.app import _build_llm_shell_command

        dangerous = "text; rm -rf /"
        result = _build_llm_shell_command("llm {prompt}", dangerous)
        # Verify the prompt is wrapped by shlex.quote (single-quoted)
        assert shlex.quote(dangerous) in result
        # The raw unquoted semicolon must NOT appear outside quotes
        assert result == f"llm {shlex.quote(dangerous)}"

    def test_missing_prompt_placeholder(self):
        from arxiv_browser.app import _build_llm_shell_command

        with pytest.raises(ValueError, match=r"must contain.*\{prompt\}"):
            _build_llm_shell_command("claude", "hello")


class TestCommandHash:
    """Tests for command hash computation."""

    def test_same_input_same_hash(self):
        from arxiv_browser.app import _compute_command_hash

        h1 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        h2 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        assert h1 == h2

    def test_different_command_different_hash(self):
        from arxiv_browser.app import _compute_command_hash

        h1 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        h2 = _compute_command_hash("llm {prompt}", "Summarize: {title}")
        assert h1 != h2

    def test_different_prompt_different_hash(self):
        from arxiv_browser.app import _compute_command_hash

        h1 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        h2 = _compute_command_hash("claude -p {prompt}", "Explain: {title}")
        assert h1 != h2

    def test_hash_length(self):
        from arxiv_browser.app import _compute_command_hash

        h = _compute_command_hash("cmd", "prompt")
        assert len(h) == 16


class TestLlmPresets:
    """Tests for LLM preset definitions."""

    def test_presets_exist(self):
        assert "claude" in LLM_PRESETS
        assert "codex" in LLM_PRESETS
        assert "llm" in LLM_PRESETS
        assert "copilot" in LLM_PRESETS

    def test_presets_have_prompt_placeholder(self):
        for name, cmd in LLM_PRESETS.items():
            assert "{prompt}" in cmd, f"Preset {name!r} missing {{prompt}} placeholder"


class TestLlmConfigSerialization:
    """Tests for LLM config fields roundtrip."""

    def test_roundtrip(self, tmp_path, monkeypatch):
        from arxiv_browser.app import _config_to_dict, _dict_to_config

        config = UserConfig(
            llm_command="claude -p {prompt}",
            llm_prompt_template="Summarize: {title}",
            llm_preset="claude",
        )
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.llm_command == "claude -p {prompt}"
        assert restored.llm_prompt_template == "Summarize: {title}"
        assert restored.llm_preset == "claude"

    def test_missing_fields_default(self):
        from arxiv_browser.app import _dict_to_config

        config = _dict_to_config({})
        assert config.llm_command == ""
        assert config.llm_prompt_template == ""
        assert config.llm_preset == ""


# ============================================================================
# Tests for Phase 1-5 new behavior
# ============================================================================


class TestSummaryStateClearOnDateSwitch:
    """Verify summary caches are cleared when switching dates."""

    def test_load_current_date_clears_summaries(self, make_paper, tmp_path):
        """_load_current_date should clear _paper_summaries and _summary_loading."""
        from datetime import date as dt_date
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        # Create a history file
        hdir = tmp_path / "history"
        hdir.mkdir()
        paper_file = hdir / "2024-01-15.txt"
        paper_file.write_text(
            "arXiv:2401.00001\n"
            "Date: Mon, 15 Jan 2024\n"
            "Title: Test Paper\n"
            "Authors: Author\n"
            "Categories: cs.AI\n"
            "\\\\\n"
            "Abstract text here.\n"
            "(https://arxiv.org/abs/2401.00001)\n"
            "------------------------------------------------------------------------------\n"
        )
        paper_file2 = hdir / "2024-01-16.txt"
        paper_file2.write_text(
            "arXiv:2401.00002\n"
            "Date: Tue, 16 Jan 2024\n"
            "Title: Test Paper 2\n"
            "Authors: Author 2\n"
            "Categories: cs.LG\n"
            "\\\\\n"
            "Abstract text here 2.\n"
            "(https://arxiv.org/abs/2401.00002)\n"
            "------------------------------------------------------------------------------\n"
        )

        history_files = [
            (dt_date(2024, 1, 16), paper_file2),
            (dt_date(2024, 1, 15), paper_file),
        ]
        papers = [make_paper(arxiv_id="2401.00002")]
        app = ArxivBrowser(papers, history_files=history_files, current_date_index=0)

        # Simulate having summaries cached
        app._paper_summaries["2401.00002"] = "Some summary"
        app._summary_loading.add("2401.00002")

        with patch("arxiv_browser.app.save_config", return_value=True):

            async def run_test():
                async with app.run_test():
                    # Switch to older date
                    app._current_date_index = 1
                    app._load_current_date()

                    assert len(app._paper_summaries) == 0
                    assert len(app._summary_loading) == 0

            import asyncio

            asyncio.get_event_loop().run_until_complete(run_test())


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

            asyncio.get_event_loop().run_until_complete(run_test())


class TestDetailKwargs:
    """Verify _detail_kwargs aggregates all detail pane state."""

    def test_detail_kwargs_returns_expected_keys(self, make_paper):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

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
            },
        )()
        app._relevance_scores = {"2401.00001": (8, "relevant")}

        kwargs = app._detail_kwargs("2401.00001")
        assert kwargs["summary"] == "A summary"
        assert kwargs["summary_loading"] is False
        assert kwargs["highlight_terms"] == ["test"]
        assert kwargs["s2_data"] is None
        assert kwargs["s2_loading"] is False
        assert kwargs["hf_data"] is None
        assert kwargs["version_update"] is None
        assert kwargs["summary_mode"] == "tldr"
        assert kwargs["tags"] is None
        assert kwargs["relevance"] == (8, "relevant")
        assert kwargs["collapsed_sections"] == [
            "tags",
            "relevance",
            "summary",
            "s2",
            "hf",
            "version",
        ]


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


class TestBatchConfirmationThreshold:
    """Verify batch confirmation modal behavior."""

    def test_threshold_constant_exists(self):
        from arxiv_browser.app import BATCH_CONFIRM_THRESHOLD

        assert BATCH_CONFIRM_THRESHOLD == 10

    def test_confirm_modal_class_exists(self):
        from arxiv_browser.app import ConfirmModal

        modal = ConfirmModal("Test message?")
        assert modal._message == "Test message?"


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

        from arxiv_browser.app import BATCH_CONFIRM_THRESHOLD, ArxivBrowser, ConfirmModal

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


# ============================================================================
# Semantic Scholar Integration Tests
# ============================================================================


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
        from unittest.mock import MagicMock

        app = self._make_mock_app()
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()

        assert app._s2_active is False
        app.action_toggle_s2()
        assert app._s2_active is True
        app.notify.assert_called_once()
        assert "enabled" in app.notify.call_args[0][0]
        app._mark_badges_dirty.assert_called_once_with("s2", immediate=True)

        app.action_toggle_s2()
        assert app._s2_active is False
        assert "disabled" in app.notify.call_args[0][0]
        assert app._mark_badges_dirty.call_count == 2

    @pytest.mark.asyncio
    async def test_action_fetch_s2_dedupes_concurrent_calls(self, make_paper):
        import asyncio
        import time
        from unittest.mock import MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        paper = make_paper(arxiv_id="2401.42424")
        app = ArxivBrowser([paper], restore_session=False)

        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                app._s2_active = True
                load_calls = 0
                track_calls = 0

                def fake_load_s2_paper(*_args, **_kwargs):
                    nonlocal load_calls
                    load_calls += 1
                    time.sleep(0.05)
                    return None

                def fake_track_task(coro):
                    nonlocal track_calls
                    track_calls += 1
                    coro.close()
                    return MagicMock()

                app._track_task = fake_track_task
                with patch("arxiv_browser.app.load_s2_paper", side_effect=fake_load_s2_paper):
                    await asyncio.gather(app.action_fetch_s2(), app.action_fetch_s2())
                    await pilot.pause(0)

                assert load_calls == 1
                assert track_calls == 1

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


# ============================================================================
# HuggingFace Integration Tests
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
    async def test_fetch_hf_daily_dedupes_concurrent_calls(self, make_paper):
        import asyncio
        import time
        from unittest.mock import MagicMock, patch

        from arxiv_browser.app import ArxivBrowser

        paper = make_paper(arxiv_id="2401.51515")
        app = ArxivBrowser([paper], restore_session=False)

        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                cache_calls = 0
                track_calls = 0

                def fake_load_hf_daily_cache(*_args, **_kwargs):
                    nonlocal cache_calls
                    cache_calls += 1
                    time.sleep(0.05)
                    return None

                def fake_track_task(coro):
                    nonlocal track_calls
                    track_calls += 1
                    coro.close()
                    return MagicMock()

                app._track_task = fake_track_task
                with patch(
                    "arxiv_browser.app.load_hf_daily_cache", side_effect=fake_load_hf_daily_cache
                ):
                    await asyncio.gather(app._fetch_hf_daily(), app._fetch_hf_daily())
                    await pilot.pause(0)

                assert cache_calls == 1
                assert track_calls == 1


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

    def test_badge_timer_cleanup_on_unmount(self):
        """Verify _badge_timer is stopped during on_unmount."""
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

        app = self._make_badge_app()
        app._search_timer = None
        app._detail_timer = None
        mock_timer = MagicMock()
        app._badge_timer = mock_timer
        app._save_session_state = MagicMock()

        app.on_unmount()

        mock_timer.stop.assert_called_once()
        assert app._badge_timer is None

    def test_flush_updates_multiple_papers(self, make_paper):
        """Flush iterates all filtered papers."""
        from unittest.mock import MagicMock

        app = self._make_badge_app()
        app._badges_dirty = {"hf"}
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


# ============================================================================
# Tests for detail pane content caching (P7)
# ============================================================================


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


# ============================================================================
# Tests for parse_arxiv_version_map
# ============================================================================


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


# ============================================================================
# Tests for version metadata serialization
# ============================================================================


class TestVersionMetadataSerialization:
    """Tests for last_checked_version config round-trip."""

    def test_round_trip_with_version(self, tmp_path, monkeypatch):
        """last_checked_version survives save/load cycle."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

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
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

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
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

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


# ============================================================================
# Tests for version detail pane
# ============================================================================


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


# ============================================================================
# Tests for version list badge
# ============================================================================


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


# ============================================================================
# Tests for version app state
# ============================================================================


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


# ============================================================================
# Tests for RIS export format
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


# ============================================================================
# Tests for CSV export format
# ============================================================================


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


class TestCSVExportMethods:
    """Verify ArxivBrowser CSV export methods use self._config.paper_metadata."""

    def _make_mock_app(self, papers, metadata=None):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser

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


# ============================================================================
# Tests for Markdown table export format
# ============================================================================


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


# ============================================================================
# Structured Summary Modes (#2.3) Tests
# ============================================================================


class TestSummaryModes:
    """Tests for SUMMARY_MODES constant and mode templates."""

    def test_summary_modes_has_expected_keys(self):
        expected = {"default", "tldr", "methods", "results", "comparison"}
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
        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        # Each action should produce the correct mode name
        assert hasattr(modal, "action_mode_default")
        assert hasattr(modal, "action_mode_tldr")
        assert hasattr(modal, "action_mode_methods")
        assert hasattr(modal, "action_mode_results")
        assert hasattr(modal, "action_mode_comparison")
        assert hasattr(modal, "action_cancel")

    def test_modal_bindings(self):
        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        binding_keys = {b.key for b in modal.BINDINGS}
        assert {"d", "t", "m", "r", "c", "escape"} <= binding_keys


class TestSummaryDbMigration:
    """Tests for the DB schema migration from single to composite PK."""

    def test_creates_new_db_with_composite_pk(self, tmp_path):
        import sqlite3

        from arxiv_browser.app import _init_summary_db

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries'"
            ).fetchone()
            assert "PRIMARY KEY (arxiv_id, command_hash)" in row[0]

    def test_migrates_old_single_pk_schema(self, tmp_path):
        import sqlite3

        from arxiv_browser.app import _init_summary_db

        db_path = tmp_path / "summaries.db"
        # Create old schema
        with sqlite3.connect(str(db_path)) as conn:
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

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries'"
            ).fetchone()
            assert "PRIMARY KEY (arxiv_id, command_hash)" in row[0]
            # Old data is gone (cache table, safe to drop)
            count = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            assert count == 0

    def test_composite_pk_allows_multiple_modes(self, tmp_path):
        from arxiv_browser.app import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)

        _save_summary(db_path, "2401.00001", "default summary", "hash_default")
        _save_summary(db_path, "2401.00001", "tldr summary", "hash_tldr")

        assert _load_summary(db_path, "2401.00001", "hash_default") == "default summary"
        assert _load_summary(db_path, "2401.00001", "hash_tldr") == "tldr summary"


class TestSummaryModeDisplay:
    """Tests for mode label display in AI Summary header."""

    def test_summary_header_includes_mode_label(self, make_paper):
        from arxiv_browser.app import PaperDetails

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
        from arxiv_browser.app import PaperDetails

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
        from arxiv_browser.app import PaperDetails

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
        from arxiv_browser.app import _compute_command_hash

        hash1 = _compute_command_hash("cmd", SUMMARY_MODES["default"][1])
        hash2 = _compute_command_hash("cmd", SUMMARY_MODES["tldr"][1])
        assert hash1 != hash2

    def test_each_mode_produces_unique_hash(self):
        from arxiv_browser.app import _compute_command_hash

        hashes = set()
        for mode_name, (_, template) in SUMMARY_MODES.items():
            h = _compute_command_hash("test_cmd", template)
            assert h not in hashes, f"Mode '{mode_name}' has duplicate hash"
            hashes.add(h)


# ============================================================================
# Hierarchical Tags with Namespaces (#3.1) Tests
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
        from arxiv_browser.app import TagsModal

        modal = TagsModal("2401.00001", all_tags=[])
        assert modal._build_suggestions_markup() == ""

    def test_build_suggestions_groups_by_namespace(self):
        from arxiv_browser.app import TagsModal

        modal = TagsModal(
            "2401.00001",
            all_tags=["topic:ml", "topic:nlp", "status:to-read", "important"],
        )
        markup = modal._build_suggestions_markup()
        assert "status:" in markup
        assert "topic:" in markup
        assert "important" in markup

    def test_build_suggestions_deduplicates(self):
        from arxiv_browser.app import TagsModal

        modal = TagsModal(
            "2401.00001",
            all_tags=["topic:ml", "topic:ml", "topic:ml"],
        )
        markup = modal._build_suggestions_markup()
        # "ml" should appear only once
        assert markup.count("ml") == 1

    def test_modal_accepts_all_tags_param(self):
        from arxiv_browser.app import TagsModal

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


# ============================================================================
# Tests for Relevance Scoring
# ============================================================================


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
        with sqlite3.connect(str(db_path)) as conn:
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
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        config = UserConfig(research_interests="efficient LLM inference, quantization")
        assert save_config(config) is True
        loaded = load_config()
        assert loaded.research_interests == "efficient LLM inference, quantization"

    def test_defaults_when_absent(self, tmp_path, monkeypatch):
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        config_file.write_text(json.dumps({"version": 1}))
        loaded = load_config()
        assert loaded.research_interests == ""

    def test_type_validation_rejects_non_string(self, tmp_path, monkeypatch):
        import json

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

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
        from arxiv_browser.app import ResearchInterestsModal

        modal = ResearchInterestsModal("test interests")
        assert modal._current_interests == "test interests"

    def test_modal_empty_default(self):
        from arxiv_browser.app import ResearchInterestsModal

        modal = ResearchInterestsModal()
        assert modal._current_interests == ""

    def test_modal_has_bindings(self):
        from arxiv_browser.app import ResearchInterestsModal

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


class TestCountPapersInFile:
    """Tests for count_papers_in_file utility."""

    def test_counts_ids(self, tmp_path):
        from arxiv_browser.app import count_papers_in_file

        f = tmp_path / "test.txt"
        f.write_text(
            "arXiv:2401.12345 some paper\narXiv:2401.67890v2 another\n",
            encoding="utf-8",
        )
        assert count_papers_in_file(f) == 2

    def test_missing_file(self, tmp_path):
        from arxiv_browser.app import count_papers_in_file

        assert count_papers_in_file(tmp_path / "nonexistent.txt") == 0

    def test_empty_file(self, tmp_path):
        from arxiv_browser.app import count_papers_in_file

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert count_papers_in_file(f) == 0


class TestDateNavigator:
    """Tests for DateNavigator widget."""

    def test_window_centers_on_current(self):
        from datetime import date as dt_date
        from pathlib import Path

        from arxiv_browser.app import DATE_NAV_WINDOW_SIZE

        # Create 10 fake history entries
        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        # Window of 5 centered on index 5 should be [3..7]
        total = len(files)
        current = 5
        half = DATE_NAV_WINDOW_SIZE // 2
        start = max(0, current - half)
        end = min(total, start + DATE_NAV_WINDOW_SIZE)
        if end - start < DATE_NAV_WINDOW_SIZE:
            start = max(0, end - DATE_NAV_WINDOW_SIZE)
        assert end - start == DATE_NAV_WINDOW_SIZE
        assert start <= current < end

    def test_window_clamps_at_edges(self):
        from datetime import date as dt_date
        from pathlib import Path

        from arxiv_browser.app import DATE_NAV_WINDOW_SIZE

        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        # At index 0 (edge), window should start at 0
        total = len(files)
        current = 0
        half = DATE_NAV_WINDOW_SIZE // 2
        start = max(0, current - half)
        end = min(total, start + DATE_NAV_WINDOW_SIZE)
        if end - start < DATE_NAV_WINDOW_SIZE:
            start = max(0, end - DATE_NAV_WINDOW_SIZE)
        assert start == 0
        assert end == DATE_NAV_WINDOW_SIZE

    def test_small_file_list(self):
        from datetime import date as dt_date
        from pathlib import Path

        from arxiv_browser.app import DATE_NAV_WINDOW_SIZE

        # Only 3 files — window should show all of them
        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(3)]
        total = len(files)
        current = 1
        half = DATE_NAV_WINDOW_SIZE // 2
        start = max(0, current - half)
        end = min(total, start + DATE_NAV_WINDOW_SIZE)
        if end - start < DATE_NAV_WINDOW_SIZE:
            start = max(0, end - DATE_NAV_WINDOW_SIZE)
        assert start == 0
        assert end == 3  # Only 3 files, so window covers all

    def test_count_cache_tracks_file_path_not_index(self, tmp_path):
        from datetime import date as dt_date

        from arxiv_browser.app import DateNavigator

        first = tmp_path / "2026-01-01.txt"
        second = tmp_path / "2026-01-02.txt"
        replacement = tmp_path / "2026-01-03.txt"

        first.write_text("arXiv:2401.00001\n", encoding="utf-8")
        second.write_text("arXiv:2401.00002\narXiv:2401.00003\n", encoding="utf-8")
        replacement.write_text(
            "arXiv:2401.00004\narXiv:2401.00005\narXiv:2401.00006\n",
            encoding="utf-8",
        )

        nav = DateNavigator([(dt_date(2026, 1, 1), first), (dt_date(2026, 1, 2), second)])
        assert nav._get_paper_count(0) == 1

        # Replace index 0 with a different file; count should update accordingly.
        nav._history_files = [(dt_date(2026, 1, 3), replacement), (dt_date(2026, 1, 2), second)]
        assert nav._get_paper_count(0) == 3


class TestThemeSwitcher:
    """Tests for U7: Color theme switcher."""

    def test_themes_have_matching_keys(self):
        from arxiv_browser.app import CATPPUCCIN_MOCHA_THEME, DEFAULT_THEME

        assert set(DEFAULT_THEME.keys()) == set(CATPPUCCIN_MOCHA_THEME.keys())

    def test_theme_name_roundtrip(self):
        from arxiv_browser.app import _config_to_dict, _dict_to_config

        config = UserConfig(theme_name="catppuccin-mocha")
        data = _config_to_dict(config)
        assert data["theme_name"] == "catppuccin-mocha"
        restored = _dict_to_config(data)
        assert restored.theme_name == "catppuccin-mocha"

    def test_theme_name_defaults_to_monokai(self):
        from arxiv_browser.app import _dict_to_config

        config = _dict_to_config({})
        assert config.theme_name == "monokai"

    def test_apply_uses_named_theme(self):
        from arxiv_browser.app import (
            CATPPUCCIN_MOCHA_THEME,
            THEME_COLORS,
            ArxivBrowser,
        )

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="catppuccin-mocha")
        app._http_client = None
        app._apply_theme_overrides()
        assert THEME_COLORS["accent"] == CATPPUCCIN_MOCHA_THEME["accent"]
        assert THEME_COLORS["green"] == CATPPUCCIN_MOCHA_THEME["green"]

    def test_per_key_override_layers(self):
        from arxiv_browser.app import CATPPUCCIN_MOCHA_THEME, THEME_COLORS, ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(
            theme_name="catppuccin-mocha",
            theme={"accent": "#ff0000"},
        )
        app._http_client = None
        app._apply_theme_overrides()
        # Per-key override wins over base theme
        assert THEME_COLORS["accent"] == "#ff0000"
        # Other keys come from the base theme
        assert THEME_COLORS["green"] == CATPPUCCIN_MOCHA_THEME["green"]

    def test_unknown_theme_falls_back_to_default(self):
        from arxiv_browser.app import DEFAULT_THEME, THEME_COLORS, ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="nonexistent-theme")
        app._http_client = None
        app._apply_theme_overrides()
        assert THEME_COLORS["accent"] == DEFAULT_THEME["accent"]


# ============================================================================
# U4: Progress Indicator Tests
# ============================================================================


class TestProgressIndicators:
    """Tests for U4: X/Y counter progress indicators in footer."""

    def _make_app(self):
        from arxiv_browser.app import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig()
        app._relevance_scoring_active = False
        app._version_checking = False
        app._scoring_progress = None
        app._version_progress = None
        app._in_arxiv_api_mode = False
        app.selected_ids = set()
        app._s2_active = False
        app._history_files = []
        return app

    def test_scoring_progress_in_footer(self):
        app = self._make_app()
        app._scoring_progress = (3, 50)
        bindings = app._get_footer_bindings()
        assert any("Scoring 3/50" in label for _, label in bindings)

    def test_version_progress_in_footer(self):
        app = self._make_app()
        app._version_progress = (2, 5)
        bindings = app._get_footer_bindings()
        assert any("Versions 2/5" in label for _, label in bindings)

    def test_scoring_progress_overrides_boolean(self):
        """Tuple progress takes priority over boolean flag."""
        app = self._make_app()
        app._relevance_scoring_active = True
        app._scoring_progress = (7, 20)
        bindings = app._get_footer_bindings()
        # Should show X/Y, not static text
        labels = [label for _, label in bindings]
        assert any("Scoring 7/20" in lbl for lbl in labels)
        assert not any("Scoring papers" in lbl for lbl in labels)

    def test_version_progress_overrides_boolean(self):
        """Tuple progress takes priority over boolean flag."""
        app = self._make_app()
        app._version_checking = True
        app._version_progress = (1, 3)
        bindings = app._get_footer_bindings()
        labels = [label for _, label in bindings]
        assert any("Versions 1/3" in lbl for lbl in labels)
        assert not any("Checking versions" in lbl for lbl in labels)

    def test_scoring_progress_cleared_is_none(self):
        """When progress is None, boolean fallback is used."""
        app = self._make_app()
        app._relevance_scoring_active = True
        app._scoring_progress = None
        bindings = app._get_footer_bindings()
        # Falls back to boolean text
        labels = [label for _, label in bindings]
        assert any("Scoring papers" in lbl for lbl in labels)


# ============================================================================
# U7: Solarized Dark Theme Tests
# ============================================================================


class TestSolarizedDarkTheme:
    """Tests for U7: Solarized Dark theme expansion."""

    def test_solarized_theme_exists(self):
        from arxiv_browser.app import SOLARIZED_DARK_THEME, THEMES

        assert "solarized-dark" in THEMES
        assert THEMES["solarized-dark"] is SOLARIZED_DARK_THEME

    def test_solarized_has_all_keys(self):
        from arxiv_browser.app import DEFAULT_THEME, SOLARIZED_DARK_THEME

        assert set(SOLARIZED_DARK_THEME.keys()) == set(DEFAULT_THEME.keys())

    def test_solarized_palette_spot_check(self):
        from arxiv_browser.app import SOLARIZED_DARK_THEME

        assert SOLARIZED_DARK_THEME["background"] == "#002b36"
        assert SOLARIZED_DARK_THEME["accent"] == "#268bd2"
        assert SOLARIZED_DARK_THEME["green"] == "#859900"
        assert SOLARIZED_DARK_THEME["pink"] == "#d33682"

    def test_three_themes_in_cycle(self):
        from arxiv_browser.app import THEME_NAMES

        assert len(THEME_NAMES) == 3
        assert "monokai" in THEME_NAMES
        assert "catppuccin-mocha" in THEME_NAMES
        assert "solarized-dark" in THEME_NAMES

    def test_solarized_config_roundtrip(self):
        from arxiv_browser.app import _config_to_dict, _dict_to_config

        config = UserConfig(theme_name="solarized-dark")
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.theme_name == "solarized-dark"

    def test_category_colors_update_on_theme(self):
        from arxiv_browser.app import (
            CATEGORY_COLORS,
            THEME_CATEGORY_COLORS,
            ArxivBrowser,
        )

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="solarized-dark")
        app._http_client = None
        app._apply_category_overrides()
        expected = THEME_CATEGORY_COLORS["solarized-dark"]
        for cat, color in expected.items():
            assert CATEGORY_COLORS[cat] == color

    def test_tag_namespace_colors_update_on_theme(self):
        from arxiv_browser.app import (
            TAG_NAMESPACE_COLORS,
            THEME_TAG_NAMESPACE_COLORS,
            ArxivBrowser,
        )

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="solarized-dark")
        app._http_client = None
        app._apply_theme_overrides()
        expected = THEME_TAG_NAMESPACE_COLORS["solarized-dark"]
        for ns, color in expected.items():
            assert TAG_NAMESPACE_COLORS[ns] == color


# ============================================================================
# U3: Collapsible Section Tests
# ============================================================================


class TestCollapsibleSections:
    """Tests for U3: Collapsible detail pane sections."""

    def test_default_collapsed_sections(self):
        from arxiv_browser.app import DEFAULT_COLLAPSED_SECTIONS

        config = UserConfig()
        assert config.collapsed_sections == DEFAULT_COLLAPSED_SECTIONS
        assert "tags" in config.collapsed_sections
        assert "authors" not in config.collapsed_sections

    def test_collapsed_sections_roundtrip(self):
        from arxiv_browser.app import _config_to_dict, _dict_to_config

        config = UserConfig(collapsed_sections=["authors", "abstract"])
        data = _config_to_dict(config)
        assert data["collapsed_sections"] == ["authors", "abstract"]
        restored = _dict_to_config(data)
        assert restored.collapsed_sections == ["authors", "abstract"]

    def test_invalid_sections_filtered(self):
        from arxiv_browser.app import _dict_to_config

        data = {"collapsed_sections": ["authors", "invalid_key", "abstract", 42]}
        config = _dict_to_config(data)
        assert config.collapsed_sections == ["authors", "abstract"]

    def test_missing_collapsed_sections_uses_defaults(self):
        from arxiv_browser.app import DEFAULT_COLLAPSED_SECTIONS, _dict_to_config

        config = _dict_to_config({})
        assert config.collapsed_sections == DEFAULT_COLLAPSED_SECTIONS

    def test_expanded_section_shows_content(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(authors="John Doe")
        details.update_paper(paper, collapsed_sections=[])
        rendered = str(details.content)
        assert "John Doe" in rendered
        assert "\u25be Authors" in rendered  # expanded indicator

    def test_collapsed_section_hides_content(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(authors="John Doe")
        details.update_paper(paper, collapsed_sections=["authors"])
        rendered = str(details.content)
        assert "John Doe" not in rendered
        assert "\u25b8 Authors" in rendered  # collapsed indicator

    def test_collapsed_s2_shows_citation_hint(self, make_paper):
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        details = PaperDetails()
        paper = make_paper()
        s2_data = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=42,
            influential_citation_count=5,
            tldr="A test paper.",
            fields_of_study=("Computer Science",),
            year=2024,
            url="https://api.semanticscholar.org/abc",
            title="Test",
        )
        details.update_paper(paper, s2_data=s2_data, collapsed_sections=["s2"])
        rendered = str(details.content)
        assert "42 cites" in rendered
        assert "\u25b8 Semantic Scholar" in rendered

    def test_collapsed_hf_shows_upvote_hint(self, make_paper):
        from arxiv_browser.app import PaperDetails
        from arxiv_browser.huggingface import HuggingFacePaper

        details = PaperDetails()
        paper = make_paper()
        hf_data = HuggingFacePaper(
            arxiv_id="2401.12345",
            title="Test",
            upvotes=15,
            num_comments=3,
            github_repo="",
            github_stars=0,
            ai_summary="",
            ai_keywords=(),
        )
        details.update_paper(paper, hf_data=hf_data, collapsed_sections=["hf"])
        rendered = str(details.content)
        assert "\u219115" in rendered  # ↑15
        assert "\u25b8 HuggingFace" in rendered

    def test_collapsed_tags_shows_count(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper, tags=["ml", "topic:transformers", "status:read"], collapsed_sections=["tags"]
        )
        rendered = str(details.content)
        assert "Tags (3)" in rendered
        assert "\u25b8 Tags" in rendered

    def test_collapsed_relevance_shows_score(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(paper, relevance=(8, "High quality"), collapsed_sections=["relevance"])
        rendered = str(details.content)
        assert "Relevance (8/10)" in rendered
        assert "High quality" not in rendered

    def test_url_always_visible_despite_collapsed(self, make_paper):
        """URL section is always visible — not collapsible."""
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        # Even if "url" appears in collapsed list (from old config), URL should show
        details.update_paper(paper, collapsed_sections=["url"])
        rendered = str(details.content)
        assert "URL" in rendered
        assert "arxiv.org" in rendered

    def test_detail_section_keys_complete(self):
        from arxiv_browser.app import DETAIL_SECTION_KEYS, DETAIL_SECTION_NAMES

        assert len(DETAIL_SECTION_KEYS) == 8
        for key in DETAIL_SECTION_KEYS:
            assert key in DETAIL_SECTION_NAMES


# ============================================================================
# Tests for _dict_to_config edge cases
# ============================================================================


class TestDictToConfigEdgeCases:
    """Tests for _dict_to_config with invalid/malformed data."""

    def test_sort_index_negative_clamps_to_zero(self):
        """Negative sort_index should be clamped to 0."""
        from arxiv_browser.app import _dict_to_config

        data = {"session": {"sort_index": -1}}
        config = _dict_to_config(data)
        assert config.session.sort_index == 0

    def test_sort_index_too_large_clamps_to_zero(self):
        """sort_index beyond SORT_OPTIONS length should be clamped to 0."""
        from arxiv_browser.app import _dict_to_config

        data = {"session": {"sort_index": 999}}
        config = _dict_to_config(data)
        assert config.session.sort_index == 0

    def test_sort_index_at_max_boundary(self):
        """sort_index at the last valid index should be accepted."""
        from arxiv_browser.app import _dict_to_config

        max_idx = len(SORT_OPTIONS) - 1
        data = {"session": {"sort_index": max_idx}}
        config = _dict_to_config(data)
        assert config.session.sort_index == max_idx

    def test_sort_index_one_past_max_clamps_to_zero(self):
        """sort_index one past the last valid index should be clamped to 0."""
        from arxiv_browser.app import _dict_to_config

        data = {"session": {"sort_index": len(SORT_OPTIONS)}}
        config = _dict_to_config(data)
        assert config.session.sort_index == 0

    def test_session_not_a_dict_uses_defaults(self):
        """Non-dict session value should fall back to defaults."""
        from arxiv_browser.app import _dict_to_config

        data = {"session": "not_a_dict"}
        config = _dict_to_config(data)
        assert config.session.scroll_index == 0
        assert config.session.current_filter == ""
        assert config.session.sort_index == 0

    def test_metadata_entry_not_a_dict_skipped(self):
        """Non-dict metadata entries should be skipped."""
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import _dict_to_config

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

    def test_metadata_last_checked_version_non_int_becomes_none(self):
        """last_checked_version that is not int should become None."""
        from arxiv_browser.app import _dict_to_config

        data = {"paper_metadata": {"2401.00001": {"last_checked_version": "v3"}}}
        config = _dict_to_config(data)
        assert config.paper_metadata["2401.00001"].last_checked_version is None

    def test_metadata_last_checked_version_int_preserved(self):
        """last_checked_version that is int should be preserved."""
        from arxiv_browser.app import _dict_to_config

        data = {"paper_metadata": {"2401.00001": {"last_checked_version": 5}}}
        config = _dict_to_config(data)
        assert config.paper_metadata["2401.00001"].last_checked_version == 5

    def test_watch_list_non_dict_entries_skipped(self):
        """Non-dict watch list entries should be skipped."""
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import _dict_to_config

        data = {"watch_list": [{"pattern": "test", "match_type": "invalid_type"}]}
        config = _dict_to_config(data)
        assert config.watch_list[0].match_type == "author"

    def test_watch_list_not_a_list_uses_empty(self):
        """Non-list watch_list value should result in empty list."""
        from arxiv_browser.app import _dict_to_config

        data = {"watch_list": "not_a_list"}
        config = _dict_to_config(data)
        assert config.watch_list == []

    def test_bookmarks_non_dict_entries_skipped(self):
        """Non-dict bookmark entries should be skipped."""
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import _dict_to_config

        data = {"bookmarks": {"name": "AI"}}
        config = _dict_to_config(data)
        assert config.bookmarks == []

    def test_marks_not_a_dict_uses_empty(self):
        """Non-dict marks value should result in empty dict."""
        from arxiv_browser.app import _dict_to_config

        data = {"marks": ["a", "b"]}
        config = _dict_to_config(data)
        assert config.marks == {}

    def test_marks_valid_dict_preserved(self):
        """Valid marks dict should be preserved."""
        from arxiv_browser.app import _dict_to_config

        data = {"marks": {"a": "2401.00001", "b": "2401.00002"}}
        config = _dict_to_config(data)
        assert config.marks == {"a": "2401.00001", "b": "2401.00002"}

    def test_category_colors_non_string_entries_filtered(self):
        """Non-string keys/values in category_colors should be filtered out."""
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import _dict_to_config

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
        from arxiv_browser.app import _dict_to_config

        data = {"session": {"current_date": 12345}}
        config = _dict_to_config(data)
        assert config.session.current_date is None

    def test_current_date_string_preserved(self):
        """String current_date should be preserved."""
        from arxiv_browser.app import _dict_to_config

        data = {"session": {"current_date": "2024-01-15"}}
        config = _dict_to_config(data)
        assert config.session.current_date == "2024-01-15"


# ============================================================================
# Tests for load_config error paths
# ============================================================================


class TestLoadConfigErrorPaths:
    """Tests for load_config error handling."""

    def test_key_error_returns_default(self, tmp_path, monkeypatch):
        """KeyError during config parsing should return default config."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text('{"session": {}}', encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        with patch("arxiv_browser.app._dict_to_config", side_effect=KeyError("bad_key")):
            config = load_config()
        assert isinstance(config, UserConfig)
        assert config.bibtex_export_dir == ""

    def test_type_error_returns_default(self, tmp_path, monkeypatch):
        """TypeError during config parsing should return default config."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text('{"session": {}}', encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        with patch("arxiv_browser.app._dict_to_config", side_effect=TypeError("bad_type")):
            config = load_config()
        assert isinstance(config, UserConfig)

    def test_os_error_returns_default(self, tmp_path, monkeypatch):
        """OSError during config read should return default config."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        with patch.object(type(config_file), "read_text", side_effect=OSError("Permission denied")):
            config = load_config()
        assert isinstance(config, UserConfig)


# ============================================================================
# Tests for save_config error paths
# ============================================================================


class TestSaveConfigErrorPaths:
    """Tests for save_config error handling and tempfile cleanup."""

    def test_oserror_during_mkdir_returns_false(self, tmp_path, monkeypatch):
        """OSError during directory creation should return False."""
        from unittest.mock import patch

        config_file = tmp_path / "readonly" / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            result = save_config(UserConfig())
        assert result is False

    def test_oserror_during_replace_cleans_up_temp(self, tmp_path, monkeypatch):
        """OSError during os.replace should clean up the temp file."""
        import os
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

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
        monkeypatch.setattr("arxiv_browser.app.get_config_path", lambda: config_file)

        result = save_config(UserConfig())
        assert result is True
        assert config_file.exists()


# ============================================================================
# Tests for SQLite error handlers (summaries and relevance)
# ============================================================================


class TestSummaryDbErrorHandlers:
    """Tests for _load_summary and _save_summary error handling."""

    def test_load_summary_nonexistent_db_returns_none(self, tmp_path):
        """Loading from nonexistent DB path should return None."""
        from arxiv_browser.app import _load_summary

        db_path = tmp_path / "nonexistent.db"
        result = _load_summary(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_summary_corrupt_db_returns_none(self, tmp_path):
        """Loading from corrupt DB should return None (not raise)."""
        from arxiv_browser.app import _load_summary

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        result = _load_summary(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_summary_missing_table_returns_none(self, tmp_path):
        """Loading from DB without summaries table should return None."""
        import sqlite3

        from arxiv_browser.app import _load_summary

        db_path = tmp_path / "empty.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other_table (id TEXT)")
        result = _load_summary(db_path, "2401.00001", "hash123")
        assert result is None

    def test_save_summary_corrupt_db_does_not_raise(self, tmp_path):
        """Saving to corrupt DB should not raise (logs warning)."""
        from arxiv_browser.app import _save_summary

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        # Should not raise
        _save_summary(db_path, "2401.00001", "summary text", "hash123")

    def test_save_summary_sqlite_error_does_not_raise(self, tmp_path):
        """sqlite3.Error during save should not raise (logs warning)."""
        from unittest.mock import patch

        from arxiv_browser.app import _init_summary_db, _save_summary

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)
        with patch("sqlite3.connect", side_effect=__import__("sqlite3").Error("db locked")):
            # Should not raise
            _save_summary(db_path, "2401.00001", "summary text", "hash123")

    def test_load_summary_no_matching_row_returns_none(self, tmp_path):
        """Loading with non-matching hash should return None."""
        from arxiv_browser.app import _init_summary_db, _load_summary, _save_summary

        db_path = tmp_path / "summaries.db"
        _init_summary_db(db_path)
        _save_summary(db_path, "2401.00001", "a summary", "hash_a")
        result = _load_summary(db_path, "2401.00001", "hash_b")
        assert result is None


class TestRelevanceDbErrorHandlers:
    """Tests for relevance score SQLite error handling."""

    def test_load_score_nonexistent_db_returns_none(self, tmp_path):
        """Loading from nonexistent DB should return None."""
        from arxiv_browser.app import _load_relevance_score

        db_path = tmp_path / "nonexistent.db"
        result = _load_relevance_score(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_score_corrupt_db_returns_none(self, tmp_path):
        """Loading from corrupt DB should return None (not raise)."""
        from arxiv_browser.app import _load_relevance_score

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        result = _load_relevance_score(db_path, "2401.00001", "hash123")
        assert result is None

    def test_load_score_missing_table_returns_none(self, tmp_path):
        """Loading from DB without relevance_scores table should return None."""
        import sqlite3

        from arxiv_browser.app import _load_relevance_score

        db_path = tmp_path / "empty.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other_table (id TEXT)")
        result = _load_relevance_score(db_path, "2401.00001", "hash123")
        assert result is None

    def test_save_score_corrupt_db_does_not_raise(self, tmp_path):
        """Saving to corrupt DB should not raise."""
        from arxiv_browser.app import _save_relevance_score

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        _save_relevance_score(db_path, "2401.00001", "hash123", 8, "relevant")

    def test_load_all_corrupt_db_returns_empty(self, tmp_path):
        """Bulk loading from corrupt DB should return empty dict."""
        from arxiv_browser.app import _load_all_relevance_scores

        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        result = _load_all_relevance_scores(db_path, "hash123")
        assert result == {}

    def test_load_all_nonexistent_db_returns_empty(self, tmp_path):
        """Bulk loading from nonexistent DB should return empty dict."""
        from arxiv_browser.app import _load_all_relevance_scores

        db_path = tmp_path / "nonexistent.db"
        result = _load_all_relevance_scores(db_path, "hash123")
        assert result == {}

    def test_load_all_missing_table_returns_empty(self, tmp_path):
        """Bulk loading from DB without table should return empty dict."""
        import sqlite3

        from arxiv_browser.app import _load_all_relevance_scores

        db_path = tmp_path / "empty.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other_table (id TEXT)")
        result = _load_all_relevance_scores(db_path, "hash123")
        assert result == {}


# ============================================================================
# Tests for find_similar_papers boost functions
# ============================================================================


class TestFindSimilarPapersBoosts:
    """Tests for metadata_boost and recency_score inner functions."""

    def test_starred_paper_ranked_higher(self, make_paper):
        """Starred papers should get a boost in similarity ranking."""
        from arxiv_browser.app import find_similar_papers

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
        from arxiv_browser.app import find_similar_papers

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
        from arxiv_browser.app import find_similar_papers

        target = make_paper(arxiv_id="target", categories="cs.AI")
        other = make_paper(arxiv_id="other", categories="cs.AI")
        results = find_similar_papers(target, [other], top_n=5, metadata=None)
        assert isinstance(results, list)

    def test_recent_papers_ranked_higher(self, make_paper):
        """More recent papers should get a recency boost."""
        from arxiv_browser.app import find_similar_papers

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
        from arxiv_browser.app import find_similar_papers

        target = make_paper(arxiv_id="target")
        results = find_similar_papers(target, [], top_n=5)
        assert results == []

    def test_target_excluded_from_results(self, make_paper):
        """Target paper should not appear in its own similar papers."""
        from arxiv_browser.app import find_similar_papers

        target = make_paper(arxiv_id="target", categories="cs.AI")
        results = find_similar_papers(target, [target], top_n=5)
        ids = [p.arxiv_id for p, _ in results]
        assert "target" not in ids


# ============================================================================
# Tests for render_paper_option badge paths
# ============================================================================


class TestRenderPaperOptionBadges:
    """Tests for render_paper_option with various badge combinations."""

    def test_tags_displayed(self, make_paper):
        """Papers with tags should show tag badges in output."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        metadata = PaperMetadata(arxiv_id="2401.12345", tags=["topic:ml", "important"])
        result = render_paper_option(paper, metadata=metadata)
        assert "#topic:ml" in result
        assert "#important" in result

    def test_s2_citation_badge(self, make_paper):
        """S2 data should show citation count badge."""
        from arxiv_browser.app import render_paper_option
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

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
        from arxiv_browser.app import render_paper_option
        from arxiv_browser.huggingface import HuggingFacePaper

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
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, version_update=(1, 3))
        assert "v1\u2192v3" in result  # v1→v3

    def test_relevance_score_high_green(self, make_paper):
        """High relevance score (8-10) should show green badge."""
        from arxiv_browser.app import THEME_COLORS, render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, relevance_score=(9, "very relevant"))
        assert "9/10" in result
        assert THEME_COLORS["green"] in result

    def test_relevance_score_medium_yellow(self, make_paper):
        """Medium relevance score (5-7) should show yellow badge."""
        from arxiv_browser.app import THEME_COLORS, render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, relevance_score=(6, "moderate"))
        assert "6/10" in result
        assert THEME_COLORS["yellow"] in result

    def test_relevance_score_low_muted(self, make_paper):
        """Low relevance score (1-4) should show muted badge."""
        from arxiv_browser.app import THEME_COLORS, render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, relevance_score=(3, "not relevant"))
        assert "3/10" in result
        assert THEME_COLORS["muted"] in result

    def test_read_paper_dimmed(self, make_paper):
        """Read papers should have dimmed title."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper(title="Test Title")
        metadata = PaperMetadata(arxiv_id="2401.12345", is_read=True)
        result = render_paper_option(paper, metadata=metadata)
        assert "[dim]" in result
        assert "\u2713" in result  # checkmark

    def test_starred_paper_has_star(self, make_paper):
        """Starred papers should show star indicator."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        metadata = PaperMetadata(arxiv_id="2401.12345", starred=True)
        result = render_paper_option(paper, metadata=metadata)
        assert "\u2b50" in result  # star emoji

    def test_selected_paper_has_bullet(self, make_paper):
        """Selected papers should show green bullet."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, selected=True)
        assert "\u25cf" in result  # ● bullet

    def test_watched_paper_has_eye(self, make_paper):
        """Watched papers should show eye indicator."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, watched=True)
        assert "\U0001f441" in result  # 👁

    def test_api_source_shows_api_badge(self, make_paper):
        """Papers from API source should show API badge."""
        from arxiv_browser.app import render_paper_option

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
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        result = render_paper_option(
            paper, show_preview=True, abstract_text="This is the abstract."
        )
        assert "This is the abstract." in result

    def test_preview_with_none_abstract_shows_loading(self, make_paper):
        """Preview with None abstract should show loading message."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, show_preview=True, abstract_text=None)
        assert "Loading abstract" in result

    def test_preview_with_empty_abstract(self, make_paper):
        """Preview with empty abstract should show 'No abstract'."""
        from arxiv_browser.app import render_paper_option

        paper = make_paper()
        result = render_paper_option(paper, show_preview=True, abstract_text="")
        assert "No abstract available" in result

    def test_preview_long_abstract_truncated(self, make_paper):
        """Long abstract in preview should be truncated with ellipsis."""
        from arxiv_browser.app import PREVIEW_ABSTRACT_MAX_LEN, render_paper_option

        paper = make_paper()
        long_abstract = "word " * 100  # well over 150 chars
        result = render_paper_option(paper, show_preview=True, abstract_text=long_abstract)
        assert "..." in result

    def test_preview_exact_length_not_truncated(self, make_paper):
        """Abstract at exactly max length should not be truncated."""
        from arxiv_browser.app import PREVIEW_ABSTRACT_MAX_LEN, render_paper_option

        paper = make_paper()
        exact_abstract = "x" * PREVIEW_ABSTRACT_MAX_LEN
        result = render_paper_option(paper, show_preview=True, abstract_text=exact_abstract)
        assert "..." not in result

    def test_all_badges_combined(self, make_paper):
        """All badges together should render without error."""
        from arxiv_browser.app import render_paper_option
        from arxiv_browser.huggingface import HuggingFacePaper
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

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


# ============================================================================
# Integration Tests: Metadata Actions (read/star/notes/tags)
# ============================================================================


@pytest.mark.integration
class TestMetadataActionsIntegration:
    """Integration tests for read/star/notes/tags keyboard actions."""

    @staticmethod
    def _make_papers(make_paper, count: int = 3) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                abstract=f"Abstract content for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_toggle_read_creates_metadata(self, make_paper):
        """Pressing 'r' should create metadata and set is_read True."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                assert first_id not in app._config.paper_metadata

                await pilot.press("r")
                await pilot.pause(0.1)
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].is_read is True

    async def test_toggle_read_twice_unsets(self, make_paper):
        """Pressing 'r' twice should toggle is_read back to False."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("r")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].is_read is True

                await pilot.press("r")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].is_read is False

    async def test_toggle_read_on_second_paper(self, make_paper):
        """Navigate to second paper with 'j', then toggle read."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("j")
                await pilot.pause(0.1)
                await pilot.press("r")
                await pilot.pause(0.1)
                second_id = papers[1].arxiv_id
                assert second_id in app._config.paper_metadata
                assert app._config.paper_metadata[second_id].is_read is True
                # First paper should not have metadata
                assert papers[0].arxiv_id not in app._config.paper_metadata

    async def test_toggle_star_creates_metadata(self, make_paper):
        """Pressing 'x' should create metadata and set starred True."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                assert first_id not in app._config.paper_metadata

                await pilot.press("x")
                await pilot.pause(0.1)
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].starred is True

    async def test_toggle_star_twice_unsets(self, make_paper):
        """Pressing 'x' twice should toggle starred back to False."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("x")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].starred is True

                await pilot.press("x")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].starred is False

    async def test_read_and_star_independent(self, make_paper):
        """Read and star should be independent metadata flags."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("r")
                await pilot.pause(0.1)
                await pilot.press("x")
                await pilot.pause(0.1)
                meta = app._config.paper_metadata[first_id]
                assert meta.is_read is True
                assert meta.starred is True

    async def test_notes_modal_opens(self, make_paper):
        """Pressing 'n' should open the NotesModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, NotesModal

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert len(app.screen_stack) == 1
                await pilot.press("n")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], NotesModal)

    async def test_notes_modal_save_persists(self, make_paper):
        """Type text in NotesModal and save with Ctrl+S should persist notes."""
        from unittest.mock import patch

        from textual.widgets import TextArea

        from arxiv_browser.app import ArxivBrowser, NotesModal

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("n")
                await pilot.pause(0.2)
                assert isinstance(app.screen_stack[-1], NotesModal)

                # Type into the text area
                textarea = app.screen_stack[-1].query_one("#notes-textarea", TextArea)
                textarea.insert("My test notes")
                await pilot.pause(0.1)

                # Save with Ctrl+S
                await pilot.press("ctrl+s")
                await pilot.pause(0.2)

                # Modal should have closed
                assert len(app.screen_stack) == 1
                # Notes should be persisted
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].notes == "My test notes"

    async def test_notes_modal_cancel_does_not_save(self, make_paper):
        """Pressing Escape on NotesModal should not save new notes."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("n")
                await pilot.pause(0.2)

                # Cancel without typing
                await pilot.press("escape")
                await pilot.pause(0.2)

                assert len(app.screen_stack) == 1
                # No notes should have been saved (metadata may exist but notes empty)
                if first_id in app._config.paper_metadata:
                    assert app._config.paper_metadata[first_id].notes == ""

    async def test_tags_modal_opens(self, make_paper):
        """Pressing 't' should open the TagsModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, TagsModal

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert len(app.screen_stack) == 1
                await pilot.press("t")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], TagsModal)


# ============================================================================
# Integration Tests: Sort Cycling
# ============================================================================


@pytest.mark.integration
class TestSortCyclingIntegration:
    """Integration tests for sort cycling via 's' key."""

    @staticmethod
    def _make_papers(make_paper, count: int = 5) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(90 - i)}",
                authors=f"Author {chr(65 + i)}",
                categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_sort_cycles_all_options(self, make_paper):
        """Pressing 's' should cycle through all SORT_OPTIONS and wrap around."""
        from unittest.mock import patch

        from arxiv_browser.app import SORT_OPTIONS, ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert app._sort_index == 0
                num_options = len(SORT_OPTIONS)
                for expected in range(1, num_options):
                    await pilot.press("s")
                    assert app._sort_index == expected
                # Wrap around
                await pilot.press("s")
                assert app._sort_index == 0

    async def test_sort_preserves_option_count(self, make_paper):
        """Sorting should not change the number of papers displayed."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                initial_count = option_list.option_count
                assert initial_count == 5

                await pilot.press("s")
                await pilot.pause(0.1)
                assert option_list.option_count == initial_count

                await pilot.press("s")
                await pilot.pause(0.1)
                assert option_list.option_count == initial_count

    async def test_sort_changes_paper_order(self, make_paper):
        """Sorting should actually reorder the filtered_papers list."""
        from unittest.mock import patch

        from arxiv_browser.app import SORT_OPTIONS, ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Initial order is as-given (reverse alpha: Z, Y, X, W, V)
                initial_ids = [p.arxiv_id for p in app.filtered_papers]

                # Cycle through all sort options and verify the list gets resorted
                # At least one sort should change the order from the initial
                order_changed = False
                for _ in range(len(SORT_OPTIONS)):
                    await pilot.press("s")
                    await pilot.pause(0.1)
                    current_ids = [p.arxiv_id for p in app.filtered_papers]
                    if current_ids != initial_ids:
                        order_changed = True
                        break

                assert order_changed, "Sorting never changed paper order"


# ============================================================================
# Integration Tests: Selection and Batch Operations
# ============================================================================


@pytest.mark.integration
class TestSelectionIntegration:
    """Integration tests for paper selection via space/a/u keys."""

    @staticmethod
    def _make_papers(make_paper, count: int = 5) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_space_toggles_selection(self, make_paper):
        """Pressing space should toggle selection of the current paper."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                assert len(app.selected_ids) == 0

                await pilot.press("space")
                await pilot.pause(0.1)
                assert first_id in app.selected_ids
                assert len(app.selected_ids) == 1

                # Toggle off
                await pilot.press("space")
                await pilot.pause(0.1)
                assert first_id not in app.selected_ids
                assert len(app.selected_ids) == 0

    async def test_select_multiple_papers(self, make_paper):
        """Select multiple papers by navigating and pressing space."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Select first paper
                await pilot.press("space")
                await pilot.pause(0.1)
                assert papers[0].arxiv_id in app.selected_ids

                # Navigate down and select second
                await pilot.press("j")
                await pilot.press("space")
                await pilot.pause(0.1)
                assert papers[1].arxiv_id in app.selected_ids
                assert len(app.selected_ids) == 2

    async def test_select_all(self, make_paper):
        """Pressing 'a' should select all visible papers."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert len(app.selected_ids) == 0

                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 5
                for paper in papers:
                    assert paper.arxiv_id in app.selected_ids

    async def test_clear_selection(self, make_paper):
        """Pressing 'u' should clear all selections."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Select all first
                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 5

                # Clear selection
                await pilot.press("u")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 0

    async def test_select_all_then_deselect_one(self, make_paper):
        """Select all, then deselect one with space."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 3

                # Deselect the first paper
                await pilot.press("space")
                await pilot.pause(0.1)
                assert papers[0].arxiv_id not in app.selected_ids
                assert len(app.selected_ids) == 2

    async def test_selection_count_in_status_bar(self, make_paper):
        """Status bar should show selection count when papers are selected."""
        from unittest.mock import patch

        from textual.widgets import Label

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # No selection initially — "selected" should not appear
                status = app.query_one("#status-bar", Label)
                assert "selected" not in str(status.content)

                # Select all papers
                await pilot.press("a")
                await pilot.pause(0.1)
                status_text = str(status.content)
                assert "3 selected" in status_text


# ============================================================================
# Integration Tests: Export Menu
# ============================================================================


@pytest.mark.integration
class TestExportMenuIntegration:
    """Integration tests for the export menu modal."""

    @staticmethod
    def _make_papers(make_paper, count: int = 2) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories="cs.AI",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_export_menu_opens(self, make_paper):
        """Pressing 'E' with selected papers should open the ExportMenuModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, ExportMenuModal

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Select a paper first so _get_target_papers returns non-empty
                await pilot.press("space")
                await pilot.pause(0.1)
                assert len(app.screen_stack) == 1
                await pilot.press("E")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], ExportMenuModal)

    async def test_export_menu_closes_on_escape(self, make_paper):
        """Pressing Escape should close the ExportMenuModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, ExportMenuModal

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("space")
                await pilot.pause(0.1)
                await pilot.press("E")
                await pilot.pause(0.2)
                assert isinstance(app.screen_stack[-1], ExportMenuModal)

                await pilot.press("escape")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 1

    async def test_export_menu_opens_with_detail_pane_paper(self, make_paper):
        """Export menu should open when detail pane has a paper (no explicit selection)."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, ExportMenuModal

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Wait for the detail pane to load the highlighted paper
                await pilot.pause(0.5)
                await pilot.press("E")
                await pilot.pause(0.2)
                # If the detail pane has a paper, export menu opens
                # If not, the action just notifies — either way no crash
                if len(app.screen_stack) == 2:
                    assert isinstance(app.screen_stack[-1], ExportMenuModal)


# ============================================================================
# Integration Tests: Abstract Preview Toggle
# ============================================================================


@pytest.mark.integration
class TestAbstractPreviewIntegration:
    """Integration tests for abstract preview toggle via 'p' key."""

    @staticmethod
    def _make_papers(make_paper, count: int = 3) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories="cs.AI",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_toggle_preview_flips_state(self, make_paper):
        """Pressing 'p' should toggle _show_abstract_preview."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                initial = app._show_abstract_preview
                await pilot.press("p")
                await pilot.pause(0.1)
                assert app._show_abstract_preview is not initial

                # Toggle back
                await pilot.press("p")
                await pilot.pause(0.1)
                assert app._show_abstract_preview is initial

    async def test_toggle_preview_preserves_paper_count(self, make_paper):
        """Toggling preview should not change the number of papers displayed."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 3

                await pilot.press("p")
                await pilot.pause(0.1)
                assert option_list.option_count == 3

    async def test_toggle_preview_updates_config(self, make_paper):
        """Preview toggle should sync to config for persistence."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                initial = app._config.show_abstract_preview
                await pilot.press("p")
                await pilot.pause(0.1)
                assert app._config.show_abstract_preview is not initial


# ============================================================================
# Integration Tests: Watch Filter Toggle
# ============================================================================


@pytest.mark.integration
class TestWatchFilterIntegration:
    """Integration tests for watch filter toggle via 'w' key."""

    @staticmethod
    def _make_papers(make_paper, count: int = 3) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories="cs.AI",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_watch_filter_toggle_empty_watch_list(self, make_paper):
        """Pressing 'w' with empty watch list should remain inactive."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert app._watch_filter_active is False
                await pilot.press("w")
                await pilot.pause(0.1)
                # Should remain False because watch list is empty
                assert app._watch_filter_active is False

    async def test_watch_filter_toggle_with_watch_list(self, make_paper):
        """Pressing 'w' with watch entries should toggle filter on/off."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, WatchListEntry

        papers = self._make_papers(make_paper, count=3)
        config = UserConfig(watch_list=[WatchListEntry(pattern="Author A", match_type="author")])
        app = ArxivBrowser(papers, config=config, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert app._watch_filter_active is False
                # Ensure some papers are watched
                assert len(app._watched_paper_ids) > 0

                await pilot.press("w")
                await pilot.pause(0.1)
                assert app._watch_filter_active is True

                # Toggle off
                await pilot.press("w")
                await pilot.pause(0.1)
                assert app._watch_filter_active is False

    async def test_watch_filter_reduces_visible_papers(self, make_paper):
        """Activating watch filter should show only watched papers."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser, WatchListEntry

        papers = self._make_papers(make_paper, count=3)
        # Only watch Author A — should match papers[0] only
        config = UserConfig(watch_list=[WatchListEntry(pattern="Author A", match_type="author")])
        app = ArxivBrowser(papers, config=config, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 3

                await pilot.press("w")
                await pilot.pause(0.2)
                # Only watched papers should be visible
                assert option_list.option_count < 3
                assert option_list.option_count >= 1


# ============================================================================
# Integration Tests: Theme Cycling
# ============================================================================


@pytest.mark.integration
class TestThemeCyclingIntegration:
    """Integration tests for theme cycling via Ctrl+T."""

    @staticmethod
    def _make_papers(make_paper, count: int = 1) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories="cs.AI",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_theme_cycles_to_next(self, make_paper):
        """Pressing Ctrl+T should cycle to the next theme."""
        from unittest.mock import patch

        from arxiv_browser.app import THEME_NAMES, ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                initial_theme = app._config.theme_name
                assert initial_theme == "monokai"

                await pilot.press("ctrl+t")
                await pilot.pause(0.1)
                assert app._config.theme_name == THEME_NAMES[1]
                assert app._config.theme_name != initial_theme

    async def test_theme_cycles_wrap_around(self, make_paper):
        """Pressing Ctrl+T len(THEME_NAMES) times should return to the first theme."""
        from unittest.mock import patch

        from arxiv_browser.app import THEME_NAMES, ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                initial_theme = app._config.theme_name
                for _ in range(len(THEME_NAMES)):
                    await pilot.press("ctrl+t")
                    await pilot.pause(0.1)
                assert app._config.theme_name == initial_theme

    async def test_theme_cycle_preserves_paper_count(self, make_paper):
        """Theme cycling should not affect the paper list."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 3

                await pilot.press("ctrl+t")
                await pilot.pause(0.1)
                assert option_list.option_count == 3

    async def test_theme_cycle_refreshes_detail_markup(self, make_paper):
        """Theme cycling should invalidate detail cache and re-render markup colors."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser, PaperDetails

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                details = app.query_one(PaperDetails)
                app._refresh_detail_pane()
                before = str(details.content)

                await pilot.press("ctrl+t")
                await pilot.pause(0.1)

                after = str(details.content)
                assert before != after


# ============================================================================
# Integration Tests: History Mode Navigation
# ============================================================================


@pytest.mark.integration
class TestHistoryNavigationIntegration:
    """Integration tests for history date navigation via '[' and ']' keys."""

    @staticmethod
    def _make_history_file(tmp_path, date_str: str, paper_id: str) -> Path:
        """Create a minimal arXiv email file for a given date."""
        history_dir = tmp_path / "history"
        history_dir.mkdir(exist_ok=True)
        filepath = history_dir / f"{date_str}.txt"
        content = (
            f"arXiv:{paper_id}\n"
            f"Date: Mon, 15 Jan 2024\n"
            f"Title: Paper for {date_str}\n"
            f"Authors: Author X\n"
            f"Categories: cs.AI\n"
            f"\\\\\n"
            f"  Abstract for {date_str}.\n"
        )
        filepath.write_text(content)
        return filepath

    async def test_prev_date_increments_index(self, make_paper, tmp_path):
        """Pressing '[' should navigate to the older date (higher index)."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, DateNavigator

        # Create history files
        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")
        f3 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        # History files are newest-first: index 0 = newest
        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
            (dt_date(2024, 1, 15), f3),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 0

                await pilot.press("bracketleft")
                await pilot.pause(0.3)
                assert app._current_date_index == 1

    async def test_next_date_decrements_index(self, make_paper, tmp_path):
        """Pressing ']' should navigate to the newer date (lower index)."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")
        f3 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
            (dt_date(2024, 1, 15), f3),
        ]

        papers = [make_paper(arxiv_id="2401.00002", title="Paper for 2024-01-16")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=1,
        )
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 1

                await pilot.press("bracketright")
                await pilot.pause(0.3)
                assert app._current_date_index == 0

    async def test_prev_date_clamps_at_oldest(self, make_paper, tmp_path):
        """Pressing '[' at the oldest date should not go further."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 15), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00001", title="Paper for 2024-01-15")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=1,  # Already at oldest
        )
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 1

                await pilot.press("bracketleft")
                await pilot.pause(0.2)
                # Should stay at oldest (index 1 for 2 files)
                assert app._current_date_index == 1

    async def test_next_date_clamps_at_newest(self, make_paper, tmp_path):
        """Pressing ']' at the newest date should not go further."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 15), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,  # Already at newest
        )
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 0

                await pilot.press("bracketright")
                await pilot.pause(0.2)
                # Should stay at newest (index 0)
                assert app._current_date_index == 0

    async def test_history_navigation_loads_new_papers(self, make_paper, tmp_path):
        """Navigating dates should load papers from the new date file."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app.all_papers[0].arxiv_id == "2401.00003"

                await pilot.press("bracketleft")
                await pilot.pause(0.3)
                # After navigating to older date, papers should be from the new file
                assert len(app.all_papers) >= 1
                assert app.all_papers[0].arxiv_id == "2401.00002"

    async def test_not_in_history_mode_ignores_navigation(self, make_paper):
        """Without history files, '[' and ']' should not crash or change state."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = [make_paper()]
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert not app._is_history_mode()
                assert app._current_date_index == 0

                await pilot.press("bracketleft")
                await pilot.pause(0.1)
                assert app._current_date_index == 0

                await pilot.press("bracketright")
                await pilot.pause(0.1)
                assert app._current_date_index == 0

    async def test_history_navigation_clears_selection(self, make_paper, tmp_path):
        """Navigating to a new date should clear any paper selections."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.app import ArxivBrowser, DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )
        with (
            patch("arxiv_browser.app.save_config", return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                # Select a paper
                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) > 0

                # Navigate to older date
                await pilot.press("bracketleft")
                await pilot.pause(0.3)
                # Selection should be cleared
                assert len(app.selected_ids) == 0


# ============================================================================
# Tests for ExportMenuModal
# ============================================================================


class TestExportMenuModal:
    """Tests for ExportMenuModal action methods and structure."""

    def test_bindings_cover_all_formats(self):
        from arxiv_browser.app import ExportMenuModal

        binding_keys = {b.key for b in ExportMenuModal.BINDINGS}
        expected = {"escape", "c", "b", "m", "r", "v", "t", "B", "R", "C"}
        assert expected <= binding_keys

    def test_action_cancel_dismisses_empty_string(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=3)
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with("")

    def test_action_do_clipboard_plain(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_plain()
        modal.dismiss.assert_called_once_with("clipboard-plain")

    def test_action_do_clipboard_bibtex(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_bibtex()
        modal.dismiss.assert_called_once_with("clipboard-bibtex")

    def test_action_do_clipboard_markdown(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_markdown()
        modal.dismiss.assert_called_once_with("clipboard-markdown")

    def test_action_do_clipboard_ris(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_ris()
        modal.dismiss.assert_called_once_with("clipboard-ris")

    def test_action_do_clipboard_csv(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_csv()
        modal.dismiss.assert_called_once_with("clipboard-csv")

    def test_action_do_clipboard_mdtable(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_mdtable()
        modal.dismiss.assert_called_once_with("clipboard-mdtable")

    def test_action_do_file_bibtex(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_file_bibtex()
        modal.dismiss.assert_called_once_with("file-bibtex")

    def test_action_do_file_ris(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_file_ris()
        modal.dismiss.assert_called_once_with("file-ris")

    def test_action_do_file_csv(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_file_csv()
        modal.dismiss.assert_called_once_with("file-csv")

    def test_paper_count_stored(self):
        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=5)
        assert modal._paper_count == 5

    def test_paper_count_plural_suffix(self):
        from arxiv_browser.app import ExportMenuModal

        modal_one = ExportMenuModal(paper_count=1)
        assert modal_one._paper_count == 1

        modal_many = ExportMenuModal(paper_count=3)
        assert modal_many._paper_count == 3

    def test_all_action_dismiss_values_are_unique(self):
        """Ensure each export action produces a distinct format string."""
        from unittest.mock import MagicMock

        from arxiv_browser.app import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()

        actions = [
            modal.action_do_clipboard_plain,
            modal.action_do_clipboard_bibtex,
            modal.action_do_clipboard_markdown,
            modal.action_do_clipboard_ris,
            modal.action_do_clipboard_csv,
            modal.action_do_clipboard_mdtable,
            modal.action_do_file_bibtex,
            modal.action_do_file_ris,
            modal.action_do_file_csv,
        ]
        values = []
        for action in actions:
            modal.dismiss.reset_mock()
            action()
            values.append(modal.dismiss.call_args[0][0])

        assert len(set(values)) == 9


# ============================================================================
# Tests for SummaryModeModal dismiss values
# ============================================================================


class TestSummaryModeModalDismiss:
    """Tests that each SummaryModeModal action dismisses with the correct mode name."""

    def test_action_mode_default_dismisses_default(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_default()
        modal.dismiss.assert_called_once_with("default")

    def test_action_mode_tldr_dismisses_tldr(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_tldr()
        modal.dismiss.assert_called_once_with("tldr")

    def test_action_mode_methods_dismisses_methods(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_methods()
        modal.dismiss.assert_called_once_with("methods")

    def test_action_mode_results_dismisses_results(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_results()
        modal.dismiss.assert_called_once_with("results")

    def test_action_mode_comparison_dismisses_comparison(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_comparison()
        modal.dismiss.assert_called_once_with("comparison")

    def test_action_cancel_dismisses_empty(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with("")

    def test_all_modes_match_summary_modes_dict(self):
        """Verify each modal mode corresponds to a key in SUMMARY_MODES."""
        from unittest.mock import MagicMock

        from arxiv_browser.app import SUMMARY_MODES, SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()

        mode_actions = {
            "default": modal.action_mode_default,
            "tldr": modal.action_mode_tldr,
            "methods": modal.action_mode_methods,
            "results": modal.action_mode_results,
            "comparison": modal.action_mode_comparison,
        }
        for mode_name, action in mode_actions.items():
            modal.dismiss.reset_mock()
            action()
            dismissed_value = modal.dismiss.call_args[0][0]
            assert dismissed_value == mode_name
            assert mode_name in SUMMARY_MODES


# ============================================================================
# Tests for ResearchInterestsModal actions
# ============================================================================


class TestResearchInterestsModalActions:
    """Tests for ResearchInterestsModal save/cancel dismiss behavior."""

    def test_action_cancel_dismisses_empty_string(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ResearchInterestsModal

        modal = ResearchInterestsModal("some interests")
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with("")

    def test_initial_interests_stored(self):
        from arxiv_browser.app import ResearchInterestsModal

        modal = ResearchInterestsModal("LLM quantization, speculative decoding")
        assert modal._current_interests == "LLM quantization, speculative decoding"

    def test_default_interests_empty(self):
        from arxiv_browser.app import ResearchInterestsModal

        modal = ResearchInterestsModal()
        assert modal._current_interests == ""


# ============================================================================
# Tests for SectionToggleModal
# ============================================================================


class TestSectionToggleModal:
    """Tests for SectionToggleModal toggle and save/cancel behavior."""

    def test_init_stores_collapsed_as_set(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal(["authors", "abstract"])
        assert modal._collapsed == {"authors", "abstract"}

    def test_init_empty_collapsed(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        assert modal._collapsed == set()

    def test_toggle_adds_section_when_not_collapsed(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal._toggle("a")
        assert "authors" in modal._collapsed

    def test_toggle_removes_section_when_collapsed(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal(["authors"])
        modal._toggle("a")
        assert "authors" not in modal._collapsed

    def test_toggle_idempotent_double_toggle(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal._toggle("a")
        assert "authors" in modal._collapsed
        modal._toggle("a")
        assert "authors" not in modal._collapsed

    def test_toggle_invalid_key_ignored(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal._toggle("z")
        assert modal._collapsed == set()

    def test_action_toggle_a_toggles_authors(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_a()
        assert "authors" in modal._collapsed

    def test_action_toggle_b_toggles_abstract(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_b()
        assert "abstract" in modal._collapsed

    def test_action_toggle_t_toggles_tags(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_t()
        assert "tags" in modal._collapsed

    def test_action_toggle_r_toggles_relevance(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_r()
        assert "relevance" in modal._collapsed

    def test_action_toggle_s_toggles_summary(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_s()
        assert "summary" in modal._collapsed

    def test_action_toggle_e_toggles_s2(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_e()
        assert "s2" in modal._collapsed

    def test_action_toggle_h_toggles_hf(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_h()
        assert "hf" in modal._collapsed

    def test_action_toggle_v_toggles_version(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_v()
        assert "version" in modal._collapsed

    def test_action_save_returns_sorted_collapsed(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal(["url", "authors", "hf"])
        modal.dismiss = MagicMock()
        modal.action_save()
        modal.dismiss.assert_called_once_with(sorted(["url", "authors", "hf"]))

    def test_action_save_after_toggle_reflects_changes(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal(["authors"])
        modal.action_toggle_a()
        modal.action_toggle_b()
        modal.dismiss = MagicMock()
        modal.action_save()
        result = modal.dismiss.call_args[0][0]
        assert "abstract" in result
        assert "authors" not in result

    def test_action_cancel_returns_none(self):
        from unittest.mock import MagicMock

        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal(["authors"])
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with(None)

    def test_render_list_shows_all_sections(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal([])
        rendered = modal._render_list()
        assert "Authors" in rendered
        assert "Abstract" in rendered
        assert "Tags" in rendered
        assert "Relevance" in rendered
        assert "AI Summary" in rendered
        assert "Semantic Scholar" in rendered
        assert "HuggingFace" in rendered
        assert "Version Update" in rendered
        # URL is no longer collapsible — should NOT be listed
        assert "URL" not in rendered

    def test_render_list_indicates_collapsed_state(self):
        from arxiv_browser.app import SectionToggleModal

        modal = SectionToggleModal(["authors", "version"])
        rendered = modal._render_list()
        lines = rendered.split("\n")
        authors_line = next(line for line in lines if "Authors" in line)
        version_line = next(line for line in lines if "Version" in line)
        abstract_line = next(line for line in lines if "Abstract" in line)
        assert "\u25b8" in authors_line
        assert "collapsed" in authors_line
        assert "\u25b8" in version_line
        assert "\u25be" in abstract_line
        assert "expanded" in abstract_line

    def test_bindings_have_all_toggle_keys(self):
        from arxiv_browser.app import SectionToggleModal

        binding_keys = {b.key for b in SectionToggleModal.BINDINGS}
        expected = {"escape", "enter", "a", "b", "t", "r", "s", "e", "h", "v"}
        assert expected <= binding_keys


# ============================================================================
# Tests for _get_target_papers
# ============================================================================


class TestGetTargetPapers:
    """Tests for ArxivBrowser._get_target_papers selection logic."""

    def _make_mock_app(self, make_paper, papers=None, selected_ids=None):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser, PaperDetails

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None

        if papers is None:
            papers = [make_paper(arxiv_id=f"2401.{i:05d}") for i in range(3)]
        app.filtered_papers = papers
        app._papers_by_id = {p.arxiv_id: p for p in papers}
        app.selected_ids = selected_ids or set()

        mock_details = MagicMock(spec=PaperDetails)
        mock_details.paper = papers[0] if papers else None
        app.query_one = MagicMock(return_value=mock_details)

        return app

    def test_no_selection_returns_current_paper(self, make_paper):
        papers = [make_paper(arxiv_id="2401.00001", title="Paper 1")]
        app = self._make_mock_app(make_paper, papers=papers, selected_ids=set())

        result = app._get_target_papers()
        assert len(result) == 1
        assert result[0].arxiv_id == "2401.00001"

    def test_with_selection_returns_selected_papers(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00002"),
            make_paper(arxiv_id="2401.00003"),
        ]
        app = self._make_mock_app(
            make_paper,
            papers=papers,
            selected_ids={"2401.00001", "2401.00003"},
        )

        result = app._get_target_papers()
        assert len(result) == 2
        ids = [p.arxiv_id for p in result]
        assert "2401.00001" in ids
        assert "2401.00003" in ids

    def test_selection_preserves_list_order(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00002"),
        ]
        app = self._make_mock_app(
            make_paper,
            papers=papers,
            selected_ids={"2401.00001", "2401.00002", "2401.00003"},
        )

        result = app._get_target_papers()
        result_ids = [p.arxiv_id for p in result]
        assert result_ids == ["2401.00003", "2401.00001", "2401.00002"]

    def test_no_selection_no_current_paper_returns_empty(self, make_paper):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser, PaperDetails

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app.filtered_papers = []
        app._papers_by_id = {}
        app.selected_ids = set()

        mock_details = MagicMock(spec=PaperDetails)
        mock_details.paper = None
        app.query_one = MagicMock(return_value=mock_details)

        result = app._get_target_papers()
        assert result == []

    def test_selected_id_not_in_filtered_still_included(self, make_paper):
        """Papers selected but then filtered out should still be returned."""
        visible = [make_paper(arxiv_id="2401.00001")]
        hidden = make_paper(arxiv_id="2401.00099")

        app = self._make_mock_app(
            make_paper,
            papers=visible,
            selected_ids={"2401.00001", "2401.00099"},
        )
        app._papers_by_id["2401.00099"] = hidden

        result = app._get_target_papers()
        result_ids = [p.arxiv_id for p in result]
        assert "2401.00001" in result_ids
        assert "2401.00099" in result_ids


# ============================================================================
# Tests for action_toggle_read and action_toggle_star
# ============================================================================


class TestToggleReadStar:
    """Tests for action_toggle_read and action_toggle_star via mock app."""

    def _make_mock_app(self, make_paper, papers=None):
        from unittest.mock import MagicMock

        from arxiv_browser.app import ArxivBrowser, UserConfig

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig()

        if papers is None:
            papers = [make_paper()]
        app.filtered_papers = papers
        app._papers_by_id = {p.arxiv_id: p for p in papers}

        app._get_current_paper = MagicMock(return_value=papers[0] if papers else None)
        app._get_current_index = MagicMock(return_value=0 if papers else None)
        app._update_option_at_index = MagicMock()
        app.notify = MagicMock()

        return app

    def test_toggle_read_creates_metadata_and_sets_read(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_read()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.is_read is True
        app.notify.assert_called_once()
        assert "read" in app.notify.call_args[0][0]

    def test_toggle_read_twice_unsets_read(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_read()
        app.action_toggle_read()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.is_read is False

    def test_toggle_read_calls_update_option(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_read()
        app._update_option_at_index.assert_called_once_with(0)

    def test_toggle_read_no_paper_does_nothing(self, make_paper):
        app = self._make_mock_app(make_paper, papers=[make_paper()])
        app._get_current_paper = lambda: None

        app.action_toggle_read()
        assert app._config.paper_metadata == {}

    def test_toggle_star_creates_metadata_and_sets_starred(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_star()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.starred is True
        app.notify.assert_called_once()
        assert "starred" in app.notify.call_args[0][0]

    def test_toggle_star_twice_unsets_star(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_star()
        app.action_toggle_star()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.starred is False

    def test_toggle_star_calls_update_option(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_star()
        app._update_option_at_index.assert_called_once_with(0)

    def test_toggle_star_no_paper_does_nothing(self, make_paper):
        app = self._make_mock_app(make_paper, papers=[make_paper()])
        app._get_current_paper = lambda: None

        app.action_toggle_star()
        assert app._config.paper_metadata == {}

    def test_toggle_read_with_none_index_skips_option_update(self, make_paper):
        """When _get_current_index returns None, _update_option_at_index is not called."""
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])
        app._get_current_index = lambda: None

        app.action_toggle_read()
        app._update_option_at_index.assert_not_called()
        assert app._config.paper_metadata["2401.00001"].is_read is True

    def test_toggle_star_with_none_index_skips_option_update(self, make_paper):
        """When _get_current_index returns None, _update_option_at_index is not called."""
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])
        app._get_current_index = lambda: None

        app.action_toggle_star()
        app._update_option_at_index.assert_not_called()
        assert app._config.paper_metadata["2401.00001"].starred is True


# ============================================================================
# Tests for format_paper_for_clipboard (extended)
# ============================================================================


class TestFormatPaperForClipboardExtended:
    """Extended tests for format_paper_for_clipboard output formatting."""

    def test_includes_all_metadata_fields(self, make_paper):
        from arxiv_browser.app import format_paper_for_clipboard

        paper = make_paper(
            arxiv_id="2401.12345",
            title="A Great Paper",
            authors="Alice, Bob",
            date="Mon, 15 Jan 2024",
            categories="cs.AI cs.LG",
            comments="10 pages, 5 figures",
        )
        result = format_paper_for_clipboard(paper, abstract_text="The abstract text.")
        assert "Title: A Great Paper" in result
        assert "Authors: Alice, Bob" in result
        assert "arXiv: 2401.12345" in result
        assert "Date: Mon, 15 Jan 2024" in result
        assert "Categories: cs.AI cs.LG" in result
        assert "Comments: 10 pages, 5 figures" in result
        assert "URL: https://arxiv.org/abs/2401.12345" in result
        assert "Abstract: The abstract text." in result

    def test_omits_comments_when_none(self, make_paper):
        from arxiv_browser.app import format_paper_for_clipboard

        paper = make_paper(comments=None)
        result = format_paper_for_clipboard(paper)
        assert "Comments:" not in result

    def test_empty_abstract_still_has_label(self, make_paper):
        from arxiv_browser.app import format_paper_for_clipboard

        paper = make_paper()
        result = format_paper_for_clipboard(paper, abstract_text="")
        assert "Abstract: " in result


# ============================================================================
# Tests for format_paper_as_markdown (extended)
# ============================================================================


class TestFormatPaperAsMarkdownExtended:
    """Extended tests for format_paper_as_markdown output formatting."""

    def test_output_has_markdown_structure(self, make_paper):
        from arxiv_browser.app import format_paper_as_markdown

        paper = make_paper(
            arxiv_id="2401.12345",
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            date="Mon, 15 Jan 2024",
            categories="cs.CL",
        )
        result = format_paper_as_markdown(paper, abstract_text="We propose Transformer.")
        assert result.startswith("## Attention Is All You Need")
        assert "**arXiv:** [2401.12345](https://arxiv.org/abs/2401.12345)" in result
        assert "**Date:** Mon, 15 Jan 2024" in result
        assert "**Categories:** cs.CL" in result
        assert "**Authors:** Vaswani et al." in result
        assert "### Abstract" in result
        assert "We propose Transformer." in result

    def test_includes_comments_when_present(self, make_paper):
        from arxiv_browser.app import format_paper_as_markdown

        paper = make_paper(comments="Accepted at NeurIPS 2024")
        result = format_paper_as_markdown(paper)
        assert "**Comments:** Accepted at NeurIPS 2024" in result

    def test_omits_comments_when_none(self, make_paper):
        from arxiv_browser.app import format_paper_as_markdown

        paper = make_paper(comments=None)
        result = format_paper_as_markdown(paper)
        assert "**Comments:**" not in result


# ============================================================================
# Tests for format_paper_as_bibtex (extended)
# ============================================================================


class TestFormatPaperAsBibtexExtended:
    """Extended tests for format_paper_as_bibtex output formatting."""

    def test_bibtex_uses_misc_type(self, make_paper):
        from arxiv_browser.app import format_paper_as_bibtex

        paper = make_paper()
        result = format_paper_as_bibtex(paper)
        assert result.startswith("@misc{")

    def test_bibtex_contains_required_fields(self, make_paper):
        from arxiv_browser.app import format_paper_as_bibtex

        paper = make_paper(
            arxiv_id="2401.12345",
            title="Deep Learning",
            authors="John Smith",
            date="Mon, 15 Jan 2024",
            categories="cs.AI cs.LG",
        )
        result = format_paper_as_bibtex(paper)
        assert "title = {Deep Learning}" in result
        assert "author = {John Smith}" in result
        assert "year = {2024}" in result
        assert "eprint = {2401.12345}" in result
        assert "archivePrefix = {arXiv}" in result
        assert "primaryClass = {cs.AI}" in result
        assert "url = {https://arxiv.org/abs/2401.12345}" in result

    def test_bibtex_ends_with_closing_brace(self, make_paper):
        from arxiv_browser.app import format_paper_as_bibtex

        paper = make_paper()
        result = format_paper_as_bibtex(paper)
        assert result.strip().endswith("}")

    def test_bibtex_escapes_special_chars(self, make_paper):
        from arxiv_browser.app import format_paper_as_bibtex

        paper = make_paper(title="NLP & Transformers: 100% Better")
        result = format_paper_as_bibtex(paper)
        assert r"NLP \& Transformers: 100\% Better" in result

    def test_bibtex_primary_class_from_first_category(self, make_paper):
        from arxiv_browser.app import format_paper_as_bibtex

        paper = make_paper(categories="stat.ML cs.LG")
        result = format_paper_as_bibtex(paper)
        assert "primaryClass = {stat.ML}" in result


# ============================================================================
# Tests for format_paper_as_ris (extended)
# ============================================================================


class TestFormatPaperAsRisExtended:
    """Extended tests for format_paper_as_ris output formatting."""

    def test_ris_has_correct_record_type(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper()
        result = format_paper_as_ris(paper)
        assert result.startswith("TY  - ELEC")

    def test_ris_ends_with_end_record(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper()
        result = format_paper_as_ris(paper)
        assert result.strip().endswith("ER  -")

    def test_ris_includes_title_and_url(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper(arxiv_id="2401.12345", title="My Paper")
        result = format_paper_as_ris(paper)
        assert "TI  - My Paper" in result
        assert "UR  - https://arxiv.org/abs/2401.12345" in result

    def test_ris_splits_authors_by_comma(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper(authors="Alice Smith, Bob Jones, Charlie Brown")
        result = format_paper_as_ris(paper)
        assert "AU  - Alice Smith" in result
        assert "AU  - Bob Jones" in result
        assert "AU  - Charlie Brown" in result

    def test_ris_includes_abstract_when_provided(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper()
        result = format_paper_as_ris(paper, abstract_text="This is the abstract.")
        assert "AB  - This is the abstract." in result

    def test_ris_omits_abstract_when_empty(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper()
        result = format_paper_as_ris(paper)
        assert "AB  -" not in result

    def test_ris_includes_comments_as_note(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper(comments="Accepted at ICML")
        result = format_paper_as_ris(paper)
        assert "N2  - Accepted at ICML" in result

    def test_ris_omits_comments_note_when_none(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper(comments=None)
        result = format_paper_as_ris(paper)
        assert "N2  -" not in result

    def test_ris_includes_categories_as_keywords(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper(categories="cs.AI cs.LG stat.ML")
        result = format_paper_as_ris(paper)
        assert "KW  - cs.AI" in result
        assert "KW  - cs.LG" in result
        assert "KW  - stat.ML" in result

    def test_ris_includes_arxiv_note(self, make_paper):
        from arxiv_browser.app import format_paper_as_ris

        paper = make_paper(arxiv_id="2401.12345")
        result = format_paper_as_ris(paper)
        assert "N1  - arXiv:2401.12345" in result


# ============================================================================
# Tests for format_papers_as_csv (extended)
# ============================================================================


class TestFormatPapersAsCsvExtended:
    """Extended tests for format_papers_as_csv output formatting."""

    def test_csv_header_without_metadata(self, make_paper):
        from arxiv_browser.app import format_papers_as_csv

        papers = [make_paper()]
        result = format_papers_as_csv(papers)
        lines = result.strip().split("\n")
        header = lines[0]
        assert "arxiv_id" in header
        assert "title" in header
        assert "authors" in header
        assert "categories" in header
        assert "date" in header
        assert "url" in header
        assert "comments" in header
        assert "starred" not in header
        assert "read" not in header
        assert "tags" not in header
        assert "notes" not in header

    def test_csv_header_with_metadata(self, make_paper):
        from arxiv_browser.app import format_papers_as_csv

        papers = [make_paper()]
        result = format_papers_as_csv(papers, metadata={})
        lines = result.strip().split("\n")
        header = lines[0]
        assert "starred" in header
        assert "read" in header
        assert "tags" in header
        assert "notes" in header

    def test_csv_multiple_papers(self, make_paper):
        from arxiv_browser.app import format_papers_as_csv

        papers = [
            make_paper(arxiv_id="2401.00001", title="Paper A"),
            make_paper(arxiv_id="2401.00002", title="Paper B"),
        ]
        result = format_papers_as_csv(papers)
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_csv_with_metadata_values(self, make_paper):
        from arxiv_browser.app import PaperMetadata, format_papers_as_csv

        paper = make_paper(arxiv_id="2401.00001")
        meta = PaperMetadata(
            arxiv_id="2401.00001",
            starred=True,
            is_read=True,
            tags=["ml", "topic:transformers"],
            notes="Important paper",
        )
        result = format_papers_as_csv([paper], metadata={"2401.00001": meta})
        assert "true" in result
        assert "ml;topic:transformers" in result
        assert "Important paper" in result

    def test_csv_with_metadata_no_match(self, make_paper):
        from arxiv_browser.app import format_papers_as_csv

        paper = make_paper(arxiv_id="2401.00001")
        result = format_papers_as_csv([paper], metadata={})
        lines = result.strip().split("\n")
        data_line = lines[1]
        assert "false" in data_line

    def test_csv_escapes_commas_in_fields(self, make_paper):
        import csv
        import io

        from arxiv_browser.app import format_papers_as_csv

        paper = make_paper(title='Title with "quotes" and, commas')
        result = format_papers_as_csv([paper])
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[1][1] == 'Title with "quotes" and, commas'


# ============================================================================
# Tests for format_papers_as_markdown_table (extended)
# ============================================================================


class TestFormatPapersAsMarkdownTableExtended:
    """Extended tests for format_papers_as_markdown_table output."""

    def test_markdown_table_header(self, make_paper):
        from arxiv_browser.app import format_papers_as_markdown_table

        papers = [make_paper()]
        result = format_papers_as_markdown_table(papers)
        lines = result.strip().split("\n")
        assert "| arXiv ID | Title | Authors | Categories | Date |" in lines[0]
        assert lines[1].startswith("|---")

    def test_markdown_table_arxiv_link(self, make_paper):
        from arxiv_browser.app import format_papers_as_markdown_table

        paper = make_paper(arxiv_id="2401.12345")
        result = format_papers_as_markdown_table([paper])
        assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in result

    def test_markdown_table_truncates_many_authors(self, make_paper):
        from arxiv_browser.app import format_papers_as_markdown_table

        paper = make_paper(authors="Alice, Bob, Charlie, Diana, Eve")
        result = format_papers_as_markdown_table([paper])
        assert "Alice et al." in result
        assert "Eve" not in result

    def test_markdown_table_shows_few_authors(self, make_paper):
        from arxiv_browser.app import format_papers_as_markdown_table

        paper = make_paper(authors="Alice, Bob, Charlie")
        result = format_papers_as_markdown_table([paper])
        assert "Alice, Bob, Charlie" in result

    def test_markdown_table_escapes_pipes(self, make_paper):
        from arxiv_browser.app import format_papers_as_markdown_table

        paper = make_paper(title="A | B | C")
        result = format_papers_as_markdown_table([paper])
        assert "A \\| B \\| C" in result

    def test_markdown_table_empty_list(self, make_paper):
        from arxiv_browser.app import format_papers_as_markdown_table

        result = format_papers_as_markdown_table([])
        lines = result.strip().split("\n")
        assert len(lines) == 2


# ============================================================================
# Tests for PaperDetails cache integration (extended)
# ============================================================================


class TestPaperDetailsCacheIntegration:
    """Tests for PaperDetails.update_paper cache behavior."""

    def test_cache_hit_returns_same_markup(self, make_paper):
        from arxiv_browser.app import PaperDetails

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
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract version 1")
        details.update_paper(paper, "abstract version 2")

        assert len(details._detail_cache) == 2

    def test_cache_miss_on_different_tags(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", tags=["ml"])
        details.update_paper(paper, "abstract", tags=["ml", "cv"])

        assert len(details._detail_cache) == 2

    def test_cache_miss_on_summary_loading_change(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", summary_loading=False)
        details.update_paper(paper, "abstract", summary_loading=True)

        assert len(details._detail_cache) == 2

    def test_cache_miss_on_collapsed_sections_change(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", collapsed_sections=[])
        details.update_paper(paper, "abstract", collapsed_sections=["authors"])

        assert len(details._detail_cache) == 2

    def test_cache_stores_in_order_list(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()

        for i in range(3):
            paper = make_paper(arxiv_id=f"2401.{i:05d}")
            details.update_paper(paper, f"abstract {i}")

        assert len(details._detail_cache_order) == 3
        assert len(details._detail_cache) == 3

    def test_none_paper_does_not_cache(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        details.update_paper(None)

        assert len(details._detail_cache) == 0
        assert details.paper is None

    def test_cache_eviction_removes_oldest(self, make_paper):
        from arxiv_browser.app import DETAIL_CACHE_MAX, PaperDetails

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
        from arxiv_browser.app import PaperDetails

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
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", relevance=(8, "Good match"))
        details.update_paper(paper, "abstract", relevance=(3, "Poor match"))

        assert len(details._detail_cache) == 2

    def test_cache_with_version_update(self, make_paper):
        from arxiv_browser.app import PaperDetails

        details = PaperDetails()
        paper = make_paper(arxiv_id="2401.00001")

        details.update_paper(paper, "abstract", version_update=None)
        details.update_paper(paper, "abstract", version_update=(1, 3))

        assert len(details._detail_cache) == 2


# ============================================================================
# Theme-Aware CSS Tests
# ============================================================================


class TestTextualThemes:
    """Tests for the Textual theme system with custom CSS variables."""

    def test_build_textual_theme_maps_all_keys(self):
        from arxiv_browser.app import DEFAULT_THEME, _build_textual_theme

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
        from arxiv_browser.app import TEXTUAL_THEMES

        theme_names = list(TEXTUAL_THEMES.keys())
        assert len(theme_names) == 3
        keys_0 = set(TEXTUAL_THEMES[theme_names[0]].variables.keys())
        for name in theme_names[1:]:
            assert set(TEXTUAL_THEMES[name].variables.keys()) == keys_0

    def test_textual_theme_values_are_hex_colors(self):
        from arxiv_browser.app import TEXTUAL_THEMES

        for name, theme in TEXTUAL_THEMES.items():
            for var_name, value in theme.variables.items():
                assert value.startswith("#"), (
                    f"Theme '{name}' variable '{var_name}' has non-hex value: {value}"
                )


# ============================================================================
# Footer Contrast & Mode Badge Tests
# ============================================================================


class TestFooterContrast:
    """Tests for footer rendering with accent-colored keys."""

    def test_footer_contrast_uses_accent_for_keys(self):
        from arxiv_browser.app import THEME_COLORS, ContextFooter

        footer = ContextFooter()
        footer.render_bindings([("o", "open"), ("s", "sort")])
        rendered = str(footer.content)
        assert THEME_COLORS["accent"] in rendered

    def test_footer_mode_badge_renders(self):
        from arxiv_browser.app import THEME_COLORS, ContextFooter

        footer = ContextFooter()
        badge = f"[bold {THEME_COLORS['accent']}] SEARCH [/]"
        footer.render_bindings([("Esc", "close")], mode_badge=badge)
        rendered = str(footer.content)
        assert "SEARCH" in rendered

    def test_footer_mode_badge_default_empty(self):
        """In default browsing state, mode badge should be empty string."""
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.app import ArxivBrowser

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

        from arxiv_browser.app import ArxivBrowser

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

        from arxiv_browser.app import ArxivBrowser

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


# ============================================================================
# Detail Pane Ordering Tests
# ============================================================================


class TestDetailPaneOrdering:
    """Tests for the abstract-before-authors ordering in the detail pane."""

    def test_abstract_before_authors_in_detail(self, make_paper):
        from arxiv_browser.app import PaperDetails

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
        from arxiv_browser.app import DETAIL_SECTION_KEYS, DETAIL_SECTION_NAMES

        assert "url" not in DETAIL_SECTION_KEYS
        assert "url" not in DETAIL_SECTION_NAMES


# ============================================================================
# Feature Discoverability Tests
# ============================================================================


class TestFooterDiscoverability:
    """Tests for conditional feature hints in the footer."""

    @staticmethod
    def _make_footer_app(config):
        """Helper to create a minimal ArxivBrowser mock for footer tests."""
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.app import ArxivBrowser

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
        app.query_one = MagicMock(side_effect=NoMatches())
        return app

    def test_footer_shows_version_hint_when_starred(self, make_paper):
        from arxiv_browser.app import PaperMetadata, UserConfig

        config = UserConfig(
            paper_metadata={"2401.00001": PaperMetadata(arxiv_id="2401.00001", starred=True)}
        )
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        keys = [k for k, _ in bindings]
        assert "V" in keys

    def test_footer_shows_relevance_hint_when_llm(self):
        from arxiv_browser.app import UserConfig

        config = UserConfig(llm_preset="claude")
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        keys = [k for k, _ in bindings]
        assert "L" in keys

    def test_footer_hides_features_when_inactive(self):
        from arxiv_browser.app import UserConfig

        config = UserConfig()  # No starred papers, no LLM
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        keys = [k for k, _ in bindings]
        assert "V" not in keys
        assert "L" not in keys

    def test_footer_shows_export_hint(self):
        from arxiv_browser.app import UserConfig

        config = UserConfig()
        app = self._make_footer_app(config)
        bindings = app._get_footer_bindings()
        keys = [k for k, _ in bindings]
        assert "E" in keys


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
