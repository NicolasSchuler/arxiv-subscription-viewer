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


class TestFormatCategories:
    """Tests for category formatting with colors."""

    def test_single_known_category(self):
        """Known category should get its assigned color."""
        result = format_categories("cs.AI")
        assert "cs.AI" in result
        assert "#fd4d8e" in result  # Monokai pink for cs.AI (WCAG AA adjusted)

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

    def test_parse_arxiv_api_feed_rejects_unsafe_entities(self):
        xml = """<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>&xxe;</title>
    <summary>Unsafe</summary>
  </entry>
</feed>"""
        with pytest.raises(ValueError, match="Invalid arXiv API XML response"):
            parse_arxiv_api_feed(xml)


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
        from tests.support.canonical_exports import FUZZY_SCORE_CUTOFF

        assert 0 <= FUZZY_SCORE_CUTOFF <= 100

    def test_stopwords_contains_common_words(self):
        """STOPWORDS should contain common English stopwords."""
        from tests.support.canonical_exports import STOPWORDS

        assert "the" in STOPWORDS
        assert "and" in STOPWORDS
        assert "or" in STOPWORDS
        assert "is" in STOPWORDS


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

        entry = WatchListEntry(pattern="test")
        assert entry.pattern == "test"
        assert entry.match_type == "author"
        assert entry.case_sensitive is False

    def test_search_bookmark_creation(self):
        """SearchBookmark should store name and query."""

        bookmark = SearchBookmark(name="AI Papers", query="cat:cs.AI")
        assert bookmark.name == "AI Papers"
        assert bookmark.query == "cat:cs.AI"

    def test_session_state_defaults(self):
        """SessionState should have correct defaults."""
        from tests.support.canonical_exports import SessionState

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


class TestConfigPersistence:
    """Tests for configuration save/load functions."""

    def test_config_to_dict_roundtrip(self):
        """Config should serialize and deserialize correctly."""
        from tests.support.canonical_exports import (
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
        from tests.support.canonical_exports import _dict_to_config

        config = _dict_to_config({})
        default = UserConfig()
        assert config.show_abstract_preview == default.show_abstract_preview
        assert config.paper_metadata == default.paper_metadata

    def test_arxiv_api_max_results_is_clamped(self):
        """arxiv_api_max_results should be clamped to configured limits."""
        from tests.support.canonical_exports import ARXIV_API_MAX_RESULTS_LIMIT, _dict_to_config

        config = _dict_to_config({"arxiv_api_max_results": 9999})
        assert config.arxiv_api_max_results == ARXIV_API_MAX_RESULTS_LIMIT
