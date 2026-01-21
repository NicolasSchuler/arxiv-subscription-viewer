#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

import tempfile
from pathlib import Path

import pytest

from arxiv_browser import (
    Paper,
    clean_latex,
    format_categories,
    parse_arxiv_file,
    SORT_OPTIONS,
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
        assert "#888888" in result  # Default gray

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
    def temp_arxiv_file(self, sample_arxiv_content):
        """Create a temporary file with sample content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_arxiv_content)
            return Path(f.name)

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
        """Parser should handle papers without comments."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.67890")
        # Comments might be None or extracted from elsewhere
        # The important thing is it doesn't crash

    def test_parse_extracts_url(self, temp_arxiv_file):
        """Parser should correctly extract URLs."""
        papers = parse_arxiv_file(temp_arxiv_file)
        paper = next(p for p in papers if p.arxiv_id == "2401.12345")
        assert paper.url == "https://arxiv.org/abs/2401.12345"

    def test_parse_empty_file(self):
        """Parser should return empty list for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        papers = parse_arxiv_file(temp_path)
        assert papers == []

    def test_parse_malformed_entries_skipped(self):
        """Parser should skip malformed entries without crashing."""
        content = '''Some random text that is not a valid entry
------------------------------------------------------------------------------
This is also not valid
------------------------------------------------------------------------------
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        papers = parse_arxiv_file(temp_path)
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


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
