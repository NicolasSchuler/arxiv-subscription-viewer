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


class TestChatSystemPrompt:
    """Tests for the CHAT_SYSTEM_PROMPT template and PaperChatScreen."""

    def test_chat_prompt_has_required_placeholders(self):
        from arxiv_browser.llm import CHAT_SYSTEM_PROMPT

        for field in ("title", "authors", "categories", "paper_content"):
            assert f"{{{field}}}" in CHAT_SYSTEM_PROMPT

    def test_chat_prompt_formats_correctly(self, make_paper):
        from arxiv_browser.llm import CHAT_SYSTEM_PROMPT

        paper = make_paper(
            title="Test Paper",
            authors="Alice, Bob",
            categories="cs.LG",
            abstract="An abstract.",
        )
        result = CHAT_SYSTEM_PROMPT.format(
            title=paper.title,
            authors=paper.authors,
            categories=paper.categories,
            paper_content="Full paper text here.",
        )
        assert "Test Paper" in result
        assert "Alice, Bob" in result
        assert "cs.LG" in result
        assert "Full paper text here." in result

    def test_chat_screen_init(self, make_paper):
        from unittest.mock import AsyncMock

        from arxiv_browser.modals import PaperChatScreen

        paper = make_paper(title="My Paper")
        provider = AsyncMock()
        screen = PaperChatScreen(paper, provider, "paper content")
        assert screen._paper is paper
        assert screen._provider is provider
        assert screen._paper_content == "paper content"
        assert screen._history == []
        assert screen._waiting is False

    def test_chat_screen_add_message(self, make_paper):
        from unittest.mock import AsyncMock

        from arxiv_browser.modals import PaperChatScreen

        paper = make_paper(title="My Paper")
        screen = PaperChatScreen(paper, AsyncMock())
        # Test message tracking (without DOM — just the history list)
        screen._history.append(("user", "What is this about?"))
        screen._history.append(("assistant", "This paper discusses..."))
        assert len(screen._history) == 2
        assert screen._history[0] == ("user", "What is this about?")
        assert screen._history[1] == ("assistant", "This paper discusses...")

    def test_chat_command_palette_entry(self):
        from arxiv_browser.browser.contracts import COMMAND_PALETTE_COMMANDS

        names = [cmd[0] for cmd in COMMAND_PALETTE_COMMANDS]
        assert "Chat with Paper" in names
        # Verify it maps to the right action
        chat_entry = next(cmd for cmd in COMMAND_PALETTE_COMMANDS if cmd[0] == "Chat with Paper")
        assert chat_entry[2] == "C"  # keybinding
        assert chat_entry[3] == "chat_with_paper"  # action

    def test_chat_context_builds_history(self, make_paper):
        from arxiv_browser.llm import CHAT_SYSTEM_PROMPT

        paper = make_paper(title="Test", authors="A", categories="cs.AI", abstract="Abstract.")
        context = CHAT_SYSTEM_PROMPT.format(
            title=paper.title,
            authors=paper.authors,
            categories=paper.categories,
            paper_content="Full text.",
        )
        # Simulate building chat context with history
        history = [("user", "Q1"), ("assistant", "A1")]
        history_text = ""
        for role, text in history:
            prefix = "User" if role == "user" else "Assistant"
            history_text += f"\n{prefix}: {text}"
        context += f"\n\nConversation so far:{history_text}"
        context += "\n\nUser: Q2\nAssistant:"

        assert "User: Q1" in context
        assert "Assistant: A1" in context
        assert "User: Q2" in context


class TestAskLlm:
    """Tests for PaperChatScreen._ask_llm async method."""

    @pytest.fixture
    def chat_screen(self, make_paper):
        from unittest.mock import AsyncMock, MagicMock

        from textual.css.query import NoMatches

        from arxiv_browser.llm_providers import LLMResult
        from arxiv_browser.modals import PaperChatScreen

        paper = make_paper(
            title="Test Paper",
            authors="Alice",
            categories="cs.AI",
            abstract="An abstract.",
        )
        provider = AsyncMock()
        provider.execute.return_value = LLMResult(output="", success=True)
        screen = PaperChatScreen(paper, provider, "Full paper text.")
        # Mock _add_message since it requires DOM
        screen._add_message = MagicMock()
        # Simulate a question already in history (as on_question_submitted does)
        screen._history.append(("user", "What is this about?"))
        screen._waiting = True
        # Mock query_one for the status update in finally block
        screen.query_one = MagicMock(side_effect=NoMatches())
        return screen

    async def test_success_adds_response(self, chat_screen):
        from arxiv_browser.llm_providers import LLMResult

        chat_screen._provider.execute.return_value = LLMResult(
            output="This paper discusses transformers.", success=True
        )
        await chat_screen._ask_llm("What is this about?")

        # Verify response was added without markup flag (escaped by default)
        chat_screen._add_message.assert_called_once_with(
            "assistant", "This paper discusses transformers."
        )
        assert chat_screen._waiting is False

    async def test_timeout_shows_error(self, chat_screen):
        from arxiv_browser.llm_providers import LLMResult

        chat_screen._provider.execute.return_value = LLMResult(
            output="", success=False, error="Timed out after 120s"
        )
        await chat_screen._ask_llm("question")

        chat_screen._add_message.assert_called_once_with(
            "assistant", "[red]Error: Timed out after 120s[/]", markup=True
        )
        assert chat_screen._waiting is False

    async def test_nonzero_exit_shows_error(self, chat_screen):
        from arxiv_browser.llm_providers import LLMResult

        chat_screen._provider.execute.return_value = LLMResult(
            output="", success=False, error="Exit 1: model not found"
        )
        await chat_screen._ask_llm("question")

        chat_screen._add_message.assert_called_once_with(
            "assistant", "[red]Error: Exit 1: model not found[/]", markup=True
        )
        assert chat_screen._waiting is False

    async def test_empty_output_shows_error(self, chat_screen):
        from arxiv_browser.llm_providers import LLMResult

        chat_screen._provider.execute.return_value = LLMResult(
            output="", success=False, error="Empty output"
        )
        await chat_screen._ask_llm("question")

        chat_screen._add_message.assert_called_once_with(
            "assistant", "[red]Error: Empty output[/]", markup=True
        )

    async def test_exception_logged_and_shown(self, chat_screen):
        chat_screen._provider.execute.side_effect = OSError("command not found")

        await chat_screen._ask_llm("question")

        chat_screen._add_message.assert_called_once_with(
            "assistant", "[red]Error: command not found[/]", markup=True
        )
        assert chat_screen._waiting is False

    async def test_rich_markup_in_response_is_escaped(self, chat_screen):
        """LLM response with brackets should not be interpreted as Rich markup."""
        from arxiv_browser.llm_providers import LLMResult

        response_text = "See [1] and [Section 3] for details"
        chat_screen._provider.execute.return_value = LLMResult(output=response_text, success=True)
        await chat_screen._ask_llm("question")

        # Response is passed WITHOUT markup=True, so _add_message will escape it
        chat_screen._add_message.assert_called_once_with("assistant", response_text)

    async def test_history_included_in_context(self, chat_screen):
        """Conversation history should be sent to the LLM."""
        from arxiv_browser.llm_providers import LLMResult

        # Add prior conversation
        chat_screen._history = [
            ("user", "First question"),
            ("assistant", "First answer"),
            ("user", "Follow up"),
        ]

        chat_screen._provider.execute.return_value = LLMResult(output="response", success=True)
        await chat_screen._ask_llm("Follow up")

        # The provider.execute should have been called with context containing history
        call_args = chat_screen._provider.execute.call_args
        context = call_args[0][0]  # first positional arg is the prompt/context
        assert "First question" in context
        assert "First answer" in context


class TestAddMessageMarkup:
    """Tests for _add_message markup parameter behavior."""

    def test_add_message_escapes_by_default(self, make_paper):
        from unittest.mock import AsyncMock

        from arxiv_browser.modals import PaperChatScreen

        screen = PaperChatScreen(make_paper(title="T"), AsyncMock())
        # Just test the history tracking (no DOM)
        screen._history = []
        assert callable(screen._add_message)
        # The method signature should accept markup kwarg
        import inspect

        sig = inspect.signature(PaperChatScreen._add_message)
        assert "markup" in sig.parameters
        assert sig.parameters["markup"].default is False


class TestPaperDetailsRenderHelpers:
    """Tests for PaperDetails._render_* helper methods."""

    def _make_details(self):
        from arxiv_browser.widgets.details import PaperDetails

        return PaperDetails()

    def test_render_title(self, make_paper):
        details = self._make_details()
        paper = make_paper(title="Test Title")
        result = details._render_title(paper)
        assert "Test Title" in result
        assert "bold" in result

    def test_render_metadata_basic(self, make_paper):
        details = self._make_details()
        paper = make_paper(arxiv_id="2401.00001", date="2024-01-01", categories="cs.AI")
        result = details._render_metadata(paper)
        assert "2401.00001" in result
        assert "2024-01-01" in result
        assert "cs.AI" in result

    def test_render_metadata_with_comments(self, make_paper):
        details = self._make_details()
        paper = make_paper(comments="10 pages, 5 figures")
        result = details._render_metadata(paper)
        assert "10 pages" in result

    def test_render_abstract_collapsed(self, make_paper):
        details = self._make_details()
        result = details._render_abstract("Some text", False, None, True, "full")
        assert "▸ Abstract" in result
        assert "Some text" not in result

    def test_render_abstract_expanded(self, make_paper):
        details = self._make_details()
        result = details._render_abstract("Deep learning is great", False, None, False, "full")
        assert "▾ Abstract" in result
        assert "Deep learning is great" in result

    def test_render_abstract_loading(self, make_paper):
        details = self._make_details()
        result = details._render_abstract("", True, None, False, "full")
        assert "Loading abstract" in result

    def test_render_abstract_empty(self, make_paper):
        details = self._make_details()
        result = details._render_abstract("", False, None, False, "full")
        assert "No abstract available" in result

    def test_render_abstract_scan_truncates(self):
        details = self._make_details()
        abstract = " ".join(["token"] * 120)
        result = details._render_abstract(abstract, False, None, False, "scan")
        assert "▾ Abstract" in result
        assert "..." in result or "\u2026" in result
        assert abstract not in result

    def test_render_authors_collapsed(self, make_paper):
        details = self._make_details()
        paper = make_paper(authors="John Doe")
        result = details._render_authors(paper, True, "full")
        assert "▸ Authors" in result
        assert "John Doe" not in result

    def test_render_authors_expanded(self, make_paper):
        details = self._make_details()
        paper = make_paper(authors="John Doe")
        result = details._render_authors(paper, False, "full")
        assert "▾ Authors" in result
        assert "John Doe" in result

    def test_render_authors_scan_truncates(self, make_paper):
        details = self._make_details()
        paper = make_paper(authors=", ".join([f"Author {i}" for i in range(20)]))
        result = details._render_authors(paper, False, "scan")
        assert "▾ Authors" in result
        assert "..." in result or "\u2026" in result

    def test_render_tags_empty(self):
        details = self._make_details()
        assert details._render_tags(None, False) == ""
        assert details._render_tags([], False) == ""

    def test_render_tags_collapsed(self):
        details = self._make_details()
        result = details._render_tags(["ml", "nlp"], True)
        assert "▸ Tags (2)" in result

    def test_render_tags_expanded(self):
        details = self._make_details()
        result = details._render_tags(["topic:ml", "flat-tag"], False)
        assert "▾ Tags" in result
        assert "topic:" in result
        assert "flat-tag" in result

    def test_render_tags_escapes_markup(self):
        details = self._make_details()
        result = details._render_tags(["topic:[red]ml[/]", "[bold]flat[/]"], False)
        assert "[red]ml[/]" not in result
        assert "[bold]flat[/]" not in result
        assert "\\[red]ml\\[/]" in result
        assert "\\[bold]flat\\[/]" in result

    def test_render_relevance_none(self):
        details = self._make_details()
        assert details._render_relevance(None, False) == ""

    def test_render_relevance_collapsed(self):
        details = self._make_details()
        result = details._render_relevance((8, "Good paper"), True)
        assert "▸ Relevance (\u26058/10)" in result

    def test_render_relevance_high_score(self):
        details = self._make_details()
        result = details._render_relevance((9, "Excellent"), False)
        assert "9/10" in result
        assert "Excellent" in result

    def test_render_summary_empty(self):
        details = self._make_details()
        assert details._render_summary(None, False, "", False) == ""

    def test_render_summary_loading(self):
        details = self._make_details()
        result = details._render_summary(None, True, "tldr", False)
        assert "Generating summary" in result
        assert "tldr" in result

    def test_render_summary_collapsed(self):
        details = self._make_details()
        result = details._render_summary("Some summary", False, "", True)
        assert "▸ AI Summary" in result
        assert "(loaded)" in result

    def test_render_s2_empty(self):
        details = self._make_details()
        assert details._render_s2(None, False, False) == ""

    def test_render_s2_loading(self):
        details = self._make_details()
        result = details._render_s2(None, True, False)
        assert "Fetching data" in result

    def test_render_s2_collapsed_with_data(self):
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        details = self._make_details()
        s2 = SemanticScholarPaper(
            arxiv_id="2401.00001",
            s2_paper_id="abc",
            title="Test",
            citation_count=42,
            influential_citation_count=5,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="",
        )
        result = details._render_s2(s2, False, True)
        assert "42 cites" in result

    def test_render_s2_escapes_fields(self):
        from arxiv_browser.semantic_scholar import SemanticScholarPaper

        details = self._make_details()
        s2 = SemanticScholarPaper(
            arxiv_id="2401.00001",
            s2_paper_id="abc",
            title="Test",
            citation_count=42,
            influential_citation_count=5,
            tldr="",
            fields_of_study=("[red]ML[/]",),
            year=2024,
            url="",
        )
        result = details._render_s2(s2, False, False)
        assert "[red]ML[/]" not in result
        assert "\\[red]ML\\[/]" in result

    def test_render_hf_empty(self):
        details = self._make_details()
        assert details._render_hf(None, False) == ""

    def test_render_hf_collapsed(self):
        from arxiv_browser.huggingface import HuggingFacePaper

        details = self._make_details()
        hf = HuggingFacePaper(
            arxiv_id="2401.00001",
            title="T",
            upvotes=15,
            num_comments=3,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
        result = details._render_hf(hf, True)
        assert "15" in result

    def test_render_hf_escapes_keywords(self):
        from arxiv_browser.huggingface import HuggingFacePaper

        details = self._make_details()
        hf = HuggingFacePaper(
            arxiv_id="2401.00001",
            title="T",
            upvotes=15,
            num_comments=3,
            ai_summary="",
            ai_keywords=("[bold]unsafe[/]",),
            github_repo="",
            github_stars=0,
        )
        result = details._render_hf(hf, False)
        assert "[bold]unsafe[/]" not in result
        assert "\\[bold]unsafe\\[/]" in result

    def test_render_version_none(self, make_paper):
        details = self._make_details()
        paper = make_paper()
        assert details._render_version(paper, None, False) == ""

    def test_render_version_collapsed(self, make_paper):
        details = self._make_details()
        paper = make_paper()
        result = details._render_version(paper, (1, 3), True)
        assert "v1" in result and "v3" in result

    def test_render_version_expanded(self, make_paper):
        details = self._make_details()
        paper = make_paper(arxiv_id="2401.00001")
        result = details._render_version(paper, (1, 3), False)
        assert "arxivdiff.org" in result
        assert "2401.00001" in result

    def test_render_url(self, make_paper):
        details = self._make_details()
        paper = make_paper(url="https://arxiv.org/abs/2401.00001")
        result = details._render_url(paper)
        assert "URL" in result
        assert "arxiv.org" in result

    def test_ascii_mode_uses_ascii_safe_detail_glyphs(self, make_paper):
        from arxiv_browser.huggingface import HuggingFacePaper
        from arxiv_browser.widgets.details import set_ascii_glyphs

        set_ascii_glyphs(True)
        try:
            details = self._make_details()
            paper = make_paper(authors="Jane Doe")

            assert "> Abstract" in details._render_abstract("Some text", False, None, True, "full")
            assert "v Authors" in details._render_authors(paper, False, "full")
            assert "> Relevance (*8/10)" in details._render_relevance((8, "fit"), True)
            assert "v AI Summary" in details._render_summary("Summary text", False, "", False)
            assert "^12" in details._render_hf(
                HuggingFacePaper(
                    arxiv_id="2401.00001",
                    title="T",
                    upvotes=12,
                    num_comments=0,
                    ai_summary="",
                    ai_keywords=(),
                    github_repo="",
                    github_stars=0,
                ),
                True,
            )
            assert "v1->v3" in details._render_version(paper, (1, 3), True)
        finally:
            set_ascii_glyphs(False)
