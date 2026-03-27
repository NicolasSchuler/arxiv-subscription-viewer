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


class TestStatusFilterRegressions:
    """Regression tests for status bar and filter query handling.

    The three fragile tests that relied on _Static__content and monkey-patched
    query_one have been migrated to TestStatusFilterIntegration (Phase 7).
    Only the source-inspection test remains here.
    """

    def test_help_sections_include_history_and_palette_keys(self):
        """Help content should include key hints from runtime bindings."""
        from tests.support.canonical_exports import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        sections = app._build_help_sections()
        entries = {(key, desc) for _, pairs in sections for key, desc in pairs}
        assert ("Ctrl+e", "Toggle S2 (browse) / Exit API (API mode)") in entries
        assert ("[", "Older") in entries
        assert ("]", "Newer") in entries
        assert ("Ctrl+p", "Commands") in entries
        assert ("Ctrl+k", "Collections") in entries
        assert ("C", "Chat") in entries
        assert ("Ctrl+g", "Auto-Tag") in entries

    def test_command_palette_entries_adapt_to_app_state(self):
        """Command palette labels should reflect the current UI state."""
        from types import SimpleNamespace

        from tests.support.canonical_exports import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._in_arxiv_api_mode = True
        app._hf_active = True
        app._watch_filter_active = True
        app._s2_active = True
        app._s2_cache = {}
        app._detail_mode = "scan"
        app._pending_query = "cat:cs.AI"
        app.filtered_papers = [SimpleNamespace(arxiv_id="2401.00001")]
        app.selected_ids = set()
        app._history_files = []
        app._config = type(
            "Config",
            (),
            {
                "show_abstract_preview": True,
                "watch_list": ["watch"],
                "marks": {},
                "paper_metadata": {},
            },
        )()
        app._get_current_paper = lambda: SimpleNamespace(arxiv_id="2401.00001")

        entries = {command.action: command for command in app._build_command_palette_commands()}

        assert entries["ctrl_e_dispatch"].name == "Exit Search Results"
        assert entries["ctrl_e_dispatch"].description == "Return to your local or history papers"
        assert entries["toggle_hf"].name == "Disable HuggingFace Trending"
        assert entries["toggle_watch_filter"].name == "Show All Papers"
        assert entries["toggle_preview"].name == "Hide Abstract Preview"
        assert entries["toggle_detail_mode"].name == "Switch to Full Details"
        assert entries["fetch_s2"].enabled is True
        assert entries["citation_graph"].enabled is False
        assert entries["citation_graph"].blocked_reason == "S2 data"
        assert entries["generate_summary"].enabled is False
        assert entries["generate_summary"].blocked_reason == "LLM configuration"
        assert entries["chat_with_paper"].blocked_reason == "LLM configuration"
        assert entries["score_relevance"].blocked_reason == "LLM configuration"

    def test_help_sections_include_getting_started_shortcuts(self):
        """Help content should lead with a concise getting-started flow."""
        from tests.support.canonical_exports import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        sections = app._build_help_sections()

        assert sections[0][0] == "Getting Started"
        getting_started_entries = set(sections[0][1])
        assert ("/", "Search papers") in getting_started_entries
        assert ("Space", "Select current paper") in getting_started_entries
        assert ("A", "Search all arXiv") in getting_started_entries
        assert ("E", "Export current or selected papers") in getting_started_entries
        assert ("Ctrl+p", "Open commands") in getting_started_entries
        assert ("[ / ]", "Change dates (history mode)") in getting_started_entries
        assert ("?", "Show full shortcuts") in getting_started_entries

    def test_binding_labels_use_long_form_naming(self):
        """Binding descriptions should use long-form action names."""
        from tests.support.canonical_exports import ArxivBrowser

        by_action = {binding.action: binding for binding in ArxivBrowser.BINDINGS}
        assert by_action["command_palette"].description == "Command palette"
        assert by_action["open_url"].description == "Open in Browser"

    def test_status_bar_compacts_for_narrow_width(self):
        """Status text should switch to compact mode on narrow terminals."""
        from arxiv_browser.widgets.chrome import build_status_bar_text

        status = build_status_bar_text(
            total=83,
            filtered=12,
            query="graph transformers",
            watch_filter_active=False,
            selected_count=5,
            sort_label="relevance",
            in_arxiv_api_mode=True,
            api_page=2,
            arxiv_api_loading=True,
            show_abstract_preview=True,
            s2_active=True,
            s2_loading=False,
            s2_count=11,
            hf_active=True,
            hf_loading=False,
            hf_match_count=4,
            version_checking=False,
            version_update_count=1,
            max_width=80,
        )
        assert " | " in status
        assert "API p2 loading" in status
        assert "sort:relevance" in status
        assert "preview" not in status.lower()
        assert "updated" not in status.lower()
        assert len(status) <= 83  # allow tiny ellipsis overhead

    def test_status_bar_compact_priority_across_width_tiers(self):
        from arxiv_browser.widgets.chrome import build_status_bar_text

        narrow = build_status_bar_text(
            total=120,
            filtered=32,
            query="transformer",
            watch_filter_active=False,
            selected_count=4,
            sort_label="date",
            in_arxiv_api_mode=True,
            api_page=3,
            arxiv_api_loading=True,
            show_abstract_preview=False,
            s2_active=True,
            s2_loading=False,
            s2_count=8,
            hf_active=True,
            hf_loading=False,
            hf_match_count=5,
            version_checking=False,
            version_update_count=0,
            max_width=80,
        )
        medium = build_status_bar_text(
            total=120,
            filtered=32,
            query="transformer",
            watch_filter_active=False,
            selected_count=4,
            sort_label="date",
            in_arxiv_api_mode=True,
            api_page=3,
            arxiv_api_loading=True,
            show_abstract_preview=False,
            s2_active=True,
            s2_loading=False,
            s2_count=8,
            hf_active=True,
            hf_loading=False,
            hf_match_count=5,
            version_checking=False,
            version_update_count=0,
            max_width=96,
        )
        wide = build_status_bar_text(
            total=120,
            filtered=32,
            query="transformer",
            watch_filter_active=False,
            selected_count=4,
            sort_label="date",
            in_arxiv_api_mode=True,
            api_page=3,
            arxiv_api_loading=True,
            show_abstract_preview=False,
            s2_active=True,
            s2_loading=False,
            s2_count=8,
            hf_active=True,
            hf_loading=False,
            hf_match_count=5,
            version_checking=False,
            version_update_count=0,
            max_width=120,
        )
        assert "API p3 loading" in narrow
        assert "4 sel" in narrow
        assert "sort:date" in narrow
        assert "HF:" not in narrow
        assert "HF:5" in medium
        assert "[dim]│[/]" in wide

    def test_empty_state_messages_include_try_guidance(self):
        from tests.support.canonical_exports import build_list_empty_message

        query_msg = build_list_empty_message(
            query="transformer",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )
        api_msg = build_list_empty_message(
            query="",
            in_arxiv_api_mode=True,
            watch_filter_active=False,
            history_mode=False,
        )
        watch_msg = build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=True,
            history_mode=False,
        )
        history_msg = build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=True,
        )

        assert "Try:" in query_msg
        assert "Try:" in api_msg
        assert "Try:" in watch_msg
        assert "Try:" in history_msg
        assert "Next:" in query_msg
        assert "Next:" in api_msg
        assert "Next:" in watch_msg
        assert "Next:" in history_msg
        assert "[bold]][/bold] next page" in api_msg
        assert "[bold][[/bold] previous page" in api_msg
        assert "[bold]Esc[/bold] or [bold]Ctrl+e[/bold]" in api_msg

    def test_actionable_error_template(self):
        from arxiv_browser.action_messages import build_actionable_error

        message = build_actionable_error(
            "run arXiv API search",
            why="rate limit reached (HTTP 429)",
            next_step="retry in a few seconds",
        )
        assert "Could not run arXiv API search." in message
        assert "Why: rate limit reached (HTTP 429)." in message
        assert "Next step: retry in a few seconds." in message

    def test_actionable_warning_template(self):
        from arxiv_browser.action_messages import build_actionable_warning

        message = build_actionable_warning(
            "No active bookmark is selected",
            next_step="press 1-9 to activate a bookmark",
        )
        assert "No active bookmark is selected." in message
        assert "Next step: press 1-9 to activate a bookmark." in message

    def test_actionable_success_template(self):
        from arxiv_browser.action_messages import build_actionable_success

        message = build_actionable_success(
            "HuggingFace trending data loaded",
            detail="3 papers matched your list",
            next_step="press Ctrl+h to hide HF badges",
        )
        assert "HuggingFace trending data loaded." in message
        assert "3 papers matched your list." in message
        assert "Next step: press Ctrl+h to hide HF badges." in message

    def test_actionable_templates_cover_optional_fields_and_punctuation(self):
        from arxiv_browser.action_messages import (
            _ensure_sentence,
            build_actionable_error,
            build_actionable_success,
            build_actionable_warning,
        )

        assert _ensure_sentence("") == ""
        assert _ensure_sentence("Already punctuated!") == "Already punctuated!"
        assert _ensure_sentence("missing punctuation") == "missing punctuation."

        error_message = build_actionable_error(
            "refresh recommendations",
            next_step="press R again",
        )
        assert "Why:" not in error_message
        assert error_message.endswith("Next step: press R again.")

        warning_message = build_actionable_warning(
            "No Semantic Scholar data was found.",
            why="already cached.",
            next_step="retry later",
        )
        assert "No Semantic Scholar data was found." in warning_message
        assert "Why: already cached." in warning_message
        assert "Next step: retry later." in warning_message

        success_message = build_actionable_success("Saved bookmarks", detail=None, next_step=None)
        assert success_message == "Saved bookmarks."


class TestModuleExports:
    """Tests for module public API."""

    def test_root_package_exports_are_importable(self):
        """All items in the root-package ``__all__`` should be importable."""
        import arxiv_browser
        from arxiv_browser import __all__

        for name in __all__:
            assert hasattr(arxiv_browser, name), f"{name} not found in module"

    def test_root_package_compatibility_exports_remain_importable(self):
        """Legacy root-package imports should keep resolving via the app shim."""
        from arxiv_browser import DEFAULT_THEME, highlight_text

        assert DEFAULT_THEME["accent"]
        assert callable(highlight_text)

    def test_main_exports_exist(self):
        """Key exports should be available."""
        from tests.support.canonical_exports import (
            ArxivBrowser,
            SessionState,
            main,
        )

        assert Paper is not None
        assert ArxivBrowser is not None


class TestHighlightText:
    """Tests for highlight_text()."""

    def test_empty_text_returns_empty(self):
        from tests.support.canonical_exports import highlight_text

        assert highlight_text("", ["foo"], "#ff0000") == ""

    def test_empty_terms_returns_escaped(self):
        from tests.support.canonical_exports import highlight_text

        # Rich's escape only escapes recognized markup-like brackets
        result = highlight_text("Hello [bold]text[/bold]", [], "#ff0000")
        assert r"\[bold]" in result

    def test_short_terms_filtered(self):
        from tests.support.canonical_exports import highlight_text

        result = highlight_text("a b c", ["a"], "#ff0000")
        assert "[bold" not in result  # "a" is too short (< 2 chars)

    def test_dedup_terms(self):
        from tests.support.canonical_exports import highlight_text

        result = highlight_text("hello world", ["hello", "HELLO"], "#ff0000")
        assert result.count("[bold") == 1  # Deduped

    def test_case_insensitive_highlight(self):
        from tests.support.canonical_exports import highlight_text

        result = highlight_text("Deep Learning", ["deep"], "#ff0000")
        assert "[bold #ff0000]Deep[/]" in result

    def test_rich_escaping_preserved(self):
        from tests.support.canonical_exports import highlight_text

        result = highlight_text("[bold]text[/bold]", ["text"], "#ff0000")
        assert r"\[bold]" in result


class TestEscapeRichText:
    """Tests for escape_rich_text()."""

    def test_empty_string(self):
        from tests.support.canonical_exports import escape_rich_text

        assert escape_rich_text("") == ""

    def test_normal_text(self):
        from tests.support.canonical_exports import escape_rich_text

        assert escape_rich_text("Hello World") == "Hello World"

    def test_brackets_escaped(self):
        from tests.support.canonical_exports import escape_rich_text

        assert escape_rich_text("[bold]text[/bold]") == r"\[bold]text\[/bold]"


class TestFormatAuthorsBibtex:
    """Tests for format_authors_bibtex()."""

    def test_single_author(self):
        from tests.support.canonical_exports import format_authors_bibtex

        assert format_authors_bibtex("John Smith") == "John Smith"

    def test_special_chars_escaped(self):
        from tests.support.canonical_exports import format_authors_bibtex

        assert format_authors_bibtex("A & B") == r"A \& B"


class TestGetConfigPath:
    """Tests for get_config_path()."""

    def test_returns_path_with_config_json(self):
        from tests.support.canonical_exports import get_config_path

        path = get_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.json"


class TestComputePaperSimilarity:
    """Tests for compute_paper_similarity()."""

    def test_identity_similarity(self, make_paper):
        from tests.support.canonical_exports import compute_paper_similarity

        paper = make_paper()
        assert compute_paper_similarity(paper, paper) == 1.0

    def test_different_papers_less_than_one(self, make_paper):
        from tests.support.canonical_exports import compute_paper_similarity

        p1 = make_paper(arxiv_id="001", categories="cs.AI", authors="Smith")
        p2 = make_paper(arxiv_id="002", categories="quant-ph", authors="Jones")
        assert compute_paper_similarity(p1, p2) < 1.0

    def test_category_weight_dominates(self, make_paper):
        from tests.support.canonical_exports import compute_paper_similarity

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
        from tests.support.canonical_exports import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("Deep learning for natural language processing")
        assert isinstance(tokens, list)
        assert "deep" in tokens
        assert "learning" in tokens
        assert "natural" in tokens
        assert "language" in tokens
        assert "processing" in tokens

    def test_tokenize_preserves_frequency(self):
        from tests.support.canonical_exports import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("transformer transformer transformer")
        assert tokens.count("transformer") == 3

    def test_tokenize_empty(self):
        from tests.support.canonical_exports import _tokenize_for_tfidf

        assert _tokenize_for_tfidf(None) == []
        assert _tokenize_for_tfidf("") == []

    def test_tokenize_min_length(self):
        from tests.support.canonical_exports import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("a bb ccc dddd")
        # Only tokens with 3+ chars matching [a-z][a-z0-9]{2,}
        assert "ccc" in tokens
        assert "dddd" in tokens
        assert "bb" not in tokens

    def test_tokenize_stopwords(self):
        from tests.support.canonical_exports import _tokenize_for_tfidf

        tokens = _tokenize_for_tfidf("this paper presents the method")
        assert "this" not in tokens
        assert "paper" not in tokens  # "paper" is in STOPWORDS
        assert "the" not in tokens
        assert "method" not in tokens  # "method" is in STOPWORDS
        assert "presents" in tokens

    def test_compute_tf_sublinear(self):
        import math

        from tests.support.canonical_exports import _compute_tf

        tf = _compute_tf(["transformer", "transformer", "transformer", "model"])
        assert tf["transformer"] == pytest.approx(1.0 + math.log(3))
        assert tf["model"] == pytest.approx(1.0 + math.log(1))

    def test_build_empty_corpus(self):
        from tests.support.canonical_exports import TfidfIndex

        index = TfidfIndex.build([], text_fn=lambda p: p.title)
        assert len(index) == 0

    def test_build_single_paper(self, make_paper):
        from tests.support.canonical_exports import TfidfIndex

        papers = [make_paper(title="Attention mechanisms in deep learning")]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        # n < 2 guard: single paper cannot compute meaningful IDF
        assert len(index) == 0

    def test_build_two_papers(self, make_paper):
        from tests.support.canonical_exports import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Attention mechanisms in deep learning"),
            make_paper(arxiv_id="002", title="Reinforcement learning for robotics"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert len(index) == 2
        assert "001" in index
        assert "002" in index

    def test_cosine_self_is_one(self, make_paper):
        from tests.support.canonical_exports import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Attention mechanisms in deep learning"),
            make_paper(arxiv_id="002", title="Reinforcement learning for robotics"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert index.cosine_similarity("001", "001") == pytest.approx(1.0)

    def test_similar_scores_higher(self, make_paper):
        from tests.support.canonical_exports import TfidfIndex

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
        from tests.support.canonical_exports import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Deep learning methods"),
            make_paper(arxiv_id="002", title="Reinforcement learning"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert index.cosine_similarity("001", "unknown") == 0.0
        assert index.cosine_similarity("unknown", "001") == 0.0

    def test_idf_downweights_common(self, make_paper):
        from tests.support.canonical_exports import TfidfIndex

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
        from tests.support.canonical_exports import TfidfIndex

        papers = [
            make_paper(arxiv_id="001", title="Testing containment protocol"),
            make_paper(arxiv_id="002", title="Another paper here"),
        ]
        index = TfidfIndex.build(papers, text_fn=lambda p: p.title)
        assert "001" in index
        assert "nonexistent" not in index

    def test_compute_similarity_with_tfidf(self, make_paper):
        from tests.support.canonical_exports import (
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
        from tests.support.canonical_exports import compute_paper_similarity

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
        from tests.support.canonical_exports import is_advanced_query

        tokens = [QueryToken(kind="term", value="attention")]
        assert is_advanced_query(tokens) is False

    def test_operator_is_advanced(self):
        from tests.support.canonical_exports import is_advanced_query

        tokens = [
            QueryToken(kind="term", value="a"),
            QueryToken(kind="op", value="AND"),
            QueryToken(kind="term", value="b"),
        ]
        assert is_advanced_query(tokens) is True

    def test_field_prefix_is_advanced(self):
        from tests.support.canonical_exports import is_advanced_query

        tokens = [QueryToken(kind="term", value="cs.AI", field="cat")]
        assert is_advanced_query(tokens) is True

    def test_quoted_phrase_is_advanced(self):
        from tests.support.canonical_exports import is_advanced_query

        tokens = [QueryToken(kind="term", value="deep learning", phrase=True)]
        assert is_advanced_query(tokens) is True

    def test_unread_virtual_term_is_advanced(self):
        from tests.support.canonical_exports import is_advanced_query

        tokens = [QueryToken(kind="term", value="unread")]
        assert is_advanced_query(tokens) is True

    def test_starred_virtual_term_is_advanced(self):
        from tests.support.canonical_exports import is_advanced_query

        tokens = [QueryToken(kind="term", value="starred")]
        assert is_advanced_query(tokens) is True


class TestMatchQueryTerm:
    """Tests for match_query_term()."""

    def test_empty_value_matches_all(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="   ")
        assert match_query_term(paper, token, None) is True

    def test_cat_field_matches(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper(categories="cs.AI cs.LG")
        token = QueryToken(kind="term", value="cs.AI", field="cat")
        assert match_query_term(paper, token, None) is True

    def test_cat_field_no_match(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper(categories="cs.AI")
        token = QueryToken(kind="term", value="cs.CV", field="cat")
        assert match_query_term(paper, token, None) is False

    def test_tag_field_matches(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper()
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["important", "to-read"])
        token = QueryToken(kind="term", value="important", field="tag")
        assert match_query_term(paper, token, meta) is True

    def test_tag_field_no_metadata(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="important", field="tag")
        assert match_query_term(paper, token, None) is False

    def test_title_field_matches(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper(title="Deep Learning for NLP")
        token = QueryToken(kind="term", value="Deep", field="title")
        assert match_query_term(paper, token, None) is True

    def test_author_field_matches(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper(authors="John Smith")
        token = QueryToken(kind="term", value="smith", field="author")
        assert match_query_term(paper, token, None) is True

    def test_abstract_field_matches(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="test abstract", field="abstract")
        assert match_query_term(paper, token, None, abstract_text="Test abstract content.") is True

    def test_unread_virtual_term(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="unread")
        # No metadata = unread
        assert match_query_term(paper, token, None) is True
        # Read = not unread
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, is_read=True)
        assert match_query_term(paper, token, meta) is False

    def test_starred_virtual_term(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper()
        token = QueryToken(kind="term", value="starred")
        # No metadata = not starred
        assert match_query_term(paper, token, None) is False
        # Starred = starred
        meta = PaperMetadata(arxiv_id=paper.arxiv_id, starred=True)
        assert match_query_term(paper, token, meta) is True

    def test_fallback_search_title_and_authors(self, make_paper):
        from tests.support.canonical_exports import match_query_term

        paper = make_paper(title="Attention Mechanism", authors="Jane Doe")
        token = QueryToken(kind="term", value="attention")
        assert match_query_term(paper, token, None) is True


class TestMatchesAdvancedQuery:
    """Tests for matches_advanced_query()."""

    def test_empty_rpn_matches_all(self, make_paper):
        from tests.support.canonical_exports import matches_advanced_query

        paper = make_paper()
        assert matches_advanced_query(paper, [], None) is True

    def test_single_term(self, make_paper):
        from tests.support.canonical_exports import matches_advanced_query

        paper = make_paper(categories="cs.AI")
        rpn = [QueryToken(kind="term", value="cs.AI", field="cat")]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_and_query(self, make_paper):
        from tests.support.canonical_exports import matches_advanced_query

        paper = make_paper(title="Deep Learning for NLP")
        rpn = [
            QueryToken(kind="term", value="deep"),
            QueryToken(kind="term", value="nlp"),
            QueryToken(kind="op", value="AND"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_or_query(self, make_paper):
        from tests.support.canonical_exports import matches_advanced_query

        paper = make_paper(title="Deep Learning")
        rpn = [
            QueryToken(kind="term", value="quantum"),
            QueryToken(kind="term", value="deep"),
            QueryToken(kind="op", value="OR"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True

    def test_not_query(self, make_paper):
        from tests.support.canonical_exports import matches_advanced_query

        paper = make_paper(title="Deep Learning")
        rpn = [
            QueryToken(kind="term", value="quantum"),
            QueryToken(kind="op", value="NOT"),
        ]
        assert matches_advanced_query(paper, rpn, None) is True


class TestPaperMatchesWatchEntry:
    """Tests for paper_matches_watch_entry()."""

    def test_author_match(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper(authors="John Smith, Jane Doe")
        entry = WatchListEntry(pattern="Smith", match_type="author")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_author_no_match(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper(authors="John Smith")
        entry = WatchListEntry(pattern="Wilson", match_type="author")
        assert paper_matches_watch_entry(paper, entry) is False

    def test_title_match(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper(title="Deep Learning for NLP")
        entry = WatchListEntry(pattern="Deep Learning", match_type="title")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_keyword_match_in_title(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper(title="Transformer Architecture", abstract_raw="Some abstract")
        entry = WatchListEntry(pattern="transformer", match_type="keyword")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_keyword_match_in_abstract(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper(title="Some Title", abstract_raw="attention mechanism")
        entry = WatchListEntry(pattern="attention", match_type="keyword")
        assert paper_matches_watch_entry(paper, entry) is True

    def test_case_sensitive(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper(authors="john smith")
        entry = WatchListEntry(pattern="John", match_type="author", case_sensitive=True)
        assert paper_matches_watch_entry(paper, entry) is False

    def test_unknown_match_type(self, make_paper):
        from tests.support.canonical_exports import paper_matches_watch_entry

        paper = make_paper()
        entry = WatchListEntry(pattern="test", match_type="unknown")
        assert paper_matches_watch_entry(paper, entry) is False


class TestSortPapers:
    """Tests for sort_papers()."""

    def test_sort_by_title(self, make_paper):
        from tests.support.canonical_exports import sort_papers

        papers = [
            make_paper(title="Zebra"),
            make_paper(title="Apple"),
            make_paper(title="Mango"),
        ]
        result = sort_papers(papers, "title")
        assert [p.title for p in result] == ["Apple", "Mango", "Zebra"]

    def test_sort_by_date_descending(self, make_paper):
        from tests.support.canonical_exports import sort_papers

        papers = [
            make_paper(date="Mon, 1 Jan 2024"),
            make_paper(date="Wed, 15 Jan 2024"),
            make_paper(date="Tue, 10 Jan 2024"),
        ]
        result = sort_papers(papers, "date")
        assert result[0].date == "Wed, 15 Jan 2024"
        assert result[-1].date == "Mon, 1 Jan 2024"

    def test_sort_by_arxiv_id_descending(self, make_paper):
        from tests.support.canonical_exports import sort_papers

        papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00002"),
        ]
        result = sort_papers(papers, "arxiv_id")
        assert [p.arxiv_id for p in result] == ["2401.00003", "2401.00002", "2401.00001"]

    def test_sort_does_not_mutate_original(self, make_paper):
        from tests.support.canonical_exports import sort_papers

        papers = [make_paper(title="B"), make_paper(title="A")]
        original_order = [p.title for p in papers]
        sort_papers(papers, "title")
        assert [p.title for p in papers] == original_order


class TestFormatPaperForClipboard:
    """Tests for format_paper_for_clipboard()."""

    def test_basic_format(self, make_paper):
        from tests.support.canonical_exports import format_paper_for_clipboard

        paper = make_paper(title="Test Paper", authors="Author", arxiv_id="2401.12345")
        result = format_paper_for_clipboard(paper, abstract_text="Some abstract")
        assert "Title: Test Paper" in result
        assert "Authors: Author" in result
        assert "Abstract: Some abstract" in result

    def test_includes_comments(self, make_paper):
        from tests.support.canonical_exports import format_paper_for_clipboard

        paper = make_paper(comments="10 pages, 5 figures")
        result = format_paper_for_clipboard(paper)
        assert "Comments: 10 pages, 5 figures" in result

    def test_omits_none_comments(self, make_paper):
        from tests.support.canonical_exports import format_paper_for_clipboard

        paper = make_paper(comments=None)
        result = format_paper_for_clipboard(paper)
        assert "Comments:" not in result


class TestFormatPaperAsMarkdown:
    """Tests for format_paper_as_markdown()."""

    def test_headers_and_sections(self, make_paper):
        from tests.support.canonical_exports import format_paper_as_markdown

        paper = make_paper(title="Test Paper", authors="Author")
        result = format_paper_as_markdown(paper, abstract_text="Some abstract")
        assert "## Test Paper" in result
        assert "### Abstract" in result
        assert "**Authors:** Author" in result

    def test_arxiv_link_format(self, make_paper):
        from tests.support.canonical_exports import format_paper_as_markdown

        paper = make_paper(arxiv_id="2401.12345")
        result = format_paper_as_markdown(paper)
        assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in result


class TestGetPdfUrl:
    """Tests for get_pdf_url()."""

    def test_standard_abs_url(self, make_paper):
        from tests.support.canonical_exports import get_pdf_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345", arxiv_id="2401.12345")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_already_pdf_url(self, make_paper):
        from tests.support.canonical_exports import get_pdf_url

        paper = make_paper(url="https://arxiv.org/pdf/2401.12345.pdf")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_pdf_url_without_extension(self, make_paper):
        from tests.support.canonical_exports import get_pdf_url

        paper = make_paper(url="https://arxiv.org/pdf/2401.12345")
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2401.12345.pdf"


class TestGetPaperUrl:
    """Tests for get_paper_url()."""

    def test_default_abs_url(self, make_paper):
        from tests.support.canonical_exports import get_paper_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345")
        assert get_paper_url(paper) == "https://arxiv.org/abs/2401.12345"

    def test_prefer_pdf(self, make_paper):
        from tests.support.canonical_exports import get_paper_url

        paper = make_paper(url="https://arxiv.org/abs/2401.12345", arxiv_id="2401.12345")
        result = get_paper_url(paper, prefer_pdf=True)
        assert "pdf" in result


class TestBuildHighlightTerms:
    """Tests for build_highlight_terms()."""

    def test_title_field(self):
        from tests.support.canonical_exports import build_highlight_terms

        tokens = [QueryToken(kind="term", value="deep", field="title")]
        result = build_highlight_terms(tokens)
        assert "deep" in result["title"]
        assert result["author"] == []

    def test_unfielded_goes_to_title_and_author(self):
        from tests.support.canonical_exports import build_highlight_terms

        tokens = [QueryToken(kind="term", value="smith")]
        result = build_highlight_terms(tokens)
        assert "smith" in result["title"]
        assert "smith" in result["author"]

    def test_operators_skipped(self):
        from tests.support.canonical_exports import build_highlight_terms

        tokens = [QueryToken(kind="op", value="AND")]
        result = build_highlight_terms(tokens)
        assert all(v == [] for v in result.values())

    def test_virtual_terms_skipped(self):
        from tests.support.canonical_exports import build_highlight_terms

        tokens = [QueryToken(kind="term", value="unread")]
        result = build_highlight_terms(tokens)
        assert all(v == [] for v in result.values())
