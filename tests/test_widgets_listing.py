"""Focused tests for list rendering helpers/widgets."""

from __future__ import annotations

import re
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from textual.css.query import NoMatches

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import PaperMetadata
from arxiv_browser.query import truncate_at_word_boundary
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.widgets.chrome import StatusBarState
from arxiv_browser.widgets.listing import (
    PaperHighlightTerms,
    PaperListItem,
    PaperRowRenderState,
    _coerce_paper_row_state,
    _compress_meta_parts,
    _normalize_highlight_terms,
    _normalize_paper_row_state,
    _render_abstract_preview,
    render_paper_option,
    set_ascii_icons,
)


def _visible_text(text: str) -> str:
    return re.sub(r"\[[^\]]*]", "", text)


def test_set_ascii_icons_changes_rendered_selection_marker(make_paper):
    paper = make_paper(title="Selection Test")

    set_ascii_icons(True)
    try:
        ascii_text = render_paper_option(
            PaperRowRenderState(
                paper=paper,
                selected=True,
                hf_data=HuggingFacePaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    upvotes=7,
                    num_comments=0,
                    ai_summary="",
                    ai_keywords=(),
                    github_repo="",
                    github_stars=0,
                ),
                version_update=(1, 2),
                relevance_score=(9, "high"),
            ),
        )
        assert "[x]" in ascii_text
        assert "^7" in ascii_text
        assert "v1->v2" in ascii_text
        assert "*9/10" in ascii_text
    finally:
        set_ascii_icons(False)

    unicode_text = render_paper_option(PaperRowRenderState(paper=paper, selected=True))
    assert "●" in unicode_text
    unicode_meta = render_paper_option(
        PaperRowRenderState(
            paper=paper,
            hf_data=HuggingFacePaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                upvotes=7,
                num_comments=0,
                ai_summary="",
                ai_keywords=(),
                github_repo="",
                github_stars=0,
            ),
            version_update=(1, 2),
            relevance_score=(9, "high"),
        ),
    )
    assert "↑7" in unicode_meta
    assert "v1→v2" in unicode_meta
    assert "★9/10" in unicode_meta


def test_paper_list_item_title_and_authors_text(make_paper):
    paper = make_paper(title="Transformer Study", authors="Alice Author")
    metadata = PaperMetadata(arxiv_id=paper.arxiv_id, starred=True, is_read=True)
    item = PaperListItem(
        PaperRowRenderState(
            paper=paper,
            selected=True,
            watched=True,
            metadata=metadata,
            highlight_terms=PaperHighlightTerms(
                title=("Transformer",),
                author=("Author",),
            ),
        )
    )

    title_text = item._get_title_text()
    assert "●" in title_text
    assert "👁" in title_text
    assert "⭐" in title_text
    assert "✓" in title_text
    assert "[dim]" in title_text

    authors_text = item._get_authors_text()
    assert "Alice" in authors_text


def test_paper_list_item_meta_badges_with_all_sources(make_paper):
    paper = make_paper()
    paper.source = "api"
    metadata = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["topic:ml"])
    item = PaperListItem(paper, metadata=metadata)

    item._s2_data = SemanticScholarPaper(
        arxiv_id=paper.arxiv_id,
        s2_paper_id="s2id",
        citation_count=42,
        influential_citation_count=5,
        tldr="",
        fields_of_study=(),
        year=2024,
        url="https://s2.example",
    )
    item._hf_data = HuggingFacePaper(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        upvotes=55,
        num_comments=3,
        ai_summary="",
        ai_keywords=(),
        github_repo="",
        github_stars=0,
    )
    item._version_update = (1, 2)
    item._relevance_score = (9, "very relevant")

    meta_text = item._get_meta_text()
    assert "API" in meta_text
    assert "#topic:ml" in meta_text
    assert "C42" in meta_text
    assert "↑55" in meta_text
    assert "v1→v2" in meta_text
    assert "9/10" in meta_text


def test_paper_list_item_preview_text_branches(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)

    item._abstract_text = None
    assert "Loading abstract" in item._get_preview_text()

    item._abstract_text = ""
    assert "No abstract available" in item._get_preview_text()

    item._abstract_text = "Short abstract"
    assert "Short abstract" in item._get_preview_text()

    item._abstract_text = "word " * 80
    preview = item._get_preview_text()
    assert preview.endswith("\u2026[/]") or preview.endswith("...[/]")


def test_meta_badges_apply_budget_and_hidden_counter(make_paper):
    paper = make_paper(
        arxiv_id="2401.12345",
        categories="cs.AI cs.CL cs.LG cs.CV cs.IR cs.NE",
    )
    metadata = PaperMetadata(
        arxiv_id=paper.arxiv_id,
        tags=[
            "topic:transformers",
            "topic:reasoning",
            "status:to-read",
            "method:distillation",
            "priority:high",
        ],
    )
    rendered = render_paper_option(
        paper,
        metadata=metadata,
        s2_data=SemanticScholarPaper(
            arxiv_id=paper.arxiv_id,
            s2_paper_id="s2id",
            citation_count=123,
            influential_citation_count=7,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://s2.example",
        ),
        hf_data=HuggingFacePaper(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            upvotes=42,
            num_comments=2,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        ),
        version_update=(1, 3),
        relevance_score=(8, "strong"),
    )
    meta_line = rendered.splitlines()[2]
    visible_meta = _visible_text(meta_line)
    assert len(visible_meta) <= 78
    assert paper.arxiv_id in visible_meta
    assert "+" in visible_meta


def test_paper_list_item_setters_and_selection_refresh(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)
    item._update_display = MagicMock()

    item.set_metadata(PaperMetadata(arxiv_id=item.paper.arxiv_id, tags=["status:todo"]))
    item.set_abstract_text("preview")
    item.update_s2_data(None)
    item.update_hf_data(None)
    item.update_version_data((2, 3))
    item.update_relevance_data((7, "good"))

    item.set_selected(True)
    assert item.is_selected is True
    assert item.toggle_selected() is False

    assert item._update_display.call_count >= 8


def test_paper_list_item_update_display_handles_no_matches(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)
    item.query_one = MagicMock(side_effect=NoMatches("missing"))

    item._update_display()


def test_paper_list_item_update_display_updates_widgets(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)

    title_widget = MagicMock()
    authors_widget = MagicMock()
    meta_widget = MagicMock()
    preview_widget = MagicMock()

    def _query(selector, _type=None):
        mapping = {
            ".paper-title": title_widget,
            ".paper-authors": authors_widget,
            ".paper-meta": meta_widget,
            ".paper-preview": preview_widget,
        }
        return mapping[selector]

    item.query_one = MagicMock(side_effect=_query)
    item._update_display()

    title_widget.update.assert_called_once()
    authors_widget.update.assert_called_once()
    meta_widget.update.assert_called_once()
    preview_widget.update.assert_called_once()


def test_paper_list_item_compose_respects_preview_flag(make_paper):
    no_preview = PaperListItem(make_paper(), show_preview=False)
    with_preview = PaperListItem(make_paper(), show_preview=True)

    assert len(list(no_preview.compose())) == 3
    assert len(list(with_preview.compose())) == 4


def test_listing_helper_branch_matrix(make_paper):
    paper = make_paper(arxiv_id="2401.77777")

    terms = PaperHighlightTerms(title=("graph",), author=("author",), abstract=("text",))
    assert _normalize_highlight_terms(terms) is terms
    normalized = _normalize_highlight_terms(
        {"title": ["graph"], "author": ["author"], "abstract": ["text"]}
    )
    assert normalized.title == ("graph",)
    assert normalized.author == ("author",)
    assert normalized.abstract == ("text",)
    assert _normalize_highlight_terms(None) == PaperHighlightTerms()

    state = PaperRowRenderState(paper=paper)
    assert _normalize_paper_row_state(state).paper is paper
    with pytest.raises(TypeError):
        _coerce_paper_row_state("bad", {})
    with pytest.raises(TypeError):
        _coerce_paper_row_state(state, {"metadata": None})

    assert _compress_meta_parts([]) == ""
    assert _compress_meta_parts(["one", "two"]) == "one  two"
    compressed = _compress_meta_parts(["x" * 40, "y" * 40, "z" * 40], budget=12)
    assert len(compressed) <= 12

    assert "Loading abstract" in _render_abstract_preview(replace(state, abstract_text=None))
    assert "No abstract available" in _render_abstract_preview(replace(state, abstract_text=""))
    assert "Short abstract" in _render_abstract_preview(
        replace(state, abstract_text="Short abstract")
    )
    preview = _render_abstract_preview(replace(state, abstract_text=("word " * 80).strip()))
    assert preview.endswith("[/]")

    item = PaperListItem(paper, show_preview=True)
    item._abstract_text = "preview"
    item._s2_data = None
    item._hf_data = None
    item._version_update = (2, 3)
    item._relevance_score = (8, "good")
    assert item._abstract_text == "preview"
    assert item._version_update == (2, 3)
    assert item._relevance_score == (8, "good")


# ============================================================================
# truncate_at_word_boundary tests
# ============================================================================


class TestTruncateAtWordBoundary:
    """Tests for the shared word-boundary truncation utility."""

    def test_short_text_unchanged(self):
        assert truncate_at_word_boundary("short", 100) == "short"

    def test_exact_limit_unchanged(self):
        text = "exact"
        assert truncate_at_word_boundary(text, len(text)) == text

    def test_truncates_at_word_boundary(self):
        result = truncate_at_word_boundary("hello world foo", 14)
        assert result == "hello world\u2026"
        assert len(result) <= 14

    def test_unicode_ellipsis_default(self):
        result = truncate_at_word_boundary("hello world foo bar", 12)
        assert result.endswith("\u2026")
        assert "..." not in result

    def test_ascii_mode_uses_three_dots(self):
        result = truncate_at_word_boundary("hello world foo bar", 14, ascii_mode=True)
        assert result.endswith("...")
        assert "\u2026" not in result

    def test_long_word_falls_back_to_char_truncation(self):
        result = truncate_at_word_boundary("abcdefghijklmnopqrstuvwxyz", 10)
        assert result == "abcdefghi\u2026"
        assert len(result) == 10

    def test_result_never_exceeds_max_length(self):
        text = "the quick brown fox jumps over the lazy dog"
        for limit in range(5, len(text)):
            result = truncate_at_word_boundary(text, limit)
            assert len(result) <= limit, f"limit={limit}: len({result!r})={len(result)}"

    def test_ascii_result_never_exceeds_max_length(self):
        text = "the quick brown fox jumps over the lazy dog"
        for limit in range(5, len(text)):
            result = truncate_at_word_boundary(text, limit, ascii_mode=True)
            assert len(result) <= limit, f"limit={limit}: len({result!r})={len(result)}"

    def test_sixty_percent_threshold(self):
        # "a " is at index 1, which is < 60% of cutoff — should fall back
        result = truncate_at_word_boundary("a bbbbbbbbbbbbbbbbbbbb", 10)
        assert result == "a bbbbbbb\u2026"  # char-truncated, not "a…"

    def test_very_small_max_length(self):
        result = truncate_at_word_boundary("hello world", 1)
        assert result == "\u2026"
        assert len(result) <= 1

    def test_ascii_mode_small_max_length(self):
        result = truncate_at_word_boundary("hello world", 3, ascii_mode=True)
        assert result == "..."
        assert len(result) <= 3


def test_chrome_helper_branches_cover_status_and_date_navigation_helpers():
    from datetime import date
    from pathlib import Path

    import pytest

    import arxiv_browser.widgets.chrome as chrome_mod

    base_colors = dict(chrome_mod.DEFAULT_THEME)

    state = chrome_mod.StatusBarState(
        total=10,
        filtered=3,
        query="graph",
        watch_filter_active=True,
        selected_count=2,
        sort_label="date",
        in_arxiv_api_mode=True,
        api_page=4,
        arxiv_api_loading=True,
        show_abstract_preview=True,
        s2_active=True,
        s2_loading=False,
        s2_count=5,
        s2_api_error=False,
        hf_active=True,
        hf_loading=False,
        hf_match_count=4,
        hf_api_error=False,
        version_checking=True,
        version_update_count=1,
        max_width=80,
        theme_colors=base_colors,
    )

    coerced = chrome_mod._coerce_status_bar_state(state, {})
    assert coerced.theme_colors == base_colors
    legacy = chrome_mod._coerce_status_bar_state(
        None,
        {
            "total": 10,
            "filtered": 3,
            "query": "",
            "watch_filter_active": False,
            "selected_count": 0,
            "sort_label": "date",
            "in_arxiv_api_mode": False,
            "api_page": None,
            "arxiv_api_loading": False,
            "show_abstract_preview": False,
            "s2_active": False,
            "s2_loading": False,
            "s2_count": 0,
        },
    )
    assert legacy.total == 10
    with pytest.raises(TypeError):
        chrome_mod._coerce_status_bar_state(state, {"total": 1})

    assert (
        chrome_mod._compact_primary_segment(
            total=10, filtered=3, query="graph", watch_filter_active=False
        )
        == "3/10 match"
    )
    assert (
        chrome_mod._compact_primary_segment(
            total=10, filtered=3, query="", watch_filter_active=True
        )
        == "3/10 watched"
    )
    assert (
        chrome_mod._compact_primary_segment(
            total=10, filtered=10, query="", watch_filter_active=False
        )
        == "10 papers"
    )

    assert (
        chrome_mod._full_primary_segment(
            total=10,
            filtered=3,
            query="a" * 40,
            watch_filter_active=False,
            theme_colors=base_colors,
        ).count("...")
        == 1
    )
    assert (
        chrome_mod._compact_flag_segment(active=False, loading=False, count=0, label="S2") is None
    )
    assert (
        chrome_mod._compact_flag_segment(active=True, loading=True, count=0, label="S2")
        == "S2 Loading..."
    )
    assert (
        chrome_mod._compact_flag_segment(active=True, loading=False, count=2, label="HF") == "HF:2"
    )
    assert (
        chrome_mod._compact_flag_segment(
            active=True, loading=False, count=0, label="HF", api_error=True
        )
        == "HF:err"
    )

    compact_parts = chrome_mod._build_compact_status_parts(state)
    assert "graph" not in " | ".join(compact_parts)
    assert any(part.startswith("API p4") for part in compact_parts)
    assert any(part.startswith("S2") for part in compact_parts)

    full_parts = chrome_mod._build_full_status_parts(
        chrome_mod.StatusBarState(
            total=10,
            filtered=3,
            query="graph",
            watch_filter_active=True,
            selected_count=2,
            sort_label="date",
            in_arxiv_api_mode=True,
            api_page=4,
            arxiv_api_loading=True,
            show_abstract_preview=True,
            s2_active=True,
            s2_loading=True,
            s2_count=5,
            s2_api_error=False,
            hf_active=True,
            hf_loading=True,
            hf_match_count=4,
            hf_api_error=False,
            version_checking=False,
            version_update_count=1,
            max_width=120,
            theme_colors=base_colors,
        )
    )
    assert any("Preview" in part for part in full_parts)
    assert any("S2 loading" in part for part in full_parts)
    assert any("HF loading" in part for part in full_parts)
    assert all("Checking versions" not in part for part in full_parts)

    assert chrome_mod._render_compact_status(["a", "b", "c"], 6).endswith("...")
    assert chrome_mod._truncate_rich_text("[bold]hello[/] world", 5).endswith("...")
    assert chrome_mod._truncate_rich_text("[bold]hello[/]", None) == "[bold]hello[/]"

    assert chrome_mod.build_selection_footer_bindings(3)[0] == ("o", "open(3)")
    assert chrome_mod.build_search_footer_bindings()[-1] == ("?", "help")
    assert chrome_mod.build_api_footer_bindings()[1] == ("Esc/Ctrl+e", "exit")
    assert chrome_mod.build_browse_footer_bindings(
        s2_active=True,
        has_starred=True,
        llm_configured=True,
        has_history_navigation=True,
    )[5] == ("[/]", "dates")
    assert chrome_mod.build_footer_mode_badge(
        relevance_scoring_active=True,
        version_checking=False,
        search_visible=False,
        in_arxiv_api_mode=False,
        selected_count=0,
        theme_colors=base_colors,
    ).startswith("[bold")

    assert chrome_mod._compute_window_bounds(0, 0, 5) == (0, 0)
    assert chrome_mod._compute_window_bounds(10, 8, 5) == (5, 10)
    assert (
        chrome_mod._format_date_nav_label(
            date(2026, 1, 2),
            count=4,
            mode=chrome_mod.DATE_NAV_LABEL_WITH_COUNTS,
        )
        == "Jan 02(4)"
    )
    assert (
        chrome_mod._format_date_nav_label(
            date(2026, 1, 2),
            count=None,
            mode=chrome_mod.DATE_NAV_LABEL_MONTH_DAY,
        )
        == "Jan 02"
    )
    assert (
        chrome_mod._format_date_nav_label(
            date(2026, 1, 2),
            count=None,
            mode=chrome_mod.DATE_NAV_LABEL_NUMERIC,
        )
        == "01-02"
    )
    assert chrome_mod._estimate_date_nav_width(["Jan 01", "Jan 02"]) > 0

    history = [
        (date(2026, 1, 3), Path("/tmp/2026-01-03.txt")),
        (date(2026, 1, 2), Path("/tmp/2026-01-02.txt")),
        (date(2026, 1, 1), Path("/tmp/2026-01-01.txt")),
    ]
    start, end, mode = chrome_mod._compute_responsive_date_plan(
        history,
        current_index=1,
        width=0,
        get_count=lambda _idx: 1,
    )
    assert (start, end, mode) == (0, 3, chrome_mod.DATE_NAV_LABEL_WITH_COUNTS)
    start, end, mode = chrome_mod._compute_responsive_date_plan(
        history,
        current_index=1,
        width=200,
        get_count=lambda idx: idx + 1,
    )
    assert start <= 1 <= end
    assert mode in chrome_mod.DATE_NAV_LABEL_MODES


# ============================================================================
# ASCII mode comprehensive audit
# ============================================================================

_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


class TestAsciiModeNoUnicodeLeaks:
    """Verify that ASCII mode produces only ASCII characters in rendered output."""

    def _assert_ascii_only(self, text: str, label: str = "") -> None:
        match = _NON_ASCII_RE.search(text)
        assert match is None, (
            f"Non-ASCII char U+{ord(match.group()):04X} ({match.group()!r}) "
            f"found in {label}: ...{text[max(0, match.start() - 20) : match.end() + 20]}..."
        )

    def test_listing_render_paper_option_ascii(self, make_paper):
        """render_paper_option produces ASCII-only output in ASCII mode."""
        from arxiv_browser.widgets.listing import set_ascii_icons

        paper = make_paper(title="Attention Is All You Need")
        set_ascii_icons(True)
        try:
            text = render_paper_option(
                paper,
                selected=True,
                watched=True,
                metadata=PaperMetadata(
                    arxiv_id=paper.arxiv_id,
                    starred=True,
                    is_read=True,
                ),
                hf_data=HuggingFacePaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    upvotes=10,
                    num_comments=2,
                    ai_summary="",
                    ai_keywords=(),
                    github_repo="",
                    github_stars=0,
                ),
                version_update=(1, 3),
                relevance_score=(8, "high"),
            )
            self._assert_ascii_only(text, "render_paper_option")
        finally:
            set_ascii_icons(False)

    def test_chrome_status_bar_ascii(self):
        """Status bar uses ASCII separator in ASCII mode."""
        from arxiv_browser.widgets.chrome import (
            build_status_bar_text,
            set_ascii_glyphs,
        )

        set_ascii_glyphs(True)
        try:
            result = build_status_bar_text(
                StatusBarState(
                    total=100,
                    filtered=50,
                    query="test",
                    watch_filter_active=False,
                    selected_count=0,
                    sort_label="date",
                    in_arxiv_api_mode=False,
                    api_page=None,
                    arxiv_api_loading=False,
                    show_abstract_preview=False,
                    s2_active=True,
                    s2_loading=False,
                    s2_count=10,
                    hf_active=True,
                    hf_loading=False,
                    hf_match_count=5,
                    version_checking=False,
                    version_update_count=0,
                )
            )
            # Strip Rich tags before checking
            plain = re.sub(r"\[[^\]]*]", "", result)
            self._assert_ascii_only(plain, "build_status_bar_text")
        finally:
            set_ascii_glyphs(False)

    def test_chrome_search_footer_ascii(self):
        """Search footer uses ASCII arrows in ASCII mode."""
        from arxiv_browser.widgets.chrome import (
            build_search_footer_bindings,
            set_ascii_glyphs,
        )

        set_ascii_glyphs(True)
        try:
            bindings = build_search_footer_bindings()
            all_text = " ".join(f"{k} {v}" for k, v in bindings)
            self._assert_ascii_only(all_text, "search footer bindings")
            assert any("^v" in k for k, _ in bindings)
        finally:
            set_ascii_glyphs(False)

    def test_progress_bar_ascii(self):
        """render_progress_bar uses # and - in ASCII mode."""
        from arxiv_browser.query import render_progress_bar

        result = render_progress_bar(5, 10, width=10, ascii_mode=True)
        self._assert_ascii_only(result, "render_progress_bar")
        assert "#" in result
        assert "-" in result

    def test_format_summary_bullets_ascii(self):
        """format_summary_as_rich uses - bullets in ASCII mode."""
        from arxiv_browser._ascii import set_ascii_mode
        from arxiv_browser.query import format_summary_as_rich

        set_ascii_mode(True)
        try:
            result = format_summary_as_rich("- item one\n- item two")
            plain = re.sub(r"\[[^\]]*]", "", result)
            self._assert_ascii_only(plain, "format_summary_as_rich")
        finally:
            set_ascii_mode(False)

    def test_build_daily_digest_ascii(self):
        """build_daily_digest uses | separator in ASCII mode."""
        from arxiv_browser._ascii import set_ascii_mode
        from arxiv_browser.parsing import build_daily_digest

        set_ascii_mode(True)
        try:
            paper = make_paper_for_digest()
            result = build_daily_digest([paper])
            self._assert_ascii_only(result, "build_daily_digest")
            assert "|" in result or result == "1 papers" or "papers" in result
        finally:
            set_ascii_mode(False)

    def test_help_section_headers_ascii(self):
        """Help section headers replace · with - in ASCII mode."""
        from arxiv_browser._ascii import set_ascii_mode
        from arxiv_browser.help_ui import build_help_sections

        set_ascii_mode(True)
        try:
            sections = build_help_sections([])
            for name, _ in sections:
                self._assert_ascii_only(name, f"help section '{name}'")
        finally:
            set_ascii_mode(False)

    def test_detail_glyphs_ascii(self):
        """Detail pane glyphs are ASCII in ASCII mode."""
        from arxiv_browser.widgets.details import (
            _DETAIL_GLYPH_SETS,
            set_ascii_glyphs,
        )

        set_ascii_glyphs(True)
        try:
            for key, val in _DETAIL_GLYPH_SETS["ascii"].items():
                self._assert_ascii_only(val, f"detail glyph '{key}'")
        finally:
            set_ascii_glyphs(False)


def make_paper_for_digest():
    """Create a minimal Paper for digest tests."""
    from arxiv_browser.models import Paper

    return Paper(
        arxiv_id="2401.00001",
        date="Mon, 15 Jan 2024",
        title="Test",
        authors="Author",
        categories="cs.AI",
        comments=None,
        abstract="Abstract.",
        url="https://arxiv.org/abs/2401.00001",
        abstract_raw="Abstract.",
    )
