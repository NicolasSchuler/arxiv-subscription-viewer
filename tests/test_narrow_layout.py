"""Regression tests for app-chrome layout and footer fixes (UX review wave)."""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.widgets import Header, OptionList

from arxiv_browser._ascii import set_ascii_mode
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.browser.options import ArxivBrowserOptions
from arxiv_browser.models import UserConfig
from arxiv_browser.query import format_categories
from arxiv_browser.whats_new import WHATS_NEW_VERSION
from arxiv_browser.widgets.chrome import ContextFooter
from arxiv_browser.widgets.details import DetailRenderState, PaperDetails


def _config() -> UserConfig:
    return UserConfig(
        onboarding_seen=True,
        last_seen_whats_new=WHATS_NEW_VERSION,
        theme_name="github-light",
    )


def _footer_plain(app) -> str:
    footer = app.query_one(ContextFooter)
    return Text.from_markup(str(footer.content)).plain


def _app(make_paper, count: int = 7) -> ArxivBrowser:
    papers = [make_paper(arxiv_id=f"2401.010{i:02d}") for i in range(count)]
    return ArxivBrowser(papers, ArxivBrowserOptions(config=_config(), restore_session=False))


@pytest.mark.asyncio
async def test_footer_is_visible_and_shows_help_at_100x30(make_paper):
    """The context footer paints at least one row with the help hint (#1)."""
    app = _app(make_paper)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        footer = app.query_one(ContextFooter)
        assert footer.content_size.height >= 1
        assert "? help" in _footer_plain(app)


@pytest.mark.asyncio
async def test_narrow_layout_keeps_usable_list_rows_at_80x24(make_paper):
    """The paper list keeps usable rows when panes stack on a narrow terminal (H2)."""
    app = _app(make_paper)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        assert app.screen.has_class("-narrow")
        paper_list = app.query_one("#paper-list", OptionList)
        # Each paper option renders ~3 lines; the list floor must show >= 3 rows.
        assert paper_list.content_size.height >= 9


@pytest.mark.asyncio
@pytest.mark.parametrize("split", ["pane-split-1", "pane-split-2", "pane-split-4"])
async def test_narrow_list_floor_holds_at_every_pane_split(make_paper, split):
    """The >=3-row list floor holds even at detail-favoring presets (H2)."""
    app = _app(make_paper)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        app.screen.remove_class("pane-split-2")
        app.screen.add_class(split)
        await pilot.pause(0.2)
        paper_list = app.query_one("#paper-list", OptionList)
        assert paper_list.content_size.height >= 9


@pytest.mark.asyncio
async def test_header_icon_uses_available_glyph(make_paper):
    """The header icon uses a widely available glyph instead of tofu U+2B58 (L6)."""
    app = _app(make_paper)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)
        assert app.query_one(Header).icon == "≡"


@pytest.mark.asyncio
async def test_header_icon_ascii_fallback(make_paper):
    """In ASCII mode the header icon falls back to '*' (L6)."""
    papers = [make_paper(arxiv_id="2401.01000")]
    app = ArxivBrowser(
        papers,
        ArxivBrowserOptions(config=_config(), restore_session=False, ascii_icons=True),
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)
        assert app.query_one(Header).icon == "*"


def test_format_title_uses_spaced_middot_separator(make_paper):
    """The header title/subtitle join uses ' · ' with no glued em-dash (M1)."""
    app = _app(make_paper)
    set_ascii_mode(False)
    rendered = str(app.format_title("arXiv Paper Browser", "Browse"))
    assert rendered == "arXiv Paper Browser · Browse"
    assert "—" not in rendered
    assert "  " not in rendered  # no double space


def test_format_title_ascii_uses_spaced_hyphen(make_paper):
    """In ASCII mode the separator is ' - ' with clean spacing (M1)."""
    app = _app(make_paper)
    set_ascii_mode(True)
    try:
        rendered = str(app.format_title("arXiv Paper Browser", "Browse"))
    finally:
        set_ascii_mode(False)
    assert rendered == "arXiv Paper Browser - Browse"
    assert "--" not in rendered
    assert "  " not in rendered


def test_format_title_without_subtitle_has_no_separator(make_paper):
    """With no subtitle the header shows only the title (M1)."""
    app = _app(make_paper)
    set_ascii_mode(False)
    assert str(app.format_title("arXiv Paper Browser", "")) == "arXiv Paper Browser"


@pytest.mark.asyncio
async def test_footer_fits_and_retains_help_at_80_cols(make_paper):
    """At 80 cols the footer drops low-priority hints but keeps help/commands (#3)."""
    app = _app(make_paper)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        plain = _footer_plain(app)
        assert "? help" in plain
        assert "Ctrl+p commands" in plain
        # It must fit inside the footer's content width (no clipping).
        assert Text(plain).cell_len <= app.query_one(ContextFooter).content_size.width


def test_unknown_category_uses_theme_muted_fallback():
    """Unknown categories fall back to the supplied (theme) colour, not #888888 (#26)."""
    out = format_categories(
        "cs.AI zz.ZZ",
        {"cs.AI": "#111111"},
        default_color="#405a60",
    )
    assert "[#405a60]zz.ZZ[/]" in out
    assert "#888888" not in out


def test_detail_line_cursor_hidden_when_pane_unfocused(make_paper):
    """The detail line cursor glyph only renders when the detail pane is focused (#38)."""
    details = PaperDetails()
    paper = make_paper(arxiv_id="2401.00099", title="Cursor Test")
    body = "line one\nline two\nline three\nline four"

    details.update_state(
        DetailRenderState(paper=paper, abstract_text=body, detail_line_cursor=2, detail_focus=False)
    )
    assert "\u276f" not in str(details.content)

    details.update_state(
        DetailRenderState(paper=paper, abstract_text=body, detail_line_cursor=2, detail_focus=True)
    )
    assert "\u276f" in str(details.content)
