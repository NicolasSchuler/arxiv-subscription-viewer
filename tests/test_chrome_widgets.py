"""Focused tests for interactive chrome widgets."""

from __future__ import annotations

from datetime import date as dt_date
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from arxiv_browser._ascii import set_ascii_mode
from arxiv_browser.models import SearchBookmark
from arxiv_browser.query import tokenize_query
from arxiv_browser.themes import TEXTUAL_THEMES, THEME_COLORS
from arxiv_browser.widgets.chrome import (
    BookmarkTabBar,
    ContextFooter,
    DateNavigator,
    FilterPillBar,
    _fit_bookmark_chips,
)


class ChromeHarness(App[None]):
    """Base app that registers the app theme variables used by chrome CSS."""

    def __init__(self) -> None:
        super().__init__()
        for textual_theme in TEXTUAL_THEMES.values():
            self.register_theme(textual_theme)
        self.theme = "monokai"


class FooterHarness(ChromeHarness):
    """Minimal app that renders a ContextFooter with given bindings."""

    def __init__(self, bindings: list[tuple[str, str]], mode_badge: str = "") -> None:
        super().__init__()
        self.bindings = bindings
        self.mode_badge = mode_badge

    def compose(self) -> ComposeResult:
        yield ContextFooter()

    def on_mount(self) -> None:
        self.query_one(ContextFooter).render_bindings(self.bindings, self.mode_badge)


class DateHarness(ChromeHarness):
    """Minimal app that mounts and updates a DateNavigator."""

    def __init__(self) -> None:
        super().__init__()
        self.history_files = [
            (dt_date(2026, 1, 3), Path("tests/no-history-3.txt")),
            (dt_date(2026, 1, 2), Path("tests/no-history-2.txt")),
            (dt_date(2026, 1, 1), Path("tests/no-history-1.txt")),
        ]

    def compose(self) -> ComposeResult:
        yield DateNavigator(self.history_files, current_index=1)

    async def on_mount(self) -> None:
        await self.query_one(DateNavigator).update_dates(self.history_files, 1)


class FilterHarness(ChromeHarness):
    """Minimal app that mounts and updates a FilterPillBar."""

    def compose(self) -> ComposeResult:
        yield FilterPillBar()

    async def on_mount(self) -> None:
        await self.query_one(FilterPillBar).update_pills(tokenize_query("cat:cs.AI"), True)


@pytest.mark.asyncio
async def test_context_footer_clickable_hints_use_action_links_and_cap_items() -> None:
    bindings = [
        ("/", "search"),
        ("Space", "select"),
        ("o", "open"),
        ("s", "sort"),
        ("[/]", "dates"),
        ("E", "export"),
        ("Ctrl+p", "commands"),
        ("?", "help"),
        ("r", "read"),
        ("x", "star"),
    ]
    badge = f"[bold {THEME_COLORS['accent']}] SEARCH [/]"
    app = FooterHarness(bindings, badge)

    async with app.run_test(size=(100, 5)) as pilot:
        footer = app.query_one(ContextFooter)
        await pilot.pause()
        content = str(footer.content)

        # Mode badge is rendered and the 10th hint is dropped (cap of 9).
        assert "SEARCH" in content
        assert "read" in content
        assert "star" not in content

        # Clickable hints are wired via @click action links to app actions.
        assert "@click=app.toggle_search" in content
        assert "@click=app.cycle_sort" in content
        assert "@click=app.command_palette" in content
        assert "@click=app.show_help" in content
        assert "@click=app.fetch_s2" not in content

        # Hints without a bound action (e.g. "[/] dates") are not links.
        assert "@click=app.dates" not in content


@pytest.mark.asyncio
async def test_contextual_footer_hints_are_clickable_when_actionable() -> None:
    app = FooterHarness([("e", "S2"), ("L", "relevance"), ("V", "versions"), ("x", "star")])

    async with app.run_test(size=(80, 5)) as pilot:
        await pilot.pause()
        content = str(app.query_one(ContextFooter).content)

        assert "@click=app.fetch_s2" in content
        assert "@click=app.score_relevance" in content
        assert "@click=app.check_versions" in content
        assert "@click=app.toggle_star" in content


def test_context_footer_unmounted_content_remains_markup_compatible() -> None:
    footer = ContextFooter()
    footer.render_bindings([("o", "open"), ("s", "sort")])
    rendered = str(footer.content)
    assert THEME_COLORS["accent"] in rendered
    assert "open" in rendered
    assert "@click=app.open_url" in rendered


def test_fit_bookmark_chips_keeps_all_when_width_unknown() -> None:
    """A width <= 0 (pre-layout) keeps every chip so nothing is hidden (M2)."""
    assert _fit_bookmark_chips(["a", "b"], 0) == ([(0, "a"), (1, "b")], 0)
    assert _fit_bookmark_chips([], 200) == ([], 0)


def test_fit_bookmark_chips_all_fit_at_wide_width() -> None:
    """When everything fits, all chips render with no overflow marker (M2)."""
    visible, hidden = _fit_bookmark_chips(["agents", "unread"], 120)
    assert visible == [(0, "agents"), (1, "unread")]
    assert hidden == 0


def test_fit_bookmark_chips_drops_whole_chips_and_reports_overflow() -> None:
    """Overflowing chips collapse into a trailing +N instead of clipping (M2)."""
    visible, hidden = _fit_bookmark_chips(
        ["agents", "unread AI", "robotics survey", "diffusion"], 76
    )
    assert visible == [(0, "agents"), (1, "unread AI")]
    assert hidden == 2


def test_fit_bookmark_chips_drops_a_chip_to_make_room_for_marker() -> None:
    """A chip is dropped so the +N marker itself fits (M2 while-loop branch)."""
    visible, hidden = _fit_bookmark_chips(["aa", "bb", "cc", "dd"], 50)
    assert visible == [(0, "aa")]
    assert hidden == 3


def test_fit_bookmark_chips_elides_single_overwide_chip() -> None:
    """A lone chip too wide for the row is elided, never clipped mid-word (M2)."""
    set_ascii_mode(False)
    visible, hidden = _fit_bookmark_chips(["a very long single bookmark name"], 40)
    assert len(visible) == 1
    assert visible[0][1].endswith("…")
    assert hidden == 0


def test_fit_bookmark_chips_elision_uses_ascii_fallback() -> None:
    """ASCII mode elides with '...' rather than the ellipsis glyph (M2)."""
    set_ascii_mode(True)
    try:
        visible, _hidden = _fit_bookmark_chips(["a very long single bookmark name"], 40)
    finally:
        set_ascii_mode(False)
    assert visible[0][1].endswith("...")
    assert "…" not in visible[0][1]


@pytest.mark.asyncio
async def test_bookmark_bar_shows_overflow_marker_at_narrow_width() -> None:
    """Live bookmark bar collapses overflow to +N with no mid-word fragment (M2)."""
    bookmarks = [
        SearchBookmark("agents", "q"),
        SearchBookmark("unread AI papers", "q"),
        SearchBookmark("robotics survey", "q"),
        SearchBookmark("diffusion", "q"),
    ]

    class BookmarkHarness(ChromeHarness):
        def compose(self) -> ComposeResult:
            yield BookmarkTabBar(bookmarks)

    async with BookmarkHarness().run_test(size=(60, 5)) as pilot:
        await pilot.pause(0.2)
        bar = pilot.app.query_one(BookmarkTabBar)
        chips = [str(child.render()) for child in bar.children]
        # The save hint survives and overflow is collapsed rather than clipped.
        assert "Ctrl+b save" in chips
        assert any(chip.startswith("+") for chip in chips)
        # No rendered chip ends on a bare mid-word fragment (they carry the
        # numbered prefix or the +N / save labels).
        assert "2: unread AI pap" not in chips


def test_fit_signature_is_none_without_bookmarks() -> None:
    """The width signature is None when there are no bookmarks (M2)."""
    assert BookmarkTabBar(bookmarks=[])._fit_signature(50) is None


def test_bookmark_on_resize_schedules_rebuild_only_when_layout_changes() -> None:
    """on_resize reschedules a rebuild once, and skips when the fit is unchanged (M2)."""
    from unittest.mock import MagicMock

    bar = BookmarkTabBar(bookmarks=[SearchBookmark("agents", "q")])
    bar.call_later = MagicMock()

    # No bookmarks -> early return, nothing scheduled.
    empty = BookmarkTabBar(bookmarks=[])
    empty.call_later = MagicMock()
    empty.on_resize(None)
    empty.call_later.assert_not_called()

    # First resize changes layout (signature was None) -> schedules a rebuild.
    bar.on_resize(None)
    assert bar.call_later.call_count == 1
    # Second resize with an unchanged fit -> no extra reschedule.
    bar.on_resize(None)
    assert bar.call_later.call_count == 1


@pytest.mark.asyncio
async def test_update_bookmarks_clear_path_removes_visible_class() -> None:
    """Clearing all bookmarks hides the bar without mounting chips (M2)."""
    from unittest.mock import AsyncMock, MagicMock

    bar = BookmarkTabBar(bookmarks=[SearchBookmark("agents", "q")])
    bar.remove_children = AsyncMock()
    bar.mount = MagicMock()
    bar.add_class = MagicMock()
    bar.remove_class = MagicMock()

    await bar.update_bookmarks(bookmarks=[])
    bar.remove_class.assert_called_with("visible")
    bar.mount.assert_not_called()


@pytest.mark.asyncio
async def test_update_bookmarks_aborts_when_superseded() -> None:
    """A rebuild superseded during remove_children stops before mounting (M2)."""
    from unittest.mock import AsyncMock, MagicMock

    bar = BookmarkTabBar(bookmarks=[])
    bar.mount = MagicMock()
    bar.add_class = MagicMock()
    bar.remove_class = MagicMock()

    async def _supersede() -> None:
        bar._bookmark_rebuild_token += 1

    bar.remove_children = AsyncMock(side_effect=_supersede)
    await bar.update_bookmarks(bookmarks=[SearchBookmark("agents", "q")])
    bar.mount.assert_not_called()


@pytest.mark.asyncio
async def test_rerender_bookmarks_respects_stale_tokens() -> None:
    """A stale resize rebuild aborts both before and after remove_children (M2)."""
    from unittest.mock import AsyncMock, MagicMock

    bm = [SearchBookmark("agents", "q")]

    # Stale before the await: nothing is touched.
    bar = BookmarkTabBar(bookmarks=bm)
    bar._bookmark_rebuild_token = 5
    bar.remove_children = AsyncMock()
    bar.mount = MagicMock()
    await bar._rerender_bookmarks(3)
    bar.remove_children.assert_not_called()
    bar.mount.assert_not_called()

    # Superseded during the await: children cleared but nothing re-mounted.
    bar2 = BookmarkTabBar(bookmarks=bm)
    bar2._bookmark_rebuild_token = 3
    bar2.mount = MagicMock()

    async def _supersede() -> None:
        bar2._bookmark_rebuild_token = 9

    bar2.remove_children = AsyncMock(side_effect=_supersede)
    await bar2._rerender_bookmarks(3)
    bar2.remove_children.assert_awaited_once()
    bar2.mount.assert_not_called()

    # Current token: rebuild proceeds and mounts chips.
    bar3 = BookmarkTabBar(bookmarks=bm)
    bar3._bookmark_rebuild_token = 4
    bar3.remove_children = AsyncMock()
    bar3.mount = MagicMock()
    await bar3._rerender_bookmarks(4)
    assert bar3.mount.call_count >= 2


@pytest.mark.asyncio
async def test_interactive_chrome_tooltips_are_set() -> None:
    bookmarks = [SearchBookmark("LLMs", "cat:cs.AI")]
    bookmark_children = list(BookmarkTabBar(bookmarks).compose())
    bookmark_tab = next(child for child in bookmark_children if child.id == "bookmark-0")
    assert bookmark_tab.tooltip == "Saved search 1 - press 1 to load"

    async with DateHarness().run_test(size=(80, 5)) as pilot:
        await pilot.pause()
        nav = pilot.app.query_one(DateNavigator)
        assert nav.query_one("#date-nav-prev").tooltip == "Older (])"
        assert nav.query_one("#date-nav-next").tooltip == "Newer ([)"
        assert nav.query_one("#date-nav-1").tooltip == "Jump to 2026-01-02"

    async with FilterHarness().run_test(size=(80, 5)) as pilot:
        await pilot.pause()
        pills = pilot.app.query_one(FilterPillBar)
        assert pills.query_one("#pill-0").tooltip == "Click to remove filter"
        assert pills.query_one("#pill-watch").tooltip == "Click to remove watch filter"
