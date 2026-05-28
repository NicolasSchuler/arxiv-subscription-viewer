"""Focused tests for interactive chrome widgets."""

from __future__ import annotations

from datetime import date as dt_date
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from arxiv_browser.models import SearchBookmark
from arxiv_browser.query import tokenize_query
from arxiv_browser.themes import TEXTUAL_THEMES, THEME_COLORS
from arxiv_browser.widgets.chrome import BookmarkTabBar, ContextFooter, DateNavigator, FilterPillBar


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

        # Hints without a bound action (e.g. "[/] dates") are not links.
        assert "@click=app.dates" not in content


def test_context_footer_unmounted_content_remains_markup_compatible() -> None:
    footer = ContextFooter()
    footer.render_bindings([("o", "open"), ("s", "sort")])
    rendered = str(footer.content)
    assert THEME_COLORS["accent"] in rendered
    assert "open" in rendered
    assert "@click=app.open_url" in rendered


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
