"""Tests for the HelpScreen progressive help overlay with tier-based tabs."""

from __future__ import annotations

import pytest
from textual.containers import Vertical
from textual.widgets import Input, Static, TabbedContent

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.help import HelpScreen, _classify_section
from tests.support.patch_helpers import patch_save_config


async def _open_modal(app: ArxivBrowser, pilot, modal: HelpScreen) -> None:
    app.push_screen(modal)
    await pilot.pause(0.05)
    assert app.screen_stack[-1] is modal


# ------------------------------------------------------------------
# Unit tests (no TUI needed)
# ------------------------------------------------------------------


class TestClassifySection:
    """_classify_section maps section names to the correct tab ID."""

    def test_getting_started(self):
        assert _classify_section("Getting Started") == "getting-started"

    def test_search_syntax(self):
        assert _classify_section("Search Syntax") == "getting-started"

    def test_core_actions(self):
        assert _classify_section("Core Actions") == "core"

    def test_standard_organize(self):
        assert _classify_section("Standard · Organize") == "standard"

    def test_standard_navigate(self):
        assert _classify_section("Standard · Navigate") == "standard"

    def test_power_llm(self):
        assert _classify_section("Power · LLM") == "power"

    def test_power_discovery(self):
        assert _classify_section("Power · Discovery") == "power"

    def test_power_research_tools(self):
        assert _classify_section("Power · Research Tools") == "power"

    def test_unknown_falls_to_all(self):
        assert _classify_section("Misc Stuff") == "all"


class TestSectionsForTab:
    """HelpScreen._sections_for_tab returns the correct section subsets."""

    @pytest.fixture
    def modal(self) -> HelpScreen:
        sections = [
            ("Getting Started", [("?", "Help")]),
            ("Search Syntax", [("cat:cs.AI", "Category")]),
            ("Core Actions", [("o", "Open")]),
            ("Standard · Organize", [("r", "Read")]),
            ("Power · LLM", [("C", "Chat")]),
        ]
        return HelpScreen(sections=sections)

    def test_getting_started_tab(self, modal: HelpScreen):
        result = modal._sections_for_tab("getting-started")
        names = [n for n, _ in result]
        assert "Getting Started" in names
        assert "Search Syntax" in names
        assert len(result) == 2

    def test_core_tab(self, modal: HelpScreen):
        result = modal._sections_for_tab("core")
        assert len(result) == 1
        assert result[0][0] == "Core Actions"

    def test_standard_tab(self, modal: HelpScreen):
        result = modal._sections_for_tab("standard")
        assert len(result) == 1
        assert result[0][0] == "Standard · Organize"

    def test_power_tab(self, modal: HelpScreen):
        result = modal._sections_for_tab("power")
        assert len(result) == 1
        assert result[0][0] == "Power · LLM"

    def test_all_tab(self, modal: HelpScreen):
        result = modal._sections_for_tab("all")
        assert len(result) == 5


class TestConstructorDefaults:
    """Constructor backward compatibility."""

    def test_default_initial_tab(self):
        modal = HelpScreen()
        assert modal._initial_tab == "getting-started"

    def test_custom_initial_tab(self):
        modal = HelpScreen(initial_tab="power")
        assert modal._initial_tab == "power"

    def test_default_footer(self):
        modal = HelpScreen()
        assert modal._footer_note == "Close: ? / Esc / q"

    def test_default_sections_used_when_none(self):
        modal = HelpScreen()
        assert len(modal._sections) == len(HelpScreen._DEFAULT_SECTIONS)


class TestFilterSections:
    """_filter_sections works unchanged after the tabbed refactor."""

    def test_empty_query_returns_all(self):
        modal = HelpScreen()
        assert modal._filter_sections("") is modal._sections

    def test_matching_query(self):
        modal = HelpScreen()
        filtered = modal._filter_sections("search")
        all_entries = [(k, d) for _, entries in filtered for k, d in entries]
        assert len(all_entries) > 0
        for key, desc in all_entries:
            assert "search" in key.lower() or "search" in desc.lower()

    def test_nonsense_query_returns_empty(self):
        modal = HelpScreen()
        assert modal._filter_sections("xyzzyplugh") == []


# ------------------------------------------------------------------
# TUI integration tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tabs_render_with_correct_section_grouping(make_paper):
    """Each tab renders only the sections that belong to it."""
    sections = [
        ("Getting Started", [("?", "Help")]),
        ("Search Syntax", [("cat:cs.AI", "Category")]),
        ("Core Actions", [("o", "Open")]),
        ("Standard · Organize", [("r", "Read")]),
        ("Power · LLM", [("C", "Chat")]),
    ]
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = HelpScreen(sections=sections)

    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)

            # Getting Started tab: 2 sections
            gs_body = modal.query_one("#help-tab-getting-started", Vertical)
            assert len(gs_body.query(".help-section-title")) == 2

            # Core tab: 1 section
            core_body = modal.query_one("#help-tab-core", Vertical)
            assert len(core_body.query(".help-section-title")) == 1

            # Standard tab: 1 section
            std_body = modal.query_one("#help-tab-standard", Vertical)
            assert len(std_body.query(".help-section-title")) == 1

            # Power tab: 1 section
            pwr_body = modal.query_one("#help-tab-power", Vertical)
            assert len(pwr_body.query(".help-section-title")) == 1

            # All tab: all 5 sections
            all_body = modal.query_one("#help-tab-all", Vertical)
            assert len(all_body.query(".help-section-title")) == 5


@pytest.mark.asyncio
async def test_filter_input_switches_to_flat_view(make_paper):
    """Filter input hides tabs and shows flat filtered results."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = HelpScreen()

    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)

            tabs = modal.query_one("#help-tabs", TabbedContent)
            flat = modal.query_one("#help-sections", Vertical)

            # Initially: tabs visible, flat hidden
            assert "hidden" not in tabs.classes
            assert "hidden" in flat.classes
            assert tabs.display is True
            assert flat.display is False

            # Apply filter
            filter_input = modal.query_one("#help-filter", Input)
            filter_input.value = "bookmark"
            await pilot.pause(0.1)

            # After filter: tabs hidden, flat visible with results
            assert "hidden" in tabs.classes
            assert "hidden" not in flat.classes
            assert tabs.display is False
            assert flat.display is True
            section_titles = flat.query(".help-section-title")
            assert len(section_titles) >= 1

            # Clear filter restores tabs
            filter_input.value = ""
            await pilot.pause(0.1)
            assert "hidden" not in tabs.classes
            assert "hidden" in flat.classes
            assert tabs.display is True
            assert flat.display is False
            assert len(flat.children) == 0


@pytest.mark.asyncio
async def test_no_matches_shown_in_flat_view(make_paper):
    """Nonsense filter shows 'No matches' in flat view."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = HelpScreen()

    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)

            filter_input = modal.query_one("#help-filter", Input)
            filter_input.value = "xyzzyplugh"
            await pilot.pause(0.1)

            flat = modal.query_one("#help-sections", Vertical)
            no_match = flat.query_one("#help-no-matches", Static)
            assert "No matches" in no_match.render().plain


@pytest.mark.asyncio
async def test_initial_tab_parameter(make_paper):
    """initial_tab parameter selects the correct starting tab."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = HelpScreen(initial_tab="power")

    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)

            tabs = modal.query_one("#help-tabs", TabbedContent)
            assert tabs.active == "power"


@pytest.mark.asyncio
async def test_default_initial_tab_is_getting_started(make_paper):
    """Default initial_tab is 'getting-started'."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = HelpScreen()

    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)

            tabs = modal.query_one("#help-tabs", TabbedContent)
            assert tabs.active == "getting-started"
