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
from tests.support.patch_helpers import patch_save_config

# ============================================================================
# Tests for clean_latex function
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

    @staticmethod
    async def _wait_for_option_count(
        pilot, option_list, expected: int, timeout: float = 2.0
    ) -> None:
        """Poll until OptionList reaches expected count or timeout."""
        import asyncio

        deadline = asyncio.get_running_loop().time() + timeout
        while option_list.option_count != expected and asyncio.get_running_loop().time() < deadline:
            await pilot.pause(0.05)
        assert option_list.option_count == expected

    async def test_app_renders_paper_list(self, make_paper):
        """App should mount and render all papers in the option list."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test():
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 5

    async def test_search_filters_papers(self, make_paper):
        """Typing in search should filter the paper list after debounce."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.browser.core import ArxivBrowser

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
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                # Open search
                await pilot.press("slash")
                # Type a query that matches only the quantum paper
                await pilot.press("Q", "u", "a", "n", "t", "u", "m")
                option_list = app.query_one("#paper-list", OptionList)
                await self._wait_for_option_count(pilot, option_list, expected=1)

    async def test_search_clear_restores_all(self, make_paper):
        """Pressing escape on search should restore all papers."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                # Open search and type an advanced query that matches nothing
                # Use "cat:" prefix to trigger exact (non-fuzzy) matching
                await pilot.press("slash")
                for ch in "cat:nonexistent":
                    await pilot.press(ch)
                option_list = app.query_one("#paper-list", OptionList)
                # Empty state shows a single placeholder option
                await self._wait_for_option_count(pilot, option_list, expected=1)

                # Cancel search with escape
                await pilot.press("escape")
                await self._wait_for_option_count(pilot, option_list, expected=3)

    async def test_filter_to_empty_cancels_pending_detail_update(self, make_paper):
        """Filtering to an empty list must not allow stale debounced detail updates."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.details import PaperDetails

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.listing import PaperListItem

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.modals import HelpScreen

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser

        local_papers = [make_paper(arxiv_id="2401.00001", title="Local paper")]
        api_paper = make_paper(
            arxiv_id="2602.00001",
            title="API result",
        )
        api_paper.source = "api"
        api_papers = [api_paper]

        app = ArxivBrowser(local_papers, restore_session=False)
        with (
            patch_save_config(return_value=True),
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

        from arxiv_browser.browser.core import ArxivBrowser

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
            patch_save_config(return_value=True),
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

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import (
            ArxivSearchModeState,
            ArxivSearchRequest,
        )

        app = ArxivBrowser([make_paper(arxiv_id="2401.00001")], restore_session=False)
        with patch_save_config(return_value=True):
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

        from arxiv_browser.browser.core import ArxivBrowser

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
            patch_save_config(return_value=True),
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

    async def test_restore_local_snapshot_keeps_live_search_query(self, make_paper):
        """Exiting API mode should restore the query text the user actually typed."""
        from unittest.mock import patch

        from textual.widgets import Input, OptionList

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import LocalBrowseSnapshot

        local_papers = [
            make_paper(arxiv_id="2401.00001", title="Transformer Architecture"),
            make_paper(arxiv_id="2401.00002", title="Other Paper"),
        ]
        app = ArxivBrowser(local_papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test():
                app._local_browse_snapshot = LocalBrowseSnapshot(
                    all_papers=local_papers,
                    papers_by_id={p.arxiv_id: p for p in local_papers},
                    selected_ids=set(),
                    sort_index=0,
                    search_query="Transformer",
                    pending_query="Transformer",
                    applied_query="",
                    watch_filter_active=False,
                    active_bookmark_index=0,
                    list_index=0,
                    sub_title="",
                    highlight_terms={"title": [], "author": [], "abstract": []},
                    match_scores={},
                )

                app._restore_local_browse_snapshot()

                search_input = app.query_one("#search-input", Input)
                option_list = app.query_one("#paper-list", OptionList)
                assert search_input.value == "Transformer"
                assert app._pending_query == "Transformer"
                assert app._applied_query == "Transformer"
                assert option_list.option_count == 1


class TestArxivApiErrorHandling:
    """Unit tests for arXiv API rate limiting and exception cleanup paths."""

    @staticmethod
    def _make_minimal_app():
        from unittest.mock import MagicMock

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig()
        app._arxiv_api_fetch_inflight = False
        app._arxiv_api_loading = False
        app._arxiv_api_request_token = 0
        app._last_arxiv_api_request_at = 0.0
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._apply_arxiv_search_results = MagicMock()
        return app

    @pytest.mark.asyncio
    async def test_apply_arxiv_rate_limit_waits_when_too_soon(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from arxiv_browser.browser.core import ArxivBrowser

        app = self._make_minimal_app()
        app.notify = MagicMock()
        app._last_arxiv_api_request_at = 100.0

        class FakeLoop:
            def __init__(self) -> None:
                self._calls = 0

            def time(self) -> float:
                self._calls += 1
                if self._calls == 1:
                    return 101.0
                return 104.5

        fake_loop = FakeLoop()
        with (
            patch(
                "arxiv_browser.actions.search_api_actions.asyncio.get_running_loop",
                return_value=fake_loop,
            ),
            patch(
                "arxiv_browser.actions.search_api_actions.asyncio.sleep", new_callable=AsyncMock
            ) as sleep_mock,
        ):
            await ArxivBrowser._apply_arxiv_rate_limit(app)

        sleep_mock.assert_awaited_once()
        assert sleep_mock.await_args.args[0] == pytest.approx(2.0)
        app.notify.assert_called_once()
        assert app._last_arxiv_api_request_at == pytest.approx(104.5)

    @pytest.mark.asyncio
    async def test_run_arxiv_search_value_error_cleans_loading_state(self):
        from unittest.mock import AsyncMock

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import ArxivSearchRequest

        app = self._make_minimal_app()
        app._fetch_arxiv_api_page = AsyncMock(side_effect=ValueError("bad query"))

        await ArxivBrowser._run_arxiv_search(app, ArxivSearchRequest(query="bad"), start=0)

        assert app._arxiv_api_fetch_inflight is False
        assert app._arxiv_api_loading is False
        app._apply_arxiv_search_results.assert_not_called()
        app.notify.assert_called_once()
        assert "bad query" in app.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_run_arxiv_search_http_status_error_cleans_loading_state(self):
        from unittest.mock import AsyncMock

        import httpx

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import ArxivSearchRequest

        app = self._make_minimal_app()
        request = httpx.Request("GET", "https://export.arxiv.org/api/query")
        response = httpx.Response(status_code=503, request=request)
        app._fetch_arxiv_api_page = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "service unavailable", request=request, response=response
            )
        )

        await ArxivBrowser._run_arxiv_search(app, ArxivSearchRequest(query="test"), start=0)

        assert app._arxiv_api_fetch_inflight is False
        assert app._arxiv_api_loading is False
        app._apply_arxiv_search_results.assert_not_called()
        app.notify.assert_called_once()
        assert "HTTP 503" in app.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_run_arxiv_search_http_error_cleans_loading_state(self):
        from unittest.mock import AsyncMock

        import httpx

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import ArxivSearchRequest

        app = self._make_minimal_app()
        request = httpx.Request("GET", "https://export.arxiv.org/api/query")
        app._fetch_arxiv_api_page = AsyncMock(
            side_effect=httpx.ConnectError("network down", request=request)
        )

        await ArxivBrowser._run_arxiv_search(app, ArxivSearchRequest(query="test"), start=0)

        assert app._arxiv_api_fetch_inflight is False
        assert app._arxiv_api_loading is False
        app._apply_arxiv_search_results.assert_not_called()
        app.notify.assert_called_once()
        message = app.notify.call_args.args[0]
        assert "Could not run arXiv API search." in message
        assert "Why:" in message
        assert "Next step:" in message


@pytest.mark.integration
class TestStatusFilterIntegration:
    """Migrated regression tests using Textual Pilot instead of fragile internals.

    Replaces the tests from TestStatusFilterRegressions that used private
    attributes (_Static__content) and monkey-patched query_one.
    """

    @staticmethod
    def _make_app(make_paper):
        from arxiv_browser.browser.core import ArxivBrowser

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
        with patch_save_config(return_value=True):
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
        with patch_save_config(return_value=True):
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
        with patch_save_config(return_value=True):
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
