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


class TestWatchListActions:
    """Unit tests for watch list management action behavior."""

    @staticmethod
    def _make_mock_app():
        from unittest.mock import MagicMock

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(watch_list=[WatchListEntry(pattern="old", match_type="title")])
        app._compute_watched_papers = MagicMock()
        app._watch_filter_active = False
        app._watched_paper_ids = set()
        app._apply_filter = MagicMock()
        app.notify = MagicMock()
        app.query_one = MagicMock(return_value=type("SearchInput", (), {"value": "cat:cs.AI"})())
        return app

    def test_manage_watch_list_persists_updates(self):
        from unittest.mock import patch

        app = self._make_mock_app()
        new_entries = [WatchListEntry(pattern="new", match_type="author")]

        def fake_push_screen(_screen, callback):
            callback(new_entries)

        app.push_screen = fake_push_screen
        with patch_save_config(return_value=True) as save_mock:
            app.action_manage_watch_list()

        assert app._config.watch_list == new_entries
        save_mock.assert_called_once_with(app._config)
        app._compute_watched_papers.assert_called_once()
        app._apply_filter.assert_called_once_with("cat:cs.AI")

    def test_manage_watch_list_reverts_on_save_failure(self):
        from unittest.mock import patch

        app = self._make_mock_app()
        old_entries = list(app._config.watch_list)
        new_entries = [WatchListEntry(pattern="new", match_type="author")]

        def fake_push_screen(_screen, callback):
            callback(new_entries)

        app.push_screen = fake_push_screen
        with patch_save_config(return_value=False):
            app.action_manage_watch_list()

        assert app._config.watch_list == old_entries
        app._compute_watched_papers.assert_not_called()
        app._apply_filter.assert_not_called()
        assert "Failed to save watch list" in app.notify.call_args[0][0]


@pytest.mark.integration
class TestWatchFilterIntegration:
    """Integration tests for watch filter toggle via 'w' key."""

    @staticmethod
    def _make_papers(make_paper, count: int = 3) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories="cs.AI",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_watch_filter_toggle_empty_watch_list(self, make_paper):
        """Pressing 'w' with empty watch list should remain inactive."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                assert app._watch_filter_active is False
                await pilot.press("w")
                await pilot.pause(0.1)
                # Should remain False because watch list is empty
                assert app._watch_filter_active is False

    async def test_watch_filter_toggle_with_watch_list(self, make_paper):
        """Pressing 'w' with watch entries should toggle filter on/off."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import WatchListEntry

        papers = self._make_papers(make_paper, count=3)
        config = UserConfig(watch_list=[WatchListEntry(pattern="Author A", match_type="author")])
        app = ArxivBrowser(papers, config=config, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                assert app._watch_filter_active is False
                # Ensure some papers are watched
                assert len(app._watched_paper_ids) > 0

                await pilot.press("w")
                await pilot.pause(0.1)
                assert app._watch_filter_active is True

                # Toggle off
                await pilot.press("w")
                await pilot.pause(0.1)
                assert app._watch_filter_active is False

    async def test_watch_filter_reduces_visible_papers(self, make_paper):
        """Activating watch filter should show only watched papers."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import WatchListEntry

        papers = self._make_papers(make_paper, count=3)
        # Only watch Author A — should match papers[0] only
        config = UserConfig(watch_list=[WatchListEntry(pattern="Author A", match_type="author")])
        app = ArxivBrowser(papers, config=config, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 3

                await pilot.press("w")
                await pilot.pause(0.2)
                # Only watched papers should be visible
                assert option_list.option_count < 3
                assert option_list.option_count >= 1


@pytest.mark.integration
class TestThemeCyclingIntegration:
    """Integration tests for theme cycling via Ctrl+T."""

    @staticmethod
    def _make_papers(make_paper, count: int = 1) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories="cs.AI",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_theme_cycles_to_next(self, make_paper):
        """Pressing Ctrl+T should cycle to the next theme."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import THEME_NAMES as APP_THEME_NAMES

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                initial_theme = app._config.theme_name
                assert initial_theme == "monokai"

                await pilot.press("ctrl+t")
                await pilot.pause(0.1)
                assert app._config.theme_name == APP_THEME_NAMES[1]
                assert app._config.theme_name != initial_theme

    async def test_theme_cycles_wrap_around(self, make_paper):
        """Pressing Ctrl+T len(THEME_NAMES) times should return to the first theme."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import THEME_NAMES as APP_THEME_NAMES

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                initial_theme = app._config.theme_name
                for _ in range(len(APP_THEME_NAMES)):
                    await pilot.press("ctrl+t")
                    await pilot.pause(0.1)
                assert app._config.theme_name == initial_theme

    async def test_theme_cycle_preserves_paper_count(self, make_paper):
        """Theme cycling should not affect the paper list."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.browser.core import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 3

                await pilot.press("ctrl+t")
                await pilot.pause(0.1)
                assert option_list.option_count == 3

    async def test_theme_cycle_refreshes_detail_markup(self, make_paper):
        """Theme cycling should invalidate detail cache and re-render markup colors."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.details import PaperDetails

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                details = app.query_one(PaperDetails)
                app._refresh_detail_pane()
                before = str(details.content)

                await pilot.press("ctrl+t")
                await pilot.pause(0.1)

                after = str(details.content)
                assert before != after


@pytest.mark.integration
class TestHistoryNavigationIntegration:
    """Integration tests for history date navigation via '[' and ']' keys."""

    @staticmethod
    def _make_history_file(tmp_path, date_str: str, paper_id: str) -> Path:
        """Create a minimal arXiv email file for a given date."""
        history_dir = tmp_path / "history"
        history_dir.mkdir(exist_ok=True)
        filepath = history_dir / f"{date_str}.txt"
        content = (
            f"arXiv:{paper_id}\n"
            f"Date: Mon, 15 Jan 2024\n"
            f"Title: Paper for {date_str}\n"
            f"Authors: Author X\n"
            f"Categories: cs.AI\n"
            f"\\\\\n"
            f"  Abstract for {date_str}.\n"
        )
        filepath.write_text(content)
        return filepath

    async def test_prev_date_increments_index(self, make_paper, tmp_path):
        """Pressing '[' should navigate to the older date (higher index)."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        # Create history files
        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")
        f3 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        # History files are newest-first: index 0 = newest
        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
            (dt_date(2024, 1, 15), f3),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )
        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 0

                await pilot.press("bracketleft")
                await pilot.pause(0.3)
                assert app._current_date_index == 1

    async def test_next_date_decrements_index(self, make_paper, tmp_path):
        """Pressing ']' should navigate to the newer date (lower index)."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")
        f3 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
            (dt_date(2024, 1, 15), f3),
        ]

        papers = [make_paper(arxiv_id="2401.00002", title="Paper for 2024-01-16")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=1,
        )
        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 1

                await pilot.press("bracketright")
                await pilot.pause(0.3)
                assert app._current_date_index == 0

    async def test_prev_date_clamps_at_oldest(self, make_paper, tmp_path):
        """Pressing '[' at the oldest date should not go further."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 15), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00001", title="Paper for 2024-01-15")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=1,  # Already at oldest
        )
        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 1

                await pilot.press("bracketleft")
                await pilot.pause(0.2)
                # Should stay at oldest (index 1 for 2 files)
                assert app._current_date_index == 1

    async def test_next_date_clamps_at_newest(self, make_paper, tmp_path):
        """Pressing ']' at the newest date should not go further."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-15", "2401.00001")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 15), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,  # Already at newest
        )
        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 0

                await pilot.press("bracketright")
                await pilot.pause(0.2)
                # Should stay at newest (index 0)
                assert app._current_date_index == 0

    async def test_history_navigation_loads_new_papers(self, make_paper, tmp_path):
        """Navigating dates should load papers from the new date file."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )
        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                assert app.all_papers[0].arxiv_id == "2401.00003"

                await pilot.press("bracketleft")
                await pilot.pause(0.3)
                # After navigating to older date, papers should be from the new file
                assert len(app.all_papers) >= 1
                assert app.all_papers[0].arxiv_id == "2401.00002"

    async def test_not_in_history_mode_ignores_navigation(self, make_paper):
        """Without history files, '[' and ']' should not crash or change state."""
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser

        papers = [make_paper()]
        app = ArxivBrowser(papers, restore_session=False)
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                assert not app._is_history_mode()
                assert app._current_date_index == 0

                await pilot.press("bracketleft")
                await pilot.pause(0.1)
                assert app._current_date_index == 0

                await pilot.press("bracketright")
                await pilot.pause(0.1)
                assert app._current_date_index == 0

    async def test_history_navigation_clears_selection(self, make_paper, tmp_path):
        """Navigating to a new date should clear any paper selections."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )
        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
        ):
            async with app.run_test() as pilot:
                # Select a paper
                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) > 0

                # Navigate to older date
                await pilot.press("bracketleft")
                await pilot.pause(0.3)
                # Selection should be cleared
                assert len(app.selected_ids) == 0

    async def test_prev_date_load_error_rolls_back_index(self, make_paper, tmp_path):
        """Failed date load should keep index on the last successfully loaded date."""
        from datetime import date as dt_date
        from unittest.mock import AsyncMock, patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = self._make_history_file(tmp_path, "2024-01-17", "2401.00003")
        f2 = self._make_history_file(tmp_path, "2024-01-16", "2401.00002")

        history_files = [
            (dt_date(2024, 1, 17), f1),
            (dt_date(2024, 1, 16), f2),
        ]

        papers = [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]
        app = ArxivBrowser(
            papers,
            restore_session=False,
            history_files=history_files,
            current_date_index=0,
        )

        def parse_with_error(path: Path):
            if path == f2:
                msg = "broken file"
                raise OSError(msg)
            return [make_paper(arxiv_id="2401.00003", title="Paper for 2024-01-17")]

        with (
            patch_save_config(return_value=True),
            patch.object(DateNavigator, "update_dates", new_callable=AsyncMock),
            patch("arxiv_browser.browser.browse.parse_arxiv_file", side_effect=parse_with_error),
        ):
            async with app.run_test() as pilot:
                assert app._current_date_index == 0
                await pilot.press("bracketleft")
                await pilot.pause(0.2)
                assert app._current_date_index == 0
                assert app.all_papers[0].arxiv_id == "2401.00003"
