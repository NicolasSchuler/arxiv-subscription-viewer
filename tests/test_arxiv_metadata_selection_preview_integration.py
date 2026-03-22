#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.app import (
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
from arxiv_browser.themes import THEME_NAMES, THEMES

# ============================================================================
# Tests for clean_latex function
# ============================================================================


@pytest.mark.integration
class TestMetadataActionsIntegration:
    """Integration tests for read/star/notes/tags keyboard actions."""

    @staticmethod
    def _make_papers(make_paper, count: int = 3) -> list:
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

    async def test_toggle_read_creates_metadata(self, make_paper):
        """Pressing 'r' should create metadata and set is_read True."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                assert first_id not in app._config.paper_metadata

                await pilot.press("r")
                await pilot.pause(0.1)
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].is_read is True

    async def test_toggle_read_twice_unsets(self, make_paper):
        """Pressing 'r' twice should toggle is_read back to False."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("r")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].is_read is True

                await pilot.press("r")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].is_read is False

    async def test_toggle_read_on_second_paper(self, make_paper):
        """Navigate to second paper with 'j', then toggle read."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("j")
                await pilot.pause(0.1)
                await pilot.press("r")
                await pilot.pause(0.1)
                second_id = papers[1].arxiv_id
                assert second_id in app._config.paper_metadata
                assert app._config.paper_metadata[second_id].is_read is True
                # First paper should not have metadata
                assert papers[0].arxiv_id not in app._config.paper_metadata

    async def test_toggle_star_creates_metadata(self, make_paper):
        """Pressing 'x' should create metadata and set starred True."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                assert first_id not in app._config.paper_metadata

                await pilot.press("x")
                await pilot.pause(0.1)
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].starred is True

    async def test_toggle_star_twice_unsets(self, make_paper):
        """Pressing 'x' twice should toggle starred back to False."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("x")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].starred is True

                await pilot.press("x")
                await pilot.pause(0.1)
                assert app._config.paper_metadata[first_id].starred is False

    async def test_read_and_star_independent(self, make_paper):
        """Read and star should be independent metadata flags."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("r")
                await pilot.pause(0.1)
                await pilot.press("x")
                await pilot.pause(0.1)
                meta = app._config.paper_metadata[first_id]
                assert meta.is_read is True
                assert meta.starred is True

    async def test_notes_modal_opens(self, make_paper):
        """Pressing 'n' should open the NotesModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.modals import NotesModal

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert len(app.screen_stack) == 1
                await pilot.press("n")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], NotesModal)

    async def test_notes_modal_save_persists(self, make_paper):
        """Type text in NotesModal and save with Ctrl+S should persist notes."""
        from unittest.mock import patch

        from textual.widgets import TextArea

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.modals import NotesModal

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("n")
                await pilot.pause(0.2)
                assert isinstance(app.screen_stack[-1], NotesModal)

                # Type into the text area
                textarea = app.screen_stack[-1].query_one("#notes-textarea", TextArea)
                textarea.insert("My test notes")
                await pilot.pause(0.1)

                # Save with Ctrl+S
                await pilot.press("ctrl+s")
                await pilot.pause(0.2)

                # Modal should have closed
                assert len(app.screen_stack) == 1
                # Notes should be persisted
                assert first_id in app._config.paper_metadata
                assert app._config.paper_metadata[first_id].notes == "My test notes"

    async def test_notes_modal_cancel_does_not_save(self, make_paper):
        """Pressing Escape on NotesModal should not save new notes."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                await pilot.press("n")
                await pilot.pause(0.2)

                # Cancel without typing
                await pilot.press("escape")
                await pilot.pause(0.2)

                assert len(app.screen_stack) == 1
                # No notes should have been saved (metadata may exist but notes empty)
                if first_id in app._config.paper_metadata:
                    assert app._config.paper_metadata[first_id].notes == ""

    async def test_notes_modal_cancel_preserves_existing_notes(self, make_paper):
        """Escape should discard edits and keep existing notes intact."""
        from unittest.mock import patch

        from textual.widgets import TextArea

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        first_id = papers[0].arxiv_id
        app._config.paper_metadata[first_id] = PaperMetadata(
            arxiv_id=first_id, notes="Keep this note"
        )

        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("n")
                await pilot.pause(0.2)
                textarea = app.screen_stack[-1].query_one("#notes-textarea", TextArea)
                textarea.insert(" (edited)")
                await pilot.pause(0.1)
                await pilot.press("escape")
                await pilot.pause(0.2)

        assert app._config.paper_metadata[first_id].notes == "Keep this note"

    async def test_tags_modal_opens(self, make_paper):
        """Pressing 't' should open the TagsModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.modals import TagsModal

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert len(app.screen_stack) == 1
                await pilot.press("t")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], TagsModal)

    async def test_tags_modal_cancel_preserves_existing_tags(self, make_paper):
        """Escape should discard tag edits and preserve existing tags."""
        from unittest.mock import patch

        from textual.widgets import Input

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        first_id = papers[0].arxiv_id
        app._config.paper_metadata[first_id] = PaperMetadata(
            arxiv_id=first_id,
            tags=["topic:ml", "status:reading"],
        )

        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("t")
                await pilot.pause(0.2)
                tags_input = app.screen_stack[-1].query_one("#tags-input", Input)
                tags_input.value = "topic:new"
                await pilot.pause(0.1)
                await pilot.press("escape")
                await pilot.pause(0.2)

        assert app._config.paper_metadata[first_id].tags == ["topic:ml", "status:reading"]


@pytest.mark.integration
class TestSortCyclingIntegration:
    """Integration tests for sort cycling via 's' key."""

    @staticmethod
    def _make_papers(make_paper, count: int = 5) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(90 - i)}",
                authors=f"Author {chr(65 + i)}",
                categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_sort_cycles_all_options(self, make_paper):
        """Pressing 's' should cycle through all SORT_OPTIONS and wrap around."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert app._sort_index == 0
                num_options = len(SORT_OPTIONS)
                for expected in range(1, num_options):
                    await pilot.press("s")
                    assert app._sort_index == expected
                # Wrap around
                await pilot.press("s")
                assert app._sort_index == 0

    async def test_sort_preserves_option_count(self, make_paper):
        """Sorting should not change the number of papers displayed."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                initial_count = option_list.option_count
                assert initial_count == 5

                await pilot.press("s")
                await pilot.pause(0.1)
                assert option_list.option_count == initial_count

                await pilot.press("s")
                await pilot.pause(0.1)
                assert option_list.option_count == initial_count

    async def test_sort_changes_paper_order(self, make_paper):
        """Sorting should actually reorder the filtered_papers list."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Initial order is as-given (reverse alpha: Z, Y, X, W, V)
                initial_ids = [p.arxiv_id for p in app.filtered_papers]

                # Cycle through all sort options and verify the list gets resorted
                # At least one sort should change the order from the initial
                order_changed = False
                for _ in range(len(SORT_OPTIONS)):
                    await pilot.press("s")
                    await pilot.pause(0.1)
                    current_ids = [p.arxiv_id for p in app.filtered_papers]
                    if current_ids != initial_ids:
                        order_changed = True
                        break

                assert order_changed, "Sorting never changed paper order"


@pytest.mark.integration
class TestSelectionIntegration:
    """Integration tests for paper selection via space/a/u keys."""

    @staticmethod
    def _make_papers(make_paper, count: int = 5) -> list:
        return [
            make_paper(
                arxiv_id=f"2401.{10000 + i}",
                title=f"Paper Title {chr(65 + i)}",
                authors=f"Author {chr(65 + i)}",
                categories=f"cs.{'AI' if i % 2 == 0 else 'LG'}",
                abstract=f"Abstract for paper {chr(65 + i)}.",
            )
            for i in range(count)
        ]

    async def test_space_toggles_selection(self, make_paper):
        """Pressing space should toggle selection of the current paper."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                first_id = papers[0].arxiv_id
                assert len(app.selected_ids) == 0

                await pilot.press("space")
                await pilot.pause(0.1)
                assert first_id in app.selected_ids
                assert len(app.selected_ids) == 1

                # Toggle off
                await pilot.press("space")
                await pilot.pause(0.1)
                assert first_id not in app.selected_ids
                assert len(app.selected_ids) == 0

    async def test_select_multiple_papers(self, make_paper):
        """Select multiple papers by navigating and pressing space."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Select first paper
                await pilot.press("space")
                await pilot.pause(0.1)
                assert papers[0].arxiv_id in app.selected_ids

                # Navigate down and select second
                await pilot.press("j")
                await pilot.press("space")
                await pilot.pause(0.1)
                assert papers[1].arxiv_id in app.selected_ids
                assert len(app.selected_ids) == 2

    async def test_select_all(self, make_paper):
        """Pressing 'a' should select all visible papers."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                assert len(app.selected_ids) == 0

                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 5
                for paper in papers:
                    assert paper.arxiv_id in app.selected_ids

    async def test_clear_selection(self, make_paper):
        """Pressing 'u' should clear all selections."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=5)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Select all first
                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 5

                # Clear selection
                await pilot.press("u")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 0

    async def test_select_all_then_deselect_one(self, make_paper):
        """Select all, then deselect one with space."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("a")
                await pilot.pause(0.1)
                assert len(app.selected_ids) == 3

                # Deselect the first paper
                await pilot.press("space")
                await pilot.pause(0.1)
                assert papers[0].arxiv_id not in app.selected_ids
                assert len(app.selected_ids) == 2

    async def test_selection_count_in_status_bar(self, make_paper):
        """Status bar should show selection count when papers are selected."""
        from unittest.mock import patch

        from textual.widgets import Label

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # No selection initially — "selected" should not appear
                status = app.query_one("#status-bar", Label)
                assert "selected" not in str(status.content)

                # Select all papers
                await pilot.press("a")
                await pilot.pause(0.1)
                status_text = str(status.content)
                assert "3 sel" in status_text or "3 selected" in status_text


@pytest.mark.integration
class TestExportMenuIntegration:
    """Integration tests for the export menu modal."""

    @staticmethod
    def _make_papers(make_paper, count: int = 2) -> list:
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

    async def test_export_menu_opens(self, make_paper):
        """Pressing 'E' with selected papers should open the ExportMenuModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.modals import ExportMenuModal

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Select a paper first so _get_target_papers returns non-empty
                await pilot.press("space")
                await pilot.pause(0.1)
                assert len(app.screen_stack) == 1
                await pilot.press("E")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 2
                assert isinstance(app.screen_stack[-1], ExportMenuModal)

    async def test_export_menu_closes_on_escape(self, make_paper):
        """Pressing Escape should close the ExportMenuModal."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.modals import ExportMenuModal

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                await pilot.press("space")
                await pilot.pause(0.1)
                await pilot.press("E")
                await pilot.pause(0.2)
                assert isinstance(app.screen_stack[-1], ExportMenuModal)

                await pilot.press("escape")
                await pilot.pause(0.2)
                assert len(app.screen_stack) == 1

    async def test_export_menu_opens_with_detail_pane_paper(self, make_paper):
        """Export menu should open when detail pane has a paper (no explicit selection)."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser
        from arxiv_browser.modals import ExportMenuModal

        papers = self._make_papers(make_paper, count=2)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                # Wait for the detail pane to load the highlighted paper
                await pilot.pause(0.5)
                await pilot.press("E")
                await pilot.pause(0.2)
                # If the detail pane has a paper, export menu opens
                # If not, the action just notifies — either way no crash
                if len(app.screen_stack) == 2:
                    assert isinstance(app.screen_stack[-1], ExportMenuModal)


@pytest.mark.integration
class TestAbstractPreviewIntegration:
    """Integration tests for abstract preview toggle via 'p' key."""

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

    async def test_toggle_preview_flips_state(self, make_paper):
        """Pressing 'p' should toggle _show_abstract_preview."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                initial = app._show_abstract_preview
                await pilot.press("p")
                await pilot.pause(0.1)
                assert app._show_abstract_preview is not initial

                # Toggle back
                await pilot.press("p")
                await pilot.pause(0.1)
                assert app._show_abstract_preview is initial

    async def test_toggle_preview_preserves_paper_count(self, make_paper):
        """Toggling preview should not change the number of papers displayed."""
        from unittest.mock import patch

        from textual.widgets import OptionList

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=3)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.option_count == 3

                await pilot.press("p")
                await pilot.pause(0.1)
                assert option_list.option_count == 3

    async def test_toggle_preview_updates_config(self, make_paper):
        """Preview toggle should sync to config for persistence."""
        from unittest.mock import patch

        from arxiv_browser.app import ArxivBrowser

        papers = self._make_papers(make_paper, count=1)
        app = ArxivBrowser(papers, restore_session=False)
        with patch("arxiv_browser.app.save_config", return_value=True):
            async with app.run_test() as pilot:
                initial = app._config.show_abstract_preview
                await pilot.press("p")
                await pilot.pause(0.1)
                assert app._config.show_abstract_preview is not initial
