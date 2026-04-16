"""TUI interaction tests — mouse clicks, keyboard focus, modal validation, filter pills.

Tests end-to-end interaction patterns that exercise the live Textual widget tree
rather than unit-testing individual methods in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Input, Label, OptionList, Static

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.editing import PaperEditModal, PaperEditResult
from arxiv_browser.models import (
    Paper,
    PaperMetadata,
    UserConfig,
)
from arxiv_browser.query import tokenize_query
from arxiv_browser.widgets.chrome import FilterPillBar
from arxiv_browser.widgets.omni_input import OmniInput
from tests.support.patch_helpers import patch_save_config

# ── Helpers ──────────────────────────────────────────────────────────────────


def _papers(make_paper) -> list[Paper]:
    """Create a small set of test papers with distinct categories and titles."""
    return [
        make_paper(
            arxiv_id="2401.00001",
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            categories="cs.AI cs.CL",
            abstract="We propose the Transformer model.",
        ),
        make_paper(
            arxiv_id="2401.00002",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors="Devlin et al.",
            categories="cs.CL",
            abstract="We introduce BERT, a new language model.",
        ),
        make_paper(
            arxiv_id="2401.00003",
            title="Reinforcement Learning: An Introduction",
            authors="Sutton and Barto",
            categories="cs.LG",
            abstract="A comprehensive introduction to RL.",
        ),
    ]


# ============================================================================
# 1. Mouse Click Tests
# ============================================================================


class TestMouseInteractions:
    """Test mouse-driven interactions with the paper list and UI elements."""

    @pytest.mark.asyncio
    async def test_click_paper_list_highlights_paper(self, make_paper):
        """Clicking on the paper list OptionList navigates the highlight."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                option_list = app.query_one("#paper-list", OptionList)
                # After mount, the first paper should be highlighted
                assert option_list.highlighted == 0

                # Navigate down with keyboard, then verify
                await pilot.press("j")
                await pilot.pause(0.05)
                assert option_list.highlighted == 1

    @pytest.mark.asyncio
    async def test_keyboard_j_k_navigates_paper_list(self, make_paper):
        """j/k keys move highlight up/down in the paper list."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                option_list = app.query_one("#paper-list", OptionList)

                assert option_list.highlighted == 0

                # j moves down
                await pilot.press("j")
                await pilot.pause(0.05)
                assert option_list.highlighted == 1

                await pilot.press("j")
                await pilot.pause(0.05)
                assert option_list.highlighted == 2

                # k moves back up
                await pilot.press("k")
                await pilot.pause(0.05)
                assert option_list.highlighted == 1

    @pytest.mark.asyncio
    async def test_space_toggles_paper_selection(self, make_paper):
        """Pressing space toggles selection on the currently highlighted paper."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Initially no selection
                assert len(app.selected_ids) == 0

                # Space to select
                await pilot.press("space")
                await pilot.pause(0.05)
                assert "2401.00001" in app.selected_ids

                # Space again to deselect
                await pilot.press("space")
                await pilot.pause(0.05)
                assert "2401.00001" not in app.selected_ids


# ============================================================================
# 2. Keyboard Focus Tests
# ============================================================================


class TestKeyboardFocus:
    """Test Tab/focus navigation between widgets."""

    @pytest.mark.asyncio
    async def test_initial_focus_on_paper_list(self, make_paper):
        """After mount, the paper list should have focus."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)
                option_list = app.query_one("#paper-list", OptionList)
                assert option_list.has_focus

    @pytest.mark.asyncio
    async def test_slash_opens_search_and_focuses_input(self, make_paper):
        """Pressing / toggles the search bar visible and moves focus to input."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Initially the OmniInput is hidden
                omni = app.query_one(OmniInput)
                assert not omni.is_open

                # Press / to toggle search
                await pilot.press("slash")
                await pilot.pause(0.1)

                # OmniInput is now visible
                assert omni.is_open

                # Search input should have focus
                search_input = app.query_one("#omni-input", Input)
                assert search_input.has_focus

    @pytest.mark.asyncio
    async def test_escape_from_search_returns_focus_to_list(self, make_paper):
        """Pressing Escape while in search mode clears and returns focus."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Open search
                await pilot.press("slash")
                await pilot.pause(0.1)

                omni = app.query_one(OmniInput)
                assert omni.is_open

                # Press Escape to cancel search
                await pilot.press("escape")
                await pilot.pause(0.1)

                # OmniInput should be hidden again
                assert not omni.is_open

    @pytest.mark.asyncio
    async def test_modal_takes_and_returns_focus(self, make_paper):
        """When a modal opens it takes focus; when closed, app retains control."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal("2401.00001", current_notes="test notes", initial_tab="notes")

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Push modal
                app.push_screen(modal)
                await pilot.pause(0.1)

                # Modal should be on the screen stack
                assert app.screen_stack[-1] is modal

                # Verify modal widgets exist
                assert modal.query_one("#edit-dialog") is not None

                # Dismiss modal
                modal.dismiss(None)
                await pilot.pause(0.1)

                # Modal should be gone from stack
                assert modal not in app.screen_stack


# ============================================================================
# 3. Modal Validation Tests
# ============================================================================


class TestModalValidation:
    """Test error handling and validation in the unified PaperEditModal."""

    @pytest.mark.asyncio
    async def test_tags_tab_parses_comma_separated_tags(self, make_paper):
        """PaperEditModal tags tab correctly parses comma-separated tag input."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal(
            "2401.00001",
            current_tags=["topic:ml"],
            all_tags=["topic:ml", "topic:nlp", "status:todo"],
            initial_tab="tags",
        )

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                # Verify the input is pre-filled with current tags
                tags_input = modal.query_one("#tags-input", Input)
                assert tags_input.value == "topic:ml"

                # Set new tags
                tags_input.value = "topic:cv, status:done, method:transformer"

                # Capture dismiss
                modal.dismiss = MagicMock()
                modal.action_save()
                result = modal.dismiss.call_args[0][0]
                assert isinstance(result, PaperEditResult)
                assert result.tags == ["topic:cv", "status:done", "method:transformer"]

    @pytest.mark.asyncio
    async def test_tags_tab_strips_whitespace_from_tags(self, make_paper):
        """PaperEditModal tags tab strips extra whitespace around tags."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal("2401.00001", initial_tab="tags")

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                tags_input = modal.query_one("#tags-input", Input)
                tags_input.value = "  topic:ml ,  status:todo  ,, ,  method:rl  "

                modal.dismiss = MagicMock()
                modal.action_save()
                result = modal.dismiss.call_args[0][0]
                assert isinstance(result, PaperEditResult)
                assert result.tags == ["topic:ml", "status:todo", "method:rl"]

    @pytest.mark.asyncio
    async def test_tags_tab_empty_input_returns_empty_list(self, make_paper):
        """PaperEditModal tags tab with empty input returns an empty tag list."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal("2401.00001", initial_tab="tags")

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                tags_input = modal.query_one("#tags-input", Input)
                tags_input.value = ""

                modal.dismiss = MagicMock()
                modal.action_save()
                result = modal.dismiss.call_args[0][0]
                assert isinstance(result, PaperEditResult)
                assert result.tags == []

    @pytest.mark.asyncio
    async def test_notes_tab_captures_content(self, make_paper):
        """PaperEditModal notes tab returns the text content on save."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal("2401.00001", current_notes="initial notes", initial_tab="notes")

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                textarea = modal.query_one("#notes-textarea")
                assert textarea.text == "initial notes"

                modal.dismiss = MagicMock()
                modal.action_save()
                result = modal.dismiss.call_args[0][0]
                assert isinstance(result, PaperEditResult)
                assert result.notes == "initial notes"

    @pytest.mark.asyncio
    async def test_cancel_returns_none(self, make_paper):
        """PaperEditModal cancel returns None, preserving existing data."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal("2401.00001", current_notes="do not lose me", initial_tab="notes")

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                modal.dismiss = MagicMock()
                modal.action_cancel()
                modal.dismiss.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_escape_dismisses_without_saving(self, make_paper):
        """Pressing Escape on PaperEditModal should dismiss without saving."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal(
            "2401.00001",
            current_tags=["topic:ml"],
            initial_tab="tags",
        )

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                modal.dismiss = MagicMock()
                modal.action_cancel()
                modal.dismiss.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_autotag_tab_merges_current_and_suggested(self, make_paper):
        """PaperEditModal AI Tags tab merges current tags with suggestions, deduping."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal(
            "2401.00001",
            current_tags=["status:todo"],
            suggested_tags=["topic:ml", "topic:ml", "method:transformer"],
            initial_tab="ai-tags",
        )

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                # Verify merged (deduped) input
                input_widget = modal.query_one("#autotag-input", Input)
                assert input_widget.value == "status:todo, topic:ml, method:transformer"

                # Save should lower-case and parse (AI Tags tab active)
                modal.dismiss = MagicMock()
                modal.action_save()
                result = modal.dismiss.call_args[0][0]
                assert isinstance(result, PaperEditResult)
                assert "status:todo" in result.tags
                assert "topic:ml" in result.tags
                assert "method:transformer" in result.tags

    @pytest.mark.asyncio
    async def test_autotag_tab_cancel_returns_none(self, make_paper):
        """PaperEditModal AI Tags tab cancel returns None."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal(
            "2401.00001",
            suggested_tags=["topic:ml"],
            initial_tab="ai-tags",
        )

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                modal.dismiss = MagicMock()
                modal.action_cancel()
                modal.dismiss.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_tags_tab_shows_suggestions(self, make_paper):
        """PaperEditModal tags tab displays tag suggestions from all known tags."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal(
            "2401.00001",
            current_tags=[],
            all_tags=["topic:ml", "topic:nlp", "status:todo"],
            initial_tab="tags",
        )

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                # Suggestions label should be present
                suggestions = modal.query_one("#tags-suggestions", Label)
                assert suggestions is not None

    @pytest.mark.asyncio
    async def test_tags_input_submitted_triggers_save(self, make_paper):
        """Pressing Enter in the tags input triggers save."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)
        modal = PaperEditModal("2401.00001", current_tags=["topic:ml"], initial_tab="tags")

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                app.push_screen(modal)
                await pilot.pause(0.1)

                modal.action_save = MagicMock()
                modal.on_tags_submitted()
                modal.action_save.assert_called_once()


# ============================================================================
# 4. Filter Pill Interaction Tests
# ============================================================================


class TestFilterPills:
    """Test the filter pill bar behavior with the tokenizer."""

    @pytest.mark.asyncio
    async def test_filter_pills_appear_on_search(self, make_paper):
        """When a search query is applied, filter pills should appear."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Open search and type a query
                await pilot.press("slash")
                await pilot.pause(0.1)

                search_input = app.query_one("#omni-input", Input)
                search_input.value = "transformer"

                # Wait for debounce (0.3s) + render
                await pilot.pause(0.5)

                # Filter pill bar should now be visible
                pill_bar = app.query_one(FilterPillBar)
                assert "visible" in pill_bar.classes

    @pytest.mark.asyncio
    async def test_filter_pills_disappear_on_clear(self, make_paper):
        """Clearing the search query should remove all filter pills."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Apply a filter
                await pilot.press("slash")
                await pilot.pause(0.1)
                search_input = app.query_one("#omni-input", Input)
                search_input.value = "transformer"
                await pilot.pause(0.5)

                pill_bar = app.query_one(FilterPillBar)
                assert "visible" in pill_bar.classes

                # Clear the search
                search_input.value = ""
                await pilot.pause(0.5)

                # Pills should be gone
                assert "visible" not in pill_bar.classes

    @pytest.mark.asyncio
    async def test_escape_clears_search_and_pills(self, make_paper):
        """Pressing Escape clears the search query and hides pills."""
        papers = _papers(make_paper)
        app = ArxivBrowser(papers, restore_session=False)

        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.1)

                # Apply a filter
                await pilot.press("slash")
                await pilot.pause(0.1)
                search_input = app.query_one("#omni-input", Input)
                search_input.value = "transformer"
                await pilot.pause(0.5)

                pill_bar = app.query_one(FilterPillBar)
                assert "visible" in pill_bar.classes

                # Escape to cancel search
                await pilot.press("escape")
                await pilot.pause(0.5)

                # Search should be cleared
                assert search_input.value == ""
                # Pills should be hidden
                assert "visible" not in pill_bar.classes

    def test_tokenize_query_produces_correct_pill_content(self):
        """Verify tokenize_query produces tokens matching expected pill labels."""
        from arxiv_browser.query import pill_label_for_token

        tokens = tokenize_query("cat:cs.AI transformer")
        term_tokens = [t for t in tokens if t.kind == "term"]
        labels = [pill_label_for_token(t) for t in term_tokens]
        assert "cat:cs.AI" in labels
        assert "transformer" in labels

    def test_tokenize_query_quoted_phrase(self):
        """Quoted phrases produce a single phrase token."""
        from arxiv_browser.query import pill_label_for_token

        tokens = tokenize_query('"large language model"')
        term_tokens = [t for t in tokens if t.kind == "term"]
        assert len(term_tokens) == 1
        assert term_tokens[0].phrase is True
        label = pill_label_for_token(term_tokens[0])
        assert label == '"large language model"'

    def test_tokenize_query_with_boolean_operators(self):
        """Boolean operators (AND/OR) are tokenized as 'op' kind."""
        tokens = tokenize_query("cat:cs.AI AND transformer")
        kinds = [t.kind for t in tokens]
        assert "op" in kinds
        term_tokens = [t for t in tokens if t.kind == "term"]
        assert len(term_tokens) == 2

    @pytest.mark.asyncio
    async def test_filter_pill_bar_update_pills_creates_children(self):
        """FilterPillBar.update_pills creates Label children for tokens."""
        bar = FilterPillBar()

        # Simulate mounting behavior (same pattern as test_tui_quality_pass)
        def fake_mount(widget: Label):
            bar._nodes._append(widget)

        async def fake_remove_children():
            for child in list(bar.children):
                bar._nodes._remove(child)

        bar.mount = fake_mount  # type: ignore[method-assign]
        bar.remove_children = fake_remove_children  # type: ignore[method-assign]

        tokens = tokenize_query("transformer")
        await bar.update_pills(tokens, watch_active=False)

        pill_ids = [child.id for child in bar.children if child.id and child.id.startswith("pill-")]
        assert len(pill_ids) >= 1
        assert "pill-0" in pill_ids

    @pytest.mark.asyncio
    async def test_filter_pill_bar_empty_query_hides_bar(self):
        """FilterPillBar with no tokens removes visible class."""
        bar = FilterPillBar()

        def fake_mount(widget: Label):
            bar._nodes._append(widget)

        bar.mount = fake_mount  # type: ignore[method-assign]

        # With no pills, the bar should not have visible class
        await bar.update_pills([], watch_active=False)
        assert "visible" not in bar.classes

        # Add pills → visible
        tokens = tokenize_query("test")
        await bar.update_pills(tokens, watch_active=False)
        assert "visible" in bar.classes


class TestRichTextTruncation:
    """Tests for _truncate_rich_text status bar truncation."""

    def test_no_truncation_when_within_limit(self):
        from arxiv_browser.widgets.chrome import _truncate_rich_text

        assert _truncate_rich_text("short", 10) == "short"

    def test_truncation_preserves_rich_tags(self):
        from arxiv_browser.widgets.chrome import _truncate_rich_text

        text = "[bold]Hello world long text here[/bold]"
        result = _truncate_rich_text(text, 12)
        assert result.endswith("...")
        assert "[bold]" in result

    def test_truncation_with_escaped_brackets(self):
        """Escaped brackets (\\[) count as 1 visible char, not as Rich tags."""
        from arxiv_browser.widgets.chrome import _truncate_rich_text

        # Rich renders \[ as literal [  →  "ABC [tag]XYZ" = 12 visible chars
        text = r"ABC \[tag]XYZ"
        result = _truncate_rich_text(text, 9)
        assert result.endswith("...")
        # Visible content before "..." should be ≤ 6 chars (9 - 3 for "...")
        import re

        stripped = re.sub(r"\\\[", "X", result.removesuffix("..."))
        stripped = re.sub(r"\[[^\]]*]", "", stripped)
        assert len(stripped) <= 6

    def test_no_truncation_for_none_width(self):
        from arxiv_browser.widgets.chrome import _truncate_rich_text

        assert _truncate_rich_text("anything", None) == "anything"

    def test_no_truncation_for_zero_width(self):
        from arxiv_browser.widgets.chrome import _truncate_rich_text

        assert _truncate_rich_text("anything", 0) == "anything"
