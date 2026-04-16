"""Focused tests for the unified PaperEditModal."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals import (
    PaperEditModal,
)
from arxiv_browser.modals.editing import PaperEditResult


@pytest.mark.asyncio
async def test_paper_edit_modal_notes_tab(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = PaperEditModal("2401.00001", current_notes="existing notes", initial_tab="notes")

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        assert modal.query_one("#edit-dialog") is not None
        assert modal.query_one("#notes-textarea").text == "existing notes"

    focus_target = MagicMock()
    modal.query_one = MagicMock(return_value=focus_target)
    modal.on_mount()
    focus_target.focus.assert_called_once_with()

    modal.dismiss = MagicMock()
    # Mock query_one to return appropriate widgets for action_save
    notes_widget = SimpleNamespace(text="updated")
    tags_widget = SimpleNamespace(value="topic:ml")
    tc_widget = SimpleNamespace(active="notes")

    def mock_query(selector, _type=None):
        if selector == "#notes-textarea":
            return notes_widget
        if selector == "#tags-input":
            return tags_widget
        if "TabbedContent" in str(_type) if _type else False:
            return tc_widget
        return tc_widget

    from textual.widgets import Input, TabbedContent, TextArea

    def mock_query_typed(selector, widget_type=None):
        if widget_type is TextArea or selector == "#notes-textarea":
            return notes_widget
        if widget_type is Input or selector == "#tags-input":
            return tags_widget
        if widget_type is TabbedContent:
            return tc_widget
        return tc_widget

    modal.query_one = mock_query_typed
    modal.action_save()
    result = modal.dismiss.call_args[0][0]
    assert isinstance(result, PaperEditResult)
    assert result.notes == "updated"
    assert result.tags == ["topic:ml"]
    assert result.active_tab == "notes"

    modal.dismiss = MagicMock()
    modal.action_cancel()
    modal.dismiss.assert_called_once_with(None)

    modal.action_save = MagicMock()
    modal.on_save_pressed()
    modal.action_save.assert_called_once_with()

    modal.action_cancel = MagicMock()
    modal.on_cancel_pressed()
    modal.action_cancel.assert_called_once_with()


@pytest.mark.asyncio
async def test_paper_edit_modal_tags_tab(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = PaperEditModal(
        "2401.00001",
        current_tags=["topic:ml"],
        all_tags=["topic:ml", "topic:nlp", "status:todo"],
        initial_tab="tags",
    )

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        assert modal.query_one("#edit-dialog") is not None
        assert modal.query_one("#tags-input").value == "topic:ml"
        assert modal.query_one("#tags-suggestions") is not None

    assert modal._parse_tags(" a, b ,, c ") == ["a", "b", "c"]

    focus_target = MagicMock()
    modal.query_one = MagicMock(return_value=focus_target)
    modal._initial_tab = "tags"
    modal.on_mount()
    focus_target.focus.assert_called_once_with()

    modal.action_save = MagicMock()
    modal.on_tags_submitted()
    modal.action_save.assert_called_once_with()


@pytest.mark.asyncio
async def test_paper_edit_modal_autotag_tab(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = PaperEditModal(
        "2401.00001",
        current_tags=["status:todo"],
        suggested_tags=["topic:ml", "topic:ml", "method:transformer"],
        initial_tab="ai-tags",
    )

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)

        assert modal.query_one("#edit-dialog") is not None
        assert modal.query_one("#edit-title") is not None
        input_widget = modal.query_one("#autotag-input")
        assert input_widget.value == "status:todo, topic:ml, method:transformer"
        assert str(modal.query_one("#save-btn", Button).label) == "Save (Ctrl+S)"
        assert str(modal.query_one("#cancel-btn", Button).label) == "Cancel"

    modal.dismiss = MagicMock()
    modal.action_cancel()
    modal.dismiss.assert_called_once_with(None)

    modal.action_save = MagicMock()
    modal.on_autotag_submitted()
    modal.action_save.assert_called_once_with()
