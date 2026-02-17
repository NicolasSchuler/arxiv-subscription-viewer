"""Focused tests for editing-related modals."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arxiv_browser.app import ArxivBrowser, AutoTagSuggestModal, NotesModal, TagsModal


@pytest.mark.asyncio
async def test_notes_modal_compose_and_actions(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = NotesModal("2401.00001", "existing notes")

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        assert modal.query_one("#notes-dialog") is not None
        assert modal.query_one("#notes-textarea").text == "existing notes"

    focus_target = MagicMock()
    modal.query_one = MagicMock(return_value=focus_target)
    modal.on_mount()
    focus_target.focus.assert_called_once_with()

    modal.dismiss = MagicMock()
    modal.query_one = MagicMock(return_value=SimpleNamespace(text="updated"))
    modal.action_save()
    modal.dismiss.assert_called_once_with("updated")

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
async def test_tags_modal_compose_parse_and_handlers(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = TagsModal(
        "2401.00001",
        current_tags=["topic:ml"],
        all_tags=["topic:ml", "topic:nlp", "status:todo"],
    )

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        assert modal.query_one("#tags-dialog") is not None
        assert modal.query_one("#tags-input").value == "topic:ml"
        assert modal.query_one("#tags-suggestions") is not None

    assert modal._parse_tags(" a, b ,, c ") == ["a", "b", "c"]

    focus_target = MagicMock()
    modal.query_one = MagicMock(return_value=focus_target)
    modal.on_mount()
    focus_target.focus.assert_called_once_with()

    modal.dismiss = MagicMock()
    modal.query_one = MagicMock(return_value=SimpleNamespace(value=" topic:cv , status:done "))
    modal.action_save()
    modal.dismiss.assert_called_once_with(["topic:cv", "status:done"])

    modal.dismiss = MagicMock()
    modal.action_cancel()
    modal.dismiss.assert_called_once_with(None)

    modal.action_save = MagicMock()
    modal.on_save_pressed()
    modal.action_save.assert_called_once_with()

    modal.action_cancel = MagicMock()
    modal.on_cancel_pressed()
    modal.action_cancel.assert_called_once_with()

    modal.action_save = MagicMock()
    modal.on_input_submitted()
    modal.action_save.assert_called_once_with()


@pytest.mark.asyncio
async def test_autotag_modal_compose_accept_cancel_and_handlers(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = AutoTagSuggestModal(
        "Very Long Paper Title",
        suggested_tags=["topic:ml", "topic:ml", "method:transformer"],
        current_tags=["status:todo"],
    )

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)

        assert modal.query_one("#autotag-dialog") is not None
        assert modal.query_one("#autotag-title") is not None
        input_widget = modal.query_one("#autotag-input")
        assert input_widget.value == "status:todo, topic:ml, method:transformer"

    modal.dismiss = MagicMock()
    modal.query_one = MagicMock(return_value=SimpleNamespace(value=" Topic:ML,  method:RAG "))
    modal.action_accept()
    modal.dismiss.assert_called_once_with(["topic:ml", "method:rag"])

    modal.dismiss = MagicMock()
    modal.action_cancel()
    modal.dismiss.assert_called_once_with(None)

    modal.action_accept = MagicMock()
    modal.on_accept_pressed()
    modal.action_accept.assert_called_once_with()

    modal.action_cancel = MagicMock()
    modal.on_cancel_pressed()
    modal.action_cancel.assert_called_once_with()
