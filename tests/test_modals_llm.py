"""Focused tests for LLM-related modals/screens."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arxiv_browser.app import (
    ArxivBrowser,
    PaperChatScreen,
    ResearchInterestsModal,
    SummaryModeModal,
)


@pytest.mark.asyncio
async def test_summary_mode_modal_compose_in_app(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = SummaryModeModal()

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)

        assert modal.query_one("#summary-mode-dialog") is not None
        assert modal.query_one("#summary-mode-title") is not None
        assert modal.query_one("#summary-mode-footer") is not None


@pytest.mark.asyncio
async def test_research_interests_modal_compose_in_app(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = ResearchInterestsModal("llm systems")

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)

        assert modal.query_one("#interests-dialog") is not None
        assert modal.query_one("#interests-title") is not None
        assert modal.query_one("#interests-textarea").text == "llm systems"


def test_research_interests_modal_mount_actions_and_handlers():
    modal = ResearchInterestsModal("")

    focus_target = MagicMock()
    modal.query_one = MagicMock(return_value=focus_target)
    modal.on_mount()
    focus_target.focus.assert_called_once_with()

    modal.dismiss = MagicMock()
    modal.query_one = MagicMock(return_value=SimpleNamespace(text="  multimodal  "))
    modal.action_save()
    modal.dismiss.assert_called_once_with("multimodal")

    modal.dismiss = MagicMock()
    modal.action_cancel()
    modal.dismiss.assert_called_once_with("")

    modal.action_save = MagicMock()
    modal.on_save_pressed()
    modal.action_save.assert_called_once_with()

    modal.action_cancel = MagicMock()
    modal.on_cancel_pressed()
    modal.action_cancel.assert_called_once_with()


def test_paper_chat_on_mount_hint_branches(make_paper):
    paper = make_paper(title="Paper")

    for paper_content, expected_hint in [
        ("full text", "Paper content loaded. Ask anything!"),
        ("", "Using abstract only (HTML not available). Ask anything!"),
    ]:
        screen = PaperChatScreen(paper, AsyncMock(), paper_content)
        input_widget = MagicMock()
        messages = MagicMock()

        def _query(selector, _type=None, *, input_widget=input_widget, messages=messages):
            if selector == "#chat-input":
                return input_widget
            if selector == "#chat-messages":
                return messages
            raise AssertionError(f"unexpected selector {selector}")

        screen.query_one = MagicMock(side_effect=_query)
        screen.on_mount()

        input_widget.focus.assert_called_once_with()
        mounted = messages.mount.call_args[0][0]
        assert expected_hint in str(mounted.render())


@pytest.mark.asyncio
async def test_paper_chat_question_submit_guards_and_valid_path(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    paper = make_paper(title="Chat paper")
    screen = PaperChatScreen(paper, AsyncMock(), "content")

    tracked = []

    def _track(coro):
        tracked.append(coro)
        coro.close()
        return MagicMock()

    async with app.run_test() as pilot:
        app._track_task = _track  # type: ignore[method-assign]
        app.push_screen(screen)
        await pilot.pause(0.05)

        screen._add_message = MagicMock()
        input_widget = screen.query_one("#chat-input")

        screen._waiting = False
        screen.on_question_submitted(SimpleNamespace(value="   ", input=input_widget))
        screen._add_message.assert_not_called()
        assert tracked == []

        screen._waiting = True
        screen.on_question_submitted(SimpleNamespace(value="Question", input=input_widget))
        screen._add_message.assert_not_called()
        assert tracked == []

        screen._waiting = False
        input_widget.value = "Question"
        screen.on_question_submitted(SimpleNamespace(value="Question", input=input_widget))

        screen._add_message.assert_called_once_with("user", "Question")
        assert tracked
        assert input_widget.value == ""
        assert screen._waiting is True


@pytest.mark.asyncio
async def test_paper_chat_add_message_renders_user_and_assistant(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    screen = PaperChatScreen(make_paper(title="Msg paper"), AsyncMock(), "content")

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        screen._add_message("user", "What changed?")
        screen._add_message("assistant", "New findings")
        await pilot.pause(0)

        messages = screen.query_one("#chat-messages")
        assert len(messages.children) >= 3
        assert screen._history[-2:] == [
            ("user", "What changed?"),
            ("assistant", "New findings"),
        ]


def test_paper_chat_action_close_dismisses_none(make_paper):
    screen = PaperChatScreen(make_paper(title="Close"), AsyncMock(), "")
    screen.dismiss = MagicMock()
    screen.action_close()
    screen.dismiss.assert_called_once_with(None)
