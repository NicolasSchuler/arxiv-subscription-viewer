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
from arxiv_browser.llm_providers import LLMResult


@pytest.mark.asyncio
async def test_summary_mode_modal_compose_in_app(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = SummaryModeModal()

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)

        assert modal.query_one("#summary-mode-dialog") is not None
        assert modal.query_one("#summary-mode-title") is not None
        footer = modal.query_one("#summary-mode-footer")
        assert footer is not None
        assert "Cancel: Esc" in str(footer.content)


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


# ── Chat error recovery tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_screen_provider_error_shows_notification(make_paper):
    """When the LLM provider returns success=False, the error is displayed in chat."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    provider = AsyncMock()
    provider.execute.return_value = LLMResult(output="", success=False, error="timeout after 120s")
    screen = PaperChatScreen(make_paper(title="Error paper"), provider, "content")

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        # Seed history as on_question_submitted would before dispatching _ask_llm
        screen._add_message("user", "What is the main result?")
        screen._waiting = True
        await screen._ask_llm("What is the main result?")
        await pilot.pause(0)

        # Provider was called
        provider.execute.assert_awaited_once()

        # Error is recorded in history as an assistant message with markup
        assert len(screen._history) >= 2
        last_role, last_text = screen._history[-1]
        assert last_role == "assistant"
        assert "Error" in last_text
        assert "timeout after 120s" in last_text

        # _waiting must be reset so user can retry
        assert screen._waiting is False

        # Error message is rendered in the chat messages container
        messages = screen.query_one("#chat-messages")
        assert len(messages.children) >= 3  # hint + user msg + error msg


@pytest.mark.asyncio
async def test_chat_screen_empty_response(make_paper):
    """When the LLM returns success=True with empty output, it is handled gracefully."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    provider = AsyncMock()
    provider.execute.return_value = LLMResult(output="", success=True)
    screen = PaperChatScreen(make_paper(title="Empty resp"), provider, "content")

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        screen._add_message("user", "Summarise")
        screen._waiting = True
        await screen._ask_llm("Summarise")
        await pilot.pause(0)

        provider.execute.assert_awaited_once()

        # Empty output still gets appended as assistant message (no crash)
        last_role, last_text = screen._history[-1]
        assert last_role == "assistant"
        assert last_text == ""

        # _waiting is cleared
        assert screen._waiting is False

        # Chat status bar is cleared (no longer shows "Thinking...")
        from textual.widgets import Static

        status = screen.query_one("#chat-status", Static)
        rendered = status.render()
        assert "Thinking" not in str(rendered)


@pytest.mark.asyncio
async def test_chat_screen_long_conversation_history(make_paper):
    """A conversation with 20+ exchanges doesn't crash and history is maintained."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    provider = AsyncMock()
    provider.execute.return_value = LLMResult(output="Response OK", success=True)
    screen = PaperChatScreen(make_paper(title="Long conv"), provider, "content")

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        # Simulate 20 prior exchanges
        for i in range(20):
            screen._add_message("user", f"Question {i}")
            screen._add_message("assistant", f"Answer {i}")
        await pilot.pause(0)

        assert len(screen._history) == 40

        # Now send one more question through _ask_llm
        screen._add_message("user", "Final question")
        screen._waiting = True
        await screen._ask_llm("Final question")
        await pilot.pause(0)

        # Provider received the full context (all 41 prior messages in the prompt)
        provider.execute.assert_awaited_once()
        prompt_sent = provider.execute.call_args[0][0]
        assert "Question 0" in prompt_sent
        assert "Answer 19" in prompt_sent
        assert "Final question" in prompt_sent

        # History now has 42 entries (40 prior + user final + assistant response)
        assert len(screen._history) == 42
        assert screen._history[-1] == ("assistant", "Response OK")
        assert screen._waiting is False

        # All messages rendered without crash
        messages = screen.query_one("#chat-messages")
        # hint + 42 messages = 43 children
        assert len(messages.children) >= 43


@pytest.mark.asyncio
async def test_chat_screen_special_chars_in_question(make_paper):
    """Questions with braces, angle brackets, quotes, and newlines don't crash."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    provider = AsyncMock()
    provider.execute.return_value = LLMResult(output="Fine.", success=True)
    screen = PaperChatScreen(make_paper(title="Special chars"), provider, "content")

    special_question = (
        "What about {prompt} and <tag> and \"quotes\" and 'single' and\nnewlines\nhere?"
    )

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        screen._add_message("user", special_question)
        screen._waiting = True
        await screen._ask_llm(special_question)
        await pilot.pause(0)

        provider.execute.assert_awaited_once()

        # The special chars should be present in the prompt sent to the provider
        prompt_sent = provider.execute.call_args[0][0]
        assert "{prompt}" in prompt_sent
        assert "<tag>" in prompt_sent
        assert '"quotes"' in prompt_sent
        assert "\n" in prompt_sent

        # Response recorded
        assert screen._history[-1] == ("assistant", "Fine.")
        assert screen._waiting is False

        # No crash — messages rendered
        messages = screen.query_one("#chat-messages")
        assert len(messages.children) >= 3  # hint + user + assistant


@pytest.mark.asyncio
async def test_chat_screen_paper_with_special_chars(make_paper):
    """Paper title/abstract with unicode, LaTeX, and quotes don't break context building."""
    app = ArxivBrowser([make_paper()], restore_session=False)
    provider = AsyncMock()
    provider.execute.return_value = LLMResult(output="Understood.", success=True)

    paper = make_paper(
        title='Résumé of "Schrödinger\'s" $\\alpha$-Cat: émigré → Pro™',
        authors="José García, François Müller, 田中太郎",
        abstract=(
            "We prove that $\\mathcal{O}(n \\log n)$ is optimal for "
            "Lévy flights in ℝ³. See §3 & Table «1» for «résultats»."
        ),
        categories="math-ph, cs.AI",
    )
    paper_content = "Full text with ∫∑∏ and ℝ³ and [bold]markup-like[/bold] tokens"
    screen = PaperChatScreen(paper, provider, paper_content)

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        screen._add_message("user", "Explain the theorem")
        screen._waiting = True
        await screen._ask_llm("Explain the theorem")
        await pilot.pause(0)

        provider.execute.assert_awaited_once()

        # Context contains the unicode-rich paper metadata
        prompt_sent = provider.execute.call_args[0][0]
        assert "Schrödinger" in prompt_sent
        assert "$\\alpha$" in prompt_sent
        assert "José García" in prompt_sent
        assert "田中太郎" in prompt_sent
        assert "ℝ³" in prompt_sent
        assert "∫∑∏" in prompt_sent

        # Response recorded
        assert screen._history[-1] == ("assistant", "Understood.")
        assert screen._waiting is False
