"""Tests for the Debate Paper LLM action."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arxiv_browser import paper_debate_actions
from arxiv_browser.llm import PaperDebateResult
from arxiv_browser.modals import PaperDebateResultModal
from arxiv_browser.models import UserConfig
from tests.support.app_stubs import _new_app_stub


def _make_debate_app(make_paper):
    paper = make_paper(arxiv_id="2401.64001", title="Paper Debate", abstract="Abstract")
    app = _new_app_stub()
    app._config = UserConfig(llm_command="echo {prompt}", llm_timeout=23)
    app._llm_provider = object()
    app.all_papers = [paper]
    app.filtered_papers = [paper]
    app._papers_by_id = {paper.arxiv_id: paper}
    app.selected_ids = set()
    app._get_current_paper = MagicMock(return_value=paper)
    app._get_active_query = MagicMock(return_value="")
    app._fetch_paper_content_async = AsyncMock(return_value="full content")
    llm_service = SimpleNamespace(
        generate_paper_debate=AsyncMock(
            return_value=(PaperDebateResult("advocate", "reviewer"), None)
        )
    )
    app._get_services = MagicMock(return_value=SimpleNamespace(llm=llm_service))
    app._capture_dataset_epoch = MagicMock(return_value=1)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    app.push_screen = MagicMock()
    return app, paper, llm_service


def test_action_debate_paper_requires_config_and_trust(make_paper):
    app, _paper, _service = _make_debate_app(make_paper)
    app._require_llm_command = MagicMock(return_value=None)
    app._ensure_llm_command_trusted = MagicMock()

    paper_debate_actions.action_debate_paper(app)

    app._ensure_llm_command_trusted.assert_not_called()

    app._require_llm_command = MagicMock(return_value="echo {prompt}")
    app._ensure_llm_command_trusted = MagicMock(return_value=False)
    with patch("arxiv_browser.paper_debate_actions._start_paper_debate_flow") as start:
        paper_debate_actions.action_debate_paper(app)
    start.assert_not_called()

    app._ensure_llm_command_trusted = MagicMock(return_value=True)
    with patch("arxiv_browser.paper_debate_actions._start_paper_debate_flow") as start:
        paper_debate_actions.action_debate_paper(app)
    start.assert_called_once_with(app)


def test_start_paper_debate_guards_duplicate_missing_paper_and_provider(make_paper):
    app, _paper, _service = _make_debate_app(make_paper)
    app._paper_debate_active = True

    paper_debate_actions._start_paper_debate_flow(app)

    assert "already in progress" in app.notify.call_args.args[0]

    app.notify.reset_mock()
    app._paper_debate_active = False
    app._get_current_paper = MagicMock(return_value=None)
    paper_debate_actions._start_paper_debate_flow(app)
    assert "No paper selected" in app.notify.call_args.args[0]

    app.notify.reset_mock()
    app._get_current_paper = MagicMock(return_value=_paper)
    app._llm_provider = None
    paper_debate_actions._start_paper_debate_flow(app)
    assert "provider unavailable" in app.notify.call_args.args[0]


def test_start_paper_debate_schedules_background_task(make_paper):
    app, _paper, _service = _make_debate_app(make_paper)
    tracked = []

    def _track(coro):
        tracked.append(coro)
        coro.close()

    app._track_dataset_task = MagicMock(side_effect=_track)

    paper_debate_actions._start_paper_debate_flow(app)

    assert app._paper_debate_active is True
    assert tracked
    app._update_footer.assert_called_once()
    assert "Generating paper debate" in app.notify.call_args.args[0]


def test_paper_debate_palette_availability_requires_llm_and_current_paper(make_paper):
    app, paper, _service = _make_debate_app(make_paper)

    app._config.llm_command = ""
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "debate_paper"
    )
    assert command.enabled is False
    assert command.blocked_reason == "Configure an LLM command first"

    app._config.llm_command = "echo {prompt}"
    app._get_current_paper = MagicMock(return_value=None)
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "debate_paper"
    )
    assert command.enabled is False
    assert command.blocked_reason == "Select a paper first"

    app._get_current_paper = MagicMock(return_value=paper)
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "debate_paper"
    )
    assert command.enabled is True
    assert command.group == "Research"


@pytest.mark.asyncio
async def test_generate_paper_debate_success_opens_result_and_cleans_up(make_paper):
    app, paper, service = _make_debate_app(make_paper)
    app._paper_debate_active = True

    await paper_debate_actions._generate_paper_debate_async(app, paper, app._llm_provider)

    service.generate_paper_debate.assert_awaited_once()
    kwargs = service.generate_paper_debate.await_args.kwargs
    assert kwargs["timeout_seconds"] == 23
    assert kwargs["max_content_chars"] == 12_000
    modal = app.push_screen.call_args.args[0]
    assert isinstance(modal, PaperDebateResultModal)
    assert "Paper debate generated" in app.notify.call_args.args[0]
    assert app._paper_debate_active is False
    assert app._update_footer.called


@pytest.mark.asyncio
async def test_generate_paper_debate_failure_and_stale_paths(make_paper):
    app, paper, service = _make_debate_app(make_paper)
    service.generate_paper_debate.return_value = (None, "boom")
    app._paper_debate_active = True

    await paper_debate_actions._generate_paper_debate_async(app, paper, app._llm_provider)

    app.push_screen.assert_not_called()
    assert "boom" in app.notify.call_args.args[0]
    assert app._paper_debate_active is False

    app, paper, _service = _make_debate_app(make_paper)
    app._is_current_dataset_epoch = MagicMock(return_value=False)
    app._paper_debate_active = True

    await paper_debate_actions._generate_paper_debate_async(app, paper, app._llm_provider)

    app.push_screen.assert_not_called()
    assert app._paper_debate_active is False
    app._update_footer.assert_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side_effect", [RuntimeError("recoverable"), Exception("unexpected")])
async def test_generate_paper_debate_exception_paths_notify(make_paper, side_effect):
    app, paper, service = _make_debate_app(make_paper)
    service.generate_paper_debate.side_effect = side_effect
    app._paper_debate_active = True

    await paper_debate_actions._generate_paper_debate_async(app, paper, app._llm_provider)

    assert "failed" in app.notify.call_args.args[0]
    assert app._paper_debate_active is False


@pytest.mark.asyncio
async def test_generate_paper_debate_reraises_cancellation(make_paper):
    app, paper, service = _make_debate_app(make_paper)
    service.generate_paper_debate.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await paper_debate_actions._generate_paper_debate_async(app, paper, app._llm_provider)
