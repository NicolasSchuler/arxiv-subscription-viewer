"""Tests for the Paper Remix ideation action."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arxiv_browser.actions import llm_actions
from arxiv_browser.modals import PaperRemixResultModal
from arxiv_browser.models import UserConfig
from tests.support.app_stubs import _new_app_stub


def _make_remix_app(make_paper, *, selected_count: int = 2):
    papers = [
        make_paper(arxiv_id=f"2401.6100{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
        for i in range(3)
    ]
    app = _new_app_stub()
    app._config = UserConfig(
        llm_command="echo {prompt}",
        llm_timeout=11,
        research_interests="adaptive RAG",
    )
    app._llm_provider = object()
    app.all_papers = papers
    app.filtered_papers = papers
    app.selected_ids = {paper.arxiv_id for paper in papers[:selected_count]}
    llm_service = SimpleNamespace(generate_paper_remix=AsyncMock(return_value=("idea", None)))
    app._get_services = MagicMock(return_value=SimpleNamespace(llm=llm_service))
    app._capture_dataset_epoch = MagicMock(return_value=1)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    app.push_screen = MagicMock()
    return app, papers, llm_service


def test_selected_papers_for_remix_prefers_visible_order(make_paper):
    app, papers, _service = _make_remix_app(make_paper, selected_count=3)
    app.filtered_papers = [papers[1], papers[0]]

    selected = llm_actions._selected_papers_for_remix(app)

    assert [paper.arxiv_id for paper in selected] == [
        papers[1].arxiv_id,
        papers[0].arxiv_id,
        papers[2].arxiv_id,
    ]


def test_action_remix_papers_requires_config_and_trust(make_paper):
    app, _papers, _service = _make_remix_app(make_paper)
    app._require_llm_command = MagicMock(return_value=None)
    app._ensure_llm_command_trusted = MagicMock()
    app._start_paper_remix_flow = MagicMock()

    llm_actions.action_remix_papers(app)

    app._ensure_llm_command_trusted.assert_not_called()
    app._start_paper_remix_flow.assert_not_called()

    app._require_llm_command = MagicMock(return_value="echo {prompt}")
    app._ensure_llm_command_trusted = MagicMock(return_value=False)
    llm_actions.action_remix_papers(app)
    app._start_paper_remix_flow.assert_not_called()

    app._ensure_llm_command_trusted = MagicMock(return_value=True)
    llm_actions.action_remix_papers(app)
    app._start_paper_remix_flow.assert_called_once_with()


@pytest.mark.parametrize("selected_count", [0, 1, 4])
def test_start_paper_remix_warns_for_wrong_selection_count(make_paper, selected_count):
    app, papers, _service = _make_remix_app(make_paper)
    app.selected_ids = {f"missing-{index}" for index in range(selected_count)}
    app.all_papers = papers
    app.filtered_papers = papers
    app._track_dataset_task = MagicMock()

    llm_actions._start_paper_remix_flow(app)

    app._track_dataset_task.assert_not_called()
    assert "Select exactly 2 or 3 papers" in app.notify.call_args.args[0]


def test_start_paper_remix_prevents_duplicate_and_missing_provider(make_paper):
    app, _papers, _service = _make_remix_app(make_paper)
    app._paper_remix_active = True

    llm_actions._start_paper_remix_flow(app)

    assert "already in progress" in app.notify.call_args.args[0]

    app.notify.reset_mock()
    app._paper_remix_active = False
    app._llm_provider = None
    llm_actions._start_paper_remix_flow(app)
    assert "provider unavailable" in app.notify.call_args.args[0]


def test_start_paper_remix_schedules_background_task(make_paper):
    app, _papers, _service = _make_remix_app(make_paper)
    tracked = []

    def _track(coro):
        tracked.append(coro)
        coro.close()

    app._track_dataset_task = MagicMock(side_effect=_track)

    llm_actions._start_paper_remix_flow(app)

    assert app._paper_remix_active is True
    assert tracked
    app._update_footer.assert_called_once()
    assert "Generating research idea" in app.notify.call_args.args[0]


def test_paper_remix_palette_availability_requires_llm_and_two_or_three_selected(make_paper):
    app, papers, _service = _make_remix_app(make_paper)
    app._get_active_query = MagicMock(return_value="")
    app._get_current_paper = MagicMock(return_value=papers[0])

    app._config.llm_command = ""
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "remix_papers"
    )
    assert command.enabled is False
    assert command.blocked_reason == "Configure an LLM command first"

    app._config.llm_command = "echo {prompt}"
    app.selected_ids = {papers[0].arxiv_id}
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "remix_papers"
    )
    assert command.enabled is False
    assert command.blocked_reason == "Select exactly 2 or 3 papers first"

    app.selected_ids = {papers[0].arxiv_id, papers[1].arxiv_id}
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "remix_papers"
    )
    assert command.enabled is True


@pytest.mark.asyncio
async def test_generate_paper_remix_success_opens_result_and_cleans_up(make_paper):
    app, papers, service = _make_remix_app(make_paper)
    app._paper_remix_active = True

    await llm_actions._generate_paper_remix_async(app, papers[:2], app._llm_provider)

    service.generate_paper_remix.assert_awaited_once()
    kwargs = service.generate_paper_remix.await_args.kwargs
    assert kwargs["research_interests"] == "adaptive RAG"
    assert kwargs["timeout_seconds"] == 11
    modal = app.push_screen.call_args.args[0]
    assert isinstance(modal, PaperRemixResultModal)
    assert "Research idea generated" in app.notify.call_args.args[0]
    assert app._paper_remix_active is False
    assert app._update_footer.called


@pytest.mark.asyncio
async def test_generate_paper_remix_failure_notifies_and_cleans_up(make_paper):
    app, papers, service = _make_remix_app(make_paper)
    service.generate_paper_remix.return_value = (None, "boom")
    app._paper_remix_active = True

    await llm_actions._generate_paper_remix_async(app, papers[:2], app._llm_provider)

    app.push_screen.assert_not_called()
    assert "boom" in app.notify.call_args.args[0]
    assert app._paper_remix_active is False


@pytest.mark.asyncio
async def test_generate_paper_remix_stale_result_suppresses_modal_but_cleans_up(make_paper):
    app, papers, _service = _make_remix_app(make_paper)
    app._is_current_dataset_epoch = MagicMock(return_value=False)
    app._paper_remix_active = True

    await llm_actions._generate_paper_remix_async(app, papers[:2], app._llm_provider)

    app.push_screen.assert_not_called()
    assert app._paper_remix_active is False
    app._update_footer.assert_called()
