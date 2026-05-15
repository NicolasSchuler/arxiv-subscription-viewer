"""Tests for the local-first paper comparison view."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arxiv_browser.actions import comparison_actions
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals import PaperComparisonScreen
from arxiv_browser.models import UserConfig
from tests.support.app_stubs import _new_app_stub


def _make_compare_app(make_paper, *, selected_count: int = 2):
    papers = [
        make_paper(arxiv_id=f"2401.6200{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
        for i in range(4)
    ]
    app = _new_app_stub()
    app._config = UserConfig(llm_command="echo {prompt}", llm_timeout=13)
    app._llm_provider = object()
    app.all_papers = papers
    app.filtered_papers = papers
    app._papers_by_id = {paper.arxiv_id: paper for paper in papers}
    app.selected_ids = {paper.arxiv_id for paper in papers[:selected_count]}
    app._get_abstract_text = MagicMock(side_effect=lambda paper, **_kwargs: paper.abstract)
    app._get_current_paper = MagicMock(return_value=papers[0])
    app._get_active_query = MagicMock(return_value="")
    app.push_screen = MagicMock()
    llm_service = SimpleNamespace(compare_papers=AsyncMock(return_value=("## Methods\nok", None)))
    app._get_services = MagicMock(return_value=SimpleNamespace(llm=llm_service))
    app._capture_dataset_epoch = MagicMock(return_value=1)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    return app, papers, llm_service


def test_selected_papers_for_comparison_preserves_visible_then_hidden_order(make_paper):
    app, papers, _service = _make_compare_app(make_paper, selected_count=3)
    app.filtered_papers = [papers[1], papers[0]]

    selected = comparison_actions._selected_papers_for_comparison(app)

    assert [paper.arxiv_id for paper in selected] == [
        papers[1].arxiv_id,
        papers[0].arxiv_id,
        papers[2].arxiv_id,
    ]


@pytest.mark.parametrize("selected_count", [0, 1, 4])
def test_action_compare_papers_warns_for_wrong_selection_count(make_paper, selected_count):
    app, papers, _service = _make_compare_app(make_paper)
    app.selected_ids = {paper.arxiv_id for paper in papers[:selected_count]}

    comparison_actions.action_compare_papers(app)

    app.push_screen.assert_not_called()
    assert "Select 2 or 3 papers" in app.notify.call_args.args[0]


@pytest.mark.parametrize("selected_count", [2, 3])
def test_action_compare_papers_opens_local_modal(make_paper, selected_count):
    app, _papers, _service = _make_compare_app(make_paper, selected_count=selected_count)

    comparison_actions.action_compare_papers(app)

    modal = app.push_screen.call_args.args[0]
    assert isinstance(modal, PaperComparisonScreen)
    assert len(modal._papers) == selected_count


def test_action_compare_papers_warns_for_stale_selected_ids(make_paper):
    app, papers, _service = _make_compare_app(make_paper)
    app.selected_ids = {papers[0].arxiv_id, "missing"}

    comparison_actions.action_compare_papers(app)

    app.push_screen.assert_not_called()
    assert "no longer available" in app.notify.call_args.args[0]


def test_compare_palette_availability_is_local_first(make_paper):
    app, papers, _service = _make_compare_app(make_paper)
    app._config.llm_command = ""
    app.selected_ids = {papers[0].arxiv_id}
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "compare_papers"
    )
    assert command.enabled is False
    assert command.blocked_reason == "Select exactly 2 or 3 papers first"

    app.selected_ids = {papers[0].arxiv_id, papers[1].arxiv_id}
    command = next(
        cmd for cmd in app._build_command_palette_commands() if cmd.action == "compare_papers"
    )
    assert command.enabled is True


def test_request_ai_comparison_without_llm_config_leaves_modal_open(make_paper):
    app, papers, _service = _make_compare_app(make_paper)
    app._require_llm_command = MagicMock(return_value=None)
    screen = MagicMock(spec=PaperComparisonScreen)

    comparison_actions._request_paper_comparison_ai(app, screen, papers[:2])

    screen.set_ai_idle.assert_called_once()
    app._get_services.assert_not_called()


@pytest.mark.parametrize("trusted", [False, True])
def test_request_ai_comparison_respects_trust_gate(make_paper, trusted):
    app, papers, _service = _make_compare_app(make_paper)
    app._require_llm_command = MagicMock(return_value="echo {prompt}")
    app._ensure_llm_command_trusted = MagicMock(return_value=trusted)
    screen = MagicMock(spec=PaperComparisonScreen, ai_running=False)
    tracked = []

    def _track(coro):
        tracked.append(coro)
        coro.close()

    app._track_dataset_task = MagicMock(side_effect=_track)

    comparison_actions._request_paper_comparison_ai(app, screen, papers[:2])

    app._ensure_llm_command_trusted.assert_called_once()
    assert bool(tracked) is trusted


def test_start_ai_comparison_guards_duplicate_and_missing_provider(make_paper):
    app, papers, _service = _make_compare_app(make_paper)
    screen = MagicMock(spec=PaperComparisonScreen, ai_running=True)

    comparison_actions._start_paper_comparison_ai(app, screen, papers[:2])
    assert "already generating" in app.notify.call_args.args[0]

    app.notify.reset_mock()
    screen.ai_running = False
    app._llm_provider = None
    comparison_actions._start_paper_comparison_ai(app, screen, papers[:2])
    screen.set_ai_error.assert_called_once_with("LLM provider unavailable")


def test_start_ai_comparison_schedules_background_task(make_paper):
    app, papers, _service = _make_compare_app(make_paper)
    screen = MagicMock(spec=PaperComparisonScreen, ai_running=False)
    tracked = []

    def _track(coro):
        tracked.append(coro)
        coro.close()

    app._track_dataset_task = MagicMock(side_effect=_track)

    comparison_actions._start_paper_comparison_ai(app, screen, papers[:2])

    screen.set_ai_loading.assert_called_once_with()
    assert tracked


@pytest.mark.asyncio
async def test_generate_ai_comparison_success_updates_open_modal(make_paper):
    app, papers, service = _make_compare_app(make_paper)
    screen = MagicMock(spec=PaperComparisonScreen, is_mounted=True)

    await comparison_actions._generate_paper_comparison_async(app, screen, papers[:2], object())

    service.compare_papers.assert_awaited_once()
    kwargs = service.compare_papers.await_args.kwargs
    assert kwargs["timeout_seconds"] == 13
    assert kwargs["max_content_chars"] == 12_000
    screen.set_ai_result.assert_called_once_with("## Methods\nok")
    assert "generated" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_generate_ai_comparison_failure_and_stale_modal_paths(make_paper):
    app, papers, service = _make_compare_app(make_paper)
    service.compare_papers.return_value = (None, "boom")
    screen = MagicMock(spec=PaperComparisonScreen, is_mounted=True)

    await comparison_actions._generate_paper_comparison_async(app, screen, papers[:2], object())

    screen.set_ai_error.assert_called_once_with("boom")

    app, papers, service = _make_compare_app(make_paper)
    app._is_current_dataset_epoch = MagicMock(return_value=False)
    stale_screen = MagicMock(spec=PaperComparisonScreen, is_mounted=True)
    await comparison_actions._generate_paper_comparison_async(
        app, stale_screen, papers[:2], object()
    )
    stale_screen.set_ai_result.assert_not_called()

    app, papers, service = _make_compare_app(make_paper)
    closed_screen = MagicMock(spec=PaperComparisonScreen, is_mounted=False)
    await comparison_actions._generate_paper_comparison_async(
        app, closed_screen, papers[:2], object()
    )
    closed_screen.set_ai_result.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_error", "expected"),
    [("x" * 300, "x" * 200), (None, "LLM command failed")],
)
async def test_generate_ai_comparison_truncates_or_defaults_provider_error(
    make_paper,
    service_error,
    expected,
):
    app, papers, service = _make_compare_app(make_paper)
    service.compare_papers.return_value = (None, service_error)
    screen = MagicMock(spec=PaperComparisonScreen, is_mounted=True)

    await comparison_actions._generate_paper_comparison_async(app, screen, papers[:2], object())

    screen.set_ai_error.assert_called_once_with(expected)
    app.notify.assert_called_once_with(
        expected,
        title="Paper Comparison",
        severity="error",
        timeout=8,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("side_effect", [RuntimeError("recoverable"), Exception("unexpected")])
async def test_generate_ai_comparison_exception_paths_leave_modal_recoverable(
    make_paper, side_effect
):
    app, papers, service = _make_compare_app(make_paper)
    service.compare_papers.side_effect = side_effect
    screen = MagicMock(spec=PaperComparisonScreen, is_mounted=True)

    await comparison_actions._generate_paper_comparison_async(app, screen, papers[:2], object())

    screen.set_ai_error.assert_called_once_with("Paper comparison failed")
    assert "failed" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_generate_ai_comparison_reraises_cancellation(make_paper):
    app, papers, service = _make_compare_app(make_paper)
    service.compare_papers.side_effect = asyncio.CancelledError()
    screen = MagicMock(spec=PaperComparisonScreen, is_mounted=True)

    with pytest.raises(asyncio.CancelledError):
        await comparison_actions._generate_paper_comparison_async(app, screen, papers[:2], object())


@pytest.mark.asyncio
async def test_ctrl_v_opens_comparison_modal_in_textual_flow(make_paper):
    papers = [
        make_paper(arxiv_id="2401.63001", title="First"),
        make_paper(arxiv_id="2401.63002", title="Second"),
    ]
    app = ArxivBrowser(papers, restore_session=False)

    async with app.run_test() as pilot:
        await pilot.press("space")
        await pilot.press("j")
        await pilot.press("space")
        await pilot.press("ctrl+v")
        await pilot.pause(0.05)

        assert isinstance(app.screen, PaperComparisonScreen)
