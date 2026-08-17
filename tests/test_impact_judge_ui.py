"""Rendering, sorting, and discoverability coverage for impact judging."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from arxiv_browser.browser import badge_refresh, dataset_reset
from arxiv_browser.browser.contracts import COMMAND_PALETTE_COMMANDS, _PaletteAppState
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.browser.detail_pane import DetailPaneMixin
from arxiv_browser.browser.runtime_state import EnrichmentScoringRuntimeState
from arxiv_browser.help_ui import HELP_BADGE_LEGEND, HELP_SECTION_ACTIONS
from arxiv_browser.judging import JudgeReasons, JudgeScore
from arxiv_browser.query import (
    PaperSortSignals,
    _build_queue_score_context,
    _queue_score,
    sort_papers,
)
from arxiv_browser.widgets.detail_state import DetailRenderState, _detail_cache_key_for_state
from arxiv_browser.widgets.details import PaperDetails
from arxiv_browser.widgets.listing import PaperRowRenderState, render_paper_option
from tests.support.patch_helpers import patch_save_config


def _score(impact: float = 7.0) -> JudgeScore:
    return JudgeScore(
        impact, 6, 5, 8, 9, JudgeReasons(impact="Useful evidence", novelty="Novel method")
    )


def test_impact_sort_queue_weight_and_legacy_queue_context_compatibility(make_paper) -> None:
    low, high, unknown = (
        make_paper(arxiv_id="low", date="invalid"),
        make_paper(arxiv_id="high", date="invalid"),
        make_paper(arxiv_id="unknown", date="invalid"),
    )
    signals = PaperSortSignals(judge_cache={"low": _score(2), "high": _score(9)})
    assert [paper.arxiv_id for paper in sort_papers([low, unknown, high], "impact", signals)] == [
        "high",
        "low",
        "unknown",
    ]
    context = _build_queue_score_context(
        [high],
        judge_cache={"high": _score(10)},
        relevance_cache={"high": (10, "relevant")},
        today=datetime(2024, 1, 15),
    )
    # Queue relevance is now 30%, judge impact is its own 20% contribution.
    assert _queue_score(high, context) == 0.5


def test_judge_badge_detail_rendering_and_cache_identity(make_paper) -> None:
    paper = make_paper()
    score = replace(_score(), pairwise_score=8.5, pairwise_wins=2, pairwise_matches=3)
    listing = render_paper_option(PaperRowRenderState(paper=paper, judge_score=score))
    assert "J:8.5" in listing

    details = PaperDetails()
    expanded = details._render_judge(score, False)
    collapsed = details._render_judge(score, True)
    assert "AI Impact Judge" in collapsed and "8.5/10" in collapsed
    assert "Pairwise refinement: 2/3 wins" in expanded
    assert "Useful evidence" in expanded and "Novel method" in expanded

    state = DetailRenderState(paper=paper, judge_score=score)
    assert _detail_cache_key_for_state(state, glyph_mode="unicode") != _detail_cache_key_for_state(
        replace(state, judge_score=_score(9)), glyph_mode="unicode"
    )


def test_palette_help_and_runtime_state_expose_judge_feature() -> None:
    assert (
        "Judge Scientific Impact",
        "Locally rate selected or visible papers with the configured LLM",
        "Ctrl+j",
        "judge_impact",
    ) in COMMAND_PALETTE_COMMANDS
    assert "judge_impact" in next(
        actions for title, actions in HELP_SECTION_ACTIONS if "Research" in title
    )


def test_badge_refresh_and_dataset_reset_helpers_cover_all_judge_state() -> None:
    app = SimpleNamespace(
        _s2_active=True,
        _s2_cache={"s2": object()},
        _hf_active=False,
        _hf_cache={"hf": object()},
        _version_updates={"version": (1, 2)},
        _relevance_scores={"rel": (7, "reason")},
        _judge_scores={"judge": _score()},
        _triage_predictions={"triage": object()},
    )
    assert badge_refresh.badge_refresh_ids(app, "s2") == ({"s2"}, False)
    assert badge_refresh.badge_refresh_ids(app, "hf") == (set(), True)
    assert badge_refresh.badge_refresh_ids(app, "version") == ({"version"}, False)
    assert badge_refresh.badge_refresh_ids(app, "relevance") == ({"rel"}, False)
    assert badge_refresh.badge_refresh_ids(app, "judge") == ({"judge"}, False)
    assert badge_refresh.badge_refresh_ids(app, "triage") == ({"triage"}, False)
    assert badge_refresh.badge_refresh_ids(app, "unknown") == (set(), True)
    assert badge_refresh.is_sort_sensitive_badge("judge", "impact")
    assert badge_refresh.is_sort_sensitive_badge("judge", "queue")
    assert not badge_refresh.is_sort_sensitive_badge("judge", "date")

    reset_app = SimpleNamespace(
        _paper_summaries={"x": "summary"},
        _summary_loading={"x"},
        _summary_mode_label={"x": "mode"},
        _summary_command_hash={"x": "hash"},
        _s2_cache={"x": object()},
        _s2_loading={"x"},
        _s2_api_error=True,
        _hf_cache={"x": object()},
        _hf_loading=True,
        _hf_api_error=True,
        _version_updates={"x": (1, 2)},
        _version_checking=True,
        _version_progress=(1, 2),
        _relevance_scores={"x": (8, "reason")},
        _relevance_scoring_active=True,
        _scoring_progress=(1, 2),
        _judge_scores={"x": _score()},
        _judge_scoring_active=True,
        _judge_progress=(1, 2),
        _judge_cancel_requested=True,
        _judge_task=object(),
        _auto_tag_active=True,
        _auto_tag_progress=(1, 2),
        _cancel_batch_requested=True,
    )
    dataset_reset.reset_dataset_enrichment_state(reset_app)
    assert reset_app._judge_scores == {}
    assert reset_app._judge_scoring_active is False
    assert reset_app._judge_progress is None
    assert reset_app._judge_cancel_requested is False
    assert reset_app._judge_task is None
    assert reset_app._paper_summaries == {}
    assert reset_app._s2_cache == {} and reset_app._hf_cache == {}
    assert ("J:n", "Local AI-judge impact score") in HELP_BADGE_LEGEND

    state = _PaletteAppState(
        in_arxiv_api_mode=False,
        hf_active=False,
        watch_filter_active=False,
        show_abstract_preview=False,
        compact_list=False,
        detail_mode="scan",
        active_query="",
        has_history_files=False,
        has_history_navigation=False,
        watch_list=[],
        has_marks=False,
        has_starred=False,
        llm_configured=False,
        has_visible_papers=True,
        has_selection=False,
        selected_count=0,
        has_current_paper=False,
        has_target_papers=False,
        s2_active=False,
        s2_data_loaded=False,
    )
    app = SimpleNamespace(
        _palette_basic_blocked_reason=DetailPaneMixin._palette_basic_blocked_reason
    )
    # The LLM-specific blocker must identify the provider configuration, not a command.
    assert DetailPaneMixin._palette_llm_blocked_reason(app, "judge_impact", state) == "LLM provider"
    http_state = replace(state, judge_llm_configured=True)
    assert DetailPaneMixin._palette_llm_blocked_reason(app, "judge_impact", http_state) == ""
    assert (
        DetailPaneMixin._palette_llm_blocked_reason(app, "generate_summary", http_state)
        == "LLM command"
    )
    hidden_selection = replace(http_state, has_visible_papers=False, has_selection=True)
    assert (
        DetailPaneMixin._palette_basic_blocked_reason(app, "judge_impact", hidden_selection) == ""
    )

    runtime = EnrichmentScoringRuntimeState(cache_db_path=__import__("pathlib").Path("cache.db"))
    assert (
        runtime.judge_scores == {}
        and runtime.judge_progress is None
        and runtime.judge_cancel_requested is False
    )


@pytest.mark.asyncio
async def test_textual_browser_refreshes_impact_badge_and_detail(make_paper) -> None:
    paper = make_paper(arxiv_id="2401.90001")
    app = ArxivBrowser([paper], restore_session=False)
    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            app._judge_scores[paper.arxiv_id] = replace(_score(), pairwise_score=8.5)
            app._mark_badges_dirty("judge", immediate=True)
            app._refresh_detail_pane()
            await pilot.pause(0.05)
            assert app._build_paper_row_state(paper).judge_score is not None
            assert app._build_paper_row_state(paper).judge_score.ranking_score == 8.5
            assert "AI Impact Judge" in str(app._get_paper_details_widget().content)
