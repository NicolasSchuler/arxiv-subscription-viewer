from __future__ import annotations

from datetime import date

import pytest
from textual.widgets import Static

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.discovery import TrendRadarModal
from arxiv_browser.trend_radar import (
    UNKNOWN_CATEGORY,
    build_trend_report,
    build_trend_report_from_papers,
    primary_category,
    render_sparkline,
)


def test_trend_report_counts_growth_and_dedupes_per_day(make_paper) -> None:
    old_day = date(2026, 1, 1)
    recent_day = date(2026, 1, 2)
    duplicate = make_paper(arxiv_id="2401.00001", categories="cs.AI cs.LG")
    report = build_trend_report_from_papers(
        [
            (
                recent_day,
                [
                    duplicate,
                    duplicate,
                    make_paper(arxiv_id="2401.00002", authors="Alice, Bob", categories="bad??"),
                ],
            ),
            (
                old_day,
                [
                    make_paper(arxiv_id="2401.00003", authors="Alice", categories="cs.LG"),
                    make_paper(arxiv_id="2401.00004", authors="Carol", categories=""),
                ],
            ),
        ],
        recent_window=1,
        top_n=5,
    )

    assert report.dates == (old_day, recent_day)
    assert report.total_papers == 4
    assert report.category_trends[0].category == "cs.AI"
    assert report.category_trends[0].counts == (0, 1)
    assert report.category_trends[0].delta == 1
    unknown = next(row for row in report.category_trends if row.category == UNKNOWN_CATEGORY)
    assert unknown.counts == (1, 1)
    assert report.top_authors[0].name == "Alice"


def test_hot_bigrams_ignore_stopwords_and_sort_ties(make_paper) -> None:
    report = build_trend_report_from_papers(
        [
            (
                date(2026, 1, 1),
                [
                    make_paper(
                        abstract=(
                            "The neural retrieval method improves neural retrieval. "
                            "A graph signal method improves graph signal."
                        )
                    )
                ],
            )
        ],
        top_n=3,
    )

    assert [item.bigram for item in report.hot_bigrams][:2] == [
        "graph signal",
        "neural retrieval",
    ]


def test_sparklines_and_primary_category_edges() -> None:
    assert render_sparkline([], ascii_mode=False) == ""
    assert render_sparkline([0, 0, 0], ascii_mode=False) == "▁▁▁"
    assert render_sparkline([5], ascii_mode=True) == "1"
    assert render_sparkline([-2, 0, 8], ascii_mode=True) == "118"
    assert primary_category("") == UNKNOWN_CATEGORY
    assert primary_category("???") == UNKNOWN_CATEGORY
    assert primary_category("hep-th/9901001") == UNKNOWN_CATEGORY
    assert primary_category("cs.AI cs.LG") == "cs.AI"


def test_trend_report_file_io_and_empty_states(tmp_path) -> None:
    missing = tmp_path / "missing.txt"
    report = build_trend_report([(date(2026, 1, 1), missing)])
    assert report.history_file_count == 1
    assert report.category_trends == ()
    assert build_trend_report_from_papers([]).history_file_count == 0


@pytest.mark.asyncio
async def test_trend_radar_modal_renders_and_closes(make_paper) -> None:
    report = build_trend_report_from_papers(
        [(date(2026, 1, 1), [make_paper(authors="Alice", categories="cs.AI")])]
    )
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = TrendRadarModal(report)

    async with app.run_test(size=(70, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        body = modal.query_one("#trend-radar-body", Static)
        assert "Growing Categories" in str(body.content)
        modal.action_close()
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_trend_radar_modal_empty_history_message(make_paper) -> None:
    report = build_trend_report_from_papers([])
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = TrendRadarModal(report)

    async with app.run_test(size=(60, 20)) as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        body = modal.query_one("#trend-radar-body", Static)
        assert "No local history files" in str(body.content)
