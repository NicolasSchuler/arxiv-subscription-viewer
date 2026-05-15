from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Static

from arxiv_browser.actions import ui_actions
from arxiv_browser.authors import (
    author_matches_exact,
    build_author_profile,
    normalize_author_name,
    split_author_names,
)
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.config import _dict_to_config, export_metadata, import_metadata
from arxiv_browser.modals.discovery import AuthorPickerModal, AuthorProfileModal
from arxiv_browser.models import UserConfig
from arxiv_browser.query import (
    build_highlight_terms,
    match_query_term,
    pill_label_for_token,
    remove_query_token,
    tokenize_query,
)
from arxiv_browser.trend_radar import build_trend_report_from_papers
from tests.support.semantic_scholar_helpers import _make_paper as make_s2_paper


def test_exact_author_query_preserves_substring_backwards_compatibility(make_paper) -> None:
    paper = make_paper(authors="Geoffrey Hinton, Alice Smith")

    substring_token = tokenize_query("author:hinton")[0]
    exact_short_token = tokenize_query("author:=hinton")[0]
    exact_phrase_token = tokenize_query('author:="Geoffrey Hinton"')[0]

    assert match_query_term(paper, substring_token, None)
    assert not match_query_term(paper, exact_short_token, None)
    assert match_query_term(paper, exact_phrase_token, None)
    assert pill_label_for_token(exact_phrase_token) == 'author:="Geoffrey Hinton"'
    assert remove_query_token('author:="Geoffrey Hinton" AND unread', 0) == "unread"
    assert build_highlight_terms([exact_phrase_token])["author"] == ["Geoffrey Hinton"]


def test_author_name_helpers_normalize_split_and_match() -> None:
    assert normalize_author_name("  Geoffrey E. Hinton ") == "geoffrey e hinton"
    assert split_author_names("Alice Smith and Bob Jones, Carol Lee") == [
        "Alice Smith",
        "Bob Jones",
        "Carol Lee",
    ]
    assert split_author_names("") == []
    assert author_matches_exact("Alice Smith", "")
    assert author_matches_exact("Alice Smith, Bob Jones", "alice smith")
    assert not author_matches_exact("Alice Smith, Bob Jones", "smith")


def test_tracked_authors_config_round_trip_and_import_dedup() -> None:
    config = UserConfig(tracked_authors=["Alice Smith"])
    exported = export_metadata(config)
    assert exported["tracked_authors"] == ["Alice Smith"]

    fresh = UserConfig(tracked_authors=["alice smith"])
    import_metadata(
        {"format": "arxiv-browser-metadata", "tracked_authors": ["Alice Smith", 42]}, fresh
    )
    assert fresh.tracked_authors == ["alice smith"]

    parsed = _dict_to_config({"tracked_authors": [" Alice Smith ", "alice  smith", 7]})
    assert parsed.tracked_authors == ["Alice Smith"]


def test_author_profile_uses_loaded_papers_and_cached_citations(make_paper) -> None:
    target = make_paper(
        arxiv_id="2401.00001",
        date="Tue, 16 Jan 2024",
        authors="Alice Smith, Bob Jones",
    )
    older = make_paper(
        arxiv_id="2401.00002",
        date="Mon, 15 Jan 2024",
        authors="Carol Lee, Alice Smith",
    )
    miss = make_paper(arxiv_id="2401.00003", authors="Alicia Smyth")
    profile = build_author_profile(
        "alice smith",
        [older, miss, target],
        {"2401.00001": make_s2_paper(arxiv_id="2401.00001", citation_count=9)},
    )

    assert [record.paper.arxiv_id for record in profile.papers] == ["2401.00001", "2401.00002"]
    assert profile.total_cached_citations == 9
    assert profile.citation_coverage == 1
    assert [(item.name, item.count) for item in profile.coauthors] == [
        ("Bob Jones", 1),
        ("Carol Lee", 1),
    ]


def test_tracked_authors_feed_watched_paper_cache(make_paper) -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._config = UserConfig(tracked_authors=["Alice Smith"])
    app._watched_paper_ids = set()
    app.all_papers = [
        make_paper(arxiv_id="2401.00001", authors="Alice Smith"),
        make_paper(arxiv_id="2401.00002", authors="Alicia Smyth"),
    ]

    ArxivBrowser._compute_watched_papers(app)

    assert app._watched_paper_ids == {"2401.00001"}


def test_track_author_action_persists_and_rolls_back_on_failure(make_paper) -> None:
    paper = make_paper(authors="Alice Smith")
    app = SimpleNamespace(
        _config=UserConfig(),
        _get_current_paper=MagicMock(return_value=paper),
        _compute_watched_papers=MagicMock(),
        _apply_filter=MagicMock(),
        _get_active_query=MagicMock(return_value=""),
        notify=MagicMock(),
    )

    with patch.object(ui_actions, "save_config", return_value=True):
        ui_actions.action_track_author(app)
    assert app._config.tracked_authors == ["Alice Smith"]
    app._compute_watched_papers.assert_called_once()

    app._config.tracked_authors = []
    with patch.object(ui_actions, "save_config", return_value=False):
        ui_actions.action_track_author(app)
    assert app._config.tracked_authors == []
    assert app.notify.call_args.kwargs["severity"] == "error"


def test_track_author_action_handles_duplicate_and_missing_author(make_paper) -> None:
    paper = make_paper(authors="Alice Smith")
    app = SimpleNamespace(
        _config=UserConfig(tracked_authors=["Alice Smith"]),
        _get_current_paper=MagicMock(return_value=paper),
        notify=MagicMock(),
    )
    ui_actions.action_track_author(app)
    assert "Already tracking" in app.notify.call_args.args[0]

    app._get_current_paper = MagicMock(return_value=make_paper(authors=""))
    ui_actions.action_track_author(app)
    assert app.notify.call_args.kwargs["severity"] == "warning"

    app._get_current_paper = MagicMock(return_value=None)
    ui_actions.action_author_profile(app)
    assert app.notify.call_args.kwargs["severity"] == "warning"


def test_author_profile_action_prompts_when_current_paper_has_multiple_authors(make_paper) -> None:
    paper = make_paper(authors="Alice Smith, Bob Jones")
    captured: dict[str, object] = {}
    app = SimpleNamespace(
        all_papers=[paper],
        _s2_cache={},
        _get_current_paper=MagicMock(return_value=paper),
        push_screen=lambda modal, callback=None: captured.update(modal=modal, callback=callback),
        notify=MagicMock(),
    )

    ui_actions.action_author_profile(app)

    assert isinstance(captured["modal"], AuthorPickerModal)
    callback = captured["callback"]
    assert callable(callback)


def test_author_profile_action_opens_profile_for_single_author(make_paper) -> None:
    paper = make_paper(authors="Alice Smith")
    captured: dict[str, object] = {}
    app = SimpleNamespace(
        all_papers=[paper],
        _s2_cache={},
        _get_current_paper=MagicMock(return_value=paper),
        push_screen=lambda modal, callback=None: captured.update(modal=modal),
        notify=MagicMock(),
    )

    ui_actions.action_author_profile(app)

    assert isinstance(captured["modal"], AuthorProfileModal)


def test_trend_radar_action_handles_empty_and_success(make_paper) -> None:
    captured: dict[str, object] = {}
    app = SimpleNamespace(_history_files=[], notify=MagicMock(), push_screen=MagicMock())
    ui_actions.action_trend_radar(app)
    assert app.notify.call_args.kwargs["severity"] == "warning"

    report = build_trend_report_from_papers([])
    app._history_files = [object()]
    app.push_screen = lambda modal: captured.update(modal=modal)
    with patch.object(ui_actions, "build_trend_report", return_value=report):
        ui_actions.action_trend_radar(app)
    assert isinstance(captured["modal"], ui_actions.TrendRadarModal)


def test_discovery_action_wrappers_delegate() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    with (
        patch.object(ui_actions, "action_trend_radar") as trend,
        patch.object(ui_actions, "action_author_profile") as profile,
        patch.object(ui_actions, "action_track_author") as track,
    ):
        app.action_trend_radar()
        app.action_author_profile()
        app.action_track_author()
    trend.assert_called_once_with(app)
    profile.assert_called_once_with(app)
    track.assert_called_once_with(app)


@pytest.mark.asyncio
async def test_author_profile_modal_renders_empty_and_cached_states(make_paper) -> None:
    profile = build_author_profile(
        "Alice Smith",
        [make_paper(authors="Alice Smith")],
        {"2401.12345": make_s2_paper(arxiv_id="2401.12345", citation_count=3)},
    )
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = AuthorProfileModal(profile)

    async with app.run_test(size=(70, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        body = modal.query_one("#author-profile-body", Static)
        assert "Cached citations" in str(body.content)
        assert "S2:3" in str(body.content)
        modal.action_close()
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_author_picker_modal_selects_and_handles_empty(make_paper) -> None:
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = AuthorPickerModal(["Alice Smith", "Bob Jones"])

    async with app.run_test(size=(60, 20)) as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        modal.dismiss = MagicMock()
        modal.action_choose()
        assert modal.dismiss.call_args.args[0] == "Alice Smith"

    empty_app = ArxivBrowser([make_paper()], restore_session=False)
    empty_modal = AuthorPickerModal([])
    async with empty_app.run_test(size=(60, 20)) as pilot:
        app.push_screen(empty_modal)
        await pilot.pause(0.05)
        empty_modal.dismiss = MagicMock()
        empty_modal.action_choose()
        empty_modal.dismiss.assert_not_called()


@pytest.mark.asyncio
async def test_author_profile_modal_empty_state(make_paper) -> None:
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = AuthorProfileModal(build_author_profile("Missing Author", []))

    async with app.run_test(size=(60, 20)) as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)
        body = modal.query_one("#author-profile-body", Static)
        assert "No papers" in str(body.content)
