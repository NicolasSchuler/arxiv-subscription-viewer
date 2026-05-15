"""Tests for quick triage mode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arxiv_browser.actions import triage_actions
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.triage import (
    TRIAGE_LATER_TAG,
    QuickTriageCallbacks,
    QuickTriageItem,
    QuickTriageRequest,
    QuickTriageScreen,
    first_two_abstract_lines,
)
from arxiv_browser.models import MAX_PAPERS_PER_COLLECTION, PaperCollection, PaperMetadata
from tests.support.app_stubs import _new_app_stub
from tests.support.patch_helpers import patch_save_config


def _callbacks(save_result: bool = True) -> QuickTriageCallbacks:
    return QuickTriageCallbacks(
        mark_starred_read=MagicMock(return_value=True),
        mark_skipped=MagicMock(return_value=True),
        tag_later=MagicMock(return_value=True),
        save_to_collection=MagicMock(return_value=save_result),
    )


def _screen(items, callbacks=None, collections=None) -> QuickTriageScreen:
    request = QuickTriageRequest(
        items=items,
        callbacks=callbacks or _callbacks(),
        collections=collections or [],
        papers_by_id={item.paper.arxiv_id: item.paper for item in items},
    )
    screen = QuickTriageScreen(request)
    screen.dismiss = MagicMock()
    return screen


def test_quick_triage_items_preserve_visible_unread_order(make_paper):
    app = _new_app_stub()
    unread = make_paper(arxiv_id="1", abstract="first")
    read = make_paper(arxiv_id="2", abstract="second")
    watched = make_paper(arxiv_id="3", abstract="third")
    app.filtered_papers = [unread, read, watched]
    app._config.paper_metadata = {"2": PaperMetadata(arxiv_id="2", is_read=True)}
    app._relevance_scores = {"1": (8, "match")}
    from arxiv_browser.triage_model import TRIAGE_BUCKET_LIKELY_STAR, TriagePrediction

    app._triage_predictions = {"1": TriagePrediction("1", 0.82, TRIAGE_BUCKET_LIKELY_STAR)}
    app._watched_paper_ids = {"3"}
    app._get_abstract_text = MagicMock(side_effect=lambda paper, allow_async: paper.abstract)

    items = triage_actions._quick_triage_items(app)

    assert [item.paper.arxiv_id for item in items] == ["1", "3"]
    assert items[0].abstract_text == "first"
    assert items[0].relevance == (8, "match")
    assert items[0].triage_prediction is app._triage_predictions["1"]
    assert items[1].watched is True


def test_action_quick_triage_warns_when_queue_empty(make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="1")
    app.filtered_papers = [paper]
    app._config.paper_metadata = {"1": PaperMetadata(arxiv_id="1", is_read=True)}
    app._get_abstract_text = MagicMock(return_value="abstract")
    app._watched_paper_ids = set()
    app.push_screen = MagicMock()

    triage_actions.action_quick_triage(app)

    app.push_screen.assert_not_called()
    assert "No unread papers" in app.notify.call_args.args[0]


def test_mark_starred_read_sets_read_and_starred(make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="1")

    assert triage_actions._mark_starred_read(app, paper) is True

    meta = app._config.paper_metadata["1"]
    assert meta.is_read is True
    assert meta.starred is True
    app._save_config_or_warn.assert_called_once_with("quick triage")


def test_mark_skipped_preserves_existing_star(make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="1")
    app._config.paper_metadata["1"] = PaperMetadata(arxiv_id="1", starred=True)

    assert triage_actions._mark_skipped(app, paper) is True

    meta = app._config.paper_metadata["1"]
    assert meta.is_read is True
    assert meta.starred is True


def test_tag_later_is_idempotent_and_marks_read(make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="1")

    triage_actions._tag_later(app, paper)
    triage_actions._tag_later(app, paper)

    meta = app._config.paper_metadata["1"]
    assert meta.is_read is True
    assert meta.tags == [TRIAGE_LATER_TAG]


def test_save_to_collection_adds_without_duplicates_and_marks_read(make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="1")
    app._config.collections = [PaperCollection(name="Reading", paper_ids=["existing"])]

    assert triage_actions._save_to_collection(app, paper, "Reading") is True
    assert triage_actions._save_to_collection(app, paper, "Reading") is True

    assert app._config.collections[0].paper_ids == ["existing", "1"]
    assert app._config.paper_metadata["1"].is_read is True


def test_save_to_collection_handles_missing_and_full_collection(make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="new")
    app._config.collections = [
        PaperCollection(
            name="Full",
            paper_ids=[str(i) for i in range(MAX_PAPERS_PER_COLLECTION)],
        )
    ]

    assert triage_actions._save_to_collection(app, paper, "Missing") is False
    assert triage_actions._save_to_collection(app, paper, "Full") is False
    assert "new" not in app._config.paper_metadata
    assert app.notify.call_count == 2


def test_quick_triage_screen_decisions_advance_and_return_summary(make_paper):
    paper = make_paper(arxiv_id="1")
    callbacks = _callbacks()
    screen = _screen([QuickTriageItem(paper=paper, abstract_text="abstract")], callbacks)

    screen.action_star_read()

    callbacks.mark_starred_read.assert_called_once_with(paper)
    result = screen.dismiss.call_args.args[0]
    assert result.reviewed == 1
    assert result.counts.starred == 1
    assert result.completed is True


def test_quick_triage_screen_save_reuses_picked_collection(make_paper):
    first = make_paper(arxiv_id="1")
    second = make_paper(arxiv_id="2")
    callbacks = _callbacks()
    items = [
        QuickTriageItem(paper=first, abstract_text="first"),
        QuickTriageItem(paper=second, abstract_text="second"),
    ]
    screen = _screen(items, callbacks, collections=[PaperCollection(name="Reading")])

    screen._on_collection_selected(None)
    assert screen._index == 0
    assert screen._counts.saved == 0

    screen._on_collection_selected("Reading")
    assert screen._index == 1
    assert screen._collection_name == "Reading"

    screen.action_save()
    assert callbacks.save_to_collection.call_args_list[0].args == (first, "Reading")
    assert callbacks.save_to_collection.call_args_list[1].args == (second, "Reading")
    result = screen.dismiss.call_args.args[0]
    assert result.counts.saved == 2


def test_quick_triage_screen_failed_save_does_not_advance(make_paper):
    paper = make_paper(arxiv_id="1")
    callbacks = _callbacks(save_result=False)
    screen = _screen([QuickTriageItem(paper=paper, abstract_text="abstract")], callbacks)
    screen._collection_name = "Reading"

    screen.action_save()

    assert screen._index == 0
    assert screen._counts.saved == 0
    screen.dismiss.assert_not_called()


def test_first_two_abstract_lines_wraps_and_escapes_rich_markup():
    text = " ".join(["[attention]"] * 40)

    rendered = first_two_abstract_lines(text)

    assert "\\[attention]" in rendered
    assert rendered.count("\n") == 1
    assert rendered.endswith("...[/]")


def test_quick_triage_badges_include_ml_prediction(make_paper):
    from arxiv_browser.triage_model import TRIAGE_BUCKET_UNSURE, TriagePrediction

    paper = make_paper(arxiv_id="1")
    screen = _screen(
        [
            QuickTriageItem(
                paper=paper,
                abstract_text="abstract",
                triage_prediction=TriagePrediction("1", 0.46, TRIAGE_BUCKET_UNSURE),
            )
        ]
    )

    assert "ML:?46%" in screen._format_badges(screen._items[0])


@pytest.mark.asyncio
async def test_quick_triage_textual_flow_marks_decisions(make_paper):
    papers = [
        make_paper(arxiv_id="1", title="Star", abstract="star abstract"),
        make_paper(arxiv_id="2", title="Skip", abstract="skip abstract"),
        make_paper(arxiv_id="3", title="Later", abstract="later abstract"),
    ]
    app = ArxivBrowser(papers, restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("T")
            await pilot.pause(0.1)
            await pilot.press("y")
            await pilot.pause(0.05)
            await pilot.press("n")
            await pilot.pause(0.05)
            await pilot.press("t")
            await pilot.pause(0.1)

    assert app._config.paper_metadata["1"].is_read is True
    assert app._config.paper_metadata["1"].starred is True
    assert app._config.paper_metadata["2"].is_read is True
    assert app._config.paper_metadata["2"].starred is False
    assert app._config.paper_metadata["3"].tags == [TRIAGE_LATER_TAG]
