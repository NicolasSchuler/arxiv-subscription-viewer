"""Focused tests for TUI runtime helper seams."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

from textual.css.query import NoMatches

from arxiv_browser.ui_runtime import UiRefreshCoordinator, UiRefs, restore_omni_chrome


def test_ui_refs_reset_clears_all_cached_widgets() -> None:
    refs = UiRefs()
    sentinel = object()
    for field in fields(UiRefs):
        setattr(refs, field.name, sentinel)

    refs.reset()

    assert all(getattr(refs, field.name) is None for field in fields(UiRefs))


def test_ui_refresh_coordinator_filter_refresh_order() -> None:
    events: list[tuple[str, tuple[object, ...]]] = []
    coordinator = UiRefreshCoordinator(
        refresh_list_view=lambda: events.append(("list", ())),
        update_list_header=lambda query: events.append(("header", (query,))),
        update_status_bar=lambda: events.append(("status", ())),
        update_filter_pills=lambda query: events.append(("pills", (query,))),
        refresh_detail_pane=lambda: events.append(("detail", ())),
        refresh_current_list_item=lambda: events.append(("current", ())),
    )

    coordinator.apply_filter_refresh("cat:cs.AI")

    assert events == [
        ("list", ()),
        ("header", ("cat:cs.AI",)),
        ("status", ()),
        ("pills", ("cat:cs.AI",)),
    ]


def test_ui_refresh_coordinator_detail_and_list_item_order() -> None:
    events: list[str] = []
    coordinator = UiRefreshCoordinator(
        refresh_list_view=lambda: events.append("list"),
        update_list_header=lambda _query: events.append("header"),
        update_status_bar=lambda: events.append("status"),
        update_filter_pills=lambda _query: events.append("pills"),
        refresh_detail_pane=lambda: events.append("detail"),
        refresh_current_list_item=lambda: events.append("current"),
    )

    coordinator.refresh_detail_and_list_item()

    assert events == ["detail", "current"]


def test_restore_omni_chrome_closes_focuses_and_updates_footer() -> None:
    search_container = SimpleNamespace(close=MagicMock())
    paper_list = SimpleNamespace(focus=MagicMock())
    app = SimpleNamespace(
        _get_search_container_widget=MagicMock(return_value=search_container),
        _get_paper_list_widget=MagicMock(return_value=paper_list),
        _update_footer=MagicMock(),
    )

    restore_omni_chrome(app)

    search_container.close.assert_called_once_with()
    paper_list.focus.assert_called_once_with()
    app._update_footer.assert_called_once_with()


def test_restore_omni_chrome_tolerates_missing_paper_list() -> None:
    search_container = SimpleNamespace(close=MagicMock())
    app = SimpleNamespace(
        _get_search_container_widget=MagicMock(return_value=search_container),
        _get_paper_list_widget=MagicMock(side_effect=AttributeError("not mounted")),
        _update_footer=MagicMock(),
    )

    restore_omni_chrome(app)

    search_container.close.assert_called_once_with()
    app._update_footer.assert_called_once_with()


def test_restore_omni_chrome_tolerates_unfocusable_paper_list() -> None:
    search_container = SimpleNamespace(close=MagicMock())
    app = SimpleNamespace(
        _get_search_container_widget=MagicMock(return_value=search_container),
        _get_paper_list_widget=MagicMock(return_value=object()),
        _update_footer=MagicMock(),
    )

    restore_omni_chrome(app)

    search_container.close.assert_called_once_with()
    app._update_footer.assert_called_once_with()


def test_restore_omni_chrome_tolerates_focus_no_matches() -> None:
    search_container = SimpleNamespace(close=MagicMock())
    paper_list = SimpleNamespace(focus=MagicMock(side_effect=NoMatches("detached")))
    app = SimpleNamespace(
        _get_search_container_widget=MagicMock(return_value=search_container),
        _get_paper_list_widget=MagicMock(return_value=paper_list),
        _update_footer=MagicMock(),
    )

    restore_omni_chrome(app)

    search_container.close.assert_called_once_with()
    paper_list.focus.assert_called_once_with()
    app._update_footer.assert_called_once_with()
