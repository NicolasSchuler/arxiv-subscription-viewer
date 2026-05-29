"""Focused browser runtime tests for cached refs and Omni routing."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from textual.css.query import NoMatches

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.ui_runtime import UiRefs
from arxiv_browser.widgets.omni_input import OmniInput


class _WidgetRef:
    def __init__(self, *, attached: bool) -> None:
        self.is_attached = attached


def test_get_cached_widget_reuses_live_ref_and_replaces_detached_ref() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._ui_refs = UiRefs()
    live = _WidgetRef(attached=True)
    app._ui_refs.paper_list = live  # type: ignore[assignment]
    resolver = MagicMock()

    assert app._get_cached_widget("paper_list", resolver) is live
    resolver.assert_not_called()

    replacement = _WidgetRef(attached=True)
    app._ui_refs.paper_list = _WidgetRef(attached=False)  # type: ignore[assignment]
    resolver = MagicMock(return_value=replacement)

    assert app._get_cached_widget("paper_list", resolver) is replacement
    assert app._ui_refs.paper_list is replacement
    resolver.assert_called_once_with()


def test_reset_ui_refs_clears_cached_refs() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._ui_refs = UiRefs()
    app._ui_refs.paper_list = object()  # type: ignore[assignment]
    app._ui_refs.status_bar = object()  # type: ignore[assignment]

    app._reset_ui_refs()

    assert app._ui_refs.paper_list is None
    assert app._ui_refs.status_bar is None


def test_prime_ui_refs_continues_after_one_widget_missing() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._ui_refs = UiRefs()
    getters: dict[str, MagicMock] = {
        "_get_search_input_widget": MagicMock(return_value=object()),
        "_get_search_container_widget": MagicMock(side_effect=NoMatches("missing")),
        "_get_paper_list_widget": MagicMock(return_value=object()),
        "_get_list_header_widget": MagicMock(return_value=object()),
        "_get_details_header_widget": MagicMock(return_value=object()),
        "_get_status_bar_widget": MagicMock(return_value=object()),
        "_get_footer_widget": MagicMock(return_value=object()),
        "_get_date_navigator_widget": MagicMock(return_value=object()),
        "_get_filter_pill_bar_widget": MagicMock(return_value=object()),
        "_get_bookmark_bar_widget": MagicMock(return_value=object()),
        "_get_paper_details_widget": MagicMock(return_value=object()),
    }
    for name, getter in getters.items():
        setattr(app, name, getter)

    app._prime_ui_refs()

    for getter in getters.values():
        getter.assert_called_once_with()


def test_lazy_ui_refresh_coordinator_invokes_current_methods() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    events: list[tuple[str, tuple[object, ...]]] = []
    app._refresh_list_view = MagicMock(side_effect=lambda: events.append(("list", ())))
    app._update_list_header = MagicMock(
        side_effect=lambda query: events.append(("header", (query,)))
    )
    app._update_status_bar = MagicMock(side_effect=lambda: events.append(("status", ())))
    app._update_filter_pills = MagicMock(
        side_effect=lambda query: events.append(("pills", (query,)))
    )
    app._refresh_detail_pane = MagicMock(side_effect=lambda: events.append(("detail", ())))
    app._refresh_current_list_item = MagicMock(side_effect=lambda: events.append(("current", ())))

    coordinator = app._get_ui_refresh_coordinator()
    coordinator.apply_filter_refresh("transformer")
    coordinator.refresh_detail_and_list_item()

    assert app._get_ui_refresh_coordinator() is coordinator
    assert events == [
        ("list", ()),
        ("header", ("transformer",)),
        ("status", ()),
        ("pills", ("transformer",)),
        ("detail", ()),
        ("current", ()),
    ]


def test_on_omni_api_search_constructs_default_request_and_tracks_task() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._run_arxiv_search_worker = MagicMock()  # type: ignore[method-assign]

    app.on_omni_api_search(OmniInput.ApiSearch("graph transformers"))

    app._run_arxiv_search_worker.assert_called_once()
    request = app._run_arxiv_search_worker.call_args.args[0]
    assert request.query == "graph transformers"
    assert request.field == "all"
    assert request.category == ""
    assert app._run_arxiv_search_worker.call_args.kwargs == {"start": 0}


def test_on_omni_command_selected_restores_chrome_before_dispatch_failure(caplog) -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    events: list[str] = []

    def action_boom() -> None:
        events.append("action")
        raise RuntimeError("boom")

    app.action_boom = action_boom  # type: ignore[attr-defined]
    caplog.set_level(logging.WARNING, logger="arxiv_browser.browser.core")

    with patch(
        "arxiv_browser.browser.core.restore_omni_chrome",
        side_effect=lambda _app: events.append("restore"),
    ) as restore_mock:
        app.on_omni_command_selected(OmniInput.CommandSelected("boom"))

    restore_mock.assert_called_once_with(app)
    assert events == ["restore", "action"]
    assert "OmniInput command failed: boom" in caplog.text


def test_on_omni_command_selected_tracks_async_action_result() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    events: list[str] = []

    async def async_result() -> None:
        events.append("awaited")

    def action_async() -> object:
        events.append("action")
        return async_result()

    def track_task(coro: object) -> None:
        events.append("track")
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    app.action_async = action_async  # type: ignore[attr-defined]
    app._track_task = MagicMock(side_effect=track_task)  # type: ignore[method-assign]

    with patch(
        "arxiv_browser.browser.core.restore_omni_chrome",
        side_effect=lambda _app: events.append("restore"),
    ):
        app.on_omni_command_selected(OmniInput.CommandSelected("async"))

    assert events == ["restore", "action", "track"]
    app._track_task.assert_called_once()


def test_on_omni_command_selected_unknown_action_only_restores_chrome(caplog) -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._track_task = MagicMock()  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING, logger="arxiv_browser.browser.core")

    with patch("arxiv_browser.browser.core.restore_omni_chrome") as restore_mock:
        app.on_omni_command_selected(OmniInput.CommandSelected("does_not_exist"))

    restore_mock.assert_called_once_with(app)
    app._track_task.assert_not_called()
    assert "OmniInput command failed" not in caplog.text


def test_on_omni_local_search_submitted_hides_input_and_focuses_list() -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    search_container = SimpleNamespace(hide=MagicMock(), close=MagicMock())
    paper_list = SimpleNamespace(focus=MagicMock())
    app._apply_filter = MagicMock()  # type: ignore[method-assign]
    app._get_search_container_widget = MagicMock(return_value=search_container)  # type: ignore[method-assign]
    app._get_paper_list_widget = MagicMock(return_value=paper_list)  # type: ignore[method-assign]
    app._update_footer = MagicMock()  # type: ignore[method-assign]

    app.on_omni_local_search_submitted(OmniInput.LocalSearchSubmitted("cat:cs.AI"))

    app._apply_filter.assert_called_once_with("cat:cs.AI")
    search_container.hide.assert_called_once_with()
    search_container.close.assert_not_called()
    paper_list.focus.assert_called_once_with()
    app._update_footer.assert_called_once_with()
