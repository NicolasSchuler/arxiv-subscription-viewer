#!/usr/bin/env python3
"""Focused TUI quality-pass tests for modal/widget coverage and hot-path behavior."""

from __future__ import annotations

from datetime import date as dt_date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from textual.events import Click
from textual.widgets import Checkbox, Input, Label, ListView, Select, Static

from arxiv_browser.app import (
    MAX_COLLECTIONS,
    AddToCollectionModal,
    ArxivBrowser,
    ArxivSearchModal,
    ArxivSearchRequest,
    CitationEntry,
    CitationGraphScreen,
    CollectionsModal,
    CollectionViewModal,
    CommandPaletteModal,
    DateNavigator,
    FilterPillBar,
    PaperCollection,
    PaperListItem,
    RecommendationSourceModal,
    RecommendationsScreen,
    WatchListEntry,
    WatchListModal,
    tokenize_query,
)


async def _open_modal(app: ArxivBrowser, pilot, modal) -> None:
    app.push_screen(modal)
    await pilot.pause(0.05)
    assert app.screen_stack[-1] is modal


def _click(widget) -> Click:
    return Click(
        widget=widget,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
    )


def _citation_entry(
    *,
    s2_paper_id: str,
    arxiv_id: str,
    title: str,
    url: str,
) -> CitationEntry:
    return CitationEntry(
        s2_paper_id=s2_paper_id,
        arxiv_id=arxiv_id,
        title=title,
        authors="Author One, Author Two",
        year=2024,
        citation_count=7,
        url=url,
    )


@pytest.mark.asyncio
async def test_collections_modal_create_rename_delete_flow(make_paper):
    papers = [make_paper(arxiv_id="2401.00001", title="First")]
    app = ArxivBrowser(papers, restore_session=False)
    base = PaperCollection(name="Reading", description="base", paper_ids=[papers[0].arxiv_id])
    modal = CollectionsModal([base], papers_by_id={papers[0].arxiv_id: papers[0]})

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            name_input = modal.query_one("#col-name", Input)
            desc_input = modal.query_one("#col-desc", Input)
            list_view = modal.query_one("#col-list", ListView)

            name_input.value = "Second"
            desc_input.value = "new desc"
            modal.on_create_pressed()
            assert [c.name for c in modal.collections] == ["Reading", "Second"]

            list_view.index = 1
            name_input.value = "Renamed"
            desc_input.value = "updated"
            modal.on_rename_pressed()
            assert modal.collections[1].name == "Renamed"
            assert modal.collections[1].description == "updated"

            list_view.index = 1
            modal.on_delete_pressed()
            assert [c.name for c in modal.collections] == ["Reading"]


@pytest.mark.asyncio
async def test_collections_modal_validates_create_inputs(make_paper):
    papers = [make_paper(arxiv_id="2401.00001")]
    app = ArxivBrowser(papers, restore_session=False)
    base = PaperCollection(name="Reading", description="", paper_ids=[])
    modal = CollectionsModal([base])

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            name_input = modal.query_one("#col-name", Input)
            modal.notify = MagicMock()

            name_input.value = ""
            modal.on_create_pressed()
            assert "Name cannot be empty" in modal.notify.call_args[0][0]

            modal.notify.reset_mock()
            name_input.value = "Reading"
            modal.on_create_pressed()
            assert "already exists" in modal.notify.call_args[0][0]


@pytest.mark.asyncio
async def test_collections_modal_enforces_max_collection_count(make_paper):
    papers = [make_paper(arxiv_id="2401.00001")]
    app = ArxivBrowser(papers, restore_session=False)
    collections = [
        PaperCollection(name=f"C{i}", description="", paper_ids=[]) for i in range(MAX_COLLECTIONS)
    ]
    modal = CollectionsModal(collections)

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            modal.notify = MagicMock()
            modal.query_one("#col-name", Input).value = "Overflow"
            modal.on_create_pressed()
            assert len(modal.collections) == MAX_COLLECTIONS
            assert "Maximum" in modal.notify.call_args[0][0]


@pytest.mark.asyncio
async def test_collection_view_modal_remove_and_done(make_paper):
    papers = [
        make_paper(arxiv_id="2401.00001", title="A"),
        make_paper(arxiv_id="2401.00002", title="B"),
    ]
    collection = PaperCollection(
        name="Reading",
        description="",
        paper_ids=[papers[0].arxiv_id, papers[1].arxiv_id],
    )
    app = ArxivBrowser(papers, restore_session=False)
    modal = CollectionViewModal(collection, papers_by_id={p.arxiv_id: p for p in papers})

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            modal.on_remove_pressed()
            assert modal._collection.paper_ids == [papers[1].arxiv_id]

            modal.dismiss = MagicMock()
            modal.on_done_pressed()
            assert modal.dismiss.call_args[0][0].paper_ids == [papers[1].arxiv_id]


@pytest.mark.asyncio
async def test_collection_view_modal_requires_selection_for_remove(make_paper):
    papers = [make_paper(arxiv_id="2401.00001", title="A")]
    collection = PaperCollection(name="Reading", description="", paper_ids=[])
    app = ArxivBrowser(papers, restore_session=False)
    modal = CollectionViewModal(collection, papers_by_id={papers[0].arxiv_id: papers[0]})

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            modal.notify = MagicMock()
            modal.on_remove_pressed()
            assert "Select a paper" in modal.notify.call_args[0][0]


@pytest.mark.asyncio
async def test_add_to_collection_modal_select_and_cancel(make_paper):
    papers = [make_paper(arxiv_id="2401.00001")]
    app = ArxivBrowser(papers, restore_session=False)
    collections = [
        PaperCollection(name="A", description="", paper_ids=[]),
        PaperCollection(name="B", description="", paper_ids=[]),
    ]
    modal = AddToCollectionModal(collections)

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            list_view = modal.query_one("#addcol-list", ListView)
            list_view.index = 1
            modal.dismiss = MagicMock()
            modal.on_list_selected(SimpleNamespace())
            modal.dismiss.assert_called_once_with("B")

            modal.dismiss.reset_mock()
            modal.on_cancel_pressed()
            modal.dismiss.assert_called_once_with(None)


def test_recommendation_source_modal_actions():
    modal = RecommendationSourceModal()
    modal.dismiss = MagicMock()

    modal.action_local()
    modal.dismiss.assert_called_with("local")
    modal.action_s2()
    modal.dismiss.assert_called_with("s2")
    modal.action_cancel()
    modal.dismiss.assert_called_with("")


@pytest.mark.asyncio
async def test_recommendations_screen_select_cursor_and_cancel(make_paper):
    target = make_paper(arxiv_id="2401.00001", title="Target")
    p2 = make_paper(arxiv_id="2401.00002", title="Second")
    p3 = make_paper(arxiv_id="2401.00003", title="Third")
    app = ArxivBrowser([target, p2, p3], restore_session=False)
    modal = RecommendationsScreen(target, [(p2, 0.92), (p3, 0.70)])

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            list_view = modal.query_one("#recommendations-list", ListView)
            assert len(list_view.children) == 2

            modal.dismiss = MagicMock()
            modal.action_select()
            modal.dismiss.assert_called_with("2401.00002")

            modal.dismiss.reset_mock()
            modal.action_cursor_down()
            modal.action_select()
            modal.dismiss.assert_called_with("2401.00003")

            modal.dismiss.reset_mock()
            modal.action_cancel()
            modal.dismiss.assert_called_with(None)

            modal.dismiss.reset_mock()
            modal.on_list_selected(SimpleNamespace(item=list_view.children[0]))
            modal.dismiss.assert_called_with("2401.00002")


@pytest.mark.asyncio
async def test_recommendations_screen_select_empty_returns_none(make_paper):
    target = make_paper(arxiv_id="2401.00001", title="Target")
    app = ArxivBrowser([target], restore_session=False)
    modal = RecommendationsScreen(target, [])

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            modal.dismiss = MagicMock()
            modal.action_select()
            modal.dismiss.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_citation_graph_switch_open_url_and_local_jump(make_paper):
    root = make_paper(arxiv_id="2401.00001", title="Root Paper")
    refs = [
        _citation_entry(
            s2_paper_id="s2:local",
            arxiv_id="2401.00002",
            title="Local Ref",
            url="https://arxiv.org/abs/2401.00002",
        )
    ]
    cites = [
        _citation_entry(
            s2_paper_id="s2:remote",
            arxiv_id="",
            title="Remote Cite",
            url="https://www.semanticscholar.org/paper/s2:remote",
        )
    ]

    async def _fetch(_paper_id: str):
        return [], []

    app = ArxivBrowser([root], restore_session=False)
    modal = CitationGraphScreen(
        root_title=root.title,
        root_paper_id=root.arxiv_id,
        references=refs,
        citations=cites,
        fetch_callback=_fetch,
        local_arxiv_ids=frozenset({"2401.00002"}),
    )

    with (
        patch("arxiv_browser.app.save_config", return_value=True),
        patch("arxiv_browser.modals.citations.webbrowser.open") as browser_open,
    ):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            assert modal._active_panel == "refs"

            modal.action_open_url()
            browser_open.assert_called_once_with("https://arxiv.org/abs/2401.00002")

            modal.dismiss = MagicMock()
            modal.action_go_to_local()
            modal.dismiss.assert_called_once_with("2401.00002")

            modal.action_switch_panel()
            assert modal._active_panel == "cites"


@pytest.mark.asyncio
async def test_citation_graph_drill_down_success_and_back(make_paper):
    root = make_paper(arxiv_id="2401.00001", title="Root Paper")
    child = _citation_entry(
        s2_paper_id="s2:child",
        arxiv_id="2401.00002",
        title="Child",
        url="https://arxiv.org/abs/2401.00002",
    )
    next_ref = _citation_entry(
        s2_paper_id="s2:grandchild",
        arxiv_id="2401.00003",
        title="Grandchild",
        url="https://arxiv.org/abs/2401.00003",
    )

    async def _fetch(_paper_id: str):
        return [next_ref], []

    app = ArxivBrowser([root], restore_session=False)
    modal = CitationGraphScreen(
        root_title=root.title,
        root_paper_id=root.arxiv_id,
        references=[child],
        citations=[],
        fetch_callback=_fetch,
        local_arxiv_ids=frozenset({"2401.00002", "2401.00003"}),
    )

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            await modal.action_drill_down()
            assert len(modal._stack) == 1
            assert modal._current_paper_id == "s2:child"
            assert len(modal._current_refs) == 1

            modal.action_back_or_close()
            assert modal._current_paper_id == root.arxiv_id
            assert modal._stack == []


@pytest.mark.asyncio
async def test_citation_graph_drill_down_failure_restores_state(make_paper):
    root = make_paper(arxiv_id="2401.00001", title="Root Paper")
    child = _citation_entry(
        s2_paper_id="s2:child",
        arxiv_id="2401.00002",
        title="Child",
        url="https://arxiv.org/abs/2401.00002",
    )

    async def _fetch(_paper_id: str):
        raise httpx.ConnectError("boom")

    app = ArxivBrowser([root], restore_session=False)
    modal = CitationGraphScreen(
        root_title=root.title,
        root_paper_id=root.arxiv_id,
        references=[child],
        citations=[],
        fetch_callback=_fetch,
        local_arxiv_ids=frozenset({"2401.00002"}),
    )

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            await modal.action_drill_down()
            assert modal._stack == []
            assert modal._current_paper_id == root.arxiv_id


@pytest.mark.asyncio
async def test_citation_graph_button_handlers_use_track_task(make_paper):
    root = make_paper(arxiv_id="2401.00001", title="Root Paper")
    child = _citation_entry(
        s2_paper_id="s2:child",
        arxiv_id="2401.00002",
        title="Child",
        url="https://arxiv.org/abs/2401.00002",
    )

    async def _fetch(_paper_id: str):
        return [], []

    app = ArxivBrowser([root], restore_session=False)
    modal = CitationGraphScreen(
        root_title=root.title,
        root_paper_id=root.arxiv_id,
        references=[child],
        citations=[],
        fetch_callback=_fetch,
        local_arxiv_ids=frozenset({"2401.00002"}),
    )

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            captured = []

            def _track_task(coro):
                captured.append(coro)
                coro.close()

            app._track_task = MagicMock(side_effect=_track_task)
            modal.on_drill_pressed()
            app._track_task.assert_called_once()
            assert len(captured) == 1

            modal.dismiss = MagicMock()
            modal.on_close_pressed()
            modal.dismiss.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_arxiv_search_modal_validation_and_submit(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = ArxivSearchModal(initial_query="seed", initial_field="bad-field", initial_category="")

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            field_select = modal.query_one("#arxiv-search-field", Select)
            assert field_select.value == "all"

            modal.dismiss = MagicMock()
            modal.query_one("#arxiv-search-query", Input).value = "transformers"
            modal.query_one("#arxiv-search-category", Input).value = "cs.AI"
            field_select.value = "title"
            modal.action_search()

            request = modal.dismiss.call_args[0][0]
            assert isinstance(request, ArxivSearchRequest)
            assert request.query == "transformers"
            assert request.field == "title"
            assert request.category == "cs.AI"

            modal.dismiss.reset_mock()
            modal.notify = MagicMock()
            modal.query_one("#arxiv-search-query", Input).value = ""
            modal.query_one("#arxiv-search-category", Input).value = ""
            modal.action_search()
            modal.notify.assert_called_once()
            modal.dismiss.assert_not_called()


@pytest.mark.asyncio
async def test_command_palette_modal_filters_and_executes(make_paper):
    commands = [
        ("Open Paper", "Open selected paper", "o", "open"),
        ("Toggle Watch", "Toggle watch filter", "w", "watch"),
        ("Export CSV", "Export list as CSV", "E", "csv"),
    ]
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = CommandPaletteModal(commands)

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            results = modal.query_one("#palette-results")
            assert "Command palette" in str(modal.query_one(Label).content)
            assert results.option_count == 3
            assert "Close: Esc" in str(modal.query_one("#palette-footer", Static).content)

            modal._populate_results("watch")
            assert results.option_count >= 1

            modal.dismiss = MagicMock()
            modal.key_enter()
            assert modal.dismiss.called

            modal.dismiss.reset_mock()
            modal._populate_results("zzzzqzzzz")
            assert results.option_count == 1
            assert "No commands match" in str(results.get_option_at_index(0).prompt)
            modal.key_enter()
            modal.dismiss.assert_not_called()

            modal.dismiss.reset_mock()
            modal._on_option_selected(SimpleNamespace(option_id="csv"))
            modal.dismiss.assert_called_once_with("csv")

            modal.dismiss.reset_mock()
            modal.action_cancel()
            modal.dismiss.assert_called_once_with("")


def test_help_screen_default_ctrl_e_copy():
    from arxiv_browser.modals.common import HelpScreen

    entries = {(key, desc) for _, pairs in HelpScreen._DEFAULT_SECTIONS for key, desc in pairs}
    assert ("Ctrl+e", "Toggle S2 (browse) / Exit API (API mode)") in entries
    assert ("Esc", "Clear search / exit API") in entries


def test_help_screen_default_footer_copy():
    from arxiv_browser.modals.common import HelpScreen

    modal = HelpScreen()
    assert modal._footer_note == "Close: ? / Esc / q"


@pytest.mark.asyncio
async def test_watch_list_modal_add_update_delete_and_save(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = WatchListModal(
        [WatchListEntry(pattern="Smith", match_type="author", case_sensitive=False)]
    )

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            pattern = modal.query_one("#watch-pattern", Input)
            match_type = modal.query_one("#watch-type", Select)
            case = modal.query_one("#watch-case", Checkbox)
            list_view = modal.query_one("#watch-list", ListView)
            empty_hint = modal.query_one("#watch-empty", Static)
            assert empty_hint.has_class("visible") is False

            modal.notify = MagicMock()
            pattern.value = ""
            modal.on_add_pressed()
            assert "Pattern cannot be empty" in modal.notify.call_args[0][0]

            pattern.value = "transformer"
            match_type.value = "title"
            case.value = True
            modal.on_add_pressed()
            assert len(modal._entries) == 2

            list_view.index = 1
            pattern.value = "diffusion"
            match_type.value = "keyword"
            case.value = False
            modal.on_update_pressed()
            assert modal._entries[1].pattern == "diffusion"
            assert modal._entries[1].match_type == "keyword"
            assert modal._entries[1].case_sensitive is False

            list_view.index = 1
            modal.on_delete_pressed()
            assert len(modal._entries) == 1

            modal.dismiss = MagicMock()
            modal.action_save()
            modal.dismiss.assert_called_once_with(modal._entries)


@pytest.mark.asyncio
async def test_watch_list_modal_update_delete_require_selection(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)
    modal = WatchListModal([])

    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await _open_modal(app, pilot, modal)
            empty_hint = modal.query_one("#watch-empty", Static)
            assert "No watch entries yet." in str(empty_hint.content)
            assert "Try:" in str(empty_hint.content)
            assert empty_hint.has_class("visible")
            modal.notify = MagicMock()
            modal.on_update_pressed()
            assert "Select a watch entry to update" in modal.notify.call_args[0][0]

            modal.notify.reset_mock()
            modal.on_delete_pressed()
            assert "Select a watch entry to delete" in modal.notify.call_args[0][0]


@pytest.mark.asyncio
async def test_date_navigator_updates_in_place_when_window_unchanged(make_paper, tmp_path):
    f1 = tmp_path / "2026-01-01.txt"
    f2 = tmp_path / "2026-01-02.txt"
    f3 = tmp_path / "2026-01-03.txt"
    for idx, path in enumerate([f1, f2, f3], start=1):
        path.write_text(f"arXiv:2401.0000{idx}\n", encoding="utf-8")

    history_files = [
        (dt_date(2026, 1, 3), f3),
        (dt_date(2026, 1, 2), f2),
        (dt_date(2026, 1, 1), f1),
    ]

    app = ArxivBrowser([make_paper()], restore_session=False, history_files=history_files)
    with patch("arxiv_browser.app.save_config", return_value=True):
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            nav = app.query_one(DateNavigator)
            await nav.update_dates(history_files, 1)

            before = {
                child.id: child
                for child in nav.children
                if isinstance(child, Label) and child.id and "date-nav-item" in child.classes
            }

            await nav.update_dates(history_files, 2)
            after = {
                child.id: child
                for child in nav.children
                if isinstance(child, Label) and child.id and "date-nav-item" in child.classes
            }
            assert before.keys() == after.keys()
            for key in before:
                assert before[key] is after[key]


def test_date_navigator_click_messages():
    nav = DateNavigator([])
    nav.post_message = MagicMock()

    nav.on_click(_click(Label("<", id="date-nav-prev")))
    msg = nav.post_message.call_args[0][0]
    assert isinstance(msg, DateNavigator.NavigateDate)
    assert msg.direction == 1

    nav.post_message.reset_mock()
    nav.on_click(_click(Label(">", id="date-nav-next")))
    msg = nav.post_message.call_args[0][0]
    assert isinstance(msg, DateNavigator.NavigateDate)
    assert msg.direction == -1

    nav.post_message.reset_mock()
    nav.on_click(_click(Label("Jan 01", id="date-nav-4")))
    msg = nav.post_message.call_args[0][0]
    assert isinstance(msg, DateNavigator.JumpToDate)
    assert msg.index == 4


@pytest.mark.asyncio
async def test_filter_pill_bar_updates_in_place_and_click_events():
    bar = FilterPillBar()

    def fake_mount(widget: Label):
        bar._nodes._append(widget)

    async def fake_remove_children():
        for child in list(bar.children):
            bar._nodes._remove(child)

    bar.mount = fake_mount  # type: ignore[method-assign]
    bar.remove_children = fake_remove_children  # type: ignore[method-assign]

    tokens = tokenize_query("cat:cs.AI AND transformer")
    await bar.update_pills(tokens, watch_active=True)
    first_by_id = {child.id: child for child in bar.children if child.id is not None}

    await bar.update_pills(tokens, watch_active=True)
    second_by_id = {child.id: child for child in bar.children if child.id is not None}
    assert first_by_id["pill-0"] is second_by_id["pill-0"]
    assert first_by_id["pill-2"] is second_by_id["pill-2"]
    assert first_by_id["pill-watch"] is second_by_id["pill-watch"]

    bar.post_message = MagicMock()
    bar.on_click(_click(second_by_id["pill-0"]))
    msg = bar.post_message.call_args[0][0]
    assert isinstance(msg, FilterPillBar.RemoveFilter)
    assert msg.token_index == 0

    bar.post_message.reset_mock()
    bar.on_click(_click(second_by_id["pill-watch"]))
    msg = bar.post_message.call_args[0][0]
    assert isinstance(msg, FilterPillBar.RemoveWatchFilter)

    # Invalid pill index should be ignored safely.
    bar.post_message.reset_mock()
    bar.on_click(_click(Label("bad", id="pill-bad")))
    bar.post_message.assert_not_called()


def test_paper_list_item_update_methods_safe_before_mount(make_paper):
    paper = make_paper(arxiv_id="2401.00001")
    item = PaperListItem(paper, show_preview=True)
    item.set_abstract_text("Preview text")
    item.set_selected(True)
    item.set_selected(False)
    item.update_s2_data(None)
    item.update_hf_data(None)
    item.update_version_data((1, 2))
    item.update_relevance_data((8, "high"))
    assert item.paper.arxiv_id == "2401.00001"


def test_badge_refresh_indices_sparse_updates(make_paper):
    app = ArxivBrowser.__new__(ArxivBrowser)
    app.filtered_papers = [
        make_paper(arxiv_id="2401.00001"),
        make_paper(arxiv_id="2401.00002"),
        make_paper(arxiv_id="2401.00003"),
    ]
    app._s2_active = True
    app._s2_cache = {"2401.00003": object()}
    app._hf_active = True
    app._hf_cache = {"2401.00002": object()}
    app._version_updates = {"2401.00001": (1, 2)}

    assert app._badge_refresh_indices({"hf"}) == [1]
    assert app._badge_refresh_indices({"s2"}) == [2]
    assert app._badge_refresh_indices({"version"}) == [0]
    assert app._badge_refresh_indices({"hf", "version"}) == [0, 1]

    app._hf_active = False
    assert app._badge_refresh_indices({"hf"}) == [0, 1, 2]


def test_flush_badge_refresh_uses_computed_sparse_indices(make_paper):
    app = ArxivBrowser.__new__(ArxivBrowser)
    app.filtered_papers = [
        make_paper(arxiv_id="2401.00001"),
        make_paper(arxiv_id="2401.00002"),
        make_paper(arxiv_id="2401.00003"),
    ]
    app._s2_active = True
    app._s2_cache = {}
    app._hf_active = True
    app._hf_cache = {"2401.00002": object()}
    app._version_updates = {}
    app._badges_dirty = {"hf"}
    app._badge_timer = None
    app._update_option_at_index = MagicMock()

    app._flush_badge_refresh()
    app._update_option_at_index.assert_called_once_with(1)
