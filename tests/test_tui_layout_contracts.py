"""Textual layout contracts for key TUI modes and terminal sizes."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Label, OptionList, Static

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.browser.options import ArxivBrowserOptions
from arxiv_browser.modals.editing import LineAnnotationModal
from arxiv_browser.models import ArxivSearchModeState, ArxivSearchRequest, UserConfig
from arxiv_browser.widgets.chrome import ContextFooter, FilterPillBar
from arxiv_browser.widgets.details import PaperDetails
from arxiv_browser.widgets.omni_input import OmniInput
from tests.support.patch_helpers import patch_save_config


def _footer_text(app: ArxivBrowser) -> str:
    return str(app.query_one(ContextFooter).content)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 40)])
async def test_browse_layout_contract_across_terminal_sizes(make_paper, size):
    papers = [
        make_paper(arxiv_id="2401.00001", title="Attention Models"),
        make_paper(arxiv_id="2401.00002", title="Diffusion Models"),
    ]
    app = ArxivBrowser(papers, restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.1)
            header = str(app.query_one("#list-header", Label).content)
            footer = _footer_text(app)
            status = str(app.query_one("#status-bar", Label).content)

            assert "Browse" in header
            assert "Space" in footer
            assert "Ctrl+p" in footer
            assert "papers" in status


@pytest.mark.asyncio
async def test_selection_and_detail_focus_layout_contracts(make_paper):
    app = ArxivBrowser([make_paper(arxiv_id="2401.00001")], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("space")
            await pilot.pause(0.05)
            assert "selected" in str(app.query_one("#list-header", Label).content)

            await pilot.press("tab")
            await pilot.pause(0.05)
            footer = _footer_text(app)
            status = str(app.query_one("#status-bar", Label).content)
            assert "DETAILS" in footer
            assert "j/k" in footer
            assert "Details focus" in status or "details" in status


@pytest.mark.asyncio
async def test_api_empty_palette_and_theme_layout_contracts(make_paper):
    papers = [make_paper(arxiv_id="2401.00001", title="Graph Search")]
    app = ArxivBrowser(
        papers,
        ArxivBrowserOptions(
            config=UserConfig(theme_name="solarized-light"),
            restore_session=False,
        ),
    )

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            app._in_arxiv_api_mode = True
            app._arxiv_search_state = ArxivSearchModeState(
                request=ArxivSearchRequest(query="graph"),
                start=10,
                max_results=10,
            )
            app._update_header()
            assert "API results" in str(app.query_one("#list-header", Label).content)
            assert "API" in _footer_text(app)

            await pilot.press("ctrl+p")
            await pilot.pause(0.05)
            assert app.query_one(OmniInput).is_open


@pytest.mark.asyncio
@pytest.mark.integration
async def test_omni_command_executes_action_and_restores_footer(make_paper):
    paper = make_paper(arxiv_id="2401.10001")
    app = ArxivBrowser([paper], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("ctrl+p")
            await pilot.pause(0.05)
            omni = app.query_one(OmniInput)
            input_widget = omni.query_one("#omni-input", Input)
            input_widget.value = ">preview"
            await pilot.pause(0.1)
            preview_index = [command.action for command in omni._filtered_commands].index(
                "toggle_preview"
            )
            omni.query_one("#omni-results", OptionList).highlighted = preview_index

            await input_widget.action_submit()
            await pilot.pause(0.05)

            assert app._show_abstract_preview is True
            assert omni.is_open is False
            assert "SEARCH" not in _footer_text(app)
            assert app.query_one("#paper-list", OptionList).has_focus


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_ui_refs_are_populated_after_mount(make_paper):
    app = ArxivBrowser([make_paper(arxiv_id="2401.11001")], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)

            for ref_name in (
                "search_input",
                "search_container",
                "paper_list",
                "list_header",
                "details_header",
                "status_bar",
                "footer",
                "filter_pill_bar",
                "bookmark_bar",
                "paper_details",
            ):
                widget = getattr(app._ui_refs, ref_name)
                assert widget is not None, ref_name
                assert widget.is_attached, ref_name


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_ui_refs_rebuild_after_manual_reset(make_paper):
    app = ArxivBrowser([make_paper(arxiv_id="2401.11002")], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            old_input = app._get_search_input_widget()
            old_list = app._get_paper_list_widget()

            app._reset_ui_refs()

            assert app._ui_refs.search_input is None
            assert app._ui_refs.paper_list is None
            assert app._get_search_input_widget() is old_input
            assert app._get_paper_list_widget() is old_list
            assert app._ui_refs.search_input is old_input
            assert app._ui_refs.paper_list is old_list


@pytest.mark.asyncio
@pytest.mark.integration
async def test_semantic_omni_submit_hides_input_and_focuses_list(make_paper):
    papers = [
        make_paper(arxiv_id="2401.11003", title="Transformer Paper"),
        make_paper(arxiv_id="2401.11004", title="Graph Paper"),
    ]
    app = ArxivBrowser(papers, restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            scheduled: list[object] = []

            def track_dataset_task(coro: object) -> None:
                scheduled.append(coro)
                close = getattr(coro, "close", None)
                if callable(close):
                    close()

            app._track_dataset_task = track_dataset_task  # type: ignore[method-assign]
            omni = app.query_one(OmniInput)
            omni.open("~ transformer")
            await pilot.pause(0.05)
            input_widget = omni.query_one("#omni-input", Input)
            scheduled.clear()

            await input_widget.action_submit()
            await pilot.pause(0.05)

            assert app._applied_query == "~ transformer"
            assert scheduled
            assert omni.is_open is False
            assert app.query_one("#paper-list", OptionList).has_focus


@pytest.mark.asyncio
@pytest.mark.integration
async def test_arxiv_api_empty_omni_submit_does_not_start_search(make_paper):
    app = ArxivBrowser([make_paper()], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            calls: list[object] = []

            def track_task(coro):
                calls.append(coro)
                close = getattr(coro, "close", None)
                if callable(close):
                    close()

            app._track_task = track_task  # type: ignore[method-assign]
            await pilot.press("A")
            await pilot.pause(0.05)
            omni = app.query_one(OmniInput)
            input_widget = omni.query_one("#omni-input", Input)
            assert input_widget.value == "@"

            await input_widget.action_submit()
            await pilot.pause(0.05)

            assert calls == []
            assert app._in_arxiv_api_mode is False
            assert omni.is_open is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_refresh_list_view_uses_contextual_empty_state(make_paper):
    app = ArxivBrowser(
        [make_paper(arxiv_id="2401.10002", title="Only Paper")],
        restore_session=False,
    )

    with patch_save_config(return_value=True):
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause(0.1)
            app._get_search_input_widget().value = "cat:does-not-exist"
            app._apply_filter("cat:does-not-exist")
            await pilot.pause(0.05)

            paper_list = app.query_one("#paper-list", OptionList)
            details = str(app.query_one(PaperDetails).content)
            prompt = str(paper_list.get_option_at_index(0).prompt)
            assert paper_list.option_count == 1
            assert paper_list.get_option_at_index(0).disabled is True
            assert "No papers match" in prompt
            assert "Try:" in prompt
            assert "Next:" in prompt
            assert "Select a paper" in details


@pytest.mark.asyncio
@pytest.mark.integration
async def test_filter_pill_removal_updates_live_query_and_list(make_paper):
    papers = [
        make_paper(arxiv_id="2401.10003", title="Transformer Paper", categories="cs.AI"),
        make_paper(arxiv_id="2401.10004", title="Transformer Paper", categories="cs.LG"),
    ]
    app = ArxivBrowser(papers, restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            search_input = app._get_search_input_widget()
            search_input.value = "cat:cs.AI transformer"
            app._apply_filter("cat:cs.AI transformer")
            await pilot.pause(0.1)
            assert [paper.arxiv_id for paper in app.filtered_papers] == ["2401.10003"]

            app.on_remove_filter(FilterPillBar.RemoveFilter(0))
            await pilot.pause(0.2)

            assert search_input.value == "transformer"
            assert [paper.arxiv_id for paper in app.filtered_papers] == [
                "2401.10003",
                "2401.10004",
            ]
            pill_text = "\n".join(
                str(child.content)
                for child in app.query_one(FilterPillBar).children
                if isinstance(child, Label)
            )
            assert "transformer" in pill_text
            assert "cat:cs.AI" not in pill_text


@pytest.mark.asyncio
@pytest.mark.integration
async def test_detail_focus_a_key_adds_annotation_without_select_all(make_paper):
    paper = make_paper(arxiv_id="2401.10005", abstract="Line one. Line two.")
    app = ArxivBrowser([paper, make_paper(arxiv_id="2401.10006")], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("tab")
            await pilot.pause(0.05)
            await pilot.press("a")
            await pilot.pause(0.05)

            assert isinstance(app.screen, LineAnnotationModal)
            assert app.selected_ids == set()
            input_widget = app.screen.query_one("#line-annotation-input", Input)
            input_widget.value = "check this line"
            await input_widget.action_submit()
            await pilot.pause(0.1)

            metadata = app._config.paper_metadata[paper.arxiv_id]
            assert [(note.line, note.text) for note in metadata.line_annotations] == [
                (1, "check this line")
            ]
            assert "check this line" in str(app.query_one(PaperDetails).content)


@pytest.mark.asyncio
async def test_ascii_high_contrast_layout_contract(make_paper):
    app = ArxivBrowser(
        [make_paper(arxiv_id="2401.00001", title="Accessible UI")],
        ArxivBrowserOptions(
            config=UserConfig(theme_name="high-contrast"),
            restore_session=False,
            ascii_icons=True,
        ),
    )

    with patch_save_config(return_value=True):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)
            visible_text = (
                str(app.query_one("#list-header", Label).content)
                + _footer_text(app)
                + str(app.query_one("#status-bar", Label).content)
            )
            assert all(ord(char) < 128 for char in visible_text)
            assert "Ctrl+p" in visible_text


@pytest.mark.asyncio
async def test_empty_state_layout_contract(make_paper):
    app = ArxivBrowser([], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)
            paper_list = app.query_one("#paper-list", OptionList)
            assert paper_list.option_count == 1
            prompt = str(paper_list.get_option_at_index(0).prompt)
            assert "Try:" in prompt
            assert "Next:" in prompt


@pytest.mark.asyncio
async def test_major_modal_footer_contracts(make_paper):
    from arxiv_browser.modals import CollectionsModal, PaperEditModal, WatchListModal
    from arxiv_browser.models import PaperCollection, WatchListEntry

    paper = make_paper(arxiv_id="2401.00001")
    app = ArxivBrowser([paper], restore_session=False)

    with patch_save_config(return_value=True):
        async with app.run_test(size=(120, 40)) as pilot:
            edit_modal = PaperEditModal(paper.arxiv_id, all_tags=["topic:ml"], initial_tab="tags")
            app.push_screen(edit_modal)
            await pilot.pause(0.05)
            assert "Ctrl+S save" in str(edit_modal.query_one("#edit-help", Static).content)
            edit_modal.dismiss(None)
            await pilot.pause(0.05)

            collections_modal = CollectionsModal(
                [PaperCollection(name="Reading", paper_ids=[paper.arxiv_id])],
                papers_by_id={paper.arxiv_id: paper},
            )
            app.push_screen(collections_modal)
            await pilot.pause(0.05)
            assert "Saved" in str(collections_modal.query_one("#col-help", Static).content)
            collections_modal.dismiss(None)
            await pilot.pause(0.05)

            watch_modal = WatchListModal(
                [WatchListEntry(pattern="graph", match_type="title", case_sensitive=False)]
            )
            app.push_screen(watch_modal)
            await pilot.pause(0.05)
            assert "Ctrl+S save" in str(watch_modal.query_one("#watch-help", Static).content)
