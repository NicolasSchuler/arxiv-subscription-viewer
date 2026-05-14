"""Textual layout contracts for key TUI modes and terminal sizes."""

from __future__ import annotations

import pytest
from textual.widgets import Label, OptionList, Static

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.browser.options import ArxivBrowserOptions
from arxiv_browser.models import ArxivSearchModeState, ArxivSearchRequest, UserConfig
from arxiv_browser.widgets.chrome import ContextFooter
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
