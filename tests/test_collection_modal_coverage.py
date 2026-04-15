"""Additional branch coverage for smaller modules."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import deque
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import arxiv_browser.cli as cli
import arxiv_browser.llm_providers as llm_providers
import arxiv_browser.semantic_scholar as s2
from arxiv_browser.actions import external_io_actions as io_actions
from arxiv_browser.actions import llm_actions as llm_actions
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, UserConfig
from arxiv_browser.services import enrichment_service as enrich
from tests.support.app_stubs import (
    _DummyInput,
    _DummyLabel,
    _DummyListView,
    _DummyTimer,
    _make_app_config,
    _new_app_stub,
    _paper,
)


class TestCollectionsCoverage:
    def test_collections_modal_create_rename_delete_and_view_branches(self, make_paper) -> None:
        base = PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        modal = CollectionsModal([base], papers_by_id={"2401.00001": make_paper()})
        list_view = _DummyListView(index=0)
        name_input = _DummyInput("")
        desc_input = _DummyInput("notes")
        modal.notify = MagicMock()
        modal.dismiss = MagicMock()
        modal._collections = [
            PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        ]
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#col-list": list_view,
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )
        modal._refresh_manage_list = MagicMock()

        modal.on_create_pressed()
        assert "cannot be empty" in modal.notify.call_args[0][0]

        name_input.value = "Reading"
        modal.notify.reset_mock()
        modal.on_create_pressed()
        assert "already exists" in modal.notify.call_args[0][0]

        name_input.value = "New"
        modal._collections = [
            PaperCollection(name=f"C{i}", paper_ids=[]) for i in range(MAX_COLLECTIONS)
        ]
        modal.notify.reset_mock()
        modal.on_create_pressed()
        assert "limit reached" in modal.notify.call_args[0][0]

        modal._collections = [PaperCollection(name="Reading", paper_ids=["2401.00001"])]
        modal._refresh_manage_list.reset_mock()
        modal.on_create_pressed()
        modal._refresh_manage_list.assert_called_once()

        modal._get_selected_index = MagicMock(return_value=None)
        modal.on_rename_pressed()
        assert "No collection is selected" in modal.notify.call_args[0][0]

        modal._get_selected_index = MagicMock(return_value=0)
        name_input.value = ""
        modal.on_rename_pressed()
        assert "cannot be empty" in modal.notify.call_args[0][0]

        name_input.value = "Renamed"
        modal.on_rename_pressed()
        assert modal._collections[0].name == "Renamed"

        modal._get_selected_index = MagicMock(return_value=None)
        modal.on_delete_pressed()
        assert "No collection is selected" in modal.notify.call_args[0][0]

        modal._get_selected_index = MagicMock(return_value=0)
        modal._collections.append(PaperCollection(name="Other", paper_ids=[]))
        before_delete = len(modal._collections)
        modal.on_delete_pressed()
        assert len(modal._collections) == before_delete - 1

    def test_detail_view_remove_and_back_branches(self, make_paper) -> None:
        collection = PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        modal = CollectionsModal([collection], papers_by_id={"2401.00001": make_paper()})
        detail_title = _DummyLabel()
        detail_list = _DummyListView(index=None)
        manage_view = MagicMock()
        manage_view.display = True
        detail_view = MagicMock()
        detail_view.display = False
        pick_view = MagicMock()
        pick_view.display = False
        col_title = _DummyLabel()
        col_list = _DummyListView(index=0)
        name_input = _DummyInput("Reading")
        desc_input = _DummyInput("desc")

        modal.notify = MagicMock()
        modal.dismiss = MagicMock()
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#detail-list": detail_list,
                "#detail-title": detail_title,
                "#manage-view": manage_view,
                "#detail-view": detail_view,
                "#pick-view": pick_view,
                "#col-title": col_title,
                "#col-list": col_list,
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )

        # Set up a viewed collection
        modal._viewing_collection = PaperCollection(
            name="Reading", description="desc", paper_ids=["2401.00001"]
        )
        # Remove with no selection
        modal.on_detail_remove_pressed()
        assert "No paper is selected" in modal.notify.call_args[0][0]
        # Remove with valid selection
        detail_list.index = 0
        modal.on_detail_remove_pressed()
        assert len(modal._viewing_collection.paper_ids) == 0
        # Back to manage
        modal.on_detail_back_pressed()
        assert modal._viewing_collection is None

    def test_pick_mode_select_and_cancel(self, make_paper) -> None:
        collection = PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        modal = CollectionsModal([collection], mode="pick")
        modal.dismiss = MagicMock()
        modal.query_one = MagicMock(return_value=_DummyListView(index=0))
        modal.on_pick_list_selected(SimpleNamespace())
        modal.dismiss.assert_called_once_with("Reading")
        modal.dismiss.reset_mock()
        modal.action_cancel_or_back()
        assert modal.dismiss.call_args_list[-1].args[0] is None

    def test_manage_view_save_close_and_view_no_selection(self, make_paper) -> None:
        modal = CollectionsModal([], papers_by_id={})
        list_view = _DummyListView(index=None)
        name_input = _DummyInput("")
        desc_input = _DummyInput("")
        modal.dismiss = MagicMock()
        modal.notify = MagicMock()
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#col-list": list_view,
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )

        # View with no selection
        modal.on_view_pressed()
        assert "No collection is selected" in modal.notify.call_args[0][0]

        # Save
        modal.on_save_pressed()
        modal.dismiss.assert_called_with("save")

        # Close
        modal.dismiss.reset_mock()
        modal.on_close_pressed()
        modal.dismiss.assert_called_with(None)

    def test_escape_from_detail_view_goes_back(self, make_paper) -> None:
        collection = PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        modal = CollectionsModal([collection], papers_by_id={"2401.00001": make_paper()})
        manage_view = MagicMock()
        detail_view = MagicMock()
        pick_view = MagicMock()
        col_title = _DummyLabel()
        detail_title = _DummyLabel()
        detail_list = _DummyListView(index=0)
        col_list = _DummyListView(index=0)
        name_input = _DummyInput("Reading")
        desc_input = _DummyInput("desc")
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#manage-view": manage_view,
                "#detail-view": detail_view,
                "#pick-view": pick_view,
                "#col-title": col_title,
                "#detail-title": detail_title,
                "#detail-list": detail_list,
                "#col-list": col_list,
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )
        modal.dismiss = MagicMock()

        # Put modal in detail-view state
        modal._viewing_collection = PaperCollection(
            name="Reading", description="desc", paper_ids=["2401.00001"]
        )
        # Escape should go back to manage, not dismiss
        modal.action_cancel_or_back()
        assert modal._viewing_collection is None
        modal.dismiss.assert_not_called()

    def test_populate_form_out_of_range(self) -> None:
        modal = CollectionsModal([])
        name_input = _DummyInput("")
        desc_input = _DummyInput("")
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )
        # Negative index — should be a no-op
        modal._populate_form(-1)
        assert name_input.value == ""
        # Out of range — should be a no-op
        modal._populate_form(5)
        assert name_input.value == ""

    def test_pick_list_selected_out_of_range(self) -> None:
        modal = CollectionsModal([], mode="pick")
        modal.dismiss = MagicMock()
        modal.query_one = MagicMock(return_value=_DummyListView(index=5))
        modal.on_pick_list_selected(SimpleNamespace())
        modal.dismiss.assert_not_called()

    def test_refresh_manage_list_empty(self) -> None:
        modal = CollectionsModal([])
        list_view = _DummyListView(index=None)
        list_view.children = []
        modal.query_one = MagicMock(return_value=list_view)
        modal._refresh_manage_list()
        # No children — should not crash or call _populate_form

    def test_refresh_pick_list_empty(self) -> None:
        modal = CollectionsModal([], mode="pick")
        list_view = _DummyListView(index=None)
        list_view.children = []
        modal.query_one = MagicMock(return_value=list_view)
        modal._refresh_pick_list()
        # No children — should not crash
