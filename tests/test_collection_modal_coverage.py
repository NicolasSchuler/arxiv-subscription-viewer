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
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionsModal,
    CollectionViewModal,
)
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
        modal._refresh_list = MagicMock()

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
        modal._refresh_list.reset_mock()
        modal.on_create_pressed()
        modal._refresh_list.assert_called_once()

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

    def test_collection_view_and_add_to_collection_branches(self, make_paper) -> None:
        collection = PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        view = CollectionViewModal(collection, papers_by_id={"2401.00001": make_paper()})
        title = _DummyLabel()
        list_view = _DummyListView(index=None)
        view.notify = MagicMock()
        view.dismiss = MagicMock()
        view.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#colview-list": list_view,
                "#colview-title": title,
            }[selector]
        )
        view.on_remove_pressed()
        assert "No paper is selected" in view.notify.call_args[0][0]
        list_view.index = 0
        view.on_remove_pressed()
        assert len(view._collection.paper_ids) == 0
        view.on_done_pressed()
        view.dismiss.assert_called_with(view._collection)

        add = AddToCollectionModal([collection])
        add.dismiss = MagicMock()
        add.query_one = MagicMock(return_value=_DummyListView(index=0))
        add.on_list_selected(SimpleNamespace())
        add.dismiss.assert_called_once_with("Reading")
        add.action_cancel()
        assert add.dismiss.call_args_list[-1].args[0] is None
