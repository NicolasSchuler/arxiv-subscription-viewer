#!/usr/bin/env python3
"""High-impact coverage tests for action-heavy paths in app.py."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.semantic_scholar import (
    CitationEntry,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)
from tests.support import canonical_exports as app_mod
from tests.support.app_stubs import (
    _DummyOptionList,
    _make_hf_paper,
    _make_s2_paper,
    _new_app,
)
from tests.support.canonical_exports import (
    ArxivBrowser,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
    _resolve_legacy_fallback,
    _resolve_papers,
)
from tests.support.patch_helpers import patch_save_config


class TestImportCollectionAndTagsCoverage:
    def _make_import_app(self, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app._config.bibtex_export_dir = str(tmp_path)
        app.notify = MagicMock()
        app._compute_watched_papers = MagicMock()
        app._refresh_list_view = MagicMock()
        return app

    def test_action_import_metadata_handles_no_files(self, tmp_path):
        app = self._make_import_app(tmp_path)
        app.action_import_metadata()
        app.notify.assert_called_once()
        assert "No metadata snapshots found" in app.notify.call_args[0][0]
        assert str(tmp_path.resolve()) in app.notify.call_args[0][0]

    def test_action_import_metadata_success_and_save_warning(self, tmp_path):
        app = self._make_import_app(tmp_path)
        export_file = tmp_path / "arxiv-2026-02-13.json"
        export_file.write_text("{}", encoding="utf-8")

        with (
            patch(
                "arxiv_browser.actions.external_io_actions.import_metadata",
                return_value=(2, 1, 1, 1),
            ),
            patch_save_config(return_value=False),
        ):
            app.action_import_metadata()

        app._compute_watched_papers.assert_called_once()
        app._refresh_list_view.assert_called_once()
        messages = [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        assert any("failed to save" in msg.lower() for msg in messages)
        assert any(
            "Imported 2 papers, 1 watch entries, 1 bookmarks, 1 collections" in msg
            for msg in messages
        )
        assert any(str(export_file.resolve()) in msg for msg in messages)

    def test_action_import_metadata_handles_parse_error(self, tmp_path):
        app = self._make_import_app(tmp_path)
        export_file = tmp_path / "arxiv-2026-02-13.json"
        export_file.write_text("{not-json", encoding="utf-8")
        app.action_import_metadata()
        assert "Import failed" in app.notify.call_args[0][0]
        assert str(export_file.resolve()) in app.notify.call_args[0][0]

    def test_action_import_metadata_prompts_for_snapshot_when_multiple_files(self, tmp_path):
        from arxiv_browser.modals import MetadataSnapshotPickerModal

        app = self._make_import_app(tmp_path)
        older = tmp_path / "arxiv-2026-02-12.json"
        newer = tmp_path / "arxiv-2026-02-13.json"
        older.write_text("{}", encoding="utf-8")
        newer.write_text("{}", encoding="utf-8")
        older.touch()
        newer.touch()

        captured = {}
        app.push_screen = lambda modal, callback: captured.update(modal=modal, callback=callback)

        with patch(
            "arxiv_browser.actions.external_io_actions.import_metadata", return_value=(1, 0, 0, 0)
        ):
            app.action_import_metadata()

        assert isinstance(captured["modal"], MetadataSnapshotPickerModal)
        assert captured["modal"]._snapshots == [newer, older]

        with (
            patch(
                "arxiv_browser.actions.external_io_actions.import_metadata",
                return_value=(1, 0, 0, 0),
            ),
            patch_save_config(return_value=True),
        ):
            captured["callback"](older)

        messages = [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        assert any(f"Imported 1 papers from {older.resolve()}" in msg for msg in messages)

    def test_action_import_metadata_picker_cancel_leaves_config_unchanged(self, tmp_path):
        app = self._make_import_app(tmp_path)
        first = tmp_path / "arxiv-2026-02-12.json"
        second = tmp_path / "arxiv-2026-02-13.json"
        first.write_text("{}", encoding="utf-8")
        second.write_text("{}", encoding="utf-8")

        captured = {}
        app.push_screen = lambda modal, callback: captured.update(modal=modal, callback=callback)

        with patch("arxiv_browser.actions.external_io_actions.import_metadata") as import_mock:
            app.action_import_metadata()
            captured["callback"](None)

        import_mock.assert_not_called()
        app._compute_watched_papers.assert_not_called()
        app._refresh_list_view.assert_not_called()

    def test_export_to_file_notifies_absolute_path(self, tmp_path):
        app = self._make_import_app(tmp_path)
        export_file = tmp_path / "arxiv-2026-02-13.csv"

        with patch(
            "arxiv_browser.actions.external_io_actions.write_timestamped_export_file",
            return_value=export_file,
        ):
            app._export_to_file("id,title\n", "csv", "CSV")

        assert str(export_file.resolve()) in app.notify.call_args[0][0]

    def test_action_add_to_collection_adds_new_ids_without_duplicates(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._config.collections = [PaperCollection(name="Reading", paper_ids=["a"])]
        app.notify = MagicMock()
        app._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id="a"), make_paper(arxiv_id="b")]
        )
        captured = {}
        app.push_screen = lambda _modal, cb: captured.setdefault("callback", cb)

        with patch_save_config(return_value=True) as save:
            app.action_add_to_collection()
            captured["callback"]("Reading")

        save.assert_called_once()
        assert app._config.collections[0].paper_ids == ["a", "b"]
        assert "Added 1 paper" in app.notify.call_args[0][0]

    def test_action_add_to_collection_handles_no_collections(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.collections = []
        app.notify = MagicMock()
        app._get_target_papers = MagicMock()
        app.action_add_to_collection()
        app._get_target_papers.assert_not_called()
        assert "No collections" in app.notify.call_args[0][0]

    def test_bulk_edit_tags_applies_added_and_removed_tags(self):
        app = _new_app()
        app._config = UserConfig()
        app.selected_ids = {"a", "b"}
        app._config.paper_metadata = {
            "a": PaperMetadata(arxiv_id="a", tags=["shared", "a-only"]),
            "b": PaperMetadata(arxiv_id="b", tags=["shared", "b-only"]),
        }
        app.notify = MagicMock()

        def get_meta(arxiv_id: str) -> PaperMetadata:
            return app._config.paper_metadata.setdefault(arxiv_id, PaperMetadata(arxiv_id=arxiv_id))

        def apply_to_selected(func, target_ids):
            for arxiv_id in target_ids:
                func(arxiv_id)

        app._get_or_create_metadata = get_meta
        app._apply_to_selected = MagicMock(side_effect=apply_to_selected)
        captured = {}
        app.push_screen = lambda _modal, cb: captured.setdefault("callback", cb)

        app._bulk_edit_tags()
        captured["callback"](["new"])

        assert "shared" not in app._config.paper_metadata["a"].tags
        assert "shared" not in app._config.paper_metadata["b"].tags
        assert "new" in app._config.paper_metadata["a"].tags
        assert "new" in app._config.paper_metadata["b"].tags
        msg = app.notify.call_args[0][0]
        assert "Added new" in msg
        assert "Removed shared" in msg


class TestVisibleIndexCache:
    def _build_index_ready_app(self, papers):
        app = _new_app()
        app.filtered_papers = papers.copy()
        app.all_papers = papers.copy()
        app._s2_cache = {}
        app._hf_cache = {}
        app._relevance_scores = {}
        app._sort_index = app_mod.SORT_OPTIONS.index("arxiv_id")
        app._cancel_pending_detail_update = MagicMock()
        app._config = UserConfig()
        app.selected_ids = set()
        app._watched_paper_ids = set()
        app._show_abstract_preview = False
        app._highlight_terms = {"title": [], "author": [], "abstract": []}
        app._s2_active = False
        app._hf_active = False
        app._version_updates = {}
        app._get_active_query = MagicMock(return_value="")
        app._in_arxiv_api_mode = False
        app._watch_filter_active = False
        app._is_history_mode = MagicMock(return_value=False)
        app._get_abstract_text = MagicMock(return_value="")
        app._get_paper_details_widget = MagicMock()
        app._get_paper_list_widget = MagicMock(return_value=_DummyOptionList())
        return app

    def test_rebuilds_visible_index_during_sort_and_refresh(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00002"),
        ]
        app = self._build_index_ready_app(papers)

        app._sort_papers()
        assert [paper.arxiv_id for paper in app.filtered_papers] == ["2401.00002", "2401.00001"]
        assert app._visible_index_by_id == {"2401.00002": 0, "2401.00001": 1}

        app._refresh_list_view()
        assert app._visible_index_by_id == {"2401.00002": 0, "2401.00001": 1}

    def test_update_option_for_paper_uses_cached_visible_index(self, make_paper):
        first = make_paper(arxiv_id="2401.00011")
        second = make_paper(arxiv_id="2401.00012")
        app = _new_app()
        app.filtered_papers = [first, second]
        app._visible_index_by_id = {second.arxiv_id: 1}
        app._update_option_at_index = MagicMock()

        app._update_option_for_paper(second.arxiv_id)

        app._update_option_at_index.assert_called_once_with(1)

    def test_update_option_for_paper_falls_back_when_cache_is_stale(self, make_paper):
        first = make_paper(arxiv_id="2401.00021")
        second = make_paper(arxiv_id="2401.00022")
        app = _new_app()
        app.filtered_papers = [first, second]
        app._visible_index_by_id = {second.arxiv_id: 0}
        app._update_option_at_index = MagicMock()

        app._update_option_for_paper(second.arxiv_id)

        app._update_option_at_index.assert_called_once_with(1)
        assert app._visible_index_by_id[second.arxiv_id] == 1
