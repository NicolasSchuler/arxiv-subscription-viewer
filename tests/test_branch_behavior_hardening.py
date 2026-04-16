"""Focused branch/behavior tests for hotspot modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.css.query import NoMatches

from arxiv_browser.actions import library_actions, llm_actions
from arxiv_browser.browser import discovery
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.browser.empty_state import build_list_empty_message
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.modals.editing import PaperEditResult
from arxiv_browser.modals.search import CommandPaletteModal
from arxiv_browser.models import PaperCollection, PaperMetadata, WatchListEntry
from arxiv_browser.palette import truncate_palette_text
from arxiv_browser.semantic_scholar import S2RecommendationsCacheSnapshot, SemanticScholarPaper
from arxiv_browser.services import enrichment_service as enrich
from tests.support.app_stubs import (
    _DummyInput,
    _DummyLabel,
    _DummyListView,
    _make_app_config,
    _new_app_stub,
    _paper,
)
from tests.support.patch_helpers import patch_save_config


class TestLibraryActionBehavior:
    def test_edit_tags_callback_updates_option_only_for_active_paper(self) -> None:
        app = _new_app_stub()
        paper = _paper("2403.10001")
        other = _paper("2403.10002")
        metadata = PaperMetadata(arxiv_id=paper.arxiv_id, tags=[])

        app.selected_ids = set()
        app._config = _make_app_config(paper_metadata={})
        app._collect_all_tags = MagicMock(return_value=["existing"])
        app._get_or_create_metadata = MagicMock(return_value=metadata)
        app._get_current_index = MagicMock(return_value=2)
        app._get_current_paper = MagicMock(return_value=paper)
        app._update_option_at_index = MagicMock()
        app.push_screen = MagicMock()

        library_actions.action_edit_tags(app)
        callback = app.push_screen.call_args.args[1]
        callback(PaperEditResult(notes="", tags=["topic:a"], active_tab="tags"))

        assert metadata.tags == ["topic:a"]
        app._update_option_at_index.assert_called_once_with(2)
        assert "Tags: topic:a" in app.notify.call_args.args[0]

        app._update_option_at_index.reset_mock()
        app._get_current_paper = MagicMock(return_value=other)
        callback(PaperEditResult(notes="", tags=["topic:b"], active_tab="tags"))
        app._update_option_at_index.assert_not_called()

    def test_toggle_watch_filter_handles_empty_and_non_empty_watch_sets(self) -> None:
        app = _new_app_stub()
        app._watch_filter_active = False
        app._watched_paper_ids = set()
        app._get_search_input_widget = MagicMock(
            return_value=SimpleNamespace(value="  cat:cs.AI  ")
        )
        app._apply_filter = MagicMock()

        library_actions.action_toggle_watch_filter(app)
        assert app._watch_filter_active is False
        app._apply_filter.assert_not_called()
        assert app.notify.call_args.kwargs["severity"] == "warning"

        app._watched_paper_ids = {"2403.20001"}
        app.notify.reset_mock()
        library_actions.action_toggle_watch_filter(app)
        assert app._watch_filter_active is True
        app._apply_filter.assert_called_once_with("cat:cs.AI")
        assert "Showing watched papers" in app.notify.call_args.args[0]

        app.notify.reset_mock()
        library_actions.action_toggle_watch_filter(app)
        assert app._watch_filter_active is False
        assert app._apply_filter.call_count == 2
        assert "Showing all papers" in app.notify.call_args.args[0]

    def test_manage_watch_list_reverts_on_save_failure(self) -> None:
        app = _new_app_stub()
        old_entry = WatchListEntry(pattern="old", match_type="title")
        new_entry = WatchListEntry(pattern="new", match_type="author")
        app._config = _make_app_config(watch_list=[old_entry])
        app._watch_filter_active = False
        app._watched_paper_ids = set()
        app._get_search_input_widget = MagicMock(return_value=SimpleNamespace(value="graph"))
        app._compute_watched_papers = MagicMock()
        app._apply_filter = MagicMock()
        app.push_screen = lambda _screen, callback: callback([new_entry])

        with patch_save_config(return_value=False):
            library_actions.action_manage_watch_list(app)

        assert app._config.watch_list == [old_entry]
        app._compute_watched_papers.assert_not_called()
        app._apply_filter.assert_not_called()
        assert "Failed to save watch list" in app.notify.call_args.args[0]

    def test_manage_watch_list_success_clears_empty_watch_filter(self) -> None:
        app = _new_app_stub()
        new_entry = WatchListEntry(pattern="new", match_type="author")
        app._config = _make_app_config(watch_list=[])
        app._watch_filter_active = True
        app._watched_paper_ids = set()
        app._get_search_input_widget = MagicMock(return_value=SimpleNamespace(value="  graph  "))
        app._compute_watched_papers = MagicMock()
        app._apply_filter = MagicMock()
        app.push_screen = lambda _screen, callback: callback([new_entry])

        with patch_save_config(return_value=True):
            library_actions.action_manage_watch_list(app)

        assert app._watch_filter_active is False
        app._compute_watched_papers.assert_called_once()
        app._apply_filter.assert_called_once_with("graph")
        assert "Watch list updated" in app.notify.call_args.args[0]

    def test_toggle_select_no_paper_and_idx_none(self) -> None:
        """Line 36 (return when no paper) and 43->45 (idx None skips update_option)."""
        app = _new_app_stub()

        # Line 36: action_toggle_select returns early when paper is None
        app._get_current_paper = MagicMock(return_value=None)
        library_actions.action_toggle_select(app)
        app._update_header.assert_not_called()

        # 43->45: idx is None → _update_option_at_index skipped, _update_header still called
        paper = _paper("2401.99001")
        app._get_current_paper = MagicMock(return_value=paper)
        app._update_option_at_index = MagicMock()
        app._get_current_index = MagicMock(return_value=None)
        library_actions.action_toggle_select(app)
        app._update_option_at_index.assert_not_called()
        app._update_header.assert_called()

    def test_toggle_read_and_star_with_selection(self) -> None:
        """Lines 80-81 and 98-99: bulk-toggle when selected_ids is non-empty."""
        app = _new_app_stub()
        paper = _paper("2401.99002")
        app.selected_ids = {paper.arxiv_id}
        app._bulk_toggle_bool = MagicMock()

        # Lines 80-81: action_toggle_read with selected_ids → bulk path
        library_actions.action_toggle_read(app)
        app._bulk_toggle_bool.assert_called_once_with(
            "is_read", "marked read", "marked unread", "Read Status"
        )

        # Lines 98-99: action_toggle_star with selected_ids → bulk path
        app._bulk_toggle_bool.reset_mock()
        library_actions.action_toggle_star(app)
        app._bulk_toggle_bool.assert_called_once_with("starred", "starred", "unstarred", "Star")

    def test_edit_notes_guard_and_callback_branches(self) -> None:
        """Line 117, 131->135, 133->135: edit_notes guard and callback edge paths."""
        app = _new_app_stub()

        # Line 117: action_edit_notes returns early when paper is None
        app._get_current_paper = MagicMock(return_value=None)
        app.push_screen = MagicMock()
        library_actions.action_edit_notes(app)
        app.push_screen.assert_not_called()

        # Set up paper with existing notes in paper_metadata
        paper = _paper("2401.99003")
        metadata = PaperMetadata(arxiv_id=paper.arxiv_id, notes="old note")
        app._config = _make_app_config(paper_metadata={paper.arxiv_id: metadata})
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_or_create_metadata = MagicMock(return_value=metadata)
        app._update_option_at_index = MagicMock()
        app.push_screen = MagicMock()
        library_actions.action_edit_notes(app)
        callback = app.push_screen.call_args.args[1]

        # 131->135: cur is None → update_option_at_index NOT called, notify still fires
        app._get_current_paper = MagicMock(return_value=None)
        callback(PaperEditResult(notes="saved note", tags=[], active_tab="notes"))
        app._update_option_at_index.assert_not_called()
        assert "Saved" in app.notify.call_args.args[0]

        # 133->135: same paper but idx is None → update_option_at_index NOT called
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_current_index = MagicMock(return_value=None)
        app.notify.reset_mock()
        callback(PaperEditResult(notes="another note", tags=[], active_tab="notes"))
        app._update_option_at_index.assert_not_called()
        assert "Saved" in app.notify.call_args.args[0]

    def test_edit_tags_guard_bulk_and_no_paper(self) -> None:
        """Lines 143-144 (bulk path), 148 (no paper guard), 167->169 (cur mismatch)."""
        app = _new_app_stub()
        paper = _paper("2401.99004")

        # Lines 143-144: action_edit_tags with selected_ids → bulk path and early return
        app.selected_ids = {paper.arxiv_id}
        app._bulk_edit_tags = MagicMock()
        library_actions.action_edit_tags(app)
        app._bulk_edit_tags.assert_called_once()

        # Line 148: action_edit_tags returns early when paper is None (no selection)
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=None)
        app._bulk_edit_tags.reset_mock()
        app.push_screen = MagicMock()
        library_actions.action_edit_tags(app)
        app.push_screen.assert_not_called()

        # 167->169: cur is None when tags callback fires → update_option_at_index skipped
        metadata = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["existing-tag"])
        app._config = _make_app_config(paper_metadata={paper.arxiv_id: metadata})
        app._get_current_paper = MagicMock(return_value=paper)
        app._collect_all_tags = MagicMock(return_value=["existing-tag"])
        app._get_or_create_metadata = MagicMock(return_value=metadata)
        app._update_option_at_index = MagicMock()
        app.push_screen = MagicMock()
        library_actions.action_edit_tags(app)
        callback = app.push_screen.call_args.args[1]

        app._get_current_paper = MagicMock(return_value=None)
        callback(PaperEditResult(notes="", tags=["new-tag"], active_tab="tags"))
        app._update_option_at_index.assert_not_called()
        assert "Tags: new-tag" in app.notify.call_args.args[0]

    def test_manage_watch_list_callback_none_is_noop(self) -> None:
        """Line 197: on_watch_list_updated with None entries returns immediately."""
        app = _new_app_stub()
        app._config = _make_app_config(watch_list=[])
        app._compute_watched_papers = MagicMock()
        app._apply_filter = MagicMock()
        app.push_screen = MagicMock()
        library_actions.action_manage_watch_list(app)
        callback = app.push_screen.call_args.args[1]

        # Passing None should hit line 197 (return) and do nothing else
        callback(None)
        app._compute_watched_papers.assert_not_called()
        app._apply_filter.assert_not_called()


class _FakeOptionList:
    def __init__(self, options: list[SimpleNamespace], highlighted: int | None = 0) -> None:
        self._options = options
        self.option_count = len(options)
        self.highlighted = highlighted

    def get_option_at_index(self, index: int) -> SimpleNamespace:
        return self._options[index]


class TestSearchModalBehavior:
    def test_truncate_palette_text_honors_short_max_len(self) -> None:
        assert truncate_palette_text("abcdef", 3) == "abc"
        assert truncate_palette_text("abcdef", 2) == "ab"
        assert truncate_palette_text("abcdef", 1) == "a"

    def test_highlight_first_enabled_handles_all_disabled(self) -> None:
        option_list = _FakeOptionList(
            [SimpleNamespace(disabled=True), SimpleNamespace(disabled=True)],
            highlighted=0,
        )
        CommandPaletteModal._highlight_first_enabled(option_list)  # type: ignore[arg-type]
        assert option_list.highlighted is None

    def test_option_selection_and_enter_guards(self) -> None:
        modal = CommandPaletteModal(commands=[])
        modal.dismiss = MagicMock()
        modal._on_option_selected(SimpleNamespace(option_id=None))
        modal.dismiss.assert_not_called()

        modal._on_option_selected(SimpleNamespace(option_id="run"))
        modal.dismiss.assert_called_once_with("run")

        modal.dismiss.reset_mock()
        option_list = _FakeOptionList([SimpleNamespace(disabled=True, id="x")], highlighted=0)
        modal.query_one = MagicMock(return_value=option_list)
        modal.key_enter()
        modal.dismiss.assert_not_called()

        option_list = _FakeOptionList([SimpleNamespace(disabled=False, id="x")], highlighted=0)
        modal.query_one = MagicMock(return_value=option_list)
        modal.key_enter()
        modal.dismiss.assert_called_once_with("x")


class TestCollectionsModalBehavior:
    def test_detail_back_with_no_viewing_collection(self) -> None:
        """Back from detail-view with no _viewing_collection is a no-op on collections."""
        collection = PaperCollection(name="Reading", paper_ids=["2401.00001"])
        modal = CollectionsModal([collection], papers_by_id={})
        manage_view = MagicMock()
        detail_view = MagicMock()
        pick_view = MagicMock()
        col_title = _DummyLabel()
        col_list = _DummyListView(index=0)
        name_input = _DummyInput("Reading")
        desc_input = _DummyInput("")
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#manage-view": manage_view,
                "#detail-view": detail_view,
                "#pick-view": pick_view,
                "#col-title": col_title,
                "#col-list": col_list,
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )
        # _viewing_collection is None by default, back should not crash
        modal.on_detail_back_pressed()
        assert modal.collections[0].paper_ids == ["2401.00001"]

    def test_detail_back_applies_paper_removal(self) -> None:
        """Back from detail-view applies paper changes to the parent collection."""
        original = PaperCollection(name="Reading", paper_ids=["2401.00001"])
        modal = CollectionsModal([original], papers_by_id={})
        manage_view = MagicMock()
        detail_view = MagicMock()
        pick_view = MagicMock()
        col_title = _DummyLabel()
        col_list = _DummyListView(index=0)
        name_input = _DummyInput("Reading")
        desc_input = _DummyInput("")
        modal.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#manage-view": manage_view,
                "#detail-view": detail_view,
                "#pick-view": pick_view,
                "#col-title": col_title,
                "#col-list": col_list,
                "#col-name": name_input,
                "#col-desc": desc_input,
            }[selector]
        )
        # Simulate viewing with modified paper list
        modal._viewing_collection = PaperCollection(name="Reading", paper_ids=["2401.00002"])
        modal.on_detail_back_pressed()
        assert modal.collections[0].paper_ids == ["2401.00002"]

        # Unknown collection name — should not change existing collections
        modal._viewing_collection = PaperCollection(name="Unknown", paper_ids=[])
        modal.on_detail_back_pressed()
        assert modal.collections[0].name == "Reading"


class TestBrowserRuntimeBehavior:
    def test_build_list_empty_message_covers_all_states(self) -> None:
        assert "No papers match your search" in build_list_empty_message(
            query="graph",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No API results on this page" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=True,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No watched papers found" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=True,
            history_mode=False,
        )
        assert "No papers available for this date" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=True,
        )
        assert "No papers available" in build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )

    def test_legacy_sync_wrappers_are_removed(self) -> None:
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("arxiv_browser.browser._runtime")


class TestLlmActionBehavior:
    @pytest.mark.asyncio
    async def test_score_relevance_recoverable_error_resets_state(self) -> None:
        app = _new_app_stub()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._relevance_db_path = Path("/tmp/relevance.db")
        app._relevance_scores = {}
        app._relevance_scoring_active = True
        app._scoring_progress = (1, 2)
        app._cancel_batch_requested = True

        with patch(
            "arxiv_browser.actions.llm_actions.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [_paper("2403.30001")],
                "echo {prompt}",
                "interest",
            )

        assert app.notify.call_args.kwargs["severity"] == "error"
        assert app._relevance_scoring_active is False
        assert app._scoring_progress is None
        assert app._cancel_batch_requested is False

    @pytest.mark.asyncio
    async def test_score_relevance_stale_epoch_suppresses_notify_and_finally(self) -> None:
        app = _new_app_stub()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._relevance_db_path = Path("/tmp/relevance.db")
        app._relevance_scores = {}
        app._relevance_scoring_active = True
        app._scoring_progress = (1, 2)
        app._cancel_batch_requested = True

        with patch(
            "arxiv_browser.actions.llm_actions.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [_paper("2403.30002")],
                "echo {prompt}",
                "interest",
            )

        app.notify.assert_not_called()
        assert app._relevance_scoring_active is True
        assert app._scoring_progress == (1, 2)
        assert app._cancel_batch_requested is True

    @pytest.mark.asyncio
    async def test_auto_tag_batch_recoverable_error_saves_partial_and_resets(self) -> None:
        app = _new_app_stub()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._config = _make_app_config(paper_metadata={})
        app._auto_tag_active = True
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._get_or_create_metadata = MagicMock(
            side_effect=lambda aid: app._config.paper_metadata.setdefault(
                aid, PaperMetadata(arxiv_id=aid, tags=["existing"])
            )
        )
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:new"], RuntimeError("boom")])

        papers = [_paper("2403.40001"), _paper("2403.40002")]
        await llm_actions._auto_tag_batch_async(app, papers, ["existing"])

        assert any(
            call.args == ("partial auto-tag results",)
            for call in app._save_config_or_warn.call_args_list
        )
        assert "1 tagged before error" in app.notify.call_args.args[0]
        assert app.notify.call_args.kwargs["severity"] == "error"
        assert app._auto_tag_active is False
        assert app._auto_tag_progress is None
        assert app._cancel_batch_requested is False

    @pytest.mark.asyncio
    async def test_auto_tag_batch_stale_epoch_skips_notify_and_state_reset(self) -> None:
        app = _new_app_stub()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        app._auto_tag_active = True
        app._auto_tag_progress = None
        app._cancel_batch_requested = True
        app._call_auto_tag_llm = AsyncMock(return_value=["topic:a"])

        await llm_actions._auto_tag_batch_async(app, [_paper("2403.40003")], ["existing"])

        app.notify.assert_not_called()
        assert app._auto_tag_active is True
        assert app._auto_tag_progress == (0, 1)
        assert app._cancel_batch_requested is True


class TestEnrichmentCachedIncludeStatusBehavior:
    @pytest.mark.asyncio
    async def test_s2_recommendations_cached_found_and_empty_paths(self, tmp_path) -> None:
        rec = SemanticScholarPaper(
            arxiv_id="2403.50001",
            s2_paper_id="s2:2403.50001",
            citation_count=1,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com/recs",
        )
        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=S2RecommendationsCacheSnapshot(status="found", papers=[rec]),
        ):
            assert await enrich.load_or_fetch_s2_recommendations_cached(
                arxiv_id=rec.arxiv_id,
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=object(),
                api_key="",
            ) == [rec]

        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=S2RecommendationsCacheSnapshot(status="empty", papers=[]),
        ):
            assert await enrich.load_or_fetch_s2_recommendations_cached(
                arxiv_id=rec.arxiv_id,
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=object(),
                api_key="",
                include_status=True,
            ) == ([], True)

    @pytest.mark.asyncio
    async def test_s2_recommendations_cached_miss_no_client_and_remote(self, tmp_path) -> None:
        rec = SemanticScholarPaper(
            arxiv_id="2403.50002",
            s2_paper_id="s2:2403.50002",
            citation_count=2,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com/recs2",
        )
        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=S2RecommendationsCacheSnapshot(status="miss", papers=[]),
        ):
            assert await enrich.load_or_fetch_s2_recommendations_cached(
                arxiv_id=rec.arxiv_id,
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=None,
                api_key="",
                include_status=True,
            ) == ([], True)

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=S2RecommendationsCacheSnapshot(status="miss", papers=[]),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.load_or_fetch_s2_recommendations_result",
                new=AsyncMock(
                    return_value=enrich.S2RecommendationsFetchResult(
                        state="found",
                        papers=[rec],
                        complete=False,
                        from_cache=False,
                    )
                ),
            ),
        ):
            assert await enrich.load_or_fetch_s2_recommendations_cached(
                arxiv_id=rec.arxiv_id,
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=object(),
                api_key="",
                include_status=True,
            ) == ([rec], False)


class TestDiscoveryAndBrowseBehavior:
    @pytest.mark.asyncio
    async def test_build_tfidf_index_async_error_and_stale_epoch_paths(self) -> None:
        app = _new_app_stub()
        app._build_tfidf_index_async = ArxivBrowser._build_tfidf_index_async.__get__(
            app, ArxivBrowser
        )
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app.all_papers = [_paper("2403.60001"), _paper("2403.60002")]
        app._tfidf_build_task = object()
        app._is_current_dataset_epoch = MagicMock(return_value=True)

        with patch(
            "arxiv_browser.browser.discovery.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await app._build_tfidf_index_async(
                discovery.build_similarity_corpus_key(app.all_papers)
            )
        assert app.notify.call_args.kwargs["severity"] == "error"
        assert app._tfidf_build_task is None

        sentinel = object()
        app.notify.reset_mock()
        app._tfidf_build_task = sentinel
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        with patch(
            "arxiv_browser.browser.discovery.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await app._build_tfidf_index_async(
                discovery.build_similarity_corpus_key(app.all_papers)
            )
        app.notify.assert_not_called()
        assert app._tfidf_build_task is sentinel

    def test_browse_fallback_paths_handle_missing_widgets(self) -> None:
        app = _new_app_stub()
        app._capture_local_browse_snapshot = ArxivBrowser._capture_local_browse_snapshot.__get__(
            app, ArxivBrowser
        )
        app._update_filter_pills = ArxivBrowser._update_filter_pills.__get__(app, ArxivBrowser)
        app._in_arxiv_api_mode = True
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        app._get_filter_pill_bar_widget = MagicMock(side_effect=NoMatches())

        assert app._capture_local_browse_snapshot() is None
        app._update_filter_pills("cat:cs.AI")
        app._track_task.assert_not_called()
