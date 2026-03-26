"""Focused branch/behavior tests for hotspot modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.css.query import NoMatches

import arxiv_browser.app as app_mod
from arxiv_browser.actions import library_actions, llm_actions
from arxiv_browser.browser import _runtime as browser_runtime
from arxiv_browser.browser import discovery
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.modals.search import CommandPaletteModal, _truncate_palette_text
from arxiv_browser.models import PaperCollection, PaperMetadata, WatchListEntry
from arxiv_browser.semantic_scholar import S2RecommendationsCacheSnapshot, SemanticScholarPaper
from arxiv_browser.services import enrichment_service as enrich
from tests.support.app_stubs import _make_app_config, _new_app_stub, _paper


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
        callback(["topic:a"])

        assert metadata.tags == ["topic:a"]
        app._update_option_at_index.assert_called_once_with(2)
        assert "Tags: topic:a" in app.notify.call_args.args[0]

        app._update_option_at_index.reset_mock()
        app._get_current_paper = MagicMock(return_value=other)
        callback(["topic:b"])
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

        with patch("arxiv_browser.app.save_config", return_value=False):
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

        with patch("arxiv_browser.app.save_config", return_value=True):
            library_actions.action_manage_watch_list(app)

        assert app._watch_filter_active is False
        app._compute_watched_papers.assert_called_once()
        app._apply_filter.assert_called_once_with("graph")
        assert "Watch list updated" in app.notify.call_args.args[0]


class _FakeOptionList:
    def __init__(self, options: list[SimpleNamespace], highlighted: int | None = 0) -> None:
        self._options = options
        self.option_count = len(options)
        self.highlighted = highlighted

    def get_option_at_index(self, index: int) -> SimpleNamespace:
        return self._options[index]


class TestSearchModalBehavior:
    def test_truncate_palette_text_honors_short_max_len(self) -> None:
        assert _truncate_palette_text("abcdef", 3) == "abc"
        assert _truncate_palette_text("abcdef", 2) == "ab"
        assert _truncate_palette_text("abcdef", 1) == "a"

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
    def test_on_view_result_none_does_not_refresh(self) -> None:
        collection = PaperCollection(name="Reading", paper_ids=["2401.00001"])
        modal = CollectionsModal([collection], papers_by_id={})
        modal._refresh_list = MagicMock()

        modal._on_view_result(None)
        modal._refresh_list.assert_not_called()

    def test_on_view_result_found_and_not_found_paths(self) -> None:
        original = PaperCollection(name="Reading", paper_ids=["2401.00001"])
        replacement = PaperCollection(name="Reading", paper_ids=["2401.00002"])
        modal = CollectionsModal([original], papers_by_id={})
        modal._refresh_list = MagicMock()

        modal._on_view_result(replacement)
        assert modal.collections[0].paper_ids == ["2401.00002"]
        modal._refresh_list.assert_called_once()

        modal._refresh_list.reset_mock()
        modal._on_view_result(PaperCollection(name="Unknown", paper_ids=[]))
        assert modal.collections[0].name == "Reading"
        modal._refresh_list.assert_called_once()


class TestBrowserRuntimeBehavior:
    def test_build_list_empty_message_covers_all_states(self) -> None:
        assert "No papers match your search" in browser_runtime.build_list_empty_message(
            query="graph",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No API results on this page" in browser_runtime.build_list_empty_message(
            query="",
            in_arxiv_api_mode=True,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No watched papers found" in browser_runtime.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=True,
            history_mode=False,
        )
        assert "No papers available for this date" in browser_runtime.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=True,
        )
        assert "No papers available" in browser_runtime.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )

    @pytest.mark.asyncio
    async def test_sync_app_methods_wraps_instance_static_class_and_async(self) -> None:
        calls: list[dict[str, object]] = []

        def _sync(namespace: dict[str, object]) -> None:
            calls.append(namespace)

        class Demo:
            def inc(self, value: int) -> int:
                return value + 1

            @staticmethod
            def twice(value: int) -> int:
                return value * 2

            @classmethod
            def plus_three(cls, value: int) -> int:
                return value + 3

            async def async_inc(self, value: int) -> int:
                return value + 1

        with patch("arxiv_browser.browser._runtime.sync_browser_globals", side_effect=_sync):
            Demo = browser_runtime.sync_app_methods(Demo)
            demo = Demo()
            assert demo.inc(2) == 3
            assert Demo.twice(3) == 6
            assert Demo.plus_three(4) == 7
            assert await demo.async_inc(5) == 6

        assert len(calls) == 4


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
        app._build_tfidf_index_async = app_mod.ArxivBrowser._build_tfidf_index_async.__get__(
            app, app_mod.ArxivBrowser
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
        app._capture_local_browse_snapshot = (
            app_mod.ArxivBrowser._capture_local_browse_snapshot.__get__(app, app_mod.ArxivBrowser)
        )
        app._update_filter_pills = app_mod.ArxivBrowser._update_filter_pills.__get__(
            app, app_mod.ArxivBrowser
        )
        app._in_arxiv_api_mode = True
        app._get_paper_list_widget = MagicMock(side_effect=NoMatches())
        app._get_filter_pill_bar_widget = MagicMock(side_effect=NoMatches())

        assert app._capture_local_browse_snapshot() is None
        app._update_filter_pills("cat:cs.AI")
        app._track_task.assert_not_called()
