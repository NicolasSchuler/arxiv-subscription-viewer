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

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.cli import (
    _resolve_legacy_fallback,
    _resolve_papers,
)
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import (
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
)
from arxiv_browser.semantic_scholar import (
    CitationEntry,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)
from tests.support.app_stubs import (
    _DummyOptionList,
    _make_hf_paper,
    _make_s2_paper,
    _new_app,
)
from tests.support.patch_helpers import patch_save_config


class TestRelevanceBatchCoverage:
    @pytest.mark.asyncio
    async def test_score_relevance_batch_returns_when_all_cached(self, make_paper, tmp_path):
        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        app = _new_app()
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._update_footer = MagicMock()
        app.notify = MagicMock()
        app._relevance_scoring_active = True
        app._scoring_progress = None
        app._llm_provider = None

        cached_scores = {
            "2401.00001": (9, "cached 1"),
            "2401.00002": (7, "cached 2"),
        }
        with patch(
            "arxiv_browser.actions.llm_actions._load_all_relevance_scores",
            return_value=cached_scores,
        ):
            await app._score_relevance_batch_async(papers, "cmd {prompt}", "relevance interests")

        assert app._relevance_scores == cached_scores
        assert app._relevance_scoring_active is False
        assert app._scoring_progress is None
        app.notify.assert_called()
        assert "already scored" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_score_relevance_batch_mixed_results(self, make_paper, tmp_path):
        papers = [make_paper(arxiv_id=f"2401.{i:05d}") for i in range(1, 6)]
        app = _new_app()
        app._config = UserConfig(llm_timeout=45)
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._update_footer = MagicMock()
        app.notify = MagicMock()
        app._relevance_scoring_active = True
        app._scoring_progress = None
        app._llm_provider = object()
        app._services = SimpleNamespace(
            llm=SimpleNamespace(
                score_relevance_once=AsyncMock(
                    side_effect=[
                        (9, "great match"),
                        None,
                        None,
                        RuntimeError("provider crash"),
                        (6, "partial match"),
                    ]
                )
            )
        )

        with (
            patch("arxiv_browser.actions.llm_actions._load_all_relevance_scores", return_value={}),
            patch("arxiv_browser.actions.llm_actions._save_relevance_score", return_value=None),
        ):
            await app._score_relevance_batch_async(papers, "cmd {prompt}", "relevance interests")

        assert set(app._relevance_scores) == {"2401.00001", "2401.00005"}
        assert app._relevance_scores["2401.00001"][0] == 9
        assert app._relevance_scores["2401.00005"][0] == 6
        assert app._relevance_scoring_active is False
        assert app._scoring_progress is None
        assert app._update_relevance_badge.call_count == 2
        assert {
            call.kwargs["timeout_seconds"]
            for call in app._services.llm.score_relevance_once.await_args_list
        } == {45}
        messages = [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        assert any("5/5" in message for message in messages)
        assert any("Relevance scoring complete" in message for message in messages)


class TestVersionCheckCoverage:
    @pytest.mark.asyncio
    async def test_check_versions_async_updates_version_metadata(self):
        app = _new_app()
        app.VERSION_CHECK_BATCH_SIZE = 2
        app._http_client = SimpleNamespace(get=AsyncMock())
        app._apply_arxiv_rate_limit = AsyncMock()
        app._update_footer = MagicMock()
        app._update_status_bar = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app.notify = MagicMock()
        app._version_updates = {}
        app._version_checking = True
        app._version_progress = None

        app._config = UserConfig()
        app._config.paper_metadata = {
            "2401.00001": PaperMetadata(
                arxiv_id="2401.00001",
                starred=True,
                last_checked_version=1,
            ),
            "2401.00002": PaperMetadata(
                arxiv_id="2401.00002",
                starred=True,
                last_checked_version=1,
            ),
            "2401.00003": PaperMetadata(
                arxiv_id="2401.00003",
                starred=False,
                last_checked_version=1,
            ),
        }

        response_1 = MagicMock()
        response_1.text = "<feed batch 1/>"
        response_2 = MagicMock()
        response_2.text = "<feed batch 2/>"
        app._http_client.get.side_effect = [response_1, response_2]

        with (
            patch(
                "arxiv_browser.browser.discovery.parse_arxiv_version_map",
                side_effect=[
                    {"2401.00001": 3, "2401.00002": 1},
                    {"2401.00003": 9},
                ],
            ),
            patch_save_config(return_value=True),
        ):
            await app._check_versions_async({"2401.00001", "2401.00002", "2401.00003"})

        assert app._config.paper_metadata["2401.00001"].last_checked_version == 3
        assert app._config.paper_metadata["2401.00002"].last_checked_version == 1
        assert app._version_updates["2401.00001"] == (1, 3)
        assert app._version_checking is False
        assert app._version_progress is None
        app._mark_badges_dirty.assert_called_once_with("version")
        app._update_status_bar.assert_called()

    @pytest.mark.asyncio
    async def test_check_versions_async_continues_when_one_batch_fails(self):
        app = _new_app()
        app.VERSION_CHECK_BATCH_SIZE = 1
        app._http_client = SimpleNamespace(get=AsyncMock())
        app._apply_arxiv_rate_limit = AsyncMock()
        app._update_footer = MagicMock()
        app._update_status_bar = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app.notify = MagicMock()
        app._version_updates = {}
        app._version_checking = True
        app._version_progress = None

        app._config = UserConfig()
        app._config.paper_metadata = {
            "2401.10001": PaperMetadata(
                arxiv_id="2401.10001",
                starred=True,
                last_checked_version=1,
            ),
            "2401.10002": PaperMetadata(
                arxiv_id="2401.10002",
                starred=True,
                last_checked_version=2,
            ),
        }

        ok_response = MagicMock()
        ok_response.text = "<feed ok/>"
        app._http_client.get.side_effect = [ok_response, httpx.ConnectError("network down")]

        with (
            patch(
                "arxiv_browser.browser.discovery.parse_arxiv_version_map",
                side_effect=[{"2401.10001": 4}],
            ),
            patch_save_config(return_value=True),
        ):
            await app._check_versions_async({"2401.10001", "2401.10002"})

        assert app._version_updates["2401.10001"] == (1, 4)
        assert app._version_checking is False
        messages = [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        assert any("new versions" in message for message in messages)

    @pytest.mark.asyncio
    async def test_check_versions_async_handles_missing_client(self):
        app = _new_app()
        app._http_client = None
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app._apply_arxiv_rate_limit = AsyncMock()
        app._update_footer = MagicMock()
        app._update_status_bar = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app.notify = MagicMock()
        app._version_updates = {}
        app._version_checking = True
        app._version_progress = (1, 1)

        await app._check_versions_async({"2401.99999"})

        assert app._version_checking is False
        assert app._version_progress is None
        app._update_status_bar.assert_called_once()


class TestAutoTagCoverage:
    @pytest.mark.asyncio
    async def test_auto_tag_batch_merges_and_reports_failures(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._config.paper_metadata = {
            "2401.20001": PaperMetadata(arxiv_id="2401.20001", tags=["existing"])
        }
        app.notify = MagicMock()
        app._update_footer = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._auto_tag_active = True
        app._auto_tag_progress = None
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:nlp"], None, ["status:todo"]])

        def get_meta(arxiv_id: str) -> PaperMetadata:
            return app._config.paper_metadata.setdefault(arxiv_id, PaperMetadata(arxiv_id=arxiv_id))

        app._get_or_create_metadata = get_meta

        papers = [
            make_paper(arxiv_id="2401.20001"),
            make_paper(arxiv_id="2401.20002"),
            make_paper(arxiv_id="2401.20003"),
        ]

        with patch_save_config(return_value=True):
            await app._auto_tag_batch_async(papers, taxonomy=["existing"])

        assert "topic:nlp" in app._config.paper_metadata["2401.20001"].tags
        assert "existing" in app._config.paper_metadata["2401.20001"].tags
        assert app._config.paper_metadata["2401.20003"].tags == ["status:todo"]
        assert app._auto_tag_active is False
        assert app._auto_tag_progress is None
        messages = [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        assert any("Auto-tagged 2 papers (1 failed)" in message for message in messages)

    @pytest.mark.asyncio
    async def test_auto_tag_batch_exception_saves_partial_results(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app.notify = MagicMock()
        app._update_footer = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._auto_tag_active = True
        app._auto_tag_progress = None
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:ml"], RuntimeError("boom")])

        def get_meta(arxiv_id: str) -> PaperMetadata:
            return app._config.paper_metadata.setdefault(arxiv_id, PaperMetadata(arxiv_id=arxiv_id))

        app._get_or_create_metadata = get_meta

        papers = [make_paper(arxiv_id="2401.21001"), make_paper(arxiv_id="2401.21002")]

        with patch_save_config(return_value=True) as save:
            await app._auto_tag_batch_async(papers, taxonomy=[])

        save.assert_called_once()
        assert app._config.paper_metadata["2401.21001"].tags == ["topic:ml"]
        assert app._auto_tag_active is False
        assert app._auto_tag_progress is None
        assert "failed" in app.notify.call_args[0][0].lower()

    def test_action_auto_tag_selected_and_single_branches(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        app._collect_all_tags = MagicMock(return_value=["topic:ml"])
        app.notify = MagicMock()
        app._update_footer = MagicMock()
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._auto_tag_batch_async = AsyncMock(return_value=None)
        app._auto_tag_single_async = AsyncMock(return_value=None)

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)

        paper = make_paper(arxiv_id="2401.22001")
        app.selected_ids = {paper.arxiv_id}
        app.all_papers = [paper]
        app.action_auto_tag()
        assert app._track_task.call_count == 1
        assert app._auto_tag_active is True
        assert app._auto_tag_progress == (0, 1)

        app._auto_tag_active = False
        app._auto_tag_progress = None
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=paper)
        app._tags_for = MagicMock(return_value=["old"])
        app.action_auto_tag()
        assert app._track_task.call_count == 2
        assert app._auto_tag_progress is None

        app._auto_tag_active = False
        app._get_current_paper = MagicMock(return_value=None)
        app.action_auto_tag()
        assert app._auto_tag_active is False
        assert "No paper selected" in app.notify.call_args[0][0]

    def test_cancel_search_ignores_single_auto_tag_and_cancels_batch_auto_tag(self):
        app = _new_app()
        app.notify = MagicMock()
        app.action_exit_arxiv_search_mode = MagicMock()
        app._apply_filter = MagicMock()
        app._in_arxiv_api_mode = False
        app._relevance_scoring_active = False
        app._auto_tag_active = True
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._get_search_container_widget = MagicMock(
            return_value=SimpleNamespace(classes=set(), remove_class=MagicMock())
        )

        app.action_cancel_search()

        assert app._cancel_batch_requested is False
        cancel_messages = [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        assert "Cancelling batch operation..." not in cancel_messages

        app._auto_tag_progress = (0, 2)
        app.action_cancel_search()

        assert app._cancel_batch_requested is True
        assert any(
            "Cancelling batch operation..." in msg
            for msg in cancel_messages
            + [str(call.args[0]) for call in app.notify.call_args_list if call.args]
        )
