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
from textual.app import ScreenStackError

import arxiv_browser.actions.ui_actions as ui_actions
import arxiv_browser.cli as cli
import arxiv_browser.llm_providers as llm_providers
import arxiv_browser.semantic_scholar as s2
from arxiv_browser.actions import external_io_actions as io_actions
from arxiv_browser.actions import llm_actions as llm_actions
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionsModal,
    CollectionViewModal,
)
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, PaperMetadata, UserConfig
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
from tests.support.patch_helpers import patch_save_config


class TestLlmActionCoverage:
    def test_trust_and_command_resolution_branches(self, tmp_path) -> None:
        app = _new_app_stub()
        app._trust_hash = llm_actions._trust_hash
        app._config = UserConfig()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._config.trusted_llm_command_hashes = []
        app._config.trusted_pdf_viewer_hashes = []

        app._config.llm_command = ""
        app._config.llm_preset = ""
        assert app._is_llm_command_trusted("echo {prompt}") is True

        preset_name = next(iter(llm_actions.LLM_PRESETS))
        app._config.llm_preset = preset_name
        assert app._is_llm_command_trusted(llm_actions.LLM_PRESETS[preset_name]) is True

        app._config.llm_preset = ""
        app._config.llm_command = "echo {prompt}"
        trusted_hash = llm_actions._trust_hash("echo {prompt}")
        app._config.trusted_llm_command_hashes = [trusted_hash]
        assert app._is_llm_command_trusted("echo {prompt}") is True
        app._config.trusted_pdf_viewer_hashes = [llm_actions._trust_hash("viewer {url}")]
        assert app._is_pdf_viewer_trusted("viewer {url}") is True

        app.notify.reset_mock()
        app._config.trusted_llm_command_hashes = []
        with patch_save_config(return_value=False):
            assert (
                app._remember_trusted_hash(
                    "echo {prompt}", app._config.trusted_llm_command_hashes, "LLM"
                )
                is True
            )
        assert "session only" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        on_trusted = MagicMock()
        request = llm_actions.CommandTrustRequest(
            command_template="viewer {url}",
            title="PDF",
            prompt_heading="Run untrusted custom PDF viewer command?",
            trust_button_label="Open",
            cancel_message="PDF open cancelled",
            trusted_hashes=app._config.trusted_pdf_viewer_hashes,
            on_trusted=on_trusted,
        )
        with patch_save_config(return_value=True):
            assert app._ensure_command_trusted(request) is False
        callback = app.push_screen.call_args.args[1]
        callback(None)
        assert "PDF open cancelled" in app.notify.call_args[0][0]
        app.notify.reset_mock()
        callback(True)
        on_trusted.assert_called_once()

        app.push_screen = MagicMock(side_effect=ScreenStackError("boom"))
        with patch_save_config(return_value=True):
            assert app._ensure_command_trusted(request) is False
        assert "could not confirm pdf command trust" in app.notify.call_args[0][0].lower()

        app._is_llm_command_trusted = MagicMock(return_value=True)
        app._ensure_command_trusted = MagicMock(return_value=False)
        assert app._ensure_llm_command_trusted("cmd {prompt}", MagicMock()) is True
        app._is_llm_command_trusted.return_value = False
        assert app._ensure_llm_command_trusted("cmd {prompt}", MagicMock()) is False
        app._is_pdf_viewer_trusted = MagicMock(return_value=True)
        assert app._ensure_pdf_viewer_trusted("viewer {url}", MagicMock()) is True
        app._is_pdf_viewer_trusted.return_value = False
        assert app._ensure_pdf_viewer_trusted("viewer {url}", MagicMock()) is False

        with patch(
            "arxiv_browser.actions.llm_actions.get_config_path",
            return_value=tmp_path / "config.json",
        ):
            app._config.llm_command = ""
            app._config.llm_preset = ""
            assert app._require_llm_command() is None
        assert "Set llm_command or llm_preset" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.llm_preset = "missing"
        assert app._require_llm_command() is None
        assert "Unknown preset" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.llm_command = "echo {prompt} | cat"
        app._config.llm_preset = ""
        app._config.allow_llm_shell_fallback = False
        with patch(
            "arxiv_browser.actions.llm_actions.llm_command_requires_shell", return_value=True
        ):
            assert app._require_llm_command() is None
        assert "allow_llm_shell_fallback is disabled" in app.notify.call_args[0][0]

        app._config.llm_command = "echo {prompt}"
        app._config.allow_llm_shell_fallback = True
        with patch("arxiv_browser.actions.llm_actions.resolve_provider", return_value="provider"):
            assert app._require_llm_command() == "echo {prompt}"
        assert app._llm_provider == "provider"

    @pytest.mark.asyncio
    async def test_summary_and_chat_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50001")
        app._config = UserConfig(
            llm_command="cmd {prompt}",
            llm_timeout=5,
            llm_prompt_template="",
            research_interests="",
        )
        app._llm_provider = object()
        app._summary_db_path = tmp_path / "summary.db"
        app._summary_loading = set()
        app._paper_summaries = {}
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._relevance_scoring_active = False
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._get_current_paper = MagicMock(return_value=paper)
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._update_abstract_display = MagicMock()
        app._update_footer = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()

        app._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        app._start_summary_flow = MagicMock()
        llm_actions.action_generate_summary(app)
        app._start_summary_flow.assert_called_once_with("cmd {prompt}")

        app._start_summary_flow.reset_mock()
        app._ensure_llm_command_trusted = MagicMock(return_value=False)
        llm_actions.action_generate_summary(app)
        app._start_summary_flow.assert_not_called()

        app._get_current_paper = MagicMock(return_value=None)
        llm_actions._start_summary_flow(app, "cmd {prompt}")
        assert "No paper selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._get_current_paper = MagicMock(return_value=paper)
        app._summary_loading = {paper.arxiv_id}
        llm_actions._start_summary_flow(app, "cmd {prompt}")
        assert "Summary already generating" in app.notify.call_args[0][0]

        app._summary_loading = set()
        app.push_screen = MagicMock()
        llm_actions._start_summary_flow(app, "cmd {prompt}")
        app.notify.reset_mock()
        with patch("arxiv_browser.actions.llm_actions._load_summary", return_value="cached"):
            llm_actions._on_summary_mode_selected(app, "default", paper, "cmd {prompt}")
        assert "Summary loaded from cache" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with patch("arxiv_browser.actions.llm_actions._load_summary", return_value=None):
            llm_actions._on_summary_mode_selected(app, "quick", paper, "cmd {prompt}")
        assert paper.arxiv_id in app._summary_loading
        assert app._track_dataset_task.called

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._services = SimpleNamespace(
            llm=SimpleNamespace(generate_summary=AsyncMock(return_value=(None, "bad summary")))
        )
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "bad summary" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(return_value=("summary", None))
        with patch(
            "arxiv_browser.actions.llm_actions.asyncio.to_thread", new=AsyncMock(return_value=None)
        ):
            await llm_actions._generate_summary_async(
                app,
                paper,
                "prompt",
                "cmd-hash",
                mode_label="Q",
                use_full_paper_content=False,
            )
        assert app._paper_summaries[paper.arxiv_id] == "summary"
        assert "Summary generated" in app.notify.call_args[0][0]

        app._start_chat_with_paper = MagicMock()
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        llm_actions.action_chat_with_paper(app)
        app._start_chat_with_paper.assert_called_once()

        app._start_chat_with_paper = llm_actions._start_chat_with_paper.__get__(app, ArxivBrowser)
        app._get_current_paper = MagicMock(return_value=None)
        llm_actions._start_chat_with_paper(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._get_current_paper = MagicMock(return_value=paper)
        app._llm_provider = None
        llm_actions._start_chat_with_paper(app)

        app._llm_provider = object()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        llm_actions._start_chat_with_paper(app)
        assert app._track_dataset_task.called

        app._fetch_paper_content_async = AsyncMock(return_value="content")
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app.push_screen = MagicMock()
        await llm_actions._open_chat_screen(app, paper, object())
        assert app.push_screen.called

        app.push_screen.reset_mock()
        app._capture_dataset_epoch = MagicMock(return_value=2)
        app._is_current_dataset_epoch = MagicMock(return_value=False)
        await llm_actions._open_chat_screen(app, paper, object())
        assert not app.push_screen.called

        app._summary_loading = set()
        app._summary_command_hash = {paper.arxiv_id: "stale"}
        app._paper_summaries[paper.arxiv_id] = "stale"
        app._config.llm_prompt_template = "custom prompt"
        with patch("arxiv_browser.actions.llm_actions._load_summary", return_value=None):
            llm_actions._on_summary_mode_selected(app, "default", paper, "cmd {prompt}")
        assert paper.arxiv_id in app._summary_loading
        assert app._paper_summaries.get(paper.arxiv_id) != "stale"
        assert app._summary_mode_label[paper.arxiv_id] == ""

        app.push_screen.reset_mock()
        llm_actions._on_summary_mode_selected(app, None, paper, "cmd {prompt}")
        assert not app.push_screen.called

        app.notify.reset_mock()
        llm_actions._on_summary_mode_selected(app, "bogus", paper, "cmd {prompt}")
        assert "Unknown summary mode" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._llm_provider = None
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert not app.notify.called

        app._llm_provider = object()
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._services.llm.generate_summary = AsyncMock(return_value=(None, None))
        app.notify.reset_mock()
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "LLM command failed" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(side_effect=ValueError("bad prompt"))
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "bad prompt" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(side_effect=RuntimeError("boom"))
        await llm_actions._generate_summary_async(
            app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )
        assert "Summary failed" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._services.llm.generate_summary = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            await llm_actions._generate_summary_async(
                app,
                paper,
                "prompt",
                "cmd-hash",
                mode_label="Q",
                use_full_paper_content=False,
            )

    @pytest.mark.asyncio
    async def test_relevance_and_auto_tag_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50011")
        other = make_paper(arxiv_id="2401.50012")
        app._config = UserConfig(
            llm_command="echo {prompt}",
            llm_timeout=8,
            llm_prompt_template="",
            research_interests="",
        )
        app._llm_provider = object()
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._paper_summaries = {}
        app._summary_loading = set()
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._summary_db_path = tmp_path / "summary.db"
        app.all_papers = [paper, other]
        app.selected_ids = {paper.arxiv_id, other.arxiv_id}
        app._papers_by_id = {paper.arxiv_id: paper, other.arxiv_id: other}
        app._get_current_paper = MagicMock(return_value=paper)
        app._collect_all_tags = MagicMock(return_value=["existing"])
        app._tags_for = MagicMock(return_value=["existing"])
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    generate_summary=AsyncMock(return_value=("summary", None)),
                    score_relevance_once=AsyncMock(return_value=(7, "reason")),
                    suggest_tags_once=AsyncMock(return_value=["topic:new"]),
                )
            )
        )
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._update_footer = MagicMock()
        app._update_abstract_display = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._save_config_or_warn = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)

        app._require_llm_command = MagicMock(return_value="echo {prompt}")
        app._ensure_llm_command_trusted = MagicMock(return_value=True)
        app._start_score_relevance_flow = MagicMock()
        llm_actions.action_score_relevance(app)
        app._start_score_relevance_flow.assert_called_once()

        app._start_auto_tag_flow = MagicMock()
        llm_actions.action_auto_tag(app)
        app._start_auto_tag_flow.assert_called_once()

        app._relevance_scoring_active = True
        llm_actions._start_score_relevance_flow(app, "echo {prompt}")
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._relevance_scoring_active = False
        app._config.research_interests = ""
        app.push_screen = MagicMock()
        llm_actions._start_score_relevance_flow(app, "echo {prompt}")
        assert app.push_screen.called
        callback = app.push_screen.call_args.args[1]
        callback(None)
        callback("new interests")

        app.notify.reset_mock()
        llm_actions._on_interests_saved_then_score(app, None, "echo {prompt}")
        app._relevance_scoring_active = True
        llm_actions._on_interests_saved_then_score(app, "saved", "echo {prompt}")
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._relevance_scoring_active = False
        app._start_relevance_scoring = MagicMock()
        llm_actions._on_interests_saved_then_score(app, "saved", "echo {prompt}")
        app._start_relevance_scoring.assert_called_once()

        app._config.research_interests = "saved"
        llm_actions._on_interests_edited(app, "saved")
        app._relevance_scores = {"x": (1, "reason")}
        llm_actions._on_interests_edited(app, "new interests")
        assert not app._relevance_scores
        assert app._mark_badges_dirty.called

        async def _run_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch(
                "arxiv_browser.actions.llm_actions._load_all_relevance_scores",
                return_value={
                    paper.arxiv_id: (5, "cached"),
                    other.arxiv_id: (4, "cached"),
                },
            ),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper, other],
                "echo {prompt}",
                "interest",
            )
        assert "already scored" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._llm_provider = None
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.actions.llm_actions._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )

        app._llm_provider = object()
        app._cancel_batch_requested = True
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.actions.llm_actions._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )
        assert "cancelled" in app.notify.call_args[0][0].lower()

        app._cancel_batch_requested = False
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    score_relevance_once=AsyncMock(side_effect=RuntimeError("boom"))
                )
            )
        )
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.actions.llm_actions._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )
        assert "failed" in app.notify.call_args[0][0].lower()

        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(score_relevance_once=AsyncMock(side_effect=Exception("boom")))
            )
        )
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch("arxiv_browser.actions.llm_actions._load_all_relevance_scores", return_value={}),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )

        app.notify.reset_mock()
        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=Exception("boom")),
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                [paper],
                "echo {prompt}",
                "interest",
            )
        app.notify.assert_not_called()

        app._get_or_create_metadata = MagicMock(
            return_value=SimpleNamespace(tags=["existing", "topic:new"])
        )
        taxonomy = ["existing"]
        llm_actions._apply_auto_tag_batch_result(
            app,
            paper=paper,
            suggested=["topic:new", "topic:extra"],
            taxonomy=taxonomy,
        )
        assert "topic:extra" in taxonomy

        app._cancel_batch_requested = False
        assert llm_actions._maybe_cancel_auto_tag_batch(app, index=1, total=2, tagged=0) is False
        app._cancel_batch_requested = True
        assert llm_actions._maybe_cancel_auto_tag_batch(app, index=2, total=3, tagged=1) is True
        assert app._save_config_or_warn.called

        assert llm_actions._auto_tag_failure_message(0) == "Auto-tagging failed"
        assert (
            llm_actions._auto_tag_failure_message(2)
            == "Auto-tagging failed (2 tagged before error)"
        )

        app._auto_tag_active = True
        llm_actions._start_auto_tag_flow(app)
        assert "already in progress" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._auto_tag_active = False
        app.selected_ids = {paper.arxiv_id}
        app.all_papers = [paper]
        llm_actions._start_auto_tag_flow(app)
        assert app._track_dataset_task.called

        app.notify.reset_mock()
        app._auto_tag_active = False
        app.selected_ids = {"missing"}
        app.all_papers = [paper]
        llm_actions._start_auto_tag_flow(app)
        assert "No selected papers found" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=None)
        llm_actions._start_auto_tag_flow(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app._get_current_paper = MagicMock(return_value=paper)
        app._tags_for = MagicMock(return_value=["existing"])
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        llm_actions._start_auto_tag_flow(app)
        assert app._track_dataset_task.called

        app.notify.reset_mock()
        app._call_auto_tag_llm = AsyncMock(return_value=None)
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])
        assert "Auto-tagging failed" in app.notify.call_args[0][0]

        app._call_auto_tag_llm = AsyncMock(return_value=["topic:new"])
        app.push_screen = MagicMock()
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])
        assert app.push_screen.called
        callback = app.push_screen.call_args.args[1]
        callback(["topic:accepted"])

        app._call_auto_tag_llm = AsyncMock(side_effect=RuntimeError("boom"))
        await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])
        app._call_auto_tag_llm = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            await llm_actions._auto_tag_single_async(app, paper, ["existing"], ["existing"])

        app._cancel_batch_requested = False
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:a"], None])
        app._save_config_or_warn = MagicMock()
        await llm_actions._auto_tag_batch_async(app, [paper, other], ["existing"])
        assert "failed" in app.notify.call_args[0][0].lower()

        app._llm_provider = None
        assert await ArxivBrowser._call_auto_tag_llm(app, paper, ["existing"]) is None

        app._llm_provider = object()
        app.notify.reset_mock()
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    suggest_tags_once=AsyncMock(return_value=None),
                )
            )
        )
        assert await ArxivBrowser._call_auto_tag_llm(app, paper, ["existing"]) is None
        assert "Could not parse LLM response" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_relevance_and_auto_tag_success_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        papers = [
            make_paper(arxiv_id="2401.50101"),
            make_paper(arxiv_id="2401.50102"),
            make_paper(arxiv_id="2401.50103"),
            make_paper(arxiv_id="2401.50104"),
            make_paper(arxiv_id="2401.50105"),
            make_paper(arxiv_id="2401.50106"),
        ]
        app._config = UserConfig(
            llm_command="echo {prompt}",
            llm_timeout=8,
            llm_prompt_template="",
            research_interests="interest",
        )
        app._llm_provider = object()
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._cancel_batch_requested = False
        app._papers_by_id = {paper.arxiv_id: paper for paper in papers}
        app.all_papers = papers
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_footer = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._save_config_or_warn = MagicMock()
        app.notify = MagicMock()
        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    score_relevance_once=AsyncMock(
                        side_effect=[
                            (10, "a"),
                            None,
                            (8, "b"),
                            (7, "c"),
                            (6, "d"),
                        ]
                    )
                )
            )
        )

        async def _run_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=_run_to_thread),
            ),
            patch(
                "arxiv_browser.actions.llm_actions._load_all_relevance_scores",
                return_value={papers[0].arxiv_id: (5, "cached")},
            ),
        ):
            await llm_actions._score_relevance_batch_async(
                app,
                papers,
                "echo {prompt}",
                "interest",
            )

        assert any("Scoring relevance 5/" in call.args[0] for call in app.notify.call_args_list)
        assert any("cached" in call.args[0].lower() for call in app.notify.call_args_list)
        assert app._update_relevance_badge.called

        app.notify.reset_mock()
        app._config.paper_metadata = {}
        app._get_or_create_metadata = MagicMock(
            side_effect=lambda aid: app._config.paper_metadata.setdefault(
                aid, PaperMetadata(arxiv_id=aid, tags=["existing"])
            )
        )
        app._call_auto_tag_llm = AsyncMock(side_effect=[["topic:new"], None, ["topic:extra"]])
        taxonomy = ["existing"]
        await llm_actions._auto_tag_batch_async(app, papers[:3], taxonomy)
        assert "Auto-tagged 2 papers (1 failed)" in app.notify.call_args[0][0]
        assert "topic:new" in app._config.paper_metadata[papers[0].arxiv_id].tags
        assert "topic:extra" in taxonomy

    def test_update_relevance_badge_skips_detail_refresh_for_non_current_paper(
        self, make_paper
    ) -> None:
        app = _new_app_stub()
        current = make_paper(arxiv_id="2401.50141")
        other = make_paper(arxiv_id="2401.50142")
        app._get_current_paper = MagicMock(return_value=other)
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()

        llm_actions._update_relevance_badge(app, current.arxiv_id)

        app._mark_badges_dirty.assert_called_once_with("relevance")
        app._refresh_detail_pane.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_cancel_stale_epoch_and_auto_tag_cleanup_edges(
        self, make_paper, tmp_path
    ) -> None:
        paper = make_paper(arxiv_id="2401.50151")
        other = make_paper(arxiv_id="2401.50152")

        summary_app = _new_app_stub()
        summary_app._config = UserConfig(llm_timeout=8)
        summary_app._llm_provider = object()
        summary_app._summary_db_path = tmp_path / "summary.db"
        summary_app._summary_loading = {paper.arxiv_id}
        summary_app._paper_summaries = {}
        summary_app._summary_mode_label = {paper.arxiv_id: "Q"}
        summary_app._summary_command_hash = {paper.arxiv_id: "old"}
        summary_app._update_abstract_display = MagicMock()
        summary_app._fetch_paper_content_async = AsyncMock(return_value="content")
        summary_app.notify = MagicMock()
        summary_app._capture_dataset_epoch = MagicMock(return_value=1)
        summary_app._is_current_dataset_epoch = MagicMock(return_value=True)
        summary_app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(
                    generate_summary=AsyncMock(side_effect=asyncio.CancelledError())
                )
            )
        )

        with pytest.raises(asyncio.CancelledError):
            await llm_actions._generate_summary_async(
                summary_app,
                paper,
                "prompt",
                "cmd-hash",
                mode_label="Q",
                use_full_paper_content=False,
            )

        assert paper.arxiv_id not in summary_app._summary_loading
        assert paper.arxiv_id not in summary_app._summary_mode_label
        assert paper.arxiv_id not in summary_app._summary_command_hash
        summary_app._update_abstract_display.assert_called_once_with(paper.arxiv_id)

        stale_summary_app = _new_app_stub()
        stale_summary_app._config = UserConfig(llm_timeout=8)
        stale_summary_app._llm_provider = object()
        stale_summary_app._summary_db_path = tmp_path / "stale-summary.db"
        stale_summary_app._summary_loading = {paper.arxiv_id}
        stale_summary_app._paper_summaries = {}
        stale_summary_app._summary_mode_label = {}
        stale_summary_app._summary_command_hash = {}
        stale_summary_app._update_abstract_display = MagicMock()
        stale_summary_app._fetch_paper_content_async = AsyncMock(return_value="content")
        stale_summary_app.notify = MagicMock()
        stale_summary_app._capture_dataset_epoch = MagicMock(return_value=1)
        stale_summary_app._is_current_dataset_epoch = MagicMock(return_value=False)
        stale_summary_app._get_services = MagicMock(
            return_value=SimpleNamespace(
                llm=SimpleNamespace(generate_summary=AsyncMock(return_value=("summary", None)))
            )
        )

        await llm_actions._generate_summary_async(
            stale_summary_app,
            paper,
            "prompt",
            "cmd-hash",
            mode_label="Q",
            use_full_paper_content=False,
        )

        assert stale_summary_app._paper_summaries == {}
        assert paper.arxiv_id in stale_summary_app._summary_loading
        stale_summary_app.notify.assert_not_called()
        stale_summary_app._update_abstract_display.assert_not_called()

        stale_relevance_app = _new_app_stub()
        stale_relevance_app._config = UserConfig(llm_timeout=8)
        stale_relevance_app._llm_provider = object()
        stale_relevance_app._relevance_db_path = tmp_path / "relevance.db"
        stale_relevance_app._relevance_scores = {}
        stale_relevance_app._relevance_scoring_active = True
        stale_relevance_app._scoring_progress = (1, 1)
        stale_relevance_app._cancel_batch_requested = True
        stale_relevance_app.notify = MagicMock()
        stale_relevance_app._update_footer = MagicMock()
        stale_relevance_app._mark_badges_dirty = MagicMock()
        stale_relevance_app._refresh_detail_pane = MagicMock()
        stale_relevance_app._capture_dataset_epoch = MagicMock(return_value=1)
        stale_relevance_app._is_current_dataset_epoch = MagicMock(return_value=False)

        with patch(
            "arxiv_browser.actions.llm_actions.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await llm_actions._score_relevance_batch_async(
                stale_relevance_app,
                [paper],
                "echo {prompt}",
                "interest",
            )

        stale_relevance_app.notify.assert_not_called()
        assert stale_relevance_app._relevance_scoring_active is True
        assert stale_relevance_app._cancel_batch_requested is True
        stale_relevance_app._update_footer.assert_not_called()

        unexpected_relevance_app = _new_app_stub()
        unexpected_relevance_app._config = UserConfig(llm_timeout=8)
        unexpected_relevance_app._llm_provider = object()
        unexpected_relevance_app._relevance_db_path = tmp_path / "unexpected-relevance.db"
        unexpected_relevance_app._relevance_scores = {}
        unexpected_relevance_app._relevance_scoring_active = True
        unexpected_relevance_app._scoring_progress = (1, 1)
        unexpected_relevance_app._cancel_batch_requested = True
        unexpected_relevance_app.notify = MagicMock()
        unexpected_relevance_app._update_footer = MagicMock()
        unexpected_relevance_app._mark_badges_dirty = MagicMock()
        unexpected_relevance_app._refresh_detail_pane = MagicMock()
        unexpected_relevance_app._capture_dataset_epoch = MagicMock(return_value=1)
        unexpected_relevance_app._is_current_dataset_epoch = MagicMock(return_value=True)

        with (
            patch(
                "arxiv_browser.actions.llm_actions.asyncio.to_thread",
                new=AsyncMock(side_effect=Exception("boom")),
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await llm_actions._score_relevance_batch_async(
                unexpected_relevance_app,
                [paper],
                "echo {prompt}",
                "interest",
            )

        unexpected_relevance_app.notify.assert_not_called()
        assert unexpected_relevance_app._relevance_scoring_active is False
        assert unexpected_relevance_app._scoring_progress is None
        assert unexpected_relevance_app._cancel_batch_requested is False

        stale_single_app = _new_app_stub()
        stale_single_app._auto_tag_active = True
        stale_single_app.notify = MagicMock()
        stale_single_app.push_screen = MagicMock()
        stale_single_app._update_footer = MagicMock()
        stale_single_app._capture_dataset_epoch = MagicMock(return_value=1)
        stale_single_app._is_current_dataset_epoch = MagicMock(return_value=False)
        stale_single_app._call_auto_tag_llm = AsyncMock(return_value=["topic:new"])

        await llm_actions._auto_tag_single_async(stale_single_app, paper, ["existing"], ["old"])

        stale_single_app.notify.assert_not_called()
        stale_single_app.push_screen.assert_not_called()
        assert stale_single_app._auto_tag_active is True
        stale_single_app._update_footer.assert_not_called()

        partial_batch_app = _new_app_stub()
        partial_batch_app._config = UserConfig(paper_metadata={})
        partial_batch_app._auto_tag_active = True
        partial_batch_app._auto_tag_progress = (0, 2)
        partial_batch_app._cancel_batch_requested = False
        partial_batch_app.notify = MagicMock()
        partial_batch_app._save_config_or_warn = MagicMock()
        partial_batch_app._mark_badges_dirty = MagicMock()
        partial_batch_app._refresh_detail_pane = MagicMock()
        partial_batch_app._update_footer = MagicMock()
        partial_batch_app._capture_dataset_epoch = MagicMock(return_value=1)
        partial_batch_app._is_current_dataset_epoch = MagicMock(return_value=True)
        partial_batch_app._get_or_create_metadata = MagicMock(
            side_effect=lambda aid: partial_batch_app._config.paper_metadata.setdefault(
                aid,
                PaperMetadata(arxiv_id=aid, tags=[]),
            )
        )
        partial_batch_app._call_auto_tag_llm = AsyncMock(
            side_effect=[["topic:new"], RuntimeError("boom")]
        )

        await llm_actions._auto_tag_batch_async(partial_batch_app, [paper, other], ["existing"])

        assert any(
            call.args == ("partial auto-tag results",)
            for call in partial_batch_app._save_config_or_warn.call_args_list
        )
        assert all(
            call.args != ("auto-tag results",)
            for call in partial_batch_app._save_config_or_warn.call_args_list
        )
        assert "1 tagged before error" in partial_batch_app.notify.call_args[0][0]
        assert partial_batch_app._auto_tag_active is False
        assert partial_batch_app._auto_tag_progress is None

        zero_batch_app = _new_app_stub()
        zero_batch_app._auto_tag_active = True
        zero_batch_app._auto_tag_progress = (0, 1)
        zero_batch_app.notify = MagicMock()
        zero_batch_app._save_config_or_warn = MagicMock()
        zero_batch_app._mark_badges_dirty = MagicMock()
        zero_batch_app._refresh_detail_pane = MagicMock()
        zero_batch_app._update_footer = MagicMock()
        zero_batch_app._capture_dataset_epoch = MagicMock(return_value=1)
        zero_batch_app._is_current_dataset_epoch = MagicMock(return_value=True)
        zero_batch_app._call_auto_tag_llm = AsyncMock(side_effect=Exception("boom"))

        with pytest.raises(Exception, match="boom"):
            await llm_actions._auto_tag_batch_async(zero_batch_app, [paper], ["existing"])

        zero_batch_app._save_config_or_warn.assert_not_called()
        zero_batch_app.notify.assert_not_called()

        cancelled_batch_app = _new_app_stub()
        cancelled_batch_app._auto_tag_active = True
        cancelled_batch_app._auto_tag_progress = (0, 1)
        cancelled_batch_app._cancel_batch_requested = False
        cancelled_batch_app.notify = MagicMock()
        cancelled_batch_app._update_footer = MagicMock()
        cancelled_batch_app._capture_dataset_epoch = MagicMock(return_value=1)
        cancelled_batch_app._is_current_dataset_epoch = MagicMock(return_value=True)
        cancelled_batch_app._call_auto_tag_llm = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await llm_actions._auto_tag_batch_async(cancelled_batch_app, [paper], ["existing"])

        assert cancelled_batch_app._auto_tag_active is False
        assert cancelled_batch_app._auto_tag_progress is None
        assert cancelled_batch_app._cancel_batch_requested is False
