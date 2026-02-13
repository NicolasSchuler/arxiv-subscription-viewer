#!/usr/bin/env python3
"""High-impact coverage tests for action-heavy paths in app.py."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arxiv_browser.app import (
    ArxivBrowser,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
    _resolve_legacy_fallback,
    _resolve_papers,
)
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.semantic_scholar import CitationEntry, SemanticScholarPaper


def _new_app() -> ArxivBrowser:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._http_client = None
    return app


def _make_s2_paper(arxiv_id: str) -> SemanticScholarPaper:
    return SemanticScholarPaper(
        arxiv_id=arxiv_id,
        s2_paper_id=f"s2:{arxiv_id}",
        citation_count=10,
        influential_citation_count=1,
        tldr="",
        fields_of_study=(),
        year=2024,
        url=f"https://api.semanticscholar.org/{arxiv_id}",
    )


def _make_hf_paper(arxiv_id: str) -> HuggingFacePaper:
    return HuggingFacePaper(
        arxiv_id=arxiv_id,
        title=f"HF {arxiv_id}",
        upvotes=5,
        num_comments=1,
        ai_summary="",
        ai_keywords=(),
        github_repo="",
        github_stars=0,
    )


class TestIoActionHelpers:
    def test_resolve_target_papers_preserves_order_and_includes_hidden(self, make_paper):
        from arxiv_browser.io_actions import resolve_target_papers

        visible = [
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00001"),
        ]
        hidden = make_paper(arxiv_id="2401.99999")

        result = resolve_target_papers(
            filtered_papers=visible,
            selected_ids={"2401.99999", "2401.00001"},
            papers_by_id={p.arxiv_id: p for p in [*visible, hidden]},
            current_paper=None,
        )

        assert [p.arxiv_id for p in result] == ["2401.00001", "2401.99999"]

    def test_filter_papers_needing_download_splits_pending_and_skipped(self, make_paper, tmp_path):
        from arxiv_browser.io_actions import filter_papers_needing_download

        existing_paper = make_paper(arxiv_id="2401.01001")
        pending_paper = make_paper(arxiv_id="2401.01002")
        existing_path = tmp_path / "exists.pdf"
        existing_path.write_bytes(b"ready")
        pending_path = tmp_path / "missing.pdf"

        path_map = {
            existing_paper.arxiv_id: existing_path,
            pending_paper.arxiv_id: pending_path,
        }
        to_download, skipped = filter_papers_needing_download(
            [existing_paper, pending_paper],
            lambda paper: path_map[paper.arxiv_id],
        )

        assert [p.arxiv_id for p in to_download] == [pending_paper.arxiv_id]
        assert skipped == [existing_paper.arxiv_id]

    def test_build_markdown_export_document_renders_expected_structure(self):
        from arxiv_browser.io_actions import build_markdown_export_document

        markdown = build_markdown_export_document(["## Paper A", "## Paper B"])
        assert markdown.startswith("# arXiv Papers Export")
        assert "*Exported 2 paper(s)*" in markdown
        assert markdown.count("\n---\n") == 2

    def test_write_timestamped_export_file_writes_atomically(self, tmp_path):
        from arxiv_browser.io_actions import write_timestamped_export_file

        export_dir = tmp_path / "exports"
        out = write_timestamped_export_file(
            content="hello",
            export_dir=export_dir,
            extension="txt",
            now=datetime(2026, 2, 13, 20, 0, 0),
        )
        assert out.name == "arxiv-2026-02-13_200000.txt"
        assert out.read_text(encoding="utf-8") == "hello"
        assert list(export_dir.glob(".txt-*.tmp")) == []


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
        with patch("arxiv_browser.app._load_all_relevance_scores", return_value=cached_scores):
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
        app._relevance_db_path = tmp_path / "relevance.db"
        app._relevance_scores = {}
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._update_relevance_badge = MagicMock()
        app._update_footer = MagicMock()
        app.notify = MagicMock()
        app._relevance_scoring_active = True
        app._scoring_progress = None
        app._llm_provider = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(success=True, output="ok-1", error=""),
                    SimpleNamespace(success=True, output="bad-parse", error=""),
                    SimpleNamespace(success=False, output="", error="timeout"),
                    RuntimeError("provider crash"),
                    SimpleNamespace(success=True, output="ok-2", error=""),
                ]
            )
        )

        with (
            patch("arxiv_browser.app._load_all_relevance_scores", return_value={}),
            patch(
                "arxiv_browser.app._parse_relevance_response",
                side_effect=[(9, "great match"), None, (6, "partial match")],
            ),
            patch("arxiv_browser.app._save_relevance_score", return_value=None),
        ):
            await app._score_relevance_batch_async(papers, "cmd {prompt}", "relevance interests")

        assert set(app._relevance_scores) == {"2401.00001", "2401.00005"}
        assert app._relevance_scores["2401.00001"][0] == 9
        assert app._relevance_scores["2401.00005"][0] == 6
        assert app._relevance_scoring_active is False
        assert app._scoring_progress is None
        assert app._update_relevance_badge.call_count == 2
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
                "arxiv_browser.app.parse_arxiv_version_map",
                side_effect=[
                    {"2401.00001": 3, "2401.00002": 1},
                    {"2401.00003": 9},
                ],
            ),
            patch("arxiv_browser.app.save_config", return_value=True),
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
        app._http_client.get.side_effect = [ok_response, RuntimeError("network down")]

        with (
            patch(
                "arxiv_browser.app.parse_arxiv_version_map",
                side_effect=[{"2401.10001": 4}],
            ),
            patch("arxiv_browser.app.save_config", return_value=True),
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

        with patch("arxiv_browser.app.save_config", return_value=True):
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

        with patch("arxiv_browser.app.save_config", return_value=True) as save:
            await app._auto_tag_batch_async(papers, taxonomy=[])

        save.assert_called_once()
        assert app._config.paper_metadata["2401.21001"].tags == ["topic:ml"]
        assert app._auto_tag_active is False
        assert app._auto_tag_progress is None
        assert "failed" in app.notify.call_args[0][0].lower()

    def test_action_auto_tag_selected_and_single_branches(self, make_paper):
        app = _new_app()
        app._require_llm_command = MagicMock(return_value="cmd {prompt}")
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

        app._auto_tag_active = False
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=paper)
        app._tags_for = MagicMock(return_value=["old"])
        app.action_auto_tag()
        assert app._track_task.call_count == 2

        app._auto_tag_active = False
        app._get_current_paper = MagicMock(return_value=None)
        app.action_auto_tag()
        assert app._auto_tag_active is False
        assert "No paper selected" in app.notify.call_args[0][0]


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
        assert "No metadata files found" in app.notify.call_args[0][0]

    def test_action_import_metadata_success_and_save_warning(self, tmp_path):
        app = self._make_import_app(tmp_path)
        export_file = tmp_path / "arxiv-2026-02-13.json"
        export_file.write_text("{}", encoding="utf-8")

        with (
            patch("arxiv_browser.app.import_metadata", return_value=(2, 1, 1, 1)),
            patch("arxiv_browser.app.save_config", return_value=False),
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

    def test_action_import_metadata_handles_parse_error(self, tmp_path):
        app = self._make_import_app(tmp_path)
        export_file = tmp_path / "arxiv-2026-02-13.json"
        export_file.write_text("{not-json", encoding="utf-8")
        app.action_import_metadata()
        assert "Import failed" in app.notify.call_args[0][0]

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

        with patch("arxiv_browser.app.save_config", return_value=True) as save:
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


class TestS2AndHfCoverage:
    @pytest.mark.asyncio
    async def test_action_fetch_s2_disabled_and_cache_paths(self, make_paper, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app._config.s2_cache_ttl_days = 7
        app._config.s2_api_key = ""
        app.notify = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._refresh_current_list_item = MagicMock()
        app._track_task = MagicMock()
        app._s2_db_path = tmp_path / "s2.db"
        app._s2_loading = set()
        app._s2_cache = {}

        paper = make_paper(arxiv_id="2401.30001")
        app._get_current_paper = MagicMock(return_value=paper)

        app._s2_active = False
        await app.action_fetch_s2()
        assert "S2 is disabled" in app.notify.call_args[0][0]

        app._s2_active = True
        with patch("arxiv_browser.app.load_s2_paper", return_value=_make_s2_paper(paper.arxiv_id)):
            await app.action_fetch_s2()

        assert paper.arxiv_id in app._s2_cache
        app._track_task.assert_not_called()
        assert paper.arxiv_id not in app._s2_loading

    @pytest.mark.asyncio
    async def test_action_fetch_s2_handles_cache_errors_and_schedules_fetch(
        self, make_paper, tmp_path
    ):
        app = _new_app()
        app._config = UserConfig()
        app._config.s2_cache_ttl_days = 7
        app._config.s2_api_key = ""
        app.notify = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._refresh_current_list_item = MagicMock()
        app._s2_db_path = tmp_path / "s2.db"
        app._s2_loading = set()
        app._s2_cache = {}
        paper = make_paper(arxiv_id="2401.30002")
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_active = True

        with patch("arxiv_browser.app.load_s2_paper", side_effect=RuntimeError("db error")):
            await app.action_fetch_s2()
        assert "S2 fetch failed" in app.notify.call_args[0][0]
        assert paper.arxiv_id not in app._s2_loading

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)
        with patch("arxiv_browser.app.load_s2_paper", return_value=None):
            await app.action_fetch_s2()
        app._track_task.assert_called_once()
        assert paper.arxiv_id in app._s2_loading

    @pytest.mark.asyncio
    async def test_fetch_s2_paper_async_success_and_no_data(self, tmp_path):
        app = _new_app()
        app._http_client = AsyncMock()
        app._config = UserConfig()
        app._config.s2_api_key = ""
        app._s2_db_path = tmp_path / "s2.db"
        app._s2_cache = {}
        app._s2_loading = {"2401.30003"}
        app.notify = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._refresh_current_list_item = MagicMock()

        with (
            patch("arxiv_browser.app.fetch_s2_paper", return_value=_make_s2_paper("2401.30003")),
            patch("arxiv_browser.app.save_s2_paper", return_value=None),
        ):
            await app._fetch_s2_paper_async("2401.30003")

        assert "2401.30003" in app._s2_cache
        assert "2401.30003" not in app._s2_loading

        app._s2_loading.add("2401.30004")
        with patch("arxiv_browser.app.fetch_s2_paper", return_value=None):
            await app._fetch_s2_paper_async("2401.30004")
        assert "No S2 data found" in app.notify.call_args[0][0]
        assert "2401.30004" not in app._s2_loading

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_cache_hit_error_and_schedule(self, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app._config.hf_cache_ttl_hours = 6
        app._hf_cache = {}
        app._hf_loading = False
        app._hf_db_path = tmp_path / "hf.db"
        app._papers_by_id = {"2401.40001": object()}
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()
        app._track_task = MagicMock()

        hf = _make_hf_paper("2401.40001")
        with patch("arxiv_browser.app.load_hf_daily_cache", return_value={"2401.40001": hf}):
            await app._fetch_hf_daily()
        assert app._hf_cache["2401.40001"] is hf
        app._track_task.assert_not_called()

        app._hf_loading = False
        with patch("arxiv_browser.app.load_hf_daily_cache", side_effect=RuntimeError("db")):
            await app._fetch_hf_daily()
        assert "HF fetch failed" in app.notify.call_args[0][0]

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)
        app._hf_loading = False
        with patch("arxiv_browser.app.load_hf_daily_cache", return_value=None):
            await app._fetch_hf_daily()
        app._track_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_hf_daily_async_success_and_empty(self, tmp_path):
        app = _new_app()
        app._http_client = AsyncMock()
        app._config = UserConfig()
        app._hf_db_path = tmp_path / "hf.db"
        app._hf_cache = {}
        app._hf_loading = True
        app._papers_by_id = {"2401.40002": object()}
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app._mark_badges_dirty = MagicMock()

        hf = _make_hf_paper("2401.40002")
        with (
            patch("arxiv_browser.app.fetch_hf_daily_papers", return_value=[hf]),
            patch("arxiv_browser.app.save_hf_daily_cache", return_value=None),
        ):
            await app._fetch_hf_daily_async()
        assert app._hf_cache["2401.40002"] is hf
        assert app._hf_loading is False

        app._hf_loading = True
        with patch("arxiv_browser.app.fetch_hf_daily_papers", return_value=[]):
            await app._fetch_hf_daily_async()
        assert "No HF trending data found" in app.notify.call_args[0][0]
        assert app._hf_loading is False


class TestDownloadClipboardAndOpenCoverage:
    @pytest.mark.asyncio
    async def test_download_pdf_async_success_and_failure(self, make_paper, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        paper = make_paper(arxiv_id="2401.50001")
        target = tmp_path / "pdfs" / "2401.50001.pdf"
        response = MagicMock()
        response.content = b"%PDF-1.4"
        response.raise_for_status = MagicMock()
        client = SimpleNamespace(get=AsyncMock(return_value=response))

        with (
            patch("arxiv_browser.app.get_pdf_url", return_value="https://example/pdf"),
            patch("arxiv_browser.app.get_pdf_download_path", return_value=target),
        ):
            ok = await app._download_pdf_async(paper, client)

        assert ok is True
        assert target.exists()
        assert target.read_bytes() == b"%PDF-1.4"

        with (
            patch("arxiv_browser.app.get_pdf_url", return_value="https://example/pdf"),
            patch("arxiv_browser.app.get_pdf_download_path", return_value=target),
        ):
            client.get = AsyncMock(side_effect=OSError("network"))
            ok = await app._download_pdf_async(paper, client)
        assert ok is False

    @pytest.mark.asyncio
    async def test_process_single_download_updates_state_and_finishes(self, make_paper):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.50002")
        app._download_pdf_async = AsyncMock(return_value=False)
        app._download_results = {}
        app._download_total = 1
        app._downloading = {paper.arxiv_id}
        app._update_download_progress = MagicMock()
        app._start_downloads = MagicMock()
        app._finish_download_batch = MagicMock()

        await app._process_single_download(paper)

        assert app._download_results[paper.arxiv_id] is False
        assert paper.arxiv_id not in app._downloading
        app._update_download_progress.assert_called_once_with(1, 1)
        app._start_downloads.assert_called_once()
        app._finish_download_batch.assert_called_once()

    def test_finish_download_batch_notifies_and_resets_state(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.pdf_download_dir = None
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._download_results = {"a": True, "b": False}
        app._download_total = 2

        app._finish_download_batch()

        assert app._download_total == 0
        assert app._download_results == {}
        app.notify.assert_called_once()
        assert "Downloaded 1/2 PDFs (1 failed)" in app.notify.call_args[0][0]

    def test_action_download_pdf_filters_existing_and_starts(self, make_paper, tmp_path):
        app = _new_app()
        app._config = UserConfig()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._do_start_downloads = MagicMock()
        app._is_download_batch_active = MagicMock(return_value=False)

        paper_existing = make_paper(arxiv_id="2401.50003")
        paper_new = make_paper(arxiv_id="2401.50004")
        app._get_target_papers = MagicMock(return_value=[paper_existing, paper_new])

        existing = tmp_path / "existing.pdf"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"ready")
        missing = tmp_path / "missing.pdf"

        with patch(
            "arxiv_browser.app.get_pdf_download_path",
            side_effect=[existing, missing],
        ):
            app.action_download_pdf()

        app._do_start_downloads.assert_called_once_with([paper_new])

    def test_action_download_pdf_handles_no_target(self):
        app = _new_app()
        app.notify = MagicMock()
        app._is_download_batch_active = MagicMock(return_value=False)
        app._get_target_papers = MagicMock(return_value=[])
        app.action_download_pdf()
        assert "No papers to download" in app.notify.call_args[0][0]

    def test_copy_to_clipboard_platform_paths(self):
        app = _new_app()

        with (
            patch("arxiv_browser.app.platform.system", return_value="Darwin"),
            patch("arxiv_browser.app.subprocess.run", return_value=None) as run,
        ):
            assert app._copy_to_clipboard("abc") is True
            assert run.call_count == 1

        with (
            patch("arxiv_browser.app.platform.system", return_value="Linux"),
            patch(
                "arxiv_browser.app.subprocess.run",
                side_effect=[FileNotFoundError(), None],
            ) as run,
        ):
            assert app._copy_to_clipboard("abc") is True
            assert run.call_count == 2

        with (
            patch("arxiv_browser.app.platform.system", return_value="Plan9"),
            patch("arxiv_browser.app.subprocess.run", return_value=None),
        ):
            assert app._copy_to_clipboard("abc") is False

    def test_copy_to_clipboard_handles_subprocess_failure(self):
        import subprocess

        app = _new_app()
        with (
            patch("arxiv_browser.app.platform.system", return_value="Darwin"),
            patch(
                "arxiv_browser.app.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="pbcopy", timeout=1),
            ),
        ):
            assert app._copy_to_clipboard("abc") is False

    def test_open_with_viewer_and_open_pdf_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app._safe_browser_open = MagicMock(return_value=True)
        app._open_with_viewer = MagicMock(return_value=True)
        app._config = UserConfig()
        app._config.pdf_viewer = "viewer {url}"

        paper = make_paper(arxiv_id="2401.50005")
        app._do_open_pdfs([paper])
        app._open_with_viewer.assert_called_once()
        app._safe_browser_open.assert_not_called()

    def test_open_with_viewer_handles_errors(self):
        app = _new_app()
        app.notify = MagicMock()

        with patch("arxiv_browser.app.subprocess.Popen", side_effect=OSError("bad viewer")):
            ok = app._open_with_viewer("broken-viewer {url}", "https://arxiv.org/pdf/1")

        assert ok is False
        assert "Failed to open PDF viewer" in app.notify.call_args[0][0]

    def test_open_with_viewer_uses_subprocess_args_not_shell(self):
        app = _new_app()
        app.notify = MagicMock()

        with patch("arxiv_browser.app.subprocess.Popen") as popen:
            ok = app._open_with_viewer("open -a Skim {path}", "/tmp/my paper.pdf")

        assert ok is True
        popen.assert_called_once()
        call_args = popen.call_args
        assert call_args.args[0] == ["open", "-a", "Skim", "/tmp/my paper.pdf"]
        assert "shell" not in call_args.kwargs

    def test_action_open_url_uses_confirmation_for_large_batches(self, make_paper):
        app = _new_app()
        app._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id=f"2401.{i:05d}") for i in range(20)]
        )
        app.push_screen = MagicMock()
        app.action_open_url()
        app.push_screen.assert_called_once()

    def test_export_dispatch_and_markdown_export(self, make_paper):
        paper = make_paper(arxiv_id="2401.50006")
        app = _new_app()
        app.action_copy_selected = MagicMock()
        app.action_copy_bibtex = MagicMock()
        app.action_export_markdown = MagicMock()
        app._export_clipboard_ris = MagicMock()
        app._export_clipboard_csv = MagicMock()
        app._export_clipboard_mdtable = MagicMock()
        app.action_export_bibtex_file = MagicMock()
        app._export_file_ris = MagicMock()
        app._export_file_csv = MagicMock()
        app._do_export("clipboard-bibtex", [paper])
        app.action_copy_bibtex.assert_called_once()

        app._do_export("file-csv", [paper])
        app._export_file_csv.assert_called_once_with([paper])

        app2 = _new_app()
        app2._get_target_papers = MagicMock(return_value=[paper])
        app2._format_paper_as_markdown = MagicMock(return_value="## Test")
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.notify = MagicMock()
        app2.action_export_markdown()
        app2._copy_to_clipboard.assert_called_once()
        assert "as Markdown" in app2.notify.call_args[0][0]


class TestStatusCommandPaletteAndChatCoverage:
    def test_update_status_bar_renders_mode_badges(self):
        class DummyLabel:
            def __init__(self):
                self.content = ""

            def update(self, text):
                self.content = text

        label = DummyLabel()
        app = _new_app()
        app.query_one = MagicMock(return_value=label)
        app._update_footer = MagicMock()
        app._get_active_query = MagicMock(return_value="graph transformers")
        app.all_papers = [object(), object(), object()]
        app.filtered_papers = [object()]
        app.selected_ids = {"a", "b"}
        app._sort_index = 0
        app._watch_filter_active = False
        app._in_arxiv_api_mode = False
        app._arxiv_search_state = None
        app._arxiv_api_loading = False
        app._show_abstract_preview = True
        app._s2_active = True
        app._s2_loading = {"a"}
        app._s2_cache = {"a": object()}
        app._hf_active = True
        app._hf_loading = False
        app._hf_cache = {"a": _make_hf_paper("a")}
        app._papers_by_id = {"a": object()}
        app._version_checking = True
        app._version_updates = {"a": (1, 2)}

        app._update_status_bar()

        content = str(label.content)
        assert "S2 loading" in content
        assert "HF:1" in content
        assert "Checking versions" in content
        assert "Preview" in content
        app._update_footer.assert_called_once()

    def test_action_command_palette_dispatches_coroutines_and_errors(self):
        app = _new_app()
        app.notify = MagicMock()
        captured = {}
        app.push_screen = lambda _modal, cb: captured.setdefault("callback", cb)

        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)

        async def fake_async():
            return None

        def action_demo():
            return fake_async()

        def action_boom():
            raise RuntimeError("boom")

        app.action_demo = action_demo
        app.action_boom = action_boom

        app.action_command_palette()
        callback = captured["callback"]
        callback("demo")
        callback("boom")
        callback("missing")
        callback(None)

        app._track_task.assert_called_once()
        assert "Command failed: boom" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_chat_and_summary_action_paths(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app.notify = MagicMock()
        app._summary_loading = set()
        app._paper_summaries = {}
        app._summary_mode_label = {}
        app._summary_command_hash = {}
        app._summary_db_path = MagicMock()
        app._update_abstract_display = MagicMock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())

        paper = make_paper(arxiv_id="2401.60001")
        app._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app._get_current_paper = MagicMock(return_value=paper)
        app._llm_provider = MagicMock()
        app.push_screen = MagicMock()

        app.action_generate_summary()
        app.push_screen.assert_called_once()

        with patch("arxiv_browser.app._load_summary", return_value="cached summary"):
            app._on_summary_mode_selected("default", paper, "cmd {prompt}")
        assert app._paper_summaries[paper.arxiv_id] == "cached summary"

        app.action_chat_with_paper()
        app._track_task.assert_called()

        with patch("arxiv_browser.app._fetch_paper_content_async", return_value="full text"):
            await app._open_chat_screen(paper, app._llm_provider)
        assert app.push_screen.call_count >= 2


class TestArxivApiAndSimilarityCoverage:
    @pytest.mark.asyncio
    async def test_fetch_arxiv_api_page_with_and_without_shared_client(self, make_paper):
        request = SimpleNamespace(query="transformers", field="all", category="")

        app = _new_app()
        app._apply_arxiv_rate_limit = AsyncMock()
        response = MagicMock()
        response.text = "<feed/>"
        app._http_client = SimpleNamespace(get=AsyncMock(return_value=response))

        with (
            patch("arxiv_browser.app.build_arxiv_search_query", return_value="all:transformers"),
            patch("arxiv_browser.app.parse_arxiv_api_feed", return_value=[make_paper()]),
        ):
            papers = await app._fetch_arxiv_api_page(request, start=0, max_results=5)
        assert len(papers) == 1

        app2 = _new_app()
        app2._apply_arxiv_rate_limit = AsyncMock()
        app2._http_client = None
        response2 = MagicMock()
        response2.text = "<feed/>"

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *_args, **_kwargs):
                return response2

        with (
            patch("arxiv_browser.app.build_arxiv_search_query", return_value="all:transformers"),
            patch("arxiv_browser.app.parse_arxiv_api_feed", return_value=[make_paper()]),
            patch("arxiv_browser.app.httpx.AsyncClient", return_value=DummyClient()),
        ):
            papers2 = await app2._fetch_arxiv_api_page(request, start=0, max_results=5)
        assert len(papers2) == 1

    def test_show_similar_actions_dispatch_and_local_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._show_recommendations = MagicMock()
        app._get_current_paper = MagicMock(return_value=None)
        app.action_show_similar()
        assert "No paper selected" in app.notify.call_args[0][0]

        paper = make_paper(arxiv_id="2401.70001")
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_active = True
        app.action_show_similar()
        app.push_screen.assert_called_once()

        app._s2_active = False
        app.action_show_similar()
        app._show_recommendations.assert_called_with(paper, "local")

    def test_show_recommendations_dispatch(self, make_paper):
        app = _new_app()
        paper = make_paper(arxiv_id="2401.70002")
        app._show_local_recommendations = MagicMock()
        tracked = []

        def track_task(coro):
            tracked.append(coro)
            coro.close()

        app._track_task = MagicMock(side_effect=track_task)

        app._show_recommendations(paper, None)
        app._show_recommendations(paper, "local")
        app._show_recommendations(paper, "s2")

        app._show_local_recommendations.assert_called_once_with(paper)
        app._track_task.assert_called_once()

    def test_show_local_recommendations_empty_and_success(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app._tfidf_index = None
        app._get_abstract_text = MagicMock(return_value="abstract")
        paper = make_paper(arxiv_id="2401.70003")
        app.all_papers = [paper, make_paper(arxiv_id="2401.70004")]

        with (
            patch("arxiv_browser.app.TfidfIndex.build", return_value=object()),
            patch("arxiv_browser.app.find_similar_papers", return_value=[]),
        ):
            app._show_local_recommendations(paper)
        assert "No similar papers found" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with (
            patch("arxiv_browser.app.TfidfIndex.build", return_value=object()),
            patch(
                "arxiv_browser.app.find_similar_papers",
                return_value=[(make_paper(arxiv_id="2401.70005"), 0.9)],
            ),
        ):
            app._show_local_recommendations(paper)
        app.push_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_s2_recommendations_and_citation_graph_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        paper = make_paper(arxiv_id="2401.70006")

        app._fetch_s2_recommendations_async = AsyncMock(return_value=[])
        await app._show_s2_recommendations(paper)
        assert "No S2 recommendations found" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(return_value=[_make_s2_paper("2401.70007")])
        app._s2_recs_to_paper_tuples = MagicMock(
            return_value=[(make_paper(arxiv_id="2401.70007"), 0.8)]
        )
        await app._show_s2_recommendations(paper)
        app.push_screen.assert_called()

        app.notify.reset_mock()
        app._fetch_s2_recommendations_async = AsyncMock(side_effect=RuntimeError("s2 error"))
        await app._show_s2_recommendations(paper)
        assert "S2 recommendations failed" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._s2_active = False
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_cache = {}
        app._track_task = MagicMock()
        app.action_citation_graph()
        assert "S2 is disabled" in app.notify.call_args[0][0]

        app._s2_active = True
        app.notify.reset_mock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app.action_citation_graph()
        app._track_task.assert_called_once()

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(return_value=([], []))
        await app._show_citation_graph("ARXIV:2401.70006", "Test")
        assert "No citation data found" in app.notify.call_args[0][0]

        refs = [
            CitationEntry(
                s2_paper_id="s2p",
                arxiv_id="2401.70008",
                title="Cited",
                authors="A",
                year=2024,
                citation_count=1,
                url="",
            )
        ]
        app._fetch_citation_graph = AsyncMock(return_value=(refs, []))
        app._papers_by_id = {"2401.70008": make_paper(arxiv_id="2401.70008")}
        await app._show_citation_graph("ARXIV:2401.70006", "Test")
        assert app.push_screen.call_count >= 2

        app.notify.reset_mock()
        app._fetch_citation_graph = AsyncMock(side_effect=RuntimeError("boom"))
        await app._show_citation_graph("ARXIV:2401.70006", "Test")
        assert "Citation graph failed" in app.notify.call_args[0][0]


class TestBookmarkRelevanceAndCopyCoverage:
    @pytest.mark.asyncio
    async def test_bookmark_add_goto_and_remove_paths(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.bookmarks = []
        app._active_bookmark_index = -1
        app._update_bookmark_bar = AsyncMock()
        app._apply_filter = MagicMock()
        app.notify = MagicMock()

        search_input = SimpleNamespace(value="")
        app.query_one = MagicMock(return_value=search_input)
        await app.action_add_bookmark()
        assert "Enter a search query first" in app.notify.call_args[0][0]

        search_input.value = "graph transformers"
        await app.action_add_bookmark()
        assert len(app._config.bookmarks) == 1
        assert app._active_bookmark_index == 0

        await app.action_goto_bookmark(99)
        await app.action_goto_bookmark(0)
        app._apply_filter.assert_called_with("graph transformers")

        app._active_bookmark_index = -1
        await app.action_remove_bookmark()
        assert "No active bookmark to remove" in app.notify.call_args[0][0]

        app._active_bookmark_index = 0
        await app.action_remove_bookmark()
        assert len(app._config.bookmarks) == 0

    @pytest.mark.asyncio
    async def test_bookmark_limit_guard(self):
        app = _new_app()
        app._config = UserConfig()
        app._config.bookmarks = [SearchBookmark(name=f"b{i}", query=f"q{i}") for i in range(9)]
        app.notify = MagicMock()
        app._update_bookmark_bar = AsyncMock()
        app.query_one = MagicMock(return_value=SimpleNamespace(value="new query"))
        await app.action_add_bookmark()
        assert "Maximum 9 bookmarks allowed" in app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_action_check_versions_and_relevance_branches(self):
        app = _new_app()
        app.notify = MagicMock()
        app._update_status_bar = MagicMock()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        app._version_checking = True
        app._config = UserConfig()
        app._config.paper_metadata = {}
        await app.action_check_versions()
        assert "already in progress" in app.notify.call_args[0][0]

        app._version_checking = False
        app.notify.reset_mock()
        await app.action_check_versions()
        assert "No starred papers" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._config.paper_metadata = {
            "2401.80001": PaperMetadata(arxiv_id="2401.80001", starred=True)
        }
        await app.action_check_versions()
        app._track_task.assert_called_once()

        app2 = _new_app()
        app2.notify = MagicMock()
        app2.push_screen = MagicMock()
        app2._start_relevance_scoring = MagicMock()
        app2._require_llm_command = MagicMock(return_value=None)
        app2._relevance_scoring_active = False
        app2._config = UserConfig()
        app2.action_score_relevance()
        app2.push_screen.assert_not_called()

        app2._require_llm_command = MagicMock(return_value="cmd {prompt}")
        app2._relevance_scoring_active = True
        app2.action_score_relevance()
        assert "already in progress" in app2.notify.call_args[0][0]

        app2._relevance_scoring_active = False
        app2._config.research_interests = ""
        app2.action_score_relevance()
        app2.push_screen.assert_called_once()

        app2._config.research_interests = "nlp"
        app2.action_score_relevance()
        app2._start_relevance_scoring.assert_called_with("cmd {prompt}", "nlp")

    def test_on_interests_edited_and_copy_selected_paths(self, make_paper):
        app = _new_app()
        app._config = UserConfig()
        app._config.research_interests = "old"
        app._relevance_scores = {"x": (8, "fit")}
        app._mark_badges_dirty = MagicMock()
        app._refresh_detail_pane = MagicMock()
        app.notify = MagicMock()

        with patch("arxiv_browser.app.save_config", return_value=True) as save:
            app._on_interests_edited("new")
            app._on_interests_edited("new")

        save.assert_called_once()
        assert app._relevance_scores == {}
        assert "updated" in app.notify.call_args[0][0]

        app2 = _new_app()
        app2.notify = MagicMock()
        app2._get_target_papers = MagicMock(return_value=[])
        app2.action_copy_selected()
        assert "No papers to copy" in app2.notify.call_args[0][0]

        p1 = make_paper(arxiv_id="2401.80002")
        p2 = make_paper(arxiv_id="2401.80003")
        app2._get_target_papers = MagicMock(return_value=[p1, p2])
        app2._format_paper_for_clipboard = MagicMock(side_effect=["A", "B"])
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.action_copy_selected()
        assert "Copied 2 papers" in app2.notify.call_args[0][0]

        app2._copy_to_clipboard = MagicMock(return_value=False)
        app2._format_paper_for_clipboard = MagicMock(return_value="X")
        app2.action_copy_selected()
        assert "Failed to copy to clipboard" in app2.notify.call_args[0][0]


class TestAutoTagAndPdfOpenCoverage:
    @pytest.mark.asyncio
    async def test_call_auto_tag_llm_and_acceptance_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app._llm_provider = SimpleNamespace(execute=AsyncMock())
        paper = make_paper(arxiv_id="2401.90001")

        app._llm_provider.execute.return_value = SimpleNamespace(
            success=False, output="", error="bad command"
        )
        result = await app._call_auto_tag_llm(paper, ["topic:ml"])
        assert result is None

        app._llm_provider.execute.return_value = SimpleNamespace(
            success=True, output='{"tags": ["topic:ml"]}', error=""
        )
        with patch("arxiv_browser.app._parse_auto_tag_response", return_value=None):
            result = await app._call_auto_tag_llm(paper, ["topic:ml"])
        assert result is None
        assert "Could not parse LLM response" in app.notify.call_args[0][0]

        with patch("arxiv_browser.app._parse_auto_tag_response", return_value=["topic:ml"]):
            result = await app._call_auto_tag_llm(paper, ["topic:ml"])
        assert result == ["topic:ml"]

        app._config = UserConfig()
        app._config.paper_metadata = {}
        app._update_option_for_paper = MagicMock()
        app._refresh_detail_pane = MagicMock()

        def get_meta(arxiv_id: str) -> PaperMetadata:
            return app._config.paper_metadata.setdefault(arxiv_id, PaperMetadata(arxiv_id=arxiv_id))

        app._get_or_create_metadata = get_meta
        with patch("arxiv_browser.app.save_config", return_value=True):
            app._on_auto_tag_accepted(None, paper.arxiv_id)
            app._on_auto_tag_accepted(["topic:ml"], paper.arxiv_id)
        assert app._config.paper_metadata[paper.arxiv_id].tags == ["topic:ml"]
        assert "Tags updated" in app.notify.call_args[0][0]

    def test_open_pdf_and_copy_bibtex_paths(self, make_paper):
        app = _new_app()
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._do_open_pdfs = MagicMock()
        app._get_target_papers = MagicMock(return_value=[])
        app.action_open_pdf()
        app._do_open_pdfs.assert_not_called()

        app._get_target_papers = MagicMock(return_value=[make_paper(arxiv_id="2401.90002")])
        app.action_open_pdf()
        app._do_open_pdfs.assert_called_once()

        app._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id=f"2401.{i:05d}") for i in range(20)]
        )
        app.action_open_pdf()
        app.push_screen.assert_called_once()

        app2 = _new_app()
        app2.notify = MagicMock()
        app2._get_target_papers = MagicMock(return_value=[])
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.action_copy_bibtex()
        assert "No paper selected" in app2.notify.call_args[0][0]

        app2._get_target_papers = MagicMock(
            return_value=[make_paper(arxiv_id="2401.90003"), make_paper(arxiv_id="2401.90004")]
        )
        app2._copy_to_clipboard = MagicMock(return_value=True)
        app2.action_copy_bibtex()
        assert "Copied 2 BibTeX entries" in app2.notify.call_args[0][0]


class TestCliResolutionCoverage:
    def test_resolve_legacy_fallback_branches(self, make_paper, tmp_path):
        missing = _resolve_legacy_fallback(tmp_path)
        assert missing == 1

        arxiv_file = tmp_path / "arxiv.txt"
        arxiv_file.write_text("placeholder", encoding="utf-8")

        with patch("arxiv_browser.app.os.access", return_value=False):
            unreadable = _resolve_legacy_fallback(tmp_path)
        assert unreadable == 1

        with (
            patch("arxiv_browser.app.os.access", return_value=True),
            patch("arxiv_browser.app.parse_arxiv_file", side_effect=OSError("read error")),
        ):
            read_error = _resolve_legacy_fallback(tmp_path)
        assert read_error == 1

        with (
            patch("arxiv_browser.app.os.access", return_value=True),
            patch("arxiv_browser.app.parse_arxiv_file", return_value=[make_paper()]),
        ):
            ok = _resolve_legacy_fallback(tmp_path)
        assert isinstance(ok, list) and len(ok) == 1

    def test_resolve_papers_uses_explicit_date_argument(self, make_paper, tmp_path):
        args = argparse.Namespace(input=None, date="2026-01-22", no_restore=False)
        config = UserConfig()
        history_files = [
            (date(2026, 1, 23), tmp_path / "2026-01-23.txt"),
            (date(2026, 1, 22), tmp_path / "2026-01-22.txt"),
        ]
        with patch("arxiv_browser.app.parse_arxiv_file", return_value=[make_paper()]):
            result = _resolve_papers(args, tmp_path, config, history_files)
        assert isinstance(result, tuple)
        assert result[2] == 1

    def test_resolve_papers_handles_invalid_saved_date(self, make_paper, tmp_path):
        args = argparse.Namespace(input=None, date=None, no_restore=False)
        config = UserConfig()
        config.session.current_date = "invalid-date"
        history_files = [
            (date(2026, 1, 23), tmp_path / "2026-01-23.txt"),
            (date(2026, 1, 22), tmp_path / "2026-01-22.txt"),
        ]
        with patch("arxiv_browser.app.parse_arxiv_file", return_value=[make_paper()]):
            result = _resolve_papers(args, tmp_path, config, history_files)
        assert isinstance(result, tuple)
        assert result[2] == 0

    def test_resolve_papers_uses_saved_current_date_when_valid(self, make_paper, tmp_path):
        args = argparse.Namespace(input=None, date=None, no_restore=False)
        config = UserConfig()
        config.session.current_date = "2026-01-23"
        history_files = [
            (date(2026, 1, 23), tmp_path / "2026-01-23.txt"),
            (date(2026, 1, 22), tmp_path / "2026-01-22.txt"),
        ]
        with patch("arxiv_browser.app.parse_arxiv_file", return_value=[make_paper()]):
            result = _resolve_papers(args, tmp_path, config, history_files)
        assert isinstance(result, tuple)
        assert result[2] == 0
