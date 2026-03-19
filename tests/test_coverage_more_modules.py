"""Additional branch coverage for smaller modules."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from contextlib import closing
from collections import deque
from datetime import UTC, datetime, date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import arxiv_browser.app as app_mod
from arxiv_browser.actions import external_io_actions as io_actions
from arxiv_browser.actions import llm_actions as llm_actions
import arxiv_browser.cli as cli
import arxiv_browser.llm_providers as llm_providers
import arxiv_browser.semantic_scholar as s2
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionViewModal,
    CollectionsModal,
)
from arxiv_browser.models import MAX_COLLECTIONS, PaperCollection, UserConfig
from arxiv_browser.services import enrichment_service as enrich


class _DummyInput:
    def __init__(self, value: str = "") -> None:
        self.value = value
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _DummyLabel:
    def __init__(self, content: str = "") -> None:
        self.content = content

    def update(self, value: str) -> None:
        self.content = value


class _DummyListView:
    def __init__(self, index: int | None = None) -> None:
        self.index = index
        self.children: list[object] = []
        self.mounted: list[object] = []
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1
        self.children.clear()
        self.mounted.clear()

    def mount(self, item: object) -> None:
        self.children.append(item)
        self.mounted.append(item)


class _DummyTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _paper(arxiv_id: str = "2401.12345", **kwargs):
    from arxiv_browser.app import Paper

    defaults = dict(
        arxiv_id=arxiv_id,
        date="Mon, 15 Jan 2024",
        title=f"Paper {arxiv_id}",
        authors="A. Author",
        categories="cs.AI",
        comments=None,
        abstract="Abstract text.",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        abstract_raw="Abstract text.",
    )
    defaults.update(kwargs)
    return Paper(**defaults)


def _make_app_config(**kwargs):
    config = UserConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def _new_app_stub():
    app = app_mod.ArxivBrowser.__new__(app_mod.ArxivBrowser)
    app.notify = MagicMock()
    app._config = UserConfig()
    app._detail_timer = None
    app._background_tasks = set()
    app._dataset_tasks = set()
    app._ui_refs = SimpleNamespace()
    app._update_status_bar = MagicMock()
    app._update_footer = MagicMock()
    app._update_header = MagicMock()
    app._update_subtitle = MagicMock()
    app._mark_badges_dirty = MagicMock()
    app._refresh_detail_pane = MagicMock()
    app._refresh_list_view = MagicMock()
    app._save_config_or_warn = MagicMock()
    app._get_ui_refresh_coordinator = MagicMock(
        return_value=SimpleNamespace(refresh_detail_pane=MagicMock())
    )
    app._track_task = MagicMock(side_effect=lambda coro: coro.close())
    app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
    app._abstract_cache = {}
    app._abstract_loading = set()
    app._abstract_queue = deque()
    app._abstract_pending_ids = set()
    app._s2_active = False
    app._hf_active = False
    app._s2_cache = {}
    app._hf_cache = {}
    app._s2_loading = set()
    app._hf_loading = False
    app._badges_dirty = set()
    app._badge_timer = None
    app._sort_refresh_dirty = set()
    app._sort_refresh_timer = None
    app._download_queue = deque()
    app._downloading = set()
    app._download_results = {}
    app._download_total = 0
    app._papers_by_id = {}
    app.filtered_papers = []
    app.all_papers = []
    app.selected_ids = set()
    app._highlight_terms = {"abstract": []}
    app._version_updates = {}
    app._relevance_scores = {}
    app._paper_summaries = {}
    app._summary_loading = set()
    app._summary_mode_label = {}
    app._match_scores = {}
    app._highlight_terms = {"abstract": []}
    app._local_browse_snapshot = None
    app._pending_detail_paper = None
    app._pending_detail_started_at = None
    app._show_abstract_preview = False
    app._detail_mode = "scan"
    app._current_date_index = 0
    app._history_files = []
    app._in_arxiv_api_mode = False
    app._active_bookmark_index = 0
    app._sort_index = 0
    return app


class TestLlmProvidersCoverage:
    def test_llm_command_requires_shell_and_windows_quote_handling(self, monkeypatch) -> None:
        assert llm_providers.llm_command_requires_shell("echo {prompt} | cat") is True
        assert llm_providers.llm_command_requires_shell("echo {prompt}") is False

        monkeypatch.setattr(llm_providers.os, "name", "nt", raising=False)
        assert llm_providers.llm_command_requires_shell('"quoted {prompt}"') is False
        assert llm_providers._strip_wrapping_quotes_windows(['"C:\\App\\tool.exe"', "x"]) == [
            "C:\\App\\tool.exe",
            "x",
        ]

    def test_build_invocation_plan_error_paths(self) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            llm_providers._build_invocation_plan("echo hello", "prompt")

        with pytest.raises(ValueError, match="disabled"):
            llm_providers._build_invocation_plan("echo {prompt} | cat", "prompt", allow_shell=False)

    @pytest.mark.asyncio
    async def test_cli_provider_retry_and_no_retry_paths(self) -> None:
        provider = llm_providers.CLIProvider("echo {prompt}", max_retries=1)
        with (
            patch.object(
                llm_providers.CLIProvider,
                "_execute_once",
                new=AsyncMock(
                    side_effect=[
                        llm_providers.LLMResult(output="", success=False, error="Timed out"),
                        llm_providers.LLMResult(output="ok", success=True),
                    ]
                ),
            ) as execute_once,
            patch("arxiv_browser.llm_providers.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await provider.execute("hello", timeout=5)

        assert result.success is True
        assert result.output == "ok"
        assert execute_once.await_count == 2
        sleep.assert_awaited_once()


class TestCliCoverage:
    def test_resolve_input_file_and_history_date_branches(self, tmp_path, make_paper, capsys) -> None:
        missing = cli._resolve_input_file(tmp_path / "missing.txt")
        assert missing == 1

        unreadable = tmp_path / "paper.txt"
        unreadable.write_text("paper", encoding="utf-8")
        with patch("arxiv_browser.cli.os.access", return_value=False):
            assert cli._resolve_input_file(unreadable) == 1

        with patch("arxiv_browser.cli.parse_arxiv_file", side_effect=OSError("boom")):
            assert cli._resolve_input_file(unreadable) == 1

        with patch("arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]):
            assert len(cli._resolve_input_file(unreadable)) == 1

        history_files = [(date(2026, 1, 23), tmp_path / "2026-01-23.txt")]
        assert cli._resolve_history_date(history_files, "2026-13-01") is None
        assert cli._resolve_history_date(history_files, "2026-01-22") is None
        assert cli._resolve_history_date(history_files, "2026-01-23") == 0

    def test_resolve_legacy_fallback_branches(self, tmp_path, make_paper) -> None:
        assert cli._resolve_legacy_fallback(tmp_path) == 1

        arxiv_txt = tmp_path / "arxiv.txt"
        arxiv_txt.write_text("placeholder", encoding="utf-8")
        with patch("arxiv_browser.cli.os.access", return_value=False):
            assert cli._resolve_legacy_fallback(tmp_path) == 1
        with patch("arxiv_browser.cli.os.access", return_value=True), patch(
            "arxiv_browser.cli.parse_arxiv_file", side_effect=OSError("boom")
        ):
            assert cli._resolve_legacy_fallback(tmp_path) == 1
        with patch("arxiv_browser.cli.os.access", return_value=True), patch(
            "arxiv_browser.cli.parse_arxiv_file", return_value=[make_paper()]
        ):
            assert len(cli._resolve_legacy_fallback(tmp_path)) == 1

    def test_resolve_arxiv_api_mode_success_and_errors(self, make_paper, capsys) -> None:
        args = argparse.Namespace(command="search", query="graph", category="cs.AI", field="title", mode="page", max_results=5)
        config = UserConfig(arxiv_api_max_results=7)
        with patch("arxiv_browser.cli._fetch_arxiv_api_papers", return_value=[make_paper()]):
            result = cli._resolve_arxiv_api_mode(args, config)
        assert isinstance(result, tuple)
        assert result[0][0].arxiv_id == "2401.12345"

        args.mode = "latest"
        with patch("arxiv_browser.cli._fetch_latest_arxiv_digest", return_value=[make_paper()]):
            result = cli._resolve_arxiv_api_mode(args, config)
        assert isinstance(result, tuple)

        with patch("arxiv_browser.cli._fetch_arxiv_api_papers", side_effect=ValueError("bad query")):
            assert cli._resolve_arxiv_api_mode(args, config) == 1

        response_429 = MagicMock(spec=httpx.Response)
        response_429.status_code = 429
        exc_429 = httpx.HTTPStatusError(
            "429",
            request=httpx.Request("GET", "https://example.com"),
            response=response_429,
        )
        with patch("arxiv_browser.cli._fetch_latest_arxiv_digest", side_effect=exc_429):
            assert cli._resolve_arxiv_api_mode(args, config) == 1

        response_503 = MagicMock(spec=httpx.Response)
        response_503.status_code = 503
        exc_503 = httpx.HTTPStatusError(
            "503",
            request=httpx.Request("GET", "https://example.com"),
            response=response_503,
        )
        with patch("arxiv_browser.cli._fetch_latest_arxiv_digest", side_effect=exc_503):
            assert cli._resolve_arxiv_api_mode(args, config) == 1

        assert cli._resolve_arxiv_api_mode(argparse.Namespace(command="browse"), config) is None

    def test_cli_helpers_and_doctor_paths(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(cli.os, "name", "posix", raising=False)
        assert cli._extract_command_binary("OPENAI_API_KEY=1 llm {prompt}") == "llm"
        assert cli._extract_command_binary("bad \"quote") is None

        cli._configure_color_mode("never")
        assert cli.os.environ["NO_COLOR"] == "1"
        cli._configure_color_mode("always")
        assert cli.os.environ["FORCE_COLOR"] == "1"
        cli._configure_color_mode("auto")

        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
        assert cli._validate_interactive_tty() is True

        config_path = tmp_path / "config.json"
        assert cli._doctor_config_issue_count(config_path, ok_marker="OK", info_marker="INFO") == 0
        config_path.write_text("{}", encoding="utf-8")
        assert cli._doctor_config_issue_count(config_path, ok_marker="OK", info_marker="INFO") == 0

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        assert cli._doctor_history_issue_count([], ok_marker="OK", warn_marker="WARN", info_marker="INFO") == 1
        (history_dir / "2026-03-19.txt").write_text("paper", encoding="utf-8")
        assert cli._doctor_history_issue_count([(date(2026, 3, 19), history_dir / "2026-03-19.txt")], ok_marker="OK", warn_marker="WARN", info_marker="INFO") == 0

        cfg = UserConfig()
        cfg.llm_command = ""
        cfg.llm_preset = ""
        assert cli._doctor_llm_issue_count(cfg, ok_marker="OK", warn_marker="WARN", info_marker="INFO") == 0


class TestEnrichmentServiceCoverage:
    @pytest.mark.asyncio
    async def test_best_effort_cache_write_and_result_states(self, tmp_path, make_paper, caplog) -> None:
        await enrich._best_effort_cache_write(lambda *_args: None)
        with patch("arxiv_browser.services.enrichment_service.asyncio.to_thread", side_effect=OSError("boom")):
            await enrich._best_effort_cache_write(lambda *_args: None)

        s2_paper = s2.SemanticScholarPaper(
            arxiv_id="2401.1",
            s2_paper_id="s2:1",
            citation_count=1,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        hf_paper = enrich.HuggingFacePaper(
            arxiv_id="2401.2",
            title="hf",
            upvotes=1,
            num_comments=0,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )

        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
            return_value=s2.S2PaperCacheSnapshot(status="found", paper=s2_paper),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.paper is s2_paper
        assert result.from_cache is True

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="not_found", paper=None),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.state == "not_found"

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_paper",
                new=AsyncMock(return_value=(None, False)),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
            )
        assert result.state == "unavailable"

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
                return_value=enrich.HFDailyCacheSnapshot(status="found", papers={"2401.2": hf_paper}),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_hf_daily_papers",
                new=AsyncMock(return_value=([hf_paper], True)),
            ),
        ):
            result = await enrich.load_or_fetch_hf_daily_result(
                db_path=tmp_path / "hf.db",
                cache_ttl_hours=6,
                client=object(),
            )
        assert result.papers == [hf_paper]
        assert result.from_cache is True

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
                return_value=s2.S2RecommendationsCacheSnapshot(status="empty", papers=[]),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.fetch_s2_recommendations_with_status",
                new=AsyncMock(return_value=([s2_paper], True)),
            ),
        ):
            result = await enrich.load_or_fetch_s2_recommendations_result(
                arxiv_id="2401.1",
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=object(),
                api_key="",
            )
        assert result.from_cache is True

    @pytest.mark.asyncio
    async def test_cached_wrappers_cover_no_client_and_status_paths(self, tmp_path, make_paper) -> None:
        paper = s2.SemanticScholarPaper(
            arxiv_id="2401.10",
            s2_paper_id="s2:10",
            citation_count=5,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
            return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
        ):
            assert await enrich.load_or_fetch_s2_paper_cached(
                arxiv_id="2401.10",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=None,
                api_key="",
            ) is None

        with (
            patch(
                "arxiv_browser.services.enrichment_service.load_s2_paper_snapshot",
                return_value=s2.S2PaperCacheSnapshot(status="miss", paper=None),
            ),
            patch(
                "arxiv_browser.services.enrichment_service.load_or_fetch_s2_paper_result",
                new=AsyncMock(return_value=enrich.S2PaperFetchResult(
                    state="found",
                    paper=paper,
                    complete=True,
                    from_cache=False,
                )),
            ),
        ):
            result = await enrich.load_or_fetch_s2_paper_cached(
                arxiv_id="2401.10",
                db_path=tmp_path / "s2.db",
                cache_ttl_days=7,
                client=object(),
                api_key="",
                include_status=True,
            )
        assert result == (paper, True)

        with patch(
            "arxiv_browser.services.enrichment_service.load_hf_daily_cache_snapshot",
            return_value=enrich.HFDailyCacheSnapshot(status="miss", papers={}),
        ):
            assert await enrich.load_or_fetch_hf_daily_cached(
                db_path=tmp_path / "hf.db",
                cache_ttl_hours=6,
                client=None,
            ) == []

        with patch(
            "arxiv_browser.services.enrichment_service.load_s2_recommendations_snapshot",
            return_value=s2.S2RecommendationsCacheSnapshot(status="miss", papers=[]),
        ):
            assert await enrich.load_or_fetch_s2_recommendations_cached(
                arxiv_id="2401.10",
                db_path=tmp_path / "recs.db",
                cache_ttl_days=3,
                client=None,
                api_key="",
            ) == []


class TestSemanticScholarCoverage:
    def test_citation_graph_cache_round_trip_and_helpers(self, tmp_path, make_paper) -> None:
        db_path = tmp_path / "s2.db"
        entry = s2.CitationEntry(
            s2_paper_id="paper-1",
            arxiv_id="2401.99999",
            title="Cited",
            authors="A. Author",
            year=2024,
            citation_count=9,
            url="https://arxiv.org/abs/2401.99999",
        )
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1") is False
        s2.save_s2_citation_graph(db_path, "paper-1", "references", [entry])
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1") is True
        assert s2.load_s2_citation_graph(db_path, "paper-1", "references") == [entry]
        assert s2.load_s2_citation_graph(db_path, "paper-1", "citations") == []
        assert s2.has_s2_citation_graph_cache(db_path, "paper-1", ttl_days=0) is False

        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO s2_citation_graph (paper_id, direction, rank, payload_json, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("paper-2", "references", 0, '{"bad": "json"}', datetime.now(UTC).isoformat()),
            )
        assert s2.load_s2_citation_graph(db_path, "paper-2", "references") == []

    def test_s2_snapshot_cache_states(self, tmp_path, make_paper) -> None:
        db_path = tmp_path / "s2.db"
        paper = s2.SemanticScholarPaper(
            arxiv_id="2401.1",
            s2_paper_id="s2:1",
            citation_count=1,
            influential_citation_count=0,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )

        assert s2.load_s2_paper_snapshot(db_path, "2401.1").status == "miss"
        s2.save_s2_paper_not_found(db_path, "2401.1")
        assert s2.load_s2_paper_snapshot(db_path, "2401.1").status == "not_found"
        s2.save_s2_paper(db_path, paper)
        assert s2.load_s2_paper_snapshot(db_path, "2401.1").status == "found"

        rec = s2.SemanticScholarPaper(
            arxiv_id="2401.2",
            s2_paper_id="s2:2",
            citation_count=2,
            influential_citation_count=1,
            tldr="",
            fields_of_study=(),
            year=2024,
            url="https://example.com",
        )
        s2.save_s2_recommendations(db_path, "paper-x", [rec])
        assert s2.load_s2_recommendations_snapshot(db_path, "paper-x").status == "found"
        assert s2.load_s2_recommendations_snapshot(db_path, "missing").status == "miss"

    @pytest.mark.asyncio
    async def test_fetch_s2_citations_and_references_edge_cases(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await s2.fetch_s2_citations("paper", client, limit=0) == []

        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"data": []}
        client.get.return_value = response
        assert await s2.fetch_s2_references("paper", client, include_status=True) == ([], True)

        response.json.return_value = {"recommendedPapers": "oops"}
        assert await s2.fetch_s2_recommendations("paper", client) == []


class TestCollectionsCoverage:
    def test_collections_modal_create_rename_delete_and_view_branches(self, make_paper) -> None:
        base = PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])
        modal = CollectionsModal([base], papers_by_id={"2401.00001": make_paper()})
        list_view = _DummyListView(index=0)
        name_input = _DummyInput("")
        desc_input = _DummyInput("notes")
        modal.notify = MagicMock()
        modal.dismiss = MagicMock()
        modal._collections = [PaperCollection(name="Reading", description="desc", paper_ids=["2401.00001"])]
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
        modal._collections = [PaperCollection(name=f"C{i}", paper_ids=[]) for i in range(MAX_COLLECTIONS)]
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


class TestAppActionBranches:
    @pytest.mark.asyncio
    async def test_ui_action_branches_cover_app_wrappers(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.50001")
        app._config = _make_app_config(
            s2_enabled=False,
            hf_enabled=False,
            s2_cache_ttl_days=7,
            hf_cache_ttl_hours=6,
            theme_name="not-a-theme",
            collections=[PaperCollection(name="Reading", paper_ids=["2401.50001"])],
        )
        app._http_client = object()
        app._s2_db_path = tmp_path / "s2.db"
        app._hf_db_path = tmp_path / "hf.db"
        app._s2_active = False
        app._hf_active = False
        app._s2_loading = set()
        app._hf_loading = False
        app._s2_cache = {}
        app._hf_cache = {}
        app._papers_by_id = {paper.arxiv_id: paper}
        app._get_current_paper = MagicMock(return_value=paper)
        app._get_target_papers = MagicMock(return_value=[paper])
        app._get_ui_refresh_coordinator.return_value.refresh_detail_pane = MagicMock()
        app._get_paper_details_widget = MagicMock(
            return_value=SimpleNamespace(clear_cache=MagicMock())
        )
        app._apply_theme_overrides = MagicMock()
        app._apply_category_overrides = MagicMock()
        app._show_recommendations = MagicMock()
        app._show_citation_graph = AsyncMock(return_value=None)
        app.push_screen = MagicMock()

        with patch("arxiv_browser.app.save_config", return_value=False):
            app.action_toggle_s2()
        assert app._config.s2_enabled is False
        assert "Failed to save Semantic Scholar setting" in app.notify.call_args[0][0]

        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_toggle_s2()
        assert app._s2_active is True
        assert app._config.s2_enabled is True

        app._get_current_paper = MagicMock(return_value=None)
        await app.action_fetch_s2()
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_loading = {paper.arxiv_id}
        await app.action_fetch_s2()
        app._s2_loading = set()
        app._s2_cache = {paper.arxiv_id: object()}
        await app.action_fetch_s2()
        app._s2_cache = {}
        app._s2_active = True
        def _track_raises(coro):
            coro.close()
            raise OSError("boom")

        app._track_dataset_task = MagicMock(side_effect=_track_raises)
        with pytest.raises(OSError):
            await app.action_fetch_s2()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())

        app._http_client = None
        await app._fetch_s2_paper_async(paper.arxiv_id)
        app._http_client = object()
        app._services = SimpleNamespace(
            enrichment=SimpleNamespace(
                load_or_fetch_s2_paper=AsyncMock(
                    return_value=SimpleNamespace(
                        state="not_found",
                        paper=None,
                        complete=True,
                        from_cache=False,
                    )
                )
            )
        )
        await app._fetch_s2_paper_async(paper.arxiv_id)
        app._services.enrichment.load_or_fetch_s2_paper = AsyncMock(
            return_value=SimpleNamespace(
                state="found",
                paper=SimpleNamespace(arxiv_id=paper.arxiv_id, s2_paper_id="s2:1"),
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_s2_paper_async(paper.arxiv_id)
        app._services.enrichment.load_or_fetch_s2_paper = AsyncMock(
            return_value=SimpleNamespace(
                state="unavailable",
                paper=None,
                complete=False,
                from_cache=False,
            )
        )
        await app._fetch_s2_paper_async(paper.arxiv_id)

        with patch("arxiv_browser.app.save_config", return_value=False):
            await app.action_toggle_hf()
        assert app._config.hf_enabled is False

        app._config.hf_enabled = True
        app._hf_active = False
        app._hf_cache = {}
        with patch("arxiv_browser.app.save_config", return_value=True):
            await app.action_toggle_hf()
        assert app._hf_active is True
        app._hf_cache = {paper.arxiv_id: object()}
        await app.action_toggle_hf()

        app._hf_loading = False
        await app._fetch_hf_daily()
        app._hf_loading = True
        await app._fetch_hf_daily()
        app._hf_loading = True
        app._http_client = None
        await app._fetch_hf_daily_async()
        app._http_client = object()
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="empty",
                papers=[],
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="found",
                papers=[SimpleNamespace(arxiv_id=paper.arxiv_id)],
                complete=True,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()
        app._services.enrichment.load_or_fetch_hf_daily = AsyncMock(
            return_value=SimpleNamespace(
                state="unavailable",
                papers=[],
                complete=False,
                from_cache=False,
            )
        )
        await app._fetch_hf_daily_async()

        app._get_current_paper = MagicMock(return_value=None)
        app.action_show_similar()
        app._get_current_paper = MagicMock(return_value=paper)
        app._s2_active = False
        app.action_show_similar()
        app._s2_active = True
        app.action_show_similar()
        app._s2_cache = {}
        app.action_citation_graph()
        app._s2_cache = {paper.arxiv_id: SimpleNamespace(s2_paper_id="s2:1")}
        app.action_citation_graph()

        captured = {}
        app.push_screen = lambda modal, cb: captured.update(modal=modal, callback=cb)
        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_cycle_theme()
            app.action_toggle_sections()
            captured["callback"](None)
            app.action_toggle_sections()
            captured["callback"](["abstract"])

            app._config.collections = [PaperCollection(name="Reading", paper_ids=["2401.50001"])]
            app._get_target_papers = MagicMock(return_value=[paper])
            app.action_collections()
            captured["callback"]("save")

        captured_add = {}
        app.push_screen = lambda modal, cb: captured_add.update(modal=modal, callback=cb)
        with patch("arxiv_browser.app.save_config", return_value=True):
            app.action_add_to_collection()
            captured_add["callback"]("Reading")


class TestAppHelperCoverage:
    def test_list_message_and_paper_content_branches(self) -> None:
        assert "No papers match your search" in app_mod.build_list_empty_message(
            query="x",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No API results on this page" in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=True,
            watch_filter_active=False,
            history_mode=False,
        )
        assert "No watched papers found" in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=True,
            history_mode=False,
        )
        assert "No papers available for this date" in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=True,
        )
        assert "No papers available." in app_mod.build_list_empty_message(
            query="",
            in_arxiv_api_mode=False,
            watch_filter_active=False,
            history_mode=False,
        )

        class _Response:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

        class _Client:
            def __init__(self, response: _Response) -> None:
                self.response = response

            async def get(self, *_args, **_kwargs):
                return self.response

        class _TempClient:
            def __init__(self, response: _Response) -> None:
                self.response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *_args, **_kwargs):
                return self.response

        paper = _paper(arxiv_id="2401.00001", abstract="Fallback abstract.")
        empty_paper = _paper(arxiv_id="2401.00002", abstract=None, abstract_raw=None)
        long_text = "x" * (app_mod.MAX_PAPER_CONTENT_LENGTH + 10)

        with patch("arxiv_browser.app.extract_text_from_html", return_value=long_text):
            text = asyncio.run(app_mod._fetch_paper_content_async(paper, _Client(_Response(200, "<p>x</p>"))))
        assert len(text) == app_mod.MAX_PAPER_CONTENT_LENGTH

        with patch("arxiv_browser.app.extract_text_from_html", return_value=""):
            text = asyncio.run(app_mod._fetch_paper_content_async(paper, _Client(_Response(404, ""))))
        assert text == "Abstract:\nFallback abstract."

        with (
            patch("arxiv_browser.app.extract_text_from_html", return_value=""),
            patch(
                "arxiv_browser.app.httpx.AsyncClient",
                return_value=_TempClient(_Response(200, "<p>x</p>")),
            ),
        ):
            text = asyncio.run(app_mod._fetch_paper_content_async(empty_paper, None))
        assert text == ""

    @pytest.mark.asyncio
    async def test_track_and_cancel_helpers_and_task_done(self) -> None:
        async def noop():
            return None

        app = _new_app_stub()
        app._track_task = app_mod.ArxivBrowser._track_task.__get__(app, app_mod.ArxivBrowser)
        app._track_dataset_task = app_mod.ArxivBrowser._track_dataset_task.__get__(
            app, app_mod.ArxivBrowser
        )
        task = app._track_task(noop())
        assert task in app._background_tasks
        await task
        assert task not in app._background_tasks

        task2 = app._track_task(noop(), dataset_bound=True)
        assert task2 in app._background_tasks
        assert task2 in app._dataset_tasks
        await task2
        assert task2 not in app._background_tasks
        assert task2 not in app._dataset_tasks

        app._track_task = MagicMock(return_value=asyncio.create_task(noop()))
        pending_coro = noop()
        tracked = app._track_dataset_task(pending_coro)
        pending_coro.close()
        assert tracked in app._dataset_tasks
        await tracked
        assert tracked not in app._dataset_tasks

        pending = asyncio.create_task(asyncio.sleep(10))
        done = asyncio.create_task(noop())
        await asyncio.sleep(0)
        app._dataset_tasks = {pending, done}
        app._cancel_dataset_tasks()
        await asyncio.sleep(0)
        assert app._dataset_tasks == set()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        app.notify = MagicMock()
        exc_task = MagicMock()
        exc_task.cancelled.return_value = False
        exc_task.exception.return_value = RuntimeError("boom")
        app._on_task_done(exc_task)
        assert app.notify.called

        app.notify.reset_mock()
        exc_task.cancelled.return_value = True
        app._on_task_done(exc_task)
        app.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_abstract_and_detail_helpers(self, make_paper) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.10001", abstract=None, abstract_raw="x^2")
        other = make_paper(arxiv_id="2401.10002", abstract=None, abstract_raw="y^2")
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())

        app._schedule_abstract_load(paper)
        assert paper.arxiv_id in app._abstract_loading
        app._schedule_abstract_load(paper)
        assert len(app._abstract_loading) == 1

        app._abstract_loading = {str(i) for i in range(app_mod.MAX_ABSTRACT_LOADS)}
        app._schedule_abstract_load(other)
        assert other.arxiv_id in app._abstract_pending_ids
        assert list(app._abstract_queue)[0] == other

        app._abstract_loading = set()
        app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
        app._drain_abstract_queue()
        assert other.arxiv_id in app._abstract_loading

        app._abstract_cache = {paper.arxiv_id: "cached"}
        assert app._get_abstract_text(paper, allow_async=False) == "cached"

        fresh = make_paper(arxiv_id="2401.10003", abstract="already cleaned", abstract_raw="raw")
        assert app._get_abstract_text(fresh, allow_async=False) == "already cleaned"

        blank = make_paper(arxiv_id="2401.10004", abstract=None, abstract_raw=None)
        assert app._get_abstract_text(blank, allow_async=False) == ""

        latex = make_paper(arxiv_id="2401.10005", abstract=None, abstract_raw="\\alpha")
        app._schedule_abstract_load = MagicMock()
        with patch("arxiv_browser.app.clean_latex", return_value="alpha"):
            assert app._get_abstract_text(latex, allow_async=False) == "alpha"
        assert app._get_abstract_text(latex, allow_async=True) == "alpha"

        app._capture_dataset_epoch = MagicMock(return_value=1)
        app._is_current_dataset_epoch = MagicMock(return_value=True)
        app._update_abstract_display = MagicMock()
        with patch("arxiv_browser.app.asyncio.to_thread", new=AsyncMock(return_value="beta")):
            await app._load_abstract_async(latex)
        assert app._abstract_cache[latex.arxiv_id] == "beta"
        app._update_abstract_display.assert_called_with(latex.arxiv_id)

        app._update_abstract_display.reset_mock()
        with patch("arxiv_browser.app.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("boom"))):
            await app._load_abstract_async(other)
        assert other.arxiv_id not in app._abstract_loading

        details = SimpleNamespace(
            paper=SimpleNamespace(arxiv_id=latex.arxiv_id),
            update_paper=MagicMock(),
        )
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._detail_kwargs = MagicMock(return_value={})
        app._update_abstract_display = app_mod.ArxivBrowser._update_abstract_display.__get__(
            app, app_mod.ArxivBrowser
        )
        app._show_abstract_preview = False
        app._abstract_cache[latex.arxiv_id] = "beta"
        app._update_abstract_display(latex.arxiv_id)
        details.update_paper.assert_called_once()

        app._show_abstract_preview = True
        app._update_option_for_paper = MagicMock()
        app._update_abstract_display(latex.arxiv_id)
        app._update_option_for_paper.assert_called_with(latex.arxiv_id)

        app.filtered_papers = [latex]
        app._get_paper_details_widget = MagicMock(return_value=details)
        details.update_paper.reset_mock()
        app.on_paper_selected(SimpleNamespace(option_index=0))
        assert details.update_paper.called

        app._pending_detail_paper = latex
        app._pending_detail_started_at = None
        app._detail_timer = _DummyTimer()
        app._get_current_paper = MagicMock(return_value=latex)
        app._get_abstract_text = MagicMock(return_value="abstract")
        app._get_paper_details_widget = MagicMock(return_value=details)
        app._debounced_detail_update()
        assert app._pending_detail_paper is None


class TestExternalIoCoverage:
    def test_export_and_import_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.30001")
        app._config = UserConfig()
        app._config.paper_metadata = {}
        app.notify = MagicMock()

        app._get_target_papers = MagicMock(return_value=[])
        io_actions.action_copy_bibtex(app)
        assert "No paper selected" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._get_target_papers = MagicMock(return_value=[paper])
        app._copy_to_clipboard = MagicMock(return_value=True)
        io_actions.action_copy_bibtex(app)
        assert "Copied 1 BibTeX entry" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        app._copy_to_clipboard = MagicMock(return_value=False)
        app._format_paper_as_markdown = MagicMock(return_value="## title")
        io_actions.action_export_markdown(app)
        assert "Failed to copy to clipboard" in app.notify.call_args[0][0]

        app._copy_to_clipboard = MagicMock(return_value=True)
        app._format_paper_for_clipboard = MagicMock(side_effect=["A", "B"])
        app._get_target_papers = MagicMock(return_value=[paper, make_paper(arxiv_id="2401.30002")])
        io_actions.action_copy_selected(app)
        assert "Copied 2 papers to clipboard" in app.notify.call_args[0][0]

        app._export_file_csv = MagicMock()
        io_actions._do_export(app, "file-csv", [paper])
        app._export_file_csv.assert_called_once_with([paper])
        app._export_clipboard_ris = MagicMock()
        io_actions._do_export(app, "clipboard-ris", [paper])
        app._export_clipboard_ris.assert_called_once_with([paper])

        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        json1 = export_dir / "arxiv-2026-01-01_010101.json"
        json1.write_text("{}", encoding="utf-8")
        json2 = export_dir / "arxiv-2026-01-02_010101.json"
        json2.write_text("{}", encoding="utf-8")

        app._get_export_dir = MagicMock(return_value=export_dir)
        app.notify.reset_mock()
        with patch("arxiv_browser.actions.external_io_actions._list_metadata_snapshots", return_value=[]):
            io_actions.action_import_metadata(app)
        assert "No metadata snapshots found" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with (
            patch(
                "arxiv_browser.actions.external_io_actions._list_metadata_snapshots",
                return_value=[json1],
            ),
            patch("arxiv_browser.actions.external_io_actions._import_metadata_file") as import_mock,
        ):
            io_actions.action_import_metadata(app)
        import_mock.assert_called_once_with(app, json1)

        app.push_screen = MagicMock()
        with patch(
            "arxiv_browser.actions.external_io_actions._list_metadata_snapshots",
            return_value=[json1, json2],
        ):
            io_actions.action_import_metadata(app)
        modal = app.push_screen.call_args.args[0]
        callback = app.push_screen.call_args.kwargs["callback"]
        assert modal is not None
        callback(None)
        callback(json2)

        bad_json = export_dir / "arxiv-2026-01-03_010101.json"
        bad_json.write_text("not json", encoding="utf-8")
        app._compute_watched_papers = MagicMock()
        app._refresh_list_view = MagicMock()
        with patch("arxiv_browser.actions.external_io_actions.import_metadata", return_value=(1, 0, 0, 0)):
            with patch("arxiv_browser.actions.external_io_actions.save_config", return_value=False):
                io_actions._import_metadata_file(app, bad_json)
        assert app.notify.call_args[0][0].startswith("Import failed") or "Imported" in app.notify.call_args[0][0]

    def test_open_and_download_branches(self, make_paper, tmp_path) -> None:
        app = _new_app_stub()
        paper = make_paper(arxiv_id="2401.40001")
        paper2 = make_paper(arxiv_id="2401.40002")
        app.notify = MagicMock()
        app.push_screen = MagicMock()
        app._safe_browser_open = MagicMock(return_value=True)
        app._ensure_pdf_viewer_trusted = MagicMock(return_value=False)
        app._open_with_viewer = MagicMock(return_value=True)
        app._is_download_batch_active = MagicMock(return_value=False)
        app._start_downloads = MagicMock()
        app._do_start_downloads = MagicMock()
        app._download_pdf_async = AsyncMock(return_value=True)
        app._update_download_progress = MagicMock()
        app._finish_download_batch = MagicMock()

        app._get_target_papers = MagicMock(return_value=[])
        io_actions.action_open_url(app)
        io_actions.action_open_pdf(app)
        io_actions.action_download_pdf(app)

        app._get_target_papers = MagicMock(return_value=[paper])
        io_actions.action_open_url(app)
        io_actions.action_open_pdf(app)
        io_actions.action_download_pdf(app)
        assert app._safe_browser_open.called or app._open_with_viewer.called

        app._config = UserConfig()
        app._config.pdf_viewer = "viewer {url}"
        io_actions._do_open_pdfs(app, [paper, paper2])
        app._ensure_pdf_viewer_trusted.assert_called()

        app._ensure_pdf_viewer_trusted = MagicMock(return_value=True)
        io_actions._do_open_pdfs(app, [paper])
        app._open_with_viewer.assert_called()

        with patch("arxiv_browser.actions.external_io_actions.subprocess.Popen", return_value=None):
            assert io_actions._open_with_viewer(app, "viewer {url}", "https://arxiv.org/abs/x")

        with patch(
            "arxiv_browser.actions.external_io_actions.subprocess.Popen",
            side_effect=OSError("bad"),
        ):
            assert io_actions._open_with_viewer(app, "viewer {url}", "https://arxiv.org/abs/x") is False

        app._download_queue = deque([paper, paper2])
        app._downloading = set()
        app._track_task = MagicMock(side_effect=lambda coro: coro.close())
        io_actions._start_downloads(app)
        assert app._track_task.called

        app._download_results = {}
        app._download_total = 2
        app._download_pdf_async = AsyncMock(return_value=True)
        app._downloading = {paper.arxiv_id}
        asyncio.run(io_actions._process_single_download(app, paper))
        assert paper.arxiv_id not in app._downloading

        app._download_pdf_async = AsyncMock(side_effect=RuntimeError("boom"))
        app._downloading = {paper2.arxiv_id}
        asyncio.run(io_actions._process_single_download(app, paper2))
        assert paper2.arxiv_id not in app._downloading

        app._download_results = {"a": True, "b": False}
        app._download_total = 2
        app._update_status_bar = MagicMock()
        io_actions._finish_download_batch(app)
        assert app._download_total == 0


class TestLlmActionCoverage:
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
        with patch("arxiv_browser.app._load_summary", return_value="cached"):
            llm_actions._on_summary_mode_selected(app, "default", paper, "cmd {prompt}")
        assert "Summary loaded from cache" in app.notify.call_args[0][0]

        app.notify.reset_mock()
        with patch("arxiv_browser.app._load_summary", return_value=None):
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
        with patch("arxiv_browser.actions.llm_actions.asyncio.to_thread", new=AsyncMock(return_value=None)):
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
        llm_actions.action_chat_with_paper(app)
        app._start_chat_with_paper.assert_called_once()

        app._start_chat_with_paper = llm_actions._start_chat_with_paper.__get__(
            app, app_mod.ArxivBrowser
        )
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
