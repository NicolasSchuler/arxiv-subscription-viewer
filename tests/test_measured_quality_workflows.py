"""Scenario tests for measurable workflow and architecture hardening."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from textual.widgets import Input, OptionList

from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
from arxiv_browser.export import format_papers_as_markdown_table
from arxiv_browser.models import ArxivSearchRequest, Paper, UserConfig
from arxiv_browser.parsing import parse_arxiv_api_feed, parse_arxiv_file
from arxiv_browser.query import reconstruct_query, tokenize_query
from arxiv_browser.services.interfaces import AppServices
from arxiv_browser.widgets.omni_input import OmniInput
from tests.support.patch_helpers import patch_save_config


@dataclass(slots=True)
class _FakeArxivApiService:
    app_ref: ArxivBrowser | None = None
    fetched_pages: list[tuple[ArxivSearchRequest, int, int]] | None = None
    stale_during_fetch: bool = False

    def format_query_label(self, request: ArxivSearchRequest) -> str:
        return f"fake:{request.field}:{request.query}:{request.category}"

    async def enforce_rate_limit(self, **_kwargs):
        return (123.0, 0.0)

    async def fetch_page(self, **kwargs) -> list[Paper]:
        request = kwargs["request"]
        start = int(kwargs["start"])
        max_results = int(kwargs["max_results"])
        if self.fetched_pages is not None:
            self.fetched_pages.append((request, start, max_results))
        if self.stale_during_fetch and self.app_ref is not None:
            self.app_ref._arxiv_api_request_token += 1
        return [
            Paper(
                arxiv_id=f"2601.{start + 1:05d}",
                date="Mon, 26 Jan 2026",
                title=f"API Result {start + 1}",
                authors="Remote Author",
                categories="cs.AI",
                comments=None,
                abstract="Remote abstract.",
                abstract_raw="Remote abstract.",
                url=f"https://arxiv.org/abs/2601.{start + 1:05d}",
                source="api",
            )
        ]


class _FakeLlmService:
    def __init__(self, *, fail_second: bool = False) -> None:
        self.scored: list[str] = []
        self.fail_second = fail_second

    async def generate_summary(self, **_kwargs):
        return ("summary", None)

    async def score_relevance_once(self, **kwargs):
        paper = kwargs["paper"]
        self.scored.append(paper.arxiv_id)
        if self.fail_second and len(self.scored) == 2:
            return None
        return (9, f"Relevant: {paper.title}")

    async def suggest_tags_once(self, **_kwargs):
        return ["priority"]


class _FakeDownloadService:
    async def download_pdf(self, **_kwargs):
        return SimpleNamespace(success=True, failure=None, detail="")


class _FakeEnrichmentService:
    async def load_or_fetch_s2_paper(self, **_kwargs):
        return SimpleNamespace(state="not_found", paper=None, complete=True, from_cache=False)

    async def load_or_fetch_hf_daily(self, **_kwargs):
        return SimpleNamespace(state="empty", papers=[], complete=True, from_cache=False)

    async def load_or_fetch_s2_recommendations(self, **_kwargs):
        return SimpleNamespace(state="empty", papers=[], complete=True, from_cache=False)


def _fake_services(
    *,
    arxiv_api: _FakeArxivApiService | None = None,
    llm: _FakeLlmService | None = None,
) -> AppServices:
    return AppServices(
        arxiv_api=arxiv_api or _FakeArxivApiService(),
        llm=llm or _FakeLlmService(),
        download=_FakeDownloadService(),
        enrichment=_FakeEnrichmentService(),
    )


def _config(tmp_path: Path, **kwargs) -> UserConfig:
    config = UserConfig(**kwargs)
    config.pdf_download_dir = str(tmp_path / "pdfs")
    config.bibtex_export_dir = str(tmp_path / "exports")
    config.llm_command = "fake {prompt}"
    return config


def _patch_cache_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import arxiv_browser.browser.core as browser_core

    monkeypatch.setattr(browser_core, "get_cache_db_path", lambda: tmp_path / "cache.db")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acceptance_first_run_welcome_persists_without_stacking_whats_new(
    tmp_path, monkeypatch, make_paper
) -> None:
    _patch_cache_db(monkeypatch, tmp_path)
    config = _config(tmp_path, onboarding_seen=False, last_seen_whats_new="")
    app = ArxivBrowser(
        [make_paper()],
        options=ArxivBrowserOptions(config=config, restore_session=False),
    )

    with patch_save_config(return_value=True):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.screen.__class__.__name__ == "WelcomeScreen"
            await pilot.press("enter")
            await pilot.pause()

    assert config.onboarding_seen is True
    assert config.last_seen_whats_new


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acceptance_local_research_workflow_updates_state_and_export(
    tmp_path, monkeypatch
) -> None:
    _patch_cache_db(monkeypatch, tmp_path)
    papers = parse_arxiv_file(Path("tests/fixtures/2026-01-26.txt"))[:4]
    config = _config(tmp_path)
    app = ArxivBrowser(
        papers,
        options=ArxivBrowserOptions(config=config, restore_session=False),
    )

    with patch_save_config(return_value=True):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            omni = app.query_one(OmniInput)
            omni.open()
            search_input = app.query_one("#omni-input", Input)
            search_input.value = "Agents"
            await pilot.pause(0.1)
            option_list = app.query_one("#paper-list", OptionList)
            assert 0 < option_list.option_count <= len(papers)
            assert app.filtered_papers
            current = app.filtered_papers[0]
            app.selected_ids.add(current.arxiv_id)
            app.action_toggle_read()
            app.action_toggle_star()
            meta = app._get_or_create_metadata(current.arxiv_id)
            meta.tags = ["priority"]

    assert current.arxiv_id in app.selected_ids
    assert config.paper_metadata[current.arxiv_id].is_read is True
    assert config.paper_metadata[current.arxiv_id].starred is True
    assert config.paper_metadata[current.arxiv_id].tags == ["priority"]
    exported = format_papers_as_markdown_table([current])
    assert current.title in exported
    assert current.arxiv_id in exported


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acceptance_arxiv_api_mode_uses_fake_service_and_drops_stale_results(
    tmp_path, monkeypatch, make_paper
) -> None:
    _patch_cache_db(monkeypatch, tmp_path)
    local_paper = make_paper(arxiv_id="2401.00001", title="Local Paper")
    fake_api = _FakeArxivApiService(fetched_pages=[])
    config = _config(tmp_path)
    app = ArxivBrowser(
        [local_paper],
        options=ArxivBrowserOptions(
            config=config,
            restore_session=False,
            services=_fake_services(arxiv_api=fake_api),
        ),
    )
    fake_api.app_ref = app

    with patch_save_config(return_value=True):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            request = ArxivSearchRequest(query="transformer", field="title", category="cs.AI")
            await app._run_arxiv_search(request, 0)
            await pilot.pause()
            assert app._in_arxiv_api_mode is True
            assert app.all_papers[0].source == "api"
            assert fake_api.fetched_pages == [(request, 0, config.arxiv_api_max_results)]

            app.action_exit_arxiv_search_mode()
            assert app.all_papers == [local_paper]

            fake_api.stale_during_fetch = True
            await app._run_arxiv_search(request, 0)
            await pilot.pause()
            assert app.all_papers == [local_paper]
            assert app._in_arxiv_api_mode is False


@pytest.mark.asyncio
async def test_acceptance_llm_batch_fake_service_success_failure_and_cancel(make_paper) -> None:
    papers = [make_paper(arxiv_id=f"2401.0000{i}", title=f"Paper {i}") for i in range(3)]
    llm = _FakeLlmService(fail_second=True)
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._config = UserConfig(llm_timeout=1)
    app._services = _fake_services(llm=llm)
    app._llm_provider = object()
    app._relevance_db_path = None
    app._relevance_scores = {}
    app._cancel_batch_requested = False
    app._relevance_scoring_active = True
    app._scoring_progress = None
    app._dataset_epoch = 0
    app._shutting_down = False
    app.notify = MagicMock()
    app._mark_badges_dirty = MagicMock()
    app._refresh_detail_pane = MagicMock()
    app._update_relevance_badge = MagicMock()
    app._update_footer = MagicMock()

    await ArxivBrowser._score_relevance_batch_async(app, papers, "fake {prompt}", "systems")

    assert llm.scored == [paper.arxiv_id for paper in papers]
    assert set(app._relevance_scores) == {papers[0].arxiv_id, papers[2].arxiv_id}
    assert app._relevance_scoring_active is False
    assert app._scoring_progress is None

    cancelled_app = ArxivBrowser.__new__(ArxivBrowser)
    cancelled_app.__dict__.update(app.__dict__)
    cancelled_app._services = _fake_services(llm=_FakeLlmService())
    cancelled_app._relevance_scores = {}
    cancelled_app._cancel_batch_requested = True
    cancelled_app._relevance_scoring_active = True
    await ArxivBrowser._score_relevance_batch_async(
        cancelled_app, papers, "fake {prompt}", "systems"
    )
    assert cancelled_app._relevance_scores == {}
    assert cancelled_app._cancel_batch_requested is False


@pytest.mark.asyncio
async def test_dataset_epoch_prevents_stale_background_mutation(make_paper) -> None:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._dataset_epoch = 0
    app._dataset_tasks = set()
    app._background_tasks = set()
    app._shutting_down = False
    app._on_task_done = MagicMock()
    mutated: list[str] = []

    async def stale_writer() -> None:
        epoch = app._capture_dataset_epoch()
        await asyncio.sleep(0)
        if app._is_current_dataset_epoch(epoch):
            mutated.append("stale")

    task = app._track_dataset_task(stale_writer())
    app._advance_dataset_epoch()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mutated == []
    assert app._dataset_epoch == 1


def test_parser_and_query_regression_cases_are_stable() -> None:
    malformed_feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2601.00001v2</id>
        <title>  A   Realistic\n        Title  </title>
        <author><name>Alice</name></author>
        <category term="cs.AI"/>
        <summary>Abstract with\n whitespace.</summary>
      </entry>
      <entry><id></id><title>missing id</title></entry>
    </feed>
    """
    papers = parse_arxiv_api_feed(malformed_feed)
    assert [paper.arxiv_id for paper in papers] == ["2601.00001"]
    assert papers[0].title == "A Realistic Title"

    query = 'cat:cs.AI AND title:"large language" OR unread'
    tokens = tokenize_query(query)
    assert reconstruct_query(tokens, -1) == query
