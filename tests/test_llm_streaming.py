"""Focused tests for LLM summary streaming helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from arxiv_browser.actions import llm_actions
from arxiv_browser.actions.llm_streaming import request_summary_streaming, should_stream_summary
from arxiv_browser.llm_providers import LLMChunk, LLMResult
from arxiv_browser.models import UserConfig


class _StreamingProvider:
    def __init__(self, chunks: list[LLMChunk]) -> None:
        self._chunks = chunks
        self.prompt = ""
        self.timeout = 0

    async def execute_stream(self, prompt: str, timeout: int) -> AsyncIterator[LLMChunk]:
        self.prompt = prompt
        self.timeout = timeout
        for chunk in self._chunks:
            yield chunk

    async def execute(self, _prompt: str, _timeout: int) -> LLMResult:
        raise AssertionError("streaming tests should not use single-shot execute")


def _streaming_app(*, enabled: bool = True, timeout: int = 12) -> Any:
    return SimpleNamespace(
        _config=UserConfig(llm_streaming_enabled=enabled, llm_timeout=timeout),
        _fetch_paper_content_async=AsyncMock(return_value="full paper content"),
        _paper_summaries={},
        _update_abstract_display=MagicMock(),
    )


def test_should_stream_summary_requires_enabled_config_and_provider_method() -> None:
    provider = _StreamingProvider([])
    assert should_stream_summary(_streaming_app(enabled=True), provider) is True
    assert should_stream_summary(_streaming_app(enabled=False), provider) is False
    assert should_stream_summary(_streaming_app(enabled=True), object()) is False


@pytest.mark.asyncio
async def test_request_summary_streaming_uses_abstract_when_full_content_disabled(
    make_paper,
) -> None:
    paper = make_paper(title="Abstract-only stream", abstract="A concise abstract")
    app = _streaming_app(timeout=7)
    provider = _StreamingProvider(
        [
            LLMChunk(delta="first "),
            LLMChunk(delta="second"),
            LLMChunk(done=True),
        ]
    )

    summary, error = await request_summary_streaming(
        app,
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
    )

    assert (summary, error) == ("first second", None)
    assert "Abstract-only stream" in provider.prompt
    assert "Abstract:\nA concise abstract" in provider.prompt
    assert provider.timeout == 7
    app._fetch_paper_content_async.assert_not_awaited()
    assert app._paper_summaries[paper.arxiv_id] == "first second"
    assert app._update_abstract_display.call_count == 2


@pytest.mark.asyncio
async def test_request_summary_streaming_returns_provider_error(make_paper) -> None:
    paper = make_paper(abstract="The fallback abstract")
    app = _streaming_app()
    provider = _StreamingProvider([LLMChunk(error="stream failed")])

    summary, error = await request_summary_streaming(
        app,
        paper=paper,
        prompt_template="{paper_content}",
        provider=provider,
        use_full_paper_content=False,
    )

    assert (summary, error) == (None, "stream failed")
    assert "Abstract:\nThe fallback abstract" in provider.prompt
    assert app._paper_summaries == {}
    app._update_abstract_display.assert_not_called()


@pytest.mark.asyncio
async def test_request_summary_streaming_rejects_empty_stream(make_paper) -> None:
    paper = make_paper(abstract="")
    app = _streaming_app()
    provider = _StreamingProvider([LLMChunk(done=True)])

    summary, error = await request_summary_streaming(
        app,
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
    )

    assert (summary, error) == (None, "Empty response content")
    assert provider.prompt
    assert app._paper_summaries == {}
    app._update_abstract_display.assert_not_called()


@pytest.mark.asyncio
async def test_generate_summary_streaming_error_after_partial_clears_partial_state(
    make_paper,
    tmp_path,
    monkeypatch,
) -> None:
    paper = make_paper(arxiv_id="2401.88001", abstract="Fallback abstract")
    provider = _StreamingProvider(
        [
            LLMChunk(delta="draft "),
            LLMChunk(error="stream broke"),
        ]
    )
    app = SimpleNamespace(
        _config=UserConfig(llm_streaming_enabled=True, llm_timeout=9),
        _llm_provider=provider,
        _summary_db_path=tmp_path / "summaries.db",
        _paper_summaries={},
        _summary_loading={paper.arxiv_id},
        _summary_mode_label={paper.arxiv_id: "Detailed"},
        _summary_command_hash={paper.arxiv_id: "hash"},
        _capture_dataset_epoch=MagicMock(return_value=1),
        _is_current_dataset_epoch=MagicMock(return_value=True),
        _fetch_paper_content_async=AsyncMock(return_value="full paper content"),
        _update_abstract_display=MagicMock(),
        notify=MagicMock(),
    )
    save_summary = MagicMock()
    monkeypatch.setattr(llm_actions, "_save_summary", save_summary)

    await llm_actions._generate_summary_async(
        app,
        paper,
        "{title}\n{paper_content}",
        "hash",
        mode_label="Detailed",
        use_full_paper_content=True,
    )

    assert app._paper_summaries == {}
    assert app._summary_loading == set()
    assert app._summary_mode_label == {}
    assert app._summary_command_hash == {}
    save_summary.assert_not_called()
    assert "stream broke" in app.notify.call_args.args[0]
    assert app._update_abstract_display.call_count >= 2
