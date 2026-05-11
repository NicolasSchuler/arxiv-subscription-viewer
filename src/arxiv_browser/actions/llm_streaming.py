"""Streaming helpers for LLM summary actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arxiv_browser.llm import build_llm_prompt
from arxiv_browser.llm_providers import LLMProvider
from arxiv_browser.models import Paper

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser


def should_stream_summary(app: ArxivBrowser, provider: LLMProvider) -> bool:
    return bool(
        getattr(app._config, "llm_streaming_enabled", False)
        and callable(getattr(provider, "execute_stream", None))
    )


async def request_summary_streaming(
    app: ArxivBrowser,
    *,
    paper: Paper,
    prompt_template: str,
    provider: LLMProvider,
    use_full_paper_content: bool,
) -> tuple[str | None, str | None]:
    if use_full_paper_content:
        paper_content = await app._fetch_paper_content_async(paper)
    else:
        abstract = paper.abstract or paper.abstract_raw or ""
        paper_content = f"Abstract:\n{abstract}" if abstract else ""

    prompt = build_llm_prompt(paper, prompt_template, paper_content)
    parts: list[str] = []
    async for chunk in provider.execute_stream(prompt, app._config.llm_timeout):
        if chunk.error:
            return None, chunk.error
        if chunk.delta:
            parts.append(chunk.delta)
            app._paper_summaries[paper.arxiv_id] = "".join(parts)
            app._update_abstract_display(paper.arxiv_id)
        if chunk.done:
            break
    summary = "".join(parts).strip()
    if not summary:
        return None, "Empty response content"
    return summary, None


__all__ = ["request_summary_streaming", "should_stream_summary"]
