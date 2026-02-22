"""Service interfaces + default adapters for app-level dependency injection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from arxiv_browser.llm_providers import LLMProvider
from arxiv_browser.models import ArxivSearchRequest, Paper, UserConfig
from arxiv_browser.services import arxiv_api_service as _arxiv_api
from arxiv_browser.services import download_service as _download
from arxiv_browser.services import llm_service as _llm


@runtime_checkable
class ArxivApiService(Protocol):
    """Interface for arXiv API-related app operations."""

    def format_query_label(self, request: ArxivSearchRequest) -> str:
        """Build a user-facing arXiv API query label."""
        ...

    async def enforce_rate_limit(
        self,
        *,
        last_request_at: float,
        min_interval_seconds: float,
        now: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> tuple[float, float]:
        """Enforce API rate limiting and return (new_timestamp, wait_seconds)."""
        ...

    async def fetch_page(
        self,
        *,
        client: httpx.AsyncClient | None,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
        timeout_seconds: int,
        user_agent: str,
    ) -> list[Paper]:
        """Fetch a page of arXiv API results."""
        ...


@runtime_checkable
class LlmService(Protocol):
    """Interface for LLM summary/relevance/auto-tag operations."""

    async def generate_summary(
        self,
        *,
        paper: Paper,
        prompt_template: str,
        provider: LLMProvider,
        use_full_paper_content: bool,
        summary_timeout_seconds: int,
        fetch_paper_content: Callable[[Paper], Awaitable[str]],
    ) -> tuple[str | None, str | None]:
        """Generate a summary and return (summary, error)."""
        ...

    async def score_relevance_once(
        self,
        *,
        paper: Paper,
        interests: str,
        provider: LLMProvider,
        timeout_seconds: int,
    ) -> tuple[int, str] | None:
        """Score one paper for relevance."""
        ...

    async def suggest_tags_once(
        self,
        *,
        paper: Paper,
        taxonomy: list[str],
        provider: LLMProvider,
        timeout_seconds: int,
    ) -> list[str] | None:
        """Suggest tags for one paper."""
        ...


@runtime_checkable
class DownloadService(Protocol):
    """Interface for PDF download app operations."""

    async def download_pdf(
        self,
        *,
        paper: Paper,
        config: UserConfig,
        client: httpx.AsyncClient | None,
        timeout_seconds: int,
    ) -> bool:
        """Download a paper PDF and return success."""
        ...


class DefaultArxivApiService:
    """Default adapter that delegates to function-based arXiv API services."""

    def format_query_label(self, request: ArxivSearchRequest) -> str:
        return _arxiv_api.format_query_label(request)

    async def enforce_rate_limit(
        self,
        *,
        last_request_at: float,
        min_interval_seconds: float,
        now: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> tuple[float, float]:
        return await _arxiv_api.enforce_rate_limit(
            last_request_at=last_request_at,
            min_interval_seconds=min_interval_seconds,
            now=now,
            sleep=sleep,
        )

    async def fetch_page(
        self,
        *,
        client: httpx.AsyncClient | None,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
        timeout_seconds: int,
        user_agent: str,
    ) -> list[Paper]:
        return await _arxiv_api.fetch_page(
            client=client,
            request=request,
            start=start,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
        )


class DefaultLlmService:
    """Default adapter that delegates to function-based LLM services."""

    async def generate_summary(
        self,
        *,
        paper: Paper,
        prompt_template: str,
        provider: LLMProvider,
        use_full_paper_content: bool,
        summary_timeout_seconds: int,
        fetch_paper_content: Callable[[Paper], Awaitable[str]],
    ) -> tuple[str | None, str | None]:
        return await _llm.generate_summary(
            paper=paper,
            prompt_template=prompt_template,
            provider=provider,
            use_full_paper_content=use_full_paper_content,
            summary_timeout_seconds=summary_timeout_seconds,
            fetch_paper_content=fetch_paper_content,
        )

    async def score_relevance_once(
        self,
        *,
        paper: Paper,
        interests: str,
        provider: LLMProvider,
        timeout_seconds: int,
    ) -> tuple[int, str] | None:
        return await _llm.score_relevance_once(
            paper=paper,
            interests=interests,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )

    async def suggest_tags_once(
        self,
        *,
        paper: Paper,
        taxonomy: list[str],
        provider: LLMProvider,
        timeout_seconds: int,
    ) -> list[str] | None:
        return await _llm.suggest_tags_once(
            paper=paper,
            taxonomy=taxonomy,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )


class DefaultDownloadService:
    """Default adapter that delegates to function-based download services."""

    async def download_pdf(
        self,
        *,
        paper: Paper,
        config: UserConfig,
        client: httpx.AsyncClient | None,
        timeout_seconds: int,
    ) -> bool:
        return await _download.download_pdf(
            paper=paper,
            config=config,
            client=client,
            timeout_seconds=timeout_seconds,
        )


@dataclass(slots=True)
class AppServices:
    """Aggregated service interfaces consumed by the app layer."""

    arxiv_api: ArxivApiService
    llm: LlmService
    download: DownloadService


def build_default_app_services() -> AppServices:
    """Build default app services backed by existing function-based modules."""
    return AppServices(
        arxiv_api=DefaultArxivApiService(),
        llm=DefaultLlmService(),
        download=DefaultDownloadService(),
    )


__all__ = [
    "AppServices",
    "ArxivApiService",
    "DownloadService",
    "LlmService",
    "build_default_app_services",
]
