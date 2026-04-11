"""Service interfaces + default adapters for app-level dependency injection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from arxiv_browser.llm_providers import LLMProvider
from arxiv_browser.models import ArxivSearchRequest, Paper, UserConfig
from arxiv_browser.services import arxiv_api_service as _arxiv_api
from arxiv_browser.services import download_service as _download
from arxiv_browser.services import enrichment_service as _enrichment
from arxiv_browser.services import llm_service as _llm
from arxiv_browser.services.download_service import DownloadResult


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
        client: httpx.AsyncClient,
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
        client: httpx.AsyncClient,
        timeout_seconds: int,
    ) -> DownloadResult: ...


@runtime_checkable
class EnrichmentService(Protocol):
    """Interface for enrichment/cache-backed app operations."""

    async def load_or_fetch_s2_paper(
        self,
        *,
        arxiv_id: str,
        db_path: Path,
        cache_ttl_days: int,
        client: httpx.AsyncClient,
        api_key: str,
    ) -> _enrichment.S2PaperFetchResult:
        """Load or fetch one S2 paper, preserving not-found state."""
        ...

    async def load_or_fetch_hf_daily(
        self,
        *,
        db_path: Path,
        cache_ttl_hours: int,
        client: httpx.AsyncClient,
    ) -> _enrichment.HFDailyFetchResult:
        """Load or fetch the HF daily snapshot, preserving empty state."""
        ...

    async def load_or_fetch_s2_recommendations(
        self,
        *,
        arxiv_id: str,
        db_path: Path,
        cache_ttl_days: int,
        client: httpx.AsyncClient,
        api_key: str,
    ) -> _enrichment.S2RecommendationsFetchResult:
        """Load or fetch S2 recommendations, preserving empty state."""
        ...


class DefaultArxivApiService:
    """Default adapter that delegates to function-based arXiv API services."""

    def format_query_label(self, request: ArxivSearchRequest) -> str:
        """Build a user-facing arXiv API query label."""
        return _arxiv_api.format_query_label(request)

    async def enforce_rate_limit(
        self,
        *,
        last_request_at: float,
        min_interval_seconds: float,
        now: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> tuple[float, float]:
        """Enforce API rate limiting and return (new_timestamp, wait_seconds)."""
        return await _arxiv_api.enforce_rate_limit(
            last_request_at=last_request_at,
            min_interval_seconds=min_interval_seconds,
            now=now,
            sleep=sleep,
        )

    async def fetch_page(
        self,
        *,
        client: httpx.AsyncClient,
        request: ArxivSearchRequest,
        start: int,
        max_results: int,
        timeout_seconds: int,
        user_agent: str,
    ) -> list[Paper]:
        """Fetch a page of arXiv API results."""
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
        """Generate a summary and return (summary, error)."""
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
        """Score one paper for relevance."""
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
        """Suggest tags for one paper."""
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
        client: httpx.AsyncClient,
        timeout_seconds: int,
    ) -> DownloadResult:
        return await _download.download_pdf(
            paper=paper,
            config=config,
            client=client,
            timeout_seconds=timeout_seconds,
        )


class DefaultEnrichmentService:
    """Default adapter that delegates to enrichment/cache services."""

    async def load_or_fetch_s2_paper(
        self,
        *,
        arxiv_id: str,
        db_path: Path,
        cache_ttl_days: int,
        client: httpx.AsyncClient,
        api_key: str,
    ) -> _enrichment.S2PaperFetchResult:
        """Load or fetch one S2 paper, preserving not-found state."""
        return await _enrichment.load_or_fetch_s2_paper_result(
            arxiv_id=arxiv_id,
            db_path=db_path,
            cache_ttl_days=cache_ttl_days,
            client=client,
            api_key=api_key,
        )

    async def load_or_fetch_hf_daily(
        self,
        *,
        db_path: Path,
        cache_ttl_hours: int,
        client: httpx.AsyncClient,
    ) -> _enrichment.HFDailyFetchResult:
        """Load or fetch the HF daily snapshot, preserving empty state."""
        return await _enrichment.load_or_fetch_hf_daily_result(
            db_path=db_path,
            cache_ttl_hours=cache_ttl_hours,
            client=client,
        )

    async def load_or_fetch_s2_recommendations(
        self,
        *,
        arxiv_id: str,
        db_path: Path,
        cache_ttl_days: int,
        client: httpx.AsyncClient,
        api_key: str,
    ) -> _enrichment.S2RecommendationsFetchResult:
        """Load or fetch S2 recommendations, preserving empty state."""
        return await _enrichment.load_or_fetch_s2_recommendations_result(
            arxiv_id=arxiv_id,
            db_path=db_path,
            cache_ttl_days=cache_ttl_days,
            client=client,
            api_key=api_key,
        )


@dataclass(slots=True)
class AppServices:
    """Aggregated service interfaces consumed by the app layer."""

    arxiv_api: ArxivApiService
    llm: LlmService
    download: DownloadService
    enrichment: EnrichmentService


def build_default_app_services() -> AppServices:
    """Build default app services backed by existing function-based modules."""
    return AppServices(
        arxiv_api=DefaultArxivApiService(),
        llm=DefaultLlmService(),
        download=DefaultDownloadService(),
        enrichment=DefaultEnrichmentService(),
    )


__all__ = [
    "AppServices",
    "ArxivApiService",
    "DownloadService",
    "EnrichmentService",
    "LlmService",
    "build_default_app_services",
]
