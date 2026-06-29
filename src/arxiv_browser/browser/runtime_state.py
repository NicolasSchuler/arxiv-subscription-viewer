"""Small runtime-state containers used by :mod:`arxiv_browser.browser.core`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LlmApiRuntimeState:
    """Mutable runtime state for LLM caches, API browsing, and shared HTTP."""

    cache_db_path: Path
    llm_provider: Any
    paper_summaries: dict[str, str] = field(default_factory=dict)
    summary_loading: set[str] = field(default_factory=set)
    summary_mode_label: dict[str, str] = field(default_factory=dict)
    summary_command_hash: dict[str, str] = field(default_factory=dict)
    in_arxiv_api_mode: bool = False
    arxiv_search_state: Any = None
    local_browse_snapshot: Any = None
    arxiv_api_fetch_inflight: bool = False
    arxiv_api_loading: bool = False
    last_arxiv_api_request_at: float = 0.0
    arxiv_api_request_token: int = 0
    http_client: Any = None

    @property
    def summary_db_path(self) -> Path:
        return self.cache_db_path


@dataclass(slots=True)
class EnrichmentScoringRuntimeState:
    """Mutable runtime state for enrichment, scoring, and local similarity."""

    cache_db_path: Path
    s2_active: bool = False
    s2_cache: dict[str, Any] = field(default_factory=dict)
    s2_loading: set[str] = field(default_factory=set)
    s2_api_error: bool = False
    hf_active: bool = False
    hf_cache: dict[str, Any] = field(default_factory=dict)
    hf_loading: bool = False
    hf_api_error: bool = False
    version_updates: dict[str, tuple[int, int]] = field(default_factory=dict)
    version_checking: bool = False
    version_progress: Any = None
    relevance_scores: dict[str, tuple[int, str]] = field(default_factory=dict)
    relevance_scoring_active: bool = False
    scoring_progress: tuple[int, int] | None = None
    auto_tag_active: bool = False
    auto_tag_progress: tuple[int, int] | None = None
    paper_remix_active: bool = False
    cancel_batch_requested: bool = False
    detail_focus_active: bool = False
    read_event_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=240))
    tfidf_index: Any = None
    tfidf_corpus_key: str | None = None
    tfidf_build_task: Any = None
    pending_similarity_paper_id: str | None = None
    semantic_search_worker: Any = None
    semantic_search_token: int = 0

    @property
    def s2_db_path(self) -> Path:
        return self.cache_db_path

    @property
    def hf_db_path(self) -> Path:
        return self.cache_db_path

    @property
    def relevance_db_path(self) -> Path:
        return self.cache_db_path


def attach_llm_api_runtime(app: Any, state: LlmApiRuntimeState) -> None:
    """Expose LLM/API runtime state through the legacy app attribute surface."""
    app._paper_summaries = state.paper_summaries
    app._summary_loading = state.summary_loading
    app._cache_db_path = state.cache_db_path
    app._summary_db_path = state.summary_db_path
    app._summary_mode_label = state.summary_mode_label
    app._summary_command_hash = state.summary_command_hash
    app._in_arxiv_api_mode = state.in_arxiv_api_mode
    app._arxiv_search_state = state.arxiv_search_state
    app._local_browse_snapshot = state.local_browse_snapshot
    app._arxiv_api_fetch_inflight = state.arxiv_api_fetch_inflight
    app._arxiv_api_loading = state.arxiv_api_loading
    app._last_arxiv_api_request_at = state.last_arxiv_api_request_at
    app._arxiv_api_request_token = state.arxiv_api_request_token
    app._http_client = state.http_client
    app._llm_provider = state.llm_provider


def attach_enrichment_scoring_runtime(app: Any, state: EnrichmentScoringRuntimeState) -> None:
    """Expose enrichment/scoring runtime state through legacy app attributes."""
    app._s2_active = state.s2_active
    app._s2_cache = state.s2_cache
    app._s2_loading = state.s2_loading
    app._s2_db_path = state.s2_db_path
    app._s2_api_error = state.s2_api_error
    app._hf_active = state.hf_active
    app._hf_cache = state.hf_cache
    app._hf_loading = state.hf_loading
    app._hf_db_path = state.hf_db_path
    app._hf_api_error = state.hf_api_error
    app._version_updates = state.version_updates
    app._version_checking = state.version_checking
    app._version_progress = state.version_progress
    app._relevance_scores = state.relevance_scores
    app._relevance_scoring_active = state.relevance_scoring_active
    app._scoring_progress = state.scoring_progress
    app._relevance_db_path = state.relevance_db_path
    app._auto_tag_active = state.auto_tag_active
    app._auto_tag_progress = state.auto_tag_progress
    app._paper_remix_active = state.paper_remix_active
    app._cancel_batch_requested = state.cancel_batch_requested
    app._detail_focus_active = state.detail_focus_active
    app._read_event_timestamps = state.read_event_timestamps
    app._tfidf_index = state.tfidf_index
    app._tfidf_corpus_key = state.tfidf_corpus_key
    app._tfidf_build_task = state.tfidf_build_task
    app._pending_similarity_paper_id = state.pending_similarity_paper_id
    app._semantic_search_worker = state.semantic_search_worker
    app._semantic_search_token = state.semantic_search_token


__all__ = [
    "EnrichmentScoringRuntimeState",
    "LlmApiRuntimeState",
    "attach_enrichment_scoring_runtime",
    "attach_llm_api_runtime",
]
