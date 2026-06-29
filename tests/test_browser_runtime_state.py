from __future__ import annotations

from types import SimpleNamespace

from arxiv_browser.browser.runtime_state import (
    EnrichmentScoringRuntimeState,
    LlmApiRuntimeState,
    attach_enrichment_scoring_runtime,
    attach_llm_api_runtime,
)


def test_runtime_state_defaults_share_cache_paths(tmp_path) -> None:
    cache_path = tmp_path / "cache.db"
    llm_state = LlmApiRuntimeState(cache_db_path=cache_path, llm_provider=object())
    enrichment_state = EnrichmentScoringRuntimeState(cache_db_path=cache_path)

    assert llm_state.summary_db_path == cache_path
    assert enrichment_state.s2_db_path == cache_path
    assert enrichment_state.hf_db_path == cache_path
    assert enrichment_state.relevance_db_path == cache_path
    assert enrichment_state.read_event_timestamps.maxlen == 240


def test_runtime_state_attach_helpers_preserve_legacy_attribute_surface(tmp_path) -> None:
    cache_path = tmp_path / "cache.db"
    provider = object()
    app = SimpleNamespace()
    llm_state = LlmApiRuntimeState(cache_db_path=cache_path, llm_provider=provider)
    enrichment_state = EnrichmentScoringRuntimeState(cache_db_path=cache_path)

    attach_llm_api_runtime(app, llm_state)
    attach_enrichment_scoring_runtime(app, enrichment_state)

    assert app._cache_db_path == cache_path
    assert app._summary_db_path == cache_path
    assert app._paper_summaries is llm_state.paper_summaries
    assert app._llm_provider is provider
    assert app._s2_db_path == cache_path
    assert app._hf_db_path == cache_path
    assert app._relevance_db_path == cache_path
    assert app._read_event_timestamps is enrichment_state.read_event_timestamps
