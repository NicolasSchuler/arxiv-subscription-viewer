"""Helpers for enrichment-state bookkeeping used by the app orchestration layer."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from arxiv_browser.models import PaperMetadata


def count_hf_matches(
    hf_cache: Mapping[str, object],
    papers_by_id: Mapping[str, object],
) -> int:
    """Count HuggingFace entries that correspond to loaded paper IDs."""
    return sum(1 for arxiv_id in hf_cache if arxiv_id in papers_by_id)


def get_starred_paper_ids_for_version_check(
    paper_metadata: Mapping[str, PaperMetadata],
) -> set[str]:
    """Collect starred paper IDs eligible for version checks."""
    return {arxiv_id for arxiv_id, meta in paper_metadata.items() if meta.starred}


def apply_version_updates(
    version_map: Mapping[str, int],
    paper_metadata: MutableMapping[str, PaperMetadata],
    version_updates: MutableMapping[str, tuple[int, int]],
) -> int:
    """Apply fetched arXiv versions to metadata and track detected upgrades."""
    updates_found = 0
    for arxiv_id, new_version in version_map.items():
        meta = paper_metadata.get(arxiv_id)
        if meta is None or not meta.starred:
            continue
        old_version = meta.last_checked_version
        if old_version is not None and new_version > old_version:
            version_updates[arxiv_id] = (old_version, new_version)
            updates_found += 1
        meta.last_checked_version = new_version
    return updates_found


__all__ = [
    "apply_version_updates",
    "count_hf_matches",
    "get_starred_paper_ids_for_version_check",
]
