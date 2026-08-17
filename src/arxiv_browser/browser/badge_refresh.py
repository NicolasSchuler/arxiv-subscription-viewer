"""Cache-backed badge refresh and sort-sensitivity helpers."""

from __future__ import annotations

from typing import Any

BADGE_REFRESH_KINDS = frozenset({"s2", "hf", "version", "relevance", "judge", "triage"})


def badge_refresh_ids(app: Any, kind: str) -> tuple[set[str], bool]:
    """Return affected paper IDs and whether the whole visible list is dirty."""
    if kind == "s2":
        return (set(app._s2_cache), False) if app._s2_active else (set(), True)
    if kind == "hf":
        return (set(app._hf_cache), False) if app._hf_active else (set(), True)
    if kind == "version":
        return set(app._version_updates), False
    if kind == "relevance":
        return set(app._relevance_scores), False
    if kind == "judge":
        return set(app._judge_scores), False
    if kind == "triage":
        return set(getattr(app, "_triage_predictions", {})), False
    return set(), True


def is_sort_sensitive_badge(kind: str, sort_key: str) -> bool:
    """Return whether changing a badge signal can change the active ordering."""
    direct_sort = {
        "s2": "citations",
        "hf": "trending",
        "relevance": "relevance",
        "judge": "impact",
        "triage": "triage",
    }
    return direct_sort.get(kind) == sort_key or (
        sort_key == "queue" and kind in {"s2", "hf", "relevance", "judge"}
    )


__all__ = ["BADGE_REFRESH_KINDS", "badge_refresh_ids", "is_sort_sensitive_badge"]
