# pyright: reportMissingImports=false
"""Optional fuzzy matching helpers."""

from __future__ import annotations

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ImportError:
    _rapidfuzz_fuzz = None


def weighted_fuzzy_score(query: str, text: str) -> float:
    """Return a WRatio-like score without requiring rapidfuzz."""
    query_norm = _normalize(query)
    text_norm = _normalize(text)
    if not query_norm or not text_norm:
        return 0.0
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.WRatio(query_norm, text_norm))
    return max(_ratio(query_norm, text_norm), _fallback_partial_ratio(query_norm, text_norm))


def partial_fuzzy_score(query: str, text: str) -> float:
    """Return a partial-ratio-like score without requiring rapidfuzz."""
    query_norm = _normalize(query)
    text_norm = _normalize(text)
    if not query_norm or not text_norm:
        return 0.0
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.partial_ratio(query_norm, text_norm))
    return _fallback_partial_ratio(query_norm, text_norm)


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _ratio(query: str, text: str) -> float:
    return SequenceMatcher(None, query, text).ratio() * 100.0


def _fallback_partial_ratio(query: str, text: str) -> float:
    shorter, longer = (query, text) if len(query) <= len(text) else (text, query)
    if shorter in longer:
        return 100.0

    window_len = len(shorter)
    if window_len == 0:
        return 0.0

    best = 0.0
    last_start = len(longer) - window_len
    for start in range(last_start + 1):
        score = _ratio(shorter, longer[start : start + window_len])
        if score > best:
            best = score
    return best
