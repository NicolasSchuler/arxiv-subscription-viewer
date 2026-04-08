"""Shared browser-layer constants and logging."""

from __future__ import annotations

import logging

logger = logging.getLogger("arxiv_browser.browser")

FUZZY_SCORE_CUTOFF = 60
FUZZY_LIMIT = 100
MAX_ABSTRACT_LOADS = 32
BADGE_COALESCE_DELAY = 0.05
PDF_DOWNLOAD_TIMEOUT = 60

__all__ = [
    "BADGE_COALESCE_DELAY",
    "FUZZY_LIMIT",
    "FUZZY_SCORE_CUTOFF",
    "MAX_ABSTRACT_LOADS",
    "PDF_DOWNLOAD_TIMEOUT",
    "logger",
]
