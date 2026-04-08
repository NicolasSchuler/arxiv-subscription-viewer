"""Shared action-layer constants and logging."""

from __future__ import annotations

import logging

logger = logging.getLogger("arxiv_browser.actions")
SUBPROCESS_TIMEOUT = 5
BATCH_CONFIRM_THRESHOLD = 10
BOOKMARK_NAME_MAX_LEN = 15
MAX_CONCURRENT_DOWNLOADS = 3
CLIPBOARD_SEPARATOR = "=" * 80

__all__ = [
    "BATCH_CONFIRM_THRESHOLD",
    "BOOKMARK_NAME_MAX_LEN",
    "CLIPBOARD_SEPARATOR",
    "MAX_CONCURRENT_DOWNLOADS",
    "SUBPROCESS_TIMEOUT",
    "logger",
]
