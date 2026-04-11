"""Shared action-layer constants and logging."""

from __future__ import annotations

import logging

logger = logging.getLogger("arxiv_browser.actions")
SUBPROCESS_TIMEOUT = 5
BATCH_CONFIRM_THRESHOLD = 10
BOOKMARK_NAME_MAX_LEN = 15
MAX_CONCURRENT_DOWNLOADS = 3
CLIPBOARD_SEPARATOR = "=" * 80

RECOVERABLE_ACTION_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
)


def log_action_failure(action: str, exc: Exception, *, unexpected: bool = False) -> None:
    """Log an action failure with a consistent message shape."""
    qualifier = "Unexpected " if unexpected else ""
    message = f"{qualifier}{action} failed ({type(exc).__name__}): {exc}"
    logger.warning(message, exc_info=True)


__all__ = [
    "BATCH_CONFIRM_THRESHOLD",
    "BOOKMARK_NAME_MAX_LEN",
    "CLIPBOARD_SEPARATOR",
    "MAX_CONCURRENT_DOWNLOADS",
    "RECOVERABLE_ACTION_ERRORS",
    "SUBPROCESS_TIMEOUT",
    "log_action_failure",
    "logger",
]
