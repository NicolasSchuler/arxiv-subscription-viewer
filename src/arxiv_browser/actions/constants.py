"""Shared action-layer constants and logging."""

from __future__ import annotations

import logging

from arxiv_browser.action_messages import build_actionable_error

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

# Shared actionable-error copy for LLM-backed actions (finding #8).
LLM_ERROR_WHY = "the LLM command failed or timed out"
LLM_ERROR_NEXT_STEP = "check llm_command in config or run arxiv-viewer doctor, then retry"
SUMMARY_FAILED_MESSAGE = build_actionable_error(
    "generate the AI summary", why=LLM_ERROR_WHY, next_step=LLM_ERROR_NEXT_STEP
)
UNKNOWN_SUMMARY_MODE_MESSAGE = build_actionable_error(
    "generate the AI summary",
    why="an unrecognized summary mode was selected",
    next_step="pick a mode from the summary menu, then retry",
)
RELEVANCE_FAILED_MESSAGE = build_actionable_error(
    "score paper relevance", why=LLM_ERROR_WHY, next_step=LLM_ERROR_NEXT_STEP
)
AUTO_TAG_FAILED_MESSAGE = build_actionable_error(
    "auto-tag the paper(s)", why=LLM_ERROR_WHY, next_step=LLM_ERROR_NEXT_STEP
)


def build_summary_config_error(why: str) -> str:
    """Build an actionable AI-summary error carrying a specific reason."""
    return build_actionable_error("generate the AI summary", why=why, next_step=LLM_ERROR_NEXT_STEP)


def build_auto_tag_failure(tagged: int) -> str:
    """Build the batch auto-tag failure message, noting partial progress."""
    if tagged:
        why = f"{tagged} paper(s) were tagged before {LLM_ERROR_WHY}"
        return build_actionable_error("auto-tag the papers", why=why, next_step=LLM_ERROR_NEXT_STEP)
    return AUTO_TAG_FAILED_MESSAGE


def log_action_failure(action: str, exc: Exception, *, unexpected: bool = False) -> None:
    """Log an action failure with a consistent message shape."""
    qualifier = "Unexpected " if unexpected else ""
    message = f"{qualifier}{action} failed ({type(exc).__name__}): {exc}"
    logger.warning(message, exc_info=True)


__all__ = [
    "AUTO_TAG_FAILED_MESSAGE",
    "BATCH_CONFIRM_THRESHOLD",
    "BOOKMARK_NAME_MAX_LEN",
    "CLIPBOARD_SEPARATOR",
    "LLM_ERROR_NEXT_STEP",
    "LLM_ERROR_WHY",
    "MAX_CONCURRENT_DOWNLOADS",
    "RECOVERABLE_ACTION_ERRORS",
    "RELEVANCE_FAILED_MESSAGE",
    "SUBPROCESS_TIMEOUT",
    "SUMMARY_FAILED_MESSAGE",
    "UNKNOWN_SUMMARY_MODE_MESSAGE",
    "build_auto_tag_failure",
    "build_summary_config_error",
    "log_action_failure",
    "logger",
]
