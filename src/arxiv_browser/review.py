"""Spaced-review scheduling helpers for paper metadata."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from arxiv_browser.models import PaperMetadata

REVIEW_INTERVAL_DAYS: tuple[int, ...] = (1, 3, 7, 14, 30)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_review_date(value: str | None) -> date | None:
    """Parse a strict ISO ``YYYY-MM-DD`` review date."""
    if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def normalize_review_schedule(
    next_review_date: Any,
    review_stage: Any,
) -> tuple[str | None, int | None]:
    """Return sanitized persisted review fields.

    A valid due date activates the schedule. A missing or invalid stage falls
    back to the first interval so old or hand-edited configs remain usable.
    """
    parsed_date = parse_review_date(next_review_date)
    if parsed_date is None:
        return None, None
    if isinstance(review_stage, bool) or not isinstance(review_stage, int):
        return parsed_date.isoformat(), 0
    if review_stage < 0 or review_stage >= len(REVIEW_INTERVAL_DAYS):
        return parsed_date.isoformat(), 0
    return parsed_date.isoformat(), review_stage


def is_review_due(metadata: PaperMetadata | None, today: date) -> bool:
    """Return whether a paper has an active review schedule due by *today*."""
    if metadata is None:
        return False
    next_review = parse_review_date(metadata.next_review_date)
    if next_review is None:
        return False
    if not isinstance(metadata.review_stage, int):
        return False
    if metadata.review_stage < 0 or metadata.review_stage >= len(REVIEW_INTERVAL_DAYS):
        return False
    return next_review <= today


def schedule_review(metadata: PaperMetadata, today: date) -> None:
    """Start or reset a paper's review schedule at the first interval."""
    metadata.review_stage = 0
    metadata.next_review_date = _due_date_for_stage(0, today)


def mark_reviewed(metadata: PaperMetadata, today: date) -> None:
    """Advance a paper's review schedule, or start it if unscheduled."""
    if parse_review_date(metadata.next_review_date) is None:
        schedule_review(metadata, today)
        return
    if not isinstance(metadata.review_stage, int):
        schedule_review(metadata, today)
        return
    current_stage = metadata.review_stage
    if current_stage < 0 or current_stage >= len(REVIEW_INTERVAL_DAYS):
        schedule_review(metadata, today)
        return
    next_stage = min(current_stage + 1, len(REVIEW_INTERVAL_DAYS) - 1)
    metadata.review_stage = next_stage
    metadata.next_review_date = _due_date_for_stage(next_stage, today)


def clear_review(metadata: PaperMetadata) -> None:
    """Remove a paper from the spaced-review queue."""
    metadata.next_review_date = None
    metadata.review_stage = None


def review_status_label(metadata: PaperMetadata | None, today: date) -> str | None:
    """Return a compact display label for an active review schedule."""
    if metadata is None:
        return None
    return review_status_label_for_schedule(
        metadata.next_review_date,
        metadata.review_stage,
        today,
    )


def review_status_label_for_schedule(
    next_review_date: str | None,
    review_stage: int | None,
    today: date,
) -> str | None:
    """Return a compact display label for persisted review fields."""
    next_review = parse_review_date(next_review_date)
    if next_review is None or not isinstance(review_stage, int):
        return None
    if review_stage < 0 or review_stage >= len(REVIEW_INTERVAL_DAYS):
        return None
    return "due" if next_review <= today else next_review.isoformat()


def _due_date_for_stage(stage: int, today: date) -> str:
    return (today + timedelta(days=REVIEW_INTERVAL_DAYS[stage])).isoformat()


__all__ = [
    "REVIEW_INTERVAL_DAYS",
    "clear_review",
    "is_review_due",
    "mark_reviewed",
    "normalize_review_schedule",
    "parse_review_date",
    "review_status_label",
    "review_status_label_for_schedule",
    "schedule_review",
]
