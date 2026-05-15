"""Tests for spaced-review scheduling helpers."""

from __future__ import annotations

from datetime import date

from arxiv_browser.models import PaperMetadata
from arxiv_browser.review import (
    REVIEW_INTERVAL_DAYS,
    clear_review,
    is_review_due,
    mark_reviewed,
    normalize_review_schedule,
    parse_review_date,
    review_status_label,
    schedule_review,
)


def test_schedule_review_starts_at_first_interval() -> None:
    metadata = PaperMetadata(arxiv_id="paper-1")

    schedule_review(metadata, date(2026, 5, 15))

    assert REVIEW_INTERVAL_DAYS == (1, 3, 7, 14, 30)
    assert metadata.review_stage == 0
    assert metadata.next_review_date == "2026-05-16"


def test_mark_reviewed_advances_intervals_and_caps() -> None:
    metadata = PaperMetadata(arxiv_id="paper-1")
    today = date(2026, 5, 15)

    schedule_review(metadata, today)
    expected = [
        (1, "2026-05-18"),
        (2, "2026-05-22"),
        (3, "2026-05-29"),
        (4, "2026-06-14"),
        (4, "2026-06-14"),
    ]
    for stage, due_date in expected:
        mark_reviewed(metadata, today)
        assert (metadata.review_stage, metadata.next_review_date) == (stage, due_date)


def test_mark_reviewed_uses_today_as_base_for_early_review() -> None:
    metadata = PaperMetadata(
        arxiv_id="paper-1",
        next_review_date="2026-12-31",
        review_stage=1,
    )

    mark_reviewed(metadata, date(2026, 5, 15))

    assert metadata.review_stage == 2
    assert metadata.next_review_date == "2026-05-22"


def test_mark_reviewed_starts_invalid_or_missing_schedule() -> None:
    metadata = PaperMetadata(arxiv_id="paper-1", next_review_date="bad", review_stage=2)

    mark_reviewed(metadata, date(2026, 5, 15))

    assert metadata.review_stage == 0
    assert metadata.next_review_date == "2026-05-16"


def test_clear_review_removes_schedule() -> None:
    metadata = PaperMetadata(
        arxiv_id="paper-1",
        next_review_date="2026-05-15",
        review_stage=2,
    )

    clear_review(metadata)

    assert metadata.next_review_date is None
    assert metadata.review_stage is None


def test_due_and_status_helpers_require_active_valid_schedule() -> None:
    today = date(2026, 5, 15)
    due = PaperMetadata(arxiv_id="due", next_review_date="2026-05-15", review_stage=0)
    future = PaperMetadata(arxiv_id="future", next_review_date="2026-05-16", review_stage=0)
    no_stage = PaperMetadata(arxiv_id="no-stage", next_review_date="2026-05-15")
    malformed = PaperMetadata(arxiv_id="bad", next_review_date="2026-13-01", review_stage=0)

    assert is_review_due(due, today) is True
    assert review_status_label(due, today) == "due"
    assert is_review_due(future, today) is False
    assert review_status_label(future, today) == "2026-05-16"
    assert is_review_due(no_stage, today) is False
    assert review_status_label(no_stage, today) is None
    assert is_review_due(malformed, today) is False
    assert review_status_label(malformed, today) is None


def test_review_date_and_schedule_normalization() -> None:
    assert parse_review_date("2026-05-15") == date(2026, 5, 15)
    assert parse_review_date("20260515") is None
    assert parse_review_date("2026-02-31") is None

    assert normalize_review_schedule("2026-05-15", 3) == ("2026-05-15", 3)
    assert normalize_review_schedule("2026-05-15", "bad") == ("2026-05-15", 0)
    assert normalize_review_schedule("2026-05-15", 99) == ("2026-05-15", 0)
    assert normalize_review_schedule("bad", 1) == (None, None)
