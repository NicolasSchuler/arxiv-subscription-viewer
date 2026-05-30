# pyright: reportAttributeAccessIssue=false
"""Detail-pane annotation and status-metric helpers."""

from __future__ import annotations

import time

from textual.css.query import NoMatches

from arxiv_browser.modals import LineAnnotationModal
from arxiv_browser.modals.editing import LineAnnotationResult
from arxiv_browser.models import LineAnnotation


class DetailAnnotationMixin:
    """Mixin for detail-line annotations and dense status-bar metrics."""

    def _status_enrichment_progress(self) -> tuple[str, int, int] | None:
        """Return the active enrichment/progress token for the status bar."""
        scoring_progress = getattr(self, "_scoring_progress", None)
        if scoring_progress is not None:
            current, total = scoring_progress
            return "Scoring", current, total
        version_progress = getattr(self, "_version_progress", None)
        if version_progress is not None:
            current, total = version_progress
            return "Versions", current, total
        download_queue = getattr(self, "_download_queue", ())
        downloading = getattr(self, "_downloading", ())
        download_total = int(getattr(self, "_download_total", 0) or 0)
        if download_queue or downloading or download_total:
            return (
                "Downloading",
                len(getattr(self, "_download_results", {})),
                download_total,
            )
        auto_tag_progress = getattr(self, "_auto_tag_progress", None)
        if auto_tag_progress is not None:
            current, total = auto_tag_progress
            return "Auto-tag", current, total
        return None

    def _record_read_velocity_events(self, count: int = 1) -> None:
        """Record papers newly marked read for the session-local velocity sparkline."""
        if count <= 0:
            return
        events = getattr(self, "_read_event_timestamps", None)
        if events is None:
            return
        now = time.monotonic()
        for _ in range(count):
            events.append(now)

    def _reading_velocity_series(self) -> tuple[float, ...]:
        """Return read events per recent bucket for the status sparkline."""
        events = getattr(self, "_read_event_timestamps", None)
        if not events:
            return ()
        now = time.monotonic()
        window_seconds = 300.0
        bucket_count = 8
        bucket_width = window_seconds / bucket_count
        while events and now - events[0] > window_seconds:
            events.popleft()
        buckets = [0.0] * bucket_count
        for event_time in events:
            age = max(0.0, now - event_time)
            bucket_index = min(bucket_count - 1, int(age / bucket_width))
            buckets[bucket_count - 1 - bucket_index] += 1.0
        return tuple(buckets)

    def _category_distribution(self) -> tuple[tuple[str, int], ...]:
        """Return top primary arXiv categories in the current visible list."""
        counts: dict[str, int] = {}
        for paper in self.filtered_papers:
            primary = (getattr(paper, "categories", "") or "").split()
            if primary:
                counts[primary[0]] = counts.get(primary[0], 0) + 1
        return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4])

    def _ensure_detail_cursor_for(self, arxiv_id: str) -> None:
        """Initialize or preserve the detail-line cursor for the current paper."""
        if getattr(self, "_detail_line_cursor_paper_id", None) != arxiv_id:
            self._detail_line_cursor_paper_id = arxiv_id
            self._detail_line_cursor = 1

    def _detail_line_count(self) -> int:
        """Return the current base detail line count for cursor clamping."""
        try:
            details = self._get_paper_details_widget()
        except NoMatches:
            return 1
        return max(1, int(getattr(details, "detail_line_count", 1) or 1))

    def _move_detail_line_cursor(self, delta: int) -> bool:
        """Move the detail-line cursor when the detail pane is focused."""
        if not self._is_detail_footer_active():
            return False
        paper = self._get_current_paper()
        if paper is None:
            return True
        self._ensure_detail_cursor_for(paper.arxiv_id)
        current = int(getattr(self, "_detail_line_cursor", 1) or 1)
        self._detail_line_cursor = max(1, min(self._detail_line_count(), current + delta))
        self._refresh_detail_pane()
        return True

    def _open_line_annotation_modal(self) -> bool:
        """Open a quick annotation input for the current detail line."""
        if not self._is_detail_footer_active():
            return False
        paper = self._get_current_paper()
        if paper is None:
            return True
        self._ensure_detail_cursor_for(paper.arxiv_id)
        line = max(1, min(self._detail_line_count(), int(self._detail_line_cursor or 1)))

        def on_annotation(result: LineAnnotationResult | None) -> None:
            """Save a line annotation from the annotation modal."""
            if result is None:
                return
            metadata = self._get_or_create_metadata(paper.arxiv_id)
            metadata.line_annotations.append(LineAnnotation(line=result.line, text=result.text))
            self._save_config_or_warn("line annotation")
            self._refresh_detail_pane()
            self.notify("Annotation saved", title="Annotate")

        self.push_screen(LineAnnotationModal(line), on_annotation)
        return True


__all__ = ["DetailAnnotationMixin"]
