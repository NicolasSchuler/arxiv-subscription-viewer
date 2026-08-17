"""Dataset-scoped enrichment and scoring reset helpers."""

from __future__ import annotations

from typing import Any


def reset_dataset_enrichment_state(app: Any) -> None:
    """Clear enrichment/scoring state that cannot cross dataset boundaries."""
    app._paper_summaries.clear()
    app._summary_loading.clear()
    app._summary_mode_label.clear()
    app._summary_command_hash.clear()
    app._s2_cache.clear()
    app._s2_loading = set()
    app._s2_api_error = False
    app._hf_cache.clear()
    app._hf_loading = False
    app._hf_api_error = False
    app._version_updates.clear()
    app._version_checking = False
    app._version_progress = None
    app._relevance_scores.clear()
    app._relevance_scoring_active = False
    app._scoring_progress = None
    judge_scores = getattr(app, "_judge_scores", None)
    if judge_scores is None:
        app._judge_scores = {}
    else:
        judge_scores.clear()
    app._judge_hash = ""
    app._judge_scoring_active = False
    app._judge_progress = None
    app._judge_cancel_requested = False
    app._judge_task = None
    app._auto_tag_active = False
    app._auto_tag_progress = None
    app._cancel_batch_requested = False


__all__ = ["reset_dataset_enrichment_state"]
