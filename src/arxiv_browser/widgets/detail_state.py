"""Normalized detail-pane state and stable render-cache identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from arxiv_browser.conference_deadlines import SubmissionTarget
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import LineAnnotation, Paper
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.themes import (
    DEFAULT_CATEGORY_COLORS,
    DEFAULT_TAG_NAMESPACE_COLORS,
    DEFAULT_THEME,
)


@dataclass(frozen=True, slots=True)
class DetailRenderState:
    """Complete render state for the detail pane."""

    paper: Paper | None
    abstract_text: str = ""
    abstract_loading: bool = False
    summary: str | None = None
    summary_loading: bool = False
    highlight_terms: tuple[str, ...] = ()
    s2_data: SemanticScholarPaper | None = None
    s2_loading: bool = False
    hf_data: HuggingFacePaper | None = None
    version_update: tuple[int, int] | None = None
    summary_mode: str = ""
    tags: tuple[str, ...] = ()
    relevance: tuple[int, str] | None = None
    submission_targets: tuple[SubmissionTarget, ...] = ()
    deadline_countdown_key: str = ""
    is_read: bool = False
    starred: bool = False
    next_review_date: str | None = None
    review_stage: int | None = None
    line_annotations: tuple[LineAnnotation, ...] = ()
    detail_line_cursor: int | None = None
    detail_focus: bool = True
    collapsed_sections: tuple[str, ...] = ()
    detail_mode: str = "full"
    theme_colors: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_THEME))
    category_colors: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_COLORS)
    )
    tag_namespace_colors: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TAG_NAMESPACE_COLORS)
    )


def _normalize_detail_state(state: DetailRenderState) -> DetailRenderState:
    """Return a normalized detail state with copied mappings and tuples."""
    return DetailRenderState(
        paper=state.paper,
        abstract_text=state.abstract_text,
        abstract_loading=state.abstract_loading,
        summary=state.summary,
        summary_loading=state.summary_loading,
        highlight_terms=tuple(state.highlight_terms or ()),
        s2_data=state.s2_data,
        s2_loading=state.s2_loading,
        hf_data=state.hf_data,
        version_update=state.version_update,
        summary_mode=state.summary_mode,
        tags=tuple(state.tags or ()),
        relevance=state.relevance,
        submission_targets=tuple(state.submission_targets or ()),
        deadline_countdown_key=state.deadline_countdown_key,
        is_read=state.is_read,
        starred=state.starred,
        next_review_date=state.next_review_date,
        review_stage=state.review_stage,
        line_annotations=tuple(state.line_annotations or ()),
        detail_line_cursor=state.detail_line_cursor,
        detail_focus=state.detail_focus,
        collapsed_sections=tuple(state.collapsed_sections or ()),
        detail_mode=state.detail_mode,
        theme_colors=dict(state.theme_colors or DEFAULT_THEME),
        category_colors=dict(state.category_colors or DEFAULT_CATEGORY_COLORS),
        tag_namespace_colors=dict(state.tag_namespace_colors or DEFAULT_TAG_NAMESPACE_COLORS),
    )


def _coerce_detail_state(
    state_or_paper: Paper | DetailRenderState | None,
    abstract_text: str | None,
    legacy_kwargs: Mapping[str, Any],
) -> DetailRenderState | None:
    """Accept either a full detail state or the legacy paper+kwargs shape."""
    if isinstance(state_or_paper, DetailRenderState):
        if abstract_text is not None or legacy_kwargs:
            raise TypeError("DetailRenderState cannot be combined with legacy detail arguments")
        return _normalize_detail_state(state_or_paper)
    if state_or_paper is None:
        if abstract_text is not None or legacy_kwargs:
            raise TypeError("Legacy detail arguments require a paper")
        return None

    kwargs = dict(legacy_kwargs)
    abstract_loading = bool(
        kwargs.pop("abstract_loading", abstract_text is None and state_or_paper.abstract is None)
    )
    resolved_abstract = state_or_paper.abstract or "" if abstract_text is None else abstract_text
    return _normalize_detail_state(
        DetailRenderState(
            paper=state_or_paper,
            abstract_text=resolved_abstract,
            abstract_loading=abstract_loading,
            **kwargs,
        )
    )


def _detail_cache_key_for_state(state: DetailRenderState, *, glyph_mode: str) -> tuple:
    """Build a stable, hashable cache key for rendered detail markup.

    ``state`` may contain mutable collections, rich objects, and very long text
    fields. This helper normalizes those pieces into hashable tuples or short
    digests so the result can be used as a compact ``dict`` key for the render
    cache. Any visible change in the detail pane should produce a different key.
    """
    if state.paper is None:
        return ("empty", glyph_mode)

    # SHA-256 keeps long visible text out of the small FIFO cache key.
    abstract_digest = (
        hashlib.sha256(state.abstract_text.encode("utf-8")).hexdigest()
        if state.abstract_text
        else ""
    )
    summary_digest = (
        hashlib.sha256(state.summary.encode("utf-8")).hexdigest() if state.summary else ""
    )
    s2_key = (
        (
            state.s2_data.citation_count,
            state.s2_data.influential_citation_count,
            tuple(state.s2_data.fields_of_study),
            state.s2_data.tldr,
        )
        if state.s2_data is not None
        else None
    )
    hf_key = (
        (
            state.hf_data.upvotes,
            state.hf_data.num_comments,
            state.hf_data.github_repo,
            state.hf_data.github_stars,
            tuple(state.hf_data.ai_keywords),
            state.hf_data.ai_summary,
        )
        if state.hf_data is not None
        else None
    )
    return (
        state.paper.arxiv_id,
        state.paper.title,
        state.paper.authors,
        state.paper.date,
        state.paper.categories,
        state.paper.comments,
        state.paper.url,
        abstract_digest,
        state.abstract_loading,
        summary_digest,
        state.summary_loading,
        state.highlight_terms,
        s2_key,
        state.s2_loading,
        hf_key,
        state.version_update,
        state.summary_mode,
        state.tags,
        state.relevance,
        state.submission_targets,
        state.deadline_countdown_key,
        state.is_read,
        state.starred,
        state.next_review_date,
        state.review_stage,
        tuple((annotation.line, annotation.text) for annotation in state.line_annotations),
        state.detail_line_cursor,
        state.detail_focus,
        state.collapsed_sections,
        state.detail_mode,
        tuple(sorted(state.theme_colors.items())),
        tuple(sorted(state.category_colors.items())),
        tuple(sorted(state.tag_namespace_colors.items())),
        glyph_mode,
    )


__all__ = ["DetailRenderState"]
