"""List rendering helpers and widgets for paper entries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widgets import ListItem, Static

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import Paper, PaperMetadata
from arxiv_browser.query import (
    escape_rich_text,
    format_categories,
    highlight_text,
    truncate_at_word_boundary,
)
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.themes import (
    DEFAULT_TAG_NAMESPACE_COLORS,
    DEFAULT_THEME,
    category_colors_for,
    get_tag_color,
    tag_namespace_colors_for,
    theme_colors_for,
)

PREVIEW_ABSTRACT_MAX_LEN = 150  # Max abstract preview length in list items
META_LINE_BUDGET = 78  # Visible character budget for list metadata row
_RICH_TAG_RE = re.compile(r"\[[^\]]*]")

_ICON_SETS: dict[str, dict[str, str]] = {
    "unicode": {
        "selected": "●",
        "watched": "👁",
        "starred": "⭐",
        "read": "✓",
    },
    "ascii": {
        "selected": "[x]",
        "watched": "[w]",
        "starred": "*",
        "read": "v",
    },
}
_ACTIVE_ICON_SET = _ICON_SETS["unicode"]
_ACTIVE_LISTING_ASCII_MODE = False
_META_GLYPH_SETS: dict[str, dict[str, str]] = {
    "unicode": {
        "hf_upvotes": "\u2191",  # ↑
        "version_arrow": "\u2192",  # →
        "relevance_high": "\u2605",  # ★
        "relevance_mid": "\u25b8",  # ▸
        "relevance_low": "\u00b7",  # ·
    },
    "ascii": {
        "hf_upvotes": "^",
        "version_arrow": "->",
        "relevance_high": "*",
        "relevance_mid": ">",
        "relevance_low": ".",
    },
}
_ACTIVE_META_GLYPHS = _META_GLYPH_SETS["unicode"]


def set_ascii_icons(enabled: bool) -> None:
    """Switch list indicators between Unicode and ASCII modes."""
    global _ACTIVE_ICON_SET, _ACTIVE_META_GLYPHS, _ACTIVE_LISTING_ASCII_MODE
    _ACTIVE_LISTING_ASCII_MODE = enabled
    _ACTIVE_ICON_SET = _ICON_SETS["ascii"] if enabled else _ICON_SETS["unicode"]
    _ACTIVE_META_GLYPHS = _META_GLYPH_SETS["ascii"] if enabled else _META_GLYPH_SETS["unicode"]


@dataclass(frozen=True, slots=True)
class PaperHighlightTerms:
    """Normalized per-field search terms used for one rendered paper row."""

    title: tuple[str, ...] = ()
    author: tuple[str, ...] = ()
    abstract: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PaperRowRenderState:
    """Complete render state for one paper row in the main list."""

    paper: Paper
    selected: bool = False
    metadata: PaperMetadata | None = None
    watched: bool = False
    show_preview: bool = False
    abstract_text: str | None = None
    highlight_terms: PaperHighlightTerms = field(default_factory=PaperHighlightTerms)
    s2_data: SemanticScholarPaper | None = None
    hf_data: HuggingFacePaper | None = None
    version_update: tuple[int, int] | None = None
    relevance_score: tuple[int, str] | None = None
    theme_colors: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_THEME))
    category_colors: Mapping[str, str] = field(
        default_factory=lambda: dict(category_colors_for(None))
    )
    tag_namespace_colors: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TAG_NAMESPACE_COLORS)
    )


def _normalize_highlight_terms(
    raw_terms: Mapping[str, list[str]] | PaperHighlightTerms | None,
) -> PaperHighlightTerms:
    """Normalize highlight terms to immutable tuples."""
    if isinstance(raw_terms, PaperHighlightTerms):
        return raw_terms
    terms = raw_terms or {}
    return PaperHighlightTerms(
        title=tuple(terms.get("title", [])),
        author=tuple(terms.get("author", [])),
        abstract=tuple(terms.get("abstract", [])),
    )


def _normalize_paper_row_state(state: PaperRowRenderState) -> PaperRowRenderState:
    """Return a normalized paper-row state with copied mappings."""
    return replace(
        state,
        highlight_terms=_normalize_highlight_terms(state.highlight_terms),
        theme_colors=dict(state.theme_colors or DEFAULT_THEME),
        category_colors=dict(state.category_colors or category_colors_for(None)),
        tag_namespace_colors=dict(state.tag_namespace_colors or DEFAULT_TAG_NAMESPACE_COLORS),
    )


def _coerce_paper_row_state(
    state_or_paper: Paper | PaperRowRenderState,
    legacy_kwargs: Mapping[str, Any],
) -> PaperRowRenderState:
    """Accept either the new render-state object or the legacy paper+kwargs shape."""
    if isinstance(state_or_paper, PaperRowRenderState):
        if legacy_kwargs:
            raise TypeError("PaperRowRenderState cannot be combined with legacy keyword args")
        return _normalize_paper_row_state(state_or_paper)
    if not isinstance(state_or_paper, Paper):
        raise TypeError("render input must be a Paper or PaperRowRenderState")
    return _normalize_paper_row_state(PaperRowRenderState(paper=state_or_paper, **legacy_kwargs))


def _visible_text_length(text: str) -> int:
    """Return printable width estimate by stripping simple Rich tags."""
    return len(_RICH_TAG_RE.sub("", text))


def _truncate_visible_text(text: str, max_width: int) -> str:
    """Truncate Rich-ish text by visible length, returning plain fallback text."""
    visible = _RICH_TAG_RE.sub("", text)
    if len(visible) <= max_width:
        return visible
    if max_width <= 3:
        return visible[:max_width]
    return visible[: max_width - 3] + "..."


def _join_meta_parts(parts: list[str]) -> str:
    """Join metadata parts with stable spacing."""
    return "  ".join(parts)


def _compress_meta_parts(parts: list[str], budget: int = META_LINE_BUDGET) -> str:
    """Compress metadata by dropping lowest-priority tail parts and showing +N."""
    if not parts:
        return ""
    rendered = _join_meta_parts(parts)
    if _visible_text_length(rendered) <= budget:
        return rendered

    kept = parts.copy()
    removed = 0
    while len(kept) > 1 and _visible_text_length(_join_meta_parts(kept)) > budget:
        kept.pop()
        removed += 1

    if removed > 0:
        summary = f"[dim]+{removed}[/]"
        while len(kept) > 1 and _visible_text_length(_join_meta_parts([*kept, summary])) > budget:
            kept.pop()
            removed += 1
            summary = f"[dim]+{removed}[/]"
        candidate = _join_meta_parts([*kept, summary])
        if _visible_text_length(candidate) <= budget:
            return candidate

    return _truncate_visible_text(_join_meta_parts(kept), budget)


def _build_meta_parts(state: PaperRowRenderState) -> list[str]:
    """Build ordered metadata parts with deterministic priority."""
    parts: list[str] = []
    if state.paper.source == "api":
        parts.append(f"[{state.theme_colors['orange']}]API[/]")
    parts.extend(
        [
            f"[dim]{state.paper.arxiv_id}[/]",
            format_categories(state.paper.categories, state.category_colors),
        ]
    )
    tags = state.metadata.tags if state.metadata else None
    if tags:
        tag_str = " ".join(
            f"[{get_tag_color(tag, state.tag_namespace_colors)}]#{escape_rich_text(tag)}[/]"
            for tag in tags
        )
        parts.append(tag_str)
    if state.s2_data is not None:
        parts.append(f"[{state.theme_colors['green']}]C{state.s2_data.citation_count}[/]")
    if state.hf_data is not None:
        hf_upvotes = _ACTIVE_META_GLYPHS["hf_upvotes"]
        parts.append(f"[{state.theme_colors['orange']}]{hf_upvotes}{state.hf_data.upvotes}[/]")
    if state.version_update is not None:
        old_v, new_v = state.version_update
        version_arrow = _ACTIVE_META_GLYPHS["version_arrow"]
        parts.append(f"[{state.theme_colors['pink']}]v{old_v}{version_arrow}v{new_v}[/]")
    if state.relevance_score is not None:
        score, _ = state.relevance_score
        color, sym = _relevance_badge_parts(score, theme_colors=state.theme_colors)
        parts.append(f"[{color}]{sym}{score}/10[/]")
    return parts


def _render_title_line(state: PaperRowRenderState) -> str:
    """Build the title line with selection/watch/star/read indicators."""
    prefix_parts: list[str] = []
    if state.selected:
        prefix_parts.append(f"[{state.theme_colors['green']}]{_ACTIVE_ICON_SET['selected']}[/]")
    if state.watched:
        prefix_parts.append(f"[{state.theme_colors['orange']}]{_ACTIVE_ICON_SET['watched']}[/]")
    if state.metadata and state.metadata.starred:
        prefix_parts.append(f"[{state.theme_colors['yellow']}]{_ACTIVE_ICON_SET['starred']}[/]")
    if state.metadata and state.metadata.is_read:
        prefix_parts.append(f"[{state.theme_colors['muted']}]{_ACTIVE_ICON_SET['read']}[/]")
    prefix = " ".join(prefix_parts)

    title_text = highlight_text(
        state.paper.title,
        list(state.highlight_terms.title),
        state.theme_colors["accent"],
    )
    if state.metadata and state.metadata.is_read:
        title_text = f"[dim]{title_text}[/]"
    return f"{prefix} {title_text}" if prefix else title_text


def _relevance_badge_parts(
    score: int,
    theme_colors: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Return (color, symbol) for a relevance score badge.

    Uses distinct symbols alongside color so the badge is accessible
    to colorblind users (WCAG 1.4.1 — Use of Color).
    """
    colors = theme_colors or DEFAULT_THEME
    if score >= 8:
        return colors["green"], _ACTIVE_META_GLYPHS["relevance_high"]
    if score >= 5:
        return colors["yellow"], _ACTIVE_META_GLYPHS["relevance_mid"]
    return colors["muted"], _ACTIVE_META_GLYPHS["relevance_low"]


def _render_meta_badges(state: PaperRowRenderState) -> str:
    """Build the meta line with arxiv_id, categories, and badges."""
    return _compress_meta_parts(_build_meta_parts(state))


def _render_abstract_preview(state: PaperRowRenderState) -> str:
    """Build the abstract preview line for the paper list."""
    if state.abstract_text is None:
        return "[dim italic]Loading abstract...[/]"
    if not state.abstract_text:
        return "[dim italic]No abstract available[/]"
    if len(state.abstract_text) <= PREVIEW_ABSTRACT_MAX_LEN:
        highlighted = highlight_text(
            state.abstract_text,
            list(state.highlight_terms.abstract),
            state.theme_colors["accent"],
        )
        return f"[dim italic]{highlighted}[/]"
    truncated = truncate_at_word_boundary(
        state.abstract_text,
        PREVIEW_ABSTRACT_MAX_LEN,
        ascii_mode=_ACTIVE_LISTING_ASCII_MODE,
    )
    highlighted = highlight_text(
        truncated,
        list(state.highlight_terms.abstract),
        state.theme_colors["accent"],
    )
    return f"[dim italic]{highlighted}[/]"


def render_paper_option(
    state_or_paper: Paper | PaperRowRenderState,
    **legacy_kwargs: Any,
) -> str:
    """Render a paper as Rich markup for OptionList display."""
    state = _coerce_paper_row_state(state_or_paper, legacy_kwargs)
    lines = [
        _render_title_line(state),
        highlight_text(
            state.paper.authors,
            list(state.highlight_terms.author),
            state.theme_colors["accent"],
        ),
        _render_meta_badges(state),
    ]

    if state.show_preview:
        lines.append(_render_abstract_preview(state))

    return "\n".join(lines)


class PaperListItem(ListItem):
    """A list item rendering a single paper in the main paper list.

    Each item renders up to four lines depending on the current display state:

    1. **Title line** — paper title with optional search highlight, prefixed
       by selection (✓), watch-list (👁), star (★), and read (·) indicators.
    2. **Authors line** — author string, optionally highlighted.
    3. **Metadata badge line** — arXiv ID, category badges, S2 citation count,
       HF upvote count, version-update badge, and relevance score badge.
    4. **Abstract preview** (optional) — truncated abstract when
       ``show_preview`` is enabled.

    Enrichment data (S2/HF) and the relevance score can be pushed in after
    initial construction via ``set_s2_data``, ``set_hf_data``,
    ``set_version_update``, and ``set_relevance_score``.
    """

    def __init__(
        self,
        state_or_paper: Paper | PaperRowRenderState,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        self._state = _coerce_paper_row_state(state_or_paper, legacy_kwargs)
        self.paper = self._state.paper
        if self._state.selected:
            self.add_class("selected")

    @property
    def is_selected(self) -> bool:
        return self._state.selected

    @property
    def metadata(self) -> PaperMetadata | None:
        return self._state.metadata

    @property
    def _abstract_text(self) -> str | None:
        """Compatibility alias for legacy tests touching internal preview state."""
        return self._state.abstract_text

    @_abstract_text.setter
    def _abstract_text(self, value: str | None) -> None:
        self._state = replace(self._state, abstract_text=value)

    @property
    def _s2_data(self) -> SemanticScholarPaper | None:
        """Compatibility alias for legacy tests touching internal S2 state."""
        return self._state.s2_data

    @_s2_data.setter
    def _s2_data(self, value: SemanticScholarPaper | None) -> None:
        self._state = replace(self._state, s2_data=value)

    @property
    def _hf_data(self) -> HuggingFacePaper | None:
        """Compatibility alias for legacy tests touching internal HF state."""
        return self._state.hf_data

    @_hf_data.setter
    def _hf_data(self, value: HuggingFacePaper | None) -> None:
        self._state = replace(self._state, hf_data=value)

    @property
    def _version_update(self) -> tuple[int, int] | None:
        """Compatibility alias for legacy tests touching version state."""
        return self._state.version_update

    @_version_update.setter
    def _version_update(self, value: tuple[int, int] | None) -> None:
        self._state = replace(self._state, version_update=value)

    @property
    def _relevance_score(self) -> tuple[int, str] | None:
        """Compatibility alias for legacy tests touching relevance state."""
        return self._state.relevance_score

    @_relevance_score.setter
    def _relevance_score(self, value: tuple[int, str] | None) -> None:
        self._state = replace(self._state, relevance_score=value)

    def set_metadata(self, metadata: PaperMetadata | None) -> None:
        """Update metadata and refresh display."""
        self._state = replace(self._state, metadata=metadata)
        self._update_display()

    def set_abstract_text(self, text: str | None) -> None:
        """Update abstract text for preview and refresh display."""
        self._state = replace(self._state, abstract_text=text)
        if self._state.show_preview:
            self._update_display()

    def update_s2_data(self, s2_data: SemanticScholarPaper | None) -> None:
        """Update Semantic Scholar data and refresh display."""
        self._state = replace(self._state, s2_data=s2_data)
        self._update_display()

    def update_hf_data(self, hf_data: HuggingFacePaper | None) -> None:
        """Update HuggingFace data and refresh display."""
        self._state = replace(self._state, hf_data=hf_data)
        self._update_display()

    def update_version_data(self, version_update: tuple[int, int] | None) -> None:
        """Update version tracking data and refresh display."""
        self._state = replace(self._state, version_update=version_update)
        self._update_display()

    def update_relevance_data(self, relevance: tuple[int, str] | None) -> None:
        """Update relevance score data and refresh display."""
        self._state = replace(self._state, relevance_score=relevance)
        self._update_display()

    def _resolved_state(self) -> PaperRowRenderState:
        """Return the current row state with widget-resolved theme mappings."""
        return replace(
            self._state,
            theme_colors=theme_colors_for(self, self._state.theme_colors),
            category_colors=category_colors_for(self, self._state.category_colors),
            tag_namespace_colors=tag_namespace_colors_for(
                self,
                self._state.tag_namespace_colors,
            ),
        )

    def _get_title_text(self) -> str:
        """Get the formatted title text based on selection and metadata state."""
        return _render_title_line(self._resolved_state())

    def _get_authors_text(self) -> str:
        """Get the formatted author text."""
        state = self._resolved_state()
        return highlight_text(
            self.paper.authors,
            list(state.highlight_terms.author),
            state.theme_colors["accent"],
        )

    def _get_meta_text(self) -> str:
        """Get the formatted metadata text."""
        return _render_meta_badges(self._resolved_state())

    def _get_preview_text(self) -> str:
        """Get truncated abstract preview text.

        Returns formatted Rich markup for the abstract preview.
        Handles empty abstracts and truncates at word boundaries.
        """
        return _render_abstract_preview(self._resolved_state())

    def _update_selection_class(self) -> None:
        """Update the CSS class based on selection state."""
        if self._state.selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")

    def toggle_selected(self) -> bool:
        """Toggle selection state and return new state."""
        self._state = replace(self._state, selected=not self._state.selected)
        self._update_selection_class()
        self._update_display()
        return self._state.selected

    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self._state = replace(self._state, selected=selected)
        self._update_selection_class()
        self._update_display()

    def _update_display(self) -> None:
        """Update the visual display based on selection state."""
        try:
            title_widget = self.query_one(".paper-title", Static)
            authors_widget = self.query_one(".paper-authors", Static)
            meta_widget = self.query_one(".paper-meta", Static)
            title_widget.update(self._get_title_text())
            authors_widget.update(self._get_authors_text())
            meta_widget.update(self._get_meta_text())
            if self._state.show_preview:
                preview_widget = self.query_one(".paper-preview", Static)
                preview_widget.update(self._get_preview_text())
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        yield Static(self._get_title_text(), classes="paper-title")
        yield Static(self._get_authors_text(), classes="paper-authors")
        yield Static(self._get_meta_text(), classes="paper-meta")
        if self._state.show_preview:
            yield Static(self._get_preview_text(), classes="paper-preview")


__all__ = [
    "PREVIEW_ABSTRACT_MAX_LEN",
    "PaperHighlightTerms",
    "PaperListItem",
    "PaperRowRenderState",
    "render_paper_option",
]
