"""List rendering helpers and widgets for paper entries."""

from __future__ import annotations

import re
from collections.abc import Mapping

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


def _build_meta_parts(
    *,
    source: str,
    arxiv_id: str,
    categories: str,
    tags: list[str] | None,
    s2_data: SemanticScholarPaper | None,
    hf_data: HuggingFacePaper | None,
    version_update: tuple[int, int] | None,
    relevance_score: tuple[int, str] | None,
    theme_colors: Mapping[str, str],
    category_colors: Mapping[str, str],
    tag_namespace_colors: Mapping[str, str],
) -> list[str]:
    """Build ordered metadata parts with deterministic priority."""
    parts: list[str] = []
    if source == "api":
        parts.append(f"[{theme_colors['orange']}]API[/]")
    parts.extend([f"[dim]{arxiv_id}[/]", format_categories(categories, category_colors)])
    if tags:
        tag_str = " ".join(
            f"[{get_tag_color(tag, tag_namespace_colors)}]#{escape_rich_text(tag)}[/]"
            for tag in tags
        )
        parts.append(tag_str)
    if s2_data is not None:
        parts.append(f"[{theme_colors['green']}]C{s2_data.citation_count}[/]")
    if hf_data is not None:
        hf_upvotes = _ACTIVE_META_GLYPHS["hf_upvotes"]
        parts.append(f"[{theme_colors['orange']}]{hf_upvotes}{hf_data.upvotes}[/]")
    if version_update is not None:
        old_v, new_v = version_update
        version_arrow = _ACTIVE_META_GLYPHS["version_arrow"]
        parts.append(f"[{theme_colors['pink']}]v{old_v}{version_arrow}v{new_v}[/]")
    if relevance_score is not None:
        score, _ = relevance_score
        color, sym = _relevance_badge_parts(score, theme_colors=theme_colors)
        parts.append(f"[{color}]{sym}{score}/10[/]")
    return parts


def _render_title_line(
    paper: Paper,
    selected: bool,
    metadata: PaperMetadata | None,
    watched: bool,
    ht: dict[str, list[str]],
    theme_colors: Mapping[str, str],
) -> str:
    """Build the title line with selection/watch/star/read indicators."""
    prefix_parts: list[str] = []
    if selected:
        prefix_parts.append(f"[{theme_colors['green']}]{_ACTIVE_ICON_SET['selected']}[/]")
    if watched:
        prefix_parts.append(f"[{theme_colors['orange']}]{_ACTIVE_ICON_SET['watched']}[/]")
    if metadata and metadata.starred:
        prefix_parts.append(f"[{theme_colors['yellow']}]{_ACTIVE_ICON_SET['starred']}[/]")
    if metadata and metadata.is_read:
        prefix_parts.append(f"[{theme_colors['muted']}]{_ACTIVE_ICON_SET['read']}[/]")
    prefix = " ".join(prefix_parts)

    title_text = highlight_text(paper.title, ht.get("title", []), theme_colors["accent"])
    if metadata and metadata.is_read:
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


def _render_meta_badges(
    paper: Paper,
    metadata: PaperMetadata | None,
    s2_data: SemanticScholarPaper | None,
    hf_data: HuggingFacePaper | None,
    version_update: tuple[int, int] | None,
    relevance_score: tuple[int, str] | None,
    theme_colors: Mapping[str, str],
    category_colors: Mapping[str, str],
    tag_namespace_colors: Mapping[str, str],
) -> str:
    """Build the meta line with arxiv_id, categories, and badges."""
    parts = _build_meta_parts(
        source=paper.source,
        arxiv_id=paper.arxiv_id,
        categories=paper.categories,
        tags=metadata.tags if metadata else None,
        s2_data=s2_data,
        hf_data=hf_data,
        version_update=version_update,
        relevance_score=relevance_score,
        theme_colors=theme_colors,
        category_colors=category_colors,
        tag_namespace_colors=tag_namespace_colors,
    )
    return _compress_meta_parts(parts)


def _render_abstract_preview(
    abstract_text: str | None,
    ht: dict[str, list[str]],
    theme_colors: Mapping[str, str],
) -> str:
    """Build the abstract preview line for the paper list."""
    if abstract_text is None:
        return "[dim italic]Loading abstract...[/]"
    if not abstract_text:
        return "[dim italic]No abstract available[/]"
    if len(abstract_text) <= PREVIEW_ABSTRACT_MAX_LEN:
        highlighted = highlight_text(abstract_text, ht.get("abstract", []), theme_colors["accent"])
        return f"[dim italic]{highlighted}[/]"
    truncated = truncate_at_word_boundary(
        abstract_text, PREVIEW_ABSTRACT_MAX_LEN, ascii_mode=_ACTIVE_LISTING_ASCII_MODE
    )
    highlighted = highlight_text(truncated, ht.get("abstract", []), theme_colors["accent"])
    return f"[dim italic]{highlighted}[/]"


def render_paper_option(
    paper: Paper,
    *,
    selected: bool = False,
    metadata: PaperMetadata | None = None,
    watched: bool = False,
    show_preview: bool = False,
    abstract_text: str | None = None,
    highlight_terms: dict[str, list[str]] | None = None,
    s2_data: SemanticScholarPaper | None = None,
    hf_data: HuggingFacePaper | None = None,
    version_update: tuple[int, int] | None = None,
    relevance_score: tuple[int, str] | None = None,
    theme_colors: Mapping[str, str] | None = None,
    category_colors: Mapping[str, str] | None = None,
    tag_namespace_colors: Mapping[str, str] | None = None,
) -> str:
    """Render a paper as Rich markup for OptionList display."""
    ht = highlight_terms or {"title": [], "author": [], "abstract": []}
    colors = theme_colors or DEFAULT_THEME
    resolved_category_colors = category_colors or category_colors_for(None)
    resolved_tag_namespace_colors = tag_namespace_colors or DEFAULT_TAG_NAMESPACE_COLORS

    lines = [
        _render_title_line(paper, selected, metadata, watched, ht, colors),
        highlight_text(paper.authors, ht.get("author", []), colors["accent"]),
        _render_meta_badges(
            paper,
            metadata,
            s2_data,
            hf_data,
            version_update,
            relevance_score,
            colors,
            resolved_category_colors,
            tag_namespace_colors or resolved_tag_namespace_colors,
        ),
    ]

    if show_preview:
        lines.append(_render_abstract_preview(abstract_text, ht, colors))

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
        paper: Paper,
        selected: bool = False,
        metadata: PaperMetadata | None = None,
        watched: bool = False,
        show_preview: bool = False,
        abstract_text: str | None = None,
        highlight_terms: dict[str, list[str]] | None = None,
        theme_colors: Mapping[str, str] | None = None,
        category_colors: Mapping[str, str] | None = None,
        tag_namespace_colors: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.paper = paper
        self._selected = selected
        self._metadata = metadata
        self._watched = watched
        self._show_preview = show_preview
        self._abstract_text = abstract_text
        self._highlight_terms = highlight_terms or {
            "title": [],
            "author": [],
            "abstract": [],
        }
        self._theme_colors = dict(theme_colors or DEFAULT_THEME)
        self._category_colors = dict(category_colors or category_colors_for(None))
        self._tag_namespace_colors = dict(tag_namespace_colors or DEFAULT_TAG_NAMESPACE_COLORS)
        self._s2_data: SemanticScholarPaper | None = None
        self._hf_data: HuggingFacePaper | None = None
        self._version_update: tuple[int, int] | None = None
        self._relevance_score: tuple[int, str] | None = None
        if selected:
            self.add_class("selected")

    @property
    def is_selected(self) -> bool:
        return self._selected

    @property
    def metadata(self) -> PaperMetadata | None:
        return self._metadata

    def set_metadata(self, metadata: PaperMetadata | None) -> None:
        """Update metadata and refresh display."""
        self._metadata = metadata
        self._update_display()

    def set_abstract_text(self, text: str | None) -> None:
        """Update abstract text for preview and refresh display."""
        self._abstract_text = text
        if self._show_preview:
            self._update_display()

    def update_s2_data(self, s2_data: SemanticScholarPaper | None) -> None:
        """Update Semantic Scholar data and refresh display."""
        self._s2_data = s2_data
        self._update_display()

    def update_hf_data(self, hf_data: HuggingFacePaper | None) -> None:
        """Update HuggingFace data and refresh display."""
        self._hf_data = hf_data
        self._update_display()

    def update_version_data(self, version_update: tuple[int, int] | None) -> None:
        """Update version tracking data and refresh display."""
        self._version_update = version_update
        self._update_display()

    def update_relevance_data(self, relevance: tuple[int, str] | None) -> None:
        """Update relevance score data and refresh display."""
        self._relevance_score = relevance
        self._update_display()

    def _get_title_text(self) -> str:
        """Get the formatted title text based on selection and metadata state."""
        prefix_parts = []
        colors = theme_colors_for(self, self._theme_colors)

        # Selection indicator
        if self._selected:
            prefix_parts.append(f"[{colors['green']}]{_ACTIVE_ICON_SET['selected']}[/]")

        # Watched indicator
        if self._watched:
            prefix_parts.append(f"[{colors['orange']}]{_ACTIVE_ICON_SET['watched']}[/]")

        # Starred indicator
        if self._metadata and self._metadata.starred:
            prefix_parts.append(f"[{colors['yellow']}]{_ACTIVE_ICON_SET['starred']}[/]")

        # Read indicator
        if self._metadata and self._metadata.is_read:
            prefix_parts.append(f"[{colors['muted']}]{_ACTIVE_ICON_SET['read']}[/]")

        prefix = " ".join(prefix_parts)
        title_text = highlight_text(
            self.paper.title,
            self._highlight_terms.get("title", []),
            colors["accent"],
        )
        # Dim title for read papers — unread titles stay bold/bright
        is_read = self._metadata and self._metadata.is_read
        if is_read:
            title_text = f"[dim]{title_text}[/]"
        if prefix:
            return f"{prefix} {title_text}"
        return title_text

    def _get_authors_text(self) -> str:
        """Get the formatted author text."""
        colors = theme_colors_for(self, self._theme_colors)
        return highlight_text(
            self.paper.authors,
            self._highlight_terms.get("author", []),
            colors["accent"],
        )

    def _get_meta_text(self) -> str:
        """Get the formatted metadata text."""
        colors = theme_colors_for(self, self._theme_colors)
        parts = _build_meta_parts(
            source=self.paper.source,
            arxiv_id=self.paper.arxiv_id,
            categories=self.paper.categories,
            tags=self._metadata.tags if self._metadata else None,
            s2_data=self._s2_data,
            hf_data=self._hf_data,
            version_update=self._version_update,
            relevance_score=self._relevance_score,
            theme_colors=colors,
            category_colors=category_colors_for(self, self._category_colors),
            tag_namespace_colors=tag_namespace_colors_for(self, self._tag_namespace_colors),
        )
        return _compress_meta_parts(parts)

    def _get_preview_text(self) -> str:
        """Get truncated abstract preview text.

        Returns formatted Rich markup for the abstract preview.
        Handles empty abstracts and truncates at word boundaries.
        """
        abstract = self._abstract_text
        colors = theme_colors_for(self, self._theme_colors)
        if abstract is None:
            return "[dim italic]Loading abstract...[/]"
        if not abstract:
            return "[dim italic]No abstract available[/]"
        if len(abstract) <= PREVIEW_ABSTRACT_MAX_LEN:
            highlighted = highlight_text(
                abstract,
                self._highlight_terms.get("abstract", []),
                colors["accent"],
            )
            return f"[dim italic]{highlighted}[/]"
        # Truncate at word boundary for cleaner display
        truncated = truncate_at_word_boundary(
            abstract, PREVIEW_ABSTRACT_MAX_LEN, ascii_mode=_ACTIVE_LISTING_ASCII_MODE
        )
        highlighted = highlight_text(
            truncated,
            self._highlight_terms.get("abstract", []),
            colors["accent"],
        )
        return f"[dim italic]{highlighted}[/]"

    def _update_selection_class(self) -> None:
        """Update the CSS class based on selection state."""
        if self._selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")

    def toggle_selected(self) -> bool:
        """Toggle selection state and return new state."""
        self._selected = not self._selected
        self._update_selection_class()
        self._update_display()
        return self._selected

    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self._selected = selected
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
            if self._show_preview:
                preview_widget = self.query_one(".paper-preview", Static)
                preview_widget.update(self._get_preview_text())
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        yield Static(self._get_title_text(), classes="paper-title")
        yield Static(self._get_authors_text(), classes="paper-authors")
        yield Static(self._get_meta_text(), classes="paper-meta")
        if self._show_preview:
            yield Static(self._get_preview_text(), classes="paper-preview")


__all__ = [
    "PREVIEW_ABSTRACT_MAX_LEN",
    "PaperListItem",
    "render_paper_option",
]
