"""Detail pane widget for rendering selected paper metadata and enrichment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from textual.widgets import Static

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import Paper
from arxiv_browser.query import (
    escape_rich_text,
    format_categories,
    format_summary_as_rich,
    highlight_text,
    truncate_at_word_boundary,
)
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.themes import (
    DEFAULT_CATEGORY_COLORS,
    DEFAULT_TAG_NAMESPACE_COLORS,
    DEFAULT_THEME,
    category_colors_for,
    get_tag_color,
    parse_tag_namespace,
    tag_namespace_colors_for,
    theme_colors_for,
)
from arxiv_browser.widgets.listing import _relevance_badge_parts

# Maximum number of cached detail pane renderings (FIFO eviction)
DETAIL_CACHE_MAX = 100
DETAIL_SCAN_ABSTRACT_LEN = 420
DETAIL_SCAN_AUTHORS_LEN = 120

_DETAIL_GLYPH_SETS: dict[str, dict[str, str]] = {
    "unicode": {
        "collapsed": "▸",
        "expanded": "▾",
        "summary_prefix": "🤖 ",
        "summary_loading": "⏳ ",
        "hf_upvotes": "↑",
        "version_arrow": "→",
    },
    "ascii": {
        "collapsed": ">",
        "expanded": "v",
        "summary_prefix": "",
        "summary_loading": "",
        "hf_upvotes": "^",
        "version_arrow": "->",
    },
}
_ACTIVE_DETAIL_GLYPH_MODE = "unicode"
_ACTIVE_DETAIL_GLYPHS = _DETAIL_GLYPH_SETS[_ACTIVE_DETAIL_GLYPH_MODE]


def set_ascii_glyphs(enabled: bool) -> None:
    """Switch detail pane glyphs between Unicode and ASCII modes."""
    global _ACTIVE_DETAIL_GLYPHS, _ACTIVE_DETAIL_GLYPH_MODE
    _ACTIVE_DETAIL_GLYPH_MODE = "ascii" if enabled else "unicode"
    _ACTIVE_DETAIL_GLYPHS = _DETAIL_GLYPH_SETS[_ACTIVE_DETAIL_GLYPH_MODE]


def _relevance_symbol_for_mode(symbol: str) -> str:
    """Convert relevance badge symbols to ASCII-safe equivalents when needed."""
    if _ACTIVE_DETAIL_GLYPH_MODE != "ascii":
        return symbol
    return {
        "\u2605": "*",  # ★
        "\u25b8": ">",  # ▸
        "\u00b7": ".",  # ·
    }.get(symbol, symbol)


def _partition_tags(tags: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    namespaced: dict[str, list[str]] = {}
    unnamespaced: list[str] = []
    for tag in tags:
        namespace, value = parse_tag_namespace(tag)
        if namespace:
            namespaced.setdefault(namespace, []).append(value)
        else:
            unnamespaced.append(value)
    return namespaced, unnamespaced


def _namespaced_tag_lines(
    namespaced: Mapping[str, list[str]],
    tag_namespace_colors: Mapping[str, str],
) -> list[str]:
    lines: list[str] = []
    for namespace in sorted(namespaced):
        color = get_tag_color(f"{namespace}:", tag_namespace_colors)
        safe_namespace = escape_rich_text(namespace)
        values = ", ".join(escape_rich_text(value) for value in namespaced[namespace])
        lines.append(f"  [{color}]{safe_namespace}:[/] {values}")
    return lines


def _unnamespaced_tag_line(
    tags: list[str],
    tag_namespace_colors: Mapping[str, str],
) -> str:
    color = get_tag_color("", tag_namespace_colors)
    safe_tags = ", ".join(escape_rich_text(tag) for tag in tags)
    return f"  [{color}]{safe_tags}[/]"


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


def _detail_cache_key_for_state(state: DetailRenderState) -> tuple:
    """Build a stable, hashable cache key for rendered detail markup.

    ``state`` may contain mutable collections, rich objects, and very long text
    fields. This helper normalizes those pieces into hashable tuples or short
    digests so the result can be used as a compact ``dict`` key for the render
    cache. Any visible change in the detail pane should produce a different key.
    """
    if state.paper is None:
        return ("empty", _ACTIVE_DETAIL_GLYPH_MODE)

    # Convert mutable/unhashable structures to tuples.
    # SHA-256 digests are used for abstract_text and summary because these
    # strings can be thousands of characters long.  Including the raw text in
    # a tuple key would cause excessive memory usage and slow dict lookups.
    # A 64-character hex digest is unique enough for a 100-entry FIFO cache and
    # keeps each key small and fast to compare.
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
        state.collapsed_sections,
        state.detail_mode,
        tuple(sorted(state.theme_colors.items())),
        tuple(sorted(state.category_colors.items())),
        tuple(sorted(state.tag_namespace_colors.items())),
        _ACTIVE_DETAIL_GLYPH_MODE,
    )


def _detail_cache_key(
    state_or_paper: Paper | DetailRenderState,
    abstract_text: str | None = None,
    **legacy_kwargs: Any,
) -> tuple:
    """Compatibility wrapper for computing detail cache keys."""
    state = _coerce_detail_state(state_or_paper, abstract_text, legacy_kwargs)
    if state is None:
        return ("empty", _ACTIVE_DETAIL_GLYPH_MODE)
    return _detail_cache_key_for_state(state)


def _truncate_detail_text(text: str, max_len: int) -> str:
    """Shorten plain text for scan mode at a word boundary."""
    return truncate_at_word_boundary(text, max_len, ascii_mode=_ACTIVE_DETAIL_GLYPH_MODE == "ascii")


class PaperDetails(Static):
    """Widget to display full paper details."""

    def __init__(
        self,
        *,
        theme_colors: Mapping[str, str] | None = None,
        category_colors: Mapping[str, str] | None = None,
        tag_namespace_colors: Mapping[str, str] | None = None,
    ) -> None:
        """Initialise the detail pane with an empty markup cache."""
        super().__init__()
        self._paper: Paper | None = None
        self._detail_cache: dict[tuple, str] = {}
        self._detail_cache_order: list[tuple] = []
        self._theme_colors = dict(theme_colors or DEFAULT_THEME)
        self._category_colors = dict(category_colors or DEFAULT_CATEGORY_COLORS)
        self._tag_namespace_colors = dict(tag_namespace_colors or DEFAULT_TAG_NAMESPACE_COLORS)

    def update_state(self, state: DetailRenderState | None) -> None:
        """Update the displayed paper details from a complete render state."""
        resolved_state = _normalize_detail_state(state) if state is not None else None
        paper = resolved_state.paper if resolved_state is not None else None
        self._paper = paper
        if resolved_state is None or paper is None:
            self.update("[dim italic]Select a paper to view details[/]")
            return

        state = DetailRenderState(
            paper=paper,
            abstract_text=resolved_state.abstract_text,
            abstract_loading=resolved_state.abstract_loading,
            summary=resolved_state.summary,
            summary_loading=resolved_state.summary_loading,
            highlight_terms=resolved_state.highlight_terms,
            s2_data=resolved_state.s2_data,
            s2_loading=resolved_state.s2_loading,
            hf_data=resolved_state.hf_data,
            version_update=resolved_state.version_update,
            summary_mode=resolved_state.summary_mode,
            tags=resolved_state.tags,
            relevance=resolved_state.relevance,
            collapsed_sections=resolved_state.collapsed_sections,
            detail_mode=resolved_state.detail_mode,
            theme_colors=theme_colors_for(self, resolved_state.theme_colors),
            category_colors=category_colors_for(self, resolved_state.category_colors),
            tag_namespace_colors=tag_namespace_colors_for(
                self,
                resolved_state.tag_namespace_colors,
            ),
        )

        # Check detail cache before rebuilding markup
        cache_key = _detail_cache_key_for_state(state)
        cached = self._detail_cache.get(cache_key)
        if cached is not None:
            self.update(cached)
            return

        collapsed = set(state.collapsed_sections)

        sections = [
            self._render_title(paper),
            self._render_metadata(paper, state.category_colors),
            self._render_abstract(
                state.abstract_text,
                state.abstract_loading,
                list(state.highlight_terms),
                "abstract" in collapsed,
                state.detail_mode,
                state.theme_colors,
            ),
            self._render_authors(
                paper, "authors" in collapsed, state.detail_mode, state.theme_colors
            ),
            self._render_tags(
                list(state.tags),
                "tags" in collapsed,
                state.theme_colors,
                state.tag_namespace_colors,
            ),
            self._render_relevance(state.relevance, "relevance" in collapsed, state.theme_colors),
            self._render_summary(
                state.summary,
                state.summary_loading,
                state.summary_mode,
                "summary" in collapsed,
            ),
            self._render_s2(state.s2_data, state.s2_loading, "s2" in collapsed, state.theme_colors),
            self._render_hf(state.hf_data, "hf" in collapsed, state.theme_colors),
            self._render_version(
                paper, state.version_update, "version" in collapsed, state.theme_colors
            ),
            self._render_url(paper, state.theme_colors),
        ]
        markup = "\n\n".join(s for s in sections if s)

        # Store in cache with FIFO eviction
        if len(self._detail_cache) >= DETAIL_CACHE_MAX:
            oldest = self._detail_cache_order.pop(0)
            self._detail_cache.pop(oldest, None)
        self._detail_cache[cache_key] = markup
        self._detail_cache_order.append(cache_key)

        self.update(markup)

    def update_paper(
        self,
        paper: Paper | None,
        abstract_text: str | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Compatibility wrapper for the legacy detail update API."""
        self.update_state(_coerce_detail_state(paper, abstract_text, legacy_kwargs))

    # -- Section renderers ------------------------------------------------
    # Each _render_* method returns a Rich markup string for one detail pane
    # section, or "" when the section should be hidden.  The update_state()
    # orchestrator joins non-empty results with newlines.
    # ----------------------------------------------------------------------

    def _render_title(self, paper: Paper) -> str:
        """Return Rich markup for the paper title."""
        safe_title = escape_rich_text(paper.title)
        return f"[bold {theme_colors_for(self, self._theme_colors)['text']}]{safe_title}[/]"

    def _render_metadata(
        self,
        paper: Paper,
        category_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for arXiv ID, date, categories, and comments."""
        colors = theme_colors_for(self, self._theme_colors)
        resolved_category_colors = category_colors or category_colors_for(
            self, self._category_colors
        )
        safe_date = escape_rich_text(paper.date)
        safe_comments = escape_rich_text(paper.comments or "")
        lines = [
            f"  [bold {colors['accent']}]arXiv:[/] [{colors['purple']}]{paper.arxiv_id}[/]",
            f"  [bold {colors['accent']}]Date:[/] {safe_date}",
            f"  [bold {colors['accent']}]Categories:[/] {format_categories(paper.categories, resolved_category_colors)}",
        ]
        if paper.comments:
            lines.append(f"  [bold {colors['accent']}]Comments:[/] [dim]{safe_comments}[/]")
        return "\n".join(lines)

    def _render_abstract(
        self,
        abstract_text: str,
        loading: bool,
        highlight_terms: list[str] | None,
        is_collapsed: bool,
        detail_mode: str,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the abstract section with optional highlighting."""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        if is_collapsed:
            return f"[dim]{collapsed_glyph} Abstract[/]"
        if highlight_terms:
            abstract_body = (
                _truncate_detail_text(abstract_text, DETAIL_SCAN_ABSTRACT_LEN)
                if detail_mode == "scan"
                else abstract_text
            )
            safe_abstract = highlight_text(
                abstract_body,
                highlight_terms,
                resolved_theme_colors["accent"],
            )
        else:
            abstract_body = (
                _truncate_detail_text(abstract_text, DETAIL_SCAN_ABSTRACT_LEN)
                if detail_mode == "scan"
                else abstract_text
            )
            safe_abstract = escape_rich_text(abstract_body)
        lines = [f"[bold {resolved_theme_colors['orange']}]{expanded_glyph} Abstract[/]"]
        if loading:
            lines.append("  [dim italic]Loading abstract...[/]")
        elif abstract_text:
            lines.append(f"  [{resolved_theme_colors['text']}]{safe_abstract}[/]")
        else:
            lines.append("  [dim italic]No abstract available[/]")
        return "\n".join(lines)

    def _render_authors(
        self,
        paper: Paper,
        is_collapsed: bool,
        detail_mode: str,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the authors section."""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        if is_collapsed:
            return f"[dim]{collapsed_glyph} Authors[/]"
        authors_text = (
            _truncate_detail_text(paper.authors, DETAIL_SCAN_AUTHORS_LEN)
            if detail_mode == "scan"
            else paper.authors
        )
        safe_authors = escape_rich_text(authors_text)
        return (
            f"[bold {resolved_theme_colors['green']}]{expanded_glyph} Authors[/]\n"
            f"  [{resolved_theme_colors['text']}]{safe_authors}[/]"
        )

    def _render_tags(
        self,
        tags: list[str] | None,
        is_collapsed: bool,
        theme_colors: Mapping[str, str] | None = None,
        tag_namespace_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for user-assigned tags grouped by namespace."""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        resolved_tag_namespace_colors = tag_namespace_colors or tag_namespace_colors_for(
            self, self._tag_namespace_colors
        )
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        if not tags:
            return ""
        if is_collapsed:
            return f"[dim]{collapsed_glyph} Tags ({len(tags)})[/]"
        lines = [f"[bold {resolved_theme_colors['accent']}]{expanded_glyph} Tags[/]"]
        namespaced, unnamespaced = _partition_tags(tags)
        lines.extend(_namespaced_tag_lines(namespaced, resolved_tag_namespace_colors))
        if unnamespaced:
            lines.append(_unnamespaced_tag_line(unnamespaced, resolved_tag_namespace_colors))
        return "\n".join(lines)

    def _render_relevance(
        self,
        relevance: tuple[int, str] | None,
        is_collapsed: bool,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the relevance score and reason."""
        if relevance is None:
            return ""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        rel_score, rel_reason = relevance
        score_color, score_sym = _relevance_badge_parts(
            rel_score,
            theme_colors=resolved_theme_colors,
        )
        score_sym = _relevance_symbol_for_mode(score_sym)
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        if is_collapsed:
            return f"[dim]{collapsed_glyph} Relevance ({score_sym}{rel_score}/10)[/]"
        lines = [
            f"[bold {resolved_theme_colors['accent']}]{expanded_glyph} Relevance[/]",
            f"  [bold {resolved_theme_colors['accent']}]Score:[/] [{score_color}]{score_sym}{rel_score}/10[/]",
        ]
        if rel_reason:
            safe_reason = escape_rich_text(rel_reason)
            lines.append(f"  [{resolved_theme_colors['text']}]{safe_reason}[/]")
        return "\n".join(lines)

    def _render_summary(
        self,
        summary: str | None,
        summary_loading: bool,
        summary_mode: str,
        is_collapsed: bool,
    ) -> str:
        """Return Rich markup for the AI-generated summary section."""
        colors = theme_colors_for(self, self._theme_colors)
        summary_header = "AI Summary"
        if summary_mode:
            summary_header += f" ({summary_mode})"
        if not summary_loading and not summary:
            return ""
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        summary_prefix = _ACTIVE_DETAIL_GLYPHS["summary_prefix"]
        summary_loading_prefix = _ACTIVE_DETAIL_GLYPHS["summary_loading"]
        if is_collapsed:
            hint = " (loaded)" if summary else ""
            return f"[dim]{collapsed_glyph} {summary_header}{hint}[/]"
        if summary_loading:
            return (
                f"[bold {colors['purple']}]{expanded_glyph} {summary_prefix}{summary_header}[/]\n"
                f"  [dim italic]{summary_loading_prefix}Generating summary...[/]"
            )
        if summary:
            rendered_summary = format_summary_as_rich(summary, theme_colors=colors)
            return (
                f"[bold {colors['purple']}]{expanded_glyph} {summary_prefix}{summary_header}[/]\n"
                f"{rendered_summary}"
            )
        return ""

    def _render_s2(
        self,
        s2_data: SemanticScholarPaper | None,
        s2_loading: bool,
        is_collapsed: bool,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the Semantic Scholar data section."""
        if not s2_loading and not s2_data:
            return ""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        if is_collapsed:
            return self._render_s2_collapsed(s2_data, collapsed_glyph)
        if s2_loading:
            return (
                f"[bold {resolved_theme_colors['green']}]{expanded_glyph} Semantic Scholar[/]\n"
                "  [dim italic]Fetching data...[/]"
            )
        if s2_data:
            return "\n".join(
                self._render_s2_data_lines(s2_data, expanded_glyph, resolved_theme_colors)
            )
        return ""

    def _render_s2_collapsed(
        self,
        s2_data: SemanticScholarPaper | None,
        collapsed_glyph: str,
    ) -> str:
        hint = f" ({s2_data.citation_count} cites)" if s2_data else ""
        return f"[dim]{collapsed_glyph} Semantic Scholar{hint}[/]"

    def _render_s2_data_lines(
        self,
        s2_data: SemanticScholarPaper,
        expanded_glyph: str,
        theme_colors: Mapping[str, str],
    ) -> list[str]:
        lines = [
            f"[bold {theme_colors['green']}]{expanded_glyph} Semantic Scholar[/]",
            f"  [bold {theme_colors['accent']}]Citations:[/] {s2_data.citation_count}",
            f"  [bold {theme_colors['accent']}]Influential:[/] {s2_data.influential_citation_count}",
        ]
        if s2_data.fields_of_study:
            lines.append(self._render_s2_fields(s2_data, theme_colors))
        if s2_data.tldr:
            lines.append(self._render_s2_tldr(s2_data, theme_colors))
        return lines

    def _render_s2_fields(
        self,
        s2_data: SemanticScholarPaper,
        theme_colors: Mapping[str, str],
    ) -> str:
        fields = ", ".join(escape_rich_text(field) for field in s2_data.fields_of_study)
        return f"  [bold {theme_colors['accent']}]Fields:[/] {fields}"

    def _render_s2_tldr(
        self,
        s2_data: SemanticScholarPaper,
        theme_colors: Mapping[str, str],
    ) -> str:
        safe_tldr = escape_rich_text(s2_data.tldr)
        return f"  [bold {theme_colors['accent']}]TLDR:[/] [{theme_colors['text']}]{safe_tldr}[/]"

    def _render_hf(
        self,
        hf_data: HuggingFacePaper | None,
        is_collapsed: bool,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the HuggingFace metadata section."""
        if not hf_data:
            return ""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        hf_upvotes = _ACTIVE_DETAIL_GLYPHS["hf_upvotes"]
        if is_collapsed:
            return f"[dim]{collapsed_glyph} HuggingFace ({hf_upvotes}{hf_data.upvotes})[/]"
        lines = [f"[bold {resolved_theme_colors['orange']}]{expanded_glyph} HuggingFace[/]"]
        hf_parts = [f"  [bold {resolved_theme_colors['accent']}]Upvotes:[/] {hf_data.upvotes}"]
        if hf_data.num_comments > 0:
            hf_parts.append(
                f"  [bold {resolved_theme_colors['accent']}]Comments:[/] {hf_data.num_comments}"
            )
        lines.extend(hf_parts)
        if hf_data.github_repo:
            stars_str = f" ({hf_data.github_stars} stars)" if hf_data.github_stars else ""
            safe_repo = escape_rich_text(hf_data.github_repo)
            lines.append(
                f"  [bold {resolved_theme_colors['accent']}]GitHub:[/] {safe_repo}{stars_str}"
            )
        if hf_data.ai_keywords:
            kw = ", ".join(escape_rich_text(keyword) for keyword in hf_data.ai_keywords)
            lines.append(f"  [bold {resolved_theme_colors['accent']}]Keywords:[/] {kw}")
        if hf_data.ai_summary:
            safe_summary = escape_rich_text(hf_data.ai_summary)
            lines.append(
                f"  [bold {resolved_theme_colors['accent']}]AI Summary:[/] "
                f"[{resolved_theme_colors['text']}]{safe_summary}[/]"
            )
        return "\n".join(lines)

    def _render_version(
        self,
        paper: Paper,
        version_update: tuple[int, int] | None,
        is_collapsed: bool,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the version update section."""
        if version_update is None:
            return ""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        old_v, new_v = version_update
        collapsed_glyph = _ACTIVE_DETAIL_GLYPHS["collapsed"]
        expanded_glyph = _ACTIVE_DETAIL_GLYPHS["expanded"]
        version_arrow = _ACTIVE_DETAIL_GLYPHS["version_arrow"]
        if is_collapsed:
            return f"[dim]{collapsed_glyph} Version Update (v{old_v}{version_arrow}v{new_v})[/]"
        return (
            f"[bold {resolved_theme_colors['pink']}]{expanded_glyph} Version Update[/]\n"
            f"  [bold {resolved_theme_colors['accent']}]Updated:[/] "
            f"[{resolved_theme_colors['pink']}]v{old_v} {version_arrow} v{new_v}[/]\n"
            f"  [bold {resolved_theme_colors['accent']}]View diff:[/] "
            f"[{resolved_theme_colors['accent']}]https://arxivdiff.org/abs/{paper.arxiv_id}[/]"
        )

    def _render_url(
        self,
        paper: Paper,
        theme_colors: Mapping[str, str] | None = None,
    ) -> str:
        """Return Rich markup for the paper URL footer."""
        resolved_theme_colors = theme_colors or theme_colors_for(self, self._theme_colors)
        safe_url = escape_rich_text(paper.url)
        return (
            f"[bold {resolved_theme_colors['pink']}]URL[/]\n"
            f"  [{resolved_theme_colors['accent']}]{safe_url}[/]"
        )

    def clear_cache(self) -> None:
        """Clear the rendered markup cache."""
        self._detail_cache.clear()
        self._detail_cache_order.clear()

    @property
    def paper(self) -> Paper | None:
        """Return the currently displayed paper, if any."""
        return self._paper


__all__ = [
    "DETAIL_CACHE_MAX",
    "DetailRenderState",
    "PaperDetails",
    "_detail_cache_key",
    "set_ascii_glyphs",
]
