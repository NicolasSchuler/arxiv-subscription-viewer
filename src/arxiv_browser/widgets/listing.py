"""List rendering helpers and widgets for paper entries."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widgets import ListItem, Static

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import Paper, PaperMetadata
from arxiv_browser.query import escape_rich_text, format_categories, highlight_text
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.themes import THEME_COLORS, get_tag_color

PREVIEW_ABSTRACT_MAX_LEN = 150  # Max abstract preview length in list items

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


def set_ascii_icons(enabled: bool) -> None:
    """Switch list indicators between Unicode and ASCII modes."""
    global _ACTIVE_ICON_SET
    _ACTIVE_ICON_SET = _ICON_SETS["ascii"] if enabled else _ICON_SETS["unicode"]


def _render_title_line(
    paper: Paper,
    selected: bool,
    metadata: PaperMetadata | None,
    watched: bool,
    ht: dict[str, list[str]],
) -> str:
    """Build the title line with selection/watch/star/read indicators."""
    prefix_parts: list[str] = []
    if selected:
        prefix_parts.append(f"[{THEME_COLORS['green']}]{_ACTIVE_ICON_SET['selected']}[/]")
    if watched:
        prefix_parts.append(f"[{THEME_COLORS['orange']}]{_ACTIVE_ICON_SET['watched']}[/]")
    if metadata and metadata.starred:
        prefix_parts.append(f"[{THEME_COLORS['yellow']}]{_ACTIVE_ICON_SET['starred']}[/]")
    if metadata and metadata.is_read:
        prefix_parts.append(f"[{THEME_COLORS['muted']}]{_ACTIVE_ICON_SET['read']}[/]")
    prefix = " ".join(prefix_parts)

    title_text = highlight_text(paper.title, ht.get("title", []), THEME_COLORS["accent"])
    if metadata and metadata.is_read:
        title_text = f"[dim]{title_text}[/]"
    return f"{prefix} {title_text}" if prefix else title_text


def _relevance_badge_parts(score: int) -> tuple[str, str]:
    """Return (color, symbol) for a relevance score badge.

    Uses distinct symbols alongside color so the badge is accessible
    to colorblind users (WCAG 1.4.1 — Use of Color).
    """
    if score >= 8:
        return THEME_COLORS["green"], "\u2605"  # ★
    if score >= 5:
        return THEME_COLORS["yellow"], "\u25b8"  # ▸
    return THEME_COLORS["muted"], "\u00b7"  # ·


def _render_meta_badges(
    paper: Paper,
    metadata: PaperMetadata | None,
    s2_data: SemanticScholarPaper | None,
    hf_data: HuggingFacePaper | None,
    version_update: tuple[int, int] | None,
    relevance_score: tuple[int, str] | None,
) -> str:
    """Build the meta line with arxiv_id, categories, and badges."""
    parts: list[str] = []
    if paper.source == "api":
        parts.append(f"[{THEME_COLORS['orange']}]API[/]")
    parts.extend([f"[dim]{paper.arxiv_id}[/]", format_categories(paper.categories)])
    if metadata and metadata.tags:
        tag_str = " ".join(
            f"[{get_tag_color(tag)}]#{escape_rich_text(tag)}[/]" for tag in metadata.tags
        )
        parts.append(tag_str)
    if s2_data is not None:
        parts.append(f"[{THEME_COLORS['green']}]C{s2_data.citation_count}[/]")
    if hf_data is not None:
        parts.append(f"[{THEME_COLORS['orange']}]\u2191{hf_data.upvotes}[/]")
    if version_update is not None:
        old_v, new_v = version_update
        parts.append(f"[{THEME_COLORS['pink']}]v{old_v}\u2192v{new_v}[/]")
    if relevance_score is not None:
        score, _ = relevance_score
        color, sym = _relevance_badge_parts(score)
        parts.append(f"[{color}]{sym}{score}/10[/]")
    return "  ".join(parts)


def _render_abstract_preview(abstract_text: str | None, ht: dict[str, list[str]]) -> str:
    """Build the abstract preview line for the paper list."""
    if abstract_text is None:
        return "[dim italic]Loading abstract...[/]"
    if not abstract_text:
        return "[dim italic]No abstract available[/]"
    if len(abstract_text) <= PREVIEW_ABSTRACT_MAX_LEN:
        highlighted = highlight_text(abstract_text, ht.get("abstract", []), THEME_COLORS["accent"])
        return f"[dim italic]{highlighted}[/]"
    truncated = abstract_text[:PREVIEW_ABSTRACT_MAX_LEN].rsplit(" ", 1)[0]
    highlighted = highlight_text(truncated, ht.get("abstract", []), THEME_COLORS["accent"])
    return f"[dim italic]{highlighted}...[/]"


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
) -> str:
    """Render a paper as Rich markup for OptionList display."""
    ht = highlight_terms or {"title": [], "author": [], "abstract": []}

    lines = [
        _render_title_line(paper, selected, metadata, watched, ht),
        highlight_text(paper.authors, ht.get("author", []), THEME_COLORS["accent"]),
        _render_meta_badges(paper, metadata, s2_data, hf_data, version_update, relevance_score),
    ]

    if show_preview:
        lines.append(_render_abstract_preview(abstract_text, ht))

    return "\n".join(lines)


class PaperListItem(ListItem):
    """A list item displaying a paper title and URL."""

    def __init__(
        self,
        paper: Paper,
        selected: bool = False,
        metadata: PaperMetadata | None = None,
        watched: bool = False,
        show_preview: bool = False,
        abstract_text: str | None = None,
        highlight_terms: dict[str, list[str]] | None = None,
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

        # Selection indicator
        if self._selected:
            prefix_parts.append(f"[{THEME_COLORS['green']}]{_ACTIVE_ICON_SET['selected']}[/]")

        # Watched indicator
        if self._watched:
            prefix_parts.append(f"[{THEME_COLORS['orange']}]{_ACTIVE_ICON_SET['watched']}[/]")

        # Starred indicator
        if self._metadata and self._metadata.starred:
            prefix_parts.append(f"[{THEME_COLORS['yellow']}]{_ACTIVE_ICON_SET['starred']}[/]")

        # Read indicator
        if self._metadata and self._metadata.is_read:
            prefix_parts.append(f"[{THEME_COLORS['muted']}]{_ACTIVE_ICON_SET['read']}[/]")

        prefix = " ".join(prefix_parts)
        title_text = highlight_text(
            self.paper.title,
            self._highlight_terms.get("title", []),
            THEME_COLORS["accent"],
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
        return highlight_text(
            self.paper.authors,
            self._highlight_terms.get("author", []),
            THEME_COLORS["accent"],
        )

    def _get_meta_text(self) -> str:
        """Get the formatted metadata text."""
        parts = []
        if self.paper.source == "api":
            parts.append(f"[{THEME_COLORS['orange']}]API[/]")
        parts.extend(
            [
                f"[dim]{self.paper.arxiv_id}[/]",
                format_categories(self.paper.categories),
            ]
        )

        # Show tags if present (namespace-colored)
        if self._metadata and self._metadata.tags:
            tag_str = " ".join(
                f"[{get_tag_color(tag)}]#{escape_rich_text(tag)}[/]" for tag in self._metadata.tags
            )
            parts.append(tag_str)

        # S2 citation badge
        if self._s2_data is not None:
            parts.append(f"[{THEME_COLORS['green']}]C{self._s2_data.citation_count}[/]")

        # HF trending badge
        if self._hf_data is not None:
            parts.append(f"[{THEME_COLORS['orange']}]\u2191{self._hf_data.upvotes}[/]")

        # Version update badge
        if self._version_update is not None:
            old_v, new_v = self._version_update
            parts.append(f"[{THEME_COLORS['pink']}]v{old_v}\u2192v{new_v}[/]")

        # Relevance score badge
        if self._relevance_score is not None:
            score, _ = self._relevance_score
            color, sym = _relevance_badge_parts(score)
            parts.append(f"[{color}]{sym}{score}/10[/]")

        return "  ".join(parts)

    def _get_preview_text(self) -> str:
        """Get truncated abstract preview text.

        Returns formatted Rich markup for the abstract preview.
        Handles empty abstracts and truncates at word boundaries.
        """
        abstract = self._abstract_text
        if abstract is None:
            return "[dim italic]Loading abstract...[/]"
        if not abstract:
            return "[dim italic]No abstract available[/]"
        if len(abstract) <= PREVIEW_ABSTRACT_MAX_LEN:
            highlighted = highlight_text(
                abstract,
                self._highlight_terms.get("abstract", []),
                THEME_COLORS["accent"],
            )
            return f"[dim italic]{highlighted}[/]"
        # Truncate at word boundary for cleaner display
        truncated = abstract[:PREVIEW_ABSTRACT_MAX_LEN].rsplit(" ", 1)[0]
        highlighted = highlight_text(
            truncated,
            self._highlight_terms.get("abstract", []),
            THEME_COLORS["accent"],
        )
        return f"[dim italic]{highlighted}...[/]"

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
