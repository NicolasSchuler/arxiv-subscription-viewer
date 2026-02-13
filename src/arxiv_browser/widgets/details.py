"""Detail pane widget for rendering selected paper metadata and enrichment."""

from __future__ import annotations

import hashlib

from textual.widgets import Static

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import Paper
from arxiv_browser.query import (
    escape_rich_text,
    format_categories,
    format_summary_as_rich,
    highlight_text,
)
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.themes import THEME_COLORS, get_tag_color, parse_tag_namespace
from arxiv_browser.widgets.listing import _relevance_badge_parts

# Maximum number of cached detail pane renderings (FIFO eviction)
DETAIL_CACHE_MAX = 100


def _detail_cache_key(
    paper: Paper,
    abstract_text: str | None,
    abstract_loading: bool = False,
    summary: str | None = None,
    summary_loading: bool = False,
    highlight_terms: list[str] | None = None,
    s2_data: SemanticScholarPaper | None = None,
    s2_loading: bool = False,
    hf_data: HuggingFacePaper | None = None,
    version_update: tuple[int, int] | None = None,
    summary_mode: str = "",
    tags: list[str] | None = None,
    relevance: tuple[int, str] | None = None,
    collapsed_sections: list[str] | None = None,
) -> tuple:
    """Build a stable cache key for rendered detail markup."""
    # Convert mutable/unhashable structures to tuples.
    abstract_digest = (
        hashlib.sha256(abstract_text.encode("utf-8")).hexdigest() if abstract_text else ""
    )
    summary_digest = hashlib.sha256(summary.encode("utf-8")).hexdigest() if summary else ""
    s2_key = (
        (
            s2_data.citation_count,
            s2_data.influential_citation_count,
            tuple(s2_data.fields_of_study),
            s2_data.tldr,
        )
        if s2_data is not None
        else None
    )
    hf_key = (
        (
            hf_data.upvotes,
            hf_data.num_comments,
            hf_data.github_repo,
            hf_data.github_stars,
            tuple(hf_data.ai_keywords),
            hf_data.ai_summary,
        )
        if hf_data is not None
        else None
    )
    return (
        paper.arxiv_id,
        paper.title,
        paper.authors,
        paper.date,
        paper.categories,
        paper.comments,
        paper.url,
        abstract_digest,
        abstract_loading,
        summary_digest,
        summary_loading,
        tuple(highlight_terms) if highlight_terms else (),
        s2_key,
        s2_loading,
        hf_key,
        version_update,
        summary_mode,
        tuple(tags) if tags else (),
        relevance,
        tuple(collapsed_sections) if collapsed_sections else (),
        tuple(sorted(THEME_COLORS.items())),
    )


class PaperDetails(Static):
    """Widget to display full paper details."""

    def __init__(self) -> None:
        super().__init__()
        self._paper: Paper | None = None
        self._detail_cache: dict[tuple, str] = {}
        self._detail_cache_order: list[tuple] = []

    def update_paper(
        self,
        paper: Paper | None,
        abstract_text: str | None = None,
        summary: str | None = None,
        summary_loading: bool = False,
        highlight_terms: list[str] | None = None,
        s2_data: SemanticScholarPaper | None = None,
        s2_loading: bool = False,
        hf_data: HuggingFacePaper | None = None,
        version_update: tuple[int, int] | None = None,
        summary_mode: str = "",
        tags: list[str] | None = None,
        relevance: tuple[int, str] | None = None,
        collapsed_sections: list[str] | None = None,
    ) -> None:
        """Update the displayed paper details."""
        self._paper = paper
        if paper is None:
            self.update("[dim italic]Select a paper to view details[/]")
            return

        loading = abstract_text is None and paper.abstract is None
        if abstract_text is None:
            abstract_text = paper.abstract or ""

        # Check detail cache before rebuilding markup
        cache_key = _detail_cache_key(
            paper,
            abstract_text,
            abstract_loading=loading,
            summary=summary,
            summary_loading=summary_loading,
            highlight_terms=highlight_terms,
            s2_data=s2_data,
            s2_loading=s2_loading,
            hf_data=hf_data,
            version_update=version_update,
            summary_mode=summary_mode,
            tags=tags,
            relevance=relevance,
            collapsed_sections=collapsed_sections,
        )
        cached = self._detail_cache.get(cache_key)
        if cached is not None:
            self.update(cached)
            return

        collapsed = set(collapsed_sections) if collapsed_sections else set()

        sections = [
            self._render_title(paper),
            self._render_metadata(paper),
            self._render_abstract(abstract_text, loading, highlight_terms, "abstract" in collapsed),
            self._render_authors(paper, "authors" in collapsed),
            self._render_tags(tags, "tags" in collapsed),
            self._render_relevance(relevance, "relevance" in collapsed),
            self._render_summary(summary, summary_loading, summary_mode, "summary" in collapsed),
            self._render_s2(s2_data, s2_loading, "s2" in collapsed),
            self._render_hf(hf_data, "hf" in collapsed),
            self._render_version(paper, version_update, "version" in collapsed),
            self._render_url(paper),
        ]
        markup = "\n".join(s for s in sections if s)

        # Store in cache with FIFO eviction
        if len(self._detail_cache) >= DETAIL_CACHE_MAX:
            oldest = self._detail_cache_order.pop(0)
            self._detail_cache.pop(oldest, None)
        self._detail_cache[cache_key] = markup
        self._detail_cache_order.append(cache_key)

        self.update(markup)

    def _render_title(self, paper: Paper) -> str:
        safe_title = escape_rich_text(paper.title)
        return f"[bold {THEME_COLORS['text']}]{safe_title}[/]"

    def _render_metadata(self, paper: Paper) -> str:
        safe_date = escape_rich_text(paper.date)
        safe_comments = escape_rich_text(paper.comments or "")
        lines = [
            f"  [bold {THEME_COLORS['accent']}]arXiv:[/] [{THEME_COLORS['purple']}]{paper.arxiv_id}[/]",
            f"  [bold {THEME_COLORS['accent']}]Date:[/] {safe_date}",
            f"  [bold {THEME_COLORS['accent']}]Categories:[/] {format_categories(paper.categories)}",
        ]
        if paper.comments:
            lines.append(f"  [bold {THEME_COLORS['accent']}]Comments:[/] [dim]{safe_comments}[/]")
        return "\n".join(lines)

    def _render_abstract(
        self,
        abstract_text: str,
        loading: bool,
        highlight_terms: list[str] | None,
        is_collapsed: bool,
    ) -> str:
        if is_collapsed:
            return "[dim]▸ Abstract[/]"
        if highlight_terms:
            safe_abstract = highlight_text(abstract_text, highlight_terms, THEME_COLORS["accent"])
        else:
            safe_abstract = escape_rich_text(abstract_text)
        lines = [f"[bold {THEME_COLORS['orange']}]▾ Abstract[/]"]
        if loading:
            lines.append("  [dim italic]Loading abstract...[/]")
        elif abstract_text:
            lines.append(f"  [{THEME_COLORS['text']}]{safe_abstract}[/]")
        else:
            lines.append("  [dim italic]No abstract available[/]")
        return "\n".join(lines)

    def _render_authors(self, paper: Paper, is_collapsed: bool) -> str:
        if is_collapsed:
            return "[dim]▸ Authors[/]"
        safe_authors = escape_rich_text(paper.authors)
        return (
            f"[bold {THEME_COLORS['green']}]▾ Authors[/]\n"
            f"  [{THEME_COLORS['text']}]{safe_authors}[/]"
        )

    def _render_tags(self, tags: list[str] | None, is_collapsed: bool) -> str:
        if not tags:
            return ""
        if is_collapsed:
            return f"[dim]▸ Tags ({len(tags)})[/]"
        lines = [f"[bold {THEME_COLORS['accent']}]▾ Tags[/]"]
        namespaced: dict[str, list[str]] = {}
        unnamespaced: list[str] = []
        for tag in tags:
            ns, val = parse_tag_namespace(tag)
            if ns:
                namespaced.setdefault(ns, []).append(val)
            else:
                unnamespaced.append(val)
        for ns in sorted(namespaced):
            color = get_tag_color(f"{ns}:")
            safe_ns = escape_rich_text(ns)
            vals = ", ".join(escape_rich_text(v) for v in namespaced[ns])
            lines.append(f"  [{color}]{safe_ns}:[/] {vals}")
        if unnamespaced:
            color = get_tag_color("")
            safe_unnamespaced = ", ".join(escape_rich_text(v) for v in unnamespaced)
            lines.append(f"  [{color}]{safe_unnamespaced}[/]")
        return "\n".join(lines)

    def _render_relevance(self, relevance: tuple[int, str] | None, is_collapsed: bool) -> str:
        if relevance is None:
            return ""
        rel_score, rel_reason = relevance
        score_color, score_sym = _relevance_badge_parts(rel_score)
        if is_collapsed:
            return f"[dim]▸ Relevance ({score_sym}{rel_score}/10)[/]"
        lines = [
            f"[bold {THEME_COLORS['accent']}]▾ Relevance[/]",
            f"  [bold {THEME_COLORS['accent']}]Score:[/] [{score_color}]{score_sym}{rel_score}/10[/]",
        ]
        if rel_reason:
            safe_reason = escape_rich_text(rel_reason)
            lines.append(f"  [{THEME_COLORS['text']}]{safe_reason}[/]")
        return "\n".join(lines)

    def _render_summary(
        self,
        summary: str | None,
        summary_loading: bool,
        summary_mode: str,
        is_collapsed: bool,
    ) -> str:
        summary_header = "AI Summary"
        if summary_mode:
            summary_header += f" ({summary_mode})"
        if not summary_loading and not summary:
            return ""
        if is_collapsed:
            hint = " (loaded)" if summary else ""
            return f"[dim]▸ {summary_header}{hint}[/]"
        if summary_loading:
            return (
                f"[bold {THEME_COLORS['purple']}]▾ 🤖 {summary_header}[/]\n"
                "  [dim italic]⏳ Generating summary...[/]"
            )
        if summary:
            rendered_summary = format_summary_as_rich(summary)
            return f"[bold {THEME_COLORS['purple']}]▾ 🤖 {summary_header}[/]\n{rendered_summary}"
        return ""

    def _render_s2(
        self,
        s2_data: SemanticScholarPaper | None,
        s2_loading: bool,
        is_collapsed: bool,
    ) -> str:
        if not s2_loading and not s2_data:
            return ""
        if is_collapsed:
            hint = ""
            if s2_data:
                hint = f" ({s2_data.citation_count} cites)"
            return f"[dim]▸ Semantic Scholar{hint}[/]"
        if s2_loading:
            return (
                f"[bold {THEME_COLORS['green']}]▾ Semantic Scholar[/]\n"
                "  [dim italic]Fetching data...[/]"
            )
        if s2_data:
            lines = [
                f"[bold {THEME_COLORS['green']}]▾ Semantic Scholar[/]",
                (
                    f"  [bold {THEME_COLORS['accent']}]Citations:[/] {s2_data.citation_count}"
                    f"  [bold {THEME_COLORS['accent']}]Influential:[/] {s2_data.influential_citation_count}"
                ),
            ]
            if s2_data.fields_of_study:
                fos = ", ".join(escape_rich_text(field) for field in s2_data.fields_of_study)
                lines.append(f"  [bold {THEME_COLORS['accent']}]Fields:[/] {fos}")
            if s2_data.tldr:
                safe_tldr = escape_rich_text(s2_data.tldr)
                lines.append(
                    f"  [bold {THEME_COLORS['accent']}]TLDR:[/] [{THEME_COLORS['text']}]{safe_tldr}[/]"
                )
            return "\n".join(lines)
        return ""

    def _render_hf(self, hf_data: HuggingFacePaper | None, is_collapsed: bool) -> str:
        if not hf_data:
            return ""
        if is_collapsed:
            return f"[dim]▸ HuggingFace (↑{hf_data.upvotes})[/]"
        lines = [f"[bold {THEME_COLORS['orange']}]▾ HuggingFace[/]"]
        hf_parts = [f"  [bold {THEME_COLORS['accent']}]Upvotes:[/] {hf_data.upvotes}"]
        if hf_data.num_comments > 0:
            hf_parts.append(f"  [bold {THEME_COLORS['accent']}]Comments:[/] {hf_data.num_comments}")
        lines.append("".join(hf_parts))
        if hf_data.github_repo:
            stars_str = f" ({hf_data.github_stars} stars)" if hf_data.github_stars else ""
            safe_repo = escape_rich_text(hf_data.github_repo)
            lines.append(f"  [bold {THEME_COLORS['accent']}]GitHub:[/] {safe_repo}{stars_str}")
        if hf_data.ai_keywords:
            kw = ", ".join(escape_rich_text(keyword) for keyword in hf_data.ai_keywords)
            lines.append(f"  [bold {THEME_COLORS['accent']}]Keywords:[/] {kw}")
        if hf_data.ai_summary:
            safe_summary = escape_rich_text(hf_data.ai_summary)
            lines.append(
                f"  [bold {THEME_COLORS['accent']}]AI Summary:[/] [{THEME_COLORS['text']}]{safe_summary}[/]"
            )
        return "\n".join(lines)

    def _render_version(
        self,
        paper: Paper,
        version_update: tuple[int, int] | None,
        is_collapsed: bool,
    ) -> str:
        if version_update is None:
            return ""
        old_v, new_v = version_update
        if is_collapsed:
            return f"[dim]▸ Version Update (v{old_v}→v{new_v})[/]"
        return (
            f"[bold {THEME_COLORS['pink']}]▾ Version Update[/]\n"
            f"  [bold {THEME_COLORS['accent']}]Updated:[/] [{THEME_COLORS['pink']}]v{old_v} → v{new_v}[/]\n"
            f"  [bold {THEME_COLORS['accent']}]View diff:[/] [{THEME_COLORS['accent']}]https://arxivdiff.org/abs/{paper.arxiv_id}[/]"
        )

    def _render_url(self, paper: Paper) -> str:
        safe_url = escape_rich_text(paper.url)
        return f"[bold {THEME_COLORS['pink']}]URL[/]\n  [{THEME_COLORS['accent']}]{safe_url}[/]"

    def clear_cache(self) -> None:
        """Clear the rendered markup cache."""
        self._detail_cache.clear()
        self._detail_cache_order.clear()

    @property
    def paper(self) -> Paper | None:
        return self._paper


__all__ = [
    "DETAIL_CACHE_MAX",
    "PaperDetails",
]
