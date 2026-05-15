"""Status bar helpers and footer binding builders for chrome widgets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import DEFAULT_THEME

_CHROME_GLYPH_SETS: dict[str, dict[str, str]] = {
    "unicode": {
        "pill_remove": "\u00d7",
        "footer_arrows": "↑↓",
        "separator": "│",
    },
    "ascii": {
        "pill_remove": "x",
        "footer_arrows": "^v",
        "separator": "|",
    },
}
_ACTIVE_CHROME_GLYPHS = _CHROME_GLYPH_SETS["unicode"]


def set_ascii_glyphs(enabled: bool) -> None:
    """Switch chrome glyphs between Unicode and ASCII modes."""
    global _ACTIVE_CHROME_GLYPHS
    _ACTIVE_CHROME_GLYPHS = (
        _CHROME_GLYPH_SETS["ascii"] if enabled else _CHROME_GLYPH_SETS["unicode"]
    )


def get_filter_pill_remove_glyph() -> str:
    """Return the symbol used for removable filter pills."""
    return _ACTIVE_CHROME_GLYPHS["pill_remove"]


_SELECTION_FOOTER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("o", "open"),
    ("c", "copy"),
    ("r", "read"),
    ("x", "star"),
    ("t", "tags"),
    ("E", "export"),
    ("d", "download"),
    ("u", "clear"),
    ("?", "help"),
)

_SEARCH_FOOTER_BINDINGS_BASE: tuple[tuple[str, str], ...] = (
    ("type to search", ""),
    ("Enter", "apply"),
    ("Esc", "close"),
    # Arrow hint is inserted dynamically via build_search_footer_bindings()
)

_API_FOOTER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("[/]", "page"),
    ("Esc/Ctrl+e", "exit"),
    ("A", "new query"),
    ("o", "open"),
    ("?", "help"),
)

_DETAIL_FOCUS_FOOTER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("Tab", "list"),
    ("j/k", "scroll"),
    ("a", "annotate"),
    ("v", "density"),
    ("Ctrl+d", "sections"),
    ("?", "help"),
)


def build_selection_footer_base_bindings() -> list[tuple[str, str]]:
    """Return canonical selection-mode footer hints."""
    return list(_SELECTION_FOOTER_BINDINGS)


def build_search_footer_bindings() -> list[tuple[str, str]]:
    """Return canonical search-mode footer hints."""
    bindings = list(_SEARCH_FOOTER_BINDINGS_BASE)
    bindings.append((_ACTIVE_CHROME_GLYPHS["footer_arrows"], "move"))
    bindings.append(("?", "help"))
    return bindings


def build_api_footer_bindings() -> list[tuple[str, str]]:
    """Return canonical API-mode footer hints."""
    return list(_API_FOOTER_BINDINGS)


def build_detail_focus_footer_bindings() -> list[tuple[str, str]]:
    """Return canonical detail-pane focus footer hints."""
    return list(_DETAIL_FOCUS_FOOTER_BINDINGS)


@dataclass(frozen=True, slots=True)
class FooterModeBadgeState:
    """Semantic mode state for the compact footer badge."""

    relevance_scoring_active: bool
    version_checking: bool
    search_visible: bool
    in_arxiv_api_mode: bool
    selected_count: int
    detail_focus: bool = False


@dataclass(frozen=True, slots=True)
class StatusBarState:
    """Semantic status-bar state for the current app view."""

    total: int
    filtered: int
    query: str
    watch_filter_active: bool
    selected_count: int
    sort_label: str
    in_arxiv_api_mode: bool
    api_page: int | None
    arxiv_api_loading: bool
    show_abstract_preview: bool
    s2_active: bool
    s2_loading: bool
    s2_count: int
    s2_api_error: bool = False
    hf_active: bool = False
    hf_loading: bool = False
    hf_match_count: int = 0
    hf_api_error: bool = False
    version_checking: bool = False
    version_update_count: int = 0
    enrichment_progress: tuple[str, int, int] | None = None
    reading_velocity: tuple[float, ...] = ()
    category_distribution: tuple[tuple[str, int], ...] = ()
    detail_focus: bool = False
    max_width: int | None = None
    theme_colors: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_THEME))


def _coerce_status_bar_state(
    state: StatusBarState | None,
    legacy_kwargs: Mapping[str, Any],
) -> StatusBarState:
    """Accept either the new status-state object or the legacy kwargs shape."""
    if state is not None:
        if legacy_kwargs:
            raise TypeError("StatusBarState cannot be combined with legacy keyword args")
        return StatusBarState(
            total=state.total,
            filtered=state.filtered,
            query=state.query,
            watch_filter_active=state.watch_filter_active,
            selected_count=state.selected_count,
            sort_label=state.sort_label,
            in_arxiv_api_mode=state.in_arxiv_api_mode,
            api_page=state.api_page,
            arxiv_api_loading=state.arxiv_api_loading,
            show_abstract_preview=state.show_abstract_preview,
            s2_active=state.s2_active,
            s2_loading=state.s2_loading,
            s2_count=state.s2_count,
            s2_api_error=state.s2_api_error,
            hf_active=state.hf_active,
            hf_loading=state.hf_loading,
            hf_match_count=state.hf_match_count,
            hf_api_error=state.hf_api_error,
            version_checking=state.version_checking,
            version_update_count=state.version_update_count,
            enrichment_progress=state.enrichment_progress,
            reading_velocity=tuple(state.reading_velocity or ()),
            category_distribution=tuple(state.category_distribution or ()),
            detail_focus=state.detail_focus,
            max_width=state.max_width,
            theme_colors=dict(state.theme_colors or DEFAULT_THEME),
        )
    kwargs = dict(legacy_kwargs)
    kwargs["theme_colors"] = dict(kwargs.get("theme_colors") or DEFAULT_THEME)
    return StatusBarState(**kwargs)


def build_selection_footer_bindings(selected_count: int) -> list[tuple[str, str]]:
    """Build selection-mode footer bindings with dynamic open(n) label."""
    bindings = build_selection_footer_base_bindings()
    if bindings:
        bindings[0] = ("o", f"open({selected_count})")
    return bindings


def build_browse_footer_bindings(
    *,
    s2_active: bool,
    has_starred: bool,
    llm_configured: bool,
    has_history_navigation: bool,
) -> list[tuple[str, str]]:
    """Build the default browsing footer with deterministic priority."""
    _ = (s2_active, has_starred, llm_configured)
    slot_a = ("[/]", "dates") if has_history_navigation else ("n", "notes")
    return [
        ("/", "search"),
        ("Space", "select"),
        ("o", "open"),
        ("s", "sort"),
        ("r", "read"),
        slot_a,
        ("E", "export"),
        ("Ctrl+p", "commands"),
        ("?", "help"),
    ]


def build_footer_mode_badge(
    state: FooterModeBadgeState | None = None,
    theme_colors: Mapping[str, str] | None = None,
    **legacy_kwargs: Any,
) -> str:
    """Build Rich-markup mode badge text for footer state."""
    if state is not None:
        if legacy_kwargs:
            raise TypeError("FooterModeBadgeState cannot be combined with legacy kwargs")
        resolved_state = state
    else:
        resolved_state = FooterModeBadgeState(**legacy_kwargs)
    colors = theme_colors or DEFAULT_THEME
    pink = colors["pink"]
    accent = colors["accent"]
    orange = colors["orange"]
    green = colors["green"]
    panel_alt = colors["panel_alt"]
    if resolved_state.relevance_scoring_active:
        return f"[bold {pink} on {panel_alt}] SCORING [/]"
    if resolved_state.version_checking:
        return f"[bold {pink} on {panel_alt}] VERSIONS [/]"
    if resolved_state.search_visible:
        return f"[bold {accent} on {panel_alt}] SEARCH [/]"
    if resolved_state.in_arxiv_api_mode:
        return f"[bold {orange} on {panel_alt}] API [/]"
    if resolved_state.detail_focus:
        return f"[bold {accent} on {panel_alt}] DETAILS [/]"
    if resolved_state.selected_count > 0:
        return f"[bold {green} on {panel_alt}] {resolved_state.selected_count} SEL [/]"
    return ""


def build_status_bar_text(
    state: StatusBarState | None = None,
    **legacy_kwargs: Any,
) -> str:
    """Build semantic status bar text for current UI/application state."""
    resolved_state = _coerce_status_bar_state(state, legacy_kwargs)
    if resolved_state.max_width is not None and resolved_state.max_width <= 100:
        compact_parts = _build_compact_status_parts(resolved_state)
        return _render_compact_status(compact_parts, resolved_state.max_width)

    parts = _build_full_status_parts(resolved_state)
    sep = _ACTIVE_CHROME_GLYPHS["separator"]
    rendered = f" [dim]{sep}[/] ".join(parts)
    return _truncate_rich_text(rendered, resolved_state.max_width)


def _compact_primary_segment(
    *,
    total: int,
    filtered: int,
    query: str,
    watch_filter_active: bool,
) -> str:
    """Build the first compact segment (query/watch/default)."""
    if query:
        return f"{filtered}/{total} match"
    if watch_filter_active:
        return f"{filtered}/{total} watched"
    return f"{total} papers"


def _full_primary_segment(
    *,
    total: int,
    filtered: int,
    query: str,
    watch_filter_active: bool,
    theme_colors: Mapping[str, str],
) -> str:
    """Build the first rich segment (query/watch/default)."""
    if query:
        truncated_query = query if len(query) <= 30 else query[:27] + "..."
        safe_query = escape_rich_text(truncated_query)
        return (
            f"[{theme_colors['accent']}]{filtered}[/][dim]/{total} matching [/]"
            f'[{theme_colors["accent"]}]"{safe_query}"[/]'
        )
    if watch_filter_active:
        return f"[{theme_colors['orange']}]{filtered}[/][dim]/{total} watched[/]"
    return f"[dim]{total} papers[/]"


def _compact_flag_segment(
    *,
    active: bool,
    loading: bool,
    count: int,
    label: str,
    api_error: bool = False,
) -> str | None:
    """Return compact flag text like S2/HF status, or None when inactive."""
    if not active:
        return None
    if api_error:
        return f"{label}:err"
    if loading:
        return f"{label} Loading..."
    if count > 0:
        return f"{label}:{count}"
    return label


def _build_compact_status_parts(state: StatusBarState) -> list[str]:
    """Build compact status tokens for narrow terminals."""
    parts = [
        _compact_primary_segment(
            total=state.total,
            filtered=state.filtered,
            query=state.query,
            watch_filter_active=state.watch_filter_active,
        )
    ]
    if state.in_arxiv_api_mode and state.api_page is not None:
        api_segment = f"API p{state.api_page}"
        if state.arxiv_api_loading:
            api_segment += " loading"
        parts.append(api_segment)
    elif state.arxiv_api_loading:
        parts.append("API loading")

    if state.selected_count > 0:
        parts.append(f"{state.selected_count} sel")
    if state.detail_focus:
        parts.append("details")

    parts.append(f"sort:{state.sort_label}")

    s2_segment = _compact_flag_segment(
        active=state.s2_active,
        loading=state.s2_loading,
        count=state.s2_count,
        label="S2",
        api_error=state.s2_api_error,
    )
    if s2_segment:
        parts.append(s2_segment)

    if state.max_width is not None and state.max_width >= 90:
        hf_segment = _compact_flag_segment(
            active=state.hf_active,
            loading=state.hf_loading,
            count=state.hf_match_count,
            label="HF",
            api_error=state.hf_api_error,
        )
        if hf_segment:
            parts.append(hf_segment)

    _ = (
        state.show_abstract_preview,
        state.version_checking,
        state.version_update_count,
    )
    return parts


def _build_full_status_parts(state: StatusBarState) -> list[str]:
    """Build rich status tokens for regular widths."""
    parts = [
        _full_primary_segment(
            total=state.total,
            filtered=state.filtered,
            query=state.query,
            watch_filter_active=state.watch_filter_active,
            theme_colors=state.theme_colors,
        ),
        f"[dim]Sort: {state.sort_label}[/]",
    ]
    if state.selected_count > 0:
        parts.insert(
            1,
            f"[bold {state.theme_colors['green']}]{state.selected_count} selected[/]",
        )
    if state.detail_focus:
        parts.insert(1, f"[{state.theme_colors['accent_alt']}]Details focus[/]")
    if state.in_arxiv_api_mode and state.api_page is not None:
        parts.extend(_full_api_segments(state))
    if state.show_abstract_preview:
        parts.append(f"[{state.theme_colors['purple']}]Preview[/]")

    render_visuals = _should_render_visual_status(state)
    visual_segments = _full_visual_segments(state) if render_visuals else []
    parts.extend(visual_segments)
    parts.extend(_full_flag_segments(state))
    suppress_version_checking = (
        render_visuals
        and state.enrichment_progress is not None
        and state.enrichment_progress[0] == "Versions"
    )
    parts.extend(_full_version_segments(state, suppress_checking=suppress_version_checking))
    return parts


def _full_api_segments(state: StatusBarState) -> list[str]:
    """Return full-width arXiv API status segments."""
    segments = [
        f"[{state.theme_colors['orange']}]API[/]",
        f"[dim]Page: {state.api_page}[/]",
    ]
    if state.arxiv_api_loading:
        segments.append(f"[{state.theme_colors['orange']}]Loading...[/]")
    return segments


def _full_flag_segments(state: StatusBarState) -> list[str]:
    """Return S2/HF full-width status segments."""
    segments = []
    if state.s2_active:
        segments.append(
            _rich_flag_segment(
                loading=state.s2_loading,
                count=state.s2_count,
                label="S2",
                color=state.theme_colors["green"],
                error_color=state.theme_colors["orange"],
                api_error=state.s2_api_error,
            )
        )
    if state.hf_active:
        segments.append(
            _rich_flag_segment(
                loading=state.hf_loading,
                count=state.hf_match_count,
                label="HF",
                color=state.theme_colors["orange"],
                error_color=state.theme_colors["orange"],
                api_error=state.hf_api_error,
            )
        )
    return segments


def _rich_flag_segment(
    *,
    loading: bool,
    count: int,
    label: str,
    color: str,
    error_color: str,
    api_error: bool = False,
) -> str:
    """Return a full-width Rich flag segment like S2/HF status."""
    if api_error:
        return f"[{error_color}]{label}:err[/]"
    if loading:
        return f"[{color}]{label} loading...[/]"
    if count > 0:
        return f"[{color}]{label}:{count}[/]"
    return f"[{color}]{label}[/]"


def _full_version_segments(state: StatusBarState, *, suppress_checking: bool = False) -> list[str]:
    """Return full-width version-checking status segments."""
    color = state.theme_colors["pink"]
    if state.version_checking and not suppress_checking:
        return [f"[{color}]Checking versions...[/]"]
    if state.version_update_count > 0:
        return [f"[{color}]{state.version_update_count} updated[/]"]
    return []


def _should_render_visual_status(state: StatusBarState) -> bool:
    """Return whether the status bar has enough room for visual density tokens."""
    return state.max_width is None or state.max_width >= 120


def _full_visual_segments(state: StatusBarState) -> list[str]:
    """Return width-aware sparkline/histogram status segments."""
    segments: list[str] = []
    if state.enrichment_progress is not None:
        label, current, total = state.enrichment_progress
        segments.append(
            f"[{state.theme_colors['accent']}]{label} "
            f"{_progress_sparkline(current, total)} {current}/{total}[/]"
        )
    if state.reading_velocity:
        rate = sum(state.reading_velocity) / max(1, len(state.reading_velocity))
        if rate > 0:
            segments.append(
                f"[{state.theme_colors['green']}]Read/m "
                f"{_sparkline(state.reading_velocity)} {rate:.1f}[/]"
            )
    if state.category_distribution:
        histogram = _category_histogram(state.category_distribution)
        if histogram:
            segments.append(f"[{state.theme_colors['purple']}]Cats {histogram}[/]")
    return segments


def _sparkline(values: tuple[float, ...]) -> str:
    """Render a small Unicode or ASCII sparkline."""
    if not values:
        return ""
    from arxiv_browser._ascii import is_ascii_mode

    glyphs = " .:-=+*#@" if is_ascii_mode() else "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    maximum = max(values)
    if maximum <= 0:
        return glyphs[0] * len(values)
    top_index = len(glyphs) - 1
    return "".join(glyphs[min(top_index, round(value / maximum * top_index))] for value in values)


def _progress_sparkline(current: int, total: int, width: int = 8) -> str:
    """Render compact progress as a visual status token."""
    from arxiv_browser._ascii import is_ascii_mode

    current = max(0, current)
    total = max(0, total)
    ratio = 0.0 if total <= 0 else min(1.0, current / total)
    filled = int(ratio * width)
    if is_ascii_mode():
        return "#" * filled + "-" * (width - filled)
    return "\u2588" * filled + "\u2581" * (width - filled)


def _category_histogram(categories: tuple[tuple[str, int], ...]) -> str:
    """Render top category counts as a compact histogram."""
    top = tuple((label, count) for label, count in categories[:3] if count > 0)
    if not top:
        return ""
    maximum = max(count for _, count in top)
    return " ".join(
        f"{escape_rich_text(label)}{_progress_sparkline(count, maximum, width=3)}"
        for label, count in top
    )


def _render_compact_status(parts: list[str], max_width: int) -> str:
    """Render compact parts and shrink if necessary."""
    compact = " | ".join(parts)
    if len(compact) <= max_width:
        return compact

    while len(parts) > 1 and len(" | ".join(parts) + " ...") > max_width:
        parts.pop()
    return " | ".join(parts) + " ..."


def _truncate_rich_text(text: str, max_width: int | None) -> str:
    """Truncate rendered Rich markup by visible width when constrained.

    Walks the string preserving Rich ``[tag]`` sequences (which contribute
    zero visible width) so that formatting is retained in the truncated
    output.  Escaped brackets (``\\[``) are correctly counted as visible
    characters.
    """
    if max_width is None or max_width <= 0:
        return text
    if _rich_visible_width(text) <= max_width:
        return text
    target = max(0, max_width - 3)
    result: list[str] = []
    visible_count = 0
    i = 0
    n = len(text)
    while i < n and visible_count < target:
        chunk, next_i, visible_delta = _next_rich_text_chunk(text, i)
        result.append(chunk)
        visible_count += visible_delta
        i = next_i
    return "".join(result) + "..."


def _rich_visible_width(text: str) -> int:
    """Return visible width after ignoring Rich tags."""
    escaped_brackets = re.sub(r"\\\[", "X", text)
    return len(re.sub(r"\[[^\]]*]", "", escaped_brackets))


def _next_rich_text_chunk(text: str, index: int) -> tuple[str, int, int]:
    """Return the next Rich-aware chunk, next index, and visible width delta."""
    if text[index] == "\\" and index + 1 < len(text) and text[index + 1] == "[":
        return text[index : index + 2], index + 2, 1
    if text[index] == "[":
        end = text.find("]", index)
        if end != -1:
            return text[index : end + 1], end + 1, 0
    return text[index], index + 1, 1


__all__ = [
    "FooterModeBadgeState",
    "StatusBarState",
    "_build_compact_status_parts",
    "_build_full_status_parts",
    "_truncate_rich_text",
    "build_api_footer_bindings",
    "build_browse_footer_bindings",
    "build_detail_focus_footer_bindings",
    "build_footer_mode_badge",
    "build_selection_footer_base_bindings",
    "build_selection_footer_bindings",
    "build_status_bar_text",
    "get_filter_pill_remove_glyph",
    "set_ascii_glyphs",
]
