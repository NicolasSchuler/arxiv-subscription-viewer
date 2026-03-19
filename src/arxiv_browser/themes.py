"""Theme system — color palettes, category colors, and Textual theme builders."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from textual.theme import Theme as TextualTheme

# Default color for unknown categories (Monokai gray)
DEFAULT_CATEGORY_COLOR = "#888888"

# Category color mapping (Monokai-inspired palette)
DEFAULT_CATEGORY_COLORS: dict[str, str] = {
    "cs.AI": "#fd4d8e",  # Monokai pink (WCAG AA)
    "cs.CL": "#66d9ef",  # Monokai blue
    "cs.LG": "#a6e22e",  # Monokai green
    "cs.CV": "#e6db74",  # Monokai yellow
    "cs.SE": "#ae81ff",  # Monokai purple
    "cs.HC": "#fd971f",  # Monokai orange
    "cs.RO": "#66d9ef",  # Monokai blue
    "cs.NE": "#fd4d8e",  # Monokai pink (WCAG AA)
    "cs.IR": "#ae81ff",  # Monokai purple
    "cs.CR": "#fd971f",  # Monokai orange
}

DEFAULT_THEME: dict[str, str] = {
    "background": "#272822",
    "panel": "#1e1e1e",
    "panel_alt": "#3e3d32",
    "border": "#75715e",
    "text": "#f8f8f2",
    "muted": "#948e7d",  # lightened for WCAG AA (4.5:1)
    "accent": "#66d9ef",
    "accent_alt": "#e6db74",
    "green": "#a6e22e",
    "yellow": "#e6db74",
    "orange": "#fd971f",
    "pink": "#fd4d8e",  # lightened for WCAG AA (4.5:1)
    "purple": "#ae81ff",
    "highlight": "#49483e",
    "highlight_focus": "#6e6d5e",
    "selection": "#3d4a32",
    "selection_highlight": "#4d5a42",
    "scrollbar_background": "#3e3d32",
    "scrollbar_background_hover": "#49483e",
    "scrollbar_background_active": "#5a5950",
    "scrollbar": "#75715e",
    "scrollbar_active": "#66d9ef",
    "scrollbar_hover": "#a8a8a2",
    "scrollbar_corner_color": "#3e3d32",
}

CATPPUCCIN_MOCHA_THEME: dict[str, str] = {
    "background": "#1e1e2e",
    "panel": "#181825",
    "panel_alt": "#313244",
    "border": "#585b70",
    "text": "#cdd6f4",
    "muted": "#8488a2",  # lightened for WCAG AA (4.5:1)
    "accent": "#89b4fa",
    "accent_alt": "#f9e2af",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "orange": "#fab387",
    "pink": "#f38ba8",
    "purple": "#cba6f7",
    "highlight": "#313244",
    "highlight_focus": "#45475a",
    "selection": "#313244",
    "selection_highlight": "#45475a",
    "scrollbar_background": "#313244",
    "scrollbar_background_hover": "#45475a",
    "scrollbar_background_active": "#585b70",
    "scrollbar": "#6c7086",
    "scrollbar_active": "#89b4fa",
    "scrollbar_hover": "#9399b2",
    "scrollbar_corner_color": "#313244",
}

SOLARIZED_DARK_THEME: dict[str, str] = {
    "background": "#002b36",
    "panel": "#073642",
    "panel_alt": "#586e75",
    "border": "#657b83",
    "text": "#839496",
    "muted": "#7c9098",  # lightened for WCAG AA (4.5:1)
    "accent": "#3c9be2",  # lightened for WCAG AA (5.0:1)
    "accent_alt": "#b58900",
    "green": "#859900",
    "yellow": "#b58900",
    "orange": "#e87d3e",  # lightened for WCAG AA (5.2:1)
    "pink": "#e85da0",  # lightened for WCAG AA (5.0:1)
    "purple": "#8b8fd6",  # lightened for WCAG AA (5.0:1)
    "highlight": "#073642",
    "highlight_focus": "#586e75",
    "selection": "#073642",
    "selection_highlight": "#586e75",
    "scrollbar_background": "#073642",
    "scrollbar_background_hover": "#586e75",
    "scrollbar_background_active": "#657b83",
    "scrollbar": "#657b83",
    "scrollbar_active": "#3c9be2",  # match accent
    "scrollbar_hover": "#93a1a1",
    "scrollbar_corner_color": "#073642",
}

HIGH_CONTRAST_THEME: dict[str, str] = {
    "background": "#000000",
    "panel": "#0a0a0a",
    "panel_alt": "#1a1a1a",
    "border": "#808080",
    "text": "#ffffff",  # 21.0:1 — WCAG AAA
    "muted": "#b0b0b0",  # 13.3:1 — WCAG AAA
    "accent": "#5cb8ff",  # 8.0:1 — WCAG AAA
    "accent_alt": "#ffdf4d",  # 14.7:1 — WCAG AAA
    "green": "#5dfa5d",  # 12.4:1 — WCAG AAA
    "yellow": "#ffdf4d",  # 14.7:1 — WCAG AAA
    "orange": "#ffaa44",  # 10.3:1 — WCAG AAA
    "pink": "#ff7ab2",  # 7.5:1 — WCAG AAA
    "purple": "#b4a7ff",  # 8.2:1 — WCAG AAA
    "highlight": "#1a1a1a",
    "highlight_focus": "#3d3d3d",
    "selection": "#1a3a1a",
    "selection_highlight": "#2a4a2a",
    "scrollbar_background": "#1a1a1a",
    "scrollbar_background_hover": "#2a2a2a",
    "scrollbar_background_active": "#3a3a3a",
    "scrollbar": "#808080",
    "scrollbar_active": "#5cb8ff",
    "scrollbar_hover": "#b0b0b0",
    "scrollbar_corner_color": "#1a1a1a",
}

THEMES: dict[str, dict[str, str]] = {
    "monokai": DEFAULT_THEME,
    "catppuccin-mocha": CATPPUCCIN_MOCHA_THEME,
    "solarized-dark": SOLARIZED_DARK_THEME,
    "high-contrast": HIGH_CONTRAST_THEME,
}
THEME_NAMES: list[str] = list(THEMES.keys())


def _build_textual_theme(name: str, colors: dict[str, str]) -> TextualTheme:
    """Convert an app color dict to a Textual Theme with custom CSS variables.

    Maps 16 color keys to $th-* CSS variables that replace hardcoded hex in TCSS.
    Also sets primary/background/foreground for Textual's built-in widget styling.
    """
    variables = {
        "th-background": colors["background"],
        "th-panel": colors["panel"],
        "th-panel-alt": colors["panel_alt"],
        "th-highlight": colors["highlight"],
        "th-highlight-focus": colors["highlight_focus"],
        "th-accent": colors["accent"],
        "th-accent-alt": colors["accent_alt"],
        "th-muted": colors["muted"],
        "th-text": colors["text"],
        "th-green": colors["green"],
        "th-orange": colors["orange"],
        "th-purple": colors["purple"],
        "th-scrollbar-bg": colors["scrollbar_background"],
        "th-scrollbar-thumb": colors["scrollbar"],
        "th-scrollbar-active": colors["scrollbar_active"],
        "th-scrollbar-hover": colors["scrollbar_hover"],
    }
    return TextualTheme(
        name=name,
        primary=colors["accent"],
        secondary=colors["accent_alt"],
        accent=colors["green"],
        foreground=colors["text"],
        background=colors["background"],
        surface=colors["panel"],
        panel=colors["panel_alt"],
        warning=colors["orange"],
        error=colors["pink"],
        success=colors["green"],
        dark=True,
        variables=variables,
    )


TEXTUAL_THEMES: dict[str, TextualTheme] = {
    name: _build_textual_theme(name, colors) for name, colors in THEMES.items()
}


# Per-theme category colors — ensures categories are readable on each background
THEME_CATEGORY_COLORS: dict[str, dict[str, str]] = {
    "catppuccin-mocha": {
        "cs.AI": "#f38ba8",  # red
        "cs.CL": "#89b4fa",  # blue
        "cs.LG": "#a6e3a1",  # green
        "cs.CV": "#f9e2af",  # yellow
        "cs.SE": "#cba6f7",  # mauve
        "cs.HC": "#fab387",  # peach
        "cs.RO": "#89b4fa",  # blue
        "cs.NE": "#f38ba8",  # red
        "cs.IR": "#cba6f7",  # mauve
        "cs.CR": "#fab387",  # peach
    },
    "solarized-dark": {
        "cs.AI": "#e85da0",  # magenta (WCAG AA)
        "cs.CL": "#3c9be2",  # blue (WCAG AA)
        "cs.LG": "#859900",  # green
        "cs.CV": "#b58900",  # yellow
        "cs.SE": "#8b8fd6",  # violet (WCAG AA)
        "cs.HC": "#e87d3e",  # orange (WCAG AA)
        "cs.RO": "#3c9be2",  # blue (WCAG AA)
        "cs.NE": "#e85da0",  # magenta (WCAG AA)
        "cs.IR": "#8b8fd6",  # violet (WCAG AA)
        "cs.CR": "#e87d3e",  # orange (WCAG AA)
    },
    "high-contrast": {
        "cs.AI": "#ff7ab2",  # pink (WCAG AAA)
        "cs.CL": "#5cb8ff",  # blue (WCAG AAA)
        "cs.LG": "#5dfa5d",  # green (WCAG AAA)
        "cs.CV": "#ffdf4d",  # yellow (WCAG AAA)
        "cs.SE": "#b4a7ff",  # purple (WCAG AAA)
        "cs.HC": "#ffaa44",  # orange (WCAG AAA)
        "cs.RO": "#5cb8ff",  # blue (WCAG AAA)
        "cs.NE": "#ff7ab2",  # pink (WCAG AAA)
        "cs.IR": "#b4a7ff",  # purple (WCAG AAA)
        "cs.CR": "#ffaa44",  # orange (WCAG AAA)
    },
}

# Per-theme tag namespace colors
THEME_TAG_NAMESPACE_COLORS: dict[str, dict[str, str]] = {
    "catppuccin-mocha": {
        "topic": "#89b4fa",
        "status": "#a6e3a1",
        "project": "#fab387",
        "method": "#cba6f7",
        "priority": "#f38ba8",
    },
    "solarized-dark": {
        "topic": "#3c9be2",  # WCAG AA
        "status": "#859900",
        "project": "#e87d3e",  # WCAG AA
        "method": "#8b8fd6",  # WCAG AA
        "priority": "#e85da0",  # WCAG AA
    },
    "high-contrast": {
        "topic": "#5cb8ff",  # WCAG AAA
        "status": "#5dfa5d",  # WCAG AAA
        "project": "#ffaa44",  # WCAG AAA
        "method": "#b4a7ff",  # WCAG AAA
        "priority": "#ff7ab2",  # WCAG AAA
    },
}

# Tag namespace colors (Monokai palette)
DEFAULT_TAG_NAMESPACE_COLORS: dict[str, str] = {
    "topic": "#66d9ef",  # blue
    "status": "#a6e22e",  # green
    "project": "#fd971f",  # orange
    "method": "#ae81ff",  # purple
    "priority": "#fd4d8e",  # pink (WCAG AA)
}
CATEGORY_COLORS: Mapping[str, str] = MappingProxyType(DEFAULT_CATEGORY_COLORS.copy())
THEME_COLORS: Mapping[str, str] = MappingProxyType(DEFAULT_THEME.copy())
TAG_NAMESPACE_COLORS: Mapping[str, str] = MappingProxyType(DEFAULT_TAG_NAMESPACE_COLORS.copy())
# Fallback palette for unknown namespaces (deterministic via hash)
_TAG_FALLBACK_COLORS = ["#66d9ef", "#a6e22e", "#fd971f", "#ae81ff", "#fd4d8e", "#e6db74"]


@dataclass(frozen=True, slots=True)
class ThemeRuntime:
    """Resolved runtime theme state owned by the app instance."""

    name: str
    colors: dict[str, str]
    category_colors: dict[str, str]
    tag_namespace_colors: dict[str, str]


def resolve_theme_colors(
    theme_name: str,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve display colors for a theme name plus user overrides."""
    resolved = dict(THEMES.get(theme_name, DEFAULT_THEME))
    if overrides:
        resolved.update(overrides)
    return resolved


def resolve_category_colors(
    theme_name: str,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve category badge colors for a theme name plus user overrides."""
    resolved = dict(DEFAULT_CATEGORY_COLORS)
    theme_colors = THEME_CATEGORY_COLORS.get(theme_name)
    if theme_colors:
        resolved.update(theme_colors)
    if overrides:
        resolved.update(overrides)
    return resolved


def resolve_tag_namespace_colors(theme_name: str) -> dict[str, str]:
    """Resolve tag namespace colors for a theme name."""
    resolved = dict(DEFAULT_TAG_NAMESPACE_COLORS)
    theme_colors = THEME_TAG_NAMESPACE_COLORS.get(theme_name)
    if theme_colors:
        resolved.update(theme_colors)
    return resolved


def build_theme_runtime(
    theme_name: str,
    *,
    theme_overrides: Mapping[str, str] | None = None,
    category_overrides: Mapping[str, str] | None = None,
) -> ThemeRuntime:
    """Build resolved runtime theme state for the current app config."""
    return ThemeRuntime(
        name=theme_name,
        colors=resolve_theme_colors(theme_name, theme_overrides),
        category_colors=resolve_category_colors(theme_name, category_overrides),
        tag_namespace_colors=resolve_tag_namespace_colors(theme_name),
    )


def _runtime_theme(owner: object | None) -> ThemeRuntime | None:
    """Return app-owned runtime theme state when available."""
    if owner is None:
        return None
    app = owner
    if not isinstance(getattr(owner, "_theme_runtime", None), ThemeRuntime):
        try:
            app = cast(Any, owner).app
        except Exception:
            app = owner
    runtime = getattr(app, "_theme_runtime", None)
    return runtime if isinstance(runtime, ThemeRuntime) else None


def theme_colors_for(
    owner: object | None,
    fallback: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Resolve the current theme color map for a widget/screen/app."""
    runtime = _runtime_theme(owner)
    if runtime is not None:
        return runtime.colors
    return fallback or THEME_COLORS


def category_colors_for(
    owner: object | None,
    fallback: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Resolve the current category color map for a widget/screen/app."""
    runtime = _runtime_theme(owner)
    if runtime is not None:
        return runtime.category_colors
    return fallback or CATEGORY_COLORS


def tag_namespace_colors_for(
    owner: object | None,
    fallback: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Resolve the current tag namespace color map for a widget/screen/app."""
    runtime = _runtime_theme(owner)
    if runtime is not None:
        return runtime.tag_namespace_colors
    return fallback or TAG_NAMESPACE_COLORS


def parse_tag_namespace(tag: str) -> tuple[str, str]:
    """Split a tag into (namespace, value).

    >>> parse_tag_namespace("topic:transformers")
    ('topic', 'transformers')
    >>> parse_tag_namespace("important")
    ('', 'important')
    """
    if ":" in tag:
        ns, _, val = tag.partition(":")
        return (ns, val)
    return ("", tag)


def get_tag_color(tag: str, tag_namespace_colors: Mapping[str, str] | None = None) -> str:
    """Return a display color for a tag based on its namespace.

    Known namespaces get their assigned color. Unknown namespaces get a
    deterministic color via hash. Tags without a namespace get default purple.
    """
    ns, _ = parse_tag_namespace(tag)
    namespace_colors = tag_namespace_colors or TAG_NAMESPACE_COLORS
    if not ns:
        return "#ae81ff"  # default purple for unnamespaced tags
    if ns in namespace_colors:
        return namespace_colors[ns]
    # Deterministic color for unknown namespaces
    digest = hashlib.sha256(ns.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:2], "big") % len(_TAG_FALLBACK_COLORS)
    return _TAG_FALLBACK_COLORS[idx]


__all__ = [
    "CATEGORY_COLORS",
    "CATPPUCCIN_MOCHA_THEME",
    "DEFAULT_CATEGORY_COLOR",
    "DEFAULT_CATEGORY_COLORS",
    "DEFAULT_TAG_NAMESPACE_COLORS",
    "DEFAULT_THEME",
    "HIGH_CONTRAST_THEME",
    "SOLARIZED_DARK_THEME",
    "TAG_NAMESPACE_COLORS",
    "TEXTUAL_THEMES",
    "THEMES",
    "THEME_CATEGORY_COLORS",
    "THEME_COLORS",
    "THEME_NAMES",
    "THEME_TAG_NAMESPACE_COLORS",
    "ThemeRuntime",
    "build_theme_runtime",
    "category_colors_for",
    "get_tag_color",
    "parse_tag_namespace",
    "resolve_category_colors",
    "resolve_tag_namespace_colors",
    "resolve_theme_colors",
    "tag_namespace_colors_for",
    "theme_colors_for",
]
