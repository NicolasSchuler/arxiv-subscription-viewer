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
    "panel_alt": "#214e59",
    "border": "#657b83",
    "text": "#b7c5c5",
    "muted": "#93a1a1",
    "accent": "#74c7ec",
    "accent_alt": "#d6b94d",
    "green": "#859900",
    "yellow": "#b58900",
    "orange": "#e87d3e",  # lightened for WCAG AA (5.2:1)
    "pink": "#e85da0",  # lightened for WCAG AA (5.0:1)
    "purple": "#8b8fd6",  # lightened for WCAG AA (5.0:1)
    "highlight": "#073642",
    "highlight_focus": "#214e59",
    "selection": "#073642",
    "selection_highlight": "#214e59",
    "scrollbar_background": "#073642",
    "scrollbar_background_hover": "#214e59",
    "scrollbar_background_active": "#657b83",
    "scrollbar": "#657b83",
    "scrollbar_active": "#74c7ec",  # match accent
    "scrollbar_hover": "#93a1a1",
    "scrollbar_corner_color": "#073642",
}

SOLARIZED_LIGHT_THEME: dict[str, str] = {
    "background": "#fdf6e3",
    "panel": "#eee8d5",
    "panel_alt": "#e3dcc8",
    "border": "#93a1a1",
    "text": "#073642",
    "muted": "#405a60",
    "accent": "#005f87",
    "accent_alt": "#7a5a00",
    "green": "#5f6f00",
    "yellow": "#7a5a00",
    "orange": "#9a4b00",
    "pink": "#b0005b",
    "purple": "#5f4b9a",
    "highlight": "#eee8d5",
    "highlight_focus": "#d7e7ec",
    "selection": "#dce8cf",
    "selection_highlight": "#c9ddb8",
    "scrollbar_background": "#eee8d5",
    "scrollbar_background_hover": "#e3dcc8",
    "scrollbar_background_active": "#d6ceb8",
    "scrollbar": "#93a1a1",
    "scrollbar_active": "#005f87",
    "scrollbar_hover": "#586e75",
    "scrollbar_corner_color": "#eee8d5",
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

DRACULA_THEME: dict[str, str] = {
    "background": "#282a36",
    "panel": "#21222c",
    "panel_alt": "#343746",
    "border": "#6272a4",
    "text": "#f8f8f2",
    "muted": "#c0c4df",
    "accent": "#8be9fd",
    "accent_alt": "#f1fa8c",
    "green": "#50fa7b",
    "yellow": "#f1fa8c",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "highlight": "#343746",
    "highlight_focus": "#44475a",
    "selection": "#3d4658",
    "selection_highlight": "#4a5568",
    "scrollbar_background": "#343746",
    "scrollbar_background_hover": "#44475a",
    "scrollbar_background_active": "#6272a4",
    "scrollbar": "#6272a4",
    "scrollbar_active": "#8be9fd",
    "scrollbar_hover": "#c0c4df",
    "scrollbar_corner_color": "#343746",
}

NORD_THEME: dict[str, str] = {
    "background": "#2e3440",
    "panel": "#242933",
    "panel_alt": "#3b4252",
    "border": "#6d788c",
    "text": "#eceff4",
    "muted": "#d8dee9",
    "accent": "#88c0d0",
    "accent_alt": "#ebcb8b",
    "green": "#a3be8c",
    "yellow": "#ebcb8b",
    "orange": "#d79784",
    "pink": "#d687b7",
    "purple": "#c39ac4",
    "highlight": "#3b4252",
    "highlight_focus": "#4c566a",
    "selection": "#374456",
    "selection_highlight": "#45566e",
    "scrollbar_background": "#3b4252",
    "scrollbar_background_hover": "#4c566a",
    "scrollbar_background_active": "#5e6a7d",
    "scrollbar": "#6d788c",
    "scrollbar_active": "#88c0d0",
    "scrollbar_hover": "#d8dee9",
    "scrollbar_corner_color": "#3b4252",
}

GRUVBOX_DARK_THEME: dict[str, str] = {
    "background": "#282828",
    "panel": "#1d2021",
    "panel_alt": "#3c3836",
    "border": "#7c6f64",
    "text": "#ebdbb2",
    "muted": "#d5c4a1",
    "accent": "#83a598",
    "accent_alt": "#fabd2f",
    "green": "#b8bb26",
    "yellow": "#fabd2f",
    "orange": "#fe8019",
    "pink": "#fb6a55",
    "purple": "#d3869b",
    "highlight": "#3c3836",
    "highlight_focus": "#504945",
    "selection": "#3f4f24",
    "selection_highlight": "#505f2e",
    "scrollbar_background": "#3c3836",
    "scrollbar_background_hover": "#504945",
    "scrollbar_background_active": "#665c54",
    "scrollbar": "#7c6f64",
    "scrollbar_active": "#83a598",
    "scrollbar_hover": "#d5c4a1",
    "scrollbar_corner_color": "#3c3836",
}

TOKYO_NIGHT_THEME: dict[str, str] = {
    "background": "#1a1b26",
    "panel": "#16161e",
    "panel_alt": "#24283b",
    "border": "#565f89",
    "text": "#c0caf5",
    "muted": "#a9b1d6",
    "accent": "#7dcfff",
    "accent_alt": "#e0af68",
    "green": "#9ece6a",
    "yellow": "#e0af68",
    "orange": "#ff9e64",
    "pink": "#f7768e",
    "purple": "#bb9af7",
    "highlight": "#24283b",
    "highlight_focus": "#414868",
    "selection": "#283457",
    "selection_highlight": "#35436b",
    "scrollbar_background": "#24283b",
    "scrollbar_background_hover": "#414868",
    "scrollbar_background_active": "#565f89",
    "scrollbar": "#565f89",
    "scrollbar_active": "#7dcfff",
    "scrollbar_hover": "#a9b1d6",
    "scrollbar_corner_color": "#24283b",
}

EVERFOREST_DARK_THEME: dict[str, str] = {
    "background": "#2d353b",
    "panel": "#232a2e",
    "panel_alt": "#343f44",
    "border": "#859289",
    "text": "#d3c6aa",
    "muted": "#c0b999",
    "accent": "#7fbbb3",
    "accent_alt": "#dbbc7f",
    "green": "#a7c080",
    "yellow": "#dbbc7f",
    "orange": "#e69875",
    "pink": "#e67e80",
    "purple": "#d699b6",
    "highlight": "#343f44",
    "highlight_focus": "#475258",
    "selection": "#3f4f3c",
    "selection_highlight": "#4d6048",
    "scrollbar_background": "#343f44",
    "scrollbar_background_hover": "#475258",
    "scrollbar_background_active": "#5b686d",
    "scrollbar": "#859289",
    "scrollbar_active": "#7fbbb3",
    "scrollbar_hover": "#c0b999",
    "scrollbar_corner_color": "#343f44",
}

GITHUB_LIGHT_THEME: dict[str, str] = {
    "background": "#ffffff",
    "panel": "#f6f8fa",
    "panel_alt": "#eaeef2",
    "border": "#8c959f",
    "text": "#24292f",
    "muted": "#57606a",
    "accent": "#0969da",
    "accent_alt": "#8250df",
    "green": "#1a7f37",
    "yellow": "#7d4e00",
    "orange": "#bc4c00",
    "pink": "#bf3989",
    "purple": "#8250df",
    "highlight": "#eaeef2",
    "highlight_focus": "#d0d7de",
    "selection": "#ddf4ff",
    "selection_highlight": "#b6e3ff",
    "scrollbar_background": "#eaeef2",
    "scrollbar_background_hover": "#d0d7de",
    "scrollbar_background_active": "#afb8c1",
    "scrollbar": "#8c959f",
    "scrollbar_active": "#0969da",
    "scrollbar_hover": "#57606a",
    "scrollbar_corner_color": "#eaeef2",
}

THEMES: dict[str, dict[str, str]] = {
    "monokai": DEFAULT_THEME,
    "catppuccin-mocha": CATPPUCCIN_MOCHA_THEME,
    "solarized-dark": SOLARIZED_DARK_THEME,
    "solarized-light": SOLARIZED_LIGHT_THEME,
    "high-contrast": HIGH_CONTRAST_THEME,
    "dracula": DRACULA_THEME,
    "nord": NORD_THEME,
    "gruvbox-dark": GRUVBOX_DARK_THEME,
    "tokyo-night": TOKYO_NIGHT_THEME,
    "everforest-dark": EVERFOREST_DARK_THEME,
    "github-light": GITHUB_LIGHT_THEME,
}
THEME_NAMES: list[str] = list(THEMES.keys())


def available_theme_names(
    custom_themes: Mapping[str, Mapping[str, str]] | None = None,
) -> list[str]:
    """Return built-in theme names plus user-defined theme names for cycling."""
    names = list(THEME_NAMES)
    if custom_themes:
        names.extend(sorted(name for name in custom_themes if name not in THEMES))
    return names


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
        "th-selection": colors["selection"],
        "th-selection-highlight": colors["selection_highlight"],
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
        dark=not name.endswith("-light"),
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
    "solarized-light": {
        "cs.AI": "#b0005b",
        "cs.CL": "#005f87",
        "cs.LG": "#5f6f00",
        "cs.CV": "#7a5a00",
        "cs.SE": "#5f4b9a",
        "cs.HC": "#9a4b00",
        "cs.RO": "#005f87",
        "cs.NE": "#b0005b",
        "cs.IR": "#5f4b9a",
        "cs.CR": "#9a4b00",
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
    "dracula": {
        "cs.AI": "#ff79c6",
        "cs.CL": "#8be9fd",
        "cs.LG": "#50fa7b",
        "cs.CV": "#f1fa8c",
        "cs.SE": "#bd93f9",
        "cs.HC": "#ffb86c",
        "cs.RO": "#8be9fd",
        "cs.NE": "#ff79c6",
        "cs.IR": "#bd93f9",
        "cs.CR": "#ffb86c",
    },
    "nord": {
        "cs.AI": "#d687b7",
        "cs.CL": "#88c0d0",
        "cs.LG": "#a3be8c",
        "cs.CV": "#ebcb8b",
        "cs.SE": "#c39ac4",
        "cs.HC": "#d79784",
        "cs.RO": "#88c0d0",
        "cs.NE": "#d687b7",
        "cs.IR": "#c39ac4",
        "cs.CR": "#d79784",
    },
    "gruvbox-dark": {
        "cs.AI": "#fb6a55",
        "cs.CL": "#83a598",
        "cs.LG": "#b8bb26",
        "cs.CV": "#fabd2f",
        "cs.SE": "#d3869b",
        "cs.HC": "#fe8019",
        "cs.RO": "#83a598",
        "cs.NE": "#fb6a55",
        "cs.IR": "#d3869b",
        "cs.CR": "#fe8019",
    },
    "tokyo-night": {
        "cs.AI": "#f7768e",
        "cs.CL": "#7dcfff",
        "cs.LG": "#9ece6a",
        "cs.CV": "#e0af68",
        "cs.SE": "#bb9af7",
        "cs.HC": "#ff9e64",
        "cs.RO": "#7dcfff",
        "cs.NE": "#f7768e",
        "cs.IR": "#bb9af7",
        "cs.CR": "#ff9e64",
    },
    "everforest-dark": {
        "cs.AI": "#e67e80",
        "cs.CL": "#7fbbb3",
        "cs.LG": "#a7c080",
        "cs.CV": "#dbbc7f",
        "cs.SE": "#d699b6",
        "cs.HC": "#e69875",
        "cs.RO": "#7fbbb3",
        "cs.NE": "#e67e80",
        "cs.IR": "#d699b6",
        "cs.CR": "#e69875",
    },
    "github-light": {
        "cs.AI": "#bf3989",
        "cs.CL": "#0969da",
        "cs.LG": "#1a7f37",
        "cs.CV": "#7d4e00",
        "cs.SE": "#8250df",
        "cs.HC": "#bc4c00",
        "cs.RO": "#0969da",
        "cs.NE": "#bf3989",
        "cs.IR": "#8250df",
        "cs.CR": "#bc4c00",
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
    "solarized-light": {
        "topic": "#005f87",
        "status": "#5f6f00",
        "project": "#9a4b00",
        "method": "#5f4b9a",
        "priority": "#b0005b",
    },
    "high-contrast": {
        "topic": "#5cb8ff",  # WCAG AAA
        "status": "#5dfa5d",  # WCAG AAA
        "project": "#ffaa44",  # WCAG AAA
        "method": "#b4a7ff",  # WCAG AAA
        "priority": "#ff7ab2",  # WCAG AAA
    },
    "dracula": {
        "topic": "#8be9fd",
        "status": "#50fa7b",
        "project": "#ffb86c",
        "method": "#bd93f9",
        "priority": "#ff79c6",
    },
    "nord": {
        "topic": "#88c0d0",
        "status": "#a3be8c",
        "project": "#d79784",
        "method": "#c39ac4",
        "priority": "#d687b7",
    },
    "gruvbox-dark": {
        "topic": "#83a598",
        "status": "#b8bb26",
        "project": "#fe8019",
        "method": "#d3869b",
        "priority": "#fb6a55",
    },
    "tokyo-night": {
        "topic": "#7dcfff",
        "status": "#9ece6a",
        "project": "#ff9e64",
        "method": "#bb9af7",
        "priority": "#f7768e",
    },
    "everforest-dark": {
        "topic": "#7fbbb3",
        "status": "#a7c080",
        "project": "#e69875",
        "method": "#d699b6",
        "priority": "#e67e80",
    },
    "github-light": {
        "topic": "#0969da",
        "status": "#1a7f37",
        "project": "#bc4c00",
        "method": "#8250df",
        "priority": "#bf3989",
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
    custom_themes: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Resolve display colors for a theme name plus user overrides."""
    custom_theme = (custom_themes or {}).get(theme_name)
    if custom_theme is not None:
        resolved = dict(DEFAULT_THEME)
        resolved.update(custom_theme)
    else:
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
    custom_themes: Mapping[str, Mapping[str, str]] | None = None,
) -> ThemeRuntime:
    """Build resolved runtime theme state for the current app config."""
    return ThemeRuntime(
        name=theme_name,
        colors=resolve_theme_colors(theme_name, theme_overrides, custom_themes),
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
    "DRACULA_THEME",
    "EVERFOREST_DARK_THEME",
    "GITHUB_LIGHT_THEME",
    "GRUVBOX_DARK_THEME",
    "HIGH_CONTRAST_THEME",
    "NORD_THEME",
    "SOLARIZED_DARK_THEME",
    "SOLARIZED_LIGHT_THEME",
    "TAG_NAMESPACE_COLORS",
    "TEXTUAL_THEMES",
    "THEMES",
    "THEME_CATEGORY_COLORS",
    "THEME_COLORS",
    "THEME_NAMES",
    "THEME_TAG_NAMESPACE_COLORS",
    "TOKYO_NIGHT_THEME",
    "ThemeRuntime",
    "available_theme_names",
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
