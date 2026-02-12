"""Theme system — color palettes, category colors, and Textual theme builders."""

from __future__ import annotations

from textual.theme import Theme as TextualTheme

# Default color for unknown categories (Monokai gray)
DEFAULT_CATEGORY_COLOR = "#888888"

# Category color mapping (Monokai-inspired palette)
DEFAULT_CATEGORY_COLORS = {
    "cs.AI": "#f92672",  # Monokai pink
    "cs.CL": "#66d9ef",  # Monokai blue
    "cs.LG": "#a6e22e",  # Monokai green
    "cs.CV": "#e6db74",  # Monokai yellow
    "cs.SE": "#ae81ff",  # Monokai purple
    "cs.HC": "#fd971f",  # Monokai orange
    "cs.RO": "#66d9ef",  # Monokai blue
    "cs.NE": "#f92672",  # Monokai pink
    "cs.IR": "#ae81ff",  # Monokai purple
    "cs.CR": "#fd971f",  # Monokai orange
}

CATEGORY_COLORS = DEFAULT_CATEGORY_COLORS.copy()

DEFAULT_THEME = {
    "background": "#272822",
    "panel": "#1e1e1e",
    "panel_alt": "#3e3d32",
    "border": "#75715e",
    "text": "#f8f8f2",
    "muted": "#75715e",
    "accent": "#66d9ef",
    "accent_alt": "#e6db74",
    "green": "#a6e22e",
    "yellow": "#e6db74",
    "orange": "#fd971f",
    "pink": "#f92672",
    "purple": "#ae81ff",
    "highlight": "#49483e",
    "highlight_focus": "#5a5950",
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
    "muted": "#6c7086",
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
    "muted": "#586e75",
    "accent": "#268bd2",
    "accent_alt": "#b58900",
    "green": "#859900",
    "yellow": "#b58900",
    "orange": "#cb4b16",
    "pink": "#d33682",
    "purple": "#6c71c4",
    "highlight": "#073642",
    "highlight_focus": "#586e75",
    "selection": "#073642",
    "selection_highlight": "#586e75",
    "scrollbar_background": "#073642",
    "scrollbar_background_hover": "#586e75",
    "scrollbar_background_active": "#657b83",
    "scrollbar": "#657b83",
    "scrollbar_active": "#268bd2",
    "scrollbar_hover": "#93a1a1",
    "scrollbar_corner_color": "#073642",
}

THEMES: dict[str, dict[str, str]] = {
    "monokai": DEFAULT_THEME,
    "catppuccin-mocha": CATPPUCCIN_MOCHA_THEME,
    "solarized-dark": SOLARIZED_DARK_THEME,
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
        "cs.AI": "#d33682",  # magenta
        "cs.CL": "#268bd2",  # blue
        "cs.LG": "#859900",  # green
        "cs.CV": "#b58900",  # yellow
        "cs.SE": "#6c71c4",  # violet
        "cs.HC": "#cb4b16",  # orange
        "cs.RO": "#268bd2",  # blue
        "cs.NE": "#d33682",  # magenta
        "cs.IR": "#6c71c4",  # violet
        "cs.CR": "#cb4b16",  # orange
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
        "topic": "#268bd2",
        "status": "#859900",
        "project": "#cb4b16",
        "method": "#6c71c4",
        "priority": "#d33682",
    },
}

THEME_COLORS = DEFAULT_THEME.copy()

# Tag namespace colors (Monokai palette)
TAG_NAMESPACE_COLORS: dict[str, str] = {
    "topic": "#66d9ef",  # blue
    "status": "#a6e22e",  # green
    "project": "#fd971f",  # orange
    "method": "#ae81ff",  # purple
    "priority": "#f92672",  # pink
}
# Fallback palette for unknown namespaces (deterministic via hash)
_TAG_FALLBACK_COLORS = ["#66d9ef", "#a6e22e", "#fd971f", "#ae81ff", "#f92672", "#e6db74"]


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


def get_tag_color(tag: str) -> str:
    """Return a display color for a tag based on its namespace.

    Known namespaces get their assigned color. Unknown namespaces get a
    deterministic color via hash. Tags without a namespace get default purple.
    """
    ns, _ = parse_tag_namespace(tag)
    if not ns:
        return "#ae81ff"  # default purple for unnamespaced tags
    if ns in TAG_NAMESPACE_COLORS:
        return TAG_NAMESPACE_COLORS[ns]
    # Deterministic color for unknown namespaces
    return _TAG_FALLBACK_COLORS[hash(ns) % len(_TAG_FALLBACK_COLORS)]


__all__ = [
    "CATEGORY_COLORS",
    "CATPPUCCIN_MOCHA_THEME",
    "DEFAULT_CATEGORY_COLOR",
    "DEFAULT_CATEGORY_COLORS",
    "DEFAULT_THEME",
    "SOLARIZED_DARK_THEME",
    "TAG_NAMESPACE_COLORS",
    "TEXTUAL_THEMES",
    "THEMES",
    "THEME_CATEGORY_COLORS",
    "THEME_COLORS",
    "THEME_NAMES",
    "THEME_TAG_NAMESPACE_COLORS",
    "get_tag_color",
    "parse_tag_namespace",
]
