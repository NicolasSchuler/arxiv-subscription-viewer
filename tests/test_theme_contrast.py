"""Theme contrast and modal width safety checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from arxiv_browser.themes import (
    THEMES,
    resolve_category_colors,
    resolve_tag_namespace_colors,
)

_NORMAL_TEXT_KEYS: Final = ("text", "muted")
_ACCENT_UI_KEYS: Final = (
    "accent",
    "accent_alt",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
)
_BACKGROUND_KEYS: Final = ("background", "panel")
_NORMAL_TEXT_MIN_RATIO: Final = 4.5
_ACCENT_UI_MIN_RATIO: Final = 3.0
_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_MODAL_CSS_DIR: Final = _REPO_ROOT / "src/arxiv_browser/modals"


def _rgb_components(color: str) -> tuple[float, float, float]:
    hex_color = color.removeprefix("#")
    red, green, blue = (int(hex_color[index : index + 2], 16) / 255 for index in (0, 2, 4))
    return red, green, blue


def _linearized_channel(channel: float) -> float:
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _relative_luminance(color: str) -> float:
    red, green, blue = (_linearized_channel(channel) for channel in _rgb_components(color))
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _contrast_ratio(foreground: str, background: str) -> float:
    fg_luminance = _relative_luminance(foreground)
    bg_luminance = _relative_luminance(background)
    lighter = max(fg_luminance, bg_luminance)
    darker = min(fg_luminance, bg_luminance)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("theme_name,palette", THEMES.items())
@pytest.mark.parametrize("background_key", _BACKGROUND_KEYS)
def test_theme_normal_text_colors_meet_wcag_aa(
    theme_name: str,
    palette: dict[str, str],
    background_key: str,
) -> None:
    """Normal text colors should meet WCAG AA on primary app surfaces."""
    background = palette[background_key]

    for color_key in _NORMAL_TEXT_KEYS:
        ratio = _contrast_ratio(palette[color_key], background)
        assert ratio >= _NORMAL_TEXT_MIN_RATIO, (
            f"{theme_name}:{color_key} on {background_key} contrast {ratio:.2f}:1 "
            f"is below {_NORMAL_TEXT_MIN_RATIO}:1"
        )


@pytest.mark.parametrize("theme_name,palette", THEMES.items())
@pytest.mark.parametrize("background_key", _BACKGROUND_KEYS)
def test_theme_accent_ui_colors_meet_minimum_contrast(
    theme_name: str,
    palette: dict[str, str],
    background_key: str,
) -> None:
    """Accent/status UI colors should remain distinguishable on app surfaces."""
    background = palette[background_key]
    accent_colors = {key: palette[key] for key in _ACCENT_UI_KEYS}
    accent_colors.update(
        {f"category:{key}": color for key, color in resolve_category_colors(theme_name).items()}
    )
    accent_colors.update(
        {f"tag:{key}": color for key, color in resolve_tag_namespace_colors(theme_name).items()}
    )

    for color_key, color in accent_colors.items():
        ratio = _contrast_ratio(color, background)
        assert ratio >= _ACCENT_UI_MIN_RATIO, (
            f"{theme_name}:{color_key} on {background_key} contrast {ratio:.2f}:1 "
            f"is below {_ACCENT_UI_MIN_RATIO}:1"
        )


def test_small_and_medium_modal_widths_have_narrow_terminal_guard() -> None:
    """Fixed-width Small/Medium modals must cap width on narrow terminals."""
    modal_css = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_MODAL_CSS_DIR.glob("*.py"))
    )
    fixed_width_blocks = re.findall(r"[^{}]+\{[^{}]*(?<!-)width: (?:52|70);[^{}]*\}", modal_css)

    assert fixed_width_blocks
    for block in fixed_width_blocks:
        assert "max-width: 90%;" in block
