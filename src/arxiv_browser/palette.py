"""Shared command-palette data model and formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass

# Stable key/name columns plus a viewport-derived description column. The row
# chrome constant includes spaces, the suggestion marker, OptionList padding,
# and borders around the adaptive palette overlay.
PALETTE_KEY_MAX_LEN = 7
PALETTE_NAME_MAX_LEN = 24
PALETTE_DESC_MIN_LEN = 12
PALETTE_DESC_MAX_LEN = 40
PALETTE_OVERLAY_MAX_WIDTH = 100
PALETTE_ROW_CHROME_WIDTH = 56


def palette_description_max_len(viewport_width: int) -> int:
    """Return the description budget for a palette inside ``viewport_width``.

    The command-mode OmniInput occupies the viewport, capped at 100 columns.
    The remaining fixed row chrome is subtracted before clamping the
    description to a useful narrow fallback and a readable wide maximum.
    """
    overlay_width = min(PALETTE_OVERLAY_MAX_WIDTH, max(0, viewport_width))
    return max(
        PALETTE_DESC_MIN_LEN,
        min(PALETTE_DESC_MAX_LEN, overlay_width - PALETTE_ROW_CHROME_WIDTH),
    )


def truncate_palette_text(text: str, max_len: int) -> str:
    """Clamp palette row text to a stable width-friendly length."""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


@dataclass(slots=True)
class PaletteCommand:
    """One command palette row prepared by the app layer."""

    name: str
    description: str
    key_hint: str
    action: str
    group: str
    enabled: bool = True
    blocked_reason: str = ""
    suggested: bool = False


__all__ = [
    "PALETTE_DESC_MAX_LEN",
    "PALETTE_DESC_MIN_LEN",
    "PALETTE_KEY_MAX_LEN",
    "PALETTE_NAME_MAX_LEN",
    "PaletteCommand",
    "palette_description_max_len",
    "truncate_palette_text",
]
