"""Shared command-palette data model and formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass

# Fixed column widths for the one-line-per-command palette rows. Chosen so a full
# row (key + marker + name + description) fits inside the Medium-width (70-col)
# palette overlay without wrapping. See widgets/omni_input.py for the layout.
PALETTE_KEY_MAX_LEN = 7
PALETTE_NAME_MAX_LEN = 24
PALETTE_DESC_MAX_LEN = 28


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
    "PALETTE_KEY_MAX_LEN",
    "PALETTE_NAME_MAX_LEN",
    "PaletteCommand",
    "truncate_palette_text",
]
