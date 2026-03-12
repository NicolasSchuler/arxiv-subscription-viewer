"""Centralized ASCII-mode flag for glyph selection across modules.

This module has zero internal dependencies so any module can safely import it
without risking circular imports.
"""

from __future__ import annotations

_ascii_mode: bool = False


def set_ascii_mode(enabled: bool) -> None:
    """Set the global ASCII-mode flag (called once from App.__init__)."""
    global _ascii_mode
    _ascii_mode = enabled


def is_ascii_mode() -> bool:
    """Return *True* when the application runs in ASCII-only mode."""
    return _ascii_mode


__all__ = ["is_ascii_mode", "set_ascii_mode"]
