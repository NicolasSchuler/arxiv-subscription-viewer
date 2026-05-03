"""CLI keybinding output helpers."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from arxiv_browser.help_ui import build_help_sections
from arxiv_browser.ui_constants import APP_BINDINGS

KeybindingSections = Sequence[tuple[str, Sequence[tuple[str, str]]]]

TIER_SECTION_NAMES: dict[str, frozenset[str]] = {
    "core": frozenset({"Getting Started", "Search Syntax", "Core Actions"}),
    "standard": frozenset({"Standard \u00b7 Organize", "Standard - Organize"}),
    "power": frozenset(
        {
            "Power \u00b7 Research Tools",
            "Power \u00b7 Advanced",
            "Power - Research Tools",
            "Power - Advanced",
        }
    ),
}


def _filter_sections_for_tier(
    sections: KeybindingSections,
    tier: str,
) -> list[tuple[str, Sequence[tuple[str, str]]]]:
    """Return sections visible for a keybinding tier."""
    if tier == "all":
        return list(sections)
    allowed = TIER_SECTION_NAMES.get(tier, frozenset())
    return [(name, entries) for name, entries in sections if name in allowed]


def _table_uses_color(args: argparse.Namespace) -> bool:
    """Return whether table output should emit ANSI color escapes."""
    if bool(getattr(args, "no_color", False)):
        return False
    color_mode = str(getattr(args, "color", "auto"))
    if color_mode == "never":
        return False
    if color_mode == "always":
        return True
    return "NO_COLOR" not in os.environ


def _render_json(sections: KeybindingSections) -> None:
    data = [
        {"section": name, "bindings": [{"key": key, "description": desc} for key, desc in entries]}
        for name, entries in sections
    ]
    print(json.dumps(data, indent=2))


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _render_markdown(sections: KeybindingSections) -> None:
    for name, entries in sections:
        print(f"## {name}\n")
        print("| Key | Action |")
        print("|-----|--------|")
        for key, desc in entries:
            print(f"| `{_escape_markdown_cell(key)}` | {_escape_markdown_cell(desc)} |")
        print()


def _render_table(sections: KeybindingSections, *, use_color: bool) -> None:
    section_style = ("\033[1m", "\033[0m") if use_color else ("", "")
    key_style = ("\033[32m", "\033[0m") if use_color else ("", "")
    for name, entries in sections:
        print(f"{section_style[0]}{name}{section_style[1]}")
        for key, desc in entries:
            print(f"  {key_style[0]}{key:<20}{key_style[1]} {desc}")
        print()


def _run_keybindings(args: argparse.Namespace) -> int:
    """Print keyboard shortcuts to stdout in the requested format."""
    sections = _filter_sections_for_tier(
        build_help_sections(APP_BINDINGS), getattr(args, "tier", "all")
    )
    kb_format = getattr(args, "kb_format", "table")

    if kb_format == "json":
        _render_json(sections)
    elif kb_format == "markdown":
        _render_markdown(sections)
    else:
        _render_table(sections, use_color=_table_uses_color(args))

    return 0


__all__ = ["_filter_sections_for_tier", "_render_markdown", "_run_keybindings"]
