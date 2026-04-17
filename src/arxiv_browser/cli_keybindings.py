"""CLI keybinding output helpers."""

from __future__ import annotations

import argparse
import json

from arxiv_browser.help_ui import build_help_sections
from arxiv_browser.ui_constants import APP_BINDINGS


def _run_keybindings(args: argparse.Namespace) -> int:
    """Print keyboard shortcuts to stdout in the requested format."""
    sections = build_help_sections(APP_BINDINGS)

    tier = getattr(args, "tier", "all")
    if tier != "all":
        tier_map: dict[str, set[str]] = {
            "core": {"Getting Started", "Search Syntax", "Core Actions"},
            "standard": {"Standard \u00b7 Organize", "Standard - Organize"},
            "power": {
                "Power \u00b7 Research Tools",
                "Power \u00b7 Advanced",
                "Power - Research Tools",
                "Power - Advanced",
            },
        }
        allowed = tier_map.get(tier, set())
        sections = [(name, entries) for name, entries in sections if name in allowed]

    kb_format = getattr(args, "kb_format", "table")

    if kb_format == "json":
        data = [
            {"section": name, "bindings": [{"key": k, "description": d} for k, d in entries]}
            for name, entries in sections
        ]
        print(json.dumps(data, indent=2))
    elif kb_format == "markdown":
        for name, entries in sections:
            print(f"## {name}\n")
            print("| Key | Action |")
            print("|-----|--------|")
            for key, desc in entries:
                print(f"| `{key}` | {desc} |")
            print()
    else:
        for name, entries in sections:
            print(f"\033[1m{name}\033[0m")
            for key, desc in entries:
                print(f"  \033[32m{key:<20}\033[0m {desc}")
            print()

    return 0


__all__ = ["_run_keybindings"]
