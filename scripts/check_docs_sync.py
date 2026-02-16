#!/usr/bin/env python3
"""Lightweight docs drift checks for CLI flags, presets, and keybindings."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src/arxiv_browser/cli.py"
APP_PATH = ROOT / "src/arxiv_browser/app.py"
LLM_PATH = ROOT / "src/arxiv_browser/llm.py"
README_PATH = ROOT / "README.md"
CLAUDE_PATH = ROOT / "CLAUDE.md"

REQUIRED_KEYBINDINGS: set[str] = {
    "/",
    "Escape",
    "A",
    "E",
    "R",
    "G",
    "V",
    "Ctrl+e",
    "Ctrl+Shift+b",
    "Ctrl+s",
    "C",
    "Ctrl+g",
    "Ctrl+h",
    "L",
    "Ctrl+l",
    "Ctrl+p",
    "Ctrl+k",
    "[",
    "]",
    "?",
    "q",
}

KEY_ALIAS_MAP: dict[str, str] = {
    "slash": "/",
    "escape": "Escape",
    "space": "Space",
    "apostrophe": "'",
    "bracketleft": "[",
    "bracketright": "]",
    "question_mark": "?",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_cli_flag_groups(cli_text: str) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    pattern = re.compile(r"parser\\.add_argument\\((.*?)\\)\\n", re.DOTALL)
    for block in pattern.findall(cli_text):
        flags = tuple(re.findall(r'"(--?[a-zA-Z0-9-]+)"', block))
        if flags:
            groups.append(flags)
    return groups


def _extract_llm_presets(llm_text: str) -> set[str]:
    match = re.search(
        r"LLM_PRESETS\s*:\s*dict\[str,\s*str\]\s*=\s*\{(.*?)\}\n\n", llm_text, re.DOTALL
    )
    if not match:
        return set()
    body = match.group(1)
    return set(re.findall(r'"([a-z0-9_-]+)"\s*:', body))


def _canonical_key(token: str) -> str:
    t = token.strip().strip("`").strip()
    if not t:
        return ""
    lower = t.lower()
    if lower in KEY_ALIAS_MAP:
        return KEY_ALIAS_MAP[lower]
    if lower.startswith("ctrl+"):
        rest = t[5:]
        rest = rest.replace("shift+", "Shift+")
        return f"Ctrl+{rest}"
    if lower == "space":
        return "Space"
    if lower == "escape":
        return "Escape"
    return t


def _split_key_field(key_field: str) -> set[str]:
    cleaned = _canonical_key(key_field)
    if not cleaned:
        return set()
    if "/" in cleaned and cleaned != "/":
        return {_canonical_key(part) for part in cleaned.split("/") if _canonical_key(part)}
    return {cleaned}


def _extract_readme_keys(readme_text: str) -> set[str]:
    start = readme_text.find("## Keyboard Shortcuts")
    if start == -1:
        return set()
    remainder = readme_text[start + len("## Keyboard Shortcuts") :]
    end = remainder.find("\n## ")
    section = remainder if end == -1 else remainder[:end]

    keys: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 2:
            continue
        if cols[0] in {"Key", "-----"}:
            continue
        keys.update(_split_key_field(cols[0]))
    return keys


def _extract_claude_keys(claude_text: str) -> set[str]:
    start = claude_text.find("## Key Bindings Reference")
    if start == -1:
        return set()
    remainder = claude_text[start + len("## Key Bindings Reference") :]

    block_match = re.search(r"```\n(.*?)\n```", remainder, re.DOTALL)
    if not block_match:
        return set()
    block = block_match.group(1)

    keys: set[str] = set()
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or "-" not in line:
            continue
        key_part = line.split("-", 1)[0].strip()
        keys.update(_split_key_field(key_part))
    return keys


def _check_cli_flags(readme_text: str, cli_text: str) -> list[str]:
    return [
        f"README is missing CLI option group: {' / '.join(group)}"
        for group in _extract_cli_flag_groups(cli_text)
        if not any(flag in readme_text for flag in group)
    ]


def _check_llm_presets(readme_text: str, claude_text: str, llm_text: str) -> list[str]:
    errors: list[str] = []
    presets = _extract_llm_presets(llm_text)
    if not presets:
        return ["Could not parse LLM_PRESETS from src/arxiv_browser/llm.py"]

    for preset in sorted(presets):
        needle = f"`{preset}`"
        if needle not in readme_text:
            errors.append(f"README is missing preset: {needle}")
        if needle not in claude_text:
            errors.append(f"CLAUDE.md is missing preset: {needle}")
    return errors


def _check_keybindings(readme_text: str, claude_text: str) -> list[str]:
    errors: list[str] = []

    readme_keys = _extract_readme_keys(readme_text)
    claude_keys = _extract_claude_keys(claude_text)

    missing_readme = sorted(REQUIRED_KEYBINDINGS - readme_keys)
    missing_claude = sorted(REQUIRED_KEYBINDINGS - claude_keys)

    if missing_readme:
        errors.append("README keyboard shortcuts missing keys: " + ", ".join(missing_readme))
    if missing_claude:
        errors.append("CLAUDE.md key bindings missing keys: " + ", ".join(missing_claude))
    return errors


def main() -> int:
    cli_text = _read(CLI_PATH)
    llm_text = _read(LLM_PATH)
    _ = _read(APP_PATH)  # Keep app.py as an explicit source-of-truth dependency.
    readme_text = _read(README_PATH)
    claude_text = _read(CLAUDE_PATH)

    errors: list[str] = []
    errors.extend(_check_cli_flags(readme_text, cli_text))
    errors.extend(_check_llm_presets(readme_text, claude_text, llm_text))
    errors.extend(_check_keybindings(readme_text, claude_text))

    if errors:
        print("Documentation sync check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Documentation sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
