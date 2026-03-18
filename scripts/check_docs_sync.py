#!/usr/bin/env python3
"""Lightweight docs drift checks for CLI flags, presets, keybindings, and completions."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src/arxiv_browser/cli.py"
APP_PATH = ROOT / "src/arxiv_browser/app.py"
LLM_PATH = ROOT / "src/arxiv_browser/llm.py"
COMPLETIONS_PATH = ROOT / "src/arxiv_browser/completions.py"
README_PATH = ROOT / "README.md"
CLAUDE_PATH = ROOT / "CLAUDE.md"
DOCS_DIR = ROOT / "docs"

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


def _extract_table_keys(text: str) -> set[str]:
    """Extract keybinding keys from all markdown tables in the given text.

    Handles single-column tables (| Key | Action |) and double-column
    tables (| Key | Action | | Key | Action |) as used in the README.
    Also extracts backtick-wrapped keys from any table column that looks
    like a keybinding (e.g. | `Ctrl+s` | Generate summary |).
    """
    keys: set[str] = set()
    # Match backtick-wrapped keybindings anywhere in table rows
    backtick_key_re = re.compile(r"`([^`]+)`")
    key_like_re = re.compile(r"^(Ctrl\+|Escape|Space|[A-Z]$|[a-z]$|/|\\?|\[|\]|[0-9]|')")

    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 2:
            continue
        # Skip header/separator rows
        if all(c.startswith("-") or c in {"Key", "-----", "---", ""} for c in cols):
            continue

        # Strategy 1: first column (and 4th column for double-width tables)
        for idx in (0, 3):
            if idx < len(cols) and cols[idx] not in {"", "Key", "-----", "---"}:
                keys.update(_split_key_field(cols[idx]))

        # Strategy 2: backtick-wrapped keys in any column
        for col in cols:
            for m in backtick_key_re.finditer(col):
                candidate = m.group(1).strip()
                if key_like_re.match(candidate):
                    keys.update(_split_key_field(candidate))
    return keys


def _extract_readme_keys(readme_text: str) -> set[str]:
    """Extract keys from README tables and all docs/*.md files."""
    keys = _extract_table_keys(readme_text)
    if DOCS_DIR.is_dir():
        for doc_file in DOCS_DIR.glob("*.md"):
            keys.update(_extract_table_keys(doc_file.read_text(encoding="utf-8")))
    return keys


def _extract_claude_keys(claude_text: str) -> set[str]:
    start = claude_text.find("## Key Bindings Reference")
    if start == -1:
        return set()
    remainder = claude_text[start + len("## Key Bindings Reference") :]

    # Try markdown table first (| `key` | action |)
    table_keys = _extract_table_keys(remainder)
    if table_keys:
        return table_keys

    # Fall back to code block format (key - description)
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
    # Combine README + docs for flag lookup
    docs_text = ""
    if DOCS_DIR.is_dir():
        docs_text = "\n".join(f.read_text(encoding="utf-8") for f in DOCS_DIR.glob("*.md"))
    combined = readme_text + "\n" + docs_text
    return [
        f"README/docs is missing CLI option group: {' / '.join(group)}"
        for group in _extract_cli_flag_groups(cli_text)
        if not any(flag in combined for flag in group)
    ]


def _check_llm_presets(readme_text: str, claude_text: str, llm_text: str) -> list[str]:
    errors: list[str] = []
    presets = _extract_llm_presets(llm_text)
    if not presets:
        return ["Could not parse LLM_PRESETS from src/arxiv_browser/llm.py"]

    # Collect all docs text for preset lookup
    docs_text = ""
    if DOCS_DIR.is_dir():
        docs_text = "\n".join(f.read_text(encoding="utf-8") for f in DOCS_DIR.glob("*.md"))
    combined_readme = readme_text + "\n" + docs_text
    combined_claude = claude_text + "\n" + docs_text

    for preset in sorted(presets):
        needle = f"`{preset}`"
        if needle not in combined_readme:
            errors.append(f"README/docs is missing preset: {needle}")
        if needle not in combined_claude:
            errors.append(f"CLAUDE.md/docs is missing preset: {needle}")
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


def _extract_cli_commands(cli_text: str) -> set[str]:
    """Extract the subcommand names from the ``CLI_COMMANDS`` tuple in cli.py."""
    match = re.search(r"CLI_COMMANDS\s*=\s*\((.*?)\)", cli_text, re.DOTALL)
    if not match:
        return set()
    return set(re.findall(r'"([a-z][-a-z]*)"', match.group(1)))


def _extract_completion_script(completions_text: str, variable_name: str) -> str | None:
    """Extract a raw completion script block from completions.py."""
    match = re.search(
        rf"{re.escape(variable_name)}\s*=\s*r?\"\"\"(.*?)\"\"\"",
        completions_text,
        re.DOTALL,
    )
    if not match:
        return None
    return match.group(1)


def _extract_bash_commands(script: str) -> set[str]:
    """Extract subcommands from the bash completion script."""
    match = re.search(r'local commands="([^"]+)"', script)
    if not match:
        return set()
    return set(match.group(1).split())


def _extract_zsh_commands(script: str) -> set[str]:
    """Extract subcommands from the zsh completion script."""
    return set(re.findall(r"'([a-z][-a-z]*):", script))


def _extract_fish_commands(script: str) -> set[str]:
    """Extract subcommands from the fish completion script."""
    return set(
        re.findall(
            r"^complete -c arxiv-viewer -n '__fish_use_subcommand' -a ([a-z][-a-z]*)\b",
            script,
            re.MULTILINE,
        )
    )


def _check_completions(cli_text: str, completions_text: str) -> list[str]:
    """Verify that every CLI subcommand appears in all three completion scripts."""
    commands = _extract_cli_commands(cli_text)
    if not commands:
        return ["Could not parse CLI_COMMANDS from cli.py"]

    shell_commands = {
        "bash": _extract_bash_commands(
            _extract_completion_script(completions_text, "_BASH_SCRIPT") or ""
        ),
        "zsh": _extract_zsh_commands(
            _extract_completion_script(completions_text, "_ZSH_SCRIPT") or ""
        ),
        "fish": _extract_fish_commands(
            _extract_completion_script(completions_text, "_FISH_SCRIPT") or ""
        ),
    }

    errors: list[str] = []
    for shell, extracted_commands in shell_commands.items():
        if not extracted_commands:
            errors.append(
                f"Could not parse {shell} completion commands from src/arxiv_browser/completions.py"
            )
            continue
        errors.extend(
            f"{shell} completions missing subcommand: {cmd}"
            for cmd in sorted(commands - extracted_commands)
        )
    return errors


def main() -> int:
    cli_text = _read(CLI_PATH)
    llm_text = _read(LLM_PATH)
    completions_text = _read(COMPLETIONS_PATH)
    _ = _read(APP_PATH)  # Keep app.py as an explicit source-of-truth dependency.
    readme_text = _read(README_PATH)
    claude_text = _read(CLAUDE_PATH)

    errors: list[str] = []
    errors.extend(_check_cli_flags(readme_text, cli_text))
    errors.extend(_check_llm_presets(readme_text, claude_text, llm_text))
    errors.extend(_check_keybindings(readme_text, claude_text))
    errors.extend(_check_completions(cli_text, completions_text))

    if errors:
        print("Documentation sync check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Documentation sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
