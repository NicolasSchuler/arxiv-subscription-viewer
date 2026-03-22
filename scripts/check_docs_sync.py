#!/usr/bin/env python3
"""Lightweight docs drift checks for CLI flags, presets, keybindings, config docs, and completions."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src/arxiv_browser/cli.py"
APP_PATH = ROOT / "src/arxiv_browser/app.py"
MODELS_PATH = ROOT / "src/arxiv_browser/models.py"
CONFIG_PATH = ROOT / "src/arxiv_browser/config.py"
LLM_PATH = ROOT / "src/arxiv_browser/llm.py"
COMPLETIONS_PATH = ROOT / "src/arxiv_browser/completions.py"
README_PATH = ROOT / "README.md"
CLAUDE_PATH = ROOT / "CLAUDE.md"
DOCS_DIR = ROOT / "docs"
DOCS_README_PATH = DOCS_DIR / "README.md"
CONFIG_REFERENCE_PATH = DOCS_DIR / "config-reference.md"
DOCS_INDEX_PATH = DOCS_DIR / "index.html"

NON_PERSISTED_USER_CONFIG_FIELDS = frozenset({"config_defaulted"})

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


def _parse_python_module(source: str) -> ast.Module | None:
    """Parse a Python module and return ``None`` on syntax errors."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


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


def _extract_user_config_fields(models_text: str) -> set[str]:
    """Extract ``UserConfig`` field names from ``models.py``."""
    module = _parse_python_module(models_text)
    if module is None:
        return set()

    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "UserConfig":
            continue
        return {
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
    return set()


def _extract_persisted_config_keys(config_text: str) -> set[str]:
    """Extract persisted config keys from ``_config_to_dict`` in ``config.py``."""
    module = _parse_python_module(config_text)
    if module is None:
        return set()

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_config_to_dict":
            continue
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Dict):
                continue

            keys: set[str] = set()
            for key_node, value_node in zip(stmt.value.keys, stmt.value.values, strict=True):
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                key = key_node.value
                if key == "session" and isinstance(value_node, ast.Dict):
                    for nested_key_node in value_node.keys:
                        if isinstance(nested_key_node, ast.Constant) and isinstance(
                            nested_key_node.value, str
                        ):
                            keys.add(f"session.{nested_key_node.value}")
                    continue
                keys.add(key)
            return keys
    return set()


def _extract_config_reference_keys(config_reference_text: str) -> set[str]:
    """Extract documented config keys from ``docs/config-reference.md``."""
    return set(
        re.findall(
            r"`([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)`",
            config_reference_text,
        )
    )


def _extract_markdown_section(text: str, heading: str) -> str:
    """Return the markdown section body for a heading, or an empty string."""
    pattern = re.compile(
        rf"^{re.escape(heading)}\n(.*?)(?=^##\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def _normalize_doc_href(target: str) -> str:
    """Normalize a relative docs link to a comparable basename."""
    return Path(target.split("#", 1)[0].split("?", 1)[0]).name


def _extract_feature_guide_links(docs_readme_text: str) -> set[str]:
    """Extract public feature-guide targets from ``docs/README.md``."""
    feature_guides = _extract_markdown_section(docs_readme_text, "## Feature Guides")
    if not feature_guides:
        return set()
    return {
        normalized
        for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", feature_guides)
        if (normalized := _normalize_doc_href(target)).endswith(".md")
    }


def _extract_html_links(html_text: str) -> set[str]:
    """Extract normalized link targets from an HTML document."""
    return {
        _normalize_doc_href(target)
        for _, target in re.findall(r"""(href)\s*=\s*["']([^"']+)["']""", html_text)
    }


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


def _check_config_reference(
    models_text: str, config_text: str, config_reference_text: str
) -> list[str]:
    """Verify persisted ``UserConfig`` keys are documented in ``config-reference.md``."""
    user_config_fields = _extract_user_config_fields(models_text)
    if not user_config_fields:
        return ["Could not parse UserConfig from src/arxiv_browser/models.py"]

    persisted_keys = _extract_persisted_config_keys(config_text)
    if not persisted_keys:
        return ["Could not parse persisted config keys from src/arxiv_browser/config.py"]

    documented_keys = _extract_config_reference_keys(config_reference_text)
    top_level_persisted_fields = {key.split(".", 1)[0] for key in persisted_keys}

    errors = [
        f"config persistence missing UserConfig field: {field}"
        for field in sorted(
            (user_config_fields - NON_PERSISTED_USER_CONFIG_FIELDS) - top_level_persisted_fields
        )
    ]
    errors.extend(
        f"docs/config-reference.md missing persisted config key: {key}"
        for key in sorted(persisted_keys - documented_keys)
    )
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


def _check_docs_index_navigation(docs_readme_text: str, docs_index_text: str) -> list[str]:
    """Verify the landing page links to the public docs guides."""
    feature_guides = _extract_feature_guide_links(docs_readme_text)
    if not feature_guides:
        return ["Could not parse feature guide links from docs/README.md"]

    docs_index_links = _extract_html_links(docs_index_text)
    return [
        f"docs/index.html missing guide navigation link: {guide}"
        for guide in sorted(feature_guides - docs_index_links)
    ]


def main() -> int:
    models_text = _read(MODELS_PATH)
    config_text = _read(CONFIG_PATH)
    cli_text = _read(CLI_PATH)
    llm_text = _read(LLM_PATH)
    completions_text = _read(COMPLETIONS_PATH)
    _ = _read(APP_PATH)  # Keep app.py as an explicit source-of-truth dependency.
    readme_text = _read(README_PATH)
    claude_text = _read(CLAUDE_PATH)
    config_reference_text = _read(CONFIG_REFERENCE_PATH)
    docs_readme_text = _read(DOCS_README_PATH)
    docs_index_text = _read(DOCS_INDEX_PATH)

    errors: list[str] = []
    errors.extend(_check_cli_flags(readme_text, cli_text))
    errors.extend(_check_llm_presets(readme_text, claude_text, llm_text))
    errors.extend(_check_keybindings(readme_text, claude_text))
    errors.extend(_check_config_reference(models_text, config_text, config_reference_text))
    errors.extend(_check_completions(cli_text, completions_text))
    errors.extend(_check_docs_index_navigation(docs_readme_text, docs_index_text))

    if errors:
        print("Documentation sync check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Documentation sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
