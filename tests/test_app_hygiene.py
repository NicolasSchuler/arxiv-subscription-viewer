"""Repository hygiene guards for the legacy app compatibility module."""

from __future__ import annotations

import ast
from pathlib import Path


def _legacy_app_references(path: Path) -> list[str]:
    """Return import and patch-style references to ``arxiv_browser.app`` in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    refs: list[str] = []
    pattern = "arxiv_browser.app"

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            refs.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name == pattern
            )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module == pattern:
                refs.append(f"line {node.lineno}: from {pattern} import ...")

        def visit_Constant(self, node: ast.Constant) -> None:
            value = node.value
            if isinstance(value, str) and (value == pattern or value.startswith(f"{pattern}.")):
                refs.append(f"line {node.lineno}: string target {value!r}")

    Visitor().visit(tree)
    return refs


def test_only_compat_tests_reference_legacy_app_module() -> None:
    root = Path(__file__).parent
    allowed = {"test_app_compat_cli.py", "test_app_compat_exports.py", "test_app_hygiene.py"}
    bad_files: list[str] = []

    for path in sorted(root.rglob("*.py")):
        rel_path = path.relative_to(root).as_posix()
        if rel_path in allowed:
            continue
        refs = _legacy_app_references(path)
        if refs:
            bad_files.append(f"{rel_path} ({'; '.join(refs)})")

    assert bad_files == [], (
        "Only dedicated compatibility tests may import or patch arxiv_browser.app. "
        f"Found legacy references in: {', '.join(bad_files)}"
    )
