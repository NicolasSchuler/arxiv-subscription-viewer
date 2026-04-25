"""Repository hygiene guards for the legacy app compatibility module."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
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


def _canonical_export_bundle_references(path: Path) -> list[str]:
    """Return import-style references to the removed test export bundle."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    refs: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module == "tests.support.canonical_exports":
                refs.append(f"line {node.lineno}: from tests.support.canonical_exports import ...")
            if node.module == "tests.support" and any(
                alias.name == "canonical_exports" for alias in node.names
            ):
                refs.append(f"line {node.lineno}: from tests.support import canonical_exports")

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


def test_src_modules_do_not_import_legacy_app_module() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "arxiv_browser"
    bad_files: list[str] = []

    for path in sorted(src_root.rglob("*.py")):
        if path.name == "app.py":
            continue
        refs = _legacy_app_references(path)
        if refs:
            bad_files.append(f"{path.relative_to(src_root).as_posix()} ({'; '.join(refs)})")

    assert bad_files == [], (
        "Only the compatibility bridge may reference arxiv_browser.app from src/. "
        f"Found legacy references in: {', '.join(bad_files)}"
    )


def test_tests_do_not_import_removed_canonical_export_bundle() -> None:
    root = Path(__file__).parent
    bad_files: list[str] = []

    for path in sorted(root.rglob("*.py")):
        if path.name == "test_app_hygiene.py":
            continue
        refs = _canonical_export_bundle_references(path)
        if refs:
            bad_files.append(f"{path.relative_to(root).as_posix()} ({'; '.join(refs)})")

    assert bad_files == [], (
        "Tests must import canonical modules directly instead of tests.support.canonical_exports. "
        f"Found bundle references in: {', '.join(bad_files)}"
    )


def test_src_tree_avoids_repo_local_import_star_and_dynamic_dunder_all() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "arxiv_browser"
    bad_files: list[str] = []

    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        rel_path = path.relative_to(src_root).as_posix()

        if "__all__ = [name for name in globals()" in text:
            bad_files.append(f"{rel_path} (dynamic __all__ from globals())")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                module_name = node.module or ""
                if module_name.startswith("arxiv_browser"):
                    bad_files.append(
                        f"{rel_path} (line {node.lineno}: wildcard import from {module_name})"
                    )

    assert bad_files == [], (
        "src/arxiv_browser must avoid repo-local wildcard imports and dynamic globals()-derived __all__. "
        f"Found violations in: {', '.join(bad_files)}"
    )


def test_browser_core_uses_service_interfaces_not_concrete_service_modules() -> None:
    """The app shell should depend on the service aggregate seam."""
    path = Path(__file__).resolve().parents[1] / "src" / "arxiv_browser" / "browser" / "core.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name.startswith("arxiv_browser.services.") and (
                module_name != "arxiv_browser.services.interfaces"
            ):
                bad_imports.append(f"line {node.lineno}: from {module_name} import ...")
        elif isinstance(node, ast.Import):
            bad_imports.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name.startswith("arxiv_browser.services.")
                and alias.name != "arxiv_browser.services.interfaces"
            )

    assert bad_imports == [], (
        "browser/core.py should consume AppServices from arxiv_browser.services.interfaces "
        f"instead of concrete service modules. Found: {', '.join(bad_imports)}"
    )


def test_ui_packages_import_in_fresh_process() -> None:
    """Public UI packages should import without entering modal cycles."""
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-c", "import arxiv_browser.widgets; import arxiv_browser.modals"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
