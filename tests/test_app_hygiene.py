"""Repository hygiene guards for the legacy app compatibility module."""

from __future__ import annotations

from pathlib import Path


def test_only_compat_tests_reference_legacy_app_module() -> None:
    root = Path(__file__).parent
    allowed = {"test_app_compat_cli.py", "test_app_compat_exports.py", "test_app_hygiene.py"}
    bad_files: list[str] = []
    pattern = "arxiv_browser" + ".app"

    for path in sorted(root.glob("test_*.py")):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern in text:
            bad_files.append(path.name)

    assert bad_files == [], (
        "Only dedicated compatibility tests may import or patch arxiv_browser.app. "
        f"Found legacy references in: {', '.join(bad_files)}"
    )
