#!/usr/bin/env python3
"""Verify that pyproject.toml version matches the latest released CHANGELOG entry.

Rules:
- `pyproject.toml` version must equal the most recent released version in CHANGELOG.md
  (the first `## [X.Y.Z]` heading), OR
- If CHANGELOG.md has no released version at all, `pyproject.toml` version is accepted as-is.

Exits non-zero on mismatch with a helpful message.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"

RELEASED_HEADING = re.compile(r"^##\s+\[(\d+\.\d+\.\d+)\]")
FOOTER_LINK = re.compile(r"^\[(\d+\.\d+\.\d+)\]:\s*(\S+)", re.MULTILINE)
UNRELEASED_LINK = re.compile(r"^\[Unreleased\]:\s*(\S+)", re.MULTILINE)


def pyproject_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def latest_released_version() -> str | None:
    for line in CHANGELOG.read_text(encoding="utf-8").splitlines():
        match = RELEASED_HEADING.match(line)
        if match:
            return match.group(1)
    return None


def changelog_footer_errors(latest: str) -> list[str]:
    """Verify the footer has a link for the latest release and a current Unreleased compare."""
    text = CHANGELOG.read_text(encoding="utf-8")
    errors: list[str] = []
    if latest not in {m.group(1) for m in FOOTER_LINK.finditer(text)}:
        errors.append(
            f"CHANGELOG.md is missing a footer reference link for released version [{latest}]"
        )
    unreleased = UNRELEASED_LINK.search(text)
    if unreleased is None:
        errors.append("CHANGELOG.md is missing the [Unreleased] footer link")
    elif f"v{latest}...HEAD" not in unreleased.group(1):
        errors.append(
            f"CHANGELOG.md [Unreleased] link should compare from v{latest}...HEAD "
            f"(found {unreleased.group(1)})"
        )
    return errors


def main() -> int:
    pyproject = pyproject_version()
    latest = latest_released_version()
    if latest is None:
        print(f"ok: pyproject {pyproject}; CHANGELOG has no released entries yet")
        return 0
    if pyproject != latest:
        print(
            f"ERROR: pyproject.toml version {pyproject!r} does not match "
            f"latest released CHANGELOG entry {latest!r}.",
            file=sys.stderr,
        )
        print(
            "Either bump pyproject.toml to match, or add a new [Unreleased] section "
            "and cut a CHANGELOG release entry for the new version before tagging.",
            file=sys.stderr,
        )
        print(
            "If the version bump was accidental, revert pyproject.toml and uv.lock "
            "to the latest documented release instead.",
            file=sys.stderr,
        )
        return 1
    footer_errors = changelog_footer_errors(latest)
    if footer_errors:
        print("ERROR: CHANGELOG.md footer links are out of sync:", file=sys.stderr)
        for err in footer_errors:
            print(f"- {err}", file=sys.stderr)
        return 1
    print(f"ok: pyproject {pyproject} matches latest CHANGELOG entry; footer links consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
