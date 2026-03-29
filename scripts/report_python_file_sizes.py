#!/usr/bin/env python3
"""Report repo-tracked Python files that are near or above the soft line cap."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SOFT_CAP_LINES = 1000
NEAR_CAP_LINES = 900
IGNORED_PATH_PARTS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "node_modules",
    "vendor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soft-cap",
        type=int,
        default=SOFT_CAP_LINES,
        help="Soft maximum line count for tracked Python files.",
    )
    parser.add_argument(
        "--near-cap",
        type=int,
        default=NEAR_CAP_LINES,
        help="Threshold for near-cap reporting.",
    )
    parser.add_argument(
        "--path-prefix",
        default="",
        help="Restrict scan to tracked files under this path prefix.",
    )
    parser.add_argument(
        "--github-actions",
        action="store_true",
        help="Emit GitHub Actions warning annotations for near-cap and over-cap files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when files exceed the soft cap.",
    )
    return parser.parse_args()


def tracked_python_files(repo_root: Path, path_prefix: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = []
    for line in result.stdout.splitlines():
        path = repo_root / line
        if any(part in IGNORED_PATH_PARTS for part in path.parts):
            continue
        if path_prefix and not path.is_relative_to(path_prefix):
            continue
        files.append(path)
    return sorted(files)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def emit_section(
    *,
    title: str,
    entries: list[tuple[Path, int]],
    repo_root: Path,
    github_actions: bool,
    annotation_level: str,
) -> None:
    print(title)
    if not entries:
        print("  (none)")
        return
    for path, lines in entries:
        rel_path = path.relative_to(repo_root)
        print(f"  {lines:>5}  {rel_path}")
        if github_actions:
            print(f"::{annotation_level} file={rel_path}::{rel_path} has {lines} lines")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    path_prefix = (repo_root / args.path_prefix).resolve() if args.path_prefix else repo_root
    files = tracked_python_files(repo_root, path_prefix)
    missing_files = [path for path in files if not path.exists()]
    counts = [(path, count_lines(path)) for path in files if path.exists()]

    over_cap = [(path, lines) for path, lines in counts if lines > args.soft_cap]
    near_cap = [(path, lines) for path, lines in counts if args.near_cap < lines <= args.soft_cap]

    scope = args.path_prefix or "tracked files"
    print(
        "Python file-size report "
        f"(scope={scope}, soft cap: {args.soft_cap} lines, near-cap threshold: {args.near_cap} lines)"
    )
    print(f"Tracked Python files scanned: {len(counts)}")
    if missing_files:
        print(f"Missing tracked files skipped: {len(missing_files)}")
    print()

    emit_section(
        title=f"Over soft cap (>{args.soft_cap} lines)",
        entries=over_cap,
        repo_root=repo_root,
        github_actions=args.github_actions,
        annotation_level="warning",
    )
    print()
    emit_section(
        title=f"Near cap (>{args.near_cap} and <= {args.soft_cap} lines)",
        entries=near_cap,
        repo_root=repo_root,
        github_actions=args.github_actions,
        annotation_level="notice" if args.github_actions else "warning",
    )
    print()

    if over_cap:
        print(
            "Recommendation: keep new code out of over-cap files and extract cohesive "
            "modules or test support helpers before adding more logic."
        )
    elif near_cap:
        print(
            "Recommendation: monitor near-cap files and prefer small extractions before they grow."
        )
    else:
        print("All tracked Python files are within the soft cap.")

    return 1 if args.strict and over_cap else 0


if __name__ == "__main__":
    sys.exit(main())
