#!/usr/bin/env python3
"""Validate statement and branch coverage thresholds from coverage JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check statement and branch coverage thresholds from coverage.json",
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="coverage.json",
        help="Path to coverage JSON report (default: coverage.json)",
    )
    parser.add_argument(
        "--statements",
        type=float,
        default=95.0,
        help="Minimum required statement coverage percentage",
    )
    parser.add_argument(
        "--branches",
        type=float,
        default=85.0,
        help="Minimum required branch coverage percentage",
    )
    return parser.parse_args()


def _load_totals(report_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Coverage report not found: {report_path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid coverage JSON in {report_path}: {exc}") from exc

    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise SystemExit(f"Coverage report missing 'totals': {report_path}")
    return totals


def main() -> int:
    args = _parse_args()
    report_path = Path(args.report)
    totals = _load_totals(report_path)

    statements = float(totals.get("percent_statements_covered", 0.0))
    branches_raw = totals.get("percent_branches_covered")
    if branches_raw is None:
        raise SystemExit(
            f"Coverage report {report_path} does not include branch coverage. "
            "Run pytest with --cov-branch."
        )
    branches = float(branches_raw)

    statements_ok = statements >= args.statements
    branches_ok = branches >= args.branches
    passed = statements_ok and branches_ok

    summary = {
        "report": str(report_path),
        "statement_coverage": round(statements, 2),
        "required_statement_coverage": args.statements,
        "branch_coverage": round(branches, 2),
        "required_branch_coverage": args.branches,
        "passed": passed,
    }
    print(json.dumps(summary, sort_keys=True))

    if passed:
        return 0

    failed_parts: list[str] = []
    if not statements_ok:
        failed_parts.append(
            f"statement coverage {statements:.2f}% < required {args.statements:.2f}%"
        )
    if not branches_ok:
        failed_parts.append(f"branch coverage {branches:.2f}% < required {args.branches:.2f}%")
    print("; ".join(failed_parts), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
