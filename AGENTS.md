# Agent Instructions

Quick-reference for AI agents. See `CLAUDE.md` for full architecture, patterns, and dependency DAG.

## Task Runner

- **Pre-commit**: `just check` (lint + types + tests)
- **Full suite**: `just quality`
- **Auto-fix**: `just format`
- **Docs drift**: `just docs-check`
- **Dashboard**: `just report`
- **Contributor docs**: `CONTRIBUTING.md`

## Before Making Changes

1. Read the relevant source files before editing
2. For code changes, run `just check` to establish a clean baseline
3. After changes, run `just check` again to verify nothing broke (`just docs-check` is the targeted minimum for docs-sync edits; see `CONTRIBUTING.md`)

## Must-Pass Gates (CI enforced)

- `just lint` — zero ruff errors, formatting matches
- `just typecheck` — zero pyright errors (basic mode)
- `just test` — all tests pass, overall coverage >= 95% (from `pyproject.toml`), `app.py` coverage >= 80%, and the signature-count structural guard passes
- `uv run xenon src/arxiv_browser/ --max-absolute C --max-modules C --max-average B`
- `just dead-code` / `just security` / `just deps` — zero findings each
- Repo-tracked Python file-size soft-cap report: warning-only at > 1250 lines, near-cap reporting at > 1000 lines
- `src/arxiv_browser/app.py` line-count guardrail: <= 5000 lines

## Key Rules

- **Avoid circular imports**: Sub-modules should import canonical modules directly; only the narrow compatibility bridge may import from `app.py` when preserving legacy patch surfaces.
- **Soft Python file-size target**: Keep repo-tracked `.py` files at or below 1250 lines when feasible; do not grow existing over-cap files, and prefer extracting cohesive modules/tests over adding more code to them
- **Function signature limit**: No new function or method in `src/` may exceed 6 effective named parameters; group related inputs into dataclasses, state objects, or request objects
- **Test mock paths**: Patch at the module where the function is *resolved* (see CLAUDE.md Import Patterns)
- **Modal test imports**: `from arxiv_browser.modals import X` (not `from arxiv_browser.app import X`)
- **`make_paper` fixture**: Sets `abstract_raw = abstract` to prevent async HTTP fetches
- **`conftest.py` autouse fixture**: Resets mutable module-level state between tests
- **Textual integration tests**: Use `pilot.pause()` for debounce waits
