# Agent Instructions

Quick-reference for AI agents. See `CLAUDE.md` for full architecture, patterns, and dependency DAG.

## Task Runner

- **Pre-commit**: `just check` (lint + types + tests)
- **Full suite**: `just quality`
- **Auto-fix**: `just format`
- **Docs drift**: `just docs-check`
- **Dashboard**: `just report`

## Before Making Changes

1. Read the relevant source files before editing
2. Run `just check` to establish a clean baseline
3. After changes, run `just check` again to verify nothing broke

## Must-Pass Gates (CI enforced)

- `just lint` — zero ruff errors, formatting matches
- `just typecheck` — zero pyright errors (basic mode)
- `just test` — all tests pass, overall coverage >= 60%, `app.py` coverage >= 80%
- `uv run xenon src/arxiv_browser/ --max-absolute C --max-modules C --max-average B`
- `just dead-code` / `just security` / `just deps` — zero findings each
- `src/arxiv_browser/app.py` line-count guardrail: <= 5000 lines

## Key Rules

- **No circular imports**: Sub-modules must never import from `app.py`
- **Test mock paths**: Patch at the module where the function is *resolved* (see CLAUDE.md Import Patterns)
- **Modal test imports**: `from arxiv_browser.modals import X` (not `from arxiv_browser.app import X`)
- **`make_paper` fixture**: Sets `abstract_raw = abstract` to prevent async HTTP fetches
- **`conftest.py` autouse fixture**: Resets mutable module-level state between tests
- **Textual integration tests**: Use `pilot.pause()` for debounce waits
