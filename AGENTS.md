# Agent Instructions

Instructions for AI agents working on this codebase.

## Quick Reference

- **Task runner**: `just` (not `make`). Run `just` to list all recipes.
- **Pre-commit checks**: `just check` (lint + types + tests)
- **Full quality suite**: `just quality`
- **Quality dashboard**: `just report`
- **Auto-fix formatting**: `just format`

## Before Making Changes

1. Read the relevant source files before editing
2. Run `just check` to establish a clean baseline
3. After changes, run `just check` again to verify nothing broke

## Code Quality Standards

### Must-pass gates (CI enforced)

- `just lint` — zero ruff errors, formatting matches
- `just typecheck` — zero pyright errors (basic mode)
- `just test` — all tests pass, total coverage >= 60% and `src/arxiv_browser/app.py` coverage >= 80%
- `uv run xenon src/arxiv_browser/ --max-absolute C --max-modules C --max-average B`
- `just dead-code` — zero vulture findings
- `just security` — zero Bandit findings (use targeted `# nosec <rule-id>` only with inline justification)
- `just deps` — no dependency issues
- `src/arxiv_browser/app.py` line-count guardrail: <= 5000 lines

### Advisory checks (non-blocking quality targets)

- `just report` dashboard metrics (coverage/complexity/maintainability trends)
- Per-module coverage ratchets for extracted UI modules (`modals/*`, `widgets/*`)

### Quality metrics to track

Run `just report` for the full dashboard. Key metrics:

| Metric | Current | Target |
|--------|---------|--------|
| Test coverage (overall) | 69% | 80% |
| Test coverage (app.py) | 81% | 80% |
| Complexity (F-rated functions) | 1 | 0 |
| Complexity (D-rated functions) | 4 | 0 |
| Maintainability index (app.py) | C (0.0) | B |
| Lines in app.py | 8,341 | <4,000 |

## Architecture Rules

- **No circular imports**: Sub-modules must never import from `app.py`
- **Dependency DAG**: See CLAUDE.md for the module dependency graph
- **`app.py` re-exports**: All public symbols from sub-modules are re-exported via star imports in `app.py` for backward compatibility
- **Test mock paths**: Patch at the module where the function is *resolved* (see CLAUDE.md Import Patterns)

## Testing Conventions

- All test imports use `from arxiv_browser.app import ...` (backward-compat bridge)
- `make_paper` fixture in `conftest.py` sets `abstract_raw = abstract` to prevent async HTTP fetches
- `conftest.py` autouse fixture resets mutable module-level state between tests
- Use `pilot.pause()` for Textual integration tests that need debounce waits

## File Organization

```
justfile                  # Task runner (quality checks, testing, CI)
pyproject.toml            # Build config + all tool configurations
vulture_whitelist.py      # Dead code detection false positive suppressions
src/arxiv_browser/        # Application source
tests/                    # Test suite
docs/                     # Documentation assets
.github/workflows/        # CI/CD pipeline
```
