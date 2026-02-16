set shell := ["bash", "-uc"]

src := "src/arxiv_browser"

# ── Help ─────────────────────────────────────────────────────────────

# List available recipes
[private]
default:
    @just --list

# ── Fast checks (pre-commit) ────────────────────────────────────────

# Run ruff linter and format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint and formatting issues
format:
    uv run ruff check --fix .
    uv run ruff format .

# Run pyright type checker
typecheck:
    uv run pyright

# Verify docs are aligned with CLI flags, presets, and keybindings
docs-check:
    python3 scripts/check_docs_sync.py

# Run tests with coverage report
test:
    uv run pytest --cov --cov-report=term-missing
    uv run coverage report --include=src/arxiv_browser/app.py --fail-under=80

# Run tests without coverage (faster, stop on first failure)
test-quick:
    uv run pytest -x -q

# Run performance benchmarks
bench:
    uv run pytest tests/test_benchmarks.py -v -m slow -s

# ── Quality tools ────────────────────────────────────────────────────

# Show complex functions (C+ rated) and maintainability index
complexity:
    @echo "=== Cyclomatic Complexity (C+ rated) ==="
    @uv run radon cc {{ src }} -a -nc -s
    @echo ""
    @echo "=== Maintainability Index ==="
    @uv run radon mi {{ src }} -s

# Run bandit security scanner
security:
    uv run bandit -c pyproject.toml -r {{ src }}

# Run vulture dead code detection
dead-code:
    uv run vulture {{ src }} vulture_whitelist.py --min-confidence 80

# Check dependency hygiene
deps:
    uv run deptry .

# ── Composite targets ────────────────────────────────────────────────

# Run all fast checks (lint + types + tests)
check: lint typecheck test

# Run all checks including quality tools
quality: check complexity security dead-code deps

# ── CI (matches GitHub Actions + adds quality gates) ─────────────────

# Run CI-equivalent checks locally
ci:
    @echo "=== Lint ==="
    @uv run ruff check .
    @uv run ruff format --check .
    @echo ""
    @echo "=== Type Check ==="
    @uv run pyright
    @echo ""
    @echo "=== Tests ==="
    @uv run pytest --cov --cov-report=term-missing
    @uv run coverage report --include=src/arxiv_browser/app.py --fail-under=80
    @echo ""
    @echo "=== Dependency Check ==="
    @uv run deptry .
    @echo ""
    @echo "=== Dead Code ==="
    @uv run vulture {{ src }} vulture_whitelist.py --min-confidence 80
    @echo ""
    @echo "=== Security ==="
    @uv run bandit -c pyproject.toml -r {{ src }}

# ── Quality Report ───────────────────────────────────────────────────

# Generate a comprehensive quality report
report:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║             Code Quality Report — $(date +%Y-%m-%d)            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    echo "── Lines of Code ──────────────────────────────────────────────"
    wc -l {{ src }}/*.py | sort -n
    echo ""

    echo "── Tests ──────────────────────────────────────────────────────"
    uv run pytest --co -q 2>/dev/null | tail -1
    echo ""

    echo "── Coverage ───────────────────────────────────────────────────"
    COV_OUT=$(uv run pytest --cov -q 2>/dev/null)
    echo "$COV_OUT" | grep -E '(^src/|^TOTAL)' | \
        awk '{printf "  %-45s %5s %6s %5s\n", $1, $2, $3, $4}'
    echo "$COV_OUT" | grep -E '^Required' || true
    echo ""

    echo "── Complexity Distribution ────────────────────────────────────"
    uv run radon cc {{ src }} -a -j 2>/dev/null | python3 -c "
    import json, sys
    data = json.load(sys.stdin)
    total = sum(len(b) for b in data.values())
    by_rank = {}
    for blocks in data.values():
        for b in blocks:
            r = b['rank']
            by_rank[r] = by_rank.get(r, 0) + 1
    print(f'  Total blocks analyzed: {total}')
    for r in sorted(by_rank):
        print(f'  {r}: {by_rank[r]:>4} ({by_rank[r]*100//total}%)')
    "
    echo ""

    echo "── High Complexity Functions (D+F rated) ──────────────────────"
    uv run radon cc {{ src }} -nd -s 2>/dev/null || echo "  (none)"
    echo ""

    echo "── Maintainability Index ──────────────────────────────────────"
    uv run radon mi {{ src }} -j 2>/dev/null | python3 -c "
    import json, sys
    data = json.load(sys.stdin)
    for p, info in sorted(data.items()):
        short = p.replace('src/arxiv_browser/', '')
        print(f'  {info[\"rank\"]} ({info[\"mi\"]:5.1f})  {short}')
    "
    echo ""

    echo "── Lint ───────────────────────────────────────────────────────"
    uv run ruff check . --statistics 2>/dev/null && echo "  No issues" || true
    echo ""

    echo "── Type Check ─────────────────────────────────────────────────"
    uv run pyright 2>&1 | tail -1
    echo ""

    echo "── Security (bandit) ──────────────────────────────────────────"
    uv run bandit -c pyproject.toml -r {{ src }} 2>&1 | grep -E '(Total issues|Files skipped)' || true
    echo ""

    echo "── Dead Code (vulture) ────────────────────────────────────────"
    DEAD=$(uv run vulture {{ src }} vulture_whitelist.py --min-confidence 80 2>/dev/null | wc -l | tr -d ' ')
    echo "  Unused code items: $DEAD"
    echo ""

    echo "── Dependencies (deptry) ──────────────────────────────────────"
    uv run deptry . 2>&1 | tail -1
    echo ""
    echo "══════════════════════════════════════════════════════════════"

# ── Cleanup ──────────────────────────────────────────────────────────

# Remove build artifacts and caches
clean:
    rm -rf .pytest_cache .ruff_cache .pyright htmlcov .coverage coverage.xml
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
