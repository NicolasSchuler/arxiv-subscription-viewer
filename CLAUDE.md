# AI Agent Instructions

> **Primary source of truth:**
> - [`AGENTS.md`](AGENTS.md) — task runner, must-pass gates, key rules
> - [`docs/architecture.md`](docs/architecture.md) — public API boundaries, canonical imports, package structure, module dependency DAG
> - [`CONTRIBUTING.md`](CONTRIBUTING.md) — contributor workflow
> - `README.md` — user-facing key binding table and feature list
>
> This file only captures AI-specific conventions that don't belong in the above.

## Test Mock Paths (AI-specific)

Patch at the module where a symbol is *resolved*, not where it is re-exported:

- Functions called by other functions in the same module: patch at the actual module
  (e.g. `"arxiv_browser.config.get_config_path"`).
- Functions called by `ArxivBrowser`: patch where resolved. Canonical implementations
  live under `arxiv_browser.browser.*`, `arxiv_browser.actions.*`, and other direct modules.
- Modal classes in tests: `from arxiv_browser.modals import X` (not `from arxiv_browser.app import X`).
- Semantic Scholar / HuggingFace: `"arxiv_browser.semantic_scholar.X"`, `"arxiv_browser.huggingface.X"`.

## Test Fixtures

- `make_paper` sets `abstract_raw = abstract` to prevent async HTTP fetches during unit tests.
- `conftest.py` has an autouse fixture that resets mutable module-level state between tests.
- Textual integration tests should use `pilot.pause()` for debounce waits.

## Textual Conventions

- `ListView`, `ListItem`, `Static`, `Input`, `TextArea`, `Button` are the standard widgets.
- Inline CSS lives in class-level `CSS` strings.
- Keybindings are `BINDINGS` class variables with `Binding` objects (see `ui_constants.APP_BINDINGS`).
- Modal screens are `ModalScreen[T]` for dialogs returning typed values.
- Some action methods are `async` to allow DOM updates to settle.

## Caches and Persistence

- Config lives at a platformdirs-resolved path per OS (see `README.md`).
- LLM summaries, Semantic Scholar data, and HuggingFace trending share a unified `cache.db`
  (WAL mode, foreign keys on). Legacy installs continue to use the per-module SQLite files.
- Summaries are invalidated when the LLM command or prompt template changes.

## When This File Drifts

Slim entries out of this file. If content is useful to humans, move it to
`docs/architecture.md` or `CONTRIBUTING.md`. If it's useful to both AIs and humans,
promote it to `AGENTS.md`.
