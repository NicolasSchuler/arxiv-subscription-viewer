# Architecture Guide

This guide is the human-oriented architecture reference for contributors. Use it when you need to understand where new code belongs, which imports are stable, and how to patch the right symbols in tests.

## Public API Boundaries

### Root package: `arxiv_browser`

The root package is a small explicit public API. Its supported exports are defined by `src/arxiv_browser/__init__.py::__all__`.

Today that surface contains:

- `ArxivBrowser`
- `main`
- core data models such as `Paper`, `PaperMetadata`, `SearchBookmark`, `SessionState`, and `UserConfig`

Do not rely on undocumented extras from the root package. Import helpers, themes, query functions, and other implementation details from their canonical modules instead.

### Compatibility shim: `arxiv_browser.app`

`src/arxiv_browser/app.py` remains only as a narrow backward-compatibility bridge.

Supported compat exports:

- `ArxivBrowser`
- `ArxivBrowserOptions`
- `main`
- `load_config`
- `discover_history_files`
- `_resolve_papers`
- `_configure_logging`
- `_configure_color_mode`
- `_validate_interactive_tty`
- `_fetch_paper_content_async`

Anything else should be treated as unsupported and imported from its canonical module.

## Canonical Internal Imports

Use these paths for new code and for test imports:

- CLI entry point: `arxiv_browser.cli.main`
- App construction: `arxiv_browser.browser.core.ArxivBrowser`
- App options: `arxiv_browser.browser.core.ArxivBrowserOptions`
- Models and constants: `arxiv_browser.models`
- Parsing and history helpers: `arxiv_browser.parsing`
- Query/search helpers: `arxiv_browser.query`
- Export helpers: `arxiv_browser.export`
- Config load/save/import/export: `arxiv_browser.config`
- LLM workflows and storage: `arxiv_browser.llm`
- External service adapters: `arxiv_browser.services.*`
- Modals: `arxiv_browser.modals`
- Reusable widgets: `arxiv_browser.widgets`

Do not import `arxiv_browser.app` from `src/` code. Do not recreate test-side export bundles like the removed `tests.support.canonical_exports`.

## Package Structure

The main package is organized by responsibility:

- `cli.py`: argument parsing, non-interactive commands, startup/bootstrap
- `browser/`: the Textual app, split into `core.py`, `browse.py`, `chrome.py`, and `discovery.py`
- `actions/`: action handlers mixed into `ArxivBrowser`
- `models.py`: data objects and shared constants
- `config.py`: persisted user configuration and metadata import/export
- `parsing.py`, `query.py`, `export.py`: pure-ish domain helpers
- `services/`: async external I/O seams for arXiv, downloads, enrichment, and LLM orchestration
- `modals/` and `widgets/`: reusable UI building blocks

## Where New Features Belong

- Add CLI flags, commands, or startup behavior in `cli.py`.
- Add browser state transitions or list/detail behavior in `browser/`.
- Add user-triggered actions in `actions/`.
- Add reusable visual pieces in `modals/` or `widgets/`.
- Add external API or subprocess orchestration in `services/`.
- Add shared domain logic in `models.py`, `parsing.py`, `query.py`, `export.py`, or `config.py` depending on ownership.

Prefer the narrowest seam that owns the behavior already. Do not add new imports back through `app.py`.

## Test Import And Patch Rules

- Import the canonical module you are testing.
- Patch the symbol where it is resolved, not where it was originally defined.
- Use `arxiv_browser.app` only in dedicated compatibility tests.
- Do not add new imports from `tests.support.canonical_exports`; that bundle has been removed.

Examples:

- Patch `arxiv_browser.cli._resolve_papers` when testing CLI paper selection.
- Patch `arxiv_browser.browser.core.SEARCH_DEBOUNCE_DELAY` when testing browser debounce behavior.
- Patch `arxiv_browser.actions.ui_actions.save_config` when a UI action resolves `save_config` in that module.

## Current Transitional State

- `browser/core.py` owns app construction and orchestration, but no longer owns the CLI entrypoint wrapper.
- `arxiv_browser.app` is intentionally small and only preserves the legacy CLI/bootstrap monkeypatch seam.
- `browser/core.py` still coordinates a broad set of subsystem state, even though behavior has already been split across browser mixins and action modules.
- `database.py` provides a unified `cache.db` for new installs. Legacy per-module SQLite files (`summaries.db`, `relevance.db`, `semantic_scholar.db`, `huggingface.db`) are still supported via dual-path resolution in `resolve_db_path()`.

## LLM Provider Architecture

The LLM subsystem uses a `LLMProvider` protocol (`llm_providers.py`) with two implementations:

- **`CLIProvider`** — invokes an external CLI command (e.g., `llm`, `ollama`)
- **`HTTPProvider`** — targets OpenAI-compatible `/v1/chat/completions` endpoints (OpenAI, Ollama, LM Studio, vLLM)

Providers are registered in a simple registry (`register_provider` / `get_provider_class`). `resolve_provider(config)` reads `config.llm_provider_type` to select the appropriate implementation.

## Follow-Up Extraction Plan

The next safe architecture moves are:

1. Continue narrowing `ArxivBrowser` state ownership by grouping related LLM, enrichment, and recommendation state.
2. Prefer service-layer request/result objects for cross-module workflows instead of more app-level attributes.
3. Keep `ArxivBrowser` as the orchestration shell for UI wiring, not the long-term owner of every mutable concern.
