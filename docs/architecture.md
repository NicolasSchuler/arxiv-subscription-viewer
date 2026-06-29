# Architecture Guide

This guide is the human-oriented architecture reference for contributors. Use it when you need to understand where new code belongs, which imports are stable, and how to patch the right symbols in tests.

## Public API Boundaries

### Root package: `arxiv_browser`

The root package is a small explicit public API. Its supported exports are defined by `src/arxiv_browser/__init__.py::__all__`.

The current exports (the authoritative list is `src/arxiv_browser/__init__.py::__all__`):

- Entry points: `ArxivBrowser`, `main`
- Data models: `Paper`, `PaperMetadata`, `SearchBookmark`, `SessionState`, `UserConfig`, `ArxivSearchModeState`, `ArxivSearchRequest`, `LocalBrowseSnapshot`, `PaperCollection`, `QueryToken`, `WatchListEntry`

Do not rely on undocumented extras from the root package. Import helpers, themes, query functions, and other implementation details from their canonical modules instead.

### Compatibility shim: `arxiv_browser.app`

`src/arxiv_browser/app.py` is a narrow compatibility shim. It re-exports a fixed set of
symbols so that monkeypatch-based tests and external callers can keep importing them from
`arxiv_browser.app`; it owns no behavior of its own.

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
- Internal service layer (async I/O + subprocess orchestration): `arxiv_browser.services.*`
- Modals: `arxiv_browser.modals`
- Reusable widgets: `arxiv_browser.widgets`

Do not import `arxiv_browser.app` from `src/` code. Tests import canonical modules directly; a hygiene check (`tests/test_app_hygiene.py`) enforces this.

## Package Structure

The main package is organized by responsibility:

- `cli.py`: argument parsing, non-interactive commands, startup/bootstrap
- `browser/`: the Textual app, split into `core.py`, `browse.py`, `detail_pane.py`, and `discovery.py`
- `actions/`: action handlers mixed into `ArxivBrowser`
- `models.py`: data objects and shared constants
- `config.py`: persisted user configuration and metadata import/export
- `parsing.py`, `query.py`, `export.py`: pure-ish domain helpers
- `sources.py`: preprint provider identity and prototype non-arXiv parsers
- `services/`: internal service layer — async I/O and subprocess orchestration for arXiv, downloads, enrichment, and LLMs
- `modals/` and `widgets/`: reusable UI building blocks

## Where New Features Belong

- Add CLI flags, commands, or startup behavior in `cli.py`.
- Add browser state transitions or list/detail behavior in `browser/`.
- Add user-triggered actions in `actions/`.
- Add reusable visual pieces in `modals/` or `widgets/`.
- Add async I/O or subprocess orchestration in `services/`.
- Add shared domain logic in `models.py`, `parsing.py`, `query.py`, `export.py`, or `config.py` depending on ownership.

Prefer the narrowest seam that owns the behavior already. Do not add new imports back through `app.py`.

## Test Import And Patch Rules

- Import the canonical module you are testing.
- Patch the symbol where it is resolved, not where it was originally defined.
- Use `arxiv_browser.app` only in dedicated compatibility tests.

Examples:

- Patch `arxiv_browser.cli._resolve_papers` when testing CLI paper selection.
- Patch `arxiv_browser.browser.core.SEARCH_DEBOUNCE_DELAY` when testing browser debounce behavior.
- Patch `arxiv_browser.actions.ui_actions.save_config` when a UI action resolves `save_config` in that module.

## Module Dependency Layers

Imports flow one way: higher layers import lower ones, never upward. This is the layering the
"avoid circular imports" rule keeps intact (sub-modules import canonical modules directly):

```
cli.py            entry point / bootstrap
  ↓
browser/          the Textual app (core, browse, detail_pane, discovery)
  ↓
actions/          user-triggered action handlers mixed into ArxivBrowser
  ↓
modals/ widgets/  reusable UI building blocks
  ↓
services/ llm.py export.py query.py parsing.py config.py database.py sources.py   (domain + I/O)
  ↓
models.py         data objects & shared constants (no intra-package imports)
```

The narrow `arxiv_browser.app` shim is the one deliberate exception: it imports from `cli.py`/`browser/`
only to re-export legacy patch surfaces, and nothing under `src/` imports it.

## State Ownership & Persistence

- `browser/core.py` owns app construction and orchestration; the CLI entrypoint lives in `cli.py`.
- `arxiv_browser.app` is a small compatibility shim that preserves the CLI/bootstrap monkeypatch seam.
- `browser/core.py` coordinates subsystem state and wires the browser mixins and action modules together.
- `database.py` resolves a unified `cache.db`. When a pre-existing per-module SQLite file (`summaries.db`, `relevance.db`, `semantic_scholar.db`, `huggingface.db`) is present, `resolve_db_path()` uses it; otherwise all cached data lives in `cache.db`.
- `Paper.source` is the local/API load origin. Cross-server identity lives separately in `Paper.provider` and `arxiv_browser.sources`, so arXiv-only workflows such as version checks, PDF downloads/previews, and arXiv HTML figure/full-text fetches reject DOI-based non-arXiv records before calling arXiv endpoints.

## LLM Provider Architecture

The LLM subsystem uses a `LLMProvider` protocol (`llm_providers.py`) with two implementations:

- **`CLIProvider`** — invokes an external CLI command (e.g., `claude`, `codex`, `llm`)
- **`HTTPProvider`** — targets OpenAI-compatible `/v1/chat/completions` endpoints (OpenAI, Ollama, LM Studio, vLLM)

Providers are registered in a simple registry (`register_provider` / `get_provider_class`). `resolve_provider(config)` reads `config.llm_provider_type` to select the appropriate implementation.

Streaming is deliberately separated from the large action module:

- `actions/llm_streaming.py` owns incremental summary prompt assembly and partial-summary UI updates.
- `modals/llm.py` owns incremental paper-chat updates.
- Providers expose streaming through `LLMProvider.execute_stream(...)`, yielding `LLMChunk` values; providers should not mutate app/UI state directly.
- Summary persistence still belongs to `actions/llm_actions.py` after a stream completes successfully, so partial output does not become durable cache state on failure.

## Design Principles

When extending the app, keep these durable principles in mind:

- Group related state (LLM, enrichment, recommendation) behind cohesive owners rather than adding more loose attributes to `ArxivBrowser`.
- Prefer service-layer request/result objects for cross-module workflows over new app-level mutable attributes.
- Keep `ArxivBrowser` as the orchestration shell for UI wiring, not the owner of every mutable concern.
