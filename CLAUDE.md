# arXiv Subscription Viewer - Project Instructions

## Project Overview

A Textual-based TUI application for browsing arXiv papers from email subscription text files. Features include fuzzy search, paper management (read/star/notes/tags), session persistence, watch lists, bookmarks, and multiple export formats.

Published on PyPI as `arxiv-subscription-viewer`. Install with `pip install arxiv-subscription-viewer` or `uv tool install arxiv-subscription-viewer`.

## Architecture

### Package Layout (`src/arxiv_browser/`)

```
src/arxiv_browser/
├── __init__.py           # Re-exports public API from app.py
├── __main__.py           # python -m arxiv_browser support
├── action_messages.py    # Error/warning message builders
├── actions/              # Extracted action methods (mixed into ArxivBrowser)
│   ├── __init__.py       # Re-exports action mixins
│   ├── _runtime.py       # Runtime context for action methods
│   ├── external_io_actions.py  # Browser, PDF, clipboard, download actions
│   ├── library_actions.py      # Read/star/notes/tags/collections actions
│   ├── llm_actions.py          # Summary, relevance, auto-tag, chat actions
│   ├── search_api_actions.py   # arXiv API search actions
│   └── ui_actions.py           # Sort, theme, navigation, UI toggle actions
├── browser/              # Browser app mixins + compatibility helpers
│   ├── __init__.py       # Browser package exports from browser/core.py
│   ├── _runtime.py       # Shared browser runtime imports + compatibility syncing
│   ├── browse.py         # Dataset/filter state mixin
│   ├── chrome.py         # Detail-pane/footer/command-palette mixin
│   ├── core.py           # ArxivBrowser implementation + CLI bootstrap
│   └── discovery.py      # Similarity/recommendation/version-tracking mixin
├── app.py                # Compatibility shim + re-export bridge
├── cli.py                # CLI argument parsing + bootstrap
├── config.py             # Config persistence: load/save/export/import
├── enrichment.py         # Enrichment toggle helpers
├── export.py             # BibTeX, RIS, CSV, Markdown export
├── help_ui.py            # Help overlay section builder
├── huggingface.py        # HF Daily Papers API client, SQLite cache
├── io_actions.py         # I/O utilities (open URL, copy, viewer args)
├── llm.py                # LLM summary/relevance/auto-tag, SQLite cache
├── llm_providers.py      # LLM provider Protocol + CLI subprocess impl
├── models.py             # Dataclasses: Paper, PaperMetadata, UserConfig, etc.
├── parsing.py            # arXiv parsing, LaTeX cleaning, history
├── query.py              # Query tokenizer, matching, sorting, text utilities
├── semantic_scholar.py   # S2 API client, SQLite cache
├── services/             # Service layer (async I/O, external APIs)
│   ├── __init__.py       # Re-exports service functions
│   ├── arxiv_api_service.py    # arXiv API search + feed parsing
│   ├── download_service.py     # PDF download to local folder
│   ├── enrichment_service.py   # S2 + HF batch enrichment
│   ├── interfaces.py           # Service facade (unified API for browser/core.py and app.py shim)
│   └── llm_service.py          # LLM subprocess orchestration
├── similarity.py         # TF-IDF index, cosine + Jaccard similarity
├── themes.py             # Color palettes, category colors, Textual themes
├── ui_constants.py       # App CSS + keybinding definitions
├── ui_runtime.py         # Runtime UI state + refresh coordination
├── modals/               # ModalScreen subclasses, domain-grouped
│   ├── __init__.py       # Re-exports all modals for flat imports
│   ├── common.py         # HelpScreen, ConfirmModal, ExportMenuModal, MetadataSnapshotPickerModal, WatchListModal, SectionToggleModal
│   ├── editing.py        # NotesModal, TagsModal, AutoTagSuggestModal
│   ├── search.py         # ArxivSearchModal, CommandPaletteModal
│   ├── collections.py    # CollectionsModal, CollectionViewModal, AddToCollectionModal
│   ├── citations.py      # RecommendationSourceModal, RecommendationsScreen, CitationGraphScreen
│   └── llm.py            # SummaryModeModal, ResearchInterestsModal, PaperChatScreen
├── widgets/              # Reusable widgets extracted from app.py
│   ├── __init__.py       # Re-exports all widget classes/helpers/constants
│   ├── listing.py        # PaperListItem + render_paper_option list helpers
│   ├── details.py        # PaperDetails + detail pane cache/render helpers
│   └── chrome.py         # FilterPillBar, BookmarkTabBar, DateNavigator, ContextFooter
└── py.typed              # PEP 561 type marker
```

### Module Dependency DAG (no cycles)

```
models.py              ← 0 internal deps (leaf)
themes.py              ← 0 internal deps (leaf)
action_messages.py     ← 0 internal deps (leaf)
ui_constants.py        ← 0 internal deps (leaf)
help_ui.py             ← 0 internal deps (leaf)
config.py              ← models
parsing.py             ← models
export.py              ← models
query.py               ← models, themes
llm.py                 ← models
llm_providers.py       ← llm, models
similarity.py          ← models
semantic_scholar.py    ← models
huggingface.py         ← models
enrichment.py          ← models
io_actions.py          ← models
cli.py                 ← action_messages, config, models, parsing
services/arxiv_api_service.py  ← models, parsing
services/download_service.py   ← export, models
services/enrichment_service.py ← huggingface, semantic_scholar
services/llm_service.py        ← llm, llm_providers, models
services/interfaces.py ← llm_providers, models, services.*
modals/common.py       ← models, themes
modals/editing.py      ← themes
modals/search.py       ← models, parsing, query, themes
modals/collections.py  ← action_messages, models
modals/citations.py    ← action_messages, models, query, semantic_scholar, themes
modals/llm.py          ← llm, llm_providers, models, query, themes
widgets/listing.py     ← models, query, themes, semantic_scholar, huggingface
widgets/details.py     ← models, query, themes, semantic_scholar, huggingface, widgets/listing
widgets/chrome.py      ← models, parsing, query, themes
ui_runtime.py          ← widgets
actions/*              ← action_messages, actions/_runtime
app.py                 ← all above
```

Submodules should import canonical modules directly; the browser compatibility seam may import `app.py` when it needs to preserve the legacy patch surface. Modal, widget, action, and service submodules otherwise follow the same DAG constraint. `app.py` re-exports all public symbols from sub-modules via `from arxiv_browser.X import *` for backward compatibility. `modals/__init__.py`, `widgets/__init__.py`, and `services/__init__.py` provide flat imports for extracted classes.

### Data Models (`models.py`)

- `Paper` - Core paper data (arXiv ID, title, authors, etc.) with `__slots__`
- `PaperMetadata` - User annotations (notes, tags, read/star status, version tracking)
- `WatchListEntry` - Author/keyword/title patterns to highlight
- `SearchBookmark` - Saved search queries (max 9)
- `SessionState` - Scroll position, filters, sort order, current date
- `PaperCollection` - Named collection of papers (reading list) with description
- `UserConfig` - Complete user configuration container

### Core Functions (by module)

- **`parsing.py`**: `parse_arxiv_file()`, `clean_latex()`, `discover_history_files()`, `parse_arxiv_date()`
- **`config.py`**: `load_config()`, `save_config()`, `export_metadata()`, `import_metadata()`
- **`similarity.py`**: `find_similar_papers()`, `TfidfIndex`, `compute_paper_similarity()`
- **`llm.py`**: `build_llm_prompt()`, `get_summary_db_path()`, `SUMMARY_MODES`, `LLM_PRESETS`
- **`query.py`**: `tokenize_query()`, `sort_papers()`, `highlight_text()`, `format_categories()`
- **`export.py`**: `format_paper_as_bibtex()`, `format_papers_as_csv()`, `format_paper_as_ris()`
- **`themes.py`**: `get_tag_color()`, `parse_tag_namespace()`, `TEXTUAL_THEMES`

### UI Components (`widgets/` + `browser/` + `app.py`)

- `ArxivBrowser` (`browser/core.py`, re-exported via `app.py`) - Main Textual App class
- `PaperListItem` (`widgets/listing.py`) - Custom ListItem with selection/metadata display
- `PaperDetails` (`widgets/details.py`) - Rich-formatted paper detail view
- `BookmarkTabBar` (`widgets/chrome.py`) - Horizontal bookmark tabs widget
- `DateNavigator` (`widgets/chrome.py`) - Date navigation widget for history mode
- `FilterPillBar` (`widgets/chrome.py`) - Active search token pills
- `ContextFooter` (`widgets/chrome.py`) - Context-sensitive footer

### Modal Screens (`modals/`)

20 ModalScreen subclasses organized by domain:
- **common.py**: HelpScreen, ConfirmModal, ExportMenuModal, MetadataSnapshotPickerModal, WatchListModal, SectionToggleModal
- **editing.py**: NotesModal, TagsModal, AutoTagSuggestModal
- **search.py**: ArxivSearchModal, CommandPaletteModal
- **collections.py**: CollectionsModal, CollectionViewModal, AddToCollectionModal
- **citations.py**: RecommendationSourceModal, RecommendationsScreen, CitationGraphScreen
- **llm.py**: SummaryModeModal, ResearchInterestsModal, PaperChatScreen

### Performance Optimizations

- Pre-compiled regex patterns at module level (`_LATEX_PATTERNS`, `_ARXIV_ID_PATTERN`, etc.)
- `@lru_cache` for `format_categories()`
- O(1) paper lookup via `_papers_by_id` dictionary
- Pre-computed watch list matches (`_watched_paper_ids` set)
- Timer-based debouncing for search input (0.3s delay)
- Batch DOM updates in `_refresh_list_view()`
- History file discovery returns newest-first results; callers can pass an explicit limit when they need to cap the list

### External Modules

- **`semantic_scholar.py`**: S2 API client, `SemanticScholarPaper` / `CitationEntry` dataclasses, SQLite cache for papers, recommendations, and citation graphs
- **`huggingface.py`**: HuggingFace Daily Papers API client, `HuggingFacePaper` dataclass, SQLite cache

### Test Suite

Representative current coverage:

**Core and model coverage:**
- `test_arxiv_core_parsing.py`: parsing, LaTeX cleaning, exports, config, and query helpers
- `test_arxiv_cli_and_restore.py`: CLI bootstrap, history restore, and startup modes
- `test_config_import_and_load.py`: config import/load edge cases
- `test_arxiv_module_and_query_helpers.py`: helper utilities and public API re-exports
- `test_main_module.py`: `python -m arxiv_browser` entry point

**Services and external APIs:**
- `test_services_*.py`: arXiv API, download, enrichment, service facade, and LLM orchestration
- `test_semantic_scholar_*.py`: S2 parsing, cache, recommendations, and citation graph
- `test_huggingface.py`: HF Daily Papers parsing and cache
- `test_llm_*.py`: LLM providers, command guards, and integration

**UI and integration:**
- `test_app_actions_*.py`: app action coverage across bookmarks, collections, I/O, search, similarity, relevance, and auto-tag
- `test_modals_*.py`: modal screens
- `test_widgets_listing.py`: paper list rendering
- `test_tui_*.py`: TUI interaction and visual regression checks
- `test_arxiv_textual_browse_integration.py`: browse-mode end-to-end flows

**Tooling and packaging:**
- `test_check_docs_sync.py`, `test_code_quality_signatures.py`, `test_completions.py`, `test_packaging_smoke.py`, `test_benchmarks.py`

## Code Style

- Type hints on all functions and methods
- Dataclasses with `__slots__` for memory efficiency
- Pre-compile regex patterns at module level (not inside functions)
- Use `@lru_cache` for expensive repeated operations
- Constants in SCREAMING_SNAKE_CASE at module level
- `__all__` defines public API per module; `app.py` re-exports all for backward compat
- Module-level logger for debug output

## Key Patterns

### Textual Patterns
- **Widgets**: `ListView`, `ListItem`, `Static`, `Input`, `TextArea`, `Button`
- **CSS styling**: Inline `CSS` class variable with Monokai theme
- **Keyboard bindings**: `BINDINGS` class variable with `Binding` objects
- **Modal screens**: `ModalScreen[T]` for dialogs returning typed values
- **Async actions**: Some action methods are `async` for DOM updates

### Application Patterns
- **Timer debouncing**: Atomic swap pattern to avoid race conditions
- **Multi-select**: Track selection state in `selected_ids` set
- **Session restore**: Load/save via `on_mount()` / `on_unmount()`
- **Watch list**: Pre-compute matches at startup for O(1) lookup
- **History mode**: Date navigation with `_history_files` list

### Import Patterns (src layout)
- **Public API**: `from arxiv_browser import Paper, main` — via `__init__.py` → `app.py` re-exports
- **Canonical module import**: `from arxiv_browser.models import Paper` — preferred for new code
- **Modal imports**: `from arxiv_browser.modals import TagsModal` — via `modals/__init__.py` re-exports
- **Backward compat**: `from arxiv_browser.app import Paper` — still works via re-export bridge
- **Mock paths in tests**: Patch at the module where the function is *resolved*, not where it's re-exported:
  - Functions called by other functions in the same module: patch at the actual module (e.g., `"arxiv_browser.config.get_config_path"`)
  - Functions called by `ArxivBrowser`: patch the module where the symbol is resolved. Canonical implementations live under `arxiv_browser.browser.*`, but the compatibility shim still exposes many names through `arxiv_browser.app.X`.
  - Modal classes in tests: `from arxiv_browser.modals import X` (not `from arxiv_browser.app import X`)
  - S2/HF modules: `"arxiv_browser.semantic_scholar.X"`, `"arxiv_browser.huggingface.X"`

### Error Handling
- `NoMatches` exception handling for DOM queries during shutdown
- Type validation in config deserialization (`_safe_get()`)
- Graceful fallbacks for clipboard operations
- Duplicate paper detection in parser

## Dependencies

**Runtime:**
- **textual** (>=7.3.0): TUI framework
- **rapidfuzz** (>=3.0.0): Fuzzy string matching for search
- **httpx** (>=0.27.0): Async HTTP client for PDF downloads
- **platformdirs** (>=3.0): Cross-platform config directories

**Development:**
- **pytest** (>=9.0.2): Test framework
- **pytest-cov**: Coverage measurement
- **pyright**: Static type checking
- **ruff**: Linting and formatting
- **deptry**: Dependency hygiene (unused/missing/transitive deps)
- **vulture**: Dead code detection
- **bandit**: Security scanning
- **radon/xenon**: Cyclomatic complexity analysis
- **diff-cover**: Coverage enforcement on changed lines
- **mutmut**: Mutation testing (targeted)

## Testing

```bash
# Run all tests with coverage
just test

# Quick test (no coverage, stop on first failure)
just test-quick

# Run specific test class
uv run pytest -v tests/test_arxiv_core_parsing.py::TestCleanLatex

# Run tests matching pattern
uv run pytest -k "bibtex"
```

All tests must pass before commits.

## Code Quality Checks

All tool configurations live in `pyproject.toml`. The vulture whitelist is in `vulture_whitelist.py`. The `justfile` provides unified commands for all quality checks — run `just` to list available recipes.

### Quick check (before committing)

```bash
just check              # lint + typecheck + tests with coverage
```

### Full suite

```bash
just quality            # all checks: lint, types, tests, complexity, security, dead-code, deps
```

### Quality report (comprehensive dashboard)

```bash
just report             # lines of code, coverage, complexity, maintainability, lint, types, security, dead code, deps
```

### Individual tools

```bash
just lint               # ruff check + format check
just format             # auto-fix lint and formatting issues
just typecheck          # pyright
just complexity         # radon cyclomatic complexity + maintainability index
just security           # bandit security scanner
just dead-code          # vulture dead code detection
just deps               # deptry dependency hygiene
just docs-check         # docs drift guardrail (CLI flags/presets/keybindings)
just ci                 # CI-equivalent checks locally
just clean              # remove build artifacts and caches
```

### Advanced checks (not in justfile — run manually)

```bash
# Complexity gate (matches CI)
uv run xenon src/arxiv_browser/ --max-absolute C --max-modules C --max-average B

# Coverage on changed lines only (80% threshold on diffs)
uv run pytest --cov --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=main --fail-under=80

# Mutation testing (targeted — full app.py would take hours)
uv run mutmut run --paths-to-mutate=src/arxiv_browser/semantic_scholar.py
uv run mutmut run --paths-to-mutate=src/arxiv_browser/huggingface.py
```

## Running the Application

```bash
uv run arxiv-viewer              # History mode: auto-loads newest from history/
uv run arxiv-viewer -i <file>    # Custom input file
uv run arxiv-viewer --debug      # Debug logging to ~/.config/arxiv-browser/debug.log
uv run arxiv-viewer --version    # Print version and exit
uv run arxiv-viewer doctor       # Check environment & configuration health
uv run arxiv-viewer config-path  # Print config file path
uv run python -m arxiv_browser   # Alternative: run as module
```

See `docs/history-mode.md` for history directory setup, date navigation, and all CLI flags.

## Configuration Storage

Config file location (via platformdirs):
- **Linux**: `~/.config/arxiv-browser/config.json`
- **macOS**: `~/Library/Application Support/arxiv-browser/config.json`
- **Windows**: `%APPDATA%/arxiv-browser/config.json`

BibTeX exports default to `~/arxiv-exports/` (configurable).

LLM summaries are cached in a SQLite database (`summaries.db`) in the same config directory. Summaries are invalidated when the LLM command or prompt template changes.

## Feature Configuration

Detailed setup and usage for each feature is in `docs/`:

- **LLM summaries, relevance scoring, auto-tag**: `docs/llm-setup.md` — presets, custom commands, prompt templates
- **Semantic Scholar**: `docs/semantic-scholar.md` — citation counts, TLDRs, recommendations, citation graph
- **HuggingFace trending**: `docs/huggingface.md` — community upvotes, keywords, auto-fetch
- **Export & PDF**: `docs/export.md` — BibTeX, RIS, CSV, Markdown, PDF download config
- **Search & filters**: `docs/search-filters.md` — query syntax, sort orders, filter pills

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):
- **CI job**: Runs on push to `main` and PRs — lint, format check, type check, tests with coverage plus an interactive-modules coverage guard on `actions/`, `browser/`, and `cli.py`
- **Quality job**: Runs after CI passes — xenon complexity (C/C/B), bandit security, vulture dead-code (80% confidence), deptry deps, a warning-only repo-wide Python file-size report (>1250 lines, near-cap >1000), and the hard `app.py` line-count guard (≤5000)
- **Publish job**: Runs on `v*` tags after CI passes — builds and publishes to PyPI via Trusted Publishers (OIDC)

### Release process

**Important:** Push the commit and tag separately. If you push both at once (`git push origin main --tags`), GitHub may coalesce them into a single branch-triggered run and skip the tag-triggered publish job.

```bash
uv version --bump minor           # bumps in pyproject.toml
uv sync                           # regenerates uv.lock
git add pyproject.toml uv.lock
git commit -m "release: v0.2.0"
git tag v0.2.0
git push origin main              # CI runs on branch push
git push origin v0.2.0            # CD publishes to PyPI on tag push
```

## Key Bindings Reference

| Key | Action |
|-----|--------|
| `/` | Toggle search (fuzzy matching) |
| `Escape` | Cancel search / exit API mode |
| `A` | Search all arXiv (API mode) |
| `o` | Open selected paper(s) in browser |
| `P` | Open selected paper(s) as PDF |
| `c` | Copy selected paper(s) to clipboard |
| `d` | Download PDF(s) to local folder |
| `E` | Export menu (BibTeX, Markdown, RIS, CSV + more) |
| `space` | Toggle selection |
| `a` | Select all visible |
| `u` | Clear selection |
| `s` | Cycle sort order (title/date/arxiv_id/citations/trending/relevance) |
| `j`/`k` | Navigate down/up (vim-style) |
| `r` | Toggle read status |
| `x` | Toggle star |
| `n` | Edit notes |
| `t` | Edit tags (namespace:value, e.g. topic:ml, status:to-read) |
| `w` | Toggle watch list filter |
| `W` | Manage watch list |
| `p` | Toggle abstract preview |
| `m` | Set mark (then press a-z) |
| `'` | Jump to mark (then press a-z) |
| `R` | Show similar papers (local or Semantic Scholar) |
| `G` | Citation graph (S2-powered, drill-down navigation) |
| `V` | Check starred papers for version updates |
| `e` | Fetch Semantic Scholar data for current paper |
| `Ctrl+e` | Exit API mode (in API mode) / toggle S2 enrichment (otherwise) |
| `Ctrl+h` | Toggle HuggingFace trending on/off |
| `Ctrl+s` | Generate AI summary (mode selector: default/TLDR/methods/results/comparison) |
| `C` | Chat with current paper (LLM-powered) |
| `Ctrl+g` | Auto-tag current/selected papers (LLM-powered) |
| `L` | Score papers by relevance (LLM-powered) |
| `Ctrl+l` | Edit research interests |
| `1-9` | Jump to bookmark |
| `Ctrl+b` | Add current search as bookmark |
| `Ctrl+Shift+b` | Remove active bookmark |
| `Ctrl+t` | Cycle color theme (Monokai / Catppuccin / Solarized / High Contrast) |
| `Ctrl+d` | Toggle detail pane sections |
| `Ctrl+k` | Manage paper collections (reading lists) |
| `Ctrl+p` | Open command palette |
| `[` | Go to previous (older) date (history mode) |
| `]` | Go to next (newer) date (history mode) |
| `?` | Show help overlay |
| `q` | Quit |
