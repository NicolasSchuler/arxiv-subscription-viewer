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
├── app.py                # ArxivBrowser app + CLI entrypoint/glue (~4800 lines)
├── models.py             # Dataclasses: Paper, PaperMetadata, UserConfig, etc. (~330 lines)
├── config.py             # Config persistence: load/save/export/import (~540 lines)
├── parsing.py            # arXiv parsing, LaTeX cleaning, history (~570 lines)
├── export.py             # BibTeX, RIS, CSV, Markdown export (~330 lines)
├── query.py              # Query tokenizer, matching, sorting, text utils (~480 lines)
├── llm.py                # LLM summary/relevance/auto-tag, SQLite cache (~540 lines)
├── llm_providers.py      # LLM provider Protocol + CLI subprocess implementation (~110 lines)
├── similarity.py         # TF-IDF index, cosine + Jaccard similarity (~320 lines)
├── themes.py             # Color palettes, category colors, Textual themes (~270 lines)
├── semantic_scholar.py   # S2 API client, SQLite cache (~820 lines)
├── huggingface.py        # HF Daily Papers API client, SQLite cache (~350 lines)
├── modals/               # ModalScreen subclasses, domain-grouped (~2700 lines)
│   ├── __init__.py       # Re-exports all modals for flat imports
│   ├── common.py         # HelpScreen, ConfirmModal, ExportMenuModal, WatchListModal, SectionToggleModal
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
config.py              ← models
parsing.py             ← models
export.py              ← models
query.py               ← models, themes
llm.py                 ← models
llm_providers.py       ← llm, models
similarity.py          ← models
semantic_scholar.py    ← models
huggingface.py         ← models
modals/common.py       ← models, themes
modals/editing.py      ← themes
modals/search.py       ← models, parsing, query, themes
modals/collections.py  ← models
modals/citations.py    ← models, query, semantic_scholar, themes
modals/llm.py          ← llm, llm_providers, models, query, themes
widgets/listing.py     ← models, query, themes, semantic_scholar, huggingface
widgets/details.py     ← models, query, themes, semantic_scholar, huggingface, widgets/listing
widgets/chrome.py      ← models, parsing, query, themes
app.py                 ← all above
```

No module imports from `app.py` — this prevents circular dependencies. Modal and widget submodules follow the same DAG constraint. `app.py` re-exports all public symbols from sub-modules via `from arxiv_browser.X import *` for backward compatibility. `modals/__init__.py` and `widgets/__init__.py` provide flat imports for extracted UI classes.

### Data Models (`models.py`)

- `Paper` - Core paper data (arXiv ID, title, authors, etc.) with `__slots__`
- `PaperMetadata` - User annotations (notes, tags, read/star status, version tracking)
- `WatchListEntry` - Author/keyword/title patterns to highlight
- `SearchBookmark` - Saved search queries (max 9)
- `SessionState` - Scroll position, filters, sort order, current date
- `UserConfig` - Complete user configuration container

### Core Functions (by module)

- **`parsing.py`**: `parse_arxiv_file()`, `clean_latex()`, `discover_history_files()`, `parse_arxiv_date()`
- **`config.py`**: `load_config()`, `save_config()`, `export_metadata()`, `import_metadata()`
- **`similarity.py`**: `find_similar_papers()`, `TfidfIndex`, `compute_paper_similarity()`
- **`llm.py`**: `build_llm_prompt()`, `get_summary_db_path()`, `SUMMARY_MODES`, `LLM_PRESETS`
- **`query.py`**: `tokenize_query()`, `sort_papers()`, `highlight_text()`, `format_categories()`
- **`export.py`**: `format_paper_as_bibtex()`, `format_papers_as_csv()`, `format_paper_as_ris()`
- **`themes.py`**: `get_tag_color()`, `parse_tag_namespace()`, `TEXTUAL_THEMES`

### UI Components (`widgets/` + `app.py`)

- `ArxivBrowser` (`app.py`) - Main Textual App class
- `PaperListItem` (`widgets/listing.py`) - Custom ListItem with selection/metadata display
- `PaperDetails` (`widgets/details.py`) - Rich-formatted paper detail view
- `BookmarkTabBar` (`widgets/chrome.py`) - Horizontal bookmark tabs widget
- `DateNavigator` (`widgets/chrome.py`) - Date navigation widget for history mode
- `FilterPillBar` (`widgets/chrome.py`) - Active search token pills
- `ContextFooter` (`widgets/chrome.py`) - Context-sensitive footer

### Modal Screens (`modals/`)

19 ModalScreen subclasses organized by domain:
- **common.py**: HelpScreen, ConfirmModal, ExportMenuModal, WatchListModal, SectionToggleModal
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
- History file discovery limited to 365 files

### External Modules

- **`semantic_scholar.py`** (~820 lines): S2 API client, `SemanticScholarPaper` / `CitationEntry` dataclasses, SQLite cache for papers, recommendations, and citation graphs
- **`huggingface.py`** (~350 lines): HuggingFace Daily Papers API client, `HuggingFacePaper` dataclass, SQLite cache

### Test Suite (~1147 tests across 6 files + conftest.py in `tests/`)

- **`tests/test_arxiv_browser.py`** (~5800 lines): Core parsing, similarity, export, config, UI integration, WCAG contrast
- **`tests/test_integration.py`** (~400 lines): End-to-end workflows with real arXiv email fixtures, export validation, resource cleanup, debug logging
- **`tests/test_semantic_scholar.py`** (~990 lines): S2 response parsing, serialization, cache CRUD, API fetch functions, citation graph
- **`tests/test_huggingface.py`** (~460 lines): HF response parsing, cache, API fetch functions
- **`tests/test_llm_providers.py`** (~160 lines): LLMProvider protocol, CLIProvider subprocess wrapper, resolve_provider factory
- **`tests/test_benchmarks.py`** (~170 lines): Performance regression tests (marked `@pytest.mark.slow`, excluded from default runs)

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
  - Functions called by `ArxivBrowser` (resolves from `app.py`): patch at `"arxiv_browser.app.X"`
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
- **platformdirs**: Cross-platform config directories (transitive)

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
uv run pytest -v tests/test_arxiv_browser.py::TestCleanLatex

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
just ci                 # CI-equivalent checks locally
just clean              # remove build artifacts and caches
```

### Advanced checks (not in justfile — run manually)

```bash
# Complexity gate (max-absolute F baseline — import_metadata is F-ranked; ratchet down)
uv run xenon src/arxiv_browser/ --max-absolute F --max-modules D --max-average B

# Coverage on changed lines only (80% threshold on diffs)
uv run pytest --cov --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=main --fail-under=80

# Mutation testing (targeted — full app.py would take hours)
uv run mutmut run --paths-to-mutate=src/arxiv_browser/semantic_scholar.py
uv run mutmut run --paths-to-mutate=src/arxiv_browser/huggingface.py
```

## Running the Application

```bash
# History mode: auto-loads newest file from history/
uv run arxiv-viewer

# List available dates
uv run arxiv-viewer --list-dates

# Open specific date
uv run arxiv-viewer --date 2026-01-23

# Custom input file (disables history mode)
uv run arxiv-viewer -i <file>

# Start fresh session (no restore)
uv run arxiv-viewer --no-restore

# Enable debug logging (writes to ~/.config/arxiv-browser/debug.log)
uv run arxiv-viewer --debug

# Alternative: run as module
uv run python -m arxiv_browser
```

## History Mode

Store arXiv emails in `history/` directory with `YYYY-MM-DD.txt` filenames:

```
history/
├── 2026-01-20.txt
├── 2026-01-21.txt
└── 2026-01-23.txt
```

- App auto-discovers and loads newest file
- Use `[` and `]` keys to navigate between dates
- Falls back to `arxiv.txt` if no history directory exists
- Limited to 365 most recent files (`MAX_HISTORY_FILES`)

## Configuration Storage

Config file location (via platformdirs):
- **Linux**: `~/.config/arxiv-browser/config.json`
- **macOS**: `~/Library/Application Support/arxiv-browser/config.json`
- **Windows**: `%APPDATA%/arxiv-browser/config.json`

BibTeX exports default to `~/arxiv-exports/` (configurable).

LLM summaries are cached in a SQLite database (`summaries.db`) in the same config directory. Summaries are invalidated when the LLM command or prompt template changes.

## LLM Summary Configuration

Generate AI-powered paper summaries using any CLI tool. The default prompt produces accessible, explanatory summaries aimed at CS students (Problem / Approach / Results / Limitations / Key Takeaway). The full paper content is automatically fetched from the arXiv HTML version and passed to the LLM.

Add to `config.json`:

```json
{
  "llm_preset": "copilot"
}
```

Available presets: `claude`, `codex`, `llm`, `copilot`. Or use a custom command:

```json
{
  "llm_command": "claude -p {prompt}",
  "llm_prompt_template": "Summarize in 3 sentences: {title}\n\n{paper_content}"
}
```

Prompt template placeholders: `{title}`, `{authors}`, `{categories}`, `{abstract}`, `{arxiv_id}`, `{paper_content}`.

The `{paper_content}` placeholder is replaced with the full paper text (fetched from arXiv HTML), falling back to the abstract if unavailable.

## Semantic Scholar Integration

Optional enrichment with citation counts, fields of study, TLDRs, and S2-powered recommendations. Fully opt-in and on-demand.

Add to `config.json`:

```json
{
  "s2_enabled": true,
  "s2_api_key": "",
  "s2_cache_ttl_days": 7
}
```

- `s2_enabled`: Enable S2 enrichment on startup (default: `false`)
- `s2_api_key`: Optional API key for higher rate limits
- `s2_cache_ttl_days`: Days to cache S2 data in SQLite (default: `7`)

Usage: Press `Ctrl+e` to toggle S2 on/off, then `e` on any paper to fetch citation data. Press `R` for recommendations (local or S2-powered when enabled). Press `G` for citation graph exploration (drill-down through references and citations). Sort by citations with `s`.

## HuggingFace Trending Integration

Optional trending signal from HuggingFace Daily Papers — shows community upvotes, comments, GitHub info, AI summaries, and keywords. Fully opt-in with auto-fetch.

Add to `config.json`:

```json
{
  "hf_enabled": true,
  "hf_cache_ttl_hours": 6
}
```

- `hf_enabled`: Enable HF trending on startup (default: `false`)
- `hf_cache_ttl_hours`: Hours to cache HF daily data in SQLite (default: `6`)

Usage: Press `Ctrl+h` to toggle HF on/off. When enabled, the daily papers list is auto-fetched and cross-matched against loaded papers. Trending papers show upvote badges in the list and a HuggingFace section in the detail pane. Sort by trending with `s`.

## Relevance Scoring

LLM-powered relevance scoring lets you define research interests and have the configured LLM score each paper 1-10. Requires an LLM preset or command configured (same as AI summaries). Scores are cached in SQLite (`relevance.db`) keyed by `(arxiv_id, interests_hash)`.

Add to `config.json`:

```json
{
  "research_interests": "efficient LLM inference, quantization, speculative decoding"
}
```

Usage: Press `L` to score all loaded papers (prompts for interests on first use). Press `Ctrl+l` to edit interests (clears cached scores). Sort by relevance with `s`. Papers show colored score badges: green (8-10), yellow (5-7), dim (1-4).

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):
- **CI job**: Runs on push to `main` and PRs — lint, format check, type check, tests with coverage
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

```
/       - Toggle search (fuzzy matching)
A       - Search all arXiv (API mode)
o       - Open selected paper(s) in browser
P       - Open selected paper(s) as PDF
c       - Copy selected paper(s) to clipboard
d       - Download PDF(s) to local folder
E       - Export menu (BibTeX, Markdown, RIS, CSV + more)
space   - Toggle selection
a       - Select all visible
u       - Clear selection
s       - Cycle sort order (title/date/arxiv_id/citations/trending/relevance)
j/k     - Navigate down/up (vim-style)
r       - Toggle read status
x       - Toggle star
n       - Edit notes
t       - Edit tags (namespace:value supported, e.g., topic:ml, status:to-read)
w       - Toggle watch list filter
W       - Manage watch list
p       - Toggle abstract preview
m       - Set mark (then press a-z)
'       - Jump to mark (then press a-z)
R       - Show similar papers (local or Semantic Scholar)
G       - Citation graph (S2-powered, drill-down navigation)
V       - Check starred papers for version updates
e       - Fetch Semantic Scholar data for current paper
Ctrl+e  - Toggle Semantic Scholar enrichment on/off
Ctrl+h  - Toggle HuggingFace trending on/off
Ctrl+s  - Generate AI summary (mode selector: default/TLDR/methods/results/comparison)
L       - Score papers by relevance (LLM-powered)
Ctrl+l  - Edit research interests
1-9     - Jump to bookmark
Ctrl+b  - Add current search as bookmark
Ctrl+t  - Cycle color theme (Monokai / Catppuccin / Solarized)
Ctrl+d  - Toggle detail pane sections
Ctrl+k  - Manage paper collections (reading lists)
[       - Go to previous (older) date (history mode)
]       - Go to next (newer) date (history mode)
?       - Show help overlay
q       - Quit
```
