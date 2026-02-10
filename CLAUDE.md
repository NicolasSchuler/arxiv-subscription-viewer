# arXiv Subscription Viewer - Project Instructions

## Project Overview

A Textual-based TUI application for browsing arXiv papers from email subscription text files. Features include fuzzy search, paper management (read/star/notes/tags), session persistence, watch lists, bookmarks, and multiple export formats.

## Architecture

### Main Application (`arxiv_browser.py` ~8300 lines)

**Data Models:**
- `Paper` - Core paper data (arXiv ID, title, authors, etc.) with `__slots__`
- `PaperMetadata` - User annotations (notes, tags, read/star status, version tracking)
- `WatchListEntry` - Author/keyword/title patterns to highlight
- `SearchBookmark` - Saved search queries (max 9)
- `SessionState` - Scroll position, filters, sort order, current date
- `UserConfig` - Complete user configuration container

**Core Functions:**
- `parse_arxiv_file()` - Parse arXiv email text files
- `clean_latex()` - Remove LaTeX commands for readable display
- `discover_history_files()` - Find YYYY-MM-DD.txt files in history/
- `load_config()` / `save_config()` - JSON persistence via platformdirs
- `find_similar_papers()` - Hybrid TF-IDF cosine + Jaccard similarity (text 50%, categories 30%, authors 20%)
- `build_llm_prompt()` - Build LLM prompt from paper data and template
- `get_summary_db_path()` - Path to SQLite summary cache

**UI Components:**
- `ArxivBrowser` - Main Textual App class
- `PaperListItem` - Custom ListItem with selection/metadata display
- `PaperDetails` - Rich-formatted paper detail view
- `NotesModal` / `TagsModal` - ModalScreen dialogs for editing
- `RecommendationsScreen` - Similar papers modal
- `CitationGraphScreen` - Citation graph drill-down modal
- `BookmarkTabBar` - Horizontal bookmark tabs widget

**Performance Optimizations:**
- Pre-compiled regex patterns at module level (`_LATEX_PATTERNS`, `_ARXIV_ID_PATTERN`, etc.)
- `@lru_cache` for `format_categories()`
- O(1) paper lookup via `_papers_by_id` dictionary
- Pre-computed watch list matches (`_watched_paper_ids` set)
- Timer-based debouncing for search input (0.3s delay)
- Batch DOM updates in `_refresh_list_view()`
- History file discovery limited to 365 files

### Supporting Modules

- **`semantic_scholar.py`** (~630 lines): S2 API client, `SemanticScholarPaper` / `CitationEntry` dataclasses, SQLite cache for papers, recommendations, and citation graphs
- **`huggingface.py`** (~300 lines): HuggingFace Daily Papers API client, `HuggingFacePaper` dataclass, SQLite cache

### Test Suite (~620 tests across 3 files)

- **`test_arxiv_browser.py`** (~5800 lines): Core parsing, similarity, export, config, UI integration
- **`test_semantic_scholar.py`** (~990 lines): S2 response parsing, serialization, cache CRUD, API fetch functions, citation graph
- **`test_huggingface.py`** (~460 lines): HF response parsing, cache, API fetch functions

## Code Style

- Type hints on all functions and methods
- Dataclasses with `__slots__` for memory efficiency
- Pre-compile regex patterns at module level (not inside functions)
- Use `@lru_cache` for expensive repeated operations
- Constants in SCREAMING_SNAKE_CASE at module level
- `__all__` defines public API (58 exports)
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
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov --cov-report=term-missing

# Run specific test class
uv run pytest -v test_arxiv_browser.py::TestCleanLatex

# Run tests matching pattern
uv run pytest -k "bibtex"
```

All tests must pass before commits.

## Code Quality Checks

All tool configurations live in `pyproject.toml`. The vulture whitelist is in `vulture_whitelist.py`.

### Quick check (before committing)

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest --cov --cov-report=term-missing
```

### Full suite

```bash
# Lint + format
uv run ruff check .
uv run ruff format --check .

# Type checking (basic mode — catches real errors without Textual framework noise)
uv run pyright

# Tests with coverage (fail_under=60, ratchet up over time)
uv run pytest --cov --cov-report=term-missing --cov-report=html

# Dependency hygiene — detects unused, missing, and transitive deps
uv run deptry .

# Dead code detection (vulture_whitelist.py suppresses Textual framework false positives)
uv run vulture arxiv_browser.py semantic_scholar.py huggingface.py vulture_whitelist.py --min-confidence 80

# Security scanning (B101/B311/B314/B404/B405 skipped in config)
uv run bandit -c pyproject.toml -r arxiv_browser.py semantic_scholar.py huggingface.py

# Complexity (show C+ rated functions, plus average)
uv run radon cc arxiv_browser.py semantic_scholar.py huggingface.py -a -nc

# Complexity gate (max-absolute E baseline — update_paper is E-ranked; ratchet down)
uv run xenon arxiv_browser.py semantic_scholar.py huggingface.py --max-absolute E --max-modules D --max-average B

# Coverage on changed lines only (80% threshold on diffs)
uv run pytest --cov --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=main --fail-under=80

# Mutation testing (targeted — full arxiv_browser.py would take hours)
uv run mutmut run --paths-to-mutate=semantic_scholar.py
uv run mutmut run --paths-to-mutate=huggingface.py
```

## Running the Application

```bash
# History mode: auto-loads newest file from history/
uv run python arxiv_browser.py

# List available dates
uv run python arxiv_browser.py --list-dates

# Open specific date
uv run python arxiv_browser.py --date 2026-01-23

# Custom input file (disables history mode)
uv run python arxiv_browser.py -i <file>

# Start fresh session (no restore)
uv run python arxiv_browser.py --no-restore
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

## Key Bindings Reference

```
/       - Toggle search (fuzzy matching)
A       - Search all arXiv (API mode)
o       - Open selected paper(s) in browser
P       - Open selected paper(s) as PDF
c       - Copy selected paper(s) to clipboard
b       - Copy as BibTeX
B       - Export BibTeX to file (for Zotero import)
d       - Download PDF(s) to local folder
M       - Copy as Markdown
E       - Export menu (RIS, CSV, Markdown table + more)
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
[       - Go to previous (older) date (history mode)
]       - Go to next (newer) date (history mode)
q       - Quit
```
