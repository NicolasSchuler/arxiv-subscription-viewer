# arXiv Subscription Viewer - Project Instructions

## Project Overview

A Textual-based TUI application for browsing arXiv papers from email subscription text files. Features include fuzzy search, paper management (read/star/notes/tags), session persistence, watch lists, bookmarks, and multiple export formats.

## Architecture

### Main Application (`arxiv_browser.py` ~3860 lines)

**Data Models:**
- `Paper` - Core paper data (arXiv ID, title, authors, etc.) with `__slots__`
- `PaperMetadata` - User annotations (notes, tags, read/star status)
- `WatchListEntry` - Author/keyword/title patterns to highlight
- `SearchBookmark` - Saved search queries (max 9)
- `SessionState` - Scroll position, filters, sort order, current date
- `UserConfig` - Complete user configuration container

**Core Functions:**
- `parse_arxiv_file()` - Parse arXiv email text files
- `clean_latex()` - Remove LaTeX commands for readable display
- `discover_history_files()` - Find YYYY-MM-DD.txt files in history/
- `load_config()` / `save_config()` - JSON persistence via platformdirs
- `find_similar_papers()` - Jaccard similarity on categories/authors/keywords
- `build_llm_prompt()` - Build LLM prompt from paper data and template
- `get_summary_db_path()` - Path to SQLite summary cache

**UI Components:**
- `ArxivBrowser` - Main Textual App class
- `PaperListItem` - Custom ListItem with selection/metadata display
- `PaperDetails` - Rich-formatted paper detail view
- `NotesModal` / `TagsModal` - ModalScreen dialogs for editing
- `RecommendationsScreen` - Similar papers modal
- `BookmarkTabBar` - Horizontal bookmark tabs widget

**Performance Optimizations:**
- Pre-compiled regex patterns at module level (`_LATEX_PATTERNS`, `_ARXIV_ID_PATTERN`, etc.)
- `@lru_cache` for `format_categories()`
- O(1) paper lookup via `_papers_by_id` dictionary
- Pre-computed watch list matches (`_watched_paper_ids` set)
- Timer-based debouncing for search input (0.3s delay)
- Batch DOM updates in `_refresh_list_view()`
- History file discovery limited to 365 files

### Test Suite (`test_arxiv_browser.py` ~1080 lines)

93 tests covering:
- LaTeX cleaning edge cases (nested commands, math mode, accents)
- Date parsing and sorting
- Category formatting and caching
- File parsing (valid, empty, malformed inputs)
- Paper dataclass behavior
- Configuration serialization roundtrip
- Paper similarity algorithm (Jaccard, keyword extraction)
- Fuzzy search constants
- Text truncation utility
- Type-safe config parsing (`_safe_get`)
- Paper deduplication
- History file discovery and limits
- BibTeX export edge cases
- Module exports (`__all__`)

## Code Style

- Type hints on all functions and methods
- Dataclasses with `__slots__` for memory efficiency
- Pre-compile regex patterns at module level (not inside functions)
- Use `@lru_cache` for expensive repeated operations
- Constants in SCREAMING_SNAKE_CASE at module level
- `__all__` defines public API (23 exports)
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

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test class
uv run pytest -v test_arxiv_browser.py::TestCleanLatex

# Run tests matching pattern
uv run pytest -k "bibtex"
```

All tests must pass before commits.

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

## Key Bindings Reference

```
/       - Toggle search (fuzzy matching)
o       - Open selected paper(s) in browser
c       - Copy selected paper(s) to clipboard
b       - Copy as BibTeX
B       - Export BibTeX to file (for Zotero import)
d       - Download PDF(s) to local folder
M       - Copy as Markdown
space   - Toggle selection
a       - Select all visible
u       - Clear selection
s       - Cycle sort order (title/date/arxiv_id)
j/k     - Navigate down/up (vim-style)
r       - Toggle read status
x       - Toggle star
n       - Edit notes
t       - Edit tags
w       - Toggle watch list filter
p       - Toggle abstract preview
m       - Set mark (then press a-z)
'       - Jump to mark (then press a-z)
R       - Show similar papers
Ctrl+s  - Generate AI summary (requires LLM CLI tool)
1-9     - Jump to bookmark
Ctrl+b  - Add current search as bookmark
[       - Go to previous (older) date (history mode)
]       - Go to next (newer) date (history mode)
q       - Quit
```
