# arXiv Subscription Viewer - Project Instructions

## Project Overview

A Textual-based TUI application for browsing arXiv papers from email subscription text files.

## Architecture

- **arxiv_browser.py**: Main application (~870 lines)
  - `Paper` dataclass with `__slots__` for memory efficiency
  - Pre-compiled regex patterns for LaTeX cleaning at module level
  - `@lru_cache` for category formatting performance
  - O(1) paper lookup via `_paper_lookup` dictionary
  - Timer-based debouncing for search input

- **test_arxiv_browser.py**: Pytest test suite (~315 lines)
  - Tests for LaTeX cleaning, category formatting, file parsing
  - Tests for Paper dataclass behavior and module constants

## Code Style

- Type hints throughout all functions and methods
- Dataclasses with `__slots__` for data structures
- Pre-compile regex patterns at module level (not inside functions)
- Use `@lru_cache` for expensive repeated operations
- Constants defined at module level in SCREAMING_SNAKE_CASE

## Testing

```bash
uv run pytest
```

All tests must pass before commits.

## Key Patterns

- **Textual widgets**: `ListView`, `ListItem`, `Static`, `Input`
- **CSS styling**: Defined in `DEFAULT_CSS` class variable
- **Keyboard bindings**: Defined via `BINDINGS` class variable
- **Timer-based debouncing**: For search input filtering
- **Multi-select**: Track selection state in `Paper.selected` attribute

## Dependencies

- **textual**: TUI framework
- **rich**: Terminal formatting (used by textual)
- **pytest**: Testing (dev dependency)

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
