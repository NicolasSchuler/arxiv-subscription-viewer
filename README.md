# arXiv Subscription Viewer

A terminal user interface (TUI) for browsing arXiv papers from email subscription archives.

## Features

- Interactive split-pane interface with paper list and detail view
- Search by title, author, or category (use `cat:` prefix for category filtering)
- Multi-select papers for batch operations
- Open papers in browser or copy metadata to clipboard
- LaTeX cleaning for readable display
- Monokai color theme with category-specific highlighting
- Vim-style navigation (j/k) plus standard arrow keys
- Sort cycling between title, date, and arXiv ID

## Installation

Requires Python 3.13+

```bash
# Clone the repository
git clone https://github.com/nschuler/arxiv-subscription-viewer.git
cd arxiv-subscription-viewer

# Install with uv
uv sync
```

## Usage

```bash
# Run with default arxiv.txt in current directory
uv run python arxiv_browser.py

# Run with custom input file
uv run python arxiv_browser.py -i papers.txt
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Toggle search |
| `o` | Open selected paper(s) in browser |
| `c` | Copy selected paper(s) to clipboard |
| `Space` | Toggle paper selection |
| `a` | Select all visible papers |
| `u` | Clear all selections |
| `s` | Cycle sort order (title/date/arxiv_id) |
| `j`/`k` | Navigate down/up (vim-style) |
| `q` | Quit |

### Category Filtering

Filter by category using the `cat:` prefix in the search box:

- `cat:cs.AI` - Show only AI papers
- `cat:cs.LG` - Show only machine learning papers
- `cat:math` - Show papers with "math" in any category

## Input File Format

The application parses arXiv email subscription text files. Expected format:

- Papers separated by `------------------------------------------------------------------------------`
- Each paper contains: arXiv ID, date, title, authors, categories, abstract, URL

Example paper entry:

```
arXiv:2501.12345
Date: Mon, 20 Jan 2025 00:00:00 GMT   (15kb)

Title: Example Paper Title
Authors: Jane Doe, John Smith
Categories: cs.AI cs.LG
Abstract: This is the abstract text...
\\ ( https://arxiv.org/abs/2501.12345 , 15kb)
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v
```

## Author

Nicolas Sebastian Schuler (nicolas.schuler@kit.edu)

## License

MIT License - see [LICENSE](LICENSE) for details.
