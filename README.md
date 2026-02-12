<p align="center">
  <img src="docs/logo.png" alt="arXiv Subscription Viewer" width="200">
</p>

<h1 align="center">arXiv Subscription Viewer</h1>

<p align="center">
  A terminal user interface (TUI) for browsing arXiv papers from email subscription archives.
</p>

![arXiv Subscription Viewer Screenshot](screenshot.svg)

## Features

### Core Browsing
- Interactive split-pane interface with paper list and detail view
- Fuzzy search by title and author (powered by RapidFuzz)
- Filter by category (`cat:cs.AI`), tag (`tag:important`), `unread`, or `starred`
- Search all arXiv via API (field + optional category) with paginated results
- Multi-select papers for batch operations
- Sort cycling between title, date, arXiv ID, citations, trending, and relevance
- Vim-style navigation (j/k) plus standard arrow keys

### Paper Management
- Mark papers as read/unread with persistent tracking
- Star important papers for quick access
- Add custom notes to any paper
- Tag papers with custom labels (e.g., `to-read`, `important`, `llm`)
- Watch list for highlighting papers by author/keyword/title

### Export Features
- Open papers in browser (single or batch)
- Copy paper metadata to clipboard
- Export as BibTeX (clipboard or file for Zotero import)
- Export as Markdown
- Download PDFs to local folder (async batch downloads)

### Productivity
- Session restore (scroll position, filters, selections persist across runs)
- Search bookmarks (save up to 9 frequent searches, access via 1-9 keys)
- Vim-style marks (set marks with `m` + letter, jump with `'` + letter)
- Similar paper recommendations based on category, author, and content
- Semantic Scholar integration: citation counts, TLDR summaries, S2-powered recommendations, and citation graph exploration
- HuggingFace trending: community upvotes, AI summaries, keywords, and GitHub info from HF Daily Papers
- Paper version tracking: check starred papers for arXiv revisions with diff links
- LLM-powered relevance scoring: score papers 1-10 based on your research interests
- Abstract preview toggle in list view

### History Mode
- Auto-discover and load arXiv emails from `history/` directory
- Navigate between dates with `[` and `]` keys
- Date-aware session restore

### Visual
- Three color themes: Monokai (default), Catppuccin, Solarized Dark — cycle with `Ctrl+t`
- Category-specific highlighting and watch list highlighting
- LaTeX cleaning for readable display
- Collapsible detail pane sections (`Ctrl+d`)

## Installation

Requires Python 3.13+

### From PyPI (recommended)

```bash
# Install globally with uv
uv tool install arxiv-subscription-viewer

# Or with pip
pip install arxiv-subscription-viewer
```

### From source (development)

```bash
# Clone the repository
git clone https://github.com/NicolasSchuler/arxiv-subscription-viewer.git
cd arxiv-subscription-viewer

# Install with uv
uv sync
```

## Quick Start

```bash
# 1. Install
pip install arxiv-subscription-viewer

# 2. Place an arXiv email file in history/ (or use -i)
mkdir -p history
# Save your arXiv email as history/2026-02-12.txt

# 3. Run
arxiv-viewer
```

## Usage

```bash
# History mode: auto-loads newest file from history/
arxiv-viewer

# Show help
arxiv-viewer --help

# List available dates in history
arxiv-viewer --list-dates

# Open specific date
arxiv-viewer --date 2026-01-23

# Custom input file (disables history mode)
arxiv-viewer -i papers.txt

# Start fresh session (ignore saved state)
arxiv-viewer --no-restore

# Alternative: run as module (useful during development)
uv run python -m arxiv_browser
```

## Keyboard Shortcuts

### Navigation & Search
| Key | Action |
|-----|--------|
| `/` | Toggle search input |
| `Escape` | Cancel search / exit API mode |
| `A` | Search all arXiv (API mode) |
| `Ctrl+e` | Exit API mode |
| `j`/`k` | Navigate down/up (vim-style) |
| `1-9` | Jump to search bookmark |
| `Ctrl+b` | Add current search as bookmark |
| `[` | Previous date (history) / previous API page (API mode) |
| `]` | Next date (history) / next API page (API mode) |

### Selection & Actions
| Key | Action |
|-----|--------|
| `Space` | Toggle paper selection |
| `a` | Select all visible papers |
| `u` | Clear all selections |
| `o` | Open selected paper(s) in browser |
| `P` | Open selected paper(s) as PDF |
| `c` | Copy selected paper(s) to clipboard |
| `s` | Cycle sort order (title/date/arxiv_id/citations/trending/relevance) |

### Paper Status
| Key | Action |
|-----|--------|
| `r` | Toggle read status |
| `x` | Toggle star |
| `n` | Edit notes |
| `t` | Edit tags |
| `w` | Toggle watch list filter |
| `W` | Manage watch list |
| `p` | Toggle abstract preview |

### Export & Download
| Key | Action |
|-----|--------|
| `E` | Export menu (BibTeX, Markdown, RIS, CSV + clipboard/file) |
| `d` | Download PDF(s) to local folder |

### Marks & Enrichment
| Key | Action |
|-----|--------|
| `m` | Set mark (then press a-z) |
| `'` | Jump to mark (then press a-z) |
| `R` | Show similar papers (local or S2-powered) |
| `G` | Explore citation graph (S2-powered, drill-down) |
| `V` | Check starred papers for version updates |
| `e` | Fetch Semantic Scholar data for current paper |
| `Ctrl+e` | Toggle Semantic Scholar enrichment on/off |
| `Ctrl+s` | Generate AI summary (mode selector) |
| `Ctrl+h` | Toggle HuggingFace trending on/off |
| `L` | Score papers by relevance (LLM-powered) |
| `Ctrl+l` | Edit research interests |

### General
| Key | Action |
|-----|--------|
| `Ctrl+t` | Cycle color theme (Monokai / Catppuccin / Solarized) |
| `Ctrl+d` | Toggle detail pane sections |
| `?` | Show help overlay |
| `q` | Quit |

## Search Filters

Use these prefixes in the search box:

| Filter | Example | Description |
|--------|---------|-------------|
| `cat:` | `cat:cs.AI` | Filter by category |
| `tag:` | `tag:to-read` | Filter by custom tag |
| `author:` | `author:hinton` | Filter by author name |
| `title:` | `title:transformer` | Filter by title substring |
| `abstract:` | `abstract:attention` | Filter by abstract substring |
| `unread` | `unread` | Show only unread papers |
| `starred` | `starred` | Show only starred papers |
| (text) | `transformer` | Fuzzy search title/authors |
| `"..."` | `"large language"` | Match exact phrase |

Combine terms with boolean operators: `cat:cs.AI AND author:hinton`, `unread OR starred`, `NOT cat:math`.

## History Mode

Store arXiv emails in the `history/` directory with `YYYY-MM-DD.txt` filenames:

```
history/
├── 2026-01-20.txt
├── 2026-01-21.txt
└── 2026-01-23.txt
```

- App auto-discovers and loads the newest file on startup
- Use `[` and `]` keys to navigate between dates
- Session state (including current date) persists across runs
- Falls back to `arxiv.txt` if no history directory exists

## PDF Downloads

Press `d` to download PDFs for selected papers (or current paper) to your local machine:

- Default location: `~/arxiv-pdfs/`
- Configure custom directory in `config.json` with `pdf_download_dir`
- Already-downloaded files are skipped
- Progress shown in status bar
- Supports batch downloads with multi-select

## Input File Format

The application parses arXiv email subscription text files. Expected format:

- Papers separated by `------------------------------------------------------------------------------`
- Each paper contains: arXiv ID, date, title, authors, categories, abstract, URL

Example paper entry:

```
------------------------------------------------------------------------------
\\
arXiv:2501.12345
Date: Mon, 20 Jan 2025 00:00:00 GMT   (15kb)

Title: Example Paper Title
Authors: Jane Doe, John Smith
Categories: cs.AI cs.LG
Comments: 10 pages, 5 figures
\\
  This is the abstract text describing the paper's contributions...
\\
( https://arxiv.org/abs/2501.12345 , 15kb)
------------------------------------------------------------------------------
```

## Configuration

User configuration is stored in a platform-specific location:

- **Linux**: `~/.config/arxiv-browser/config.json`
- **macOS**: `~/Library/Application Support/arxiv-browser/config.json`
- **Windows**: `%APPDATA%/arxiv-browser/config.json`

Configuration includes:
- Paper metadata (read status, stars, notes, tags)
- Watch list entries
- Search bookmarks
- Vim-style marks
- Session state (scroll position, filters, sort order)
- UI preferences (abstract preview toggle)
- arXiv API search preferences (`arxiv_api_max_results`)
- LLM summary settings (command, prompt template, preset)

### arXiv API Search Settings

Optional config key:

```json
{
  "arxiv_api_max_results": 50
}
```

Values are clamped to a safe range (`1..200`).

### AI Summary Setup

Generate paper summaries using any LLM CLI tool. Press `Ctrl+s` on a paper to generate an accessible, explanatory summary aimed at CS students. The full paper content is automatically fetched from the arXiv HTML version and passed to the LLM.

Add one of these to your `config.json`:

```json
{ "llm_preset": "copilot" }
```

Available presets: `claude` (`claude -p`), `codex` (`codex exec`), `llm` (`llm`), `copilot` (`copilot -p`).

Or configure a custom command with `{prompt}` placeholder:

```json
{
  "llm_command": "claude -p {prompt}",
  "llm_prompt_template": "Summarize: {title}\n\n{paper_content}"
}
```

Prompt placeholders: `{title}`, `{authors}`, `{categories}`, `{abstract}`, `{arxiv_id}`, `{paper_content}`.

Summaries are cached in a local SQLite database and persist across sessions.

### HuggingFace Trending Setup

Surface trending signals from HuggingFace Daily Papers — community upvotes, comments, GitHub info, AI summaries, and keywords. Press `Ctrl+h` to toggle on/off. Data is auto-fetched when enabled and cross-matched against loaded papers.

```json
{
  "hf_enabled": true,
  "hf_cache_ttl_hours": 6
}
```

- `hf_enabled`: Enable HF trending on startup (default: `false`)
- `hf_cache_ttl_hours`: Hours to cache HF data (default: `6`, trending data changes frequently)

Trending papers show upvote badges in the list view and a HuggingFace section in the detail pane with upvotes, comments, GitHub repository, AI summary, and keywords.

### Semantic Scholar Setup

Enrich papers with citation counts, fields of study, TLDRs, S2-powered recommendations, and citation graph exploration. Press `Ctrl+e` to toggle on/off.

```json
{
  "s2_enabled": true,
  "s2_api_key": "",
  "s2_cache_ttl_days": 7
}
```

- `s2_enabled`: Enable S2 enrichment on startup (default: `false`)
- `s2_api_key`: Optional API key for higher rate limits
- `s2_cache_ttl_days`: Days to cache S2 data (default: `7`)

Press `e` to fetch data for the current paper, `R` for S2-powered recommendations, and `G` to explore the citation graph with drill-down navigation.

### Relevance Scoring Setup

Score papers 1-10 based on your research interests using the configured LLM. Requires an LLM preset or command (same as AI summaries).

```json
{
  "research_interests": "efficient LLM inference, quantization, speculative decoding"
}
```

Press `L` to score all loaded papers, `Ctrl+l` to edit interests. Sort by relevance with `s`. Papers show colored score badges: green (8-10), yellow (5-7), dim (1-4).

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test class
uv run pytest -v test_arxiv_browser.py::TestCleanLatex

# Lint + type check
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

## Dependencies

- **textual** (>=7.3.0): TUI framework
- **rapidfuzz** (>=3.0.0): Fuzzy string matching
- **httpx** (>=0.27.0): Async HTTP client for API calls and PDF downloads
- **platformdirs**: Cross-platform config directory (transitive via textual)
- **pytest** (>=9.0.2): Testing (dev dependency)

## Author

Nicolas Sebastian Schuler (nicolas.schuler@kit.edu)

## License

MIT License - see [LICENSE](LICENSE) for details.
