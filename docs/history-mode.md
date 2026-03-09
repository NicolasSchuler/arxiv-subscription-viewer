# 📅 History Mode & Email Ingestion

Browse daily arXiv email digests stored as dated text files.

## Directory Setup

Create a `history/` directory where you run `arxiv-viewer`:

```
~/research/arxiv/
└── history/
    ├── 2026-01-20.txt
    ├── 2026-01-21.txt
    └── 2026-01-23.txt
```

Files must be named `YYYY-MM-DD.txt`. The app auto-loads the newest file on startup.

## Navigation

| Key | Action |
|-----|--------|
| `[` | Previous (older) date |
| `]` | Next (newer) date |

Session state (including current date) persists across runs. Falls back to `arxiv.txt` if no `history/` directory exists. Limited to the 365 most recent files.

## CLI Options

```bash
arxiv-viewer                    # Auto-load newest from history/
arxiv-viewer --list-dates       # List available dates
arxiv-viewer --date 2026-01-23  # Open specific date
arxiv-viewer -i papers.txt      # Custom file (disables history mode)
arxiv-viewer --no-restore       # Ignore saved session state
```

## Automating Email Ingestion

`arxiv-viewer` only needs dated text files in `history/`, so any email automation that writes `history/YYYY-MM-DD.txt` will work.

**Recommended setup:**

1. Create a dedicated mailbox folder/label for your arXiv digest emails
2. Add a mail rule or scheduled job that exports the latest digest body as plain text to `history/YYYY-MM-DD.txt`
3. Keep one file per date (overwrite if needed); the viewer auto-loads the newest file

**Manual fallback on macOS:**

```bash
pbpaste > "history/$(date +%F).txt"
```

## Input File Format

The parser expects arXiv email subscription format:

```
------------------------------------------------------------------------------
\
arXiv:2501.12345
Date: Mon, 20 Jan 2025 00:00:00 GMT   (15kb)

Title: Example Paper Title
Authors: Jane Doe, John Smith
Categories: cs.AI cs.LG
Comments: 10 pages, 5 figures
\
  This is the abstract text describing the paper's contributions...
\
( https://arxiv.org/abs/2501.12345 , 15kb)
------------------------------------------------------------------------------
```
