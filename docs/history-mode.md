# 📅 History Mode & Email Ingestion

Use history mode when you already save daily digests locally and want the app to behave like a keyboard-first review queue. If you do not keep local files, skip to [Search & Filters](search-filters.md) and start with `arxiv-viewer search ...` instead.

## Directory Setup

Create a `history/` directory in the workspace where you run `arxiv-viewer`:

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
| `T` | Quick triage visible unread papers |

Session state (including current date) persists across runs. Falls back to `arxiv.txt` if no `history/` directory exists.

## Smart Reading Queue

Cycle sorting with `s` until `sort:queue` appears to use the Smart Reading Queue. Queue mode is a priority sort over the loaded date file: relevance scores, watch-list matches, recency, HuggingFace upvotes, and Semantic Scholar citation velocity all contribute when available. It does not create a separate queue or change read/unread state.

When queue mode is selected, it stays active as you move between date files, so each newly loaded digest is auto-ranked with the same priority formula.

## Local Triage Model

Once you have enough saved decisions, open `Ctrl+p` and run **Train Triage Model**. The model uses local sklearn TF-IDF features plus logistic regression to learn from stars, collection saves, and skips. It adds `ML:` badges and a `sort:triage` mode with likely-star papers first, uncertain papers next, and likely skips last. Tagged-only decisions such as `triage:later` are treated as neutral training data and are not forced into either class.

Use **Clear Triage Model** from the command palette to delete the local model artifacts.

## Quick Triage

Press `T` to review the current visible unread queue one paper at a time. Each card shows the title, the first two abstract lines, any existing relevance score, local ML triage badge, and a watch-list badge when matched.

| Key | Action |
|-----|--------|
| `y` | Star and mark read |
| `n` | Skip by marking read |
| `t` | Add `triage:later` and mark read |
| `s` | Save to a collection and mark read |
| `Esc` / `q` | Close with a partial summary |

## Trend Radar

Open `Ctrl+p` and run **Trend Radar** to inspect your local `history/` archive. The overlay reads the same dated digest files used for navigation; it does not query live arXiv API results.

Trend Radar shows:

- growing primary categories, with sparklines ordered oldest to newest
- top authors across the parsed local history
- hot recent abstract bigrams from the newest history dates

The recent window defaults to the newest 10 history files and compares them with the previous 10 when available. A positive category delta means the category appeared more often in the recent window than in the previous one.

## CLI Options

```bash
arxiv-viewer                          # Alias for browse; auto-load newest from history/
arxiv-viewer dates                    # List available dates
arxiv-viewer browse --date 2026-01-23 # Open specific date
arxiv-viewer browse -i papers.txt     # Custom file (disables history mode)
arxiv-viewer browse --no-restore      # Ignore saved session state
```

## Automating Email Ingestion

`arxiv-viewer` only needs dated text files in `history/`, so any automation that writes `history/YYYY-MM-DD.txt` will work. If the app is not finding your files, run `arxiv-viewer doctor` to confirm the current working directory and history discovery status.

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

---

**See also:** [Search & Filters](search-filters.md) · [Config Reference](config-reference.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
