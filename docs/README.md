# Documentation

This file is the **repository-friendly docs index** for people browsing on GitHub. If you want the polished published landing page instead, start at the **[GitHub Pages guide hub](https://nicolasschuler.github.io/arxiv-subscription-viewer/)**. For install commands and the fastest CLI overview, use the **[main README](../README.md)**.

The app supports two first-class entry paths:

- Run from a local `history/` archive when you already save daily digests and want date navigation plus persistent local state.
- Start from `arxiv-viewer search ...` when you want live arXiv API results without managing local files first.

If setup or discovery looks wrong, run `arxiv-viewer doctor` before digging deeper.

Review organization is local-first: use read state, stars, tags, notes, marks, bookmarks, collections, and command-palette spaced-review scheduling to keep a durable paper queue. Scheduled papers can be surfaced later with the `review-due` filter.

## Fast Path

1. **Install** from the [main README](../README.md), then run `arxiv-viewer doctor`.
2. **Scan** live results with `arxiv-viewer search ...` or local digest files with `arxiv-viewer browse`.
3. **Enrich** the shortlist with LLM summaries, Semantic Scholar, HuggingFace, version checks, and figure/PDF previews.
4. **Organize** papers with read state, stars, tags, notes, bookmarks, marks, collections, and review scheduling.
5. **Export** selected papers or collections with the [export guide](export.md).
6. **Configure** defaults, paths, themes, and API keys with [config-reference.md](config-reference.md).

## Choose The Right Entry Point

- **[Main README](../README.md)**: install, quick commands, highlights, and key bindings
- **[Published guide hub](https://nicolasschuler.github.io/arxiv-subscription-viewer/)**: the static landing page used for GitHub Pages
- **[config-reference.md](config-reference.md)**: direct configuration schema and option reference
- **This file**: GitHub-native map of the end-user docs set

## Recommended Reading Order

1. [Main README](../README.md) for install, quick commands, and core shortcuts
2. [search-filters.md](search-filters.md) for live arXiv scans and complex local filtering
3. [history-mode.md](history-mode.md) if you keep local digest files
4. [llm-setup.md](llm-setup.md), [semantic-scholar.md](semantic-scholar.md), and [huggingface.md](huggingface.md) when you are ready to enrich papers
5. [export.md](export.md) when you want to move selected papers into BibTeX, RIS, CSV, Markdown, or PDF folders
6. [config-reference.md](config-reference.md) once you start customizing paths, themes, API keys, or enrichment defaults
7. [troubleshooting.md](troubleshooting.md) when `doctor` or first-run behavior surfaces problems

## Feature Guides

| Document | Description |
|----------|-------------|
| [history-mode.md](history-mode.md) | `history/` setup, date navigation, and digest-ingestion workflow |
| [search-filters.md](search-filters.md) | Query syntax, filter prefixes, boolean operators, bookmarks, and API search |
| [digest.md](digest.md) | Non-interactive daily/weekly Markdown digests for cron, email, or Slack piping |
| [llm-setup.md](llm-setup.md) | AI summaries, chat, paper comparison, relevance scoring, auto-tagging |
| [semantic-scholar.md](semantic-scholar.md) | Citation counts, recommendations, citation graph |
| [huggingface.md](huggingface.md) | Trending papers, community upvotes |
| [export.md](export.md) | BibTeX, RIS, CSV, Markdown, PDF downloads, collections |
| [config-reference.md](config-reference.md) | Full configuration schema reference |
| [troubleshooting.md](troubleshooting.md) | Common issues and solutions |

## Getting Started

See the [main README](../README.md) for installation, usage, and key bindings, then pick the guide that matches your workflow.

## Internal Development Docs

These references are development-only and are not intended as first-run user guides.

| Document | Purpose |
|----------|---------|
| [architecture.md](architecture.md) | Contributor architecture, canonical imports, compatibility boundaries, and extraction guidance |
| [tui-style-guide.md](tui-style-guide.md) | TUI copy, layout, footer, accessibility, keybinding, and verification conventions |
