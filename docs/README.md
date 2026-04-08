# Documentation

This file is the **repository-friendly docs index** for people browsing on GitHub. If you want the polished published landing page instead, start at the **[GitHub Pages guide hub](https://nicolasschuler.github.io/arxiv-subscription-viewer/)**. For install commands and the fastest CLI overview, use the **[main README](../README.md)**.

The app supports two first-class entry paths:

- Run from a local `history/` archive when you already save daily digests and want date navigation plus persistent local state.
- Start from `arxiv-viewer search ...` when you want live arXiv API results without managing local files first.

If setup or discovery looks wrong, run `arxiv-viewer doctor` before digging deeper.

## Choose The Right Entry Point

- **[Main README](../README.md)**: install, quick commands, highlights, and key bindings
- **[Published guide hub](https://nicolasschuler.github.io/arxiv-subscription-viewer/)**: the static landing page used for GitHub Pages
- **[config-reference.md](config-reference.md)**: direct configuration schema and option reference
- **This file**: GitHub-native map of the end-user docs set

## Recommended Reading Order

1. [Main README](../README.md) for install, quick commands, and core shortcuts
2. [Published guide hub](https://nicolasschuler.github.io/arxiv-subscription-viewer/) for the short visual overview and workflow chooser
3. [history-mode.md](history-mode.md) if you keep local digest files
4. [search-filters.md](search-filters.md) if you want API search or complex local filtering
5. [config-reference.md](config-reference.md) once you start customizing exports, themes, or enrichment
6. [troubleshooting.md](troubleshooting.md) when `doctor` or first-run behavior surfaces problems

## Feature Guides

| Document | Description |
|----------|-------------|
| [history-mode.md](history-mode.md) | `history/` setup, date navigation, and digest-ingestion workflow |
| [search-filters.md](search-filters.md) | Query syntax, filter prefixes, boolean operators, bookmarks, and API search |
| [llm-setup.md](llm-setup.md) | AI summaries, chat, relevance scoring, auto-tagging |
| [semantic-scholar.md](semantic-scholar.md) | Citation counts, recommendations, citation graph |
| [huggingface.md](huggingface.md) | Trending papers, community upvotes |
| [export.md](export.md) | BibTeX, RIS, CSV, Markdown, PDF downloads, collections |
| [config-reference.md](config-reference.md) | Full configuration schema reference |
| [troubleshooting.md](troubleshooting.md) | Common issues and solutions |

## Getting Started

See the [main README](../README.md) for installation, usage, and key bindings, then pick the guide that matches your workflow.

## Internal Development Docs

The following are development-only references and not intended for end users:

- [architecture.md](architecture.md) — contributor architecture, canonical imports, and compatibility boundaries
- [tui-style-guide.md](tui-style-guide.md) — UI/UX conventions for TUI development
- [code-quality-pipeline-prompt.md](code-quality-pipeline-prompt.md) — CI quality pipeline configuration
