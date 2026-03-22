# Documentation

Start here if you want the shortest path from install to a working review workflow. The app supports two first-class entry paths:

- Run from a local `history/` archive when you already save daily digests and want date navigation plus persistent local state.
- Start from `arxiv-viewer search ...` when you want live arXiv API results without managing local files first.

If setup or discovery looks wrong, run `arxiv-viewer doctor` before digging deeper.

## Recommended Reading Order

1. [Main README](../README.md) for install, quick commands, and core shortcuts
2. [history-mode.md](history-mode.md) if you keep local digest files
3. [search-filters.md](search-filters.md) if you want API search or complex local filtering
4. [config-reference.md](config-reference.md) once you start customizing exports, themes, or enrichment
5. [troubleshooting.md](troubleshooting.md) when `doctor` or first-run behavior surfaces problems

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

- [tui-style-guide.md](tui-style-guide.md) — UI/UX conventions for TUI development
- [code-quality-pipeline-prompt.md](code-quality-pipeline-prompt.md) — CI quality pipeline configuration
