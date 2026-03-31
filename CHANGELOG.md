# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Shell completion scripts for bash, zsh, and fish (`arxiv-viewer completions`)
- HTTP retry utility with exponential backoff for all external API calls
- Search/filter input inside the HelpScreen modal for quick keybinding lookup
- `arxiv-viewer doctor` command for environment and configuration health checks
- Onboarding flag (`onboarding_seen`) to track first-run state

### Changed
- Refactored browser mixins into `browser/browse.py`, `browser/chrome.py`, and `browser/discovery.py` for maintainability
- Split `semantic_scholar.py` cache helpers into `semantic_scholar_cache.py`
- Extracted `chrome_status.py` widget for status-bar rendering
- Hardened LLM command safety: shell metacharacter detection, timeout protection, and clearer error feedback
- Improved ASCII-mode fallbacks for all Unicode glyphs across widgets
- Raised overall test coverage to 94.5%; added branch-coverage gating (≥85%)
- Stabilised CI timing sensitivity in date-navigator and debounce tests

### Fixed
- Dataset state and theme-refresh regressions after mixin extraction
- LLM action config and batch-cancellation edge cases
- Date navigator CI flakiness on slow hosts

## [0.1.2] - 2025-01-26

### Added
- `arxiv-viewer search` subcommand: search arXiv API directly from the CLI without preparing local files
- Latest-day startup mode (`--latest`) that auto-fetches the current day's papers on launch
- GitHub Pages documentation site at <https://nicolasschuler.github.io/arxiv-subscription-viewer/>

### Changed
- Added `app.py` coverage guardrail (≥80%) to CI
- Quality job now enforces xenon complexity gates (C/C/B), bandit, vulture, and deptry
- Improved TUI style guide and help/footer UX copy
- CLI extraction into `cli.py`; async similarity indexing

### Fixed
- Parser version deduplication for papers with multiple arXiv versions
- Stable tag-color assignment across sessions (bounded LRU cache)
- Streamed PDF downloads to avoid memory spikes on large files

## [0.1.1] - 2024-12-15

### Added
- Modal extraction refactor: 20 `ModalScreen` subclasses split into `modals/` package
  (`common`, `editing`, `search`, `collections`, `citations`, `llm`)
- LLM provider abstraction (`llm_providers.py`): bring-your-own-LLM via configurable subprocess command
- Hypothesis property-based tests covering parsing, export, similarity, query, and config round-trips
- Performance benchmarks (`just bench`) guarding against O(n²) regressions
- Integration tests with real fixture data and resource-cleanup verification

### Changed
- Reduced cyclomatic complexity: five F-rated functions refactored to pass xenon gate
- BibTeX escaping fix for special characters in author names and titles
- Async cleanup hardened to prevent resource leaks on shutdown

## [0.1.0] - 2024-11-01

### Added
- Initial release: Textual-based TUI for browsing arXiv papers from email subscription digests
- History mode with date navigation (`[`/`]`), persistent bookmarks (1–9), and local archive
- Fuzzy search with query syntax (`author:`, `cat:`, `tag:`, boolean operators `AND`/`OR`/`NOT`)
- Paper annotations: read/star status, free-text notes, namespaced tags (`topic:ml`, `status:to-read`)
- Paper collections (named reading lists) with `Ctrl+k`
- Watch list: highlight papers matching author/keyword/title patterns
- Semantic Scholar enrichment: citation counts, TLDRs, recommendations, citation graph
- HuggingFace Daily Papers trending data and community upvotes
- LLM summaries, relevance scoring, auto-tagging, and paper chat via configurable subprocess
- Export formats: BibTeX, RIS, CSV, Markdown table, PDF download
- TF-IDF similarity index for local "find similar papers"
- Multiple color themes: Monokai, Catppuccin, Solarized, High Contrast
- ASCII mode for terminals without Unicode support (`--ascii`)
- Session persistence: scroll position, active filters, sort order

[Unreleased]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/releases/tag/v0.1.0
