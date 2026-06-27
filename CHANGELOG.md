# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-06-27

### Added
- Added runtime list/detail pane resizing with `Alt+Left` / `Alt+Right`, plus `Alt+0` to restore the default split.
- Persisted the `pane_split` preference so the selected split is restored on the next launch and adapts to both wide and narrow layouts.

### Changed
- Documented pane resizing in the README, cheat sheet, config reference, TUI style guide, help overlay, and command palette.

## [0.3.1] - 2026-06-18

### Added
- Six additional built-in themes — Dracula, Nord, Gruvbox Dark, Tokyo Night, Everforest Dark, and GitHub Light — bringing the total to 11 (`Ctrl+t` to cycle)
- Added a one-page quick reference cheat sheet for common CLI commands, keybindings, and search syntax.

### Changed
- Refreshed documentation for command-palette-only actions, export formats, metadata import scope, config/cache behavior, and troubleshooting diagnostics.
- Expanded docs/version drift checks to verify theme counts, export format documentation, and changelog footer links.

### Fixed
- Corrected older changelog release dates and restored the missing `0.3.0` comparison footer.

## [0.3.0] - 2026-05-29

### Added
- Added a responsive narrow-terminal layout: below 80 columns the list and detail panes stack vertically (list first) instead of squeezing the side-by-side split.
- Added built-in loading indicators on the paper-list and detail panes while background fetches run.
- Added mouse-clickable footer hints (each bound hint invokes the same action as its key) and tooltips for interactive chrome elements.

### Changed
- Migrated application state (selection, sort, view modes, enrichment flags, version-check progress) to Textual reactive attributes with watcher-driven UI refreshes, replacing scattered manual refresh calls.
- Migrated background operations to Textual workers (`@work`) for consistent cancellation and shutdown handling.
- Replaced manual string truncation with Rich-based rendering for the paper list and detail pane so wide characters and markup truncate correctly.
- Extracted `ReactiveStateMixin` and `WorkerRuntimeMixin` from the core app module to keep `browser/core.py` within its size budget.

## [0.2.0] - 2026-05-15

### Added
- Added focused runtime tests for cached UI refs, OmniInput command routing, chrome restoration, and TUI layout contracts.
- Expanded regression coverage across enrichment services, downloads, Semantic Scholar parsing, figure/PDF previews, LLM streaming, quick triage, export safety, database snapshots, and onboarding overlays.

### Changed
- Centralized Omni command chrome restoration so command execution consistently closes the command input, restores list focus when available, and refreshes footer state.

### Fixed
- Triage model prediction failures now clear stale predictions and report scoring errors without refreshing stale UI badges.

## [0.1.9] - 2026-05-15

### Added
- Added terminal HTML figure preview on `I`, with arXiv HTML figure extraction, cached image rendering, and graceful fallback warnings.
- Added width-aware status bar visual tokens for enrichment progress, reading velocity, and category distribution.
- Added detail-pane line annotations with context-sensitive `a` behavior and persisted `paper_metadata.line_annotations`.

### Changed
- Expanded tests and docs drift checks for the new keybinding, status visuals, figure cache/error paths, and metadata subfield documentation.

## [0.1.8] - 2026-05-14

### Added
- Added a Solarized Light theme with matching category/tag colors and CLI/docs coverage.
- Added TUI layout-contract tests across terminal sizes, ASCII/high-contrast mode, API mode, selection/detail focus, empty states, command palette, and major modals.

### Changed
- Aligned browse/detail footers, status text, and list headers with clearer mode and focus affordances.
- Improved paper-list badges with labeled S2, HuggingFace, relevance, and version-update metadata plus width-aware compression.
- Added a triage strip to the detail pane for read/star/tags/relevance/enrichment/version signals.
- Improved command-palette search, grouping, suggested markers, and disabled-action guidance.
- Added saved/unsaved footer state to editing, collections, and watch-list modals, plus filterable tag chips in the paper edit modal.

## [0.1.7] - 2026-05-14

### Changed
- Updated the GitHub Pages deployment workflow to use `actions/configure-pages@v6` and `actions/deploy-pages@v5`.

## [0.1.6] - 2026-05-14

### Changed
- Clarified LLM streaming summary ownership by separating prompt-content resolution from partial-summary UI publication.
- Documented the streaming architecture boundary between summary actions, paper-chat modals, providers, and durable cache writes.

### Fixed
- Added focused regression coverage for streaming summary enablement, abstract-only prompts, provider errors, and empty streamed responses.

## [0.1.5] - 2026-05-11

### Added
- Default MIT-compatible PDF support with `pypdf`, `pypdfium2`, and `Pillow`.
- Full-paper content caching with arXiv HTML first, PDF text fallback, then abstract fallback.
- Terminal PDF preview on `F` with cached PNG renders and half-block/ASCII display.
- Opt-in streaming LLM summaries and chat for CLI and OpenAI-compatible HTTP providers.

### Changed
- Documented PDF preview, full-paper content caching, and `llm_streaming_enabled`.
- Hardened LLM CLI timeout handling for shell-backed commands.

### Fixed
- Avoided repeated full-paper network fetches for summary/chat workflows.

## [0.1.4] - 2026-05-09

### Added
- Published-docs hero redesign with product-first copy, stronger screenshot placement, skip-link support, and live copy feedback.
- Documentation link hygiene checks for tracked local docs links.
- HTTP/OpenAI-compatible LLM provider diagnostics in `arxiv-viewer doctor`.
- Keyboard affordances for detail-pane focus toggling and inline command-palette result navigation.
- Theme contrast regression tests for core foreground/background pairs.

### Changed
- Folded docs drift checks into local quality gates and pre-commit.
- Expanded install, contributor, release, and LLM setup documentation.
- Clarified command-palette disabled-state and no-match guidance.
- Improved Solarized Dark contrast for selected and header states.
- Centralized action registration expectations by removing the unused parallel registry.

### Fixed
- Search cancel now returns focus to the paper list.
- Welcome-screen `?` now opens full help instead of only dismissing onboarding.
- ASCII mode now avoids Unicode hints in the command palette and release-notes modal.
- Removed a broken docs link to an ignored local prompt file.

## [0.1.3] - 2026-05-03

### Added
- Shell completion scripts for bash, zsh, and fish (`arxiv-viewer completions`)
- HTTP retry utility with exponential backoff for all external API calls
- Search/filter input inside the HelpScreen modal for quick keybinding lookup
- `arxiv-viewer doctor` command for environment and configuration health checks
- Onboarding flag (`onboarding_seen`) to track first-run state

### Changed
- Tightened import boundaries: the root package now exposes only its explicit `__all__`, `arxiv_browser.app` is a narrow compatibility shim, and tests import canonical modules directly instead of using a repo-local export bundle
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

## [0.1.2] - 2026-02-16

### Added
- `arxiv-viewer search` subcommand: search arXiv API directly from the CLI without preparing local files
- Latest-day startup mode (`--latest`) that auto-fetches the current day's papers on launch
- GitHub Pages documentation site at <https://nicolasschuler.github.io/arxiv-subscription-viewer/>

### Changed
- Added an interactive-modules coverage guardrail (≥85%) to CI for `actions/`, `browser/`, and `cli.py`
- Quality job now enforces xenon complexity gates (C/C/B), bandit, vulture, and deptry
- Improved TUI style guide and help/footer UX copy
- CLI extraction into `cli.py`; async similarity indexing

### Fixed
- Parser version deduplication for papers with multiple arXiv versions
- Stable tag-color assignment across sessions (bounded LRU cache)
- Streamed PDF downloads to avoid memory spikes on large files

## [0.1.1] - 2026-02-13

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

## [0.1.0] - 2026-02-12

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

[Unreleased]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NicolasSchuler/arxiv-subscription-viewer/releases/tag/v0.1.0
