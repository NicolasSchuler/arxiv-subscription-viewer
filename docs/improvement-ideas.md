# Improvement Ideas for arxiv-subscription-viewer

> Brainstormed 2026-02-12. Baseline metrics at time of writing:
> - 12,900 lines across 13 modules (`app.py` still 8,341 lines)
> - 1,051 tests, 69% overall coverage (57% on `app.py`)
> - Average cyclomatic complexity: C (17.4), with 5 F-rated functions
> - CI runs: ruff, pyright, pytest — no complexity, security, or dead-code gates

---

## 1. ~~Extract Modals into a `modals/` Package~~ DONE

**Category:** Architecture
**Effort:** Medium
**Impact:** High

`app.py` contained 19 `ModalScreen` subclasses (~2,500+ lines). Extracted into
`src/arxiv_browser/modals/` with 6 domain-grouped submodules: common.py, editing.py,
search.py, collections.py, citations.py, llm.py. Reduced `app.py` from ~8,300 to
~5,800 lines. See local planning docs for the detailed extraction design notes.

## 2. ~~Extract Widgets into a `widgets/` Package~~ DONE

**Category:** Architecture
**Effort:** Medium
**Impact:** High

Extracted reusable widgets into `src/arxiv_browser/widgets/`:
`listing.py` (`PaperListItem`, `render_paper_option`), `details.py`
(`PaperDetails`), and `chrome.py` (`FilterPillBar`, `BookmarkTabBar`,
`DateNavigator`, `ContextFooter`). `app.py` now acts as composition/glue and
re-exports widget symbols for backward compatibility (`from arxiv_browser.app import ...`).
This reduced `app.py` from ~5,806 to ~4,792 lines.

## 3. Raise `app.py` Coverage from 57% to 80%

**Category:** Testing
**Effort:** High
**Impact:** High

The extracted modules are 92-100% covered, but `app.py` at 57% drags the overall
number. The ~1,700 uncovered lines include action handlers, the status bar, footer
bindings, and the chat screen. Textual's `async with app.run_test()` pilot pattern
would cover many UI interaction paths.

## 4. ~~Reduce Cyclomatic Complexity of F-rated Functions~~ DONE

**Category:** Code Quality
**Effort:** Medium
**Impact:** Medium

Five functions rated F for complexity:

| Function | Complexity | Module |
|---|---|---|
| `import_metadata` | 47 | config.py |
| `_dict_to_config` | 29 | config.py |
| `main` | 26 | app.py |
| `render_paper_option` | 25 | app.py |
| `tokenize_query` | 21 | query.py |

`import_metadata` at 47 is extreme — splitting by data type (papers, watchlist,
bookmarks, collections) would drop it dramatically.

## 5. ~~Property-Based Testing with Hypothesis~~ DONE

**Category:** Testing
**Effort:** Medium
**Impact:** Medium

Pure functions in `parsing.py`, `query.py`, `export.py`, and `similarity.py` are
ideal candidates:

- `clean_latex(clean_latex(x)) == clean_latex(x)` (idempotency)
- `tokenize_query()` -> `reconstruct_query()` round-trips
- `format_paper_as_bibtex()` always produces valid BibTeX
- `parse_arxiv_file()` never crashes on arbitrary input

## 6. ~~Integration Tests with Real arXiv Emails~~ DONE

**Category:** Testing
**Effort:** Low
**Impact:** Medium

No end-to-end test loads a real `.txt` file, navigates dates, stars a paper,
exports BibTeX, and verifies config round-trips. A `tests/fixtures/` directory with
2-3 sanitized email samples + Textual pilot-driven integration tests would catch
full-pipeline regressions.

## 7. ~~Async Resource Management Audit~~ DONE

**Category:** Reliability
**Effort:** Low
**Impact:** Medium

Audit `httpx.AsyncClient` and SQLite connections for proper cleanup on all exit
paths. Consider `asyncio.TaskGroup` (3.13+) to replace the manual `_track_task()`
pattern. Verify exception propagation from fire-and-forget tasks.

## 8. ~~Structured Logging to File~~ DONE

**Category:** Observability
**Effort:** Low
**Impact:** Medium

A `--debug` flag that enables file logging to `~/.config/arxiv-browser/debug.log`
with rotation. Log API calls, cache hits/misses, and timing. Makes it possible for
users to share logs when reporting issues. Currently debugging a TUI app is hard
because stdout is captured by Textual.

## 9. ~~Declarative Config Validation~~ DONE

**Category:** Robustness
**Effort:** Medium
**Impact:** Medium

`_dict_to_config` (complexity 29) manually validates every field. Replace with
type-checked deserialization — `cattrs`, a `from_dict()` classmethod, or even
`__post_init__` validators. Would eliminate hand-written validation and make adding
new config fields trivial.

## 10. ~~Performance Profiling & Startup Benchmark~~ DONE

**Category:** Performance
**Effort:** Low
**Impact:** Low-Medium

Establish baselines for startup time, search latency with 500+ papers, and memory
footprint. A `tests/benchmarks/` directory with `pytest-benchmark` or
`time.perf_counter` snapshots would catch regressions.

## 11. ~~Plugin / Hook System for LLM Providers~~ DONE

**Category:** Extensibility
**Effort:** Medium
**Impact:** Low-Medium

The LLM integration shells out to CLI tools. A lightweight plugin interface
(a `Protocol` class with a `__call__` method) would allow direct API calls
(faster than subprocess), custom providers, and easier testing with mock providers.

## 12. ~~Accessibility Audit~~ DONE

**Category:** UX
**Effort:** Low
**Impact:** Medium

The app uses color extensively for state (read/unread, star, watch, badges).
Verify all color-coded information is also conveyed by text/symbols. Add a
high-contrast theme option. Test screen reader compatibility with Textual's
built-in accessibility support.

---

## Priority Matrix

| Priority | Ideas | Rationale |
|---|---|---|
| **Now** | Quality tooling (prerequisite) | Need measurement before improvement |
| **High** | 1, 2 (modal/widget extraction) | Highest leverage for maintainability |
| **High** | 3, 4 (coverage + complexity) | Quality gates prevent regressions |
| **Medium** | 5, 6 (testing depth) | Safety net for future refactoring |
| **Medium** | 7, 8, 9 (reliability) | Production-readiness improvements |
| **Lower** | 10, 11, 12 (nice-to-have) | Polish and extensibility |
