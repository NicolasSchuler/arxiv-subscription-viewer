# Design: Extract Modals into `modals/` Package

> Date: 2026-02-13
> Status: Completed
> Roadmap item: #1 (Extract Modals into a `modals/` Package)

## Goal

Extract 19 `ModalScreen` subclasses (~2,500 lines) from `app.py` into a domain-grouped `modals/` package. Reduce `app.py` from 8,325 lines to ~5,800 lines. Preserve all behavior, update all test imports (no backward-compat shims).

## Package Structure

```
src/arxiv_browser/modals/
├── __init__.py           # Re-exports all 19 modal classes + helper ListItems
├── editing.py            # NotesModal, TagsModal, AutoTagSuggestModal
├── search.py             # ArxivSearchModal, CommandPaletteModal
├── collections.py        # CollectionsModal, CollectionViewModal, AddToCollectionModal
├── citations.py          # RecommendationsScreen, RecommendationSourceModal,
│                         #   CitationGraphScreen + RecommendationListItem, CitationGraphListItem
├── llm.py                # SummaryModeModal, ResearchInterestsModal, PaperChatScreen
└── common.py             # HelpScreen, ConfirmModal, ExportMenuModal,
                          #   SectionToggleModal, WatchListModal + WatchListItem
```

### File Sizes (estimated)

| File | Modals | Lines |
|------|--------|-------|
| editing.py | NotesModal, TagsModal, AutoTagSuggestModal | ~430 |
| search.py | ArxivSearchModal, CommandPaletteModal | ~280 |
| collections.py | CollectionsModal, CollectionViewModal, AddToCollectionModal | ~450 |
| citations.py | RecommendationsScreen, RecommendationSourceModal, CitationGraphScreen | ~560 |
| llm.py | SummaryModeModal, ResearchInterestsModal, PaperChatScreen | ~340 |
| common.py | HelpScreen, ConfirmModal, ExportMenuModal, SectionToggleModal, WatchListModal | ~530 |

## What Moves With Each Modal

- **Custom ListItem subclasses** stay with their modal: `WatchListItem` → `common.py`, `RecommendationListItem` → `citations.py`, `CitationGraphListItem` → `citations.py`
- **Module-level constants used only by one group** move with them: `_SECTION_TOGGLE_KEYS` → `common.py`, `RECOMMENDATION_TITLE_MAX_LEN` → `citations.py`
- **Constants shared with ArxivBrowser** stay in `app.py`: `COMMAND_PALETTE_COMMANDS`, `BATCH_CONFIRM_THRESHOLD`, `LLM_COMMAND_TIMEOUT`

## Dependency DAG

```
models.py              <- 0 internal deps (leaf)
themes.py              <- 0 internal deps (leaf)
config.py              <- models
parsing.py             <- models
export.py              <- models
query.py               <- models, themes
llm.py                 <- models
llm_providers.py       <- llm, models
similarity.py          <- models
semantic_scholar.py    <- models
huggingface.py         <- models
modals/                <- models, themes, query, parsing, llm, llm_providers, semantic_scholar
app.py                 <- all above (including modals/)
```

No module in `modals/` imports from `app.py` — the DAG remains acyclic.

### Per-File Dependencies

| File | Internal Deps |
|------|---------------|
| editing.py | themes (parse_tag_namespace, get_tag_color) |
| search.py | models (ArxivSearchRequest), parsing (ARXIV_QUERY_FIELDS, build_arxiv_search_query), themes (THEME_COLORS), query (escape_rich_text) |
| collections.py | models (PaperCollection, MAX_COLLECTIONS, Paper) |
| citations.py | models (Paper), semantic_scholar (CitationEntry), query (truncate_text, escape_rich_text), themes (THEME_COLORS) |
| llm.py | models (Paper), llm (CHAT_SYSTEM_PROMPT), llm_providers (CLIProvider), query (escape_rich_text), themes (THEME_COLORS) |
| common.py | models (WatchListEntry, WATCH_MATCH_TYPES, DETAIL_SECTION_NAMES), themes (THEME_COLORS) |

## App-Level Coupling

Three modals access `self.app`:

1. **CitationGraphScreen**: `self.app._track_task()` for async task management
2. **PaperChatScreen**: `self.app._track_task()` for async LLM calls
3. **CollectionsModal**: `self.app.push_screen()` to open CollectionViewModal

These use Textual's built-in `self.app` reference (available on all widgets). No import of `ArxivBrowser` is needed. The coupling is accepted as standard Textual practice.

## Import Strategy

- **No backward-compat re-exports**: All test imports updated from `from arxiv_browser.app import SomeModal` to `from arxiv_browser.modals import SomeModal`
- **`app.py`**: Adds `from arxiv_browser.modals import (...)` to access modal classes
- **`modals/__init__.py`**: Re-exports all 19 modals + 3 helper ListItems for flat imports
- **`__init__.py` (package root)**: Chain: `modals/*.py` -> `modals/__init__.py` -> `app.py __all__` -> `__init__.py`

## What Stays in `app.py` (~5,800 lines)

- `PaperListItem` widget (~300 lines) -- future extraction in Item #2
- `PaperDetails` widget (~600 lines) -- future extraction in Item #2
- `FilterPillBar`, `BookmarkTabBar`, `DateNavigator` widgets -- future Item #2
- `ArxivBrowser` app class (~3,500 lines)
- Module-level constants shared with the app
- `main()` CLI entry point

## Testing

- All existing tests remain, only import paths change
- No new tests needed (pure structural refactor, no behavior change)
- Verification: `just check` (lint + typecheck + tests with coverage) must pass
- `just quality` for full quality suite confirmation

## Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Package vs flat files | `modals/` package | Matches roadmap intent, scales for 19 classes |
| Grouping strategy | Domain-grouped (6 files) | Natural cohesion, 300-500 lines each |
| Backward compat | None (update all imports) | Clean break, no re-export shims |
| App coupling | Accept `self.app` calls | Standard Textual pattern, no abstraction needed |
