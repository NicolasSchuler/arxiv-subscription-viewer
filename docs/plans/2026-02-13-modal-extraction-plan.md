# Modal Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract 19 ModalScreen subclasses from `app.py` into a domain-grouped `modals/` package, reducing `app.py` from 8,325 to ~5,800 lines.

**Architecture:** Create `src/arxiv_browser/modals/` with 6 domain-grouped files plus `__init__.py`. Each modal moves with its helper ListItem subclasses and private constants. No backward-compat re-exports — all test imports updated to `from arxiv_browser.modals import X`.

**Tech Stack:** Python 3.13, Textual ModalScreen, pytest

**Design doc:** `docs/plans/2026-02-13-modal-extraction-design.md`

---

## Task 1: Create `modals/` Package Skeleton

**Files:**
- Create: `src/arxiv_browser/modals/__init__.py`
- Create: `src/arxiv_browser/modals/common.py`
- Create: `src/arxiv_browser/modals/editing.py`
- Create: `src/arxiv_browser/modals/search.py`
- Create: `src/arxiv_browser/modals/collections.py`
- Create: `src/arxiv_browser/modals/citations.py`
- Create: `src/arxiv_browser/modals/llm.py`

**Step 1: Create the package directory and empty files**

```bash
mkdir -p src/arxiv_browser/modals
touch src/arxiv_browser/modals/__init__.py
touch src/arxiv_browser/modals/common.py
touch src/arxiv_browser/modals/editing.py
touch src/arxiv_browser/modals/search.py
touch src/arxiv_browser/modals/collections.py
touch src/arxiv_browser/modals/citations.py
touch src/arxiv_browser/modals/llm.py
```

**Step 2: Write `__init__.py` with all re-exports (initially commented out)**

The `__init__.py` will re-export all 19 modal classes plus 3 helper ListItems. Write it with the full import list, but leave them commented until each module is populated:

```python
"""Modal dialogs for the arXiv Browser TUI.

Domain-grouped ModalScreen subclasses extracted from app.py.
Import modals from this package: ``from arxiv_browser.modals import TagsModal``
"""

# common.py — general-purpose dialogs
from arxiv_browser.modals.common import (
    ConfirmModal,
    ExportMenuModal,
    HelpScreen,
    SectionToggleModal,
    WatchListItem,
    WatchListModal,
)

# editing.py — paper metadata editing
from arxiv_browser.modals.editing import (
    AutoTagSuggestModal,
    NotesModal,
    TagsModal,
)

# search.py — search and command palette
from arxiv_browser.modals.search import (
    ArxivSearchModal,
    CommandPaletteModal,
)

# collections.py — paper collections / reading lists
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionViewModal,
    CollectionsModal,
)

# citations.py — recommendations and citation graph
from arxiv_browser.modals.citations import (
    CitationGraphListItem,
    CitationGraphScreen,
    RecommendationListItem,
    RecommendationSourceModal,
    RecommendationsScreen,
)

# llm.py — LLM summary, relevance, and chat
from arxiv_browser.modals.llm import (
    PaperChatScreen,
    ResearchInterestsModal,
    SummaryModeModal,
)

__all__ = [
    # common
    "ConfirmModal",
    "ExportMenuModal",
    "HelpScreen",
    "SectionToggleModal",
    "WatchListItem",
    "WatchListModal",
    # editing
    "AutoTagSuggestModal",
    "NotesModal",
    "TagsModal",
    # search
    "ArxivSearchModal",
    "CommandPaletteModal",
    # collections
    "AddToCollectionModal",
    "CollectionViewModal",
    "CollectionsModal",
    # citations
    "CitationGraphListItem",
    "CitationGraphScreen",
    "RecommendationListItem",
    "RecommendationSourceModal",
    "RecommendationsScreen",
    # llm
    "PaperChatScreen",
    "ResearchInterestsModal",
    "SummaryModeModal",
]
```

**Step 3: Commit skeleton**

```bash
git add src/arxiv_browser/modals/
git commit -m "chore: create modals/ package skeleton"
```

---

## Task 2: Extract `common.py` — HelpScreen, ConfirmModal, ExportMenuModal, SectionToggleModal, WatchListModal

**Files:**
- Modify: `src/arxiv_browser/app.py` (remove classes)
- Create: `src/arxiv_browser/modals/common.py`
- Modify: `tests/test_arxiv_browser.py` (update imports)

**What to move (line ranges in current app.py):**
- `HelpScreen` (lines 1326–1473)
- `ConfirmModal` (lines 2587–2650)
- `ExportMenuModal` (lines 2652–2758)
- `_SECTION_TOGGLE_KEYS` dict (lines 3202–3211)
- `SectionToggleModal` (lines 3214–3326)
- `WatchListItem` (lines 1695–1701)
- `WatchListModal` (lines 1703–1911)

**Step 1: Write `common.py`**

Copy the classes listed above into `src/arxiv_browser/modals/common.py`. Add these imports at the top:

```python
"""General-purpose modal dialogs."""

from __future__ import annotations

import logging

from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, ListItem, ListView, Select, Static

from arxiv_browser.models import DETAIL_SECTION_NAMES, WATCH_MATCH_TYPES, WatchListEntry
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)
```

Then paste the class bodies verbatim. The `_SECTION_TOGGLE_KEYS` dict goes above `SectionToggleModal` in this file.

**Step 2: Remove the moved classes from `app.py`**

Delete the line ranges listed above from `app.py`. Be careful to keep surrounding code intact (the classes before `HelpScreen` at line 1326 and after `WatchListModal` at line 1911 must remain).

**Step 3: Add modals import to `app.py`**

In `app.py`, after the existing module imports (around line 270), add:

```python
from arxiv_browser.modals import (
    ConfirmModal,
    ExportMenuModal,
    HelpScreen,
    SectionToggleModal,
    WatchListItem,
    WatchListModal,
)
```

This is temporary — we'll consolidate the import once all modals are extracted.

**Step 4: Update test imports**

In `tests/test_arxiv_browser.py`, find all local imports of these modals and change `from arxiv_browser.app import` to `from arxiv_browser.modals import`:

- `HelpScreen` — lines 1521, 2608
- `ConfirmModal` — lines 4264, 4301
- `ExportMenuModal` — lines 9012, 9031, 9051, 9695, 9704, 9714, 9724, 9734, 9744, 9754, 9764, 9774, 9784, 9794, 9802, 9808, 9820
- `SectionToggleModal` — lines 9988, 9994, 10000, 10007, 10014, 10023, 10030, 10037, 10044, 10051, 10058, 10065, 10072, 10079, 10088, 10098, 10112, 10120, 10136, 10151

For lines that import multiple names (e.g., `from arxiv_browser.app import ArxivBrowser, HelpScreen`), split into two import statements — one for `arxiv_browser.app` and one for `arxiv_browser.modals`.

**Step 5: Run tests to verify**

```bash
just test-quick
```

Expected: All tests pass. If any import errors, check that all names are correctly re-exported from `modals/__init__.py`.

**Step 6: Run lint + typecheck**

```bash
just lint && just typecheck
```

Fix any issues (likely ruff import sorting).

**Step 7: Commit**

```bash
git add src/arxiv_browser/modals/common.py src/arxiv_browser/modals/__init__.py src/arxiv_browser/app.py tests/test_arxiv_browser.py
git commit -m "refactor: extract common modals to modals/common.py"
```

---

## Task 3: Extract `editing.py` — NotesModal, TagsModal, AutoTagSuggestModal

**Files:**
- Modify: `src/arxiv_browser/app.py` (remove classes)
- Create: `src/arxiv_browser/modals/editing.py`
- Modify: `tests/test_arxiv_browser.py` (update imports)

**What to move:**
- `NotesModal` (lines 1475–1555)
- `TagsModal` (lines 1557–1693)
- `AutoTagSuggestModal` (lines 2930–3031)

**Step 1: Write `editing.py`**

```python
"""Paper metadata editing modals — notes, tags, auto-tag suggestions."""

from __future__ import annotations

import logging

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, TextArea

from arxiv_browser.themes import get_tag_color, parse_tag_namespace

logger = logging.getLogger(__name__)
```

Paste `NotesModal`, `TagsModal`, and `AutoTagSuggestModal` class bodies verbatim.

**Step 2: Remove from `app.py` and add import**

Delete the three class definitions from `app.py`. Add to the modals import block:

```python
from arxiv_browser.modals import (
    AutoTagSuggestModal,
    NotesModal,
    TagsModal,
)
```

**Step 3: Update test imports**

Change these test imports from `arxiv_browser.app` to `arxiv_browser.modals`:
- `TagsModal` — lines 6431, 6437, 6449, 6460, 8732
- `NotesModal` — lines 8661, 8679

**Step 4: Run tests + lint + typecheck**

```bash
just test-quick && just lint && just typecheck
```

**Step 5: Commit**

```bash
git add src/arxiv_browser/modals/editing.py src/arxiv_browser/app.py tests/test_arxiv_browser.py
git commit -m "refactor: extract editing modals to modals/editing.py"
```

---

## Task 4: Extract `search.py` — ArxivSearchModal, CommandPaletteModal

**Files:**
- Modify: `src/arxiv_browser/app.py` (remove classes)
- Create: `src/arxiv_browser/modals/search.py`

**What to move:**
- `ArxivSearchModal` (lines 1913–2048)
- `CommandPaletteModal` (lines 3329–3425)

**Step 1: Write `search.py`**

```python
"""Search and command palette modals."""

from __future__ import annotations

import logging

from rapidfuzz import fuzz
from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from arxiv_browser.models import ArxivSearchRequest
from arxiv_browser.parsing import ARXIV_QUERY_FIELDS, build_arxiv_search_query
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)
```

Paste `ArxivSearchModal` and `CommandPaletteModal` class bodies. Note: `CommandPaletteModal` references `COMMAND_PALETTE_COMMANDS` which stays in `app.py`. The modal receives commands via `__init__` parameter, so check how it accesses the commands list. If it references the module-level constant directly, it needs to accept it as a constructor parameter instead, or import it from `app.py` — but we cannot import from `app.py` (circular dependency).

**Important:** Check `CommandPaletteModal.__init__` to see if `COMMAND_PALETTE_COMMANDS` is passed as an argument or accessed as a global. If it's a global, it must be passed as a constructor argument. Read the class carefully before extracting.

**Step 2: Remove from `app.py` and add import**

**Step 3: Run tests + lint + typecheck**

```bash
just test-quick && just lint && just typecheck
```

No test imports to update — `CommandPaletteModal` and `ArxivSearchModal` aren't directly imported in tests (except `COMMAND_PALETTE_COMMANDS` which stays in `app.py`).

**Step 4: Commit**

```bash
git add src/arxiv_browser/modals/search.py src/arxiv_browser/app.py
git commit -m "refactor: extract search modals to modals/search.py"
```

---

## Task 5: Extract `collections.py` — CollectionsModal, CollectionViewModal, AddToCollectionModal

**Files:**
- Modify: `src/arxiv_browser/app.py` (remove classes)
- Create: `src/arxiv_browser/modals/collections.py`

**What to move:**
- `CollectionsModal` (lines 3461–3701)
- `CollectionViewModal` (lines 3703–3806)
- `AddToCollectionModal` (lines 3808–3886)

**Step 1: Write `collections.py`**

```python
"""Paper collections (reading lists) modals."""

from __future__ import annotations

import logging
from datetime import datetime

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView

from arxiv_browser.models import MAX_COLLECTIONS, Paper, PaperCollection

logger = logging.getLogger(__name__)
```

Note: `CollectionsModal` calls `self.app.push_screen(CollectionViewModal(...))` — this works because `CollectionViewModal` is defined in the same file. No circular dependency.

**Step 2: Remove from `app.py` and add import**

Also note: `ContextFooter` (line 3427) sits between `CommandPaletteModal` and `CollectionsModal`. `ContextFooter` is a widget, NOT a modal — it stays in `app.py`.

**Step 3: Run tests + lint + typecheck**

```bash
just test-quick && just lint && just typecheck
```

**Step 4: Commit**

```bash
git add src/arxiv_browser/modals/collections.py src/arxiv_browser/app.py
git commit -m "refactor: extract collection modals to modals/collections.py"
```

---

## Task 6: Extract `citations.py` — RecommendationSourceModal, RecommendationsScreen, CitationGraphScreen

**Files:**
- Modify: `src/arxiv_browser/app.py` (remove classes)
- Create: `src/arxiv_browser/modals/citations.py`

**What to move:**
- `RecommendationSourceModal` (lines 2050–2112)
- `RecommendationListItem` (lines 2114–2120)
- `RecommendationsScreen` (lines 2122–2252)
- `RECOMMENDATION_TITLE_MAX_LEN` constant (line 547)
- `CitationGraphListItem` (lines 2254–2266)
- `CitationGraphScreen` (lines 2269–2585)

**Step 1: Write `citations.py`**

```python
"""Recommendation and citation graph modals."""

from __future__ import annotations

import asyncio
import logging
import webbrowser

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView, Static

from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text, truncate_text
from arxiv_browser.semantic_scholar import CitationEntry
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)

RECOMMENDATION_TITLE_MAX_LEN = 60
```

Note: `CitationGraphScreen` uses `self.app._track_task()`. This is fine — Textual provides `self.app` to all widgets. The type checker may warn about `_track_task` not being on `App`; use `self.app._track_task(...)  # type: ignore[attr-defined]` if needed.

**Step 2: Remove from `app.py` and add import. Also remove `RECOMMENDATION_TITLE_MAX_LEN` from app.py line 547.**

**Step 3: Run tests + lint + typecheck**

```bash
just test-quick && just lint && just typecheck
```

**Step 4: Commit**

```bash
git add src/arxiv_browser/modals/citations.py src/arxiv_browser/app.py
git commit -m "refactor: extract citation modals to modals/citations.py"
```

---

## Task 7: Extract `llm.py` — SummaryModeModal, ResearchInterestsModal, PaperChatScreen

**Files:**
- Modify: `src/arxiv_browser/app.py` (remove classes)
- Create: `src/arxiv_browser/modals/llm.py`
- Modify: `tests/test_arxiv_browser.py` (update imports)

**What to move:**
- `SummaryModeModal` (lines 2760–2838)
- `ResearchInterestsModal` (lines 2840–2928)
- `PaperChatScreen` (lines 3033–3200)

**Step 1: Write `llm.py`**

```python
"""LLM-powered modals — summaries, relevance scoring, paper chat."""

from __future__ import annotations

import asyncio
import logging

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, TextArea

from arxiv_browser.llm import CHAT_SYSTEM_PROMPT, SUMMARY_MODES
from arxiv_browser.llm_providers import CLIProvider
from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)
```

Note: `PaperChatScreen` references `LLM_COMMAND_TIMEOUT` from `app.py`. Check if it's used directly as a module-level reference. If so, move it to `modals/llm.py` (it's only used by PaperChatScreen). If it's also used elsewhere in `app.py`, keep it in `app.py` and pass as constructor arg or import from a shared location.

**Step 2: Remove from `app.py` and add import**

**Step 3: Update test imports**

Change these from `arxiv_browser.app` to `arxiv_browser.modals`:
- `SummaryModeModal` — lines 6156, 6169, 9856, 9866, 9876, 9886, 9896, 9906, 9916, 9927
- `ResearchInterestsModal` — lines 6925, 6931, 6937, 9959, 9967, 9973
- `PaperChatScreen` — lines 11382, 11396, 11450, 11571

For line 9927 (`from arxiv_browser.app import SUMMARY_MODES, SummaryModeModal`): split into two imports — `SUMMARY_MODES` stays from `arxiv_browser.app` (or `arxiv_browser.llm`), `SummaryModeModal` from `arxiv_browser.modals`.

**Step 4: Run tests + lint + typecheck**

```bash
just test-quick && just lint && just typecheck
```

**Step 5: Commit**

```bash
git add src/arxiv_browser/modals/llm.py src/arxiv_browser/app.py tests/test_arxiv_browser.py
git commit -m "refactor: extract LLM modals to modals/llm.py"
```

---

## Task 8: Consolidate `app.py` Imports and Clean Up `__all__`

**Files:**
- Modify: `src/arxiv_browser/app.py`

**Step 1: Consolidate the temporary per-task modal imports into one block**

Replace the 6 separate `from arxiv_browser.modals import (...)` blocks added during tasks 2–7 with a single consolidated import:

```python
from arxiv_browser.modals import (
    AddToCollectionModal,
    AutoTagSuggestModal,
    ArxivSearchModal,
    CitationGraphListItem,
    CitationGraphScreen,
    CollectionViewModal,
    CollectionsModal,
    CommandPaletteModal,
    ConfirmModal,
    ExportMenuModal,
    HelpScreen,
    NotesModal,
    PaperChatScreen,
    RecommendationListItem,
    RecommendationSourceModal,
    RecommendationsScreen,
    ResearchInterestsModal,
    SectionToggleModal,
    SummaryModeModal,
    TagsModal,
    WatchListItem,
    WatchListModal,
)
```

**Step 2: Add `modals` re-export to `__all__`**

Add the modal names to `app.py`'s `__all__` list so they remain accessible via `from arxiv_browser import SomeModal`. The modal names that are already in `__all__` (like `AddToCollectionModal`, `CollectionsModal`, etc.) should stay. No new entries needed — they were already there from the re-export pattern.

**Step 3: Update `__init__.py` (package root)**

Check `src/arxiv_browser/__init__.py`. It re-exports from `app.py` via `*` import. Since `app.py` re-imports all modals and includes them in `__all__`, no changes to `__init__.py` should be needed. Verify this.

**Step 4: Run full quality suite**

```bash
just quality
```

This runs lint, typecheck, tests with coverage, complexity, security, dead code, and dependency checks. Everything must pass.

**Step 5: Commit**

```bash
git add src/arxiv_browser/app.py
git commit -m "refactor: consolidate modal imports in app.py"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `CLAUDE.md` (update Architecture section)
- Modify: `docs/plans/2026-02-13-modal-extraction-design.md` (mark as completed)

**Step 1: Update `CLAUDE.md` Architecture section**

Add `modals/` to the package layout:

```
src/arxiv_browser/
├── ...
├── modals/               # Modal dialogs (extracted from app.py)
│   ├── __init__.py       # Re-exports all modal classes
│   ├── common.py         # HelpScreen, ConfirmModal, ExportMenuModal, SectionToggleModal, WatchListModal
│   ├── editing.py        # NotesModal, TagsModal, AutoTagSuggestModal
│   ├── search.py         # ArxivSearchModal, CommandPaletteModal
│   ├── collections.py    # CollectionsModal, CollectionViewModal, AddToCollectionModal
│   ├── citations.py      # RecommendationsScreen, RecommendationSourceModal, CitationGraphScreen
│   └── llm.py            # SummaryModeModal, ResearchInterestsModal, PaperChatScreen
├── ...
```

Update the dependency DAG to include `modals/`. Update `app.py` line count (~5,800). Update the "UI Components" subsection to note modals are in `modals/`.

Update the Import Patterns section to add:

```
- **Modals**: `from arxiv_browser.modals import TagsModal` — canonical import path for all modals
```

**Step 2: Mark design doc as completed**

Change status in `docs/plans/2026-02-13-modal-extraction-design.md` from `Approved` to `Completed`.

**Step 3: Mark improvement ideas item #1 as DONE**

In `docs/improvement-ideas.md`, update item #1 title to `~~Extract Modals into a `modals/` Package~~ DONE` (matching the pattern of items 4–12).

**Step 4: Commit**

```bash
git add CLAUDE.md docs/plans/2026-02-13-modal-extraction-design.md docs/improvement-ideas.md
git commit -m "docs: update architecture docs after modal extraction"
```

---

## Task 10: Final Verification

**Step 1: Run the full quality suite one final time**

```bash
just quality
```

All checks must pass: lint, typecheck, tests, complexity, security, dead code, deps.

**Step 2: Verify line count reduction**

```bash
wc -l src/arxiv_browser/app.py
```

Expected: ~5,800 lines (down from 8,325).

```bash
wc -l src/arxiv_browser/modals/*.py
```

Expected: ~2,600 lines total across 7 files.

**Step 3: Verify no circular imports**

```bash
python -c "from arxiv_browser.modals import *; print('OK')"
python -c "from arxiv_browser.app import ArxivBrowser; print('OK')"
python -c "from arxiv_browser import main; print('OK')"
```

All should print `OK` without ImportError.

**Step 4: Verify the app launches**

```bash
uv run arxiv-viewer --help
```

Should print CLI help without errors.

---

## Execution Order Summary

| Task | Description | Est. Lines Changed |
|------|-------------|-------------------|
| 1 | Package skeleton | ~80 new |
| 2 | Extract `common.py` (5 modals) | ~530 moved, ~60 test imports |
| 3 | Extract `editing.py` (3 modals) | ~430 moved, ~7 test imports |
| 4 | Extract `search.py` (2 modals) | ~280 moved, ~0 test imports |
| 5 | Extract `collections.py` (3 modals) | ~450 moved, ~0 test imports |
| 6 | Extract `citations.py` (3 modals + 2 ListItems) | ~560 moved, ~0 test imports |
| 7 | Extract `llm.py` (3 modals) | ~340 moved, ~20 test imports |
| 8 | Consolidate imports + quality check | ~30 changed |
| 9 | Update docs | ~50 changed |
| 10 | Final verification | 0 changed |

## Key Gotchas

1. **`COMMAND_PALETTE_COMMANDS`** (line 475 in app.py): This is a list of tuples that `CommandPaletteModal` receives. Check whether it's passed via `__init__` or accessed as a module-level global. If global, refactor to accept as constructor argument to avoid circular import.

2. **`LLM_COMMAND_TIMEOUT`** (line 567): Used by `PaperChatScreen`. Check if also used in `ArxivBrowser`. If only in PaperChatScreen, move to `modals/llm.py`. If shared, keep in `app.py` and pass as constructor arg.

3. **`SUMMARY_MODES`** (from `llm.py` module, not `app.py`): `SummaryModeModal` references this. Since it's in the `llm` module (not `app.py`), the modal can import it directly — no circular dependency.

4. **`ContextFooter`** (line 3427): This is a widget, not a modal. It sits between `CommandPaletteModal` and `CollectionsModal` in the file. Do NOT move it — it stays in `app.py`.

5. **Test imports with multiple names**: Lines like `from arxiv_browser.app import BATCH_CONFIRM_THRESHOLD, ArxivBrowser, ConfirmModal` need to be split: constants and `ArxivBrowser` stay from `arxiv_browser.app`, modals come from `arxiv_browser.modals`.

6. **`self.app._track_task()`**: Used in `CitationGraphScreen` and `PaperChatScreen`. This works via Textual's runtime `self.app` reference. Add `# type: ignore[attr-defined]` if pyright complains, since `_track_task` is on `ArxivBrowser` not the base `App`.
