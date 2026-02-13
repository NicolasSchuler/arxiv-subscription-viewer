"""Modal dialogs for the arXiv Browser TUI.

Domain-grouped ModalScreen subclasses extracted from app.py.
Import modals from this package: ``from arxiv_browser.modals import HelpScreen``
"""

# collections.py — paper collections (reading lists) modals
from arxiv_browser.modals.collections import (
    AddToCollectionModal,
    CollectionsModal,
    CollectionViewModal,
)

# common.py — general-purpose dialogs
from arxiv_browser.modals.common import (
    ConfirmModal,
    ExportMenuModal,
    HelpScreen,
    SectionToggleModal,
    WatchListItem,
    WatchListModal,
)

# editing.py — notes, tags, auto-tag suggestions
from arxiv_browser.modals.editing import (
    AutoTagSuggestModal,
    NotesModal,
    TagsModal,
)

# search.py — arXiv search form, command palette
from arxiv_browser.modals.search import (
    ArxivSearchModal,
    CommandPaletteModal,
)

# Future extractions (Tasks 6-7) will add imports here:
# - citations.py: CitationGraphScreen, RecommendationsScreen, etc.
# - llm.py: SummaryModeModal, ResearchInterestsModal, PaperChatScreen

__all__ = [
    "AddToCollectionModal",
    "ArxivSearchModal",
    "AutoTagSuggestModal",
    "CollectionViewModal",
    "CollectionsModal",
    "CommandPaletteModal",
    "ConfirmModal",
    "ExportMenuModal",
    "HelpScreen",
    "NotesModal",
    "SectionToggleModal",
    "TagsModal",
    "WatchListItem",
    "WatchListModal",
]
