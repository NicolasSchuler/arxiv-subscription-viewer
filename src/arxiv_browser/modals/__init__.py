"""Modal dialogs for the arXiv Browser TUI.

Domain-grouped ModalScreen subclasses extracted from app.py.
Import modals from this package: ``from arxiv_browser.modals import HelpScreen``
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

# Future extractions (Tasks 5-7) will add imports here:
# - collections.py: CollectionsModal, CollectionViewModal, AddToCollectionModal
# - citations.py: CitationGraphScreen, RecommendationsScreen, etc.
# - llm.py: SummaryModeModal, ResearchInterestsModal, PaperChatScreen

__all__ = [
    "ArxivSearchModal",
    "AutoTagSuggestModal",
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
