"""Modal dialogs for the arXiv Browser TUI.

Domain-grouped ModalScreen subclasses extracted from app.py.
Import modals from this package: ``from arxiv_browser.modals import HelpScreen``
"""

# citations.py — recommendation and citation graph modals
from arxiv_browser.modals.citations import (
    CitationGraphListItem,
    CitationGraphScreen,
    RecommendationListItem,
    RecommendationSourceModal,
    RecommendationsScreen,
)

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

# Future extractions (Task 7) will add imports here:
# - llm.py: SummaryModeModal, ResearchInterestsModal, PaperChatScreen

__all__ = [
    "AddToCollectionModal",
    "ArxivSearchModal",
    "AutoTagSuggestModal",
    "CitationGraphListItem",
    "CitationGraphScreen",
    "CollectionViewModal",
    "CollectionsModal",
    "CommandPaletteModal",
    "ConfirmModal",
    "ExportMenuModal",
    "HelpScreen",
    "NotesModal",
    "RecommendationListItem",
    "RecommendationSourceModal",
    "RecommendationsScreen",
    "SectionToggleModal",
    "TagsModal",
    "WatchListItem",
    "WatchListModal",
]
