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

# llm.py — LLM-powered modals (summaries, relevance, chat)
from arxiv_browser.modals.llm import (
    PaperChatScreen,
    ResearchInterestsModal,
    SummaryModeModal,
)

# search.py — arXiv search form, command palette
from arxiv_browser.modals.search import (
    ArxivSearchModal,
    CommandPaletteModal,
)

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
    "PaperChatScreen",
    "RecommendationListItem",
    "RecommendationSourceModal",
    "RecommendationsScreen",
    "ResearchInterestsModal",
    "SectionToggleModal",
    "SummaryModeModal",
    "TagsModal",
    "WatchListItem",
    "WatchListModal",
]
