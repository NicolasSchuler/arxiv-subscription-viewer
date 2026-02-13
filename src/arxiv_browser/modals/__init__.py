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
