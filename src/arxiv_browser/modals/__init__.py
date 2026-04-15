"""Modal dialogs for the arXiv Browser TUI.

Domain-grouped ModalScreen subclasses extracted from app.py.
Import modals from this package: ``from arxiv_browser.modals import HelpScreen``
"""

# base.py — shared base class for all modals
from arxiv_browser.modals.base import ModalBase

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
    MetadataSnapshotPickerModal,
    SectionToggleModal,
)

# editing.py — notes, tags, auto-tag suggestions
from arxiv_browser.modals.editing import (
    AutoTagSuggestModal,
    NotesModal,
    TagsModal,
)

# help.py — full-screen keyboard shortcut overlay
from arxiv_browser.modals.help import HelpScreen

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
    PaletteCommand,
)

# watchlist.py — watch list management modals
from arxiv_browser.modals.watchlist import WatchListItem, WatchListModal

# welcome.py — first-run tutorial overlay
from arxiv_browser.modals.welcome import WelcomeScreen

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
    "MetadataSnapshotPickerModal",
    "ModalBase",
    "NotesModal",
    "PaletteCommand",
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
    "WelcomeScreen",
]
