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
    RecommendationsScreen,
)

# collections.py — paper collections (reading lists) modal
from arxiv_browser.modals.collections import CollectionsModal

# common.py — general-purpose dialogs
from arxiv_browser.modals.common import (
    ConfirmModal,
    ExportMenuModal,
    MetadataSnapshotPickerModal,
    SectionToggleModal,
)

# editing.py — unified paper editing modal (notes + tags + auto-tag)
from arxiv_browser.modals.editing import (
    PaperEditModal,
    PaperEditResult,
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
    "ArxivSearchModal",
    "CitationGraphListItem",
    "CitationGraphScreen",
    "CollectionsModal",
    "CommandPaletteModal",
    "ConfirmModal",
    "ExportMenuModal",
    "HelpScreen",
    "MetadataSnapshotPickerModal",
    "ModalBase",
    "PaletteCommand",
    "PaperChatScreen",
    "PaperEditModal",
    "PaperEditResult",
    "RecommendationListItem",
    "RecommendationsScreen",
    "ResearchInterestsModal",
    "SectionToggleModal",
    "SummaryModeModal",
    "WatchListItem",
    "WatchListModal",
    "WelcomeScreen",
]
