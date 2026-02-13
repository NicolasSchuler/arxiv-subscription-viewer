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

# Future extractions (Tasks 3-7) will add imports here:
# - editing.py: NotesModal, TagsModal, AutoTagSuggestModal
# - search.py: ArxivSearchModal, CommandPaletteModal
# - collections.py: CollectionsModal, CollectionViewModal, AddToCollectionModal
# - citations.py: CitationGraphScreen, RecommendationsScreen, etc.
# - llm.py: SummaryModeModal, ResearchInterestsModal, PaperChatScreen

__all__ = [
    # common
    "ConfirmModal",
    "ExportMenuModal",
    "HelpScreen",
    "SectionToggleModal",
    "WatchListItem",
    "WatchListModal",
]
