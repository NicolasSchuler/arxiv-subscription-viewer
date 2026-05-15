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

# discovery.py — local analytics and author profiles
from arxiv_browser.modals.discovery import (
    AuthorListItem,
    AuthorPickerModal,
    AuthorProfileModal,
    TrendRadarModal,
)

# editing.py — unified paper editing modal (notes + tags + auto-tag)
from arxiv_browser.modals.editing import (
    LineAnnotationModal,
    LineAnnotationResult,
    PaperEditModal,
    PaperEditResult,
)

# help.py — full-screen keyboard shortcut overlay
from arxiv_browser.modals.help import HelpScreen

# llm.py — LLM-powered modals (summaries, relevance, chat)
from arxiv_browser.modals.llm import (
    PaperChatScreen,
    PaperComparisonScreen,
    PaperDebateResultModal,
    PaperRemixResultModal,
    ResearchInterestsModal,
    SummaryModeModal,
)

# pdf.py — terminal PDF/figure preview
from arxiv_browser.modals.pdf import FigurePreviewScreen, PdfPreviewScreen

# search.py — arXiv search form, command palette
from arxiv_browser.modals.search import (
    ArxivSearchModal,
    CommandPaletteModal,
    PaletteCommand,
)

# triage.py — rapid unread-paper triage
from arxiv_browser.modals.triage import QuickTriageScreen

# watchlist.py — watch list management modals
from arxiv_browser.modals.watchlist import WatchListItem, WatchListModal

# welcome.py — first-run tutorial overlay
from arxiv_browser.modals.welcome import WelcomeScreen

# whats_new.py — version-bump changelog overlay
from arxiv_browser.modals.whats_new import WhatsNewScreen

__all__ = [
    "ArxivSearchModal",
    "AuthorListItem",
    "AuthorPickerModal",
    "AuthorProfileModal",
    "CitationGraphListItem",
    "CitationGraphScreen",
    "CollectionsModal",
    "CommandPaletteModal",
    "ConfirmModal",
    "ExportMenuModal",
    "FigurePreviewScreen",
    "HelpScreen",
    "LineAnnotationModal",
    "LineAnnotationResult",
    "MetadataSnapshotPickerModal",
    "ModalBase",
    "PaletteCommand",
    "PaperChatScreen",
    "PaperComparisonScreen",
    "PaperDebateResultModal",
    "PaperEditModal",
    "PaperEditResult",
    "PaperRemixResultModal",
    "PdfPreviewScreen",
    "QuickTriageScreen",
    "RecommendationListItem",
    "RecommendationsScreen",
    "ResearchInterestsModal",
    "SectionToggleModal",
    "SummaryModeModal",
    "TrendRadarModal",
    "WatchListItem",
    "WatchListModal",
    "WelcomeScreen",
    "WhatsNewScreen",
]
