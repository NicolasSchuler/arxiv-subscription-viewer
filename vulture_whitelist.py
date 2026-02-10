"""Vulture whitelist for Textual framework false positives.

Textual uses string-based dispatch for action_* methods (via BINDINGS),
lifecycle hooks, event handlers, and compose() methods. Vulture can't
trace these, so we declare them here.
"""

# ── ArxivBrowser (App) ────────────────────────────────────────────────
from arxiv_browser import ArxivBrowser

ArxivBrowser.compose
ArxivBrowser.on_mount
ArxivBrowser.on_unmount
ArxivBrowser.on_key
ArxivBrowser.action_toggle_search
ArxivBrowser.action_cancel_search
ArxivBrowser.action_ctrl_e_dispatch
ArxivBrowser.action_toggle_s2
ArxivBrowser.action_exit_arxiv_search_mode
ArxivBrowser.action_arxiv_search
ArxivBrowser.action_toggle_select
ArxivBrowser.action_select_all
ArxivBrowser.action_clear_selection
ArxivBrowser.action_cycle_sort
ArxivBrowser.action_toggle_read
ArxivBrowser.action_toggle_star
ArxivBrowser.action_edit_notes
ArxivBrowser.action_edit_tags
ArxivBrowser.action_toggle_watch_filter
ArxivBrowser.action_manage_watch_list
ArxivBrowser.action_toggle_preview
ArxivBrowser.action_start_mark
ArxivBrowser.action_start_goto_mark
ArxivBrowser.action_copy_bibtex
ArxivBrowser.action_export_bibtex_file
ArxivBrowser.action_export_markdown
ArxivBrowser.action_export_menu
ArxivBrowser.action_show_similar
ArxivBrowser.action_citation_graph
ArxivBrowser.action_show_help
ArxivBrowser.action_generate_summary
ArxivBrowser.action_score_relevance
ArxivBrowser.action_edit_interests
ArxivBrowser.action_prev_date
ArxivBrowser.action_next_date
ArxivBrowser.action_open_pdf
ArxivBrowser.action_download_pdf
ArxivBrowser.action_copy_selected
ArxivBrowser.action_fetch_s2
ArxivBrowser.action_toggle_hf
ArxivBrowser.action_check_versions
ArxivBrowser.action_goto_bookmark
ArxivBrowser.action_add_bookmark
ArxivBrowser.action_remove_bookmark

# ── PaperListItem (ListItem) ─────────────────────────────────────────
from arxiv_browser import PaperListItem

PaperListItem.compose

# ── PaperDetails (Static) ────────────────────────────────────────────
from arxiv_browser import PaperDetails

PaperDetails.on_mount

# ── HelpScreen (ModalScreen) ─────────────────────────────────────────
from arxiv_browser import HelpScreen

HelpScreen.compose
HelpScreen.action_dismiss

# ── NotesModal (ModalScreen) ─────────────────────────────────────────
from arxiv_browser import NotesModal

NotesModal.compose
NotesModal.action_save
NotesModal.action_cancel

# ── TagsModal (ModalScreen) ──────────────────────────────────────────
from arxiv_browser import TagsModal

TagsModal.compose
TagsModal.action_save
TagsModal.action_cancel

# ── WatchListItem (ListItem) ─────────────────────────────────────────
from arxiv_browser import WatchListItem

WatchListItem.compose

# ── WatchListModal (ModalScreen) ──────────────────────────────────────
from arxiv_browser import WatchListModal

WatchListModal.compose
WatchListModal.on_mount
WatchListModal.action_save
WatchListModal.action_cancel

# ── ArxivSearchModal (ModalScreen) ────────────────────────────────────
from arxiv_browser import ArxivSearchModal

ArxivSearchModal.compose
ArxivSearchModal.action_search
ArxivSearchModal.action_cancel

# ── RecommendationSourceModal (ModalScreen) ───────────────────────────
from arxiv_browser import RecommendationSourceModal

RecommendationSourceModal.compose
RecommendationSourceModal.action_local
RecommendationSourceModal.action_s2

# ── RecommendationListItem (ListItem) ─────────────────────────────────
from arxiv_browser import RecommendationListItem

RecommendationListItem.compose

# ── RecommendationsScreen (ModalScreen) ───────────────────────────────
from arxiv_browser import RecommendationsScreen

RecommendationsScreen.compose
RecommendationsScreen.on_mount
RecommendationsScreen.action_select
RecommendationsScreen.action_dismiss
RecommendationsScreen.action_cursor_down
RecommendationsScreen.action_cursor_up

# ── CitationGraphListItem (ListItem) ──────────────────────────────────
from arxiv_browser import CitationGraphListItem

CitationGraphListItem.compose

# ── CitationGraphScreen (ModalScreen) ─────────────────────────────────
from arxiv_browser import CitationGraphScreen

CitationGraphScreen.compose
CitationGraphScreen.on_mount
CitationGraphScreen.action_dismiss
CitationGraphScreen.action_back_or_close
CitationGraphScreen.action_switch_panel
CitationGraphScreen.action_open_url
CitationGraphScreen.action_go_to_local
CitationGraphScreen.action_drill_down
CitationGraphScreen.action_cursor_down
CitationGraphScreen.action_cursor_up

# ── ConfirmModal (ModalScreen) ────────────────────────────────────────
from arxiv_browser import ConfirmModal

ConfirmModal.compose
ConfirmModal.action_confirm
ConfirmModal.action_cancel

# ── ExportMenuModal (ModalScreen) ─────────────────────────────────────
from arxiv_browser import ExportMenuModal

ExportMenuModal.compose
ExportMenuModal.action_do_clipboard_plain
ExportMenuModal.action_do_clipboard_bibtex
ExportMenuModal.action_do_clipboard_markdown
ExportMenuModal.action_do_clipboard_ris
ExportMenuModal.action_do_clipboard_csv
ExportMenuModal.action_do_clipboard_mdtable
ExportMenuModal.action_do_file_bibtex
ExportMenuModal.action_do_file_ris
ExportMenuModal.action_do_file_csv
ExportMenuModal.action_dismiss

# ── SummaryModeModal (ModalScreen) ────────────────────────────────────
from arxiv_browser import SummaryModeModal

SummaryModeModal.compose
SummaryModeModal.action_mode_default
SummaryModeModal.action_mode_tldr
SummaryModeModal.action_mode_methods
SummaryModeModal.action_mode_results
SummaryModeModal.action_mode_comparison

# ── ResearchInterestsModal (ModalScreen) ──────────────────────────────
from arxiv_browser import ResearchInterestsModal

ResearchInterestsModal.compose
ResearchInterestsModal.action_save
ResearchInterestsModal.action_cancel

# ── BookmarkTabBar (Horizontal) ───────────────────────────────────────
from arxiv_browser import BookmarkTabBar

BookmarkTabBar.compose
