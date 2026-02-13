"""Vulture whitelist for Textual framework false positives.

Textual uses string-based dispatch for action_* methods (via BINDINGS),
lifecycle hooks, event handlers (@on decorators), and compose() methods.
Vulture can't trace these, so we declare them here.
"""

# ── ArxivBrowser (App) ────────────────────────────────────────────────
from arxiv_browser.app import ArxivBrowser

ArxivBrowser.TITLE
ArxivBrowser.CSS
ArxivBrowser.BINDINGS
ArxivBrowser.compose
ArxivBrowser.on_mount
ArxivBrowser.on_unmount
ArxivBrowser.on_key
ArxivBrowser.on_search_submitted
ArxivBrowser.on_search_changed
ArxivBrowser.on_date_jump
ArxivBrowser.on_date_navigate
ArxivBrowser.on_remove_filter
ArxivBrowser.on_remove_watch_filter
ArxivBrowser.on_paper_selected
ArxivBrowser.on_paper_highlighted
ArxivBrowser._match_query_term
ArxivBrowser._mark_badges_dirty
ArxivBrowser._flush_badge_refresh
ArxivBrowser.is_paper_watched
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
ArxivBrowser.action_export_metadata
ArxivBrowser.action_import_metadata
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
ArxivBrowser.action_cycle_theme
ArxivBrowser.action_toggle_sections
ArxivBrowser.action_command_palette
ArxivBrowser.action_collections
ArxivBrowser.action_add_to_collection
ArxivBrowser.action_chat_with_paper
ArxivBrowser.action_auto_tag

# ── App-level constants ──────────────────────────────────────────────
from arxiv_browser.app import MIN_LIST_WIDTH, MAX_LIST_WIDTH, MAX_HISTORY_FILES

MIN_LIST_WIDTH
MAX_LIST_WIDTH
MAX_HISTORY_FILES

# ── PaperListItem (ListItem) ─────────────────────────────────────────
from arxiv_browser.app import PaperListItem

PaperListItem.compose
PaperListItem.is_selected
PaperListItem.set_metadata
PaperListItem.set_abstract_text
PaperListItem.update_s2_data
PaperListItem.update_hf_data
PaperListItem.update_version_data
PaperListItem.update_relevance_data
PaperListItem.toggle_selected
PaperListItem.set_selected

# ── PaperDetails (Static) ────────────────────────────────────────────
from arxiv_browser.app import PaperDetails

PaperDetails.on_mount
PaperDetails.clear_cache

# ── BookmarkTabBar (Horizontal) ───────────────────────────────────────
from arxiv_browser.app import BookmarkTabBar

BookmarkTabBar.DEFAULT_CSS
BookmarkTabBar.compose

# ── DateNavigator (Horizontal) ──────────────────────────────────────
from arxiv_browser.app import DateNavigator

DateNavigator.DEFAULT_CSS
DateNavigator.compose
DateNavigator.on_click
DateNavigator._current_index

# ── ContextFooter (Static) ──────────────────────────────────────────
from arxiv_browser.app import ContextFooter

ContextFooter.DEFAULT_CSS

# ── FilterPillBar (Horizontal) ──────────────────────────────────────
from arxiv_browser.app import FilterPillBar

FilterPillBar.DEFAULT_CSS

# ── HelpScreen ──────────────────────────────────────────────────────
from arxiv_browser.modals.common import HelpScreen

HelpScreen.BINDINGS
HelpScreen.CSS
HelpScreen.compose
HelpScreen.action_dismiss

# ── ConfirmModal ────────────────────────────────────────────────────
from arxiv_browser.modals.common import ConfirmModal

ConfirmModal.BINDINGS
ConfirmModal.CSS
ConfirmModal.compose
ConfirmModal.action_confirm
ConfirmModal.action_cancel
ConfirmModal.on_yes
ConfirmModal.on_no

# ── ExportMenuModal ─────────────────────────────────────────────────
from arxiv_browser.modals.common import ExportMenuModal

ExportMenuModal.BINDINGS
ExportMenuModal.CSS
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

# ── WatchListItem ───────────────────────────────────────────────────
from arxiv_browser.modals.common import WatchListItem

WatchListItem.compose

# ── WatchListModal ──────────────────────────────────────────────────
from arxiv_browser.modals.common import WatchListModal

WatchListModal.BINDINGS
WatchListModal.CSS
WatchListModal.compose
WatchListModal.on_mount
WatchListModal.action_save
WatchListModal.action_cancel
WatchListModal.on_list_highlighted
WatchListModal.on_add_pressed
WatchListModal.on_update_pressed
WatchListModal.on_delete_pressed
WatchListModal.on_save_pressed
WatchListModal.on_cancel_pressed

# ── SectionToggleModal ──────────────────────────────────────────────
from arxiv_browser.modals.common import SectionToggleModal

SectionToggleModal.BINDINGS
SectionToggleModal.CSS
SectionToggleModal.compose
SectionToggleModal.action_toggle_a
SectionToggleModal.action_toggle_b
SectionToggleModal.action_toggle_t
SectionToggleModal.action_toggle_r
SectionToggleModal.action_toggle_s
SectionToggleModal.action_toggle_e
SectionToggleModal.action_toggle_h
SectionToggleModal.action_toggle_v
SectionToggleModal.action_save
SectionToggleModal.action_cancel

# ── NotesModal ──────────────────────────────────────────────────────
from arxiv_browser.modals.editing import NotesModal

NotesModal.BINDINGS
NotesModal.CSS
NotesModal.compose
NotesModal.action_save
NotesModal.action_cancel
NotesModal.on_save_pressed
NotesModal.on_cancel_pressed

# ── TagsModal ───────────────────────────────────────────────────────
from arxiv_browser.modals.editing import TagsModal

TagsModal.BINDINGS
TagsModal.CSS
TagsModal.compose
TagsModal.action_save
TagsModal.action_cancel
TagsModal.on_save_pressed
TagsModal.on_cancel_pressed
TagsModal.on_input_submitted

# ── AutoTagSuggestModal ─────────────────────────────────────────────
from arxiv_browser.modals.editing import AutoTagSuggestModal

AutoTagSuggestModal.BINDINGS
AutoTagSuggestModal.CSS
AutoTagSuggestModal.compose
AutoTagSuggestModal.action_accept
AutoTagSuggestModal.action_cancel
AutoTagSuggestModal.on_accept_pressed
AutoTagSuggestModal.on_cancel_pressed

# ── ArxivSearchModal ────────────────────────────────────────────────
from arxiv_browser.modals.search import ArxivSearchModal

ArxivSearchModal.BINDINGS
ArxivSearchModal.CSS
ArxivSearchModal.compose
ArxivSearchModal.action_search
ArxivSearchModal.action_cancel
ArxivSearchModal.on_search_pressed
ArxivSearchModal.on_cancel_pressed
ArxivSearchModal.on_query_submitted
ArxivSearchModal.on_category_submitted

# ── CommandPaletteModal ─────────────────────────────────────────────
from arxiv_browser.modals.search import CommandPaletteModal

CommandPaletteModal.BINDINGS
CommandPaletteModal.DEFAULT_CSS
CommandPaletteModal.compose
CommandPaletteModal.on_mount
CommandPaletteModal.action_cancel
CommandPaletteModal.key_enter
CommandPaletteModal._on_search_changed
CommandPaletteModal._on_option_selected

# ── CollectionsModal ────────────────────────────────────────────────
from arxiv_browser.modals.collections import CollectionsModal

CollectionsModal.BINDINGS
CollectionsModal.CSS
CollectionsModal.compose
CollectionsModal.on_mount
CollectionsModal.action_cancel
CollectionsModal.on_list_highlighted
CollectionsModal.on_create_pressed
CollectionsModal.on_rename_pressed
CollectionsModal.on_delete_pressed
CollectionsModal.on_view_pressed
CollectionsModal.on_save_pressed
CollectionsModal.on_close_pressed

# ── CollectionViewModal ─────────────────────────────────────────────
from arxiv_browser.modals.collections import CollectionViewModal

CollectionViewModal.BINDINGS
CollectionViewModal.CSS
CollectionViewModal.compose
CollectionViewModal.on_mount
CollectionViewModal.action_cancel
CollectionViewModal.on_remove_pressed
CollectionViewModal.on_done_pressed

# ── AddToCollectionModal ────────────────────────────────────────────
from arxiv_browser.modals.collections import AddToCollectionModal

AddToCollectionModal.BINDINGS
AddToCollectionModal.CSS
AddToCollectionModal.compose
AddToCollectionModal.on_mount
AddToCollectionModal.action_cancel
AddToCollectionModal.on_list_selected
AddToCollectionModal.on_cancel_pressed

# ── RecommendationSourceModal ───────────────────────────────────────
from arxiv_browser.modals.citations import RecommendationSourceModal

RecommendationSourceModal.BINDINGS
RecommendationSourceModal.CSS
RecommendationSourceModal.compose
RecommendationSourceModal.action_local
RecommendationSourceModal.action_s2
RecommendationSourceModal.on_local_pressed
RecommendationSourceModal.on_s2_pressed

# ── RecommendationListItem ──────────────────────────────────────────
from arxiv_browser.modals.citations import RecommendationListItem

RecommendationListItem.compose

# ── RecommendationsScreen ───────────────────────────────────────────
from arxiv_browser.modals.citations import RecommendationsScreen

RecommendationsScreen.BINDINGS
RecommendationsScreen.CSS
RecommendationsScreen.compose
RecommendationsScreen.on_mount
RecommendationsScreen.action_select
RecommendationsScreen.action_dismiss
RecommendationsScreen.action_cursor_down
RecommendationsScreen.action_cursor_up
RecommendationsScreen.on_close_pressed
RecommendationsScreen.on_select_pressed
RecommendationsScreen.on_list_selected

# ── CitationGraphListItem ──────────────────────────────────────────
from arxiv_browser.modals.citations import CitationGraphListItem

CitationGraphListItem.compose

# ── CitationGraphScreen ────────────────────────────────────────────
from arxiv_browser.modals.citations import CitationGraphScreen

CitationGraphScreen.BINDINGS
CitationGraphScreen.CSS
CitationGraphScreen.compose
CitationGraphScreen.on_mount
CitationGraphScreen._root_title
CitationGraphScreen._root_paper_id
CitationGraphScreen.action_dismiss
CitationGraphScreen.action_back_or_close
CitationGraphScreen.action_switch_panel
CitationGraphScreen.action_open_url
CitationGraphScreen.action_go_to_local
CitationGraphScreen.action_drill_down
CitationGraphScreen.action_cursor_down
CitationGraphScreen.action_cursor_up
CitationGraphScreen.on_close_pressed
CitationGraphScreen.on_drill_pressed

# ── SummaryModeModal ────────────────────────────────────────────────
from arxiv_browser.modals.llm import SummaryModeModal

SummaryModeModal.BINDINGS
SummaryModeModal.CSS
SummaryModeModal.compose
SummaryModeModal.action_cancel
SummaryModeModal.action_mode_default
SummaryModeModal.action_mode_quick
SummaryModeModal.action_mode_tldr
SummaryModeModal.action_mode_methods
SummaryModeModal.action_mode_results
SummaryModeModal.action_mode_comparison

# ── ResearchInterestsModal ──────────────────────────────────────────
from arxiv_browser.modals.llm import ResearchInterestsModal

ResearchInterestsModal.BINDINGS
ResearchInterestsModal.CSS
ResearchInterestsModal.compose
ResearchInterestsModal.action_save
ResearchInterestsModal.action_cancel
ResearchInterestsModal.on_save_pressed
ResearchInterestsModal.on_cancel_pressed

# ── PaperChatScreen ─────────────────────────────────────────────────
from arxiv_browser.modals.llm import PaperChatScreen

PaperChatScreen.BINDINGS
PaperChatScreen.CSS
PaperChatScreen.compose
PaperChatScreen.on_mount
PaperChatScreen.action_close
PaperChatScreen.on_question_submitted

# ── HTMLParser subclasses (parsing.py) ──────────────────────────────
from arxiv_browser.parsing import ArxivVersionParser

ArxivVersionParser.handle_starttag
ArxivVersionParser.handle_endtag
ArxivVersionParser.handle_data

# ── LLM internal functions ──────────────────────────────────────────
from arxiv_browser.llm import _load_relevance_score

_load_relevance_score
