# pyright: reportAttributeAccessIssue=false
"""Browse and dataset-state mixin for ArxivBrowser."""

from __future__ import annotations

import logging
import time
import webbrowser
from collections.abc import Callable
from datetime import date

import httpx
from textual.css.query import NoMatches
from textual.widgets.option_list import Option, OptionDoesNotExist

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.browser.constants import (
    FUZZY_LIMIT,
    FUZZY_SCORE_CUTOFF,
    PDF_DOWNLOAD_TIMEOUT,
    logger,
)
from arxiv_browser.empty_state import build_list_empty_message
from arxiv_browser.export import format_paper_as_markdown
from arxiv_browser.fuzzy import weighted_fuzzy_score
from arxiv_browser.io_actions import resolve_target_papers
from arxiv_browser.modals.editing import PaperEditModal, PaperEditResult
from arxiv_browser.models import SORT_OPTIONS, LocalBrowseSnapshot, Paper, PaperMetadata
from arxiv_browser.parsing import build_daily_digest, parse_arxiv_file
from arxiv_browser.query import (
    _HIGHLIGHT_PATTERN_CACHE,
    QueryToken,
    apply_watch_filter,
    execute_query_filter,
    get_query_tokens,
    match_query_term,
    matches_advanced_query,
    paper_matches_watch_entry,
    sort_papers,
)
from arxiv_browser.widgets import render_paper_option


class BrowseMixin:
    """Dataset and filter state transitions shared by the main browser app."""

    def _capture_local_browse_snapshot(self) -> LocalBrowseSnapshot | None:
        """Capture the local-library view before switching into API search mode.
        The snapshot intentionally preserves everything needed to restore the
        user-visible browsing state later: current dataset, sort/filter state,
        selection, highlight terms, visible-list position, and subtitle text.
        """
        try:
            list_view = self._get_paper_list_widget()
        except NoMatches:
            return None
        search_query = self._get_live_query()
        if search_query.startswith(("@", ">")):
            search_query = self._get_active_query()
        return LocalBrowseSnapshot(
            all_papers=self.all_papers,
            papers_by_id=self._papers_by_id,
            selected_ids=set(self.selected_ids),
            sort_index=self._sort_index,
            search_query=search_query,
            pending_query=self._pending_query,
            applied_query=self._applied_query,
            watch_filter_active=self._watch_filter_active,
            active_bookmark_index=self._active_bookmark_index,
            list_index=list_view.highlighted if list_view.highlighted is not None else 0,
            sub_title=self.sub_title,
            highlight_terms={key: terms.copy() for key, terms in self._highlight_terms.items()},
            match_scores=dict(self._match_scores),
        )

    def _restore_local_browse_snapshot(self) -> None:
        """Restore the local-library view saved before API search mode.
        Restoring advances the dataset epoch first so stale background work from
        the temporary API dataset will not publish into the restored local
        dataset. After the core state is restored, this method recomputes watch
        matches, reapplies the saved query, and rebuilds the visible list focus
        so the UI returns to the same logical place the user left.
        """
        snapshot = self._local_browse_snapshot
        if snapshot is None:
            return
        self._advance_dataset_epoch()
        self.all_papers = snapshot.all_papers
        self._papers_by_id = snapshot.papers_by_id
        self.filtered_papers = self.all_papers.copy()
        self._reset_dataset_view_state()
        self.selected_ids = set(snapshot.selected_ids)
        self._sort_index = snapshot.sort_index
        self._pending_query = snapshot.pending_query
        self._applied_query = snapshot.applied_query
        self._watch_filter_active = snapshot.watch_filter_active
        self._active_bookmark_index = snapshot.active_bookmark_index
        self._highlight_terms = {
            key: terms.copy() for key, terms in snapshot.highlight_terms.items()
        }
        self._match_scores = dict(snapshot.match_scores)
        self.sub_title = snapshot.sub_title
        # Recompute watch matches for the restored local dataset
        self._compute_watched_papers()
        try:
            self._get_search_input_widget().value = snapshot.search_query
        except NoMatches:
            pass
        self._apply_filter(snapshot.search_query)
        try:
            option_list = self._get_paper_list_widget()
            if option_list.option_count > 0:
                max_index = max(0, option_list.option_count - 1)
                option_list.highlighted = min(max(0, snapshot.list_index), max_index)
                option_list.focus()
        except NoMatches:
            pass
        if self._config.bookmarks:
            self._track_task(self._update_bookmark_bar())

    def _debounced_filter(self) -> None:
        """Apply filter after debounce delay."""
        self._search_timer = None
        if getattr(self, "_shutting_down", False):
            return
        try:
            self._get_paper_list_widget()
        except NoMatches:
            return
        self._apply_filter(self._pending_query)

    def _get_active_query(self) -> str:
        """Get the filter query currently applied to the paper list."""
        return str(getattr(self, "_applied_query", "")).strip()

    def _get_live_query(self) -> str:
        """Get the current search text, even if debounce has not applied it yet."""
        try:
            return self._get_search_input_widget().value.strip()
        except (AttributeError, NoMatches):
            return str(getattr(self, "_pending_query", "")).strip()

    def _format_header_text(self, query: str = "") -> str:
        """Format the left-pane header text."""
        if query:
            return f" [bold]Papers[/] ({len(self.filtered_papers)}/{len(self.all_papers)})"
        return " [bold]Papers[/]"

    def _matches_advanced_query(self, paper: Paper, rpn: list[QueryToken]) -> bool:
        """Test whether a paper matches an advanced RPN query with metadata context."""
        metadata = self._config.paper_metadata.get(paper.arxiv_id)
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return matches_advanced_query(paper, rpn, metadata, abstract_text)

    def _match_query_term(self, paper: Paper, token: QueryToken) -> bool:
        """Test whether a paper matches a single query token with metadata context."""
        metadata = self._config.paper_metadata.get(paper.arxiv_id)
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return match_query_term(paper, token, metadata, abstract_text)

    def _fuzzy_search(self, query: str, papers: list[Paper] | None = None) -> list[Paper]:
        """Perform fuzzy search on title and authors.
        Populates self._match_scores with relevance scores.
        """
        query_lower = query.lower()
        scored_papers = []
        search_space = papers if papers is not None else self.all_papers
        for paper in search_space:
            # Combine title and authors for matching
            text = f"{paper.title} {paper.authors}"
            score = weighted_fuzzy_score(query_lower, text)
            if score >= FUZZY_SCORE_CUTOFF:
                scored_papers.append((paper, score))
        # Sort by score descending
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        top_papers = scored_papers[:FUZZY_LIMIT]
        # Store scores for display (optional enhancement)
        self._match_scores = {p.arxiv_id: s for p, s in top_papers}
        return [p for p, _ in top_papers]

    def _apply_filter(self, query: str) -> None:
        """Apply the current query and refresh all dependent dataset UI state.
        Query execution runs through the shared query engine, then intersects
        with the optional watch filter, reapplies the active sort order, and
        refreshes list/detail/bookmark UI surfaces. Keeping the whole sequence
        here ensures the visible dataset, highlight terms, status text, and
        bookmark state stay in sync after both local typing and API-mode
        restoration.
        """
        perf_start = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
        query = query.strip()
        # Keep status/empty-state context synchronized with the applied filter.
        self._pending_query = query
        self._applied_query = query
        # Clear match scores by default (only fuzzy search populates them)
        self._match_scores.clear()
        _HIGHLIGHT_PATTERN_CACHE.clear()
        self.filtered_papers, self._highlight_terms = execute_query_filter(
            query,
            self.all_papers,
            fuzzy_search=self._fuzzy_search,
            advanced_match=self._matches_advanced_query,
        )
        # Apply watch filter if active (intersects with other filters)
        self.filtered_papers = apply_watch_filter(
            self.filtered_papers, self._watched_paper_ids, self._watch_filter_active
        )
        # Apply current sort order and refresh UI
        self._sort_papers()
        self._get_ui_refresh_coordinator().apply_filter_refresh(query)
        self._update_subtitle()
        self._track_task(self._update_bookmark_bar())
        logger.debug(
            "Filter applied: query=%r, matched=%d/%d papers",
            query,
            len(self.filtered_papers),
            len(self.all_papers),
        )
        if perf_start is not None:
            logger.debug(
                "Search->list refresh latency: %.2fms (query=%r, matched=%d)",
                (time.perf_counter() - perf_start) * 1000.0,
                query,
                len(self.filtered_papers),
            )

    def _update_filter_pills(self, query: str) -> None:
        """Update the filter pill bar with current active filters."""
        if self._in_arxiv_api_mode:
            try:
                self._get_filter_pill_bar_widget().remove_class("visible")
            except NoMatches:
                pass
            return
        tokens = get_query_tokens(query)
        try:
            pill_bar = self._get_filter_pill_bar_widget()
            self._track_task(pill_bar.update_pills(tokens, self._watch_filter_active))
        except NoMatches:
            pass

    def _sort_papers(self) -> None:
        """Sort filtered_papers according to current sort order."""
        sort_key = SORT_OPTIONS[self._sort_index]
        self.filtered_papers = sort_papers(
            self.filtered_papers,
            sort_key,
            s2_cache=self._s2_cache,
            hf_cache=self._hf_cache,
            relevance_cache=self._relevance_scores,
        )
        self._rebuild_visible_index()

    def _rebuild_visible_index(self) -> None:
        """Rebuild the visible-paper index cache keyed by arXiv ID."""
        self._visible_index_by_id = {
            paper.arxiv_id: idx for idx, paper in enumerate(self.filtered_papers)
        }

    def _get_visible_index_map(self) -> dict[str, int]:
        """Return visible-index cache, rebuilding when absent or stale-sized."""
        visible_index_by_id = getattr(self, "_visible_index_by_id", None)
        if not isinstance(visible_index_by_id, dict) or len(visible_index_by_id) != len(
            self.filtered_papers
        ):
            self._rebuild_visible_index()
            visible_index_by_id = self._visible_index_by_id
        return visible_index_by_id

    def _get_visible_index(self, arxiv_id: str) -> int | None:
        """Return validated visible index for arxiv_id, if available."""
        visible_index_by_id = self._get_visible_index_map()
        cached_index = visible_index_by_id.get(arxiv_id)
        if (
            cached_index is not None
            and 0 <= cached_index < len(self.filtered_papers)
            and self.filtered_papers[cached_index].arxiv_id == arxiv_id
        ):
            return cached_index
        return None

    def _resolve_visible_index(self, arxiv_id: str) -> int | None:
        """Resolve visible index for arxiv_id, repairing stale cache entries."""
        cached_index = self._get_visible_index(arxiv_id)
        if cached_index is not None:
            return cached_index
        visible_index_by_id = self._get_visible_index_map()
        for index, paper in enumerate(self.filtered_papers):
            if paper.arxiv_id == arxiv_id:
                visible_index_by_id[arxiv_id] = index
                return index
        visible_index_by_id.pop(arxiv_id, None)
        return None

    def _update_option_for_paper(self, arxiv_id: str) -> None:
        """Update the list option display for a specific paper by arXiv ID."""
        visible_index = self._resolve_visible_index(arxiv_id)
        if visible_index is not None:
            self._update_option_at_index(visible_index)

    def _refresh_list_view(self) -> None:
        """Refresh the list view with current filtered papers.
        Uses OptionList for virtual rendering — only visible lines are drawn.
        """
        highlighted_id = None
        current = self._get_current_paper()
        if current is not None:
            highlighted_id = current.arxiv_id
        self._cancel_pending_detail_update()
        self._rebuild_visible_index()
        option_list = self._get_paper_list_widget()
        option_list.clear_options()
        if self.filtered_papers:
            options = [
                Option(self._render_option(paper), id=paper.arxiv_id)
                for paper in self.filtered_papers
            ]
            option_list.add_options(options)
            restored_index = (
                self._resolve_visible_index(highlighted_id) if highlighted_id is not None else None
            )
            option_list.highlighted = restored_index if restored_index is not None else 0
        else:
            empty_msg = build_list_empty_message(
                query=self._get_active_query(),
                in_arxiv_api_mode=self._in_arxiv_api_mode,
                watch_filter_active=self._watch_filter_active,
                history_mode=self._is_history_mode(),
            )
            option_list.add_option(Option(empty_msg, disabled=True))
            try:
                details = self._get_paper_details_widget()
                details.update_state(None)
            except NoMatches:
                pass

    def _render_option(self, paper: Paper) -> str:
        """Render a single paper as Rich markup for OptionList."""
        return render_paper_option(self._build_paper_row_state(paper))

    def _update_option_at_index(self, index: int) -> None:
        """Re-render a single option at the given index."""
        if index < 0 or index >= len(self.filtered_papers):
            return
        paper = self.filtered_papers[index]
        markup = self._render_option(paper)
        try:
            option_list = self._get_paper_list_widget()
            option_list.replace_option_prompt_at_index(index, markup)
        except (NoMatches, OptionDoesNotExist):
            pass

    def _get_or_create_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Get or create metadata for a paper."""
        if arxiv_id not in self._config.paper_metadata:
            self._config.paper_metadata[arxiv_id] = PaperMetadata(arxiv_id=arxiv_id)
        return self._config.paper_metadata[arxiv_id]

    def _get_current_paper(self) -> Paper | None:
        """Get the currently highlighted paper."""
        try:
            option_list = self._get_paper_list_widget()
        except NoMatches:
            return None
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            return self.filtered_papers[idx]
        return None

    def _get_current_index(self) -> int | None:
        """Get the index of the currently highlighted paper."""
        try:
            option_list = self._get_paper_list_widget()
        except NoMatches:
            return None
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            return idx
        return None

    def _apply_to_selected(
        self,
        fn: Callable[[str], None],
        target_ids: set[str] | None = None,
    ) -> None:
        """Apply fn(arxiv_id) to all selected papers, refreshing visible list items.
        Uses target_ids if provided, otherwise self.selected_ids.
        """
        ids = target_ids if target_ids is not None else self.selected_ids
        visible_ids: set[str] = set()
        for i, paper in enumerate(self.filtered_papers):
            if paper.arxiv_id in ids:
                fn(paper.arxiv_id)
                self._update_option_at_index(i)
                visible_ids.add(paper.arxiv_id)
        for aid in ids - visible_ids:
            fn(aid)

    def _bulk_toggle_bool(
        self,
        attr: str,
        true_label: str,
        false_label: str,
        title: str,
    ) -> None:
        """Toggle a boolean metadata attribute for all selected papers.
        If any selected paper has the attribute False, sets all to True;
        otherwise sets all to False.
        """
        target = any(
            not getattr(
                self._config.paper_metadata.get(aid, PaperMetadata(arxiv_id=aid)),
                attr,
            )
            for aid in self.selected_ids
        )
        self._apply_to_selected(
            lambda aid: setattr(self._get_or_create_metadata(aid), attr, target)
        )
        status = true_label if target else false_label
        self.notify(f"{len(self.selected_ids)} papers {status}", title=title)

    def _apply_tag_diff(self, arxiv_id: str, added: set[str], removed: set[str]) -> None:
        """Apply tag additions and removals to a single paper's metadata."""
        meta = self._get_or_create_metadata(arxiv_id)
        tag_set = set(meta.tags)
        tag_set |= added
        tag_set -= removed
        meta.tags = sorted(tag_set)

    def _bulk_edit_tags(self) -> None:
        """Open tags editor for bulk-tagging all selected papers."""
        n = len(self.selected_ids)
        tag_sets = [
            set(self._config.paper_metadata.get(aid, PaperMetadata(arxiv_id=aid)).tags)
            for aid in self.selected_ids
        ]
        common_tags = sorted(set.intersection(*tag_sets)) if tag_sets else []
        all_tags = self._collect_all_tags()
        target_ids = set(self.selected_ids)

        def on_bulk_tags_saved(result: PaperEditResult | None) -> None:
            if result is None:
                return
            tags = result.tags
            new_tag_set = set(tags)
            old_common = set(common_tags)
            added = new_tag_set - old_common
            removed = old_common - new_tag_set
            self._apply_to_selected(
                lambda aid: self._apply_tag_diff(aid, added, removed),
                target_ids=target_ids,
            )
            parts = []
            if added:
                parts.append(f"Added {', '.join(sorted(added))}")
            if removed:
                parts.append(f"Removed {', '.join(sorted(removed))}")
            msg = " / ".join(parts) if parts else "Tags unchanged"
            self.notify(f"{msg} on {len(target_ids)} papers", title="Bulk Tags")

        self.push_screen(
            PaperEditModal(
                f"bulk:{n}", current_tags=common_tags, all_tags=all_tags, initial_tab="tags"
            ),
            on_bulk_tags_saved,
        )

    def _compute_watched_papers(self) -> None:
        """Pre-compute which papers match watch list patterns.
        This runs once at startup and when watch list is modified,
        enabling O(1) lookup during display.
        """
        self._watched_paper_ids.clear()
        if not self._config.watch_list:
            return
        for paper in self.all_papers:
            for entry in self._config.watch_list:
                if paper_matches_watch_entry(paper, entry):
                    self._watched_paper_ids.add(paper.arxiv_id)
                    break  # Paper already matched, no need to check more entries

    def _notify_watch_list_matches(self) -> None:
        """Show a notification if any papers match the watch list."""
        if not self._watched_paper_ids:
            return
        n = len(self._watched_paper_ids)
        self.notify(
            f"{n} paper{'s' if n != 1 else ''} match your watch list",
            title="Watch List",
        )

    def _show_daily_digest(self) -> None:
        """Show a brief digest notification summarizing the day's papers."""
        if not self.all_papers:
            return
        digest = build_daily_digest(
            self.all_papers,
            watched_ids=self._watched_paper_ids,
            metadata=self._config.paper_metadata,
        )
        self.notify(digest, title="Daily Digest", timeout=8)

    def is_paper_watched(self, arxiv_id: str) -> bool:
        """Check if a paper is on the watch list. O(1) lookup."""
        return arxiv_id in self._watched_paper_ids

    async def _update_bookmark_bar(self) -> None:
        """Update the bookmark tab bar display."""
        bookmark_bar = self._get_bookmark_bar_widget()
        await bookmark_bar.update_bookmarks(
            self._config.bookmarks,
            self._active_bookmark_index,
            active_search=bool(self._get_active_query()),
        )

    def action_toggle_preview(self) -> None:
        """Toggle abstract preview in list items."""
        self._show_abstract_preview = not self._show_abstract_preview
        self._config.show_abstract_preview = self._show_abstract_preview
        status = "on" if self._show_abstract_preview else "off"
        self.notify(f"Abstract preview {status}", title="Preview")
        # Refresh list to show/hide previews
        self._refresh_list_view()
        self._update_status_bar()

    def action_toggle_detail_mode(self) -> None:
        """Toggle the detail pane between scan and full reading modes."""
        self._detail_mode = "full" if self._detail_mode == "scan" else "scan"
        self._config.detail_mode = self._detail_mode
        self._save_config_or_warn("detail density preference")
        self._update_details_header()
        self._refresh_detail_pane()
        self.notify(f"Detail view: {self._detail_mode}", title="Details")

    def action_start_mark(self) -> None:
        """Start mark-set mode. Next letter key will set a mark."""
        self._pending_mark_action = "set"
        self.notify("Press a-z to set mark", title="Mark")

    def action_start_goto_mark(self) -> None:
        """Start goto-mark mode. Next letter key will jump to that mark."""
        self._pending_mark_action = "goto"
        self.notify("Press a-z to jump to mark", title="Mark")

    def _set_mark(self, letter: str) -> None:
        """Set a mark at the current paper."""
        paper = self._get_current_paper()
        if not paper:
            self.notify("No paper selected", title="Mark", severity="warning")
            return
        self._config.marks[letter] = paper.arxiv_id
        self.notify(f"Mark '{letter}' set on {paper.arxiv_id}", title="Mark")

    def _goto_mark(self, letter: str) -> None:
        """Jump to a marked paper."""
        if letter not in self._config.marks:
            self.notify(f"Mark '{letter}' not set", title="Mark", severity="warning")
            return
        arxiv_id = self._config.marks[letter]
        paper = self._get_paper_by_id(arxiv_id)
        if not paper:
            self.notify(f"Paper {arxiv_id} not found", title="Mark", severity="warning")
            return
        # Find and scroll to the paper in the current list
        option_list = self._get_paper_list_widget()
        visible_index = self._resolve_visible_index(arxiv_id)
        if visible_index is not None:
            option_list.highlighted = visible_index
            self.notify(f"Jumped to mark '{letter}'", title="Mark")
            return
        # Paper not in current filtered list
        self.notify(
            "Paper not in current view (try clearing filter)",
            title="Mark",
            severity="warning",
        )

    def _format_paper_as_markdown(self, paper: Paper) -> str:
        """Format a paper as Markdown."""
        abstract_text = self._get_abstract_text(paper, allow_async=False) or ""
        return format_paper_as_markdown(paper, abstract_text)

    def _get_target_papers(self) -> list[Paper]:
        """Get papers to export (selected or current)."""
        return resolve_target_papers(
            filtered_papers=self.filtered_papers,
            selected_ids=self.selected_ids,
            papers_by_id=self._papers_by_id,
            current_paper=self._get_current_paper(),
        )

    def _reset_dataset_view_state(self) -> None:
        """Clear view-scoped caches and progress for a dataset swap."""
        self._cancel_pending_detail_update()
        badge_timer = self._badge_timer
        self._badge_timer = None
        if badge_timer is not None:
            badge_timer.stop()
        self._badges_dirty.clear()
        sort_timer = self._sort_refresh_timer
        self._sort_refresh_timer = None
        if sort_timer is not None:
            sort_timer.stop()
        self._sort_refresh_dirty.clear()
        self._abstract_cache.clear()
        self._abstract_loading.clear()
        self._abstract_queue.clear()
        self._abstract_pending_ids.clear()
        try:
            self._get_paper_details_widget().clear_cache()
        except NoMatches:
            pass
        self._paper_summaries.clear()
        self._summary_loading.clear()
        self._summary_mode_label.clear()
        self._summary_command_hash.clear()
        self._s2_cache.clear()
        self._s2_loading.clear()
        self._s2_api_error = False
        self._hf_cache.clear()
        self._hf_loading = False
        self._hf_api_error = False
        self._version_updates.clear()
        self._version_checking = False
        self._version_progress = None
        self._relevance_scores.clear()
        self._relevance_scoring_active = False
        self._scoring_progress = None
        self._auto_tag_active = False
        self._auto_tag_progress = None
        self._cancel_batch_requested = False
        self._tfidf_index = None
        self._tfidf_corpus_key = None
        self._pending_similarity_paper_id = None
        if self._tfidf_build_task is not None and not self._tfidf_build_task.done():
            self._tfidf_build_task.cancel()
        self._tfidf_build_task = None

    def _is_history_mode(self) -> bool:
        """Check if we're in history mode (multiple date files available)."""
        return len(self._history_files) > 0

    def _get_current_date(self) -> date | None:
        """Get the currently loaded date, or None if not in history mode."""
        if not self._is_history_mode():
            return None
        return self._history_files[self._current_date_index][0]

    def _load_current_date(self) -> bool:
        """Load papers from the current date file and refresh UI."""
        if not self._is_history_mode():
            return False
        _current_date, path = self._history_files[self._current_date_index]
        try:
            papers = parse_arxiv_file(path)
        except OSError as e:
            self.notify(
                f"Failed to load {path.name}: {e}",
                title="Load Error",
                severity="error",
            )
            return False
        self._advance_dataset_epoch()
        self.all_papers = papers
        self._papers_by_id = {p.arxiv_id: p for p in self.all_papers}
        self.filtered_papers = self.all_papers.copy()
        self._reset_dataset_view_state()
        # Clear selection when switching dates
        self.selected_ids.clear()
        # Recompute watched papers for new paper set
        self._compute_watched_papers()
        self._notify_watch_list_matches()
        self._show_daily_digest()
        # Apply current filter and sort
        query = self._get_live_query()
        self._apply_filter(query)
        # Re-fetch HF data if active (since HF data is date-specific)
        if self._hf_active:
            self._track_dataset_task(self._fetch_hf_daily())
        # Update subtitle
        self._update_subtitle()
        # Update date navigator
        self.call_after_refresh(self._refresh_date_navigator)
        return True

    def _set_history_index(self, target_index: int) -> bool:
        """Set and load a history index, rolling back on load failure."""
        if not (0 <= target_index < len(self._history_files)):
            return False
        old_index = self._current_date_index
        if target_index == old_index:
            return True
        self._current_date_index = target_index
        if self._load_current_date():
            return True
        self._current_date_index = old_index
        return False

    def _get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Look up a paper by its arXiv ID. O(1) dict lookup."""
        return self._papers_by_id.get(arxiv_id)

    async def _download_pdf_async(
        self, paper: Paper, client: httpx.AsyncClient | None = None
    ) -> bool:
        if client is None:
            logger.warning(
                "Download skipped for %s: shared HTTP client unavailable", paper.arxiv_id
            )
            return False
        result = await self._get_services().download.download_pdf(
            paper=paper,
            config=self._config,
            client=client,
            timeout_seconds=PDF_DOWNLOAD_TIMEOUT,
        )
        if result.success:
            logger.debug("Downloaded PDF for %s", paper.arxiv_id)
        else:
            logger.warning("Download failed for %s: %s", paper.arxiv_id, result.failure)
        return result.success

    def _is_download_batch_active(self) -> bool:
        """Return True when a download batch is active or pending."""
        return bool(self._download_queue or self._downloading or self._download_total)

    def _update_download_progress(self, completed: int, total: int) -> None:
        """Update status bar and footer with download progress."""
        try:
            status_bar = self._get_status_bar_widget()
            status_bar.update(f"Downloading: {completed}/{total} complete")
        except NoMatches:
            pass
        self._update_footer()

    def _safe_browser_open(self, url: str) -> bool:
        """Open a URL in the browser with error handling. Returns True on success."""
        try:
            webbrowser.open(url)
            return True
        except (webbrowser.Error, OSError) as e:
            logger.warning("Failed to open browser for %s: %s", url, e)
            self.notify(
                build_actionable_error(
                    "open your browser",
                    why="the system browser command failed",
                    next_step="copy the URL with c or export it with E",
                ),
                title="Browser",
                severity="error",
                timeout=8,
            )
            return False
