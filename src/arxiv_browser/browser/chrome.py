# ruff: noqa: F403, F405
# pyright: reportAttributeAccessIssue=false, reportUndefinedVariable=false
"""Detail-pane, footer, and command-palette mixin for ArxivBrowser."""

from __future__ import annotations

from arxiv_browser.browser._runtime import *


class ChromeMixin:
    def _apply_category_overrides(self) -> None:
        """Apply category color overrides from config.
        Layers: default → per-theme → user overrides.
        """
        theme_runtime = build_theme_runtime(
            self._config.theme_name,
            theme_overrides=self._config.theme,
            category_overrides=self._config.category_colors,
        )
        self._theme_runtime = theme_runtime
        format_categories.cache_clear()

    def _apply_theme_overrides(self) -> None:
        """Apply theme overrides from config to both Rich markup and CSS variables.
        Layers: named base theme → per-key overrides from config.
        Also refreshes app-owned runtime theme state for tag/category styling.
        """
        theme_runtime = build_theme_runtime(
            self._config.theme_name,
            theme_overrides=self._config.theme,
            category_overrides=self._config.category_colors,
        )
        self._theme_runtime = theme_runtime
        # Rebuild and activate Textual theme for CSS variable resolution
        if self._config.theme:
            try:
                self.register_theme(
                    _build_textual_theme(self._config.theme_name, theme_runtime.colors)
                )
            except Exception as e:
                logger.debug("Skipping theme registration in current context: %s", e, exc_info=True)
        try:
            self.theme = self._config.theme_name
        except Exception as e:
            logger.debug("Skipping theme activation in current context: %s", e, exc_info=True)

    def _schedule_abstract_load(self, paper: Paper) -> None:
        """Schedule an abstract load with concurrency limits."""
        if paper.arxiv_id in self._abstract_loading or paper.arxiv_id in self._abstract_pending_ids:
            return
        if len(self._abstract_loading) < MAX_ABSTRACT_LOADS:
            self._abstract_loading.add(paper.arxiv_id)
            self._track_dataset_task(self._load_abstract_async(paper))
            return
        self._abstract_queue.append(paper)
        self._abstract_pending_ids.add(paper.arxiv_id)

    def _drain_abstract_queue(self) -> None:
        """Start queued abstract loads while capacity is available."""
        while self._abstract_queue and len(self._abstract_loading) < MAX_ABSTRACT_LOADS:
            paper = self._abstract_queue.popleft()
            self._abstract_pending_ids.discard(paper.arxiv_id)
            if paper.arxiv_id in self._abstract_loading:
                continue
            if paper.arxiv_id in self._abstract_cache:
                continue
            self._abstract_loading.add(paper.arxiv_id)
            self._track_dataset_task(self._load_abstract_async(paper))

    def _get_abstract_text(self, paper: Paper, allow_async: bool) -> str | None:
        """Return cached abstract text, scheduling async load if needed."""
        cached = self._abstract_cache.get(paper.arxiv_id)
        if cached is not None:
            return cached
        if paper.abstract is not None:
            self._abstract_cache[paper.arxiv_id] = paper.abstract
            return paper.abstract
        if not paper.abstract_raw:
            self._abstract_cache[paper.arxiv_id] = ""
            paper.abstract = ""
            return ""
        if not allow_async:
            cleaned = clean_latex(paper.abstract_raw)
            self._abstract_cache[paper.arxiv_id] = cleaned
            paper.abstract = cleaned
            return cleaned
        self._schedule_abstract_load(paper)
        return None

    async def _load_abstract_async(self, paper: Paper) -> None:
        """Clean a paper's LaTeX abstract off-thread and update the display."""
        task_epoch = self._capture_dataset_epoch()
        try:
            cleaned = await asyncio.to_thread(clean_latex, paper.abstract_raw)
            if not self._is_current_dataset_epoch(task_epoch):
                return
            self._abstract_cache[paper.arxiv_id] = cleaned
            # Only update if not already set (idempotent to avoid race conditions)
            if paper.abstract is None:
                paper.abstract = cleaned
            self._update_abstract_display(paper.arxiv_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Abstract load failed for %s", paper.arxiv_id, exc_info=True)
        finally:
            self._abstract_loading.discard(paper.arxiv_id)
            self._drain_abstract_queue()

    def _tags_for(self, arxiv_id: str) -> list[str] | None:
        """Return tags for a paper, or None if none set."""
        meta = self._config.paper_metadata.get(arxiv_id)
        return meta.tags if meta and meta.tags else None

    def _build_detail_state(
        self,
        arxiv_id: str,
        paper: Paper,
        abstract_text: str | None = None,
    ) -> DetailRenderState:
        """Build the full detail-pane render state for one paper."""
        s2_data, s2_loading = self._s2_state_for(arxiv_id)
        theme_runtime = self._resolved_theme_runtime()
        resolved_abstract = paper.abstract or "" if abstract_text is None else abstract_text
        return DetailRenderState(
            paper=paper,
            abstract_text=resolved_abstract,
            abstract_loading=abstract_text is None and paper.abstract is None,
            summary=self._paper_summaries.get(arxiv_id),
            summary_loading=arxiv_id in self._summary_loading,
            highlight_terms=tuple(self._highlight_terms.get("abstract", [])),
            s2_data=s2_data,
            s2_loading=s2_loading,
            hf_data=self._hf_state_for(arxiv_id),
            version_update=self._version_update_for(arxiv_id),
            summary_mode=self._summary_mode_label.get(arxiv_id, ""),
            tags=tuple(self._tags_for(arxiv_id) or ()),
            relevance=self._relevance_scores.get(arxiv_id),
            collapsed_sections=tuple(self._config.collapsed_sections),
            detail_mode=getattr(self, "_detail_mode", "scan"),
            theme_colors=theme_runtime.colors,
            category_colors=theme_runtime.category_colors,
            tag_namespace_colors=theme_runtime.tag_namespace_colors,
        )

    def _build_paper_row_state(self, paper: Paper) -> PaperRowRenderState:
        """Build the list-row render state for one visible paper."""
        aid = paper.arxiv_id
        theme_runtime = self._resolved_theme_runtime()
        return PaperRowRenderState(
            paper=paper,
            selected=aid in self.selected_ids,
            metadata=self._config.paper_metadata.get(aid),
            watched=aid in self._watched_paper_ids,
            show_preview=self._show_abstract_preview,
            abstract_text=self._get_abstract_text(paper, allow_async=self._show_abstract_preview),
            highlight_terms=PaperHighlightTerms(
                title=tuple(self._highlight_terms.get("title", [])),
                author=tuple(self._highlight_terms.get("author", [])),
                abstract=tuple(self._highlight_terms.get("abstract", [])),
            ),
            s2_data=self._s2_cache.get(aid) if self._s2_active else None,
            hf_data=self._hf_cache.get(aid) if self._hf_active else None,
            version_update=self._version_updates.get(aid),
            relevance_score=self._relevance_scores.get(aid),
            theme_colors=theme_runtime.colors,
            category_colors=theme_runtime.category_colors,
            tag_namespace_colors=theme_runtime.tag_namespace_colors,
        )

    def _build_status_bar_state(self) -> StatusBarState:
        """Build the current semantic status-bar state."""
        api_page: int | None = None
        if self._in_arxiv_api_mode and self._arxiv_search_state is not None:
            api_page = (self._arxiv_search_state.start // self._arxiv_search_state.max_results) + 1
        size = getattr(self, "size", None)
        theme_runtime = self._resolved_theme_runtime()
        return StatusBarState(
            total=len(self.all_papers),
            filtered=len(self.filtered_papers),
            query=self._get_active_query(),
            watch_filter_active=self._watch_filter_active,
            selected_count=len(self.selected_ids),
            sort_label=SORT_OPTIONS[self._sort_index],
            in_arxiv_api_mode=self._in_arxiv_api_mode,
            api_page=api_page,
            arxiv_api_loading=self._arxiv_api_loading,
            show_abstract_preview=self._show_abstract_preview,
            s2_active=self._s2_active,
            s2_loading=bool(self._s2_loading),
            s2_count=len(self._s2_cache),
            s2_api_error=self._s2_api_error,
            hf_active=self._hf_active,
            hf_loading=self._hf_loading,
            hf_match_count=count_hf_matches(self._hf_cache, self._papers_by_id),
            hf_api_error=self._hf_api_error,
            version_checking=self._version_checking,
            version_update_count=len(self._version_updates),
            max_width=getattr(size, "width", None),
            theme_colors=theme_runtime.colors,
        )

    def _format_details_header_text(self) -> str:
        """Return the right-pane header text for the current detail density."""
        from arxiv_browser._ascii import is_ascii_mode

        sep = " - " if is_ascii_mode() else " \u00b7 "
        return f" Paper Details{sep}{self._detail_mode}"

    def _update_details_header(self) -> None:
        """Refresh the detail pane header text."""
        try:
            self._get_details_header_widget().update(self._format_details_header_text())
        except NoMatches:
            pass

    def _build_subtitle_text(self) -> str:
        """Build the app subtitle for the current dataset and mode."""
        from arxiv_browser._ascii import is_ascii_mode

        sep = " - " if is_ascii_mode() else " \u00b7 "
        if self._in_arxiv_api_mode and self._arxiv_search_state is not None:
            state = self._arxiv_search_state
            page = (state.start // state.max_results) + 1
            query_label = truncate_text(self._format_arxiv_search_label(state.request), 60)
            return f"Search{sep}{query_label}{sep}page {page}"
        query = self._get_active_query()
        if query:
            return f"Filtered{sep}{len(self.filtered_papers)}/{len(self.all_papers)} papers"
        current_date = self._get_current_date()
        if current_date is not None:
            return f"Browse{sep}{len(self.all_papers)} papers{sep}{current_date.strftime(HISTORY_DATE_FORMAT)}"
        return f"Browse{sep}{len(self.all_papers)} papers"

    def _update_subtitle(self) -> None:
        """Refresh the app subtitle from current state."""
        self.sub_title = self._build_subtitle_text()

    def _update_abstract_display(self, arxiv_id: str) -> None:
        """Refresh the detail pane and list preview after an abstract finishes loading."""
        try:
            details = self._get_paper_details_widget()
            if details.paper and details.paper.arxiv_id == arxiv_id:
                abstract_text = self._abstract_cache.get(arxiv_id, "")
                details.update_state(
                    self._build_detail_state(arxiv_id, details.paper, abstract_text)
                )
        except NoMatches:
            pass
        # Update list option if showing preview
        if self._show_abstract_preview:
            self._update_option_for_paper(arxiv_id)

    def _save_config_or_warn(self, context: str) -> bool:
        """Save config and notify the user on failure.
        Returns True on success, False on failure.
        """
        if not save_config(self._config):
            self.notify(f"Failed to save {context}.", severity="warning")
            return False
        return True

    def _save_session_state(self) -> None:
        """Save current session state to config.
        Handles the case where DOM widgets may already be destroyed during unmount.
        """
        # API mode is intentionally session-ephemeral; persist the underlying local state.
        snapshot = self._local_browse_snapshot if self._in_arxiv_api_mode else None
        # Get current date for history mode
        current_date = self._get_current_date()
        current_date_str = current_date.strftime(HISTORY_DATE_FORMAT) if current_date else None
        if snapshot is not None:
            self._config.session = SessionState(
                scroll_index=snapshot.list_index,
                current_filter=snapshot.applied_query.strip(),
                sort_index=snapshot.sort_index,
                selected_ids=list(snapshot.selected_ids),
                current_date=current_date_str,
            )
            if not save_config(self._config):
                logger.warning("Failed to save session state to config file")
                self.notify(
                    "Failed to save session -- changes may be lost",
                    title="Save Error",
                    severity="error",
                    timeout=8,
                )
            return
        try:
            list_view = self._get_paper_list_widget()
            self._config.session = SessionState(
                scroll_index=list_view.highlighted if list_view.highlighted is not None else 0,
                current_filter=self._get_active_query(),
                sort_index=self._sort_index,
                selected_ids=list(self.selected_ids),
                current_date=current_date_str,
            )
        except (NoMatches, ScreenStackError):
            # DOM already torn down during shutdown, save with defaults
            self._config.session = SessionState(
                scroll_index=0,
                current_filter=self._get_active_query(),
                sort_index=self._sort_index,
                selected_ids=list(self.selected_ids),
                current_date=current_date_str,
            )
        if not save_config(self._config):
            logger.warning("Failed to save session state to config file")
            self.notify(
                "Failed to save session -- changes may be lost",
                title="Save Error",
                severity="error",
                timeout=8,
            )

    def _debounced_detail_update(self) -> None:
        """Apply detail pane update after debounce delay."""
        self._detail_timer = None
        paper = self._pending_detail_paper
        self._pending_detail_paper = None
        started_at = self._pending_detail_started_at
        self._pending_detail_started_at = None
        if paper is None:
            return
        current = self._get_current_paper()
        if current is None or current.arxiv_id != paper.arxiv_id:
            return
        try:
            details = self._get_paper_details_widget()
        except NoMatches:
            return  # Widget tree torn down during shutdown
        aid = current.arxiv_id
        abstract_text = self._get_abstract_text(current, allow_async=True)
        details.update_state(self._build_detail_state(aid, current, abstract_text))
        if started_at is not None:
            logger.debug(
                "Selection->detail latency: %.2fms (paper=%s)",
                (time.perf_counter() - started_at) * 1000.0,
                aid,
            )

    def _cancel_pending_detail_update(self) -> None:
        """Cancel any pending debounced detail-pane update."""
        timer = self._detail_timer
        self._detail_timer = None
        if timer is not None:
            timer.stop()
        self._pending_detail_paper = None
        self._pending_detail_started_at = None

    def _mark_badges_dirty(
        self,
        *badge_types: str,
        immediate: bool = False,
    ) -> None:
        """Schedule a coalesced badge refresh for the given types.
        Use immediate=True for toggle-off cases where UX needs instant feedback.
        """
        self._badges_dirty.update(badge_types)
        self._schedule_sort_sensitive_refresh(*badge_types, immediate=immediate)
        if immediate:
            old = self._badge_timer
            self._badge_timer = None
            if old is not None:
                old.stop()
            self._flush_badge_refresh()
            return
        # Atomic swap timer pattern (same as search/detail debounce)
        old = self._badge_timer
        self._badge_timer = None
        if old is not None:
            old.stop()
        self._badge_timer = self.set_timer(BADGE_COALESCE_DELAY, self._flush_badge_refresh)

    def _badge_refresh_indices(self, dirty: set[str]) -> list[int]:
        """Return visible list indices requiring badge redraw for dirty badge types."""
        if not self.filtered_papers:
            return []
        refresh_all = False
        dirty_ids: set[str] = set()
        if "s2" in dirty:
            if self._s2_active:
                dirty_ids.update(self._s2_cache.keys())
            else:
                refresh_all = True
        if "hf" in dirty:
            if self._hf_active:
                dirty_ids.update(self._hf_cache.keys())
            else:
                refresh_all = True
        if "version" in dirty:
            dirty_ids.update(self._version_updates.keys())
        if "relevance" in dirty:
            dirty_ids.update(self._relevance_scores.keys())
        # Unknown badge type, or explicit full redraw request: fall back to full repaint.
        if (
            not dirty
            or refresh_all
            or any(kind not in {"s2", "hf", "version", "relevance"} for kind in dirty)
        ):
            return list(range(len(self.filtered_papers)))
        if not dirty_ids:
            return list(range(len(self.filtered_papers)))
        visible_index_by_id = self._get_visible_index_map()
        return sorted(
            visible_index_by_id[paper_id]
            for paper_id in dirty_ids
            if paper_id in visible_index_by_id
        )

    def _flush_badge_refresh(self) -> None:
        """Coalesced badge refresh for only affected visible papers."""
        self._badge_timer = None
        dirty = self._badges_dirty.copy()
        self._badges_dirty.clear()
        if not dirty:
            return
        indices = self._badge_refresh_indices(dirty)
        for i in indices:
            self._update_option_at_index(i)
        logger.debug(
            "Badge refresh: dirty=%s updated=%d/%d",
            sorted(dirty),
            len(indices),
            len(self.filtered_papers),
        )

    def _sort_sensitive_badge_kind(self, badge_kind: str) -> bool:
        """Return whether the active sort order depends on a cache-backed badge."""
        sort_index = getattr(self, "_sort_index", 0)
        if sort_index < 0 or sort_index >= len(SORT_OPTIONS):
            return False
        sort_key = SORT_OPTIONS[sort_index]
        return (
            (badge_kind == "s2" and sort_key == "citations")
            or (badge_kind == "hf" and sort_key == "trending")
            or (badge_kind == "relevance" and sort_key == "relevance")
        )

    def _schedule_sort_sensitive_refresh(
        self,
        *badge_types: str,
        immediate: bool = False,
    ) -> None:
        """Debounce re-sorts triggered by async cache updates."""
        if not any(self._sort_sensitive_badge_kind(kind) for kind in badge_types):
            return
        self._sort_refresh_dirty.update(badge_types)
        if immediate:
            timer = self._sort_refresh_timer
            self._sort_refresh_timer = None
            if timer is not None:
                timer.stop()
            self._flush_sort_sensitive_refresh()
            return
        timer = self._sort_refresh_timer
        self._sort_refresh_timer = None
        if timer is not None:
            timer.stop()
        self._sort_refresh_timer = self.set_timer(
            BADGE_COALESCE_DELAY,
            self._flush_sort_sensitive_refresh,
        )

    def _flush_sort_sensitive_refresh(self) -> None:
        """Re-sort the list when async cache updates affect ordering."""
        self._sort_refresh_timer = None
        dirty = self._sort_refresh_dirty.copy()
        self._sort_refresh_dirty.clear()
        if not dirty or not any(self._sort_sensitive_badge_kind(kind) for kind in dirty):
            return
        highlighted = self._get_current_paper()
        highlighted_id = highlighted.arxiv_id if highlighted is not None else None
        self._sort_papers()
        self._refresh_list_view()
        if highlighted_id is not None:
            visible_index = self._resolve_visible_index(highlighted_id)
            if visible_index is not None:
                try:
                    self._get_paper_list_widget().highlighted = visible_index
                except NoMatches:
                    return
        self._refresh_detail_pane()

    def _refresh_detail_pane(self) -> None:
        """Re-render the detail pane for the currently highlighted paper."""
        paper = self._get_current_paper()
        if not paper:
            return
        try:
            details = self._get_paper_details_widget()
        except NoMatches:
            return
        aid = paper.arxiv_id
        abstract_text = self._get_abstract_text(paper, allow_async=False)
        details.update_state(self._build_detail_state(aid, paper, abstract_text))

    def _refresh_current_list_item(self) -> None:
        """Update the current list item's option display."""
        idx = self._get_current_index()
        if idx is not None:
            self._update_option_at_index(idx)

    def _build_help_sections(
        self,
        *,
        search_first: bool = False,
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """Build help sections from the runtime key binding table."""
        return build_help_sections(self.BINDINGS, search_first=search_first)

    @staticmethod
    def _palette_group_for_action(action_name: str) -> str:
        """Return a compact group label for a palette action."""
        group_map = {
            "toggle_search": "Core",
            "show_search_syntax": "Core",
            "arxiv_search": "Research",
            "prev_date": "Advanced",
            "next_date": "Advanced",
            "open_url": "Core",
            "open_pdf": "Core",
            "download_pdf": "Core",
            "copy_selected": "Core",
            "toggle_read": "Organize",
            "toggle_star": "Organize",
            "edit_notes": "Organize",
            "edit_tags": "Organize",
            "select_all": "Core",
            "clear_selection": "Core",
            "toggle_select": "Core",
            "cycle_sort": "Core",
            "toggle_watch_filter": "Organize",
            "manage_watch_list": "Organize",
            "toggle_preview": "Advanced",
            "export_menu": "Core",
            "export_metadata": "Advanced",
            "import_metadata": "Advanced",
            "fetch_s2": "Research",
            "ctrl_e_dispatch": "Research",
            "toggle_hf": "Research",
            "check_versions": "Research",
            "citation_graph": "Research",
            "generate_summary": "Research",
            "chat_with_paper": "Research",
            "score_relevance": "Research",
            "edit_interests": "Research",
            "auto_tag": "Research",
            "show_similar": "Research",
            "add_bookmark": "Organize",
            "collections": "Organize",
            "add_to_collection": "Organize",
            "toggle_detail_mode": "Advanced",
            "cycle_theme": "Advanced",
            "toggle_sections": "Advanced",
            "show_help": "Core",
            "start_mark": "Advanced",
            "start_goto_mark": "Advanced",
        }
        return group_map.get(action_name, "Commands")

    def _command_palette_state(self) -> _PaletteAppState:
        """Capture the app state needed to shape command-palette entries."""
        config = getattr(self, "_config", None)
        history_files = getattr(self, "_history_files", [])
        in_arxiv_api_mode = bool(getattr(self, "_in_arxiv_api_mode", False))
        watch_list = list(getattr(config, "watch_list", []))
        has_marks = bool(getattr(config, "marks", {}))
        metadata_values = getattr(config, "paper_metadata", {}).values() if config else []
        filtered_papers = getattr(self, "filtered_papers", [])
        try:
            current_paper = self._get_current_paper()
        except AttributeError:
            current_paper = None
        s2_cache = getattr(self, "_s2_cache", {})
        has_selection = bool(getattr(self, "selected_ids", set()))
        return _PaletteAppState(
            in_arxiv_api_mode=in_arxiv_api_mode,
            hf_active=bool(getattr(self, "_hf_active", False)),
            watch_filter_active=bool(getattr(self, "_watch_filter_active", False)),
            show_abstract_preview=bool(getattr(config, "show_abstract_preview", False)),
            detail_mode=getattr(self, "_detail_mode", "scan"),
            active_query=self._get_active_query(),
            has_history_navigation=bool(history_files and len(history_files) > 1),
            watch_list=watch_list,
            has_marks=has_marks,
            has_starred=any(getattr(meta, "starred", False) for meta in metadata_values),
            llm_configured=bool(isinstance(config, UserConfig) and _resolve_llm_command(config)),
            has_visible_papers=bool(filtered_papers),
            has_selection=has_selection,
            has_current_paper=current_paper is not None,
            has_target_papers=has_selection or current_paper is not None,
            s2_active=bool(getattr(self, "_s2_active", False)),
            s2_data_loaded=bool(current_paper and current_paper.arxiv_id in s2_cache),
        )

    def _palette_suggested_actions(self, state: _PaletteAppState) -> set[str]:
        """Return the set of actions that should be visually suggested."""
        suggested_actions = {
            "toggle_search",
            "open_url",
            "toggle_select",
            "toggle_read",
            "export_menu",
        }
        if state.in_arxiv_api_mode:
            suggested_actions.update({"ctrl_e_dispatch", "arxiv_search"})
        if state.active_query:
            suggested_actions.update({"add_bookmark", "show_search_syntax"})
        if state.has_selection:
            suggested_actions.update({"edit_tags", "download_pdf"})
        return suggested_actions

    def _palette_entry_copy(
        self,
        name: str,
        description: str,
        action_name: str,
        state: _PaletteAppState,
    ) -> tuple[str, str]:
        """Return the display copy for one palette command in the current state."""
        if action_name == "ctrl_e_dispatch":
            if state.in_arxiv_api_mode:
                return "Exit Search Results", "Return to your local or history papers"
            return "Toggle Semantic Scholar", "Enable or disable Semantic Scholar enrichment"
        if action_name == "toggle_hf":
            if state.hf_active:
                return (
                    "Disable HuggingFace Trending",
                    "Hide HuggingFace badges and detail-pane matches",
                )
            return (
                "Enable HuggingFace Trending",
                "Show HuggingFace badges and detail-pane matches",
            )
        if action_name == "toggle_watch_filter" and state.watch_filter_active:
            return "Show All Papers", "Return to the full paper list"
        if action_name == "toggle_preview":
            if state.show_abstract_preview:
                return "Hide Abstract Preview", "Return to a denser paper list without snippets"
            return "Show Abstract Preview", "Reveal abstract snippets in the paper list"
        if action_name == "toggle_detail_mode":
            if state.detail_mode == "scan":
                return "Switch to Full Details", "Expand the detail pane for long-form reading"
            return "Switch to Scan Details", "Return to a faster triage-focused detail view"
        return name, description

    def _palette_action_availability(
        self,
        action_name: str,
        state: _PaletteAppState,
    ) -> tuple[bool, str]:
        """Return whether a palette command is enabled and what blocks it."""
        blocked_reason = self._palette_basic_blocked_reason(action_name, state)
        if blocked_reason:
            return False, blocked_reason
        blocked_reason = self._palette_enrichment_blocked_reason(action_name, state)
        if blocked_reason:
            return False, blocked_reason
        blocked_reason = self._palette_llm_blocked_reason(action_name, state)
        if blocked_reason:
            return False, blocked_reason
        return True, ""

    def _palette_basic_blocked_reason(
        self,
        action_name: str,
        state: _PaletteAppState,
    ) -> str:
        """Return generic non-LLM, non-enrichment blockers for a palette action."""
        if (
            action_name
            in {
                "open_url",
                "open_pdf",
                "download_pdf",
                "copy_selected",
                "export_menu",
                "toggle_read",
                "toggle_star",
                "edit_notes",
                "edit_tags",
                "show_similar",
                "add_to_collection",
                "start_mark",
            }
            and not state.has_target_papers
        ):
            return "selection"
        if action_name == "select_all" and not state.has_visible_papers:
            return "visible papers"
        if action_name == "clear_selection" and not state.has_selection:
            return "selection"
        if action_name == "add_bookmark" and not state.active_query:
            return "an active search"
        if action_name in {"prev_date", "next_date"} and not state.has_history_navigation:
            return "history mode"
        if action_name == "start_goto_mark" and not state.has_marks:
            return "saved marks"
        if action_name == "toggle_watch_filter" and not state.watch_list:
            return "watch list entries"
        if action_name == "check_versions" and not state.has_starred:
            return "starred papers"
        return ""

    def _palette_enrichment_blocked_reason(
        self,
        action_name: str,
        state: _PaletteAppState,
    ) -> str:
        """Return Semantic Scholar-related blockers for a palette action."""
        if action_name == "fetch_s2":
            if not state.has_current_paper:
                return "selection"
            if not state.s2_active:
                return "Semantic Scholar enabled"
        if action_name == "citation_graph":
            if not state.has_current_paper:
                return "selection"
            if not state.s2_data_loaded:
                return "S2 data"
        return ""

    def _palette_llm_blocked_reason(
        self,
        action_name: str,
        state: _PaletteAppState,
    ) -> str:
        """Return LLM-related blockers for a palette action."""
        if action_name not in {
            "generate_summary",
            "chat_with_paper",
            "score_relevance",
            "auto_tag",
        }:
            return ""
        if not state.llm_configured:
            return "LLM configuration"
        if action_name != "score_relevance" and not state.has_target_papers:
            return "selection"
        return ""

    def _build_command_palette_commands(self) -> list[PaletteCommand]:
        """Return command palette rows with labels adapted to current app state."""
        state = self._command_palette_state()
        suggested_actions = self._palette_suggested_actions(state)
        commands: list[PaletteCommand] = []
        for name, description, key_hint, action_name in COMMAND_PALETTE_COMMANDS:
            name, description = self._palette_entry_copy(name, description, action_name, state)
            enabled, blocked_reason = self._palette_action_availability(action_name, state)
            commands.append(
                PaletteCommand(
                    name=name,
                    description=description,
                    key_hint=key_hint,
                    action=action_name,
                    group=self._palette_group_for_action(action_name),
                    enabled=enabled,
                    blocked_reason=blocked_reason,
                    suggested=action_name in suggested_actions,
                )
            )
        return commands

    def _update_list_header(self, query: str) -> None:
        """Update the list header text for the current query/context."""
        try:
            self._get_list_header_widget().update(self._format_header_text(query))
        except NoMatches:
            pass

    def _update_header(self) -> None:
        """Update pane headers and status text."""
        query = self._get_active_query()
        self._update_list_header(query)
        self._update_details_header()
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Update the status bar with semantic, context-aware information."""
        try:
            status = self._get_status_bar_widget()
        except NoMatches:
            return
        status.update(_widget_chrome.build_status_bar_text(self._build_status_bar_state()))
        self._update_footer()

    def _get_footer_bindings(self) -> list[tuple[str, str]]:
        """Return context-sensitive binding hints for the footer."""
        from arxiv_browser._ascii import is_ascii_mode

        ellipsis = "..." if is_ascii_mode() else "\u2026"
        # Progress operations take highest priority (visual progress bar)
        if self._scoring_progress is not None:
            current, total = self._scoring_progress
            bar = render_progress_bar(current, total)
            return [("", f"Scoring {bar} {current}/{total}"), ("?", "help")]
        if self._relevance_scoring_active:
            return [("", f"Scoring papers{ellipsis}"), ("?", "help")]
        if self._version_progress is not None:
            batch, total = self._version_progress
            bar = render_progress_bar(batch, total)
            return [("", f"Versions {bar} {batch}/{total}"), ("?", "help")]
        if self._version_checking:
            return [("", f"Checking versions{ellipsis}"), ("?", "help")]
        if self._is_download_batch_active():
            completed = len(self._download_results)
            total = self._download_total
            bar = render_progress_bar(completed, total)
            return [("", f"Downloading {bar} {completed}/{total}"), ("?", "help")]
        if self._auto_tag_progress is not None:
            current, total = self._auto_tag_progress
            bar = render_progress_bar(current, total)
            return [("", f"Auto-tagging {bar} {current}/{total}"), ("?", "help")]
        if self._auto_tag_active:
            return [("", f"Auto-tagging{ellipsis}"), ("?", "help")]
        # Search mode — search container visible
        try:
            container = self._get_search_container_widget()
            if container.has_class("visible"):
                return _widget_chrome.build_search_footer_bindings()
        except NoMatches:
            pass
        # arXiv API search mode
        if self._in_arxiv_api_mode:
            return _widget_chrome.build_api_footer_bindings()
        # Selection mode — papers selected
        if self.selected_ids:
            return _widget_chrome.build_selection_footer_bindings(len(self.selected_ids))
        # Default browsing — dynamically show contextual hints
        has_starred = any(m.starred for m in self._config.paper_metadata.values())
        llm_configured = bool(_resolve_llm_command(self._config))
        return _widget_chrome.build_browse_footer_bindings(
            s2_active=self._s2_active,
            has_starred=has_starred,
            llm_configured=llm_configured,
            has_history_navigation=bool(self._history_files and len(self._history_files) > 1),
        )

    def _get_footer_mode_badge(self) -> str:
        """Return a Rich-markup mode badge string for the current state."""
        search_visible = False
        try:
            container = self._get_search_container_widget()
            if container.has_class("visible"):
                search_visible = True
        except NoMatches:
            pass
        theme_runtime = self._resolved_theme_runtime()
        return _widget_chrome.build_footer_mode_badge(
            relevance_scoring_active=self._relevance_scoring_active,
            version_checking=self._version_checking,
            search_visible=search_visible,
            in_arxiv_api_mode=self._in_arxiv_api_mode,
            selected_count=len(self.selected_ids),
            theme_colors=theme_runtime.colors,
        )

    def _update_footer(self) -> None:
        """Update the context-sensitive footer based on current state."""
        try:
            footer = self._get_footer_widget()
        except (NoMatches, AttributeError):
            # AttributeError: app not fully composed (e.g. __new__ mock tests)
            return
        footer.render_bindings(self._get_footer_bindings(), self._get_footer_mode_badge())
