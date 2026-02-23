# ruff: noqa: F403, F405, UP037
# pyright: reportUndefinedVariable=false, reportAttributeAccessIssue=false
"""Extracted ArxivBrowser action handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arxiv_browser.actions._runtime import *

if TYPE_CHECKING:
    from arxiv_browser.app import ArxivBrowser


def _sync_app_globals() -> None:
    """Sync patched globals from arxiv_browser.app without importing it."""
    sync_app_globals(globals())


def action_ctrl_e_dispatch(app: "ArxivBrowser") -> None:
    """Context-sensitive Ctrl+e: exit API mode if active, else toggle S2."""
    _sync_app_globals()
    if app._in_arxiv_api_mode:
        app.action_exit_arxiv_search_mode()
    else:
        app.action_toggle_s2()


def action_toggle_s2(app: "ArxivBrowser") -> None:
    """Toggle Semantic Scholar enrichment and persist the setting."""
    _sync_app_globals()
    prev_state = app._s2_active
    app._s2_active = not app._s2_active
    app._config.s2_enabled = app._s2_active
    if not save_config(app._config):
        app._s2_active = prev_state
        app._config.s2_enabled = prev_state
        app.notify(
            "Failed to save Semantic Scholar setting",
            title="S2",
            severity="error",
        )
        return
    state = "enabled" if app._s2_active else "disabled"
    app.notify(f"Semantic Scholar {state}", title="S2")
    app._update_status_bar()
    app._get_ui_refresh_coordinator().refresh_detail_pane()
    app._mark_badges_dirty("s2", immediate=True)


async def action_fetch_s2(app: "ArxivBrowser") -> None:
    """Fetch Semantic Scholar data for the currently highlighted paper."""
    _sync_app_globals()
    if not app._s2_active:
        app.notify("S2 is disabled (Ctrl+e to enable)", title="S2", severity="warning")
        return
    paper = app._get_current_paper()
    if not paper:
        return
    aid = paper.arxiv_id
    if aid in app._s2_loading:
        return  # Already fetching
    if aid in app._s2_cache:
        app.notify("S2 data already loaded", title="S2")
        return
    app._s2_loading.add(aid)
    app._get_ui_refresh_coordinator().refresh_detail_pane()  # Show loading indicator immediately
    # Try SQLite cache first (off main thread)
    try:
        cached = await asyncio.to_thread(
            load_s2_paper, app._s2_db_path, aid, app._config.s2_cache_ttl_days
        )
    except Exception:
        app._s2_loading.discard(aid)
        logger.warning("S2 cache lookup failed for %s", aid, exc_info=True)
        app.notify("S2 fetch failed", title="S2", severity="error")
        return
    if cached:
        app._s2_cache[aid] = cached
        app._s2_loading.discard(aid)
        app._get_ui_refresh_coordinator().refresh_detail_and_list_item()
        return
    # Fetch from API
    try:
        app._track_task(app._fetch_s2_paper_async(aid))
    except Exception:
        app._s2_loading.discard(aid)
        raise


async def _fetch_s2_paper_async(app: "ArxivBrowser", arxiv_id: str) -> None:
    """Fetch S2 paper data and update UI on completion."""
    _sync_app_globals()
    try:
        result = await _load_or_fetch_s2_paper_cached(
            arxiv_id=arxiv_id,
            db_path=app._s2_db_path,
            cache_ttl_days=app._config.s2_cache_ttl_days,
            client=app._http_client,
            api_key=app._config.s2_api_key,
        )
        if result is None:
            app.notify("No S2 data found", title="S2", severity="warning")
            return
        # Cache in memory + SQLite
        app._s2_cache[arxiv_id] = result
        # Update UI if still relevant
        app._get_ui_refresh_coordinator().refresh_detail_and_list_item()
    except Exception:
        logger.warning("S2 fetch failed for %s", arxiv_id, exc_info=True)
        app.notify("S2 fetch failed", title="S2", severity="error")
    finally:
        app._s2_loading.discard(arxiv_id)


async def action_toggle_hf(app: "ArxivBrowser") -> None:
    """Toggle HuggingFace trending on/off and persist the setting."""
    _sync_app_globals()
    prev_state = app._hf_active
    app._hf_active = not app._hf_active
    app._config.hf_enabled = app._hf_active
    if not save_config(app._config):
        app._hf_active = prev_state
        app._config.hf_enabled = prev_state
        app.notify(
            "Failed to save HuggingFace setting",
            title="HF",
            severity="error",
        )
        return
    if app._hf_active:
        app.notify("HuggingFace trending enabled", title="HF")
        if not app._hf_cache:
            await app._fetch_hf_daily()
    else:
        app.notify("HuggingFace trending disabled", title="HF")
    app._update_status_bar()
    app._get_ui_refresh_coordinator().refresh_detail_pane()
    app._mark_badges_dirty("hf", immediate=True)


async def _fetch_hf_daily(app: "ArxivBrowser") -> None:
    """Fetch HF daily papers list and update caches."""
    _sync_app_globals()
    if app._hf_loading:
        return
    app._hf_loading = True
    app._update_status_bar()
    # Try SQLite cache first
    try:
        cached = await asyncio.to_thread(
            load_hf_daily_cache, app._hf_db_path, app._config.hf_cache_ttl_hours
        )
    except Exception:
        app._hf_loading = False
        app._update_status_bar()
        logger.warning("HF cache lookup failed", exc_info=True)
        app.notify("HF fetch failed", title="HF", severity="error")
        return
    if cached is not None:
        app._hf_cache = cached
        app._hf_loading = False
        app._get_ui_refresh_coordinator().refresh_detail_pane()
        app._mark_badges_dirty("hf")
        matched = count_hf_matches(app._hf_cache, app._papers_by_id)
        app.notify(f"HF: {matched} trending papers matched", title="HF")
        app._update_status_bar()
        return
    # Fetch from API
    try:
        app._track_task(app._fetch_hf_daily_async())
    except Exception:
        app._hf_loading = False
        app._update_status_bar()
        raise


async def _fetch_hf_daily_async(app: "ArxivBrowser") -> None:
    """Background task: fetch HF daily papers and update UI."""
    _sync_app_globals()
    try:
        papers = await _load_or_fetch_hf_daily_cached(
            db_path=app._hf_db_path,
            cache_ttl_hours=app._config.hf_cache_ttl_hours,
            client=app._http_client,
        )
        if not papers:
            app.notify("No HF trending data found", title="HF", severity="warning")
            return
        app._hf_cache = {p.arxiv_id: p for p in papers}
        app._get_ui_refresh_coordinator().refresh_detail_pane()
        app._mark_badges_dirty("hf")
        matched = count_hf_matches(app._hf_cache, app._papers_by_id)
        app.notify(f"HF: {matched} trending papers matched", title="HF")
    except Exception:
        logger.warning("HF daily fetch failed", exc_info=True)
        app.notify("HF fetch failed", title="HF", severity="error")
    finally:
        app._hf_loading = False
        app._update_status_bar()


async def action_check_versions(app: "ArxivBrowser") -> None:
    """Check starred papers for newer arXiv versions."""
    _sync_app_globals()
    if app._version_checking:
        app.notify("Version check already in progress", title="Versions")
        return

    starred_ids = get_starred_paper_ids_for_version_check(app._config.paper_metadata)
    if not starred_ids:
        app.notify("No starred papers to check", title="Versions")
        return

    app._version_checking = True
    app._update_status_bar()
    app.notify(
        f"Checking {len(starred_ids)} starred papers...",
        title="Versions",
    )
    app._track_task(app._check_versions_async(starred_ids))


def action_show_similar(app: "ArxivBrowser") -> None:
    """Show papers similar to the currently highlighted paper."""
    _sync_app_globals()
    paper = app._get_current_paper()
    if not paper:
        app.notify("No paper selected", title="Similar", severity="warning")
        return

    if app._s2_active:
        app.push_screen(
            RecommendationSourceModal(),
            callback=lambda source: app._show_recommendations(paper, source),
        )
    else:
        app._show_recommendations(paper, "local")


async def _fetch_s2_recommendations_async(
    app: "ArxivBrowser", arxiv_id: str
) -> list[SemanticScholarPaper]:
    """Fetch S2 recommendations with SQLite cache."""
    _sync_app_globals()
    cached = await asyncio.to_thread(
        load_s2_recommendations,
        app._s2_db_path,
        arxiv_id,
        S2_REC_CACHE_TTL_DAYS,
    )
    if cached:
        return cached
    client = app._http_client
    if client is None:
        return []
    recs = await fetch_s2_recommendations(arxiv_id, client, api_key=app._config.s2_api_key)
    if recs:
        try:
            await asyncio.to_thread(save_s2_recommendations, app._s2_db_path, arxiv_id, recs)
        except (OSError, sqlite3.Error):
            logger.warning("Failed to cache S2 recommendations for %s", arxiv_id, exc_info=True)
    return recs


def action_citation_graph(app: "ArxivBrowser") -> None:
    """Open the citation graph modal for the current paper."""
    _sync_app_globals()
    if not app._s2_active:
        app.notify("S2 is disabled (Ctrl+e to enable)", title="S2", severity="warning")
        return
    paper = app._get_current_paper()
    if not paper:
        return
    # Determine S2 paper ID: prefer cached S2 data, fallback to ARXIV:id
    s2_data = app._s2_cache.get(paper.arxiv_id)
    paper_id = s2_data.s2_paper_id if s2_data else f"ARXIV:{paper.arxiv_id}"
    app.notify("Fetching citation graph...", title="Citations")
    app._track_task(app._show_citation_graph(paper_id, paper.title))


async def _fetch_citation_graph(
    app, paper_id: str
) -> tuple[list[CitationEntry], list[CitationEntry]]:
    """Fetch references + citations with SQLite cache."""
    _sync_app_globals()
    cache_hit = await asyncio.to_thread(
        has_s2_citation_graph_cache,
        app._s2_db_path,
        paper_id,
        S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    )
    if cache_hit:
        cached_refs = await asyncio.to_thread(
            load_s2_citation_graph,
            app._s2_db_path,
            paper_id,
            "references",
            S2_CITATION_GRAPH_CACHE_TTL_DAYS,
        )
        cached_cites = await asyncio.to_thread(
            load_s2_citation_graph,
            app._s2_db_path,
            paper_id,
            "citations",
            S2_CITATION_GRAPH_CACHE_TTL_DAYS,
        )
        return cached_refs, cached_cites

    # Fetch from API
    client = app._http_client
    if client is None:
        return [], []
    api_key = app._config.s2_api_key
    refs, refs_ok = await fetch_s2_references(
        paper_id,
        client,
        api_key=api_key,
        include_status=True,
    )
    cites, cites_ok = await fetch_s2_citations(
        paper_id,
        client,
        api_key=api_key,
        include_status=True,
    )

    # Cache only when both directions completed cleanly.
    if refs_ok and cites_ok:
        await asyncio.to_thread(
            save_s2_citation_graph,
            app._s2_db_path,
            paper_id,
            "references",
            refs,
        )
        await asyncio.to_thread(
            save_s2_citation_graph,
            app._s2_db_path,
            paper_id,
            "citations",
            cites,
        )
    else:
        logger.info(
            "Skipping citation graph cache write for %s due to fetch error "
            "(refs_ok=%s cites_ok=%s)",
            paper_id,
            refs_ok,
            cites_ok,
        )
    return refs, cites


def action_cycle_theme(app: "ArxivBrowser") -> None:
    """Cycle through available color themes."""
    _sync_app_globals()
    current = app._config.theme_name
    try:
        idx = THEME_NAMES.index(current)
    except ValueError:
        idx = 0
    next_idx = (idx + 1) % len(THEME_NAMES)
    app._config.theme_name = THEME_NAMES[next_idx]
    app._apply_theme_overrides()
    app._apply_category_overrides()
    try:
        app._get_paper_details_widget().clear_cache()
    except NoMatches:
        pass
    app._refresh_list_view()
    app._refresh_detail_pane()
    app._update_status_bar()
    app._save_config_or_warn("theme preference")
    app.notify(f"Theme: {THEME_NAMES[next_idx]}", title="Theme")


def action_toggle_sections(app: "ArxivBrowser") -> None:
    """Open the section toggle modal to collapse/expand detail pane sections."""
    _sync_app_globals()

    def _on_result(result: list[str] | None) -> None:
        if result is not None:
            app._config.collapsed_sections = result
            app._save_config_or_warn("section toggle")
            app._refresh_detail_pane()

    app.push_screen(SectionToggleModal(app._config.collapsed_sections), _on_result)


def action_show_help(app: "ArxivBrowser") -> None:
    """Show the help overlay with all keyboard shortcuts."""
    _sync_app_globals()
    app.push_screen(
        HelpScreen(
            sections=app._build_help_sections(),
        )
    )


def action_command_palette(app: "ArxivBrowser") -> None:
    """Open the fuzzy-searchable command palette."""
    _sync_app_globals()

    def _on_command_selected(action_name: str | None) -> None:
        if not action_name:
            return
        method = getattr(app, f"action_{action_name}", None)
        if method is not None:
            try:
                result = method()
                if asyncio.iscoroutine(result):
                    app._track_task(result)
            except Exception:
                logger.warning("Command palette action %s failed", action_name, exc_info=True)
                app.notify(f"Command failed: {action_name}", title="Error", severity="error")
        else:
            logger.warning("Unknown command palette action: %s", action_name)

    app.push_screen(CommandPaletteModal(COMMAND_PALETTE_COMMANDS), _on_command_selected)


def action_collections(app: "ArxivBrowser") -> None:
    """Open the collections manager modal."""
    _sync_app_globals()
    modal = CollectionsModal(app._config.collections, app._papers_by_id)

    def _save_collections(result: str | None) -> None:
        if result != "save":
            return
        app._config.collections = modal.collections
        app._save_config_or_warn("collections")
        count = len(app._config.collections)
        app.notify(
            f"Saved {count} collection{'s' if count != 1 else ''}",
            title="Collections",
        )

    app.push_screen(modal, _save_collections)


def action_add_to_collection(app: "ArxivBrowser") -> None:
    """Add selected papers to a collection."""
    _sync_app_globals()
    if not app._config.collections:
        app.notify(
            "No collections. Press Ctrl+k to create one.",
            title="Collections",
            severity="warning",
        )
        return
    papers = app._get_target_papers()
    if not papers:
        return
    paper_ids = [p.arxiv_id for p in papers]

    def _on_collection_selected(name: str | None) -> None:
        if not name:
            return
        for col in app._config.collections:
            if col.name == name:
                existing = set(col.paper_ids)
                added = 0
                for pid in paper_ids:
                    if pid not in existing and len(col.paper_ids) < MAX_PAPERS_PER_COLLECTION:
                        col.paper_ids.append(pid)
                        existing.add(pid)
                        added += 1
                app._save_config_or_warn("collection update")
                app.notify(
                    f"Added {added} paper{'s' if added != 1 else ''} to '{name}'",
                    title="Collections",
                )
                break

    app.push_screen(AddToCollectionModal(app._config.collections), _on_collection_selected)
