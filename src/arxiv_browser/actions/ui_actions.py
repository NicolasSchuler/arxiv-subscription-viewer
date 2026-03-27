# ruff: noqa: F403, F405, UP037
# pyright: reportUndefinedVariable=false, reportAttributeAccessIssue=false
"""UI, enrichment, and navigation action handlers for ArxivBrowser.

Covers: theme cycling, S2/HF enrichment toggles, HuggingFace daily-paper
fetching, version-update checks, similar-paper recommendations, citation graph,
help overlay, command palette, and paper-collections management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arxiv_browser.action_messages import (
    build_actionable_error,
    build_actionable_success,
    build_actionable_warning,
)
from arxiv_browser.actions._runtime import *

if TYPE_CHECKING:
    from arxiv_browser.app import ArxivBrowser


_RECOVERABLE_ACTION_ERRORS = (OSError, RuntimeError, ValueError, TypeError)


def _log_action_failure(action: str, exc: Exception, *, unexpected: bool = False) -> None:
    """Log an action failure with consistent severity/context formatting."""
    qualifier = "Unexpected " if unexpected else ""
    message = f"{qualifier}{action} failed ({type(exc).__name__}): {exc}"
    logger.warning(message, exc_info=True)


def _notify_hf_matches(app: "ArxivBrowser", matched: int) -> None:
    """Notify with canonical HuggingFace match success copy."""
    app.notify(
        build_actionable_success(
            "HuggingFace trending data loaded",
            detail=f"{matched} paper{'s' if matched != 1 else ''} matched your list",
            next_step="press Ctrl+h to hide HF badges if you want a cleaner view",
        ),
        title="HF",
    )


def action_ctrl_e_dispatch(app: "ArxivBrowser") -> None:
    """Context-sensitive Ctrl+e: exit API mode if active, else toggle S2."""
    if app._in_arxiv_api_mode:
        app.action_exit_arxiv_search_mode()
    else:
        app.action_toggle_s2()


def action_toggle_s2(app: "ArxivBrowser") -> None:
    """Toggle Semantic Scholar enrichment and persist the setting."""
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
    if app._s2_active:
        app.notify(
            build_actionable_success(
                "Semantic Scholar enabled",
                next_step="press e on a paper to fetch Semantic Scholar data",
            ),
            title="S2",
        )
    else:
        app.notify("Semantic Scholar disabled", title="S2")
    app._update_status_bar()
    app._get_ui_refresh_coordinator().refresh_detail_pane()
    app._mark_badges_dirty("s2", immediate=True)


async def action_fetch_s2(app: "ArxivBrowser") -> None:
    """Fetch Semantic Scholar data for the currently highlighted paper."""
    if not app._s2_active:
        app.notify(
            build_actionable_warning(
                "Semantic Scholar is disabled",
                next_step="press Ctrl+e to enable S2, then press e again",
            ),
            title="S2",
            severity="warning",
        )
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
    app._update_status_bar()
    app._get_ui_refresh_coordinator().refresh_detail_pane()  # Show loading indicator immediately
    try:
        app._track_dataset_task(app._fetch_s2_paper_async(aid))
    except _RECOVERABLE_ACTION_ERRORS as exc:
        app._s2_loading.discard(aid)
        _log_action_failure(f"S2 fetch scheduling for {aid}", exc)
        raise
    except Exception as exc:
        app._s2_loading.discard(aid)
        _log_action_failure(f"S2 fetch scheduling for {aid}", exc, unexpected=True)
        raise


async def _fetch_s2_paper_async(app: "ArxivBrowser", arxiv_id: str) -> None:
    """Fetch S2 paper data and update UI on completion."""
    task_epoch = app._capture_dataset_epoch()
    try:
        client = app._http_client
        if client is None:
            app._s2_api_error = True
            return

        result = await app._get_services().enrichment.load_or_fetch_s2_paper(
            arxiv_id=arxiv_id,
            db_path=app._s2_db_path,
            cache_ttl_days=app._config.s2_cache_ttl_days,
            client=client,
            api_key=app._config.s2_api_key,
        )
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if not result.complete or result.state == "unavailable":
            app._s2_api_error = True
            app.notify(
                build_actionable_error(
                    "fetch Semantic Scholar data",
                    why="an API or network error occurred",
                    next_step="press e to retry after a moment",
                ),
                title="S2",
                severity="error",
            )
            return
        if result.state == "not_found" or result.paper is None:
            app._s2_api_error = False
            if not result.from_cache:
                app.notify(
                    build_actionable_warning(
                        "No Semantic Scholar data was found for this paper",
                        next_step="press e to retry later or continue with local metadata",
                    ),
                    title="S2",
                    severity="warning",
                )
            return
        app._s2_cache[arxiv_id] = result.paper
        app._s2_api_error = False
        app._get_ui_refresh_coordinator().refresh_detail_pane()
        app._mark_badges_dirty("s2")
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, TypeError) as exc:
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _log_action_failure(f"S2 fetch for {arxiv_id}", exc)
        app._s2_api_error = True
        app.notify(
            build_actionable_error(
                "fetch Semantic Scholar data",
                why="an API or network error occurred",
                next_step="press e to retry after a moment",
            ),
            title="S2",
            severity="error",
        )
    except Exception as exc:
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _log_action_failure(f"S2 fetch for {arxiv_id}", exc, unexpected=True)
        app._s2_api_error = True
        app.notify(
            build_actionable_error(
                "fetch Semantic Scholar data",
                why="an API or network error occurred",
                next_step="press e to retry after a moment",
            ),
            title="S2",
            severity="error",
        )
    finally:
        if app._is_current_dataset_epoch(task_epoch):
            app._s2_loading.discard(arxiv_id)
            app._update_status_bar()


async def action_toggle_hf(app: "ArxivBrowser") -> None:
    """Toggle HuggingFace trending on/off and persist the setting."""
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
        app.notify(
            build_actionable_success(
                "HuggingFace trending enabled",
                next_step="badges and detail matches will populate automatically",
            ),
            title="HF",
        )
        if not app._hf_cache:
            await app._fetch_hf_daily()
    else:
        app.notify("HuggingFace trending disabled", title="HF")
    app._update_status_bar()
    app._get_ui_refresh_coordinator().refresh_detail_pane()
    app._mark_badges_dirty("hf", immediate=True)


async def _fetch_hf_daily(app: "ArxivBrowser") -> None:
    """Fetch HF daily papers list and update caches."""
    if app._hf_loading:
        return
    app._hf_loading = True
    app._update_status_bar()
    try:
        app._track_dataset_task(app._fetch_hf_daily_async())
    except _RECOVERABLE_ACTION_ERRORS as exc:
        app._hf_loading = False
        app._hf_api_error = True
        app._update_status_bar()
        _log_action_failure("HF cache lookup", exc)
        app.notify(
            build_actionable_error(
                "fetch HuggingFace trending data",
                why="local cache lookup failed",
                next_step="retry in a moment or press Ctrl+h to disable HF",
            ),
            title="HF",
            severity="error",
        )
    except Exception as exc:
        app._hf_loading = False
        app._update_status_bar()
        _log_action_failure("HF fetch scheduling", exc, unexpected=True)
        raise


async def _fetch_hf_daily_async(app: "ArxivBrowser") -> None:
    """Background task: fetch HF daily papers from the API and update the UI.

    Only reached when the SQLite cache is cold (``_fetch_hf_daily`` handles
    the cache-hit fast path).  On success, ``app._hf_cache`` is populated
    with a ``{arxiv_id: HuggingFacePaper}`` mapping and badges are marked
    dirty.  On partial failure (``complete=False`` from the service layer),
    ``app._hf_api_error`` is set so the status bar can display an indicator.

    Args:
        app: The running ``ArxivBrowser`` application instance.
    """
    task_epoch = app._capture_dataset_epoch()
    try:
        client = app._http_client
        if client is None:
            app._hf_api_error = True
            return

        result = await app._get_services().enrichment.load_or_fetch_hf_daily(
            db_path=app._hf_db_path,
            cache_ttl_hours=app._config.hf_cache_ttl_hours,
            client=client,
        )
        if not app._is_current_dataset_epoch(task_epoch):
            return
        if not result.complete or result.state == "unavailable":
            app._hf_api_error = True
            app.notify(
                build_actionable_error(
                    "fetch HuggingFace trending data",
                    why="an API or network error occurred",
                    next_step="retry later or press Ctrl+h to disable HF",
                ),
                title="HF",
                severity="error",
            )
            return
        if result.state == "empty":
            app._hf_api_error = False
            if not result.from_cache:
                app.notify(
                    build_actionable_warning(
                        "No HuggingFace trending data was returned",
                        next_step="retry later or press Ctrl+h to disable HF",
                    ),
                    title="HF",
                    severity="warning",
                )
            return
        app._hf_cache = {p.arxiv_id: p for p in result.papers}
        app._hf_api_error = False
        app._get_ui_refresh_coordinator().refresh_detail_pane()
        app._mark_badges_dirty("hf")
        matched = count_hf_matches(app._hf_cache, app._papers_by_id)
        _notify_hf_matches(app, matched)
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, TypeError) as exc:
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _log_action_failure("HF daily fetch", exc)
        app._hf_api_error = True
        app.notify(
            build_actionable_error(
                "fetch HuggingFace trending data",
                why="an API or network error occurred",
                next_step="retry later or press Ctrl+h to disable HF",
            ),
            title="HF",
            severity="error",
        )
    except Exception as exc:
        if not app._is_current_dataset_epoch(task_epoch):
            return
        _log_action_failure("HF daily fetch", exc, unexpected=True)
        app._hf_api_error = True
        app.notify(
            build_actionable_error(
                "fetch HuggingFace trending data",
                why="an API or network error occurred",
                next_step="retry later or press Ctrl+h to disable HF",
            ),
            title="HF",
            severity="error",
        )
    finally:
        if app._is_current_dataset_epoch(task_epoch):
            app._hf_loading = False
            app._update_status_bar()


async def action_check_versions(app: "ArxivBrowser") -> None:
    """Check starred papers for newer arXiv versions and notify the user.

    Requires at least one starred paper (``starred_ids`` from
    ``get_starred_paper_ids_for_version_check``).  A "newer version"
    means arXiv has published a higher version suffix (e.g. v2 while the
    locally loaded entry is v1).

    Args:
        app: The running ``ArxivBrowser`` application instance.
    """
    if app._version_checking:
        app.notify(
            build_actionable_warning(
                "Version check is already in progress",
                next_step="wait for completion before pressing V again",
            ),
            title="Versions",
            severity="warning",
        )
        return

    starred_ids = get_starred_paper_ids_for_version_check(app._config.paper_metadata)
    if not starred_ids:
        app.notify(
            build_actionable_warning(
                "No starred papers are available for version checks",
                next_step="star papers with x, then press V",
            ),
            title="Versions",
            severity="warning",
        )
        return

    app._version_checking = True
    app._update_status_bar()
    app.notify(
        f"Checking {len(starred_ids)} starred papers...",
        title="Versions",
    )
    app._track_dataset_task(app._check_versions_async(starred_ids))


def action_show_similar(app: "ArxivBrowser") -> None:
    """Show papers similar to the currently highlighted paper.

    When Semantic Scholar is active (``app._s2_active``), a source-selection
    modal is pushed first so the user can choose between local TF-IDF and
    S2 recommendations.  When S2 is disabled, the local-only path is taken
    directly without prompting.

    Args:
        app: The running ``ArxivBrowser`` application instance.
    """
    paper = app._get_current_paper()
    if not paper:
        app.notify(
            build_actionable_warning(
                "No paper is selected",
                next_step="move with j/k to a paper, then press R",
            ),
            title="Similar",
            severity="warning",
        )
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
    client = app._http_client
    if client is None:
        return []
    result = await app._get_services().enrichment.load_or_fetch_s2_recommendations(
        arxiv_id=arxiv_id,
        db_path=app._s2_db_path,
        cache_ttl_days=S2_REC_CACHE_TTL_DAYS,
        client=client,
        api_key=app._config.s2_api_key,
    )
    return result.papers


def action_citation_graph(app: "ArxivBrowser") -> None:
    """Open the citation graph modal for the current paper."""
    if not app._s2_active:
        app.notify(
            build_actionable_warning(
                "Semantic Scholar is disabled",
                next_step="press Ctrl+e to enable S2, then press G again",
            ),
            title="S2",
            severity="warning",
        )
        return
    paper = app._get_current_paper()
    if not paper:
        return
    # Determine S2 paper ID: prefer cached S2 data, fallback to ARXIV:id
    s2_data = app._s2_cache.get(paper.arxiv_id)
    paper_id = s2_data.s2_paper_id if s2_data else f"ARXIV:{paper.arxiv_id}"
    app.notify("Fetching citation graph...", title="Citations")
    app._track_dataset_task(app._show_citation_graph(paper_id, paper.title))


async def _fetch_citation_graph(
    app, paper_id: str
) -> tuple[list[CitationEntry], list[CitationEntry]]:
    """Fetch references and citations for a paper, with SQLite caching.

    Cache coherence uses a single ``has_s2_citation_graph_cache`` probe
    rather than loading both directions separately, to avoid partial cache
    reads.  Both directions are written to the cache **only when both API
    fetches complete successfully** (``refs_ok and cites_ok``); a partial
    write is suppressed to prevent stale one-sided data from poisoning
    future loads.

    Args:
        app: The running ``ArxivBrowser`` application instance.
        paper_id: Semantic Scholar paper ID (or ``"ARXIV:<id>"`` fallback).

    Returns:
        A ``(references, citations)`` tuple.  Either list may be empty on
        API error or when the paper has no connections.
    """
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

    def _on_result(result: list[str] | None) -> None:
        if result is not None:
            app._config.collapsed_sections = result
            app._save_config_or_warn("section toggle")
            app._refresh_detail_pane()

    app.push_screen(SectionToggleModal(app._config.collapsed_sections), _on_result)


def action_show_help(app: "ArxivBrowser") -> None:
    """Show the help overlay with all keyboard shortcuts."""
    app.push_screen(
        HelpScreen(
            sections=app._build_help_sections(),
        )
    )


def action_command_palette(app: "ArxivBrowser") -> None:
    """Open the fuzzy-searchable command palette."""
    commands = app._build_command_palette_commands()
    command_labels = {command.action: command.name for command in commands}

    def _on_command_selected(action_name: str | None) -> None:
        if not action_name:
            return
        command_label = command_labels.get(action_name, action_name)
        method = getattr(app, f"action_{action_name}", None)
        if method is not None:
            try:
                result = method()
                if asyncio.iscoroutine(result):
                    app._track_task(result)
            except _RECOVERABLE_ACTION_ERRORS as exc:
                logger.warning(
                    "Command palette action %s failed (%s): %s",
                    action_name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                app.notify(
                    f"{command_label} failed. Try: retry from Ctrl+p or press ? for help.",
                    title="Commands",
                    severity="error",
                )
            except Exception as exc:
                logger.warning(
                    "Unexpected command palette action failure %s (%s): %s",
                    action_name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                app.notify(
                    f"{command_label} failed. Try: retry from Ctrl+p or press ? for help.",
                    title="Commands",
                    severity="error",
                )
        else:
            logger.warning("Unknown command palette action: %s", action_name)

    app.push_screen(CommandPaletteModal(commands), _on_command_selected)


def action_collections(app: "ArxivBrowser") -> None:
    """Open the collections manager modal."""
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
