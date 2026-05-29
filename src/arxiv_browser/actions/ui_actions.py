# ruff: noqa: UP037
"""UI, enrichment, and navigation action handlers for ArxivBrowser.

Covers: theme cycling, S2/HF enrichment toggles, HuggingFace daily-paper
fetching, version-update checks, similar-paper recommendations, citation graph,
help overlay, command palette, and paper-collections management.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx
from textual import work
from textual.css.query import NoMatches

from arxiv_browser.action_messages import (
    build_actionable_error,
    build_actionable_success,
    build_actionable_warning,
)
from arxiv_browser.actions.constants import RECOVERABLE_ACTION_ERRORS, log_action_failure, logger
from arxiv_browser.authors import (
    build_author_profile,
    dedupe_author_names,
    normalize_author_name,
    split_author_names,
)
from arxiv_browser.config import save_config
from arxiv_browser.enrichment import count_hf_matches, get_starred_paper_ids_for_version_check
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.modals.common import SectionToggleModal
from arxiv_browser.modals.discovery import AuthorPickerModal, AuthorProfileModal, TrendRadarModal
from arxiv_browser.modals.help import HelpScreen
from arxiv_browser.models import MAX_PAPERS_PER_COLLECTION
from arxiv_browser.semantic_scholar import (
    S2_CITATION_GRAPH_CACHE_TTL_DAYS,
    S2_REC_CACHE_TTL_DAYS,
    SemanticScholarPaper,
    fetch_s2_citations,
    fetch_s2_references,
    has_s2_citation_graph_cache,
    load_s2_citation_graph,
    save_s2_citation_graph,
)
from arxiv_browser.semantic_scholar_models import CitationEntry
from arxiv_browser.similarity import (
    SerendipityRequest,
    build_similarity_corpus_key,
    find_serendipitous_papers,
)
from arxiv_browser.themes import THEME_NAMES
from arxiv_browser.trend_radar import build_trend_report

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser
    from arxiv_browser.services.enrichment_service import HFDailyFetchResult, S2PaperFetchResult


_RECOVERABLE_ACTION_ERRORS = RECOVERABLE_ACTION_ERRORS
_ENRICHMENT_FETCH_ERRORS = (httpx.HTTPError, OSError, RuntimeError, ValueError, TypeError)


def _log_action_failure(action: str, exc: Exception, *, unexpected: bool = False) -> None:
    return log_action_failure(action, exc, unexpected=unexpected)


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
    next_state = not app._s2_active
    app._config.s2_enabled = next_state
    if not save_config(app._config):
        app._config.s2_enabled = app._s2_active
        app.notify(
            "Failed to save Semantic Scholar setting",
            title="S2",
            severity="error",
        )
        return
    app._s2_active = next_state
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
    app._set_details_loading(True)
    app._s2_loading = {*app._s2_loading, aid}
    try:
        app._start_dataset_worker_compat(
            app._fetch_s2_paper_worker,
            lambda: app._fetch_s2_paper_async(aid),
            aid,
        )
    except _RECOVERABLE_ACTION_ERRORS as exc:
        app._set_details_loading(False)
        app._s2_loading = app._s2_loading - {aid}
        _log_action_failure(f"S2 fetch scheduling for {aid}", exc)
        raise
    except Exception as exc:
        app._set_details_loading(False)
        app._s2_loading = app._s2_loading - {aid}
        _log_action_failure(f"S2 fetch scheduling for {aid}", exc, unexpected=True)
        raise


@work(exclusive=True, group="s2-paper", exit_on_error=False)
async def _fetch_s2_paper_worker(app: "ArxivBrowser", arxiv_id: str) -> None:
    """Worker wrapper for the current-paper Semantic Scholar fetch."""
    await _fetch_s2_paper_async(app, arxiv_id)


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
        _apply_s2_paper_result(app, arxiv_id, result)
    except asyncio.CancelledError:
        raise
    except _ENRICHMENT_FETCH_ERRORS as exc:
        _handle_s2_fetch_exception(app, arxiv_id, task_epoch, exc)
    except Exception as exc:
        _handle_s2_fetch_exception(app, arxiv_id, task_epoch, exc, unexpected=True)
    finally:
        _finish_s2_fetch(app, arxiv_id, task_epoch)


def _apply_s2_paper_result(
    app: "ArxivBrowser",
    arxiv_id: str,
    result: "S2PaperFetchResult",
) -> None:
    if not result.complete or result.state == "unavailable":
        _notify_s2_fetch_error(app)
        return
    if result.state == "not_found" or result.paper is None:
        _notify_s2_not_found(app, result)
        return
    app._s2_cache[arxiv_id] = result.paper
    app._s2_api_error = False
    app._get_ui_refresh_coordinator().refresh_detail_pane()
    app._mark_badges_dirty("s2")


def _notify_s2_fetch_error(app: "ArxivBrowser") -> None:
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


def _notify_s2_not_found(app: "ArxivBrowser", result: "S2PaperFetchResult") -> None:
    app._s2_api_error = False
    if result.from_cache:
        return
    app.notify(
        build_actionable_warning(
            "No Semantic Scholar data was found for this paper",
            next_step="press e to retry later or continue with local metadata",
        ),
        title="S2",
        severity="warning",
    )


def _handle_s2_fetch_exception(
    app: "ArxivBrowser",
    arxiv_id: str,
    task_epoch: int,
    exc: Exception,
    *,
    unexpected: bool = False,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    _log_action_failure(f"S2 fetch for {arxiv_id}", exc, unexpected=unexpected)
    _notify_s2_fetch_error(app)


def _finish_s2_fetch(app: "ArxivBrowser", arxiv_id: str, task_epoch: int) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    next_loading = app._s2_loading - {arxiv_id}
    if not next_loading:
        app._set_details_loading(False)
    app._s2_loading = next_loading


async def action_toggle_hf(app: "ArxivBrowser") -> None:
    """Toggle HuggingFace trending on/off and persist the setting."""
    next_state = not app._hf_active
    app._config.hf_enabled = next_state
    if not save_config(app._config):
        app._config.hf_enabled = app._hf_active
        app.notify(
            "Failed to save HuggingFace setting",
            title="HF",
            severity="error",
        )
        return
    app._hf_active = next_state
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


def _safe_update_status_bar(app: "ArxivBrowser") -> None:
    try:
        app._update_status_bar()
    except Exception:
        logger.debug("Status bar refresh failed during async scheduling", exc_info=True)


def _queue_hf_daily(app: "ArxivBrowser"):
    """Start the HF daily worker if one is not already active."""
    if app._hf_loading:
        return None
    app._hf_loading = True
    try:
        return app._start_dataset_worker_compat(
            app._fetch_hf_daily_worker, app._fetch_hf_daily_async
        )
    except _RECOVERABLE_ACTION_ERRORS as exc:
        app._hf_api_error = True
        app._hf_loading = False
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
        _log_action_failure("HF fetch scheduling", exc, unexpected=True)
        raise


async def _fetch_hf_daily(app: "ArxivBrowser") -> None:
    """Fetch HF daily papers list and update caches."""
    _queue_hf_daily(app)


@work(exclusive=True, group="hf-daily", exit_on_error=False)
async def _fetch_hf_daily_worker(app: "ArxivBrowser") -> None:
    """Worker wrapper for HF daily enrichment."""
    await _fetch_hf_daily_async(app)


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
        _apply_hf_daily_result(app, result)
    except asyncio.CancelledError:
        raise
    except _ENRICHMENT_FETCH_ERRORS as exc:
        _handle_hf_fetch_exception(app, task_epoch, exc)
    except Exception as exc:
        _handle_hf_fetch_exception(app, task_epoch, exc, unexpected=True)
    finally:
        _finish_hf_fetch(app, task_epoch)


def _apply_hf_daily_result(app: "ArxivBrowser", result: "HFDailyFetchResult") -> None:
    if not result.complete or result.state == "unavailable":
        _notify_hf_fetch_error(app)
        return
    if result.state == "empty":
        _notify_hf_empty(app, result)
        return
    app._hf_cache = {paper.arxiv_id: paper for paper in result.papers}
    app._hf_api_error = False
    app._get_ui_refresh_coordinator().refresh_detail_pane()
    app._mark_badges_dirty("hf")
    matched = count_hf_matches(app._hf_cache, app._papers_by_id)
    _notify_hf_matches(app, matched)


def _notify_hf_fetch_error(app: "ArxivBrowser") -> None:
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


def _notify_hf_empty(app: "ArxivBrowser", result: "HFDailyFetchResult") -> None:
    app._hf_api_error = False
    if result.from_cache:
        return
    app.notify(
        build_actionable_warning(
            "No HuggingFace trending data was returned",
            next_step="retry later or press Ctrl+h to disable HF",
        ),
        title="HF",
        severity="warning",
    )


def _handle_hf_fetch_exception(
    app: "ArxivBrowser",
    task_epoch: int,
    exc: Exception,
    *,
    unexpected: bool = False,
) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    _log_action_failure("HF daily fetch", exc, unexpected=unexpected)
    _notify_hf_fetch_error(app)


def _finish_hf_fetch(app: "ArxivBrowser", task_epoch: int) -> None:
    if not app._is_current_dataset_epoch(task_epoch):
        return
    app._hf_loading = False


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
    app.notify(
        f"Checking {len(starred_ids)} starred papers...",
        title="Versions",
    )
    app._start_dataset_worker_compat(
        app._check_versions_worker,
        lambda: app._check_versions_async(starred_ids),
        starred_ids,
    )


@work(exclusive=True, group="version-check", exit_on_error=False)
async def _check_versions_worker(app: "ArxivBrowser", arxiv_ids: set[str]) -> None:
    """Worker wrapper for arXiv version checks."""
    await app._check_versions_async(arxiv_ids)


def action_show_similar(app: "ArxivBrowser") -> None:
    """Show papers similar to the currently highlighted paper.

    Always starts with local TF-IDF recommendations.  When Semantic Scholar
    is active (``app._s2_active``), the ``RecommendationsScreen`` displays
    an inline source toggle so the user can switch between local and S2
    results without a separate pre-flight modal.

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

    app._show_recommendations(paper, "local", s2_available=app._s2_active)


def action_serendipity(app: "ArxivBrowser") -> None:
    """Jump to an intentionally surprising visible paper."""
    visible_papers = list(getattr(app, "filtered_papers", []))
    if not visible_papers:
        app.notify(
            build_actionable_warning(
                "No visible papers are available for serendipity",
                next_step="clear or adjust the filter with /, then try again",
            ),
            title="Surprise Me",
            severity="warning",
        )
        return

    corpus_key = build_similarity_corpus_key(app.all_papers)
    tfidf_index = getattr(app, "_tfidf_index", None)
    tfidf_corpus_key = getattr(app, "_tfidf_corpus_key", None)
    if tfidf_index is None or tfidf_corpus_key != corpus_key:
        app._pending_serendipity = True
        build_task = getattr(app, "_tfidf_build_task", None)
        if app._is_background_work_running(build_task):
            app.notify("Similarity indexing in progress...", title="Surprise Me")
            return
        app.notify("Indexing papers for serendipity...", title="Surprise Me")
        app._set_paper_list_loading(True)
        app._tfidf_build_task = app._start_dataset_worker_compat(
            app._build_tfidf_index_worker,
            lambda: app._build_tfidf_index_async(corpus_key),
            corpus_key,
        )
        return

    app._pending_serendipity = False
    current_paper = app._get_current_paper()
    current_id = current_paper.arxiv_id if current_paper is not None else None
    candidates = find_serendipitous_papers(
        SerendipityRequest(
            papers=visible_papers,
            metadata=app._config.paper_metadata,
            tfidf_index=tfidf_index,
            current_paper_id=current_id,
            top_n=1,
        )
    )
    if not candidates:
        app.notify(
            build_actionable_warning(
                "No serendipitous paper could be selected",
                next_step="load or reveal more papers, then try again",
            ),
            title="Surprise Me",
            severity="warning",
        )
        return

    candidate = candidates[0]
    visible_index = app._resolve_visible_index(candidate.paper.arxiv_id)
    if visible_index is None:
        app.notify(
            build_actionable_warning(
                "The surprise paper is not in the current filtered view",
                next_step="clear or adjust the filter with /, then try again",
            ),
            title="Surprise Me",
            severity="warning",
        )
        return

    app._get_paper_list_widget().highlighted = visible_index
    app.notify(candidate.reason, title="Surprise Me")


def action_trend_radar(app: "ArxivBrowser") -> None:
    """Open local-history trend analytics."""
    history_files = getattr(app, "_history_files", [])
    if not history_files:
        app.notify(
            build_actionable_warning(
                "Trend Radar needs local history files",
                next_step="open a history-backed digest directory, then try Trend Radar again",
            ),
            title="Trend Radar",
            severity="warning",
        )
        return
    report = build_trend_report(history_files)
    app.push_screen(TrendRadarModal(report))


def action_author_profile(app: "ArxivBrowser") -> None:
    """Open a local profile for one author on the current paper."""

    def _show_profile(author: str) -> None:
        profile = build_author_profile(author, app.all_papers, app._s2_cache)
        app.push_screen(AuthorProfileModal(profile))

    _with_current_author(app, "Choose Author Profile", _show_profile)


def action_track_author(app: "ArxivBrowser") -> None:
    """Track one author from the current paper using exact author matching."""

    def _track(author: str) -> None:
        old_authors = list(app._config.tracked_authors)
        existing = {normalize_author_name(name) for name in old_authors}
        if normalize_author_name(author) in existing:
            app.notify(f"Already tracking {author}", title="Authors")
            return
        app._config.tracked_authors = dedupe_author_names([*old_authors, author])
        if not save_config(app._config):
            app._config.tracked_authors = old_authors
            app.notify("Failed to save tracked author", title="Authors", severity="error")
            return
        app._compute_watched_papers()
        app._apply_filter(app._get_active_query())
        app.notify(f"Tracking {author}", title="Authors")

    _with_current_author(app, "Track Author", _track)


def _with_current_author(
    app: "ArxivBrowser",
    title: str,
    callback: Callable[[str], None],
) -> None:
    paper = app._get_current_paper()
    if paper is None:
        app.notify(
            build_actionable_warning(
                "No paper is selected",
                next_step="move with j/k to a paper, then choose the author command again",
            ),
            title="Authors",
            severity="warning",
        )
        return
    authors = split_author_names(paper.authors)
    if not authors:
        app.notify("No authors are available for this paper", title="Authors", severity="warning")
        return
    if len(authors) == 1:
        callback(authors[0])
        return
    app.push_screen(
        AuthorPickerModal(authors, title=title), lambda author: callback(author) if author else None
    )


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
    app._start_dataset_worker_compat(
        app._show_citation_graph_worker,
        lambda: app._show_citation_graph(
            paper_id,
            paper.title,
            root_arxiv_id=paper.arxiv_id,
            root_s2_data=s2_data,
        ),
        paper_id,
        paper.title,
        root_arxiv_id=paper.arxiv_id,
        root_s2_data=s2_data,
    )


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


@work(exclusive=True, group="tfidf-index", exit_on_error=False)
async def _build_tfidf_index_worker(app: "ArxivBrowser", corpus_key: str) -> None:
    """Worker wrapper for similarity index builds."""
    try:
        await app._build_tfidf_index_async(corpus_key)
    finally:
        app._set_paper_list_loading(False)


@work(exclusive=True, group="citation-graph", exit_on_error=False)
async def _show_citation_graph_worker(
    app: "ArxivBrowser",
    paper_id: str,
    title: str,
    *,
    root_arxiv_id: str = "",
    root_s2_data: SemanticScholarPaper | None = None,
) -> None:
    """Worker wrapper for loading the citation graph modal."""
    app._set_details_loading(True)
    try:
        await app._show_citation_graph(
            paper_id,
            title,
            root_arxiv_id=root_arxiv_id,
            root_s2_data=root_s2_data,
        )
    finally:
        app._set_details_loading(False)


def action_cycle_theme(app: "ArxivBrowser") -> None:
    """Cycle through available color themes."""
    current = app._effective_theme_name()
    try:
        idx = THEME_NAMES.index(current)
    except ValueError:
        idx = 0
    next_idx = (idx + 1) % len(THEME_NAMES)
    next_theme = THEME_NAMES[next_idx]
    app._theme_override = None
    app._config.theme_name = next_theme
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
    app.notify(f"Theme: {next_theme}", title="Theme")


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


def action_toggle_focus_pane(app: "ArxivBrowser") -> None:
    """Move keyboard focus between the paper list and the detail scroller."""
    try:
        paper_list = app._get_paper_list_widget()
        details = app.query_one("#details-scroll")
    except NoMatches:
        return

    try:
        details.can_focus = True
        detail_active = bool(
            getattr(app, "_detail_focus_active", False)
            or getattr(details, "has_focus", False)
            or getattr(details, "has_focus_within", False)
        )
        if detail_active:
            app._detail_focus_active = False
            paper_list.focus()
        else:
            app._detail_focus_active = True
            details.focus()
        app._update_footer()
        try:
            app._update_status_bar()
        except AttributeError:
            pass
    except _RECOVERABLE_ACTION_ERRORS as exc:
        _log_action_failure("toggle pane focus", exc)


def action_command_palette(app: "ArxivBrowser") -> None:
    """Open the OmniInput in command palette mode."""
    omni = app._get_search_container_widget()
    commands = app._build_command_palette_commands()
    omni.set_commands(commands)
    omni.open(">")


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

    app.push_screen(CollectionsModal(app._config.collections, mode="pick"), _on_collection_selected)
