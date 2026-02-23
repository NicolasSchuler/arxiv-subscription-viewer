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


def action_toggle_search(app: "ArxivBrowser") -> None:
    """Toggle search input visibility."""
    _sync_app_globals()
    container = app._get_search_container_widget()
    if "visible" in container.classes:
        container.remove_class("visible")
    else:
        container.add_class("visible")
        app._get_search_input_widget().focus()
    app._update_footer()


def action_cancel_search(app: "ArxivBrowser") -> None:
    """Cancel search and hide input."""
    _sync_app_globals()
    container = app._get_search_container_widget()
    if "visible" in container.classes:
        container.remove_class("visible")
        search_input = app._get_search_input_widget()
        search_input.value = ""
        app._apply_filter("")
    if app._in_arxiv_api_mode:
        app.action_exit_arxiv_search_mode()


def action_exit_arxiv_search_mode(app: "ArxivBrowser") -> None:
    """Exit API search mode and restore local papers."""
    _sync_app_globals()
    if not app._in_arxiv_api_mode:
        return

    # Invalidate in-flight responses from older requests.
    app._arxiv_api_request_token += 1

    app._in_arxiv_api_mode = False
    app._arxiv_search_state = None
    app._arxiv_api_fetch_inflight = False
    app._arxiv_api_loading = False
    app._restore_local_browse_snapshot()
    app._local_browse_snapshot = None
    app._update_header()
    app.notify("Exited arXiv API mode", title="arXiv Search")


def action_arxiv_search(app: "ArxivBrowser") -> None:
    """Open modal to search all arXiv."""
    _sync_app_globals()
    default_query = ""
    default_field = "all"
    default_category = ""
    if app._arxiv_search_state is not None:
        default_query = app._arxiv_search_state.request.query
        default_field = app._arxiv_search_state.request.field
        default_category = app._arxiv_search_state.request.category

    def on_search(request: ArxivSearchRequest | None) -> None:
        if request is None:
            return
        app._track_task(app._run_arxiv_search(request, start=0))

    app.push_screen(
        ArxivSearchModal(
            initial_query=default_query,
            initial_field=default_field,
            initial_category=default_category,
        ),
        on_search,
    )


def _format_arxiv_search_label(app: "ArxivBrowser", request: ArxivSearchRequest) -> str:
    """Build a human-readable query label for API mode UI."""
    _sync_app_globals()
    return app._get_services().arxiv_api.format_query_label(request)


async def _apply_arxiv_rate_limit(app: "ArxivBrowser") -> None:
    """Sleep as needed to respect arXiv API rate limits."""
    _sync_app_globals()
    loop = asyncio.get_running_loop()
    new_last_request_at, wait_seconds = await app._get_services().arxiv_api.enforce_rate_limit(
        last_request_at=app._last_arxiv_api_request_at,
        min_interval_seconds=ARXIV_API_MIN_INTERVAL_SECONDS,
        now=loop.time,
        sleep=asyncio.sleep,
    )
    if wait_seconds > 0:
        app.notify(
            f"Waiting {wait_seconds:.1f}s for arXiv API rate limit",
            title="arXiv Search",
        )
    app._last_arxiv_api_request_at = new_last_request_at


async def _fetch_arxiv_api_page(
    app,
    request: ArxivSearchRequest,
    start: int,
    max_results: int,
) -> list[Paper]:
    """Fetch one page of results from arXiv API."""
    _sync_app_globals()
    await app._apply_arxiv_rate_limit()
    return await app._get_services().arxiv_api.fetch_page(
        client=app._http_client,
        request=request,
        start=start,
        max_results=max_results,
        timeout_seconds=ARXIV_API_TIMEOUT,
        user_agent="arxiv-subscription-viewer/1.0",
    )


def _apply_arxiv_search_results(
    app,
    request: ArxivSearchRequest,
    start: int,
    max_results: int,
    papers: list[Paper],
) -> None:
    """Switch UI to API mode and render fetched papers."""
    _sync_app_globals()
    was_in_api_mode = app._in_arxiv_api_mode
    if not was_in_api_mode and app._local_browse_snapshot is None:
        app._local_browse_snapshot = app._capture_local_browse_snapshot()

    app._in_arxiv_api_mode = True
    app._arxiv_search_state = ArxivSearchModeState(
        request=request,
        start=start,
        max_results=max_results,
    )

    # API mode has its own paper set and selection state.
    app.all_papers = papers
    app.filtered_papers = papers.copy()
    app._papers_by_id = {paper.arxiv_id: paper for paper in papers}
    app.selected_ids.clear()
    if not was_in_api_mode:
        # First API entry starts unfiltered; subsequent pages preserve user choice.
        app._watch_filter_active = False
    app._pending_query = ""
    app._highlight_terms = {"title": [], "author": [], "abstract": []}
    app._match_scores.clear()
    try:
        app._get_search_input_widget().value = ""
    except NoMatches:
        pass

    app._compute_watched_papers()
    if app._watch_filter_active:
        app.filtered_papers = [
            paper for paper in app.filtered_papers if paper.arxiv_id in app._watched_paper_ids
        ]
    app._sort_papers()
    app._refresh_list_view()
    app._update_header()

    query_label = app._format_arxiv_search_label(request)
    app.sub_title = f"API search · {truncate_text(query_label, 60)}"

    try:
        app._get_paper_list_widget().focus()
    except NoMatches:
        pass


async def _run_arxiv_search(app: "ArxivBrowser", request: ArxivSearchRequest, start: int) -> None:
    """Execute an arXiv API search and display one results page."""
    _sync_app_globals()
    if app._arxiv_api_fetch_inflight:
        app.notify("Search already in progress", title="arXiv Search")
        return

    max_results = _coerce_arxiv_api_max_results(app._config.arxiv_api_max_results)
    app._config.arxiv_api_max_results = max_results
    start = max(0, start)

    app._arxiv_api_request_token += 1
    request_token = app._arxiv_api_request_token
    app._arxiv_api_fetch_inflight = True
    app._arxiv_api_loading = True
    app._update_status_bar()

    try:
        papers = await app._fetch_arxiv_api_page(request, start, max_results)
    except ValueError as exc:
        app.notify(str(exc), title="arXiv Search", severity="error")
        return
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code == 429:
            message = build_actionable_error(
                "run arXiv API search",
                why="arXiv API rate limit reached (HTTP 429)",
                next_step="wait a few seconds and retry with A",
            )
        elif status_code >= 500:
            message = build_actionable_error(
                "run arXiv API search",
                why=f"arXiv API is unavailable right now (HTTP {status_code})",
                next_step="retry in a minute",
            )
        else:
            message = build_actionable_error(
                "run arXiv API search",
                why=f"arXiv API rejected the request (HTTP {status_code})",
                next_step="refine the query and retry with A",
            )
        app.notify(message, title="arXiv Search", severity="error", timeout=8)
        return
    except (httpx.HTTPError, OSError) as exc:
        app.notify(
            build_actionable_error(
                "run arXiv API search",
                why="a network or I/O error occurred",
                next_step="check connectivity and retry with A",
            ),
            title="arXiv Search",
            severity="error",
            timeout=8,
        )
        logger.warning("arXiv search failed: %s", exc, exc_info=True)
        return
    finally:
        if request_token == app._arxiv_api_request_token:
            app._arxiv_api_fetch_inflight = False
            app._arxiv_api_loading = False
            app._update_status_bar()

    # Ignore stale responses after mode exits or newer requests.
    if request_token != app._arxiv_api_request_token:
        return

    if start > 0 and not papers:
        app.notify("No more results", title="arXiv Search")
        return

    app._apply_arxiv_search_results(request, start, max_results, papers)
    page_number = (start // max_results) + 1
    if papers:
        app.notify(
            f"Loaded {len(papers)} results (page {page_number})",
            title="arXiv Search",
        )
    else:
        app.notify("No results found", title="arXiv Search")


async def action_goto_bookmark(app: "ArxivBrowser", index: int) -> None:
    """Switch to a bookmarked search query."""
    _sync_app_globals()
    if index < 0 or index >= len(app._config.bookmarks):
        return

    bookmark = app._config.bookmarks[index]
    app._active_bookmark_index = index

    # Update search input and apply filter
    search_input = app._get_search_input_widget()
    search_input.value = bookmark.query
    app._apply_filter(bookmark.query)

    # Update bookmark bar to show active tab
    await app._update_bookmark_bar()
    app.notify(f"Bookmark: {bookmark.name}", title="Search")


async def action_add_bookmark(app: "ArxivBrowser") -> None:
    """Add current search query as a bookmark."""
    _sync_app_globals()
    query = app._get_search_input_widget().value.strip()

    if not query:
        app.notify("Enter a search query first", title="Bookmark", severity="warning")
        return

    if len(app._config.bookmarks) >= 9:
        app.notify("Maximum 9 bookmarks allowed", title="Bookmark", severity="warning")
        return

    # Generate a short name from the query
    name = truncate_text(query, BOOKMARK_NAME_MAX_LEN)

    bookmark = SearchBookmark(name=name, query=query)
    app._config.bookmarks.append(bookmark)
    app._active_bookmark_index = len(app._config.bookmarks) - 1

    await app._update_bookmark_bar()
    app.notify(f"Added bookmark: {name}", title="Bookmark")


async def action_remove_bookmark(app: "ArxivBrowser") -> None:
    """Remove the currently active bookmark."""
    _sync_app_globals()
    if app._active_bookmark_index < 0 or app._active_bookmark_index >= len(app._config.bookmarks):
        app.notify("No active bookmark to remove", title="Bookmark", severity="warning")
        return

    removed = app._config.bookmarks.pop(app._active_bookmark_index)
    app._active_bookmark_index = -1

    await app._update_bookmark_bar()
    app.notify(f"Removed bookmark: {removed.name}", title="Bookmark")


def action_prev_date(app: "ArxivBrowser") -> None:
    """Navigate to previous (older) date file."""
    _sync_app_globals()
    if app._in_arxiv_api_mode:
        app._track_task(app._change_arxiv_page(-1))
        return

    if not app._is_history_mode():
        app.notify("Not in history mode", title="Navigate", severity="warning")
        return

    if app._current_date_index >= len(app._history_files) - 1:
        app.notify("Already at oldest", title="Navigate")
        return

    if not app._set_history_index(app._current_date_index + 1):
        return
    current_date = app._get_current_date()
    if current_date:
        app.notify(f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate")


def action_next_date(app: "ArxivBrowser") -> None:
    """Navigate to next (newer) date file."""
    _sync_app_globals()
    if app._in_arxiv_api_mode:
        app._track_task(app._change_arxiv_page(1))
        return

    if not app._is_history_mode():
        app.notify("Not in history mode", title="Navigate", severity="warning")
        return

    if app._current_date_index <= 0:
        app.notify("Already at newest", title="Navigate")
        return

    if not app._set_history_index(app._current_date_index - 1):
        return
    current_date = app._get_current_date()
    if current_date:
        app.notify(f"Loaded {current_date.strftime(HISTORY_DATE_FORMAT)}", title="Navigate")
