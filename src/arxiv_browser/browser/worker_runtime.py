# pyright: reportAttributeAccessIssue=false
"""Worker, shutdown, and loading-state runtime helpers for ``ArxivBrowser``.

This mixin groups the application lifecycle teardown (``on_unmount`` and its
timer/task cancellation helpers), the Textual worker management helpers, and the
pane loading-state setters. It is mixed into ``ArxivBrowser`` and relies on
methods/attributes provided by the other mixins and ``App``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import httpx
from textual.css.query import NoMatches

logger = logging.getLogger(__name__)


class WorkerRuntimeMixin:
    """App teardown, Textual worker, and pane loading-state helpers."""

    if TYPE_CHECKING:

        def _save_session_state(self) -> None: ...
        def _cancel_dataset_tasks(self) -> None: ...
        def _update_status_bar(self) -> None: ...
        def _track_dataset_task(self, coro: Coroutine[Any, Any, None]) -> Any: ...
        def _get_paper_list_widget(self) -> Any: ...
        def _get_paper_details_widget(self) -> Any: ...

    async def on_unmount(self) -> None:
        """Save session state and clean up timers/tasks when unmounted."""
        self._shutting_down = True
        self._stop_shutdown_timers()
        self._save_session_state()
        self._cancel_dataset_tasks()
        await self._cancel_background_tasks()
        self._tfidf_build_task = None
        await self._cancel_textual_workers()
        await self._close_http_client()
        self._reset_ui_refs()

    def _stop_shutdown_timers(self) -> None:
        for attr in ("_search_timer", "_detail_timer", "_badge_timer", "_sort_refresh_timer"):
            self._stop_timer_attr(attr)

    def _stop_timer_attr(self, attr: str) -> None:
        timer = getattr(self, attr, None)
        setattr(self, attr, None)
        if timer is not None:
            timer.stop()

    async def _cancel_background_tasks(self) -> None:
        background_tasks = getattr(self, "_background_tasks", set())
        pending = [task for task in background_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=0.5)
            for task in still_pending:
                logger.debug("Background task did not cancel before shutdown: %r", task)
        if hasattr(self, "_background_tasks"):
            self._background_tasks.clear()

    def _safe_update_status_bar(self) -> None:
        try:
            self._update_status_bar()
        except Exception:
            logger.debug("Status bar refresh failed", exc_info=True)

    async def _cancel_textual_workers(self) -> None:
        workers = getattr(self, "workers", None)
        if workers is None:
            return
        try:
            workers.cancel_node(self)
            await workers.wait_for_complete()
        except Exception:
            logger.debug("Failed while waiting for Textual workers during shutdown", exc_info=True)

    def _cancel_worker_group(self, group: str) -> None:
        workers = getattr(self, "workers", None)
        if workers is None:
            return
        try:
            workers.cancel_group(self, group)
        except Exception:
            logger.debug("Failed to cancel worker group %s", group, exc_info=True)

    @staticmethod
    def _is_background_work_running(work_item: Any) -> bool:
        if work_item is None:
            return False
        if isinstance(work_item, asyncio.Task):
            return not work_item.done()
        is_running = getattr(work_item, "is_running", None)
        if isinstance(is_running, bool):
            return is_running
        is_finished = getattr(work_item, "is_finished", None)
        if isinstance(is_finished, bool):
            return not is_finished
        done = getattr(work_item, "done", None)
        if callable(done):
            return not bool(done())
        return False

    def _start_dataset_worker_compat(
        self,
        worker_method: Callable[..., Any],
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self, "_thread_id"):
            return worker_method(*args, **kwargs)
        return self._track_dataset_task(coro_factory())

    def _set_paper_list_loading(self, loading: bool) -> None:
        try:
            self._get_paper_list_widget().loading = loading
        except (AttributeError, NoMatches):
            return

    def _set_details_loading(self, loading: bool) -> None:
        try:
            self._get_paper_details_widget().loading = loading
        except (AttributeError, NoMatches):
            return

    async def _close_http_client(self) -> None:
        client: httpx.AsyncClient | None = self._http_client
        self._http_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception as e:
                logger.debug(
                    "Failed to close shared HTTP client during shutdown: %s", e, exc_info=True
                )

    def _reset_ui_refs(self) -> None:
        ui_refs = getattr(self, "_ui_refs", None)
        if ui_refs is not None:
            ui_refs.reset()
