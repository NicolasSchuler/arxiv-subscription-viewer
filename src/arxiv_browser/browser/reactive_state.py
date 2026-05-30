# pyright: reportAssignmentType=false, reportAttributeAccessIssue=false
"""Reactive state declarations and watchers for :class:`ArxivBrowser`.

This mixin holds the Textual ``reactive`` descriptors, the not-yet-mounted
attribute-access shim used by the unit-test suite, and the ``watch_*`` methods
that drive declarative UI refreshes. It is mixed into ``ArxivBrowser`` ahead of
``App`` so the ``__getattribute__``/``__setattr__`` overrides resolve first.
"""

from __future__ import annotations

from typing import Any

from textual.reactive import ReactiveError


class ReactiveStateMixin:
    """Reactive attributes plus watcher-driven refresh side-effects."""

    _REACTIVE_STATE_NAMES = frozenset(
        {
            "selected_ids",
            "_sort_index",
            "_watch_filter_active",
            "_show_abstract_preview",
            "_compact_list",
            "_detail_mode",
            "_in_arxiv_api_mode",
            "_arxiv_api_loading",
            "_s2_active",
            "_s2_loading",
            "_hf_active",
            "_hf_loading",
            "_version_checking",
            "_version_progress",
        }
    )

    def __getattribute__(self, name: str) -> Any:
        # Reactive state is declared as Textual ``reactive`` descriptors so the
        # running app gets declarative watcher-driven UI refreshes. The large
        # unit-test suite, however, constructs lightweight instances via
        # ``ArxivBrowser.__new__`` (no ``__init__``), where the reactive
        # machinery is not initialised and the descriptor raises ReactiveError.
        # For those not-yet-mounted instances (detected by the absence of
        # ``_screen_stacks``) we store/read reactive values directly in
        # ``__dict__``, bypassing both the descriptor and its watchers. Once the
        # app is mounted, access falls through to normal reactive behaviour.
        if name in object.__getattribute__(self, "_REACTIVE_STATE_NAMES"):
            data = object.__getattribute__(self, "__dict__")
            if "_screen_stacks" not in data:
                if name not in data:
                    data[name] = set() if name in {"selected_ids", "_s2_loading"} else 0
                    if name in {
                        "_watch_filter_active",
                        "_show_abstract_preview",
                        "_compact_list",
                        "_in_arxiv_api_mode",
                        "_arxiv_api_loading",
                        "_s2_active",
                        "_hf_active",
                        "_hf_loading",
                        "_version_checking",
                    }:
                        data[name] = False
                    elif name == "_detail_mode":
                        data[name] = "scan"
                    elif name == "_version_progress":
                        data[name] = None
                return data[name]
            try:
                return super().__getattribute__(name)
            except ReactiveError:
                if name in data:
                    return data[name]
                raise
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in object.__getattribute__(self, "_REACTIVE_STATE_NAMES"):
            data = object.__getattribute__(self, "__dict__")
            if "_screen_stacks" not in data:
                data[name] = value
                return
        try:
            super().__setattr__(name, value)
        except ReactiveError:
            if name not in object.__getattribute__(self, "_REACTIVE_STATE_NAMES"):
                raise
            object.__getattribute__(self, "__dict__")[name] = value

    def _reactive_ui_ready(self) -> bool:
        return (
            "_screen_stacks" in self.__dict__
            and bool(getattr(self, "is_mounted", False))
            and not bool(getattr(self, "_reactive_watchers_suspended", False))
        )

    def _refresh_reactive_status(self) -> None:
        if self._reactive_ui_ready():
            self._safe_update_status_bar()

    def watch_selected_ids(self, _old: set[str], _new: set[str]) -> None:
        """Refresh header counts when selected paper IDs change."""
        if self._reactive_ui_ready():
            self._update_header()

    def watch__sort_index(self, _old: int, _new: int) -> None:
        """Resort and redraw the list when the active sort changes."""
        if not self._reactive_ui_ready():
            return
        self._sort_papers()
        self._refresh_list_view()
        self._update_header()

    def watch__watch_filter_active(self, _old: bool, _new: bool) -> None:
        """Reapply the live query when the watch-only filter toggles."""
        if self._reactive_ui_ready():
            self._apply_filter(self._get_live_query())

    def watch__show_abstract_preview(self, _old: bool, _new: bool) -> None:
        """Refresh list rows when inline abstract previews toggle."""
        if not self._reactive_ui_ready():
            return
        self._refresh_list_view()
        self._safe_update_status_bar()

    def watch__compact_list(self, _old: bool, _new: bool) -> None:
        """Refresh list rows when compact title-only mode toggles."""
        if not self._reactive_ui_ready():
            return
        self._refresh_list_view()
        self._safe_update_status_bar()

    def watch__detail_mode(self, _old: str, _new: str) -> None:
        """Refresh detail-pane chrome and content when density changes."""
        if not self._reactive_ui_ready():
            return
        self._update_details_header()
        self._refresh_detail_pane()

    def watch__in_arxiv_api_mode(self, _old: bool, _new: bool) -> None:
        """Refresh mode-aware header, subtitle, and filter pills."""
        if not self._reactive_ui_ready():
            return
        self._update_header()
        self._update_subtitle()
        self._update_filter_pills(self._get_active_query())

    def watch__arxiv_api_loading(self, _old: bool, _new: bool) -> None:
        """Refresh compact status when API loading state changes."""
        self._refresh_reactive_status()

    def watch__s2_active(self, _old: bool, _new: bool) -> None:
        """Refresh Semantic Scholar status, details, and badges."""
        if not self._reactive_ui_ready():
            return
        self._safe_update_status_bar()
        self._get_ui_refresh_coordinator().refresh_detail_pane()
        self._mark_badges_dirty("s2", immediate=True)

    def watch__s2_loading(self, _old: set[str], _new: set[str]) -> None:
        """Refresh status and details while S2 paper fetches change."""
        if not self._reactive_ui_ready():
            return
        self._safe_update_status_bar()
        self._get_ui_refresh_coordinator().refresh_detail_pane()

    def watch__hf_active(self, _old: bool, _new: bool) -> None:
        """Refresh HuggingFace status, details, and badges."""
        if not self._reactive_ui_ready():
            return
        self._safe_update_status_bar()
        self._get_ui_refresh_coordinator().refresh_detail_pane()
        self._mark_badges_dirty("hf", immediate=True)

    def watch__hf_loading(self, _old: bool, _new: bool) -> None:
        """Refresh compact status while HuggingFace data loads."""
        self._refresh_reactive_status()

    def watch__version_checking(self, _old: bool, _new: bool) -> None:
        """Refresh compact status while version checks run."""
        self._refresh_reactive_status()

    def watch__version_progress(
        self,
        _old: tuple[int, int] | None,
        _new: tuple[int, int] | None,
    ) -> None:
        """Refresh compact status when version-check progress changes."""
        self._refresh_reactive_status()
