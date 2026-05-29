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

    def watch_selected_ids(self, old: set[str], new: set[str]) -> None:
        if self._reactive_ui_ready():
            self._update_header()

    def watch__sort_index(self, old: int, new: int) -> None:
        if not self._reactive_ui_ready():
            return
        self._sort_papers()
        self._refresh_list_view()
        self._update_header()

    def watch__watch_filter_active(self, old: bool, new: bool) -> None:
        if self._reactive_ui_ready():
            self._apply_filter(self._get_live_query())

    def watch__show_abstract_preview(self, old: bool, new: bool) -> None:
        if not self._reactive_ui_ready():
            return
        self._refresh_list_view()
        self._safe_update_status_bar()

    def watch__detail_mode(self, old: str, new: str) -> None:
        if not self._reactive_ui_ready():
            return
        self._update_details_header()
        self._refresh_detail_pane()

    def watch__in_arxiv_api_mode(self, old: bool, new: bool) -> None:
        if not self._reactive_ui_ready():
            return
        self._update_header()
        self._update_subtitle()
        self._update_filter_pills(self._get_active_query())

    def watch__arxiv_api_loading(self, old: bool, new: bool) -> None:
        self._refresh_reactive_status()

    def watch__s2_active(self, old: bool, new: bool) -> None:
        if not self._reactive_ui_ready():
            return
        self._safe_update_status_bar()
        self._get_ui_refresh_coordinator().refresh_detail_pane()
        self._mark_badges_dirty("s2", immediate=True)

    def watch__s2_loading(self, old: set[str], new: set[str]) -> None:
        if not self._reactive_ui_ready():
            return
        self._safe_update_status_bar()
        self._get_ui_refresh_coordinator().refresh_detail_pane()

    def watch__hf_active(self, old: bool, new: bool) -> None:
        if not self._reactive_ui_ready():
            return
        self._safe_update_status_bar()
        self._get_ui_refresh_coordinator().refresh_detail_pane()
        self._mark_badges_dirty("hf", immediate=True)

    def watch__hf_loading(self, old: bool, new: bool) -> None:
        self._refresh_reactive_status()

    def watch__version_checking(self, old: bool, new: bool) -> None:
        self._refresh_reactive_status()

    def watch__version_progress(
        self,
        old: tuple[int, int] | None,
        new: tuple[int, int] | None,
    ) -> None:
        self._refresh_reactive_status()
