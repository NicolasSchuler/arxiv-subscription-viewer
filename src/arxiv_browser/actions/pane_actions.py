"""Resizable list/detail pane actions for ArxivBrowser."""

from __future__ import annotations

from typing import Any, Protocol, cast

from arxiv_browser.config import save_config
from arxiv_browser.models import (
    PANE_SPLIT_DEFAULT,
    PANE_SPLIT_MAX,
    PANE_SPLIT_MIN,
    PANE_SPLIT_TOTAL,
    UserConfig,
    coerce_pane_split,
)


class _PaneApp(Protocol):
    _pane_split: int
    _config: UserConfig
    screen: Any

    def notify(self, message: object, **kwargs: Any) -> object: ...


def _pane_app(owner: object) -> _PaneApp:
    return cast(_PaneApp, owner)


class PaneResizeMixin:
    """Actions and helpers for the resizable list/detail layout."""

    def _initialize_pane_split(self) -> None:
        app = _pane_app(self)
        app._pane_split = coerce_pane_split(app._config.pane_split)
        app._config.pane_split = app._pane_split

    def _pane_split_shares(self) -> tuple[int, int]:
        """Return wide-layout list/detail split shares for the current preference."""
        list_share = coerce_pane_split(getattr(self, "_pane_split", PANE_SPLIT_DEFAULT))
        return list_share, PANE_SPLIT_TOTAL - list_share

    def _apply_pane_split(self) -> None:
        """Apply the configured split through responsive screen classes."""
        try:
            screen = _pane_app(self).screen
        except AttributeError:
            return

        list_share, _detail_share = self._pane_split_shares()
        for split in range(PANE_SPLIT_MIN, PANE_SPLIT_MAX + 1):
            screen.remove_class(f"pane-split-{split}")
        screen.add_class(f"pane-split-{list_share}")

    def _notify_pane_split(self) -> None:
        app = _pane_app(self)
        list_share, detail_share = self._pane_split_shares()
        app.notify(f"List {list_share} / Details {detail_share}", title="Pane Split")

    def _set_pane_split(self, value: int) -> bool:
        app = _pane_app(self)
        next_split = coerce_pane_split(value)
        app._pane_split = next_split
        app._config.pane_split = next_split
        self._apply_pane_split()
        if not save_config(app._config):
            app.notify(
                "Failed to save pane split preference.",
                title="Pane Split",
                severity="warning",
            )
            return False
        self._notify_pane_split()
        return True

    def action_grow_detail_pane(self) -> None:
        """Give more space to the detail pane."""
        list_share, _detail_share = self._pane_split_shares()
        if list_share <= PANE_SPLIT_MIN:
            _pane_app(self).notify("Detail pane is already at maximum size.", title="Pane Split")
            return
        self._set_pane_split(list_share - 1)

    def action_grow_list_pane(self) -> None:
        """Give more space to the paper list pane."""
        list_share, _detail_share = self._pane_split_shares()
        if list_share >= PANE_SPLIT_MAX:
            _pane_app(self).notify("Paper list is already at maximum size.", title="Pane Split")
            return
        self._set_pane_split(list_share + 1)

    def action_reset_pane_sizes(self) -> None:
        """Restore the default list/detail pane split."""
        self._set_pane_split(PANE_SPLIT_DEFAULT)


__all__ = ["PaneResizeMixin"]
