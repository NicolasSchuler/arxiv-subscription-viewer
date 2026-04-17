"""Options dataclass + legacy-args coercion for :class:`ArxivBrowser`.

Extracted from :mod:`arxiv_browser.browser.core` to isolate the
forward-looking constructor shape from the rest of the App body.
The legacy positional/keyword form is preserved here for backwards
compatibility; callers that pass a single :class:`ArxivBrowserOptions`
instance remain the preferred form.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from arxiv_browser.models import UserConfig
from arxiv_browser.services.interfaces import AppServices


@dataclass(slots=True)
class ArxivBrowserOptions:
    """Normalized constructor inputs for ``ArxivBrowser``.

    This is the forward-looking constructor shape. The browser still accepts a
    legacy positional/keyword argument form, and those calls are coerced into
    this dataclass before app initialization continues.
    """

    config: UserConfig | None = None
    restore_session: bool = True
    history_files: list[tuple[date, Path]] | None = None
    current_date_index: int = 0
    ascii_icons: bool = False
    services: AppServices | None = None


_LEGACY_BROWSER_OPTION_FIELDS = (
    "config",
    "restore_session",
    "history_files",
    "current_date_index",
    "ascii_icons",
    "services",
)


def _coerce_browser_options(
    options: Any,
    legacy_args: tuple[Any, ...],
    legacy_kwargs: dict[str, Any],
) -> ArxivBrowserOptions:
    """Normalize new-style options plus the legacy constructor calling convention.

    The compatibility goal is that existing callers can continue passing the
    older positional/keyword shape while new code can pass one options object.
    This helper rejects ambiguous mixed usage and always returns a fresh
    ``ArxivBrowserOptions`` instance for downstream initialization.
    """
    if options is not None:
        if isinstance(options, ArxivBrowserOptions):
            if legacy_args or legacy_kwargs:
                raise TypeError("ArxivBrowserOptions cannot be combined with legacy arguments")
            return ArxivBrowserOptions(
                config=options.config,
                restore_session=options.restore_session,
                history_files=list(options.history_files)
                if options.history_files is not None
                else None,
                current_date_index=options.current_date_index,
                ascii_icons=options.ascii_icons,
                services=options.services,
            )
        legacy_args = (options, *legacy_args)
    if legacy_args:
        if len(legacy_args) > len(_LEGACY_BROWSER_OPTION_FIELDS):
            raise TypeError(
                "ArxivBrowser() accepts at most "
                f"{len(_LEGACY_BROWSER_OPTION_FIELDS) + 1} positional arguments"
            )
        for field_name in _LEGACY_BROWSER_OPTION_FIELDS[: len(legacy_args)]:
            if field_name in legacy_kwargs:
                raise TypeError(f"ArxivBrowser() got multiple values for argument '{field_name}'")
        legacy_kwargs = {
            **dict(zip(_LEGACY_BROWSER_OPTION_FIELDS, legacy_args, strict=False)),
            **legacy_kwargs,
        }
    return ArxivBrowserOptions(**legacy_kwargs)


__all__ = ["ArxivBrowserOptions", "_coerce_browser_options"]
