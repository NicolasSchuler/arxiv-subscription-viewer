"""Browser implementation package for ArxivBrowser."""

from arxiv_browser.browser.contracts import _PaletteAppState
from arxiv_browser.browser.core import (
    ArxivBrowser,
    ArxivBrowserOptions,
    _coerce_browser_options,
    _fetch_paper_content_async,
    build_list_empty_message,
    main,
)

__all__ = [
    "ArxivBrowser",
    "ArxivBrowserOptions",
    "_PaletteAppState",
    "_coerce_browser_options",
    "_fetch_paper_content_async",
    "build_list_empty_message",
    "main",
]
