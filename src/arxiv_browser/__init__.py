"""Public package API for arXiv Browser."""

import importlib

from arxiv_browser.models import (
    ArxivSearchModeState,
    ArxivSearchRequest,
    LocalBrowseSnapshot,
    Paper,
    PaperCollection,
    PaperMetadata,
    QueryToken,
    SearchBookmark,
    SessionState,
    UserConfig,
    WatchListEntry,
)

__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    "ArxivBrowser",
    "ArxivSearchModeState",
    "ArxivSearchRequest",
    "LocalBrowseSnapshot",
    "Paper",
    "PaperCollection",
    "PaperMetadata",
    "QueryToken",
    "SearchBookmark",
    "SessionState",
    "UserConfig",
    "WatchListEntry",
    "main",
]


def __getattr__(name: str):
    """Lazily resolve heavier compatibility exports."""
    if name == "ArxivBrowser":
        from arxiv_browser.app import ArxivBrowser as browser

        return browser
    if name == "main":
        from arxiv_browser.cli import main as cli_main

        return cli_main
    app_module = importlib.import_module("arxiv_browser.app")
    if hasattr(app_module, name):
        return getattr(app_module, name)
    raise AttributeError(f"module 'arxiv_browser' has no attribute {name!r}")
