"""Public package API for arXiv Browser."""

from arxiv_browser.models import (
    ArxivSearchModeState as ArxivSearchModeState,
)
from arxiv_browser.models import (
    ArxivSearchRequest as ArxivSearchRequest,
)
from arxiv_browser.models import (
    LocalBrowseSnapshot as LocalBrowseSnapshot,
)
from arxiv_browser.models import (
    Paper as Paper,
)
from arxiv_browser.models import (
    PaperCollection as PaperCollection,
)
from arxiv_browser.models import (
    PaperMetadata as PaperMetadata,
)
from arxiv_browser.models import (
    QueryToken as QueryToken,
)
from arxiv_browser.models import (
    SearchBookmark as SearchBookmark,
)
from arxiv_browser.models import (
    SessionState as SessionState,
)
from arxiv_browser.models import (
    UserConfig as UserConfig,
)
from arxiv_browser.models import (
    WatchListEntry as WatchListEntry,
)

_PUBLIC_EXPORTS = [
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
__all__ = list(_PUBLIC_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    """Lazily resolve supported public exports."""
    if name == "ArxivBrowser":
        from arxiv_browser.browser.core import ArxivBrowser as browser

        return browser
    if name == "main":
        from arxiv_browser.cli import main as cli_main

        return cli_main
    raise AttributeError(f"module 'arxiv_browser' has no attribute {name!r}")
