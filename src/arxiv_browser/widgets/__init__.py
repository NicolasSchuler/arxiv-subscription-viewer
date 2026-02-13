"""Widget classes extracted from app.py for modular UI composition."""

from arxiv_browser.widgets.chrome import (
    DATE_NAV_WINDOW_SIZE,
    BookmarkTabBar,
    ContextFooter,
    DateNavigator,
    FilterPillBar,
)
from arxiv_browser.widgets.details import DETAIL_CACHE_MAX, PaperDetails
from arxiv_browser.widgets.listing import (
    PREVIEW_ABSTRACT_MAX_LEN,
    PaperListItem,
    render_paper_option,
)

__all__ = [
    "DATE_NAV_WINDOW_SIZE",
    "DETAIL_CACHE_MAX",
    "PREVIEW_ABSTRACT_MAX_LEN",
    "BookmarkTabBar",
    "ContextFooter",
    "DateNavigator",
    "FilterPillBar",
    "PaperDetails",
    "PaperListItem",
    "render_paper_option",
]
