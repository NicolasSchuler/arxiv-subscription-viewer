"""Widget classes extracted from app.py for modular UI composition."""

from arxiv_browser.widgets.chrome import (
    DATE_NAV_WINDOW_SIZE,
    BookmarkTabBar,
    ContextFooter,
    DateNavigator,
    FilterPillBar,
    StatusBarState,
)
from arxiv_browser.widgets.chrome import (
    set_ascii_glyphs as set_ascii_chrome_glyphs,
)
from arxiv_browser.widgets.details import DETAIL_CACHE_MAX, DetailRenderState, PaperDetails
from arxiv_browser.widgets.listing import (
    PREVIEW_ABSTRACT_MAX_LEN,
    PaperHighlightTerms,
    PaperListItem,
    PaperRowRenderState,
    render_paper_option,
    set_ascii_icons,
)

__all__ = [
    "DATE_NAV_WINDOW_SIZE",
    "DETAIL_CACHE_MAX",
    "PREVIEW_ABSTRACT_MAX_LEN",
    "BookmarkTabBar",
    "ContextFooter",
    "DateNavigator",
    "DetailRenderState",
    "FilterPillBar",
    "PaperDetails",
    "PaperHighlightTerms",
    "PaperListItem",
    "PaperRowRenderState",
    "StatusBarState",
    "render_paper_option",
    "set_ascii_chrome_glyphs",
    "set_ascii_icons",
]
