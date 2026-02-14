"""Internal runtime helpers for TUI widget refs and refresh orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from textual.widget import Widget
from textual.widgets import Input, Label, OptionList

from arxiv_browser.widgets import BookmarkTabBar, ContextFooter, DateNavigator, FilterPillBar
from arxiv_browser.widgets.details import PaperDetails


@dataclass(slots=True)
class UiRefs:
    """Cached widget references for hot UI paths.

    These refs are internal-only and must not be treated as a public API.
    """

    search_input: Input | None = None
    search_container: Widget | None = None
    paper_list: OptionList | None = None
    list_header: Label | None = None
    status_bar: Label | None = None
    footer: ContextFooter | None = None
    date_navigator: DateNavigator | None = None
    filter_pill_bar: FilterPillBar | None = None
    bookmark_bar: BookmarkTabBar | None = None
    paper_details: PaperDetails | None = None

    def reset(self) -> None:
        """Clear all cached refs (for unmount/teardown)."""
        self.search_input = None
        self.search_container = None
        self.paper_list = None
        self.list_header = None
        self.status_bar = None
        self.footer = None
        self.date_navigator = None
        self.filter_pill_bar = None
        self.bookmark_bar = None
        self.paper_details = None


@dataclass(slots=True)
class UiRefreshCoordinator:
    """Small boundary object for orchestrating common refresh sequences."""

    refresh_list_view: Callable[[], None]
    update_list_header: Callable[[str], None]
    update_status_bar: Callable[[], None]
    update_filter_pills: Callable[[str], None]
    refresh_detail_pane: Callable[[], None]
    refresh_current_list_item: Callable[[], None]

    def apply_filter_refresh(self, query: str) -> None:
        """Run the post-filter refresh sequence in one place."""
        self.refresh_list_view()
        self.update_list_header(query)
        self.update_status_bar()
        self.update_filter_pills(query)

    def refresh_detail_and_list_item(self) -> None:
        """Refresh both detail pane and current list item."""
        self.refresh_detail_pane()
        self.refresh_current_list_item()


__all__ = [
    "UiRefreshCoordinator",
    "UiRefs",
]
