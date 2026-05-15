"""Discovery analytics and author-profile modals."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Label, ListItem, ListView, Static

from arxiv_browser.authors import AuthorProfile
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.query import escape_rich_text, truncate_text
from arxiv_browser.trend_radar import TrendRadarReport, render_sparkline


class TrendRadarModal(ModalBase[None]):
    """Read-only local-history analytics overlay."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = """
    TrendRadarModal {
        align: center middle;
    }

    #trend-radar-dialog {
        width: 86;
        max-width: 95%;
        height: 82%;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #trend-radar-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #trend-radar-body {
        height: 1fr;
        overflow-y: auto;
    }

    #trend-radar-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, report: TrendRadarReport) -> None:
        super().__init__()
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="trend-radar-dialog"):
            yield Label("Trend Radar", id="trend-radar-title")
            yield Static(_render_trend_report(self._report), id="trend-radar-body")
            yield Static("Esc/q close", id="trend-radar-footer")

    def action_close(self) -> None:
        """Close the overlay."""
        self.dismiss(None)


class AuthorListItem(ListItem):
    """Selectable author row."""

    def __init__(self, author: str, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.author = author


class AuthorPickerModal(ModalBase[str | None]):
    """Pick one author from the current paper."""

    BINDINGS = [
        Binding("enter", "choose", "Choose"),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel", show=False),
    ]

    CSS = """
    AuthorPickerModal {
        align: center middle;
    }

    #author-picker-dialog {
        width: 64;
        max-width: 95%;
        height: 70%;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #author-picker-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #author-picker-list {
        height: 1fr;
        background: $th-panel;
        border: none;
    }
    """

    def __init__(self, authors: list[str], title: str = "Choose Author") -> None:
        super().__init__()
        self._authors = authors
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="author-picker-dialog"):
            yield Label(self._title, id="author-picker-title")
            yield ListView(id="author-picker-list")
            yield Static("Enter choose | Esc/q cancel", id="author-picker-footer")

    def on_mount(self) -> None:
        list_view = self.query_one("#author-picker-list", ListView)
        for author in self._authors:
            list_view.mount(AuthorListItem(author, Label(escape_rich_text(author))))
        if list_view.children:
            list_view.index = 0
            list_view.focus()

    def action_choose(self) -> None:
        list_view = self.query_one("#author-picker-list", ListView)
        item = list_view.highlighted_child
        if isinstance(item, AuthorListItem):
            self.dismiss(item.author)

    @on(ListView.Selected, "#author-picker-list")
    def on_author_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, AuthorListItem):
            self.dismiss(event.item.author)


class AuthorProfileModal(ModalBase[None]):
    """Read-only profile for one author across loaded papers."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = """
    AuthorProfileModal {
        align: center middle;
    }

    #author-profile-dialog {
        width: 86;
        max-width: 95%;
        height: 82%;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #author-profile-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    #author-profile-body {
        height: 1fr;
        overflow-y: auto;
    }

    #author-profile-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def __init__(self, profile: AuthorProfile) -> None:
        super().__init__()
        self._profile = profile

    def compose(self) -> ComposeResult:
        with Vertical(id="author-profile-dialog"):
            yield Label(f"Author Profile: {self._profile.author}", id="author-profile-title")
            yield Static(_render_author_profile(self._profile), id="author-profile-body")
            yield Static("Esc/q close", id="author-profile-footer")

    def action_close(self) -> None:
        """Close the profile."""
        self.dismiss(None)


def _render_trend_report(report: TrendRadarReport) -> str:
    if report.history_file_count == 0:
        return "[dim]No local history files are available for Trend Radar.[/]"
    date_label = _date_range_label(report)
    lines = [
        f"[bold]History[/] {report.history_file_count} files | {report.total_papers} papers",
        f"[dim]{date_label} | recent {report.recent_file_count}, previous {report.previous_file_count}[/]",
        "",
        "[bold]Growing Categories[/]",
    ]
    if report.category_trends:
        for trend in report.category_trends:
            spark = render_sparkline(trend.counts)
            delta = f"+{trend.delta}" if trend.delta >= 0 else str(trend.delta)
            lines.append(
                f"  {escape_rich_text(trend.category):<12} {spark} "
                f"[bold]{trend.recent_count}[/] ({delta})"
            )
    else:
        lines.append("  [dim]No category data.[/]")
    lines.extend(["", "[bold]Top Authors[/]"])
    lines.extend(_count_lines((item.name, item.count) for item in report.top_authors))
    lines.extend(["", "[bold]Hot Topics[/]"])
    lines.extend(_count_lines((item.bigram, item.count) for item in report.hot_bigrams))
    return "\n".join(lines)


def _render_author_profile(profile: AuthorProfile) -> str:
    paper_count = len(profile.papers)
    lines = [
        f"[bold]Papers[/] {paper_count}",
        f"[bold]Cached citations[/] {profile.total_cached_citations} "
        f"[dim]across {profile.citation_coverage} paper(s)[/]",
        "",
        "[bold]Co-authors[/]",
    ]
    lines.extend(_count_lines((item.name, item.count) for item in profile.coauthors[:10]))
    lines.extend(["", "[bold]Library Papers[/]"])
    if not profile.papers:
        lines.append("  [dim]No papers in the loaded library match this author.[/]")
        return "\n".join(lines)
    for record in profile.papers:
        paper = record.paper
        cites = "" if record.citation_count is None else f" | S2:{record.citation_count}"
        title = escape_rich_text(truncate_text(paper.title, 74))
        lines.append(f"  [dim]{escape_rich_text(paper.date)}[/] {title}")
        lines.append(f"    [dim]{escape_rich_text(paper.arxiv_id)}{cites}[/]")
    return "\n".join(lines)


def _count_lines(items) -> list[str]:
    rows = [f"  {escape_rich_text(name)} [dim]({count})[/]" for name, count in items]
    return rows or ["  [dim]No data.[/]"]


def _date_range_label(report: TrendRadarReport) -> str:
    if not report.dates:
        return "no dates"
    return f"{report.dates[0].isoformat()} to {report.dates[-1].isoformat()}"


__all__ = [
    "AuthorListItem",
    "AuthorPickerModal",
    "AuthorProfileModal",
    "TrendRadarModal",
]
