#!/usr/bin/env python3
"""arXiv Paper Browser TUI - Browse arXiv papers from a text file.

Usage:
    python arxiv_browser.py                    # Use default arxiv.txt
    python arxiv_browser.py -i papers.txt      # Use custom file

Key bindings:
    /       - Toggle search (prefix with "cat:" to filter by category)
    o       - Open selected paper(s) in browser
    c       - Copy selected paper(s) to clipboard
    space   - Toggle selection
    a       - Select all visible
    u       - Clear selection
    s       - Cycle sort order (title/date/arxiv_id)
    j/k     - Navigate down/up (vim-style)
    q       - Quit
"""

import argparse
import functools
import platform
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

# ============================================================================
# Constants
# ============================================================================

# Placeholder using control characters (cannot appear in academic text)
_ESCAPED_DOLLAR = "\x00ESCAPED_DOLLAR\x00"

# UI Layout constants
MIN_LIST_WIDTH = 50
MAX_LIST_WIDTH = 100
CLIPBOARD_SEPARATOR = "=" * 80

# Sort order options
SORT_OPTIONS = ["title", "date", "arxiv_id"]


@dataclass(slots=True)
class Paper:
    """Represents an arXiv paper entry."""
    arxiv_id: str
    date: str
    title: str
    authors: str
    categories: str
    comments: str | None
    abstract: str
    url: str


# Pre-compiled regex patterns for LaTeX cleaning (performance optimization)
# Each tuple is (pattern, replacement) applied in order
_LATEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Text formatting commands: \textbf{text} -> text, \emph{text} -> text
    (re.compile(r"\\text(?:tt|bf|it|rm|sf)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\emph\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:bf|it|tt|rm|sf)\{([^}]*)\}"), r"\1"),
    # Escaped dollar signs: \$ -> placeholder (restored later)
    (re.compile(r"\\\$"), _ESCAPED_DOLLAR),
    # Math mode: $x^2$ -> x^2 (extracts content, non-greedy)
    (re.compile(r"\$([^$]*)\$"), r"\1"),
    # Restore escaped dollar signs: placeholder -> $
    (re.compile(re.escape(_ESCAPED_DOLLAR)), "$"),
    # Accented characters: \'e -> é, \"a -> ä, \c{c} -> ç, etc.
    (re.compile(r"\\c\{c\}"), "ç"),
    (re.compile(r"\\c\{C\}"), "Ç"),
    (re.compile(r"\\'e"), "é"),
    (re.compile(r"\\'a"), "á"),
    (re.compile(r"\\'o"), "ó"),
    (re.compile(r"\\'i"), "í"),
    (re.compile(r"\\'u"), "ú"),
    (re.compile(r'\\"\{a\}'), "ä"),
    (re.compile(r'\\"\{o\}'), "ö"),
    (re.compile(r'\\"\{u\}'), "ü"),
    (re.compile(r"\\~n"), "ñ"),
    (re.compile(r"\\&"), "&"),
    # Generic command with braces: \foo{content} -> content
    (re.compile(r"\\[a-zA-Z]+\{([^}]*)\}"), r"\1"),
    # Standalone commands: \foo -> (removed)
    (re.compile(r"\\[a-zA-Z]+(?:\s|$)"), " "),
]

# Pre-compiled regex patterns for parsing arXiv entries
# Matches: "arXiv:2301.12345" -> captures "2301.12345"
_ARXIV_ID_PATTERN = re.compile(r"arXiv:(\S+)")
# Matches: "Date: Mon, 15 Jan 2024 (v1)" -> captures "Mon, 15 Jan 2024"
_DATE_PATTERN = re.compile(r"Date:\s*(.+?)(?:\s*\(|$)", re.MULTILINE)
# Matches multi-line title up to "Authors:" label
_TITLE_PATTERN = re.compile(r"Title:\s*(.+?)(?=\nAuthors:)", re.DOTALL)
# Matches multi-line authors up to "Categories:" label
_AUTHORS_PATTERN = re.compile(r"Authors:\s*(.+?)(?=\nCategories:)", re.DOTALL)
# Matches: "Categories: cs.AI cs.LG" -> captures "cs.AI cs.LG"
_CATEGORIES_PATTERN = re.compile(r"Categories:\s*(.+?)$", re.MULTILINE)
# Matches: "Comments: 10 pages, 5 figures" -> captures "10 pages, 5 figures"
_COMMENTS_PATTERN = re.compile(r"Comments:\s*(.+?)$", re.MULTILINE)
# Matches abstract text between \\ markers after Categories/Comments line
_ABSTRACT_PATTERN = re.compile(r"(?:Categories|Comments):[^\n]*\n\\\\\n(.+?)\n\\\\", re.DOTALL)
# Matches: "( https://arxiv.org/abs/2301.12345" -> captures the URL
_URL_PATTERN = re.compile(r"\(\s*(https://arxiv\.org/abs/\S+)")
# Matches 70+ dashes used as entry separator
_ENTRY_SEPARATOR = re.compile(r"-{70,}")


def clean_latex(text: str) -> str:
    """Remove or convert common LaTeX commands to plain text.

    Args:
        text: Input text potentially containing LaTeX commands.

    Returns:
        Plain text with LaTeX commands removed or converted.
    """
    # Short-circuit: skip regex processing for text without LaTeX markers
    if "\\" not in text and "$" not in text:
        return " ".join(text.split())

    for pattern, replacement in _LATEX_PATTERNS:
        text = pattern.sub(replacement, text)

    # Clean up extra whitespace
    return " ".join(text.split())


def parse_arxiv_file(filepath: Path) -> list[Paper]:
    """Parse arxiv.txt and return a list of Paper objects."""
    # Use errors="replace" to handle any non-UTF-8 characters gracefully
    content = filepath.read_text(encoding="utf-8", errors="replace")
    papers = []

    # Split by paper separator using pre-compiled pattern
    entries = _ENTRY_SEPARATOR.split(content)

    for entry in entries:
        entry = entry.strip()
        if not entry or not entry.startswith("\\"):
            continue

        # Extract arXiv ID
        arxiv_match = _ARXIV_ID_PATTERN.search(entry)
        if not arxiv_match:
            continue
        arxiv_id = arxiv_match.group(1)

        # Extract date
        date_match = _DATE_PATTERN.search(entry)
        date = date_match.group(1).strip() if date_match else ""

        # Extract title (may span multiple lines)
        title_match = _TITLE_PATTERN.search(entry)
        if title_match:
            title = " ".join(title_match.group(1).split())
        else:
            title = ""

        # Extract authors (may span multiple lines)
        authors_match = _AUTHORS_PATTERN.search(entry)
        if authors_match:
            authors = " ".join(authors_match.group(1).split())
        else:
            authors = ""

        # Extract categories
        categories_match = _CATEGORIES_PATTERN.search(entry)
        categories = categories_match.group(1).strip() if categories_match else ""

        # Extract comments (optional)
        comments_match = _COMMENTS_PATTERN.search(entry)
        comments = comments_match.group(1).strip() if comments_match else None

        # Extract abstract (text between \\ markers)
        abstract_match = _ABSTRACT_PATTERN.search(entry)
        if abstract_match:
            abstract = " ".join(abstract_match.group(1).split())
        else:
            abstract = ""

        # Extract URL
        url_match = _URL_PATTERN.search(entry)
        url = url_match.group(1) if url_match else f"https://arxiv.org/abs/{arxiv_id}"

        papers.append(Paper(
            arxiv_id=arxiv_id,
            date=date,
            title=clean_latex(title),
            authors=clean_latex(authors),
            categories=categories,
            comments=clean_latex(comments) if comments else None,
            abstract=clean_latex(abstract),
            url=url,
        ))

    return papers


# Category color mapping (Monokai-inspired palette)
CATEGORY_COLORS = {
    "cs.AI": "#f92672",  # Monokai pink
    "cs.CL": "#66d9ef",  # Monokai blue
    "cs.LG": "#a6e22e",  # Monokai green
    "cs.CV": "#e6db74",  # Monokai yellow
    "cs.SE": "#ae81ff",  # Monokai purple
    "cs.HC": "#fd971f",  # Monokai orange
    "cs.RO": "#66d9ef",  # Monokai blue
    "cs.NE": "#f92672",  # Monokai pink
    "cs.IR": "#ae81ff",  # Monokai purple
    "cs.CR": "#fd971f",  # Monokai orange
}

@functools.lru_cache(maxsize=256)
def format_categories(categories: str) -> str:
    """Format categories with colors. Results are automatically cached via lru_cache."""
    parts = []
    for cat in categories.split():
        color = CATEGORY_COLORS.get(cat, "#888888")
        parts.append(f"[{color}]{cat}[/]")
    return " ".join(parts)


class PaperListItem(ListItem):
    """A list item displaying a paper title and URL."""

    def __init__(self, paper: Paper, index: int, selected: bool = False) -> None:
        super().__init__()
        self.paper = paper
        self.index = index
        self._selected = selected
        if selected:
            self.add_class("selected")

    @property
    def is_selected(self) -> bool:
        return self._selected

    def _get_title_text(self) -> str:
        """Get the formatted title text based on selection state."""
        if self._selected:
            return f"[#a6e22e]●[/] {self.paper.title}"  # Monokai green
        return self.paper.title

    def _get_meta_text(self) -> str:
        """Get the formatted metadata text."""
        return f"[dim]{self.paper.arxiv_id}[/]  {format_categories(self.paper.categories)}"

    def _update_selection_class(self) -> None:
        """Update the CSS class based on selection state."""
        if self._selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")

    def toggle_selected(self) -> bool:
        """Toggle selection state and return new state."""
        self._selected = not self._selected
        self._update_selection_class()
        self._update_display()
        return self._selected

    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self._selected = selected
        self._update_selection_class()
        self._update_display()

    def _update_display(self) -> None:
        """Update the visual display based on selection state."""
        title_widget = self.query_one(".paper-title", Static)
        meta_widget = self.query_one(".paper-meta", Static)
        title_widget.update(self._get_title_text())
        meta_widget.update(self._get_meta_text())

    def compose(self) -> ComposeResult:
        yield Static(self._get_title_text(), classes="paper-title")
        yield Static(self._get_meta_text(), classes="paper-meta")


class PaperDetails(Static):
    """Widget to display full paper details."""

    def __init__(self) -> None:
        super().__init__()
        self._paper: Paper | None = None

    def update_paper(self, paper: Paper | None) -> None:
        """Update the displayed paper details."""
        self._paper = paper
        if paper is None:
            self.update("[dim italic]Select a paper to view details[/]")
            return

        lines = []

        # Title section (Monokai foreground)
        lines.append(f"[bold #f8f8f2]{paper.title}[/]")
        lines.append("")

        # Metadata section (Monokai blue for labels, purple for values)
        lines.append(f"[bold #66d9ef]arXiv:[/] [#ae81ff]{paper.arxiv_id}[/]")
        lines.append(f"[bold #66d9ef]Date:[/] {paper.date}")
        lines.append(f"[bold #66d9ef]Categories:[/] {format_categories(paper.categories)}")
        if paper.comments:
            lines.append(f"[bold #66d9ef]Comments:[/] [dim]{paper.comments}[/]")
        lines.append("")

        # Authors section (Monokai green)
        lines.append("[bold #a6e22e]Authors[/]")
        lines.append(f"[#f8f8f2]{paper.authors}[/]")
        lines.append("")

        # Abstract section (Monokai orange)
        lines.append("[bold #fd971f]Abstract[/]")
        lines.append(f"[#f8f8f2]{paper.abstract}[/]")
        lines.append("")

        # URL section (Monokai pink/red for label, blue for URL)
        lines.append(f"[bold #f92672]URL:[/] [#66d9ef]{paper.url}[/]")

        self.update("\n".join(lines))

    @property
    def paper(self) -> Paper | None:
        return self._paper


class ArxivBrowser(App):
    """A TUI application to browse arXiv papers."""

    TITLE = "arXiv Paper Browser"

    # Monokai color theme
    CSS = """
    Screen {
        background: #272822;
    }

    Header {
        background: #3e3d32;
        color: #f8f8f2;
    }

    Footer {
        background: #3e3d32;
    }

    #main-container {
        height: 1fr;
    }

    #left-pane {
        width: 2fr;
        min-width: 50;
        max-width: 100;
        height: 100%;
        border: round #75715e;
        background: #1e1e1e;
    }

    #right-pane {
        width: 3fr;
        height: 100%;
        border: round #75715e;
        background: #1e1e1e;
    }

    #list-header {
        padding: 1 2;
        background: #3e3d32;
        color: #66d9ef;
        text-style: bold;
        border-bottom: solid #75715e;
    }

    #details-header {
        padding: 1 2;
        background: #3e3d32;
        color: #e6db74;
        text-style: bold;
        border-bottom: solid #75715e;
    }

    #paper-list {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #details-scroll {
        height: 1fr;
        padding: 1 2;
    }

    #search-container {
        height: auto;
        padding: 1;
        background: #3e3d32;
        display: none;
    }

    #search-container.visible {
        display: block;
    }

    #search-input {
        width: 100%;
        border: round #66d9ef;
        background: #272822;
    }

    #search-input:focus {
        border: round #e6db74;
    }

    PaperListItem {
        padding: 1 1;
        height: auto;
        border-bottom: dashed #3e3d32;
    }

    PaperListItem:hover {
        background: #3e3d32;
    }

    PaperListItem.-highlight {
        background: #49483e;
    }

    ListView > ListItem.--highlight {
        background: #49483e;
    }

    ListView:focus > ListItem.--highlight {
        background: #5a5950;
    }

    .paper-title {
        color: #f8f8f2;
        text-style: bold;
    }

    .paper-meta {
        color: #75715e;
        margin-top: 0;
    }

    PaperListItem.selected {
        background: #3d4a32;
    }

    PaperListItem.selected.--highlight {
        background: #4d5a42;
    }

    PaperDetails {
        padding: 0;
    }

    VerticalScroll {
        scrollbar-background: #3e3d32;
        scrollbar-color: #75715e;
        scrollbar-color-hover: #a6a68a;
        scrollbar-color-active: #66d9ef;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "toggle_search", "Search"),
        Binding("escape", "cancel_search", "Cancel", show=False),
        Binding("o", "open_url", "Open Selected"),
        Binding("c", "copy_selected", "Copy"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("space", "toggle_select", "Select", show=False),
        Binding("a", "select_all", "Select All"),
        Binding("u", "clear_selection", "Clear Selection"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    DEBOUNCE_DELAY = 0.3  # seconds

    def __init__(self, papers: list[Paper]) -> None:
        super().__init__()
        self.all_papers = papers
        self.filtered_papers = papers.copy()
        # Build O(1) lookup dict for papers by arxiv_id
        self._papers_by_id: dict[str, Paper] = {p.arxiv_id: p for p in papers}
        self.selected_ids: set[str] = set()  # Track selected arxiv_ids
        self._search_timer: Timer | None = None
        self._pending_query: str = ""
        self._sort_index: int = 0  # Index into SORT_OPTIONS

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-pane"):
                yield Label(f" Papers ({len(self.all_papers)} total)", id="list-header")
                with Vertical(id="search-container"):
                    yield Input(placeholder=" Filter by title/author (or cat:cs.AI for category)", id="search-input")
                yield ListView(
                    *[PaperListItem(p, i) for i, p in enumerate(self.filtered_papers)],
                    id="paper-list"
                )
            with Vertical(id="right-pane"):
                yield Label(" Paper Details", id="details-header")
                with VerticalScroll(id="details-scroll"):
                    yield PaperDetails()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.sub_title = f"{len(self.all_papers)} papers loaded"
        # Select first item if available
        list_view = self.query_one("#paper-list", ListView)
        if list_view.children:
            list_view.index = 0

    def on_unmount(self) -> None:
        """Called when app is unmounted. Clean up timers."""
        if self._search_timer is not None:
            self._search_timer.stop()
            self._search_timer = None

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle paper selection."""
        if isinstance(event.item, PaperListItem):
            details = self.query_one(PaperDetails)
            details.update_paper(event.item.paper)

    @on(ListView.Highlighted)
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle paper highlight (keyboard navigation)."""
        if isinstance(event.item, PaperListItem):
            details = self.query_one(PaperDetails)
            details.update_paper(event.item.paper)

    def action_toggle_search(self) -> None:
        """Toggle search input visibility."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
        else:
            container.add_class("visible")
            self.query_one("#search-input", Input).focus()

    def action_cancel_search(self) -> None:
        """Cancel search and hide input."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
            self._apply_filter("")

    def action_cursor_down(self) -> None:
        """Move cursor down (vim-style j key)."""
        list_view = self.query_one("#paper-list", ListView)
        list_view.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up (vim-style k key)."""
        list_view = self.query_one("#paper-list", ListView)
        list_view.action_cursor_up()

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        self._apply_filter(event.value)
        # Hide search after submission
        self.query_one("#search-container").remove_class("visible")
        # Focus the list
        self.query_one("#paper-list", ListView).focus()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Handle search input change with debouncing."""
        self._pending_query = event.value
        # Cancel existing timer if any
        if self._search_timer is not None:
            self._search_timer.stop()
        # Set new timer for debounced filter
        self._search_timer = self.set_timer(
            self.DEBOUNCE_DELAY,
            self._debounced_filter,
        )

    def _debounced_filter(self) -> None:
        """Apply filter after debounce delay."""
        self._search_timer = None
        self._apply_filter(self._pending_query)

    def _format_header_text(self, query: str = "") -> str:
        """Format the header text with paper count and selection info."""
        selection_info = f" [{len(self.selected_ids)} selected]" if self.selected_ids else ""
        sort_info = f" [dim]sorted by {SORT_OPTIONS[self._sort_index]}[/]"
        if query:
            return f" Papers ({len(self.filtered_papers)}/{len(self.all_papers)}){selection_info}{sort_info}"
        return f" Papers ({len(self.all_papers)} total){selection_info}{sort_info}"

    def _apply_filter(self, query: str) -> None:
        """Filter papers by title, author, or category (prefix with 'cat:')."""
        query = query.strip()
        list_view = self.query_one("#paper-list", ListView)

        if not query:
            self.filtered_papers = self.all_papers.copy()
        elif query.lower().startswith("cat:"):
            # Category filter: "cat:cs.AI" matches papers with cs.AI in categories
            category = query[4:].strip()
            self.filtered_papers = [
                p for p in self.all_papers
                if category.lower() in p.categories.lower()
            ]
        else:
            # Title/author filter
            query_lower = query.lower()
            self.filtered_papers = [
                p for p in self.all_papers
                if query_lower in p.title.lower() or query_lower in p.authors.lower()
            ]

        # Apply current sort order
        self._sort_papers()

        # Update list view, preserving selection state
        list_view.clear()
        for i, paper in enumerate(self.filtered_papers):
            is_selected = paper.arxiv_id in self.selected_ids
            list_view.append(PaperListItem(paper, i, selected=is_selected))

        # Update header
        self.query_one("#list-header", Label).update(self._format_header_text(query))

        # Select first item if available
        if list_view.children:
            list_view.index = 0

    def action_toggle_select(self) -> None:
        """Toggle selection of the currently highlighted paper."""
        list_view = self.query_one("#paper-list", ListView)
        if list_view.highlighted_child is None:
            return

        item = list_view.highlighted_child
        if isinstance(item, PaperListItem):
            new_state = item.toggle_selected()
            if new_state:
                self.selected_ids.add(item.paper.arxiv_id)
            else:
                self.selected_ids.discard(item.paper.arxiv_id)
            self._update_header()

    def action_select_all(self) -> None:
        """Select all currently visible papers."""
        list_view = self.query_one("#paper-list", ListView)
        for item in list_view.children:
            if isinstance(item, PaperListItem):
                item.set_selected(True)
                self.selected_ids.add(item.paper.arxiv_id)
        self._update_header()

    def action_clear_selection(self) -> None:
        """Clear all selections."""
        list_view = self.query_one("#paper-list", ListView)
        for item in list_view.children:
            if isinstance(item, PaperListItem):
                item.set_selected(False)
        self.selected_ids.clear()
        self._update_header()

    def _sort_papers(self) -> None:
        """Sort filtered_papers according to current sort order."""
        sort_key = SORT_OPTIONS[self._sort_index]
        if sort_key == "title":
            self.filtered_papers.sort(key=lambda p: p.title.lower())
        elif sort_key == "date":
            # Sort by date descending (newest first)
            self.filtered_papers.sort(key=lambda p: p.date, reverse=True)
        elif sort_key == "arxiv_id":
            # Sort by arxiv_id descending (newest first)
            self.filtered_papers.sort(key=lambda p: p.arxiv_id, reverse=True)

    def action_cycle_sort(self) -> None:
        """Cycle through sort options: title, date, arxiv_id."""
        self._sort_index = (self._sort_index + 1) % len(SORT_OPTIONS)
        sort_key = SORT_OPTIONS[self._sort_index]
        self.notify(f"Sorted by {sort_key}", title="Sort")

        # Re-sort and refresh the list
        self._sort_papers()
        list_view = self.query_one("#paper-list", ListView)
        list_view.clear()
        for i, paper in enumerate(self.filtered_papers):
            is_selected = paper.arxiv_id in self.selected_ids
            list_view.append(PaperListItem(paper, i, selected=is_selected))

        self._update_header()

        # Maintain focus on first item
        if list_view.children:
            list_view.index = 0

    def _update_header(self) -> None:
        """Update header with selection count and sort info."""
        query = self.query_one("#search-input", Input).value.strip()
        self.query_one("#list-header", Label).update(self._format_header_text(query))

    def _get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Look up a paper by its arXiv ID. O(1) dict lookup."""
        return self._papers_by_id.get(arxiv_id)

    def action_open_url(self) -> None:
        """Open selected papers' URLs in the default browser."""
        # If papers are selected, open all of them
        if self.selected_ids:
            for arxiv_id in self.selected_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper:
                    webbrowser.open(paper.url)
            self.notify(f"Opening {len(self.selected_ids)} papers", title="Browser")
        else:
            # Otherwise, open the currently highlighted paper
            details = self.query_one(PaperDetails)
            if details.paper:
                webbrowser.open(details.paper.url)
                self.notify(f"Opening {details.paper.arxiv_id}", title="Browser")

    def _format_paper_for_clipboard(self, paper: Paper) -> str:
        """Format a paper's metadata for clipboard export."""
        lines = [
            f"Title: {paper.title}",
            f"Authors: {paper.authors}",
            f"arXiv: {paper.arxiv_id}",
            f"Date: {paper.date}",
            f"Categories: {paper.categories}",
        ]
        if paper.comments:
            lines.append(f"Comments: {paper.comments}")
        lines.append(f"URL: {paper.url}")
        lines.append("")
        lines.append(f"Abstract: {paper.abstract}")
        return "\n".join(lines)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard. Returns True on success."""
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(
                    ["pbcopy"],
                    input=text.encode("utf-8"),
                    check=True,
                )
            elif system == "Linux":
                # Try xclip first, then xsel
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=text.encode("utf-8"),
                        check=True,
                    )
                except FileNotFoundError:
                    subprocess.run(
                        ["xsel", "--clipboard", "--input"],
                        input=text.encode("utf-8"),
                        check=True,
                    )
            elif system == "Windows":
                subprocess.run(
                    ["clip"],
                    input=text.encode("utf-16"),
                    check=True,
                )
            else:
                return False
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def action_copy_selected(self) -> None:
        """Copy selected papers' metadata to clipboard."""
        # Get papers to copy
        if self.selected_ids:
            papers_to_copy = [
                self._get_paper_by_id(arxiv_id)
                for arxiv_id in self.selected_ids
            ]
            papers_to_copy = [p for p in papers_to_copy if p is not None]
        else:
            # Copy currently highlighted paper if none selected
            details = self.query_one(PaperDetails)
            if details.paper:
                papers_to_copy = [details.paper]
            else:
                self.notify("No paper selected", title="Copy", severity="warning")
                return

        if not papers_to_copy:
            self.notify("No papers to copy", title="Copy", severity="warning")
            return

        # Format papers with separator between them
        separator = f"\n\n{CLIPBOARD_SEPARATOR}\n\n"
        formatted = separator.join(
            self._format_paper_for_clipboard(p) for p in papers_to_copy
        )

        # Copy to clipboard
        if self._copy_to_clipboard(formatted):
            count = len(papers_to_copy)
            self.notify(
                f"Copied {count} paper{'s' if count > 1 else ''} to clipboard",
                title="Copy",
            )
        else:
            self.notify(
                "Failed to copy to clipboard",
                title="Copy",
                severity="error",
            )


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Browse arXiv papers from a text file in a TUI"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path(__file__).parent / "arxiv.txt",
        help="Input file containing arXiv metadata (default: arxiv.txt)",
    )
    args = parser.parse_args()

    arxiv_file: Path = args.input

    if not arxiv_file.is_file():
        if arxiv_file.is_dir():
            print(f"Error: {arxiv_file} is a directory, not a file", file=sys.stderr)
        else:
            print(f"Error: {arxiv_file} not found", file=sys.stderr)
        return 1

    papers = parse_arxiv_file(arxiv_file)

    if not papers:
        print("No papers found in the file", file=sys.stderr)
        return 1

    # Sort papers alphabetically by title
    papers.sort(key=lambda p: p.title.lower())

    app = ArxivBrowser(papers)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
