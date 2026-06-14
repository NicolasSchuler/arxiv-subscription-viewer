"""PDF preview modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from arxiv_browser.figure_preview import FigurePreview
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import Paper
from arxiv_browser.pdf_preview import PdfPreviewPage
from arxiv_browser.query import escape_rich_text, truncate_text


class PdfPreviewScreen(ModalBase[None]):
    """Full-screen PDF page preview using terminal-safe block rendering."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = """
    #pdf-preview-dialog {
        width: 92%;
        height: 92%;
        min-width: 60;
        min-height: 20;
        /* deliberately accent-alt border, not the shared accent */
        border: tall $th-accent-alt;
        /* override shared 1 2 padding: preview needs tighter padding */
        padding: 0 1;
    }

    #pdf-preview-title {
        color: $th-accent-alt;
        height: auto;
    }

    #pdf-preview-pages {
        height: 1fr;
        background: $th-panel;
        padding: 1;
    }

    .pdf-preview-page-title {
        color: $th-accent;
        text-style: bold;
        margin: 1 0 0 0;
    }

    .pdf-preview-page {
        margin-bottom: 1;
    }

    #pdf-preview-footer {
        text-align: center;
        height: auto;
    }
    """

    def __init__(self, paper: Paper, pages: list[PdfPreviewPage]) -> None:
        super().__init__()
        self._paper = paper
        self._pages = pages

    def compose(self) -> ComposeResult:
        """Yield title, rendered pages, and footer."""
        title = escape_rich_text(truncate_text(self._paper.title, 90))
        with Vertical(id="pdf-preview-dialog", classes="modal-dialog"):
            yield Static(f"PDF Preview: {title}", id="pdf-preview-title", classes="modal-title")
            with VerticalScroll(id="pdf-preview-pages"):
                for page in self._pages:
                    yield Static(
                        f"Page {page.page_number}",
                        classes="pdf-preview-page-title",
                    )
                    yield Static(page.markup, classes="pdf-preview-page")
            yield Static("Close: Esc / q", id="pdf-preview-footer", classes="modal-footer")

    def action_close(self) -> None:
        """Close the preview."""
        self.dismiss(None)


class FigurePreviewScreen(ModalBase[None]):
    """Full-screen arXiv HTML figure preview using terminal-safe block rendering."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = PdfPreviewScreen.CSS.replace("PdfPreviewScreen", "FigurePreviewScreen")

    def __init__(self, paper: Paper, preview: FigurePreview) -> None:
        super().__init__()
        self._paper = paper
        self._preview = preview

    def compose(self) -> ComposeResult:
        """Yield title, rendered figure, caption, and footer."""
        title = escape_rich_text(truncate_text(self._paper.title, 90))
        with Vertical(id="pdf-preview-dialog", classes="modal-dialog"):
            yield Static(f"Figure Preview: {title}", id="pdf-preview-title", classes="modal-title")
            with VerticalScroll(id="pdf-preview-pages"):
                yield Static("First HTML figure", classes="pdf-preview-page-title")
                yield Static(self._preview.markup, classes="pdf-preview-page")
                if self._preview.caption:
                    caption = escape_rich_text(truncate_text(self._preview.caption, 240))
                    yield Static(f"[dim]{caption}[/]", classes="pdf-preview-page")
            yield Static("Close: Esc / q", id="pdf-preview-footer", classes="modal-footer")

    def action_close(self) -> None:
        """Close the preview."""
        self.dismiss(None)


__all__ = ["FigurePreviewScreen", "PdfPreviewScreen"]
