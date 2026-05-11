"""Tests for terminal PDF preview rendering helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from pypdf import PdfWriter

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.export import get_pdf_download_path
from arxiv_browser.modals.pdf import PdfPreviewScreen
from arxiv_browser.models import UserConfig
from arxiv_browser.pdf_preview import (
    PDF_PREVIEW_CACHE_DIRNAME,
    PdfPreviewError,
    PdfPreviewPage,
    build_pdf_preview_pages,
    render_pdf_pages_to_pngs,
    render_png_as_terminal_markup,
)


def test_halfblock_and_ascii_renderers_emit_terminal_text(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (255, 255, 255))
    image.save(image_path)

    ascii_preview = render_png_as_terminal_markup(
        image_path, max_width=2, max_rows=2, ascii_mode=True
    )
    halfblock_preview = render_png_as_terminal_markup(
        image_path,
        max_width=2,
        max_rows=1,
        ascii_mode=False,
    )

    assert ascii_preview
    assert "[" not in ascii_preview
    assert "▀" in halfblock_preview
    assert " on #" in halfblock_preview


def test_pdf_pages_render_to_preview_cache(make_paper, tmp_path: Path) -> None:
    paper = make_paper(arxiv_id="2401.77777")
    config = UserConfig(pdf_download_dir=str(tmp_path / "pdfs"), pdf_preview_max_pages=1)
    pdf_path = get_pdf_download_path(paper, config)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    writer.add_blank_page(width=120, height=120)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    pages = build_pdf_preview_pages(
        pdf_path=pdf_path,
        paper=paper,
        config=config,
        max_pages=1,
    )

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image_path.parent.name == PDF_PREVIEW_CACHE_DIRNAME
    assert pages[0].image_path.name == "2401.77777-p1.png"
    assert pages[0].image_path.is_file()
    assert pages[0].markup

    cached_paths = render_pdf_pages_to_pngs(
        pdf_path=pdf_path,
        paper=paper,
        config=config,
        max_pages=1,
    )
    assert cached_paths == [pages[0].image_path]


def test_pdf_preview_reports_invalid_pdf(make_paper, tmp_path: Path) -> None:
    paper = make_paper(arxiv_id="2401.77778")
    config = UserConfig(pdf_download_dir=str(tmp_path / "pdfs"))
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a pdf")

    with pytest.raises(PdfPreviewError):
        render_pdf_pages_to_pngs(
            pdf_path=bad_pdf,
            paper=paper,
            config=config,
            max_pages=1,
        )


@pytest.mark.asyncio
async def test_pdf_preview_modal_compose_and_close(make_paper, tmp_path: Path) -> None:
    app = ArxivBrowser([make_paper()], restore_session=False)
    page = PdfPreviewPage(page_number=1, image_path=tmp_path / "page.png", markup="preview")
    screen = PdfPreviewScreen(make_paper(title="A very interesting PDF"), [page])

    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause(0.05)

        assert "PDF Preview" in str(screen.query_one("#pdf-preview-title").render())
        assert "Page 1" in str(screen.query(".pdf-preview-page-title").first().render())
        assert "preview" in str(screen.query(".pdf-preview-page").first().render())

    screen.dismiss = MagicMock()
    screen.action_close()
    screen.dismiss.assert_called_once_with(None)
