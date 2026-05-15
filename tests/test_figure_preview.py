"""Tests for arXiv HTML figure preview helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from textual.widgets import Static

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.figure_preview import (
    FigurePreview,
    FigurePreviewError,
    build_figure_preview,
    cached_figure_is_valid,
    extract_first_html_figure,
    figure_preview_cache_path,
    save_figure_bytes_to_cache,
    validate_figure_content_type,
)
from arxiv_browser.modals.pdf import FigurePreviewScreen
from arxiv_browser.models import UserConfig


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_first_latex_figure_ignores_header_images() -> None:
    html = """
    <html>
      <body>
        <header><img src="/static/logo.png"></header>
        <figure class="ltx_figure">
          <figure><img src="2402.08954v1/first.png" alt="Refer to caption"></figure>
          <figcaption>Figure 1: First result</figcaption>
        </figure>
        <figure class="ltx_figure"><img src="2402.08954v1/second.png"></figure>
      </body>
    </html>
    """

    figure = extract_first_html_figure(html, "https://arxiv.org/html/2402.08954")

    assert figure.image_url == "https://arxiv.org/html/2402.08954v1/first.png"
    assert figure.caption == "Figure 1: First result"


def test_extract_first_latex_figure_reports_missing_figure() -> None:
    with pytest.raises(FigurePreviewError, match="No figure"):
        extract_first_html_figure("<img src='logo.png'>", "https://arxiv.org/html/1")


@pytest.mark.parametrize(
    "src",
    [
        "javascript:alert(1)",
        "data:image/png;base64,AAAA",
        "ftp://example.com/figure.png",
    ],
)
def test_extract_first_latex_figure_rejects_unsafe_image_url_schemes(src: str) -> None:
    html = f'<figure class="ltx_figure"><img src="{src}"></figure>'

    with pytest.raises(FigurePreviewError, match="Unsupported figure image URL scheme"):
        extract_first_html_figure(html, "https://arxiv.org/html/2401.00001")


def test_figure_cache_validation_and_rendering(make_paper, tmp_path: Path) -> None:
    paper = make_paper(arxiv_id="2401.99999")
    config = UserConfig(pdf_download_dir=str(tmp_path / "pdfs"))
    cache_path = figure_preview_cache_path(paper, config)

    assert cache_path.name == "2401.99999-figure.png"
    assert cached_figure_is_valid(cache_path) is False

    save_figure_bytes_to_cache(_png_bytes(), cache_path)
    assert cached_figure_is_valid(cache_path) is True

    figure = extract_first_html_figure(
        '<figure class="ltx_figure"><img src="fig.png"></figure>',
        "https://arxiv.org/html/2401.99999",
    )
    preview = build_figure_preview(figure=figure, image_path=cache_path)
    assert preview.markup
    assert preview.image_path == cache_path

    cache_path.write_bytes(b"not an image")
    assert cached_figure_is_valid(cache_path) is False
    with pytest.raises(FigurePreviewError, match="empty"):
        save_figure_bytes_to_cache(b"", cache_path)
    with pytest.raises(FigurePreviewError, match="Could not read figure image"):
        save_figure_bytes_to_cache(b"not an image", cache_path)


def test_validate_figure_content_type_rejects_unsupported_images() -> None:
    validate_figure_content_type("image/png; charset=binary")
    validate_figure_content_type("")
    with pytest.raises(FigurePreviewError, match="Unsupported"):
        validate_figure_content_type("image/svg+xml")


@pytest.mark.asyncio
async def test_figure_preview_screen_escapes_caption_and_closes(make_paper, tmp_path: Path) -> None:
    paper = make_paper(title="[red]Unsafe[/] Figure Paper")
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(_png_bytes())
    preview = FigurePreview(
        image_url="https://example.test/figure.png",
        image_path=image_path,
        caption="[bold]Untrusted[/] " + "caption " * 80,
        markup="terminal figure markup",
    )
    app = ArxivBrowser([paper], restore_session=False)
    modal = FigurePreviewScreen(paper, preview)

    async with app.run_test() as pilot:
        app.push_screen(modal)
        await pilot.pause(0.05)

        title = str(modal.query_one("#pdf-preview-title", Static).content)
        body = "\n".join(str(child.content) for child in modal.query("#pdf-preview-pages Static"))
        assert "\\[red]Unsafe\\[/]" in title
        assert "First HTML figure" in body
        assert "terminal figure markup" in body
        assert "\\[bold]Untrusted\\[/]" in body
        assert len(body) < 500

        await pilot.press("q")
        await pilot.pause(0.05)
        assert modal not in app.screen_stack
