"""MIT-compatible PDF preview rendering helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from arxiv_browser._ascii import is_ascii_mode
from arxiv_browser.export import get_pdf_download_path
from arxiv_browser.models import Paper, UserConfig

logger = logging.getLogger(__name__)

PDF_PREVIEW_CACHE_DIRNAME = ".preview-cache"
PDF_PREVIEW_RENDER_SCALE = 2
PDF_PREVIEW_MAX_PIXEL_SIZE = (900, 1300)
HALFBLOCK_MAX_WIDTH = 88
HALFBLOCK_MAX_ROWS = 72
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_ASCII_RAMP = " .:-=+*#%@"


class PdfPreviewError(RuntimeError):
    """Raised when PDF preview rendering cannot produce pages."""


@dataclass(slots=True, frozen=True)
class PdfPreviewPage:
    """A rendered terminal-preview page."""

    page_number: int
    image_path: Path
    markup: str


def _preview_cache_dir(paper: Paper, config: UserConfig) -> Path:
    return get_pdf_download_path(paper, config).parent / PDF_PREVIEW_CACHE_DIRNAME


def _safe_pdf_stem(arxiv_id: str) -> str:
    return _SAFE_STEM_RE.sub("_", arxiv_id).strip("._") or "paper"


def _cached_page_is_fresh(png_path: Path, pdf_path: Path) -> bool:
    try:
        return png_path.is_file() and png_path.stat().st_mtime >= pdf_path.stat().st_mtime
    except OSError:
        return False


def render_pdf_pages_to_pngs(
    *,
    pdf_path: Path,
    paper: Paper,
    config: UserConfig,
    max_pages: int,
) -> list[Path]:
    """Render the first *max_pages* of *pdf_path* to cached PNG files."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise PdfPreviewError(f"PDF preview dependency missing: {exc.name}") from exc

    cache_dir = _preview_cache_dir(paper, config)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PdfPreviewError(f"Could not create preview cache: {exc}") from exc

    try:
        document = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise PdfPreviewError(f"Could not open PDF: {exc}") from exc

    rendered: list[Path] = []
    try:
        page_count = min(max(1, max_pages), len(document))
        for index in range(page_count):
            png_path = cache_dir / f"{_safe_pdf_stem(paper.arxiv_id)}-p{index + 1}.png"
            if _cached_page_is_fresh(png_path, pdf_path):
                rendered.append(png_path)
                continue
            page = document[index]
            try:
                bitmap = page.render(scale=PDF_PREVIEW_RENDER_SCALE)
                image = bitmap.to_pil().convert("RGB")
                image.thumbnail(PDF_PREVIEW_MAX_PIXEL_SIZE, Image.Resampling.LANCZOS)
                image.save(png_path, format="PNG")
            finally:
                close = getattr(page, "close", None)
                if callable(close):
                    close()
            rendered.append(png_path)
    except Exception as exc:
        logger.warning("PDF preview render failed for %s", paper.arxiv_id, exc_info=True)
        raise PdfPreviewError(f"Could not render PDF preview: {exc}") from exc
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    if not rendered:
        raise PdfPreviewError("PDF has no renderable pages")
    return rendered


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _coerce_rgb(pixel: object) -> tuple[int, int, int]:
    if isinstance(pixel, int | float):
        value = max(0, min(255, int(pixel)))
        return value, value, value
    if isinstance(pixel, tuple | list) and len(pixel) >= 3:
        return (
            max(0, min(255, int(pixel[0]))),
            max(0, min(255, int(pixel[1]))),
            max(0, min(255, int(pixel[2]))),
        )
    return 0, 0, 0


def _pixel_to_ascii(rgb: tuple[int, int, int]) -> str:
    luminance = int(rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    index = min(len(_ASCII_RAMP) - 1, luminance * len(_ASCII_RAMP) // 256)
    return _ASCII_RAMP[index]


def render_png_as_terminal_markup(
    image_path: Path,
    *,
    max_width: int = HALFBLOCK_MAX_WIDTH,
    max_rows: int = HALFBLOCK_MAX_ROWS,
    ascii_mode: bool | None = None,
) -> str:
    """Render a PNG as Rich markup using half-blocks or ASCII fallback."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise PdfPreviewError(f"Image dependency missing: {exc.name}") from exc

    ascii_mode = is_ascii_mode() if ascii_mode is None else ascii_mode
    try:
        with Image.open(image_path) as original:
            image = original.convert("RGB")
            width = min(max_width, max(1, image.width))
            scale = width / max(1, image.width)
            pixel_height = max(1, int(image.height * scale))
            if ascii_mode:
                pixel_height = min(pixel_height, max_rows)
            else:
                pixel_height = min(pixel_height, max_rows * 2)
            image = image.resize((width, pixel_height), Image.Resampling.LANCZOS)
            if ascii_mode:
                return "\n".join(
                    "".join(
                        _pixel_to_ascii(_coerce_rgb(image.getpixel((x, y)))) for x in range(width)
                    )
                    for y in range(pixel_height)
                )
            lines: list[str] = []
            for y in range(0, pixel_height, 2):
                parts: list[str] = []
                for x in range(width):
                    upper = _coerce_rgb(image.getpixel((x, y)))
                    lower = (
                        _coerce_rgb(image.getpixel((x, y + 1)))
                        if y + 1 < pixel_height
                        else (0, 0, 0)
                    )
                    parts.append(f"[{_rgb_to_hex(upper)} on {_rgb_to_hex(lower)}]▀[/]")
                lines.append("".join(parts))
            return "\n".join(lines)
    except OSError as exc:
        raise PdfPreviewError(f"Could not read preview image: {exc}") from exc


def build_pdf_preview_pages(
    *,
    pdf_path: Path,
    paper: Paper,
    config: UserConfig,
    max_pages: int,
) -> list[PdfPreviewPage]:
    """Render PDF pages and convert them to terminal-friendly markup."""
    png_paths = render_pdf_pages_to_pngs(
        pdf_path=pdf_path,
        paper=paper,
        config=config,
        max_pages=max_pages,
    )
    return [
        PdfPreviewPage(
            page_number=index,
            image_path=path,
            markup=render_png_as_terminal_markup(path),
        )
        for index, path in enumerate(png_paths, start=1)
    ]


__all__ = [
    "PDF_PREVIEW_CACHE_DIRNAME",
    "PdfPreviewError",
    "PdfPreviewPage",
    "build_pdf_preview_pages",
    "render_pdf_pages_to_pngs",
    "render_png_as_terminal_markup",
]
