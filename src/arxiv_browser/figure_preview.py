"""arXiv HTML figure preview helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

from arxiv_browser.export import get_pdf_download_path
from arxiv_browser.models import Paper, UserConfig
from arxiv_browser.pdf_preview import render_png_as_terminal_markup

logger = logging.getLogger(__name__)

FIGURE_PREVIEW_CACHE_DIRNAME = ".figure-cache"
FIGURE_PREVIEW_MAX_PIXEL_SIZE = (900, 900)
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SUPPORTED_IMAGE_TYPES = frozenset(
    {
        "",
        "image/gif",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)
_SUPPORTED_IMAGE_URL_SCHEMES = frozenset({"http", "https"})


class FigurePreviewError(RuntimeError):
    """Raised when an arXiv HTML figure preview cannot be built."""


@dataclass(slots=True, frozen=True)
class HtmlFigureImage:
    """First figure image discovered in an arXiv HTML paper."""

    image_url: str
    caption: str = ""


@dataclass(slots=True, frozen=True)
class FigurePreview:
    """Rendered terminal preview for an arXiv HTML figure."""

    image_url: str
    image_path: Path
    markup: str
    caption: str = ""


class _FirstFigureParser(HTMLParser):
    """Find the first image inside a LaTeXML figure and collect its caption."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._figure_depth = 0
        self._caption_depth = 0
        self._caption_parts: list[str] = []
        self._done = False
        self.image_url = ""
        self.caption = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        attr_map = {name.lower(): value or "" for name, value in attrs}
        if tag == "figure" and self._is_latex_figure(attr_map.get("class", "")):
            self._figure_depth += 1
            return
        if self._figure_depth <= 0:
            return
        if tag == "figure":
            self._figure_depth += 1
        elif tag == "figcaption":
            self._caption_depth += 1
        elif tag == "img" and not self.image_url:
            src = attr_map.get("src", "").strip()
            if src:
                self.image_url = urljoin(self._base_url, src)

    def handle_endtag(self, tag: str) -> None:
        if self._figure_depth <= 0:
            return
        if tag == "figcaption" and self._caption_depth > 0:
            self._caption_depth -= 1
            if self._caption_depth == 0:
                self.caption = " ".join("".join(self._caption_parts).split())
        elif tag == "figure":
            self._figure_depth -= 1
            if self.image_url and self._figure_depth == 0:
                self._done = True

    def handle_data(self, data: str) -> None:
        if self._caption_depth > 0:
            self._caption_parts.append(data)

    @staticmethod
    def _is_latex_figure(class_value: str) -> bool:
        return "ltx_figure" in {part.strip() for part in class_value.split()}


def extract_first_html_figure(html: str, base_url: str) -> HtmlFigureImage:
    """Extract the first LaTeXML figure image URL from arXiv HTML."""
    parser = _FirstFigureParser(base_url)
    parser.feed(html)
    if not parser.image_url:
        raise FigurePreviewError("No figure image found in arXiv HTML")
    scheme = urlparse(parser.image_url).scheme.lower()
    if scheme not in _SUPPORTED_IMAGE_URL_SCHEMES:
        raise FigurePreviewError(f"Unsupported figure image URL scheme: {scheme or 'missing'}")
    return HtmlFigureImage(image_url=parser.image_url, caption=parser.caption)


def figure_preview_cache_dir(paper: Paper, config: UserConfig) -> Path:
    """Return the cache directory for downloaded arXiv HTML figure images."""
    return get_pdf_download_path(paper, config).parent / FIGURE_PREVIEW_CACHE_DIRNAME


def _safe_stem(value: str) -> str:
    return _SAFE_STEM_RE.sub("_", value).strip("._") or "paper"


def figure_preview_cache_path(paper: Paper, config: UserConfig) -> Path:
    """Return the normalized PNG cache path for the current paper's first figure."""
    return figure_preview_cache_dir(paper, config) / f"{_safe_stem(paper.arxiv_id)}-figure.png"


def cached_figure_is_valid(image_path: Path) -> bool:
    """Return whether the cached figure image can be decoded by Pillow."""
    if not image_path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def validate_figure_content_type(content_type: str) -> None:
    """Reject image payloads that the terminal renderer cannot decode."""
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized not in _SUPPORTED_IMAGE_TYPES:
        raise FigurePreviewError(f"Unsupported figure image type: {normalized or 'unknown'}")


def save_figure_bytes_to_cache(image_bytes: bytes, image_path: Path) -> Path:
    """Validate and normalize downloaded figure bytes into a cached PNG."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise FigurePreviewError(f"Image dependency missing: {exc.name}") from exc

    if not image_bytes:
        raise FigurePreviewError("Figure image was empty")
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(BytesIO(image_bytes)) as original:
            image = original.convert("RGB")
            image.thumbnail(FIGURE_PREVIEW_MAX_PIXEL_SIZE, Image.Resampling.LANCZOS)
            image.save(image_path, format="PNG")
    except (OSError, ValueError) as exc:
        logger.warning("Could not cache arXiv HTML figure", exc_info=True)
        raise FigurePreviewError(f"Could not read figure image: {exc}") from exc
    return image_path


def build_figure_preview(
    *,
    figure: HtmlFigureImage,
    image_path: Path,
) -> FigurePreview:
    """Build terminal-safe markup for a cached arXiv HTML figure image."""
    return FigurePreview(
        image_url=figure.image_url,
        image_path=image_path,
        caption=figure.caption,
        markup=render_png_as_terminal_markup(image_path),
    )


__all__ = [
    "FIGURE_PREVIEW_CACHE_DIRNAME",
    "FigurePreview",
    "FigurePreviewError",
    "HtmlFigureImage",
    "build_figure_preview",
    "cached_figure_is_valid",
    "extract_first_html_figure",
    "figure_preview_cache_path",
    "save_figure_bytes_to_cache",
    "validate_figure_content_type",
]
