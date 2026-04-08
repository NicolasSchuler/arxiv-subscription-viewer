"""Backward-compatible compatibility shim for the public arxiv_browser.app surface."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys

import httpx

from arxiv_browser.browser.content import ARXIV_HTML_TIMEOUT, MAX_PAPER_CONTENT_LENGTH
from arxiv_browser.parsing import extract_text_from_html as _extract_text_from_html

logger = logging.getLogger(__name__)

_PUBLIC_EXPORTS = [
    "ArxivBrowser",
    "ArxivBrowserOptions",
    "_configure_color_mode",
    "_configure_logging",
    "discover_history_files",
    "load_config",
    "main",
    "_fetch_paper_content_async",
    "_resolve_papers",
    "_validate_interactive_tty",
]
__all__ = list(_PUBLIC_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]

_EXPORT_SPECS = {
    "ArxivBrowser": ("arxiv_browser.browser.core", "ArxivBrowser"),
    "ArxivBrowserOptions": ("arxiv_browser.browser.core", "ArxivBrowserOptions"),
    "_configure_color_mode": ("arxiv_browser.cli", "_configure_color_mode"),
    "_configure_logging": ("arxiv_browser.cli", "_configure_logging"),
    "discover_history_files": ("arxiv_browser.parsing", "discover_history_files"),
    "load_config": ("arxiv_browser.config", "load_config"),
    "_resolve_papers": ("arxiv_browser.cli", "_resolve_papers"),
    "_validate_interactive_tty": ("arxiv_browser.cli", "_validate_interactive_tty"),
}


async def _fetch_paper_content_async(
    paper, client: httpx.AsyncClient | None = None, timeout: int | None = None
) -> str:
    """Fetch full paper text through the compatibility module patch surface."""
    html_url = f"https://arxiv.org/html/{paper.arxiv_id}"
    request_timeout = ARXIV_HTML_TIMEOUT if timeout is None else timeout
    extract_html = globals().get("extract_text_from_html", _extract_text_from_html)

    try:
        if client is not None:
            response = await client.get(html_url, timeout=request_timeout, follow_redirects=True)
        else:
            async with httpx.AsyncClient() as temp_client:
                response = await temp_client.get(
                    html_url,
                    timeout=request_timeout,
                    follow_redirects=True,
                )
        if response.status_code == 200:
            text = await asyncio.to_thread(extract_html, response.text)
            if text:
                return text[:MAX_PAPER_CONTENT_LENGTH]
        else:
            logger.warning(
                "arXiv HTML fetch returned %d for %s",
                response.status_code,
                paper.arxiv_id,
            )
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to fetch HTML for %s", paper.arxiv_id, exc_info=True)

    abstract = paper.abstract or paper.abstract_raw or ""
    return f"Abstract:\n{abstract}" if abstract else ""


def __getattr__(name: str):
    export_spec = _EXPORT_SPECS.get(name)
    if export_spec is None:
        raise AttributeError(f"module 'arxiv_browser.app' has no attribute {name!r}")
    module_name, attr_name = export_spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


def main() -> int:
    """CLI entry point that preserves the legacy arxiv_browser.app patch surface."""
    cli_module = importlib.import_module("arxiv_browser.cli")
    return cli_module.main(
        deps=cli_module.CliDependencies(
            load_config_fn=globals().get("load_config") or __getattr__("load_config"),
            discover_history_files_fn=globals().get("discover_history_files")
            or __getattr__("discover_history_files"),
            resolve_papers_fn=globals().get("_resolve_papers") or __getattr__("_resolve_papers"),
            configure_logging_fn=globals().get("_configure_logging")
            or __getattr__("_configure_logging"),
            configure_color_mode_fn=globals().get("_configure_color_mode")
            or __getattr__("_configure_color_mode"),
            validate_interactive_tty_fn=globals().get("_validate_interactive_tty")
            or __getattr__("_validate_interactive_tty"),
            app_factory=globals().get("ArxivBrowser") or __getattr__("ArxivBrowser"),
        )
    )


if __name__ == "__main__":
    sys.exit(main())
