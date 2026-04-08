"""CLI/bootstrap entry point for the browser package."""

from __future__ import annotations

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.cli import (
    CliDependencies,
    _configure_color_mode,
    _configure_logging,
    _resolve_papers,
    _validate_interactive_tty,
)
from arxiv_browser.cli import main as _cli_main
from arxiv_browser.config import load_config
from arxiv_browser.parsing import discover_history_files


def main() -> int:
    """Run the CLI using the canonical browser constructor."""
    return _cli_main(
        deps=CliDependencies(
            load_config_fn=load_config,
            discover_history_files_fn=discover_history_files,
            resolve_papers_fn=_resolve_papers,
            configure_logging_fn=_configure_logging,
            configure_color_mode_fn=_configure_color_mode,
            validate_interactive_tty_fn=_validate_interactive_tty,
            app_factory=ArxivBrowser,
            app_factory_supports_options=True,
        )
    )


__all__ = ["main"]
