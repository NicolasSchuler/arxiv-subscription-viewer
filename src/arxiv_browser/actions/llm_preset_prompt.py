"""Inline LLM preset prompt helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ScreenStackError

from arxiv_browser.action_messages import build_actionable_success
from arxiv_browser.llm_providers import resolve_provider

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser


def prompt_llm_preset(
    app: ArxivBrowser,
    *,
    get_config_path_fn: Callable[[], Path],
    notify_timeout: int,
) -> None:
    """Offer an inline LLM preset picker when no command is configured."""
    from arxiv_browser.modals.settings import LLMPresetPickerModal

    def _on_pick(preset: str | None) -> None:
        if not preset:
            return
        app._config.llm_preset = preset
        if app._save_config_or_warn("LLM preset"):
            app._llm_provider = resolve_provider(app._config)
            app.notify(
                build_actionable_success(
                    f"LLM preset set to '{preset}'",
                    next_step="press the AI shortcut again to run it",
                ),
                title="LLM configured",
                timeout=notify_timeout,
            )

    try:
        app.push_screen(LLMPresetPickerModal(), _on_pick)
    except ScreenStackError:
        app.notify(
            f"Set llm_command or llm_preset in config.json ({get_config_path_fn()})",
            title="LLM not configured",
            severity="warning",
            timeout=notify_timeout,
        )
