"""Settings action handlers for the in-app settings editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.css.query import NoMatches

from arxiv_browser.themes import available_theme_names

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser
    from arxiv_browser.modals.settings import SettingsResult


def action_open_settings(app: ArxivBrowser) -> None:
    """Open the in-app settings editor and apply the result on save."""
    from arxiv_browser.modals.settings import SettingsModal, SettingsResult

    current = SettingsResult(
        llm_preset="" if app._config.llm_command else app._config.llm_preset,
        theme_name=app._effective_theme_name(),
        s2_enabled=app._s2_active,
        hf_enabled=app._hf_active,
        research_interests=app._config.research_interests,
    )

    def _on_result(result: SettingsResult | None) -> None:
        if result is not None:
            _apply_settings(app, result)

    app.push_screen(
        SettingsModal(current, available_theme_names(app._config.custom_themes)),
        _on_result,
    )


def _apply_settings(app: ArxivBrowser, result: SettingsResult) -> None:
    """Apply settings from the settings modal, persisting and refreshing as needed."""
    config = app._config
    changed: list[str] = []

    llm_preset_changed = result.llm_preset != config.llm_preset
    llm_custom_cleared = bool(result.llm_preset and config.llm_command)
    if llm_preset_changed or llm_custom_cleared:
        config.llm_preset = result.llm_preset
        if llm_custom_cleared:
            config.llm_command = ""
        changed.append("LLM preset")
    if result.research_interests != config.research_interests:
        config.research_interests = result.research_interests
        changed.append("research interests")
    if result.theme_name != app._effective_theme_name():
        app._theme_override = None
        config.theme_name = result.theme_name
        app._apply_theme_overrides()
        app._apply_category_overrides()
        try:
            app._get_paper_details_widget().clear_cache()
        except NoMatches:
            pass
        app._refresh_list_view()
        app._refresh_detail_pane()
        changed.append("theme")
    if changed:
        app._save_config_or_warn("settings")

    # Reuse toggle actions so persistence, data fetch, and notifications stay identical.
    s2_changed = result.s2_enabled != app._s2_active
    hf_changed = result.hf_enabled != app._hf_active
    if s2_changed:
        app.action_toggle_s2()
    if hf_changed:
        app._track_task(app.action_toggle_hf())

    app._update_status_bar()
    if changed:
        app.notify("Updated " + ", ".join(changed), title="Settings")
    elif not (s2_changed or hf_changed):
        app.notify("No changes", title="Settings")
