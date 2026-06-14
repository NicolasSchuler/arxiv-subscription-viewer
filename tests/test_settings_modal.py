"""Tests for the in-app settings modal, inline LLM preset picker, and apply logic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.css.query import NoMatches

from arxiv_browser.actions import ui_actions
from arxiv_browser.modals.settings import (
    LLMPresetPickerModal,
    SettingsModal,
    SettingsResult,
)
from arxiv_browser.models import UserConfig


def _make_app(**config_kwargs) -> SimpleNamespace:
    config = UserConfig(**config_kwargs)
    return SimpleNamespace(
        _config=config,
        _s2_active=config.s2_enabled,
        _hf_active=config.hf_enabled,
        _theme_override=None,
        _effective_theme_name=lambda: config.theme_name,
        _apply_theme_overrides=MagicMock(),
        _apply_category_overrides=MagicMock(),
        _get_paper_details_widget=MagicMock(return_value=SimpleNamespace(clear_cache=MagicMock())),
        _refresh_list_view=MagicMock(),
        _refresh_detail_pane=MagicMock(),
        _save_config_or_warn=MagicMock(return_value=True),
        _update_status_bar=MagicMock(),
        action_toggle_s2=MagicMock(),
        action_toggle_hf=MagicMock(return_value=None),
        _track_task=MagicMock(),
        push_screen=MagicMock(),
        notify=MagicMock(),
    )


def _result(app, **overrides) -> SettingsResult:
    base = {
        "llm_preset": app._config.llm_preset,
        "theme_name": app._config.theme_name,
        "s2_enabled": app._s2_active,
        "hf_enabled": app._hf_active,
        "research_interests": app._config.research_interests,
    }
    base.update(overrides)
    return SettingsResult(**base)


class TestApplySettings:
    def test_sets_preset_and_interests(self):
        app = _make_app()
        ui_actions._apply_settings(
            app, _result(app, llm_preset="claude", research_interests="ML safety")
        )
        assert app._config.llm_preset == "claude"
        assert app._config.research_interests == "ML safety"
        app._save_config_or_warn.assert_called_once()
        assert "LLM preset" in str(app.notify.call_args)
        assert "research interests" in str(app.notify.call_args)

    def test_preset_selection_clears_custom_llm_command(self):
        app = _make_app(llm_command="custom {prompt}")
        ui_actions._apply_settings(app, _result(app, llm_preset="claude"))
        assert app._config.llm_preset == "claude"
        assert app._config.llm_command == ""
        app._save_config_or_warn.assert_called_once()

    def test_changes_theme_and_refreshes(self):
        app = _make_app(theme_name="monokai")
        ui_actions._apply_settings(app, _result(app, theme_name="catppuccin"))
        assert app._config.theme_name == "catppuccin"
        app._apply_theme_overrides.assert_called_once()
        app._refresh_detail_pane.assert_called_once()
        assert "theme" in str(app.notify.call_args)

    def test_theme_change_tolerates_missing_detail_widget(self):
        app = _make_app(theme_name="monokai")
        app._get_paper_details_widget.side_effect = NoMatches("missing")
        ui_actions._apply_settings(app, _result(app, theme_name="catppuccin"))
        assert app._config.theme_name == "catppuccin"
        app._refresh_detail_pane.assert_called_once()

    def test_toggles_s2_and_hf_only_when_changed(self):
        app = _make_app()  # s2/hf default off
        ui_actions._apply_settings(app, _result(app, s2_enabled=True, hf_enabled=True))
        app.action_toggle_s2.assert_called_once()
        app._track_task.assert_called_once()  # hf toggle scheduled as a task

    def test_no_toggle_when_unchanged(self):
        app = _make_app(s2_enabled=True)
        app._s2_active = True
        ui_actions._apply_settings(app, _result(app, s2_enabled=True))
        app.action_toggle_s2.assert_not_called()

    def test_no_changes_notifies(self):
        app = _make_app()
        ui_actions._apply_settings(app, _result(app))
        app.action_toggle_s2.assert_not_called()
        app._save_config_or_warn.assert_not_called()
        assert "No changes" in str(app.notify.call_args)


class TestActionOpenSettings:
    def test_pushes_settings_modal(self):
        app = _make_app()
        ui_actions.action_open_settings(app)
        app.push_screen.assert_called_once()
        assert isinstance(app.push_screen.call_args[0][0], SettingsModal)

    def test_custom_llm_command_shows_custom_setting(self):
        app = _make_app(llm_command="custom {prompt}", llm_preset="claude")
        ui_actions.action_open_settings(app)
        modal = app.push_screen.call_args[0][0]
        assert isinstance(modal, SettingsModal)
        assert modal._current.llm_preset == ""

    def test_callback_applies_result(self):
        app = _make_app()
        ui_actions.action_open_settings(app)
        callback = app.push_screen.call_args[0][1]
        callback(None)  # cancel — no change
        app._save_config_or_warn.assert_not_called()
        callback(_result(app, llm_preset="llm"))
        assert app._config.llm_preset == "llm"


@pytest.mark.asyncio
class TestSettingsModalTUI:
    async def test_settings_modal_save_returns_result(self, make_paper):
        from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
        from tests.support.patch_helpers import patch_save_config

        config = UserConfig(onboarding_seen=True)
        app = ArxivBrowser(
            [make_paper()], options=ArxivBrowserOptions(config=config, restore_session=False)
        )
        captured: list = []
        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                current = SettingsResult(
                    llm_preset="",
                    theme_name="monokai",
                    s2_enabled=False,
                    hf_enabled=False,
                    research_interests="",
                )
                app.push_screen(SettingsModal(current, ["monokai", "catppuccin"]), captured.append)
                await pilot.pause()
                assert isinstance(app.screen, SettingsModal)
                await pilot.press("ctrl+s")
                await pilot.pause()
        assert captured and isinstance(captured[0], SettingsResult)

    async def test_llm_preset_picker_lists_and_returns_preset(self, make_paper):
        from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
        from tests.support.patch_helpers import patch_save_config

        config = UserConfig(onboarding_seen=True)
        app = ArxivBrowser(
            [make_paper()], options=ArxivBrowserOptions(config=config, restore_session=False)
        )
        captured: list = []
        with patch_save_config(return_value=True):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.push_screen(LLMPresetPickerModal(), captured.append)
                await pilot.pause()
                assert isinstance(app.screen, LLMPresetPickerModal)
                await pilot.press("enter")
                await pilot.pause()
        from arxiv_browser.llm import LLM_PRESETS

        assert captured and captured[0] in LLM_PRESETS
