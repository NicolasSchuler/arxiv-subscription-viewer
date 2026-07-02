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


class TestSettingsAsciiSafety:
    """The settings modal must not leak Unicode glyphs in --ascii mode."""

    @pytest.fixture
    def ascii_mode(self):
        from arxiv_browser import _ascii

        _ascii.set_ascii_mode(True)
        try:
            yield
        finally:
            _ascii.set_ascii_mode(False)

    def test_edit_button_label_is_ascii(self, ascii_mode):
        label = SettingsModal._edit_label()
        assert "…" not in label
        assert label == "Edit..."

    def test_interests_summary_ellipsis_is_ascii(self, ascii_mode):
        modal = SettingsModal(
            SettingsResult(
                llm_preset="",
                theme_name="monokai",
                s2_enabled=False,
                hf_enabled=False,
                research_interests="x" * 80,
            ),
            theme_names=["monokai"],
        )
        summary = modal._interests_summary()
        assert "…" not in summary
        assert summary.endswith("...")

    def test_interests_summary_uses_unicode_ellipsis_by_default(self):
        modal = SettingsModal(
            SettingsResult(
                llm_preset="",
                theme_name="monokai",
                s2_enabled=False,
                hf_enabled=False,
                research_interests="x" * 80,
            ),
            theme_names=["monokai"],
        )
        assert modal._interests_summary().endswith("…")


class TestSettingsInterestsSummaryLayout:
    """The interests preview stays short enough for the inline Edit button."""

    def _modal(self, interests: str) -> SettingsModal:
        return SettingsModal(
            SettingsResult(
                llm_preset="",
                theme_name="monokai",
                s2_enabled=False,
                hf_enabled=False,
                research_interests=interests,
            ),
            theme_names=["monokai"],
        )

    def test_long_interest_is_truncated_to_fit_row(self):
        # Value column (auto) + inline Edit button must fit the 72-wide dialog;
        # cap the preview so a long interest never overflows the row.
        summary = self._modal("word " * 40)._interests_summary()
        assert len(summary) <= 30

    def test_short_interest_is_shown_verbatim(self):
        summary = self._modal("ML safety")._interests_summary()
        assert summary == "ML safety"

    def test_empty_interest_shows_placeholder(self):
        assert self._modal("")._interests_summary() == "(not set)"


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

    async def test_footer_shows_only_cancel_hint(self, make_paper):
        from textual.widgets import Static

        from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
        from tests.support.patch_helpers import patch_save_config

        config = UserConfig(onboarding_seen=True)
        app = ArxivBrowser(
            [make_paper()], options=ArxivBrowserOptions(config=config, restore_session=False)
        )
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
                app.push_screen(SettingsModal(current, ["monokai"]))
                await pilot.pause()
                footer = app.screen.query_one("#settings-footer", Static)
                assert str(footer.content) == "Cancel: Esc"

    async def test_edit_button_shares_interests_row(self, make_paper):
        from textual.widgets import Button, Static

        from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
        from tests.support.patch_helpers import patch_save_config

        config = UserConfig(onboarding_seen=True)
        app = ArxivBrowser(
            [make_paper()], options=ArxivBrowserOptions(config=config, restore_session=False)
        )
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
                app.push_screen(SettingsModal(current, ["monokai"]))
                await pilot.pause()
                value = app.screen.query_one("#settings-interests-value", Static)
                edit = app.screen.query_one("#settings-interests-edit", Button)
                # Inline association: Edit lives in the same row as its value,
                # not down in the Cancel/Save button row.
                assert edit.parent is value.parent
                buttons_row = app.screen.query_one("#settings-buttons")
                assert edit.parent is not buttons_row

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
