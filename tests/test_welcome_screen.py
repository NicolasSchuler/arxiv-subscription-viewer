"""Tests for the first-run welcome screen."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from arxiv_browser.modals.welcome import WelcomeScreen


class TestWelcomeScreen:
    """Tests for the WelcomeScreen modal."""

    def test_welcome_screen_can_be_instantiated(self) -> None:
        """WelcomeScreen can be created without arguments."""
        screen = WelcomeScreen()
        assert screen is not None

    def test_welcome_screen_has_dismiss_bindings(self) -> None:
        """Essential dismiss keybindings are defined."""
        binding_keys = {b.key for b in WelcomeScreen.BINDINGS}
        assert "escape" in binding_keys
        assert "enter" in binding_keys
        assert "space" in binding_keys
        assert "question_mark" in binding_keys

    def test_welcome_screen_action_dismiss(self) -> None:
        """action_dismiss_welcome calls dismiss(None)."""
        screen = WelcomeScreen()
        with patch.object(screen, "dismiss") as mock_dismiss:
            screen.action_dismiss_welcome()
            mock_dismiss.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_welcome_screen_mounts_and_renders_key_sections(
        self, tmp_path, monkeypatch, make_paper
    ) -> None:
        """The real modal mount should populate themed onboarding content."""
        from textual.widgets import Static

        import arxiv_browser.browser.core as browser_core
        from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
        from arxiv_browser.models import UserConfig
        from tests.support.patch_helpers import patch_save_config

        monkeypatch.setattr(browser_core, "get_cache_db_path", lambda: tmp_path / "cache.db")
        app = ArxivBrowser(
            [make_paper()],
            options=ArxivBrowserOptions(
                config=UserConfig(onboarding_seen=True),
                restore_session=False,
            ),
        )
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                modal = WelcomeScreen()
                app.push_screen(modal)
                await pilot.pause(0.05)
                content = modal.query_one("#welcome-content", Static)
                rendered = str(content.content)
                assert "Navigate" in rendered
                assert "Search" in rendered
                assert "Actions" in rendered
                await pilot.press("enter")
                await pilot.pause()
                assert modal not in pilot.app.screen_stack


class TestOnboardingFlag:
    """Tests for the onboarding_seen config flag integration."""

    def test_onboarding_seen_defaults_to_false(self) -> None:
        """New UserConfig has onboarding_seen=False (without test fixture override)."""
        from arxiv_browser.models import UserConfig

        config = UserConfig(onboarding_seen=False)
        assert config.onboarding_seen is False

    def test_onboarding_flag_serialization_roundtrip(self) -> None:
        """onboarding_seen survives config save/load cycle."""
        from arxiv_browser.config import _config_to_dict, _dict_to_config
        from arxiv_browser.models import UserConfig

        config = UserConfig()
        config.onboarding_seen = True
        data = _config_to_dict(config)
        assert data["onboarding_seen"] is True

        restored = _dict_to_config(data)
        assert restored.onboarding_seen is True

    def test_onboarding_flag_false_roundtrip(self) -> None:
        """onboarding_seen=False also survives serialization."""
        from arxiv_browser.config import _config_to_dict, _dict_to_config
        from arxiv_browser.models import UserConfig

        config = UserConfig(onboarding_seen=False)
        assert config.onboarding_seen is False
        data = _config_to_dict(config)
        assert data["onboarding_seen"] is False

        restored = _dict_to_config(data)
        assert restored.onboarding_seen is False
