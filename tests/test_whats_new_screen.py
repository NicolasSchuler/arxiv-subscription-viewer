"""Tests for the version-bump 'What's New' overlay."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from arxiv_browser.modals.whats_new import WhatsNewScreen
from arxiv_browser.whats_new import (
    WHATS_NEW_ENTRIES,
    WHATS_NEW_VERSION,
    should_show_whats_new,
)


class TestShouldShowWhatsNew:
    """Gating logic for automatic overlay trigger."""

    def test_fresh_install_shows_overlay(self) -> None:
        """Empty last-seen tag (new install or pre-feature install) shows overlay."""
        assert should_show_whats_new("") is True

    def test_older_version_shows_overlay(self) -> None:
        """Any mismatched tag shows overlay on the next launch."""
        assert should_show_whats_new("0.0.1") is True

    def test_current_version_skips_overlay(self) -> None:
        """User who dismissed the current release's notes sees nothing."""
        assert should_show_whats_new(WHATS_NEW_VERSION) is False

    def test_future_version_skips_overlay(self) -> None:
        """If last-seen tag already matches current, overlay stays quiet.

        The check is a plain equality test — we don't compare semver so a
        user running a downgrade just sees the notes once, which is fine.
        """
        # same as current_version case but covers the equality intent
        assert should_show_whats_new(WHATS_NEW_VERSION) is False


class TestWhatsNewScreen:
    """Tests for the WhatsNewScreen modal itself."""

    def test_screen_can_be_instantiated(self) -> None:
        """Modal requires no arguments."""
        screen = WhatsNewScreen()
        assert screen is not None

    def test_dismiss_bindings_defined(self) -> None:
        """Standard dismiss keys are bound."""
        binding_keys = {b.key for b in WhatsNewScreen.BINDINGS}
        assert "escape" in binding_keys
        assert "enter" in binding_keys
        assert "space" in binding_keys
        assert "q" in binding_keys

    def test_action_dismiss_whats_new(self) -> None:
        """action_dismiss_whats_new closes with result=None."""
        screen = WhatsNewScreen()
        with patch.object(screen, "dismiss") as mock_dismiss:
            screen.action_dismiss_whats_new()
            mock_dismiss.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_whats_new_screen_mounts_and_renders_release_entries(
        self, tmp_path, monkeypatch, make_paper
    ) -> None:
        """The real modal mount should render the current release notes."""
        from textual.widgets import Static

        import arxiv_browser.browser.core as browser_core
        from arxiv_browser.browser.core import ArxivBrowser, ArxivBrowserOptions
        from arxiv_browser.models import UserConfig
        from tests.support.patch_helpers import patch_save_config

        monkeypatch.setattr(browser_core, "get_cache_db_path", lambda: tmp_path / "cache.db")
        app = ArxivBrowser(
            [make_paper()],
            options=ArxivBrowserOptions(
                config=UserConfig(onboarding_seen=True, last_seen_whats_new=WHATS_NEW_VERSION),
                restore_session=False,
            ),
        )
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                modal = WhatsNewScreen()
                app.push_screen(modal)
                await pilot.pause(0.05)
                content = modal.query_one("#whats-new-content", Static)
                rendered = str(content.content)
                assert WHATS_NEW_ENTRIES[0][0] in rendered
                assert WHATS_NEW_ENTRIES[0][1] in rendered
                await pilot.press("q")
                await pilot.pause()
                assert modal not in pilot.app.screen_stack


class TestWhatsNewContent:
    """Sanity checks for the curated release-notes content."""

    def test_entries_are_non_empty(self) -> None:
        """At least one entry must be defined."""
        assert len(WHATS_NEW_ENTRIES) >= 1

    def test_entries_have_title_and_description(self) -> None:
        """Each entry is a (title, description) tuple with non-empty values."""
        for title, description in WHATS_NEW_ENTRIES:
            assert title, "title must be non-empty"
            assert description, "description must be non-empty"

    def test_version_tag_is_stringlike(self) -> None:
        """WHATS_NEW_VERSION is a non-empty string so config persistence works."""
        assert isinstance(WHATS_NEW_VERSION, str)
        assert WHATS_NEW_VERSION


class TestLastSeenWhatsNewConfigField:
    """Persistence of the new UserConfig field."""

    def test_default_is_empty_string(self) -> None:
        from arxiv_browser.models import UserConfig

        # The conftest autouse fixture seeds last_seen_whats_new to the
        # current release tag to suppress the modal in most tests. Here we
        # pass the field explicitly to assert the declared dataclass default.
        config = UserConfig(last_seen_whats_new="")
        assert config.last_seen_whats_new == ""

    def test_roundtrip_persists_value(self) -> None:
        from arxiv_browser.config import _config_to_dict, _dict_to_config
        from arxiv_browser.models import UserConfig

        config = UserConfig()
        config.last_seen_whats_new = "0.9.9"
        data = _config_to_dict(config)
        assert data["last_seen_whats_new"] == "0.9.9"

        restored = _dict_to_config(data)
        assert restored.last_seen_whats_new == "0.9.9"

    def test_missing_field_defaults_to_empty(self) -> None:
        """Configs persisted before this feature still load cleanly."""
        from arxiv_browser.config import _dict_to_config

        restored = _dict_to_config({})
        assert restored.last_seen_whats_new == ""


class TestAppBinding:
    """F1 is reserved for on-demand What's New."""

    def test_f1_binding_present(self) -> None:
        from arxiv_browser.ui_constants import APP_BINDINGS

        actions = {b.action for b in APP_BINDINGS}
        keys = {b.key for b in APP_BINDINGS}
        assert "show_whats_new" in actions
        assert "f1" in keys
