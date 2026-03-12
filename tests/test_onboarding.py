"""Tests for first-run onboarding detection."""

from __future__ import annotations

from arxiv_browser.config import _config_to_dict, _dict_to_config
from arxiv_browser.models import UserConfig


class TestOnboardingSeen:
    """Tests for the onboarding_seen flag on UserConfig."""

    def test_default_is_false(self) -> None:
        config = UserConfig()
        assert config.onboarding_seen is False

    def test_serializes_true(self) -> None:
        config = UserConfig(onboarding_seen=True)
        data = _config_to_dict(config)
        assert data["onboarding_seen"] is True

    def test_serializes_false(self) -> None:
        config = UserConfig(onboarding_seen=False)
        data = _config_to_dict(config)
        assert data["onboarding_seen"] is False

    def test_deserializes_true(self) -> None:
        config = _dict_to_config({"onboarding_seen": True})
        assert config.onboarding_seen is True

    def test_deserializes_false(self) -> None:
        config = _dict_to_config({"onboarding_seen": False})
        assert config.onboarding_seen is False

    def test_deserializes_missing_as_false(self) -> None:
        config = _dict_to_config({})
        assert config.onboarding_seen is False

    def test_roundtrip(self) -> None:
        original = UserConfig(onboarding_seen=True)
        data = _config_to_dict(original)
        restored = _dict_to_config(data)
        assert restored.onboarding_seen is True

    def test_deserializes_non_bool_as_false(self) -> None:
        config = _dict_to_config({"onboarding_seen": "yes"})
        assert config.onboarding_seen is False
