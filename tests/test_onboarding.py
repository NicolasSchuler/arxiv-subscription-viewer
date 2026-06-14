"""Tests for first-run onboarding detection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.config import _config_to_dict, _dict_to_config
from arxiv_browser.models import UserConfig


class TestOnboardingSeen:
    """Tests for the onboarding_seen flag on UserConfig."""

    def test_default_is_false(self) -> None:
        config = UserConfig(onboarding_seen=False)
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


class TestShortcutsHint:
    """Tests for the one-time 'Press ? for all shortcuts' nudge."""

    def _stub(self, **config_kwargs) -> SimpleNamespace:
        config = UserConfig(onboarding_seen=True, **config_kwargs)
        return SimpleNamespace(
            _config=config,
            screen_stack=[object()],  # only the base screen, no modal on top
            _save_config_or_warn=MagicMock(),
            notify=MagicMock(),
        )

    def _hint(self, stub: SimpleNamespace) -> None:
        ArxivBrowser._maybe_hint_shortcuts(stub)

    def test_shows_once_then_persists_flag(self) -> None:
        stub = self._stub(shortcuts_hint_seen=False)
        self._hint(stub)
        assert stub._config.shortcuts_hint_seen is True
        stub.notify.assert_called_once()
        assert "?" in stub.notify.call_args[0][0]
        stub._save_config_or_warn.assert_called_once()

        # A second view does not nag.
        stub.notify.reset_mock()
        self._hint(stub)
        stub.notify.assert_not_called()

    def test_suppressed_before_onboarding(self) -> None:
        stub = self._stub(shortcuts_hint_seen=False)
        stub._config.onboarding_seen = False
        self._hint(stub)
        stub.notify.assert_not_called()
        assert stub._config.shortcuts_hint_seen is False

    def test_suppressed_while_modal_open(self) -> None:
        stub = self._stub(shortcuts_hint_seen=False)
        stub.screen_stack = [object(), object()]  # a modal is on top
        self._hint(stub)
        stub.notify.assert_not_called()
        assert stub._config.shortcuts_hint_seen is False

    def test_flag_roundtrips_through_config(self) -> None:
        data = _config_to_dict(UserConfig(shortcuts_hint_seen=True))
        assert data["shortcuts_hint_seen"] is True
        assert _dict_to_config(data).shortcuts_hint_seen is True


class TestBadgeLegendHint:
    """Tests for the one-time 'badges → ? legend' nudge."""

    def _stub(self, *, has_enrichment: bool, **config_kwargs) -> SimpleNamespace:
        config = UserConfig(onboarding_seen=True, **config_kwargs)
        aid = "2401.00001"
        stub = SimpleNamespace(
            _config=config,
            screen_stack=[object()],
            _s2_cache={aid: object()} if has_enrichment else {},
            _hf_cache={},
            _relevance_scores={},
            _version_updates={},
            _digest_inbox_context=None,
            _save_config_or_warn=MagicMock(),
            notify=MagicMock(),
        )
        stub._paper_has_enrichment = lambda paper, s=stub: ArxivBrowser._paper_has_enrichment(
            s, paper
        )
        return stub

    def _hint(self, stub: SimpleNamespace) -> None:
        ArxivBrowser._maybe_hint_badge_legend(stub, _make_paper("2401.00001"))

    def test_shows_once_when_enriched(self) -> None:
        stub = self._stub(has_enrichment=True)
        self._hint(stub)
        assert stub._config.badge_legend_hint_seen is True
        stub.notify.assert_called_once()
        assert "Badge Legend" in stub.notify.call_args[0][0]
        # Does not nag again.
        stub.notify.reset_mock()
        self._hint(stub)
        stub.notify.assert_not_called()

    def test_suppressed_without_enrichment(self) -> None:
        stub = self._stub(has_enrichment=False)
        self._hint(stub)
        stub.notify.assert_not_called()
        assert stub._config.badge_legend_hint_seen is False

    def test_relevance_score_counts_as_enrichment(self) -> None:
        stub = self._stub(has_enrichment=False)
        stub._relevance_scores["2401.00001"] = 0.9
        paper = _make_paper("2401.00001")
        assert ArxivBrowser._paper_has_enrichment(stub, paper) is True

    def test_suppressed_while_modal_open(self) -> None:
        stub = self._stub(has_enrichment=True)
        stub.screen_stack = [object(), object()]
        self._hint(stub)
        stub.notify.assert_not_called()

    def test_flag_roundtrips_through_config(self) -> None:
        data = _config_to_dict(UserConfig(badge_legend_hint_seen=True))
        assert data["badge_legend_hint_seen"] is True
        assert _dict_to_config(data).badge_legend_hint_seen is True


def _make_paper(arxiv_id: str):
    from arxiv_browser.models import Paper

    return Paper(
        arxiv_id=arxiv_id,
        date="Mon, 1 Jan 2024",
        title="T",
        authors="A",
        categories="cs.AI",
        comments=None,
        abstract="x",
        abstract_raw="x",
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )
