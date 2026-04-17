"""Shared patch helpers for tests that target resolved module seams."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import patch

SAVE_CONFIG_TARGETS = (
    "arxiv_browser.actions.external_io_actions.save_config",
    "arxiv_browser.actions.library_actions.save_config",
    "arxiv_browser.actions.llm_actions.save_config",
    "arxiv_browser.actions.search_api_actions.save_config",
    "arxiv_browser.actions.trust_gate.save_config",
    "arxiv_browser.actions.ui_actions.save_config",
    "arxiv_browser.browser.detail_pane.save_config",
    "arxiv_browser.config.save_config",
)


@dataclass
class _PatchedCallGroup:
    mocks: list[object]

    @property
    def _called_mocks(self) -> list[object]:
        return [mock for mock in self.mocks if getattr(mock, "call_count", 0)]

    @property
    def call_count(self) -> int:
        return sum(getattr(mock, "call_count", 0) for mock in self.mocks)

    @property
    def call_args(self):
        called = self._called_mocks
        return called[-1].call_args if called else None

    @property
    def call_args_list(self):
        combined = []
        for mock in self.mocks:
            combined.extend(getattr(mock, "call_args_list", ()))
        return combined

    def assert_not_called(self) -> None:
        if self.call_count != 0:
            raise AssertionError(f"Expected no calls, saw {self.call_count}")

    def assert_called_once(self) -> None:
        if self.call_count != 1:
            raise AssertionError(f"Expected 1 call, saw {self.call_count}")

    def assert_called_once_with(self, *args, **kwargs) -> None:
        self.assert_called_once()
        self._called_mocks[0].assert_called_once_with(*args, **kwargs)


@contextmanager
def patch_save_config(*, return_value=None, side_effect=None):
    """Patch every in-repo resolution site for ``save_config``."""
    with ExitStack() as stack:
        mocks = [
            stack.enter_context(patch(target, return_value=return_value, side_effect=side_effect))
            for target in SAVE_CONFIG_TARGETS
        ]
        yield _PatchedCallGroup(mocks)
