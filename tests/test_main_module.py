"""Tests for `python -m arxiv_browser` entrypoint."""

from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest


def test_main_module_calls_sys_exit_with_main_return_value():
    with (
        patch("arxiv_browser.app.main", return_value=7) as main_mock,
        patch("sys.exit", side_effect=SystemExit) as exit_mock,
        pytest.raises(SystemExit),
    ):
        runpy.run_module("arxiv_browser.__main__", run_name="__main__")

    main_mock.assert_called_once_with()
    exit_mock.assert_called_once_with(7)
