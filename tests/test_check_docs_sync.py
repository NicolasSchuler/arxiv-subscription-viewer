"""Tests for the docs sync drift checker script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_check_docs_sync_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_docs_sync.py"
    spec = importlib.util.spec_from_file_location("check_docs_sync", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckCompletions:
    """Regression tests for per-shell completion drift detection."""

    def test_check_completions_passes_when_all_shells_define_all_commands(self):
        module = _load_check_docs_sync_module()

        cli_text = 'CLI_COMMANDS = ("browse", "search", "doctor")'
        completions_text = '''
_BASH_SCRIPT = r"""
local commands="browse search doctor"
"""

_ZSH_SCRIPT = r"""
commands=(
    'browse:Browse papers'
    'search:Search arXiv'
    'doctor:Run diagnostics'
)
"""

_FISH_SCRIPT = r"""
complete -c arxiv-viewer -n '__fish_use_subcommand' -a browse -d 'Browse papers'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a search -d 'Search arXiv'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a doctor -d 'Run diagnostics'
"""
'''

        assert module._check_completions(cli_text, completions_text) == []

    def test_check_completions_reports_shell_specific_missing_command(self):
        module = _load_check_docs_sync_module()

        cli_text = 'CLI_COMMANDS = ("browse", "search", "doctor")'
        completions_text = '''
_BASH_SCRIPT = r"""
local commands="browse search doctor"
"""

_ZSH_SCRIPT = r"""
commands=(
    'browse:Browse papers'
    'search:Search arXiv'
)
"""

_FISH_SCRIPT = r"""
complete -c arxiv-viewer -n '__fish_use_subcommand' -a browse -d 'Browse papers'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a search -d 'Search arXiv'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a doctor -d 'Run diagnostics'
"""
'''

        assert module._check_completions(cli_text, completions_text) == [
            "zsh completions missing subcommand: doctor"
        ]
