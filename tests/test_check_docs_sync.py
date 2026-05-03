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


def _load_check_version_sync_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_version_sync.py"
    spec = importlib.util.spec_from_file_location("check_version_sync", script_path)
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


class TestCheckConfigReference:
    """Regression tests for persisted config documentation coverage."""

    def test_check_config_reference_passes_for_documented_persisted_keys(self):
        module = _load_check_docs_sync_module()

        models_text = """
from dataclasses import dataclass


@dataclass
class UserConfig:
    version: int = 1
    llm_max_retries: int = 1
    llm_timeout: int = 120
    onboarding_seen: bool = False
    config_defaulted: bool = False
"""
        config_text = """
def _config_to_dict(config):
    return {
        "version": config.version,
        "llm_max_retries": config.llm_max_retries,
        "llm_timeout": config.llm_timeout,
        "onboarding_seen": config.onboarding_seen,
        "session": {
            "scroll_index": config.session.scroll_index,
        },
    }
"""
        config_reference_text = """
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | `int` | `1` | Managed by the app. |
| `llm_max_retries` | `int` | `1` | Retry count. |
| `llm_timeout` | `int` | `120` | Timeout in seconds. |
| `onboarding_seen` | `bool` | `false` | First-run onboarding dismissed. |
| `session.scroll_index` | `int` | `0` | Restored scroll position. |
"""

        assert module._check_config_reference(models_text, config_text, config_reference_text) == []

    def test_check_config_reference_reports_missing_persisted_keys(self):
        module = _load_check_docs_sync_module()

        models_text = """
from dataclasses import dataclass


@dataclass
class UserConfig:
    llm_max_retries: int = 1
    llm_timeout: int = 120
    onboarding_seen: bool = False
    config_defaulted: bool = False
"""
        config_text = """
def _config_to_dict(config):
    return {
        "llm_max_retries": config.llm_max_retries,
        "llm_timeout": config.llm_timeout,
        "onboarding_seen": config.onboarding_seen,
    }
"""
        config_reference_text = """
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_max_retries` | `int` | `1` | Retry count. |
"""

        assert module._check_config_reference(models_text, config_text, config_reference_text) == [
            "docs/config-reference.md missing persisted config key: llm_timeout",
            "docs/config-reference.md missing persisted config key: onboarding_seen",
        ]


class TestCheckDocsIndexNavigation:
    """Regression tests for docs landing-page guide navigation."""

    def test_check_docs_index_navigation_passes_when_feature_guides_are_linked(self):
        module = _load_check_docs_sync_module()

        docs_readme_text = """
## Feature Guides

| Document | Description |
|----------|-------------|
| [history-mode.md](history-mode.md) | History workflow |
| [config-reference.md](config-reference.md) | Config reference |
| [troubleshooting.md](troubleshooting.md) | Common issues |

## Internal Development Docs

- [tui-style-guide.md](tui-style-guide.md)
"""
        docs_index_text = """
<html>
  <body>
    <a href="./history-mode.md">History mode</a>
    <a href="config-reference.md#top">Config reference</a>
    <a href="/docs/troubleshooting.md?ref=landing">Troubleshooting</a>
  </body>
</html>
"""

        assert module._check_docs_index_navigation(docs_readme_text, docs_index_text) == []

    def test_check_docs_index_navigation_reports_missing_feature_guide_links(self):
        module = _load_check_docs_sync_module()

        docs_readme_text = """
## Feature Guides

| Document | Description |
|----------|-------------|
| [history-mode.md](history-mode.md) | History workflow |
| [config-reference.md](config-reference.md) | Config reference |
"""
        docs_index_text = """
<html>
  <body>
    <a href="history-mode.md">History mode</a>
  </body>
</html>
"""

        assert module._check_docs_index_navigation(docs_readme_text, docs_index_text) == [
            "docs/index.html missing guide navigation link: config-reference.md"
        ]


class TestCheckVersionSync:
    """Regression tests for release metadata drift detection."""

    def test_latest_released_version_ignores_unreleased_heading(self, tmp_path, monkeypatch):
        module = _load_check_version_sync_module()
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            """# Changelog

## [Unreleased]

## [0.1.3] - 2026-05-03

## [0.1.2] - 2025-01-26
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(module, "CHANGELOG", changelog)

        assert module.latest_released_version() == "0.1.3"

    def test_main_passes_when_pyproject_matches_latest_release(self, tmp_path, monkeypatch, capsys):
        module = _load_check_version_sync_module()
        pyproject = tmp_path / "pyproject.toml"
        changelog = tmp_path / "CHANGELOG.md"
        pyproject.write_text('[project]\nversion = "0.1.3"\n', encoding="utf-8")
        changelog.write_text("# Changelog\n\n## [0.1.3] - 2026-05-03\n", encoding="utf-8")
        monkeypatch.setattr(module, "PYPROJECT", pyproject)
        monkeypatch.setattr(module, "CHANGELOG", changelog)

        assert module.main() == 0
        assert "matches latest CHANGELOG entry" in capsys.readouterr().out

    def test_main_reports_how_to_fix_version_mismatch(self, tmp_path, monkeypatch, capsys):
        module = _load_check_version_sync_module()
        pyproject = tmp_path / "pyproject.toml"
        changelog = tmp_path / "CHANGELOG.md"
        pyproject.write_text('[project]\nversion = "0.1.3"\n', encoding="utf-8")
        changelog.write_text("# Changelog\n\n## [0.1.2] - 2025-01-26\n", encoding="utf-8")
        monkeypatch.setattr(module, "PYPROJECT", pyproject)
        monkeypatch.setattr(module, "CHANGELOG", changelog)

        assert module.main() == 1
        err = capsys.readouterr().err
        assert "cut a CHANGELOG release entry" in err
        assert "revert pyproject.toml and uv.lock" in err

    def test_main_accepts_projects_without_released_changelog_entries(
        self, tmp_path, monkeypatch, capsys
    ):
        module = _load_check_version_sync_module()
        pyproject = tmp_path / "pyproject.toml"
        changelog = tmp_path / "CHANGELOG.md"
        pyproject.write_text('[project]\nversion = "0.1.3"\n', encoding="utf-8")
        changelog.write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")
        monkeypatch.setattr(module, "PYPROJECT", pyproject)
        monkeypatch.setattr(module, "CHANGELOG", changelog)

        assert module.main() == 0
        assert "CHANGELOG has no released entries yet" in capsys.readouterr().out
