"""Environment diagnostics for the arxiv-viewer CLI."""

from __future__ import annotations

import importlib.metadata
import os
import re
import shlex
import shutil
import sys
from datetime import date
from pathlib import Path

from arxiv_browser.models import UserConfig

_PACKAGE_NAME = "arxiv-subscription-viewer"
_POSIX_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_DOCTOR_OK_MARKER = "  ok   "
_DOCTOR_WARN_MARKER = "  WARN "
_DOCTOR_INFO_MARKER = "  info "


def _get_version() -> str:
    """Return the installed package version, or 'dev' if not installed."""
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _extract_command_binary(command_template: str) -> str | None:
    """Extract the executable token from a command template for diagnostics."""
    try:
        argv = shlex.split(command_template, posix=os.name != "nt")
    except ValueError:
        return None
    if not argv:
        return None
    if os.name != "nt":
        while len(argv) > 1 and _POSIX_ENV_ASSIGNMENT_RE.match(argv[0]):
            argv.pop(0)
        if not argv:
            return None
    if os.name == "nt" and len(argv[0]) >= 2 and argv[0].startswith('"') and argv[0].endswith('"'):
        return argv[0][1:-1]
    return argv[0]


def _print_doctor_header() -> None:
    """Print the diagnostic header banner."""
    print(f"arxiv-viewer {_get_version()}")
    print(f"Python {sys.version.split()[0]}")
    print()


def _doctor_config_issue_count(config_path: Path, *, ok_marker: str, info_marker: str) -> int:
    """Report config-file status and return issue count."""
    if config_path.exists():
        print(f"{ok_marker} Config file: {config_path}")
    else:
        print(f"{info_marker} Config file: {config_path} (not created yet; using defaults)")
    return 0


def _doctor_history_issue_count(
    history_files: list[tuple[date, Path]],
    *,
    ok_marker: str,
    warn_marker: str,
    info_marker: str,
) -> int:
    """Report history-directory status and return issue count."""
    base_dir = Path.cwd()
    history_dir = base_dir / "history"
    if history_files:
        print(f"{ok_marker} History: {len(history_files)} date(s) in {history_dir}")
        return 0
    if history_dir.is_dir():
        print(f"{warn_marker} History: {history_dir} exists but contains no YYYY-MM-DD.txt files")
        return 1
    print(f"{info_marker} History: no history/ directory in {base_dir} (use search mode instead)")
    return 0


def _doctor_llm_issue_count(
    config: UserConfig,
    *,
    ok_marker: str,
    warn_marker: str,
    info_marker: str,
) -> int:
    """Report LLM command diagnostics and return issue count."""
    from arxiv_browser.llm import LLM_PRESETS, _resolve_llm_command
    from arxiv_browser.llm_providers import llm_command_requires_shell

    resolved_llm_command = _resolve_llm_command(config)
    if config.llm_command and resolved_llm_command:
        if "{prompt}" not in resolved_llm_command:
            print(f"{warn_marker} LLM command: missing required {{prompt}} placeholder")
            return 1
        if not config.allow_llm_shell_fallback and llm_command_requires_shell(resolved_llm_command):
            print(
                f"{warn_marker} LLM command: requires shell execution, "
                "but allow_llm_shell_fallback is disabled"
            )
            return 1
        cmd_binary = _extract_command_binary(resolved_llm_command)
        if cmd_binary and shutil.which(cmd_binary):
            print(f"{ok_marker} LLM command: {cmd_binary} found on PATH")
            return 0
        print(
            f"{warn_marker} LLM command: "
            f"{cmd_binary or 'unable to parse command'} NOT found on PATH"
        )
        return 1

    if not config.llm_preset:
        print(f"{info_marker} LLM: not configured (set llm_preset or llm_command in config)")
        return 0

    preset_cmd_name = config.llm_preset
    if preset_cmd_name not in LLM_PRESETS:
        print(f"{warn_marker} LLM preset: unknown preset '{preset_cmd_name}'")
        return 1
    if "{prompt}" not in resolved_llm_command:
        print(f"{warn_marker} LLM preset: missing required {{prompt}} placeholder")
        return 1
    if not config.allow_llm_shell_fallback and llm_command_requires_shell(resolved_llm_command):
        print(
            f"{warn_marker} LLM preset: {preset_cmd_name} requires shell execution, "
            "but allow_llm_shell_fallback is disabled"
        )
        return 1
    cmd_binary = _extract_command_binary(resolved_llm_command)
    if cmd_binary and shutil.which(cmd_binary):
        print(f"{ok_marker} LLM preset: {preset_cmd_name} ({cmd_binary} found on PATH)")
        return 0
    print(
        f"{warn_marker} LLM preset: {preset_cmd_name} "
        f"({cmd_binary or 'unable to parse command'} NOT found on PATH)"
    )
    return 1


def _doctor_feature_summary(config: UserConfig, *, ok_marker: str, info_marker: str) -> None:
    """Report feature toggles that don't contribute issues."""
    if config.s2_enabled:
        s2_status = "enabled"
        if config.s2_api_key:
            s2_status += " (API key set)"
        print(f"{ok_marker} Semantic Scholar: {s2_status}")
    else:
        print(f"{info_marker} Semantic Scholar: disabled (enable with s2_enabled in config)")

    if config.hf_enabled:
        print(f"{ok_marker} HuggingFace trending: enabled")
    else:
        print(f"{info_marker} HuggingFace trending: disabled (enable with hf_enabled in config)")


def _doctor_export_dirs(config: UserConfig, *, ok_marker: str, info_marker: str) -> None:
    """Report export-directory configuration."""
    export_dir = Path(config.bibtex_export_dir).expanduser() if config.bibtex_export_dir else None
    pdf_dir = Path(config.pdf_download_dir).expanduser() if config.pdf_download_dir else None
    for label, path_value, default in [
        ("Export dir", export_dir, "~/arxiv-exports/"),
        ("PDF dir", pdf_dir, "~/arxiv-pdfs/"),
    ]:
        if path_value is None:
            print(f"{info_marker} {label}: {default} (default; will be created on first use)")
        elif path_value.is_dir():
            print(f"{ok_marker} {label}: {path_value}")
        else:
            print(f"{info_marker} {label}: {path_value} (will be created on first export)")


def _doctor_terminal_summary(*, ok_marker: str, info_marker: str) -> None:
    """Report terminal interactivity."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        print(f"{ok_marker} Terminal: interactive TTY")
    else:
        print(f"{info_marker} Terminal: not an interactive TTY (TUI will not start)")


def _run_doctor(config: UserConfig, history_files: list[tuple[date, Path]]) -> int:
    """Run diagnostic checks and print a summary report."""
    from arxiv_browser.config import get_config_path

    ok_marker = _DOCTOR_OK_MARKER
    warn_marker = _DOCTOR_WARN_MARKER
    info_marker = _DOCTOR_INFO_MARKER

    _print_doctor_header()

    issues = 0
    issues += _doctor_config_issue_count(
        get_config_path(),
        ok_marker=ok_marker,
        info_marker=info_marker,
    )
    issues += _doctor_history_issue_count(
        history_files,
        ok_marker=ok_marker,
        warn_marker=warn_marker,
        info_marker=info_marker,
    )
    issues += _doctor_llm_issue_count(
        config,
        ok_marker=ok_marker,
        warn_marker=warn_marker,
        info_marker=info_marker,
    )

    _doctor_feature_summary(config, ok_marker=ok_marker, info_marker=info_marker)
    _doctor_export_dirs(config, ok_marker=ok_marker, info_marker=info_marker)
    _doctor_terminal_summary(ok_marker=ok_marker, info_marker=info_marker)

    print()
    if issues:
        print(f"{issues} issue(s) found. See warnings above.")
    else:
        print("No issues found.")
    return 1 if issues else 0


__all__ = [
    "_doctor_config_issue_count",
    "_doctor_export_dirs",
    "_doctor_feature_summary",
    "_doctor_history_issue_count",
    "_doctor_llm_issue_count",
    "_doctor_terminal_summary",
    "_extract_command_binary",
    "_get_version",
    "_run_doctor",
]
