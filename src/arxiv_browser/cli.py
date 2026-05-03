"""CLI/bootstrap helpers for the arXiv browser application."""

from __future__ import annotations

import argparse
import inspect
import logging
import logging.handlers
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.cli_doctor import (
    _doctor_config_issue_count,
    _doctor_export_dirs,
    _doctor_feature_summary,
    _doctor_history_issue_count,
    _doctor_llm_issue_count,
    _doctor_terminal_summary,
    _extract_command_binary,
    _get_version,
    _run_doctor,
)
from arxiv_browser.cli_keybindings import _run_keybindings
from arxiv_browser.cli_resolver import (
    ARXIV_API_MIN_INTERVAL_SECONDS,
    ARXIV_API_TIMEOUT,
    ARXIV_API_URL,
    ARXIV_API_USER_AGENT,
    ResolvePapersResult,
    _api_mode_requested,
    _fetch_arxiv_api_papers,
    _fetch_latest_arxiv_digest,
    _find_history_index,
    _resolve_arxiv_api_mode,
    _resolve_history_date,
    _resolve_input_file,
    _resolve_legacy_fallback,
    _resolve_papers,
)
from arxiv_browser.config import CONFIG_APP_NAME, load_config
from arxiv_browser.models import ARXIV_API_MAX_RESULTS_LIMIT, Paper, UserConfig
from arxiv_browser.parsing import (
    ARXIV_QUERY_FIELDS,
    HISTORY_DATE_FORMAT,
    discover_history_files,
)
from arxiv_browser.themes import THEME_NAMES

logger = logging.getLogger(__name__)
_DOCTOR_SHUTIL_PATCH_SURFACE = shutil  # Preserve arxiv_browser.cli.shutil patch target.
CLI_COMMANDS = ("browse", "search", "dates", "completions", "config-path", "doctor", "keybindings")
CLI_ROOT_DESCRIPTION = "Browse and search arXiv papers from your terminal."
CLI_ROOT_EPILOG = """Examples:
  arxiv-viewer
  arxiv-viewer browse --date 2026-01-23
  arxiv-viewer browse --input papers.txt
  arxiv-viewer search --category cs.AI
  arxiv-viewer search --query "diffusion transformer" --field title
  arxiv-viewer dates
  arxiv-viewer config-path
  arxiv-viewer doctor
  arxiv-viewer keybindings
  arxiv-viewer keybindings --format markdown --tier core
  eval "$(arxiv-viewer completions bash)"

Exit codes:
  0   Success
  1   Application error (missing files, API failures, bad input)
  2   Usage error (bad arguments, non-interactive terminal)
"""

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3


_ROOT_FLAG_OPTIONS = frozenset({"--debug", "--ascii", "--no-color", "-V", "--version"})
_ROOT_VALUE_OPTIONS = frozenset({"--color", "--theme"})


def _configure_logging(debug: bool) -> None:
    """Configure logging. When debug=True, logs to file at DEBUG level."""
    if not debug:
        # Default: suppress all logging (TUI captures stderr)
        logging.disable(logging.CRITICAL)
        return

    log_dir = Path(user_config_dir(CONFIG_APP_NAME))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "debug.log"

    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)


def _configure_color_mode(color_mode: str) -> None:
    """Configure environment hints for terminal color behavior."""
    if color_mode == "never":
        os.environ["NO_COLOR"] = "1"
        os.environ.pop("FORCE_COLOR", None)
        return
    if color_mode == "always":
        os.environ["FORCE_COLOR"] = "1"
        os.environ.pop("NO_COLOR", None)
        return
    # auto
    os.environ.pop("FORCE_COLOR", None)
    os.environ.pop("NO_COLOR", None)


def _validate_interactive_tty() -> bool:
    """Return True when stdin/stdout are interactive terminals."""
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _print_config_path() -> int:
    """Print the platform-specific config file path and return exit code."""
    from arxiv_browser.config import get_config_path

    print(get_config_path())
    return 0


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="arxiv-viewer",
        description=CLI_ROOT_DESCRIPTION,
        epilog=CLI_ROOT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to file (~/.config/arxiv-browser/debug.log)",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color output mode for terminal UI (default: auto)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable terminal colors (equivalent to --color never; honors NO_COLOR standard)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use ASCII-only status icons for compatibility with limited terminals",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(THEME_NAMES),
        default=None,
        help=("Override the UI color theme for this session (does not persist to config)"),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    browse_parser = subparsers.add_parser(
        "browse",
        help="Open local history or a local paper file",
        description="Browse local digest history or a specific input file.",
        epilog=(
            "Examples:\n"
            "  arxiv-viewer browse                     # auto-load newest history\n"
            "  arxiv-viewer browse --date 2026-01-23   # open specific date\n"
            "  arxiv-viewer browse -i papers.txt       # custom file\n"
            "  arxiv-viewer browse --no-restore         # ignore saved session"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    browse_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input file containing arXiv metadata (overrides history mode)",
    )
    browse_parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Start with fresh session (ignore saved scroll position, filters, etc.)",
    )
    browse_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Open specific history date in YYYY-MM-DD format",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Fetch startup papers from the arXiv API",
        description="Search arXiv online and open the results directly in the TUI.",
        epilog=(
            "Examples:\n"
            "  arxiv-viewer search --category cs.AI\n"
            '  arxiv-viewer search --query "diffusion transformer" --field title\n'
            '  arxiv-viewer search --query "attention" --mode page --max-results 100\n'
            '  arxiv-viewer search --query "LLM" --category cs.CL'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Query text for the arXiv API search",
    )
    search_parser.add_argument(
        "--field",
        choices=sorted(ARXIV_QUERY_FIELDS),
        default="all",
        help="Field for --query: all, title, author, abstract (default: all)",
    )
    search_parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Optional arXiv category filter (for example: cs.AI)",
    )
    search_parser.add_argument(
        "--mode",
        choices=["latest", "page"],
        default="latest",
        help="latest collects the newest matching day; page loads a single API page",
    )
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help=(f"API request page size (1-{ARXIV_API_MAX_RESULTS_LIMIT}; default: config value)"),
    )

    subparsers.add_parser(
        "dates",
        help="List available local history dates and exit",
        description="Print the available YYYY-MM-DD history files and exit.",
    )

    completions_parser = subparsers.add_parser(
        "completions",
        help="Generate shell completion scripts",
        description=(
            "Print a shell completion script to stdout.\n\n"
            'Usage: eval "$(arxiv-viewer completions bash)"'
        ),
    )
    completions_parser.add_argument(
        "shell",
        choices=("bash", "zsh", "fish"),
        help="Target shell (bash, zsh, or fish)",
    )

    subparsers.add_parser(
        "config-path",
        help="Print the configuration file path and exit",
        description="Print the platform-specific config.json path to stdout.",
    )

    subparsers.add_parser(
        "doctor",
        help="Check environment and configuration health",
        description="Run diagnostic checks and report potential issues.",
    )

    keybindings_parser = subparsers.add_parser(
        "keybindings",
        help="Print keyboard shortcuts and exit",
        description="Display all keyboard shortcuts grouped by category.",
    )
    keybindings_parser.add_argument(
        "--format",
        choices=("table", "json", "markdown"),
        default="table",
        dest="kb_format",
        help="Output format (default: table)",
    )
    keybindings_parser.add_argument(
        "--tier",
        choices=("core", "standard", "power", "all"),
        default="all",
        help="Filter shortcuts by tier (default: all)",
    )

    return parser


def _normalize_cli_argv(argv: list[str] | None) -> list[str]:
    """Insert the browse subcommand for bare or browse-style invocations."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return ["browse"]

    index = 0
    while index < len(args):
        token = args[index]
        if token in CLI_COMMANDS or token in {"-h", "--help"}:
            return args
        if token in _ROOT_FLAG_OPTIONS:
            index += 1
            continue
        if _is_root_value_flag(token):
            if "=" in token:
                index += 1
                continue
            if index + 1 >= len(args):
                return args
            index += 2
            continue
        break

    return [*args[:index], "browse", *args[index:]]


def _is_color_flag(token: str) -> bool:
    """Return whether a token spells the CLI color flag."""
    return token == "--color" or token.startswith("--color=")  # nosec B105


def _is_root_value_flag(token: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for option in _ROOT_VALUE_OPTIONS)


@dataclass(slots=True)
class CliDependencies:
    """Injectable collaborators for CLI entry-point tests and wrappers."""

    load_config_fn: Callable[[], UserConfig] = load_config
    discover_history_files_fn: Callable[[Path], list[tuple[date, Path]]] = discover_history_files
    resolve_papers_fn: Callable[
        [argparse.Namespace, Path, UserConfig, list[tuple[date, Path]]],
        ResolvePapersResult,
    ] = _resolve_papers
    configure_logging_fn: Callable[[bool], None] = _configure_logging
    configure_color_mode_fn: Callable[[str], None] = _configure_color_mode
    validate_interactive_tty_fn: Callable[[], bool] = _validate_interactive_tty
    app_factory: Callable[..., Any] | None = None
    app_factory_supports_options: bool | None = None


def _app_factory_accepts_options(app_factory: Callable[..., Any]) -> bool:
    """Return whether the factory explicitly opts into the new `options=` seam."""
    try:
        return "options" in inspect.signature(app_factory).parameters
    except (TypeError, ValueError):
        return False


def _handle_config_free_command(args: argparse.Namespace) -> int | None:
    """Run subcommands that do not need config, history, or TTY state."""
    command = getattr(args, "command", None)
    if command == "completions":
        from arxiv_browser.completions import get_completion_script

        print(get_completion_script(args.shell))
        return 0
    if command == "config-path":
        return _print_config_path()
    if command == "keybindings":
        return _run_keybindings(args)
    return None


def _resolve_color_mode(args: argparse.Namespace, normalized_argv: list[str]) -> str:
    color_flag_explicit = any(_is_color_flag(token) for token in normalized_argv)
    if args.no_color:
        return "never"
    if color_flag_explicit:
        return str(args.color)
    if os.environ.get("NO_COLOR"):
        return "never"
    return str(args.color)


def _run_dates_command(history_files: list[tuple[date, Path]]) -> int:
    if not history_files:
        print(
            build_actionable_error(
                "list history dates",
                why="no history files were found in history/",
                next_step=(
                    'add history/YYYY-MM-DD.txt files or run "arxiv-viewer search --category cs.AI"'
                ),
            ),
            file=sys.stderr,
        )
        return 1
    print("Available dates:")
    for d, path in history_files:
        print(f"  {d.strftime(HISTORY_DATE_FORMAT)}  ({path.name})")
    return 0


def _print_empty_papers_error() -> None:
    print(
        build_actionable_error(
            "start arxiv-viewer",
            why="the selected source contained no papers",
            next_step='choose another input/date or run "arxiv-viewer search --category cs.AI"',
        ),
        file=sys.stderr,
    )


def _print_non_interactive_error() -> None:
    print(
        "Error: arxiv-viewer requires an interactive TTY for the full UI.",
        file=sys.stderr,
    )
    print("Next steps:", file=sys.stderr)
    print("  - Run arxiv-viewer directly in a terminal session", file=sys.stderr)
    print("  - Use arxiv-viewer dates for non-interactive output", file=sys.stderr)
    print(
        "  - Use arxiv-viewer browse --date YYYY-MM-DD in an interactive terminal",
        file=sys.stderr,
    )
    print("  - Use --help for command documentation", file=sys.stderr)


def _run_browser_app(
    args: argparse.Namespace,
    dependencies: CliDependencies,
    papers: list[Paper],
    config: UserConfig,
    history_files: list[tuple[date, Path]],
    current_date_index: int,
) -> int:
    """Create and run the Textual browser app."""
    from arxiv_browser.browser.core import ArxivBrowserOptions as _ArxivBrowserOptions

    app_factory = dependencies.app_factory
    if app_factory is None:
        from arxiv_browser.browser.core import ArxivBrowser as _ArxivBrowser

        app_factory = _ArxivBrowser

    restore_session = (
        False
        if getattr(args, "command", None) == "search"
        else not bool(getattr(args, "no_restore", False))
    )
    browser_options = _ArxivBrowserOptions(
        config=config,
        restore_session=restore_session,
        history_files=history_files,
        current_date_index=current_date_index,
        ascii_icons=args.ascii,
        theme_override=getattr(args, "theme", None),
    )

    supports_options = dependencies.app_factory_supports_options
    if supports_options is None:
        supports_options = _app_factory_accepts_options(app_factory)

    if supports_options:
        app = app_factory(papers, options=browser_options)
    else:
        app = app_factory(
            papers,
            config=config,
            restore_session=restore_session,
            history_files=history_files,
            current_date_index=current_date_index,
            ascii_icons=args.ascii,
        )
    app.run()
    return 0


def main(
    argv: list[str] | None = None,
    *,
    deps: CliDependencies | None = None,
) -> int:
    """Main entry point. Returns exit code."""
    dependencies = deps or CliDependencies()
    parser = _build_cli_parser()
    normalized_argv = _normalize_cli_argv(argv)
    args = parser.parse_args(normalized_argv)

    early_exit = _handle_config_free_command(args)
    if early_exit is not None:
        return early_exit

    dependencies.configure_color_mode_fn(_resolve_color_mode(args, normalized_argv))
    dependencies.configure_logging_fn(args.debug)
    logger.debug("arxiv-viewer starting, cwd=%s", Path.cwd())

    base_dir = Path.cwd()

    # Load user config early (needed for session restore)
    config = dependencies.load_config_fn()

    history_files = dependencies.discover_history_files_fn(base_dir)

    if getattr(args, "command", None) == "doctor":
        return _run_doctor(config, history_files)

    if getattr(args, "command", None) == "dates":
        return _run_dates_command(history_files)

    result = dependencies.resolve_papers_fn(args, base_dir, config, history_files)
    if isinstance(result, int):
        return result
    papers, history_files, current_date_index = result

    if not papers:
        _print_empty_papers_error()
        return 1

    if not dependencies.validate_interactive_tty_fn():
        _print_non_interactive_error()
        return 2

    # Sort papers alphabetically by title
    papers.sort(key=lambda p: p.title.lower())
    return _run_browser_app(
        args,
        dependencies,
        papers,
        config,
        history_files,
        current_date_index,
    )


__all__ = [
    "ARXIV_API_MIN_INTERVAL_SECONDS",
    "ARXIV_API_TIMEOUT",
    "ARXIV_API_URL",
    "ARXIV_API_USER_AGENT",
    "CliDependencies",
    "_api_mode_requested",
    "_build_cli_parser",
    "_configure_color_mode",
    "_configure_logging",
    "_doctor_config_issue_count",
    "_doctor_export_dirs",
    "_doctor_feature_summary",
    "_doctor_history_issue_count",
    "_doctor_llm_issue_count",
    "_doctor_terminal_summary",
    "_extract_command_binary",
    "_fetch_arxiv_api_papers",
    "_fetch_latest_arxiv_digest",
    "_find_history_index",
    "_get_version",
    "_normalize_cli_argv",
    "_print_config_path",
    "_resolve_arxiv_api_mode",
    "_resolve_history_date",
    "_resolve_input_file",
    "_resolve_legacy_fallback",
    "_resolve_papers",
    "_run_doctor",
    "_run_keybindings",
    "_validate_interactive_tty",
    "main",
]
