"""CLI/bootstrap helpers for the arXiv browser application."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_config_dir

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.config import CONFIG_APP_NAME, _coerce_arxiv_api_max_results, load_config
from arxiv_browser.models import ARXIV_API_MAX_RESULTS_LIMIT, Paper, UserConfig
from arxiv_browser.parsing import (
    ARXIV_QUERY_FIELDS,
    HISTORY_DATE_FORMAT,
    build_arxiv_search_query,
    discover_history_files,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
)

logger = logging.getLogger(__name__)

ResolvePapersResult = tuple[list[Paper], list[tuple[date, Path]], int] | int

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_API_TIMEOUT = 30
ARXIV_API_USER_AGENT = "arxiv-subscription-viewer/1.0"
ARXIV_API_MIN_INTERVAL_SECONDS = 3.0


def _fetch_arxiv_api_papers(
    *,
    query: str,
    field: str,
    category: str,
    max_results: int,
    start: int = 0,
) -> list[Paper]:
    """Fetch one page of papers from the arXiv API."""
    search_query = build_arxiv_search_query(query, field, category)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": max(0, start),
        "max_results": max_results,
    }
    response = httpx.get(
        ARXIV_API_URL,
        params=params,
        headers={"User-Agent": ARXIV_API_USER_AGENT},
        timeout=ARXIV_API_TIMEOUT,
    )
    response.raise_for_status()
    return parse_arxiv_api_feed(response.text)


def _fetch_latest_arxiv_digest(
    *,
    query: str,
    field: str,
    category: str,
    max_results: int,
) -> list[Paper]:
    """Fetch all papers from the newest matching arXiv day.

    This mirrors email-digest behavior: collect papers from the latest
    available day, then stop when older days begin.
    """
    start = 0
    target_day: date | None = None
    papers: list[Paper] = []
    seen_ids: set[str] = set()

    while True:
        page = _fetch_arxiv_api_papers(
            query=query,
            field=field,
            category=category,
            max_results=max_results,
            start=start,
        )
        if not page:
            break

        reached_older_day = False
        for paper in page:
            parsed_date = parse_arxiv_date(paper.date)
            if parsed_date == datetime.min:
                continue
            day = parsed_date.date()
            if target_day is None:
                target_day = day
            if day != target_day:
                reached_older_day = True
                break
            if paper.arxiv_id not in seen_ids:
                papers.append(paper)
                seen_ids.add(paper.arxiv_id)

        if reached_older_day or len(page) < max_results:
            break

        start += max_results
        time.sleep(ARXIV_API_MIN_INTERVAL_SECONDS)

    return papers


def _api_mode_requested(args: argparse.Namespace) -> bool:
    """Return True when CLI flags request startup via arXiv API."""
    return bool(getattr(args, "api_query", None) is not None or getattr(args, "api_category", None))


def _resolve_arxiv_api_mode(
    args: argparse.Namespace, config: UserConfig
) -> ResolvePapersResult | None:
    """Resolve startup papers from arXiv API flags.

    Returns:
        ``None`` when API mode was not requested.
        ``ResolvePapersResult`` when API mode is requested.
    """
    if not _api_mode_requested(args):
        return None

    api_query = (getattr(args, "api_query", None) or "").strip()
    api_category = (getattr(args, "api_category", None) or "").strip()
    api_field = str(getattr(args, "api_field", "all"))
    api_page_mode = bool(getattr(args, "api_page_mode", False))
    max_results_arg = getattr(args, "api_max_results", None)
    max_results = _coerce_arxiv_api_max_results(
        max_results_arg if max_results_arg is not None else config.arxiv_api_max_results
    )

    try:
        if api_page_mode:
            papers = _fetch_arxiv_api_papers(
                query=api_query,
                field=api_field,
                category=api_category,
                max_results=max_results,
            )
        else:
            papers = _fetch_latest_arxiv_digest(
                query=api_query,
                field=api_field,
                category=api_category,
                max_results=max_results,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code == 429:
            print(
                "Error: arXiv API rate limit reached (HTTP 429). Wait a few seconds and retry.",
                file=sys.stderr,
            )
        elif status_code >= 500:
            print(
                f"Error: arXiv API is unavailable right now (HTTP {status_code}). Retry later.",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: arXiv API rejected the request (HTTP {status_code}). "
                "Check --api-query/--api-category and retry.",
                file=sys.stderr,
            )
        return 1
    except (httpx.HTTPError, OSError):
        print(
            "Error: Failed to fetch papers from arXiv API (network or I/O error).",
            file=sys.stderr,
        )
        return 1

    if not papers:
        print("Error: No papers found for the requested arXiv API query.", file=sys.stderr)
        return 1

    return (papers, [], 0)


def _resolve_input_file(input_path: Path) -> list[Paper] | int:
    """Validate and parse an explicit input file. Returns papers or exit code."""
    arxiv_file = input_path.resolve()
    if not arxiv_file.exists():
        print(f"Error: {arxiv_file} not found", file=sys.stderr)
        return 1
    if arxiv_file.is_dir():
        print(f"Error: {arxiv_file} is a directory, not a file", file=sys.stderr)
        return 1
    if not os.access(arxiv_file, os.R_OK):
        print(f"Error: {arxiv_file} is not readable (permission denied)", file=sys.stderr)
        return 1
    try:
        return parse_arxiv_file(arxiv_file)
    except OSError as e:
        print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
        return 1


def _find_history_index(
    history_files: list[tuple[date, Path]],
    target_date: date,
) -> int | None:
    """Return the index for target_date in history_files, if present."""
    for i, (d, _) in enumerate(history_files):
        if d == target_date:
            return i
    return None


def _resolve_history_date(history_files: list[tuple[date, Path]], date_str: str) -> int | None:
    """Find the index for --date argument. Returns index or None on error."""
    try:
        target_date = datetime.strptime(date_str, HISTORY_DATE_FORMAT).date()
    except ValueError:
        print(
            f"Error: Invalid date format '{date_str}', expected YYYY-MM-DD",
            file=sys.stderr,
        )
        return None
    idx = _find_history_index(history_files, target_date)
    if idx is not None:
        return idx
    print(f"Error: No file found for date {date_str}", file=sys.stderr)
    return None


def _resolve_legacy_fallback(base_dir: Path) -> list[Paper] | int:
    """Find and parse arxiv.txt in the current directory. Returns papers or exit code."""
    arxiv_file = base_dir / "arxiv.txt"
    if not arxiv_file.exists():
        print(
            build_actionable_error(
                "find startup papers",
                why="no history files or arxiv.txt were found in the current directory",
                next_step=(
                    "create history/YYYY-MM-DD.txt, create arxiv.txt, "
                    "or use --api-query/--api-category"
                ),
            ),
            file=sys.stderr,
        )
        return 1
    if not os.access(arxiv_file, os.R_OK):
        print(f"Error: {arxiv_file} is not readable (permission denied)", file=sys.stderr)
        return 1
    try:
        return parse_arxiv_file(arxiv_file)
    except OSError as e:
        print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
        return 1


def _resolve_papers(
    args: argparse.Namespace,
    base_dir: Path,
    config: UserConfig,
    history_files: list[tuple[date, Path]],
) -> ResolvePapersResult:
    """Resolve which papers to load based on CLI args."""
    if args.input is not None:
        result = _resolve_input_file(args.input)
        if isinstance(result, int):
            return result
        return (result, [], 0)

    api_result = _resolve_arxiv_api_mode(args, config)
    if api_result is not None:
        return api_result

    if history_files:
        current_date_index = 0
        if args.date:
            idx = _resolve_history_date(history_files, args.date)
            if idx is None:
                return 1
            current_date_index = idx
        elif not args.no_restore and config.session.current_date:
            try:
                saved_date = datetime.strptime(
                    config.session.current_date, HISTORY_DATE_FORMAT
                ).date()
            except ValueError:
                saved_date = None
            if saved_date is not None:
                idx = _find_history_index(history_files, saved_date)
                if idx is not None:
                    current_date_index = idx
        _, arxiv_file = history_files[current_date_index]
        try:
            papers = parse_arxiv_file(arxiv_file)
        except OSError as e:
            print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
            return 1
        return (papers, history_files, current_date_index)

    result = _resolve_legacy_fallback(base_dir)
    if isinstance(result, int):
        return result
    return (result, [], 0)


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
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
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


def _validate_interactive_tty() -> bool:
    """Return True when stdin/stdout are interactive terminals."""
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def main(
    argv: list[str] | None = None,
    *,
    load_config_fn: Callable[[], UserConfig] = load_config,
    discover_history_files_fn: Callable[[Path], list[tuple[date, Path]]] = discover_history_files,
    resolve_papers_fn: Callable[
        [argparse.Namespace, Path, UserConfig, list[tuple[date, Path]]],
        ResolvePapersResult,
    ] = _resolve_papers,
    configure_logging_fn: Callable[[bool], None] = _configure_logging,
    configure_color_mode_fn: Callable[[str], None] = _configure_color_mode,
    validate_interactive_tty_fn: Callable[[], bool] = _validate_interactive_tty,
    app_factory: Callable[..., Any] | None = None,
) -> int:
    """Main entry point. Returns exit code."""
    parser = argparse.ArgumentParser(description="Browse arXiv papers from a text file in a TUI")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input file containing arXiv metadata (overrides history mode)",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Start with fresh session (ignore saved scroll position, filters, etc.)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Open specific date in YYYY-MM-DD format (history mode only)",
    )
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates in history/ and exit",
    )
    parser.add_argument(
        "--api-query",
        type=str,
        default=None,
        help="Fetch startup papers from arXiv API using this query text",
    )
    parser.add_argument(
        "--api-field",
        choices=sorted(ARXIV_QUERY_FIELDS),
        default="all",
        help="Field for --api-query: all, title, author, abstract (default: all)",
    )
    parser.add_argument(
        "--api-category",
        type=str,
        default=None,
        help="Optional arXiv category filter for API mode (for example: cs.AI)",
    )
    parser.add_argument(
        "--api-max-results",
        type=int,
        default=None,
        help=(f"API request page size (1-{ARXIV_API_MAX_RESULTS_LIMIT}; default: config value)"),
    )
    parser.add_argument(
        "--api-page-mode",
        action="store_true",
        help="Load a single API page (default mode loads the latest matching day)",
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
        help="Disable terminal colors (equivalent to --color never)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use ASCII-only status icons for compatibility with limited terminals",
    )
    args = parser.parse_args(argv)
    api_mode = _api_mode_requested(args)
    if api_mode and args.list_dates:
        print(
            "Error: --list-dates cannot be combined with --api-query/--api-category",
            file=sys.stderr,
        )
        return 1
    if api_mode and args.date:
        print("Error: --date cannot be combined with --api-query/--api-category", file=sys.stderr)
        return 1

    color_mode = "never" if args.no_color else args.color
    configure_color_mode_fn(color_mode)
    configure_logging_fn(args.debug)
    logger.debug("arxiv-viewer starting, cwd=%s", Path.cwd())

    base_dir = Path.cwd()

    # Load user config early (needed for session restore)
    config = load_config_fn()

    # Discover history files
    history_files = discover_history_files_fn(base_dir)

    # Handle --list-dates
    if args.list_dates:
        if not history_files:
            print(
                build_actionable_error(
                    "list history dates",
                    why="no history files were found in history/",
                    next_step="add history/YYYY-MM-DD.txt files or run --api-category cs.AI",
                ),
                file=sys.stderr,
            )
            return 1
        print("Available dates:")
        for d, path in history_files:
            print(f"  {d.strftime(HISTORY_DATE_FORMAT)}  ({path.name})")
        return 0

    # Determine which file(s) to load
    result = resolve_papers_fn(args, base_dir, config, history_files)
    if isinstance(result, int):
        return result
    papers, history_files, current_date_index = result

    if not papers:
        print(
            build_actionable_error(
                "start arxiv-viewer",
                why="the selected source contained no papers",
                next_step="choose another input/date or run --api-category cs.AI",
            ),
            file=sys.stderr,
        )
        return 1

    if not validate_interactive_tty_fn():
        print(
            "Error: arxiv-viewer requires an interactive TTY for the full UI.",
            file=sys.stderr,
        )
        print("Next steps:", file=sys.stderr)
        print("  - Run arxiv-viewer directly in a terminal session", file=sys.stderr)
        print("  - Use --list-dates for non-interactive output", file=sys.stderr)
        print("  - Use --date YYYY-MM-DD in an interactive terminal", file=sys.stderr)
        print("  - Use --help for command documentation", file=sys.stderr)
        return 2

    # Sort papers alphabetically by title
    papers.sort(key=lambda p: p.title.lower())

    if app_factory is None:
        from arxiv_browser.app import ArxivBrowser as _ArxivBrowser

        app_factory = _ArxivBrowser

    app = app_factory(
        papers,
        config=config,
        restore_session=not args.no_restore,
        history_files=history_files,
        current_date_index=current_date_index,
        ascii_icons=args.ascii,
    )
    app.run()
    return 0


__all__ = [
    "_api_mode_requested",
    "_configure_color_mode",
    "_configure_logging",
    "_fetch_arxiv_api_papers",
    "_fetch_latest_arxiv_digest",
    "_find_history_index",
    "_resolve_arxiv_api_mode",
    "_resolve_history_date",
    "_resolve_input_file",
    "_resolve_legacy_fallback",
    "_resolve_papers",
    "_validate_interactive_tty",
    "main",
]
