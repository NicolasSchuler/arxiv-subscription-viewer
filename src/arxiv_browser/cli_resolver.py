"""Resolve CLI startup paper sources for the arXiv browser."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.config import _coerce_arxiv_api_max_results
from arxiv_browser.models import ArxivSearchRequest, Paper, UserConfig
from arxiv_browser.parsing import HISTORY_DATE_FORMAT, parse_arxiv_file
from arxiv_browser.services import arxiv_api_service as _arxiv_api_service

ResolvePapersResult = tuple[list[Paper], list[tuple[date, Path]], int] | int

ARXIV_API_URL = _arxiv_api_service.ARXIV_API_URL
ARXIV_API_TIMEOUT = _arxiv_api_service.ARXIV_API_TIMEOUT
ARXIV_API_USER_AGENT = _arxiv_api_service.ARXIV_API_USER_AGENT
ARXIV_API_MIN_INTERVAL_SECONDS = _arxiv_api_service.ARXIV_API_MIN_INTERVAL_SECONDS


def _fetch_arxiv_api_papers(
    *,
    query: str,
    field: str,
    category: str,
    max_results: int,
    start: int = 0,
) -> list[Paper]:
    """Fetch one page of papers from the arXiv API."""
    return _arxiv_api_service.fetch_page_sync(
        request=ArxivSearchRequest(query=query, field=field, category=category),
        start=start,
        max_results=max_results,
        timeout_seconds=ARXIV_API_TIMEOUT,
        user_agent=ARXIV_API_USER_AGENT,
    )


def _fetch_latest_arxiv_digest(
    *,
    query: str,
    field: str,
    category: str,
    max_results: int,
) -> list[Paper]:
    """Fetch all papers from the newest matching arXiv day.

    This mirrors email-digest behavior: collect papers from the latest
    available day, then stop when older days begin. Paginates through the
    arXiv API with ``max_results``-sized pages, sleeping between requests
    to respect rate limits.
    """
    return _arxiv_api_service.fetch_latest_day_digest(
        request=ArxivSearchRequest(query=query, field=field, category=category),
        max_results=max_results,
        fetch_page_fn=lambda **kwargs: _fetch_arxiv_api_papers(
            query=query,
            field=field,
            category=category,
            max_results=int(kwargs["max_results"]),
            start=int(kwargs["start"]),
        ),
        sleep_fn=time.sleep,
        min_interval_seconds=ARXIV_API_MIN_INTERVAL_SECONDS,
    )


def _api_mode_requested(args: argparse.Namespace) -> bool:
    """Return True when CLI args request startup via arXiv API."""
    return getattr(args, "command", None) == "search"


def _fetch_api_mode_papers(
    *,
    query: str,
    field: str,
    category: str,
    max_results: int,
    page_mode: bool,
) -> list[Paper]:
    if page_mode:
        return _fetch_arxiv_api_papers(
            query=query,
            field=field,
            category=category,
            max_results=max_results,
        )
    return _fetch_latest_arxiv_digest(
        query=query,
        field=field,
        category=category,
        max_results=max_results,
    )


def _print_arxiv_api_status_error(exc: httpx.HTTPStatusError) -> None:
    status_code = exc.response.status_code
    if status_code == 429:
        message = "Error: arXiv API rate limit reached (HTTP 429). Wait a few seconds and retry."
    elif status_code >= 500:
        message = f"Error: arXiv API is unavailable right now (HTTP {status_code}). Retry later."
    else:
        message = (
            f"Error: arXiv API rejected the request (HTTP {status_code}). "
            "Check --query/--category and retry."
        )
    print(message, file=sys.stderr)


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

    api_query = str(getattr(args, "query", "") or "").strip()
    api_category = str(getattr(args, "category", "") or "").strip()
    api_field = str(getattr(args, "field", "all"))
    max_results_arg = getattr(args, "max_results", None)
    max_results = _coerce_arxiv_api_max_results(
        max_results_arg if max_results_arg is not None else config.arxiv_api_max_results
    )

    try:
        papers = _fetch_api_mode_papers(
            query=api_query,
            field=api_field,
            category=api_category,
            max_results=max_results,
            page_mode=str(getattr(args, "mode", "latest")) == "page",
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        _print_arxiv_api_status_error(exc)
        return 1
    except (httpx.HTTPError, OSError):
        print(
            "Error: Failed to fetch papers from arXiv API (network or I/O error).",
            file=sys.stderr,
        )
        return 1

    if not papers:
        print("Error: No papers found for the requested search.", file=sys.stderr)
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
                    'or run "arxiv-viewer search --category cs.AI"'
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


def _restore_history_index(
    history_files: list[tuple[date, Path]],
    config: UserConfig,
) -> int | None:
    if not config.session.current_date:
        return None
    try:
        saved_date = datetime.strptime(config.session.current_date, HISTORY_DATE_FORMAT).date()
    except ValueError:
        return None
    return _find_history_index(history_files, saved_date)


def _resolve_history_source(
    args: argparse.Namespace,
    config: UserConfig,
    history_files: list[tuple[date, Path]],
) -> tuple[list[Paper], list[tuple[date, Path]], int] | int:
    current_date_index = 0
    selected_date = getattr(args, "date", None)
    if selected_date:
        idx = _resolve_history_date(history_files, selected_date)
        if idx is None:
            return 1
        current_date_index = idx
    elif not bool(getattr(args, "no_restore", False)):
        current_date_index = _restore_history_index(history_files, config) or 0

    _, arxiv_file = history_files[current_date_index]
    try:
        papers = parse_arxiv_file(arxiv_file)
    except OSError as e:
        print(f"Error: Failed to read {arxiv_file}: {e}", file=sys.stderr)
        return 1
    return (papers, history_files, current_date_index)


def _resolve_papers(
    args: argparse.Namespace,
    base_dir: Path,
    config: UserConfig,
    history_files: list[tuple[date, Path]],
) -> ResolvePapersResult:
    """Resolve which papers to load based on CLI args.

    Returns:
        A ``(papers, history_files, date_index)`` tuple on success, or an
        ``int`` exit code on failure.
    """
    input_path = getattr(args, "input", None)
    if input_path is not None:
        result = _resolve_input_file(input_path)
        if isinstance(result, int):
            return result
        return (result, [], 0)

    api_result = _resolve_arxiv_api_mode(args, config)
    if api_result is not None:
        return api_result

    if history_files:
        return _resolve_history_source(args, config, history_files)

    result = _resolve_legacy_fallback(base_dir)
    if isinstance(result, int):
        return result
    return (result, [], 0)
