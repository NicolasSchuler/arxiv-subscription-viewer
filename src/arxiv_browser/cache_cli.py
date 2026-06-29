"""Cache inspection and cleanup helpers for CLI subcommands."""

from __future__ import annotations

import sqlite3
import sys
from argparse import Namespace, _SubParsersAction
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from arxiv_browser.database import get_cache_db_path, init_cache_db


@dataclass(frozen=True, slots=True)
class CacheSection:
    """A named group of cache tables exposed through the CLI."""

    key: str
    label: str
    tables: tuple[str, ...]


CACHE_SECTIONS: tuple[CacheSection, ...] = (
    CacheSection("llm", "LLM summaries and relevance", ("summaries", "relevance_scores")),
    CacheSection("semantic", "Semantic search embeddings", ("semantic_embeddings",)),
    CacheSection("paper-content", "Full-paper content", ("paper_content",)),
    CacheSection(
        "enrichment",
        "External enrichment data",
        (
            "s2_papers",
            "s2_paper_fetch_state",
            "s2_recommendations",
            "s2_recommendation_fetch_state",
            "s2_citation_graph",
            "s2_citation_graph_fetches",
            "hf_daily_papers",
            "hf_daily_fetch_state",
            "conference_deadlines",
            "conference_deadlines_fetch_state",
        ),
    ),
)
CACHE_SECTION_KEYS = tuple(section.key for section in CACHE_SECTIONS)
CACHE_COMMANDS = ("cache-info", "cache-clear")


def add_cache_cli_subparsers(subparsers: _SubParsersAction[Any]) -> None:
    """Register cache-management subcommands on the root CLI parser."""
    subparsers.add_parser(
        "cache-info",
        help="Show local cache database location and row counts",
        description="Print local cache database status and grouped row counts.",
    )
    cache_clear_parser = subparsers.add_parser(
        "cache-clear",
        help="Clear selected local cache tables",
        description="Clear one local cache section. Without --yes, only prints a dry run.",
    )
    cache_clear_group = cache_clear_parser.add_mutually_exclusive_group(required=True)
    cache_clear_group.add_argument("--all", action="store_true", help="Clear all cache sections")
    cache_clear_group.add_argument("--llm", action="store_true", help="Clear LLM cache rows")
    cache_clear_group.add_argument(
        "--semantic",
        action="store_true",
        help="Clear semantic-search embedding cache rows",
    )
    cache_clear_group.add_argument(
        "--enrichment",
        action="store_true",
        help="Clear Semantic Scholar, HuggingFace, and conference cache rows",
    )
    cache_clear_group.add_argument(
        "--paper-content",
        action="store_true",
        help="Clear full-paper content cache rows",
    )
    cache_clear_parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete rows instead of printing a dry run",
    )


def run_cache_command(args: Namespace) -> int | None:
    """Run a config-free cache command, returning ``None`` for non-cache commands."""
    command = getattr(args, "command", None)
    if command == "cache-info":
        return run_cache_info()
    if command == "cache-clear":
        return run_cache_clear(args)
    return None


def run_cache_info() -> int:
    """Print cache location and table-group row counts."""
    db_path = get_cache_db_path()
    print(f"Cache database: {db_path}")
    if not db_path.exists():
        print("Status: not created yet")
        return 0

    try:
        init_cache_db(db_path)
        with closing(sqlite3.connect(str(db_path))) as conn:
            for section in CACHE_SECTIONS:
                total = sum(_count_rows(conn, table) for table in section.tables)
                print(f"{section.label}: {total} row{'s' if total != 1 else ''}")
    except sqlite3.Error as exc:
        print(f"Error: failed to inspect cache database {db_path}: {exc}", file=sys.stderr)
        return 1
    return 0


def run_cache_clear(args: Any) -> int:
    """Clear one cache section, performing a dry run unless --yes was passed."""
    section_key = _selected_section_key(args)
    if section_key is None:
        print("Error: choose one of --all, --llm, --semantic, --enrichment, --paper-content")
        return 2
    db_path = get_cache_db_path()
    if not db_path.exists():
        print(f"Cache database does not exist yet: {db_path}")
        return 0

    tables = _tables_for_key(section_key)
    try:
        init_cache_db(db_path)
        with closing(sqlite3.connect(str(db_path))) as conn:
            counts = {table: _count_rows(conn, table) for table in tables}
            total = sum(counts.values())
            if not bool(getattr(args, "yes", False)):
                print(f"Dry run: would clear {total} cache row{'s' if total != 1 else ''}.")
                _print_table_counts(counts)
                print("Pass --yes to delete these rows.")
                return 0
            with conn:
                for table in tables:
                    if _table_exists(conn, table):
                        conn.execute(f"DELETE FROM {table}")  # nosec B608
            print(f"Cleared {total} cache row{'s' if total != 1 else ''}.")
    except sqlite3.Error as exc:
        print(f"Error: failed to clear cache database {db_path}: {exc}", file=sys.stderr)
        return 1
    return 0


def _selected_section_key(args: Any) -> str | None:
    for key in ("all", *CACHE_SECTION_KEYS):
        attr = key.replace("-", "_")
        if bool(getattr(args, attr, False)):
            return key
    return None


def _tables_for_key(key: str) -> tuple[str, ...]:
    if key == "all":
        return tuple(
            _unique_tables(table for section in CACHE_SECTIONS for table in section.tables)
        )
    for section in CACHE_SECTIONS:
        if section.key == key:
            return section.tables
    raise ValueError(f"unknown cache section: {key}")


def _unique_tables(tables: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for table in tables:
        if table not in seen:
            seen.add(table)
            result.append(table)
    return result


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # nosec B608
    return int(row[0]) if row else 0


def _print_table_counts(counts: dict[str, int]) -> None:
    for table, count in counts.items():
        print(f"  {table}: {count}")


__all__ = [
    "CACHE_COMMANDS",
    "CACHE_SECTIONS",
    "CACHE_SECTION_KEYS",
    "add_cache_cli_subparsers",
    "run_cache_clear",
    "run_cache_command",
    "run_cache_info",
]
