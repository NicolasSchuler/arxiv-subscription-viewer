from __future__ import annotations

import sqlite3
from argparse import ArgumentParser, Namespace
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_browser import cache_cli
from arxiv_browser.database import init_cache_db


def test_cache_info_reports_missing_database(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    assert cache_cli.run_cache_info() == 0

    captured = capsys.readouterr()
    assert str(db_path) in captured.out
    assert "not created yet" in captured.out
    assert not db_path.exists()


def test_cache_clear_dry_run_keeps_rows(tmp_path, monkeypatch, capsys) -> None:
    db_path = _seed_cache(tmp_path)
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    result = cache_cli.run_cache_clear(Namespace(llm=True, yes=False))

    assert result == 0
    captured = capsys.readouterr()
    assert "Dry run" in captured.out
    assert "summaries: 1" in captured.out
    assert _table_count(db_path, "summaries") == 1
    assert _table_count(db_path, "semantic_embeddings") == 1


def test_cache_clear_confirmed_deletes_only_selected_section(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    db_path = _seed_cache(tmp_path)
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    result = cache_cli.run_cache_clear(Namespace(llm=True, yes=True))

    assert result == 0
    captured = capsys.readouterr()
    assert "Cleared 2 cache rows" in captured.out
    assert _table_count(db_path, "summaries") == 0
    assert _table_count(db_path, "relevance_scores") == 0
    assert _table_count(db_path, "semantic_embeddings") == 1


def test_cache_info_reports_existing_database_counts(tmp_path, monkeypatch, capsys) -> None:
    db_path = _seed_cache(tmp_path)
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    assert cache_cli.run_cache_info() == 0

    out = capsys.readouterr().out
    assert "LLM summaries and relevance: 2 rows" in out
    assert "Semantic search embeddings: 1 row" in out


def test_cache_clear_missing_selection_and_missing_database(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    assert cache_cli.run_cache_clear(Namespace(yes=True)) == 2
    assert "choose one" in capsys.readouterr().out

    assert cache_cli.run_cache_clear(Namespace(all=True, yes=True)) == 0
    assert "does not exist" in capsys.readouterr().out


def test_cache_clear_all_deletes_all_known_sections(tmp_path, monkeypatch) -> None:
    db_path = _seed_cache(tmp_path)
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    assert cache_cli.run_cache_clear(Namespace(all=True, yes=True)) == 0

    assert _table_count(db_path, "summaries") == 0
    assert _table_count(db_path, "relevance_scores") == 0
    assert _table_count(db_path, "semantic_embeddings") == 0


def test_cache_cli_parser_and_dispatcher(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cache_cli.add_cache_cli_subparsers(subparsers)

    info_args = parser.parse_args(["cache-info"])
    clear_args = parser.parse_args(["cache-clear", "--paper-content"])

    assert cache_cli.run_cache_command(info_args) == 0
    assert cache_cli.run_cache_command(clear_args) == 0
    assert cache_cli.run_cache_command(Namespace(command="browse")) is None
    out = capsys.readouterr().out
    assert "Cache database" in out
    assert "does not exist" in out


def test_cache_cli_sqlite_errors_return_failure(tmp_path, monkeypatch, capsys) -> None:
    db_path = _seed_cache(tmp_path)
    monkeypatch.setattr(cache_cli, "get_cache_db_path", lambda: db_path)

    with patch("arxiv_browser.cache_cli.sqlite3.connect", side_effect=sqlite3.Error("locked")):
        assert cache_cli.run_cache_info() == 1
    assert "failed to inspect" in capsys.readouterr().err

    with patch("arxiv_browser.cache_cli.sqlite3.connect", side_effect=sqlite3.Error("locked")):
        assert cache_cli.run_cache_clear(Namespace(llm=True, yes=True)) == 1
    assert "failed to clear" in capsys.readouterr().err


def test_cache_table_helpers_handle_missing_and_unknown_tables(tmp_path) -> None:
    db_path = _seed_cache(tmp_path)
    with closing(sqlite3.connect(str(db_path))) as conn:
        assert cache_cli._count_rows(conn, "not_a_table") == 0
    assert cache_cli._tables_for_key("all").count("summaries") == 1
    with pytest.raises(ValueError, match="unknown cache section"):
        cache_cli._tables_for_key("unknown")


def _seed_cache(tmp_path: Path) -> Path:
    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "INSERT INTO summaries VALUES (?, ?, ?, ?)",
            ("2401.00001", "hash", "summary", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO relevance_scores VALUES (?, ?, ?, ?, ?)",
            ("2401.00001", "interests", 8, "reason", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO semantic_embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "2401.00001",
                "fastembed",
                "model",
                "text-hash",
                3,
                "float32",
                b"abc",
                "2026-01-01T00:00:00Z",
            ),
        )
    return db_path


def _table_count(db_path: Path, table: str) -> int:
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # nosec B608
    return int(row[0])
