#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.themes import THEME_NAMES, THEMES
from tests.support.canonical_exports import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    ARXIV_DATE_FORMAT,
    DEFAULT_CATEGORY_COLOR,
    DEFAULT_LLM_PROMPT,
    LLM_PRESETS,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    SUBPROCESS_TIMEOUT,
    SUMMARY_MODES,
    TAG_NAMESPACE_COLORS,
    Paper,
    PaperCollection,
    PaperMetadata,
    QueryToken,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
    build_arxiv_search_query,
    build_llm_prompt,
    clean_latex,
    escape_bibtex,
    export_metadata,
    extract_text_from_html,
    extract_year,
    format_categories,
    format_collection_as_markdown,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    format_summary_as_rich,
    generate_citation_key,
    get_pdf_download_path,
    get_summary_db_path,
    get_tag_color,
    import_metadata,
    insert_implicit_and,
    load_config,
    normalize_arxiv_id,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
    parse_arxiv_version_map,
    parse_tag_namespace,
    pill_label_for_token,
    reconstruct_query,
    save_config,
    to_rpn,
    tokenize_query,
)

# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    def test_list_dates_with_files(self, tmp_path, monkeypatch, capsys):
        """`dates` should print dates and return 0."""
        from datetime import date as datemod

        from arxiv_browser.app import main

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        (history_dir / "2024-01-15.txt").write_text("test content")

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "dates"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), history_dir / "2024-01-15.txt")],
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "2024-01-15" in captured.out

    def test_list_dates_empty_history(self, monkeypatch, capsys):
        """`dates` with no files should return 1."""
        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "dates"])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Could not list history dates." in captured.err
        assert "Next step:" in captured.err

    def test_input_file_not_found(self, tmp_path, monkeypatch, capsys):
        """-i nonexistent.txt should return 1."""
        from arxiv_browser.app import main

        nonexistent = str(tmp_path / "nonexistent.txt")
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", nonexistent])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "not found" in captured.err

    def test_input_file_is_directory(self, tmp_path, monkeypatch, capsys):
        """-i /some/dir should return 1."""
        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", str(tmp_path)])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "directory" in captured.err

    def test_history_file_read_error_returns_1(self, tmp_path, monkeypatch, capsys):
        """Unreadable selected history file should return 1 instead of crashing."""
        from datetime import date as datemod

        from arxiv_browser.app import main

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        history_file = history_dir / "2024-01-15.txt"
        history_file.write_text("dummy")

        def raise_read_error(_path):
            msg = "boom"
            raise OSError(msg)

        monkeypatch.setattr("sys.argv", ["arxiv_browser"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), history_file)],
        )
        monkeypatch.setattr("arxiv_browser.cli.parse_arxiv_file", raise_read_error)
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Failed to read" in captured.err

    def test_no_papers_exits_with_error(self, tmp_path, monkeypatch, capsys):
        """Empty file should return 1 with actionable guidance."""
        from arxiv_browser.app import main

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-i", str(empty_file)])
        monkeypatch.setattr("arxiv_browser.app.discover_history_files", lambda base_dir: [])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Could not start arxiv-viewer." in captured.err
        assert "Next step:" in captured.err

    def test_invalid_date_format(self, monkeypatch, capsys):
        """--date Jan-15-2024 should return 1 with 'Invalid date'."""
        from datetime import date as datemod

        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "Jan-15-2024"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), Path("/fake/2024-01-15.txt"))],
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Invalid date" in captured.err

    def test_date_not_found(self, monkeypatch, capsys):
        """--date 2099-01-01 should return 1 with 'No file found'."""
        from datetime import date as datemod

        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "2099-01-01"])
        monkeypatch.setattr(
            "arxiv_browser.app.discover_history_files",
            lambda base_dir: [(datemod(2024, 1, 15), Path("/fake/2024-01-15.txt"))],
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "No file found" in captured.err

    def test_non_tty_returns_actionable_error(self, monkeypatch, capsys, make_paper):
        """Running the interactive path without a TTY should fail with guidance."""
        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2401.99999")
        monkeypatch.setattr("sys.argv", ["arxiv_browser"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr(
            "arxiv_browser.app._resolve_papers",
            lambda args, base_dir, config, history_files: ([paper], [], 0),
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        result = main()
        captured = capsys.readouterr()
        assert result == 2
        assert "requires an interactive TTY" in captured.err
        assert "arxiv-viewer dates" in captured.err

    def test_main_applies_ascii_and_color_flags(self, monkeypatch, make_paper):
        """CLI flags should configure ASCII icon mode and color environment hints."""
        import os

        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2401.99998")
        captured_kwargs = {}

        class FakeApp:
            def __init__(self, papers, *, options):
                captured_kwargs["papers"] = papers
                captured_kwargs["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--ascii", "--color", "never"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr(
            "arxiv_browser.app._resolve_papers",
            lambda args, base_dir, config, history_files: ([paper], [], 0),
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)
        monkeypatch.setenv("NO_COLOR", "")
        monkeypatch.setenv("FORCE_COLOR", "")

        result = main()
        assert result == 0
        assert captured_kwargs["options"].ascii_icons is True
        assert os.environ.get("NO_COLOR") == "1"

    def test_main_honors_no_color_env_var(self, monkeypatch, make_paper):
        """NO_COLOR should disable terminal colors even without --no-color."""
        import os

        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2401.99997")

        class FakeApp:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self):
                return None

        monkeypatch.setattr("sys.argv", ["arxiv_browser"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr(
            "arxiv_browser.app._resolve_papers",
            lambda args, base_dir, config, history_files: ([paper], [], 0),
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")

        result = main()
        assert result == 0
        assert os.environ.get("NO_COLOR") == "1"
        assert os.environ.get("FORCE_COLOR") is None

    def test_main_explicit_color_flag_overrides_no_color_env_var(self, monkeypatch, make_paper):
        """``--color always`` should override a globally exported ``NO_COLOR``."""
        import os

        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2401.99996")

        class FakeApp:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self):
                return None

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--color", "always"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr(
            "arxiv_browser.app._resolve_papers",
            lambda args, base_dir, config, history_files: ([paper], [], 0),
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)

        result = main()
        assert result == 0
        assert os.environ.get("FORCE_COLOR") == "1"
        assert os.environ.get("NO_COLOR") is None

    def test_search_category_fetches_latest_day_and_runs_app(self, monkeypatch, make_paper):
        """`search --category` should load startup papers in latest-day digest mode."""
        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2602.00001")
        captured_kwargs = {}
        captured_papers = []
        api_calls: list[dict[str, object]] = []

        class FakeApp:
            def __init__(self, papers, *, options):
                captured_papers.extend(papers)
                captured_kwargs["options"] = options

            def run(self):
                return None

        def fake_fetch(**kwargs):
            api_calls.append(kwargs)
            return [paper]

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "search", "--category", "cs.AI"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr("arxiv_browser.cli._fetch_latest_arxiv_digest", fake_fetch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)

        result = main()
        assert result == 0
        assert captured_papers == [paper]
        assert captured_kwargs["options"].history_files == []
        assert captured_kwargs["options"].restore_session is False
        assert api_calls[0]["category"] == "cs.AI"

    def test_cli_main_preserves_legacy_factory_kwargs(self, make_paper):
        """Injected legacy app factories should still receive the old keyword shape."""
        import arxiv_browser.cli as cli_mod

        paper = make_paper(arxiv_id="2602.00003")
        config = UserConfig()
        captured: dict[str, object] = {}

        class LegacyFactory:
            def __init__(
                self,
                papers,
                config=None,
                restore_session=True,
                history_files=None,
                current_date_index=0,
                ascii_icons=False,
            ):
                captured["papers"] = papers
                captured["config"] = config
                captured["restore_session"] = restore_session
                captured["history_files"] = history_files
                captured["current_date_index"] = current_date_index
                captured["ascii_icons"] = ascii_icons

            def run(self):
                return None

        result = cli_mod.main(
            ["browse"],
            deps=cli_mod.CliDependencies(
                load_config_fn=lambda: config,
                discover_history_files_fn=lambda _base: [],
                resolve_papers_fn=lambda args, base_dir, cfg, history_files: ([paper], [], 0),
                configure_logging_fn=lambda _debug: None,
                configure_color_mode_fn=lambda _mode: None,
                validate_interactive_tty_fn=lambda: True,
                app_factory=LegacyFactory,
            ),
        )

        assert result == 0
        assert captured["papers"] == [paper]
        assert captured["config"] is config
        assert captured["restore_session"] is True
        assert captured["history_files"] == []
        assert captured["current_date_index"] == 0
        assert captured["ascii_icons"] is False

    def test_app_main_preserves_legacy_factory_kwargs(self, monkeypatch, make_paper):
        """Compatibility wrapper should still support legacy injected app factories."""
        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2602.00004")
        captured: dict[str, object] = {}

        class LegacyFactory:
            def __init__(
                self,
                papers,
                config=None,
                restore_session=True,
                history_files=None,
                current_date_index=0,
                ascii_icons=False,
            ):
                captured["papers"] = papers
                captured["config"] = config
                captured["restore_session"] = restore_session
                captured["history_files"] = history_files
                captured["current_date_index"] = current_date_index
                captured["ascii_icons"] = ascii_icons

            def run(self):
                return None

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "browse"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr(
            "arxiv_browser.app._resolve_papers",
            lambda args, base_dir, config, history_files: ([paper], [], 0),
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", LegacyFactory)

        result = main()

        assert result == 0
        assert captured["papers"] == [paper]
        assert isinstance(captured["config"], UserConfig)
        assert captured["restore_session"] is True
        assert captured["history_files"] == []
        assert captured["current_date_index"] == 0
        assert captured["ascii_icons"] is False

    def test_package_main_uses_app_patch_surface(self, monkeypatch, make_paper):
        """Top-level package main should preserve the app-module patch surface."""
        from arxiv_browser import main

        paper = make_paper(arxiv_id="2602.00005")
        seen: dict[str, bool] = {"load_config": False, "resolve_papers": False}

        class FakeApp:
            def __init__(self, papers, *_args, **_kwargs):
                assert papers == [paper]

            def run(self):
                return None

        def fake_load_config() -> UserConfig:
            seen["load_config"] = True
            return UserConfig()

        def fake_resolve(args, base_dir, config, history_files):
            seen["resolve_papers"] = True
            return ([paper], [], 0)

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "browse"])
        monkeypatch.setattr("arxiv_browser.app.load_config", fake_load_config)
        monkeypatch.setattr("arxiv_browser.app._resolve_papers", fake_resolve)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)

        result = main()

        assert result == 0
        assert seen == {"load_config": True, "resolve_papers": True}

    def test_search_page_mode_fetches_single_page(self, monkeypatch, make_paper):
        """`search --mode page` should use a single API page instead of latest-day mode."""
        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2602.00002")
        captured_papers = []
        page_calls: list[dict[str, object]] = []

        class FakeApp:
            def __init__(self, papers, *_args, **_kwargs):
                captured_papers.extend(papers)

            def run(self):
                return None

        def fail_digest_fetch(**_kwargs):
            raise AssertionError("latest-day fetch should not run in --api-page-mode")

        def fake_page_fetch(**kwargs):
            page_calls.append(kwargs)
            return [paper]

        monkeypatch.setattr(
            "sys.argv", ["arxiv_browser", "search", "--query", "transformer", "--mode", "page"]
        )
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr("arxiv_browser.cli._fetch_latest_arxiv_digest", fail_digest_fetch)
        monkeypatch.setattr("arxiv_browser.cli._fetch_arxiv_api_papers", fake_page_fetch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)

        result = main()
        assert result == 0
        assert captured_papers == [paper]
        assert page_calls[0]["query"] == "transformer"

    def test_search_requires_query_or_category(self, monkeypatch, capsys):
        """`search` should require either a query or a category."""
        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "search"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Search query or category must be provided" in captured.err

    def test_legacy_api_flags_are_rejected(self, monkeypatch, capsys):
        """Legacy flat API flags should fail fast under the subcommand CLI."""
        from arxiv_browser.app import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--api-category", "cs.AI"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "unrecognized arguments" in captured.err

    def test_browse_alias_inserts_subcommand_for_browse_flags(self, monkeypatch, make_paper):
        """Bare invocations with browse flags should still resolve as `browse`."""
        from arxiv_browser.app import main

        paper = make_paper(arxiv_id="2401.42424")
        captured_args: dict[str, object] = {}

        class FakeApp:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self):
                return None

        def fake_resolve(args, _base_dir, _config, _history_files):
            captured_args["command"] = args.command
            captured_args["date"] = args.date
            return ([paper], [], 0)

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--date", "2024-01-15"])
        monkeypatch.setattr("arxiv_browser.app.load_config", lambda: UserConfig())
        monkeypatch.setattr("arxiv_browser.app._resolve_papers", fake_resolve)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("arxiv_browser.app.ArxivBrowser", FakeApp)

        result = main()
        assert result == 0
        assert captured_args == {"command": "browse", "date": "2024-01-15"}


class TestResolvePapersHistoryRestore:
    """Tests for history/session restore precedence in CLI paper resolution."""

    def test_restores_saved_date_even_when_older(self, monkeypatch, tmp_path):
        import argparse
        from datetime import date as dt_date

        from arxiv_browser.cli import _resolve_papers

        older = tmp_path / "2024-01-14.txt"
        newer = tmp_path / "2024-01-15.txt"
        history_files = [(dt_date(2024, 1, 15), newer), (dt_date(2024, 1, 14), older)]
        config = UserConfig()
        config.session.current_date = "2024-01-14"
        parsed: list[Path] = []

        def fake_parse(path: Path) -> list[Paper]:
            parsed.append(path)
            return [
                Paper(
                    arxiv_id="2401.00001",
                    date="Mon, 14 Jan 2024",
                    title="Test",
                    authors="A",
                    categories="cs.AI",
                    comments=None,
                    abstract="",
                    abstract_raw="",
                    url="https://arxiv.org/abs/2401.00001",
                )
            ]

        monkeypatch.setattr("arxiv_browser.cli.parse_arxiv_file", fake_parse)
        args = argparse.Namespace(input=None, date=None, no_restore=False)
        result = _resolve_papers(args, tmp_path, config, history_files)

        assert not isinstance(result, int)
        _, _, current_date_index = result
        assert current_date_index == 1
        assert parsed == [older]

    def test_invalid_saved_date_falls_back_to_newest(self, monkeypatch, tmp_path):
        import argparse
        from datetime import date as dt_date

        from arxiv_browser.cli import _resolve_papers

        older = tmp_path / "2024-01-14.txt"
        newer = tmp_path / "2024-01-15.txt"
        history_files = [(dt_date(2024, 1, 15), newer), (dt_date(2024, 1, 14), older)]
        config = UserConfig()
        config.session.current_date = "not-a-date"
        parsed: list[Path] = []

        def fake_parse(path: Path) -> list[Paper]:
            parsed.append(path)
            return [
                Paper(
                    arxiv_id="2401.00002",
                    date="Tue, 15 Jan 2024",
                    title="Test 2",
                    authors="B",
                    categories="cs.LG",
                    comments=None,
                    abstract="",
                    abstract_raw="",
                    url="https://arxiv.org/abs/2401.00002",
                )
            ]

        monkeypatch.setattr("arxiv_browser.cli.parse_arxiv_file", fake_parse)
        args = argparse.Namespace(input=None, date=None, no_restore=False)
        result = _resolve_papers(args, tmp_path, config, history_files)

        assert not isinstance(result, int)
        _, _, current_date_index = result
        assert current_date_index == 0
        assert parsed == [newer]

    def test_explicit_date_overrides_session_restore(self, monkeypatch, tmp_path):
        import argparse
        from datetime import date as dt_date

        from arxiv_browser.cli import _resolve_papers

        older = tmp_path / "2024-01-14.txt"
        newer = tmp_path / "2024-01-15.txt"
        history_files = [(dt_date(2024, 1, 15), newer), (dt_date(2024, 1, 14), older)]
        config = UserConfig()
        config.session.current_date = "2024-01-14"
        parsed: list[Path] = []

        def fake_parse(path: Path) -> list[Paper]:
            parsed.append(path)
            return [
                Paper(
                    arxiv_id="2401.00003",
                    date="Tue, 15 Jan 2024",
                    title="Test 3",
                    authors="C",
                    categories="cs.CL",
                    comments=None,
                    abstract="",
                    abstract_raw="",
                    url="https://arxiv.org/abs/2401.00003",
                )
            ]

        monkeypatch.setattr("arxiv_browser.cli.parse_arxiv_file", fake_parse)
        args = argparse.Namespace(input=None, date="2024-01-15", no_restore=False)
        result = _resolve_papers(args, tmp_path, config, history_files)

        assert not isinstance(result, int)
        _, _, current_date_index = result
        assert current_date_index == 0
        assert parsed == [newer]

    def test_api_mode_takes_precedence_over_history(
        self,
        monkeypatch,
        tmp_path,
        make_paper,
    ):
        import argparse
        from datetime import date as dt_date

        from arxiv_browser.cli import _resolve_papers

        history_file = tmp_path / "2024-01-15.txt"
        history_files = [(dt_date(2024, 1, 15), history_file)]
        config = UserConfig()
        api_paper = make_paper(arxiv_id="2602.00077")
        api_calls: list[dict[str, object]] = []

        def fail_parse(_path: Path) -> list[Paper]:
            raise AssertionError("history parser should not be called in API mode")

        def fake_fetch(**kwargs):
            api_calls.append(kwargs)
            return [api_paper]

        monkeypatch.setattr("arxiv_browser.cli.parse_arxiv_file", fail_parse)
        monkeypatch.setattr("arxiv_browser.cli._fetch_latest_arxiv_digest", fake_fetch)

        args = argparse.Namespace(
            command="search",
            input=None,
            date=None,
            no_restore=False,
            query=None,
            field="all",
            category="cs.LG",
            max_results=None,
            mode="latest",
        )

        result = _resolve_papers(args, tmp_path, config, history_files)
        assert not isinstance(result, int)
        papers, resolved_history, current_date_index = result
        assert papers == [api_paper]
        assert resolved_history == []
        assert current_date_index == 0
        assert api_calls[0]["category"] == "cs.LG"

    def test_fetch_latest_digest_paginates_until_older_day(self, monkeypatch, make_paper):
        from arxiv_browser.cli import _fetch_latest_arxiv_digest

        latest_day = "Mon, 17 Feb 2026"
        older_day = "Sun, 16 Feb 2026"
        page_calls: list[int] = []
        pages = [
            [
                make_paper(arxiv_id="2602.00010", date=latest_day),
                make_paper(arxiv_id="2602.00011", date=latest_day),
            ],
            [
                make_paper(arxiv_id="2602.00012", date=latest_day),
                make_paper(arxiv_id="2602.00013", date=older_day),
            ],
        ]

        def fake_fetch(**kwargs):
            page_calls.append(int(kwargs["start"]))
            return pages.pop(0)

        monkeypatch.setattr("arxiv_browser.cli._fetch_arxiv_api_papers", fake_fetch)
        monkeypatch.setattr("arxiv_browser.cli.time.sleep", lambda _seconds: None)

        papers = _fetch_latest_arxiv_digest(
            query="",
            field="all",
            category="cs.AI",
            max_results=2,
        )

        assert [p.arxiv_id for p in papers] == ["2602.00010", "2602.00011", "2602.00012"]
        assert page_calls == [0, 2]

    def test_fetch_latest_digest_skips_invalid_first_date(self, monkeypatch, make_paper):
        from arxiv_browser.cli import _fetch_latest_arxiv_digest

        latest_day = "Mon, 17 Feb 2026"
        older_day = "Sun, 16 Feb 2026"
        page_calls: list[int] = []
        pages = [
            [
                make_paper(arxiv_id="2602.00010", date="not-a-date"),
                make_paper(arxiv_id="2602.00011", date=latest_day),
                make_paper(arxiv_id="2602.00012", date=latest_day),
            ],
            [
                make_paper(arxiv_id="2602.00013", date=latest_day),
                make_paper(arxiv_id="2602.00014", date=older_day),
            ],
        ]

        def fake_fetch(**kwargs):
            page_calls.append(int(kwargs["start"]))
            return pages.pop(0)

        monkeypatch.setattr("arxiv_browser.cli._fetch_arxiv_api_papers", fake_fetch)
        monkeypatch.setattr("arxiv_browser.cli.time.sleep", lambda _seconds: None)

        papers = _fetch_latest_arxiv_digest(
            query="",
            field="all",
            category="cs.AI",
            max_results=3,
        )

        assert [p.arxiv_id for p in papers] == ["2602.00011", "2602.00012", "2602.00013"]
        assert page_calls == [0, 3]
