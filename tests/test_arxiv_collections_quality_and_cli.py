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


class TestUpdatePaperParity:
    """Verify update_paper() produces correct output via section helpers."""

    def test_full_paper_all_sections(self, make_paper):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        paper = make_paper(
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            arxiv_id="1706.03762",
            date="2017-06-12",
            categories="cs.CL cs.LG",
            url="https://arxiv.org/abs/1706.03762",
            abstract="The dominant sequence transduction models...",
        )
        details.update_paper(paper, tags=["topic:transformers"])
        output = details.content
        assert "Attention Is All You Need" in output
        assert "1706.03762" in output
        assert "Vaswani" in output
        assert "URL" in output
        assert "▾ Abstract" in output
        assert "▾ Authors" in output
        assert "▾ Tags" in output

    def test_none_paper(self):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        details.update_paper(None)
        assert "Select a paper" in details.content

    def test_collapsed_sections(self, make_paper):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        paper = make_paper(abstract="Test abstract")
        details.update_paper(paper, collapsed_sections=["abstract", "authors"])
        output = details.content
        assert "▸ Abstract" in output
        assert "▸ Authors" in output
        assert "Test abstract" not in output

    def test_scan_mode_truncates_long_sections(self, make_paper):
        from tests.support.canonical_exports import PaperDetails

        details = PaperDetails()
        paper = make_paper(
            authors=", ".join([f"Author {i}" for i in range(20)]),
            abstract=" ".join(["transformer"] * 140),
        )
        details.update_paper(paper, detail_mode="scan")
        output = str(details.content)
        assert "▾ Abstract" in output
        assert "▾ Authors" in output
        assert "..." in output or "\u2026" in output


class TestPillLabelForToken:
    """Tests for pill_label_for_token()."""

    def test_plain_term(self):
        tok = QueryToken(kind="term", value="transformer")
        assert pill_label_for_token(tok) == "transformer"

    def test_field_term(self):
        tok = QueryToken(kind="term", value="cs.AI", field="cat")
        assert pill_label_for_token(tok) == "cat:cs.AI"

    def test_phrase_term(self):
        tok = QueryToken(kind="term", value="neural network", phrase=True)
        assert pill_label_for_token(tok) == '"neural network"'

    def test_field_and_phrase(self):
        tok = QueryToken(kind="term", value="John Smith", field="author", phrase=True)
        assert pill_label_for_token(tok) == 'author:"John Smith"'

    def test_virtual_starred(self):
        tok = QueryToken(kind="term", value="starred")
        assert pill_label_for_token(tok) == "starred"

    def test_virtual_unread(self):
        tok = QueryToken(kind="term", value="unread")
        assert pill_label_for_token(tok) == "unread"


class TestPaperCollectionSerialization:
    """Tests for PaperCollection config round-trip."""

    def test_roundtrip(self):
        from tests.support.canonical_exports import _config_to_dict, _dict_to_config

        config = UserConfig()
        config.collections = [
            PaperCollection(
                name="ML Papers",
                description="Top ML",
                paper_ids=["2401.00001"],
                created="2026-01-01",
            ),
            PaperCollection(name="NLP", paper_ids=["2401.00002", "2401.00003"]),
        ]
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert len(restored.collections) == 2
        assert restored.collections[0].name == "ML Papers"
        assert restored.collections[0].description == "Top ML"
        assert restored.collections[0].paper_ids == ["2401.00001"]
        assert restored.collections[1].name == "NLP"
        assert len(restored.collections[1].paper_ids) == 2

    def test_max_collections_enforced(self):
        from tests.support.canonical_exports import _dict_to_config

        data = {"collections": [{"name": f"col-{i}", "paper_ids": []} for i in range(30)]}
        config = _dict_to_config(data)
        assert len(config.collections) <= MAX_COLLECTIONS

    def test_max_papers_per_collection_enforced(self):
        from tests.support.canonical_exports import _dict_to_config

        data = {"collections": [{"name": "big", "paper_ids": [f"id-{i}" for i in range(600)]}]}
        config = _dict_to_config(data)
        assert len(config.collections[0].paper_ids) <= MAX_PAPERS_PER_COLLECTION

    def test_invalid_entries_skipped(self):
        from tests.support.canonical_exports import _dict_to_config

        data = {"collections": ["not-a-dict", {"name": ""}, {"name": "valid", "paper_ids": []}]}
        config = _dict_to_config(data)
        assert len(config.collections) == 1
        assert config.collections[0].name == "valid"

    def test_non_string_paper_ids_filtered(self):
        from tests.support.canonical_exports import _dict_to_config

        data = {"collections": [{"name": "test", "paper_ids": ["ok", 123, None, "also-ok"]}]}
        config = _dict_to_config(data)
        assert config.collections[0].paper_ids == ["ok", "also-ok"]


class TestCollectionExportImport:
    """Tests for collection export/import via metadata."""

    def test_export_includes_collections(self):
        config = UserConfig()
        config.collections = [PaperCollection(name="Test", paper_ids=["id1"])]
        exported = export_metadata(config)
        assert "collections" in exported
        assert len(exported["collections"]) == 1
        assert exported["collections"][0]["name"] == "Test"

    def test_import_merges_by_name(self):
        config = UserConfig()
        config.collections = [PaperCollection(name="Existing", paper_ids=["a"])]
        data = {
            "format": "arxiv-browser-metadata",
            "collections": [
                {"name": "Existing", "paper_ids": ["b"]},
                {"name": "New", "paper_ids": ["c"]},
            ],
        }
        _, _, _, col_n = import_metadata(data, config)
        assert col_n == 1
        assert len(config.collections) == 2
        assert config.collections[0].name == "Existing"
        assert config.collections[0].paper_ids == ["a"]  # unchanged
        assert config.collections[1].name == "New"

    def test_import_returns_4_tuple(self):
        config = UserConfig()
        data = {"format": "arxiv-browser-metadata"}
        result = import_metadata(data, config)
        assert len(result) == 4


class TestFormatCollectionAsMarkdown:
    """Tests for format_collection_as_markdown()."""

    def test_basic_format(self, make_paper):
        p = make_paper(title="My Paper", arxiv_id="2401.00001")
        col = PaperCollection(
            name="Reading List", description="Q1 papers", paper_ids=["2401.00001"]
        )
        md = format_collection_as_markdown(col, {"2401.00001": p})
        assert "# Reading List" in md
        assert "Q1 papers" in md
        assert "My Paper" in md
        assert "1 papers" in md

    def test_missing_papers_handled(self, make_paper):
        col = PaperCollection(name="Test", paper_ids=["unknown-id"])
        md = format_collection_as_markdown(col, {})
        assert "unknown-id" in md
        assert "not loaded" in md

    def test_empty_collection(self, make_paper):
        col = PaperCollection(name="Empty")
        md = format_collection_as_markdown(col, {})
        assert "# Empty" in md
        assert "0 papers" in md


class TestCollectionActions:
    """Tests for add_to_collection dedup and max enforcement."""

    def test_add_dedup(self):
        col = PaperCollection(name="Test", paper_ids=["a", "b"])
        # Simulate add logic
        existing = set(col.paper_ids)
        for pid in ["b", "c"]:
            if pid not in existing and len(col.paper_ids) < MAX_PAPERS_PER_COLLECTION:
                col.paper_ids.append(pid)
                existing.add(pid)
        assert col.paper_ids == ["a", "b", "c"]

    def test_max_papers_enforced(self):
        col = PaperCollection(
            name="Test", paper_ids=[f"p{i}" for i in range(MAX_PAPERS_PER_COLLECTION)]
        )
        existing = set(col.paper_ids)
        for pid in ["new1", "new2"]:
            if pid not in existing and len(col.paper_ids) < MAX_PAPERS_PER_COLLECTION:
                col.paper_ids.append(pid)
                existing.add(pid)
        assert len(col.paper_ids) == MAX_PAPERS_PER_COLLECTION


class TestReconstructQuery:
    """Tests for reconstruct_query()."""

    def test_remove_single_term(self):
        tokens = tokenize_query("transformer")
        result = reconstruct_query(tokens, 0)
        assert result == ""

    def test_remove_first_of_two(self):
        tokens = tokenize_query("cat:cs.AI transformer")
        result = reconstruct_query(tokens, 0)
        assert result == "transformer"

    def test_remove_last_of_two(self):
        tokens = tokenize_query("cat:cs.AI transformer")
        result = reconstruct_query(tokens, 1)
        assert result == "cat:cs.AI"

    def test_remove_middle_with_and(self):
        tokens = tokenize_query("cat:cs.AI AND starred AND transformer")
        # tokens: [cat:cs.AI, AND, starred, AND, transformer]
        result = reconstruct_query(tokens, 2)  # remove "starred"
        assert result == "cat:cs.AI AND transformer"

    def test_remove_first_with_and(self):
        tokens = tokenize_query("cat:cs.AI AND transformer")
        result = reconstruct_query(tokens, 0)  # remove "cat:cs.AI"
        assert result == "transformer"

    def test_remove_last_with_or(self):
        tokens = tokenize_query("cat:cs.AI OR transformer")
        result = reconstruct_query(tokens, 2)  # remove "transformer"
        assert result == "cat:cs.AI"

    def test_remove_quoted_phrase(self):
        tokens = tokenize_query('"neural network" transformer')
        result = reconstruct_query(tokens, 0)
        assert result == "transformer"

    def test_remove_field_value(self):
        tokens = tokenize_query("cat:cs.AI author:Smith")
        result = reconstruct_query(tokens, 0)
        assert result == "author:Smith"

    def test_invalid_index_returns_full_query(self):
        tokens = tokenize_query("transformer")
        result = reconstruct_query(tokens, 99)
        assert result == "transformer"

    def test_negative_index_returns_full_query(self):
        tokens = tokenize_query("transformer")
        result = reconstruct_query(tokens, -1)
        assert result == "transformer"

    def test_empty_tokens(self):
        result = reconstruct_query([], 0)
        assert result == ""

    def test_preserve_field_phrase(self):
        tokens = tokenize_query('author:"John Smith" cat:cs.AI')
        result = reconstruct_query(tokens, 1)
        assert result == 'author:"John Smith"'


class TestWcagContrastCompliance:
    """Verify WCAG contrast ratios for all theme colors against their backgrounds."""

    TEXT_COLOR_KEYS = [
        "text",
        "accent",
        "accent_alt",
        "green",
        "yellow",
        "orange",
        "pink",
        "purple",
        "muted",
    ]

    @staticmethod
    def _relative_luminance(hex_color: str) -> float:
        """Compute WCAG relative luminance from a hex color string."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

        def linearize(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

    @classmethod
    def _contrast_ratio(cls, hex1: str, hex2: str) -> float:
        """Compute WCAG contrast ratio between two hex colors."""
        l1 = cls._relative_luminance(hex1)
        l2 = cls._relative_luminance(hex2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    @pytest.mark.parametrize("theme_name", THEME_NAMES)
    def test_text_colors_meet_aa(self, theme_name: str) -> None:
        """All text colors in every theme must meet WCAG AA (4.5:1)."""
        theme = THEMES[theme_name]
        bg = theme["background"]
        failures = []
        for key in self.TEXT_COLOR_KEYS:
            if key not in theme:
                continue
            ratio = self._contrast_ratio(theme[key], bg)
            if ratio < 4.5:
                failures.append(f"{key}={theme[key]} ratio={ratio:.2f}")
        assert not failures, f"WCAG AA failures in {theme_name}: {', '.join(failures)}"

    def test_high_contrast_meets_aaa(self) -> None:
        """High Contrast theme must meet WCAG AAA (7.0:1) for all text colors."""
        theme = THEMES["high-contrast"]
        bg = theme["background"]
        failures = []
        for key in self.TEXT_COLOR_KEYS:
            if key not in theme:
                continue
            ratio = self._contrast_ratio(theme[key], bg)
            if ratio < 7.0:
                failures.append(f"{key}={theme[key]} ratio={ratio:.2f}")
        assert not failures, f"WCAG AAA failures in high-contrast: {', '.join(failures)}"


class TestAtomicWriteBaseException:
    """Fix 1: BaseException → Exception in save_config atomic write."""

    def test_keyboard_interrupt_propagates_from_save(self, tmp_path, monkeypatch):
        """KeyboardInterrupt during os.replace should not be caught."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with (
            patch("os.replace", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            save_config(UserConfig())


class TestCountPapersLogging:
    """Fix 2: count_papers_in_file logs warning on OSError."""

    def test_logs_warning_on_read_error(self, tmp_path, caplog):
        """OSError during file read should log a warning and return 0."""
        import logging as _logging
        from unittest.mock import patch

        from arxiv_browser.parsing import count_papers_in_file

        path = tmp_path / "test.txt"
        with (
            patch.object(type(path), "read_text", side_effect=OSError("permission denied")),
            caplog.at_level(_logging.WARNING, logger="arxiv_browser.parsing"),
        ):
            result = count_papers_in_file(path)
        assert result == 0
        assert "permission denied" in caplog.text


class TestInitDbOsError:
    """Fix 3: init_*_db guards against OSError from mkdir."""

    def test_init_summary_db_oserror(self, tmp_path):
        """_init_summary_db should convert mkdir OSError to sqlite3.OperationalError."""
        import sqlite3
        from unittest.mock import patch

        from arxiv_browser.llm import _init_summary_db

        db_path = tmp_path / "sub" / "db.sqlite"
        with (
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
            pytest.raises(sqlite3.OperationalError, match="Cannot create DB directory"),
        ):
            _init_summary_db(db_path)

    def test_init_relevance_db_oserror(self, tmp_path):
        """_init_relevance_db should convert mkdir OSError to sqlite3.OperationalError."""
        import sqlite3
        from unittest.mock import patch

        from arxiv_browser.llm import _init_relevance_db

        db_path = tmp_path / "sub" / "db.sqlite"
        with (
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
            pytest.raises(sqlite3.OperationalError, match="Cannot create DB directory"),
        ):
            _init_relevance_db(db_path)


class TestCorruptConfigBackup:
    """Fix 4: load_config backs up corrupt config and sets config_defaulted."""

    def test_corrupt_json_creates_backup(self, tmp_path, monkeypatch):
        """Corrupt JSON config should be backed up to .corrupt."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json!!!")
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        result = load_config()
        assert result.config_defaulted is True
        assert (tmp_path / "config.json.corrupt").exists()
        assert not config_file.exists()

    def test_invalid_structure_creates_backup(self, tmp_path, monkeypatch):
        """Config with invalid structure should be backed up to .corrupt."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text('{"valid": "json"}')
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with patch(
            "arxiv_browser.config._dict_to_config",
            side_effect=KeyError("missing_field"),
        ):
            result = load_config()
        assert result.config_defaulted is True
        assert (tmp_path / "config.json.corrupt").exists()

    def test_oserror_does_not_create_backup(self, tmp_path, monkeypatch):
        """OSError (temp inaccessibility) should NOT create a backup."""
        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        config_file.write_text('{"valid": "json"}')
        monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

        with patch.object(type(config_file), "read_text", side_effect=OSError("temporary")):
            result = load_config()
        assert result.config_defaulted is False
        assert not (tmp_path / "config.json.corrupt").exists()

    def test_config_defaulted_default_is_false(self):
        """config_defaulted should default to False."""
        config = UserConfig()
        assert config.config_defaulted is False


class TestCLIVersionAndSubcommands:
    """Tests for --version, config-path, and doctor CLI subcommands."""

    def test_version_flag_exits_zero(self, monkeypatch, capsys):
        """``--version`` should print version info and exit 0."""
        from tests.support.canonical_exports import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "--version"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "arxiv-viewer" in captured.out

    def test_short_version_flag_exits_zero(self, monkeypatch, capsys):
        """``-V`` should behave the same as ``--version``."""
        from tests.support.canonical_exports import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "-V"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "arxiv-viewer" in captured.out

    def test_get_version_returns_string(self):
        """``_get_version`` should return a non-empty string."""
        from arxiv_browser.cli import _get_version

        version = _get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_version_fallback(self, monkeypatch):
        """``_get_version`` should return 'dev' when package is not installed."""
        import importlib.metadata

        from arxiv_browser.cli import _get_version

        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda _name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
        )
        assert _get_version() == "dev"

    def test_config_path_subcommand(self, monkeypatch, capsys):
        """``config-path`` should print the config path and exit 0."""
        from tests.support.canonical_exports import main

        monkeypatch.setattr("sys.argv", ["arxiv_browser", "config-path"])

        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "config.json" in captured.out

    def test_print_config_path_function(self, capsys):
        """``_print_config_path`` should print a path containing config.json."""
        from arxiv_browser.cli import _print_config_path

        result = _print_config_path()
        assert result == 0
        captured = capsys.readouterr()
        assert "config.json" in captured.out

    def test_doctor_subcommand_no_history(self, tmp_path, monkeypatch, capsys):
        """``doctor`` should run diagnostics and return 0."""
        from arxiv_browser.cli import CliDependencies
        from arxiv_browser.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["arxiv_browser", "doctor"])
        result = cli_main(
            deps=CliDependencies(
                load_config_fn=lambda: UserConfig(),
                discover_history_files_fn=lambda _d: [],
                resolve_papers_fn=lambda *_args, **_kwargs: ([], [], 0),
                configure_logging_fn=lambda _debug: None,
                configure_color_mode_fn=lambda _mode: None,
                validate_interactive_tty_fn=lambda: True,
            )
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "arxiv-viewer" in captured.out
        assert "Python" in captured.out

    def test_run_doctor_reports_config_exists(self, tmp_path, monkeypatch, capsys):
        """``_run_doctor`` should report when config exists."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig()

        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        from unittest.mock import patch

        monkeypatch.chdir(tmp_path)

        with patch("arxiv_browser.config.get_config_path", return_value=config_file):
            result = _run_doctor(config, [])

        assert result == 0
        captured = capsys.readouterr()
        assert "ok" in captured.out.lower() or "Config file" in captured.out

    def test_run_doctor_warns_empty_history_dir(self, tmp_path, monkeypatch, capsys):
        """``_run_doctor`` should warn when history dir exists but has no files."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig()
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        from unittest.mock import patch

        config_file = tmp_path / "config.json"
        with patch("arxiv_browser.config.get_config_path", return_value=config_file):
            result = _run_doctor(config, [])

        assert result == 1
        captured = capsys.readouterr()
        assert "WARN" in captured.out

    def test_run_doctor_reports_llm_preset(self, tmp_path, capsys, monkeypatch):
        """``_run_doctor`` should check LLM preset binary on PATH."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig()
        config.llm_preset = "nonexistent_preset_xyz"

        config_file = tmp_path / "config.json"
        monkeypatch.chdir(tmp_path)

        from unittest.mock import patch

        with patch("arxiv_browser.config.get_config_path", return_value=config_file):
            result = _run_doctor(config, [])

        assert result == 1
        captured = capsys.readouterr()
        assert "WARN" in captured.out or "LLM" in captured.out

    def test_run_doctor_reports_tty_status(self, tmp_path, capsys, monkeypatch):
        """``_run_doctor`` should report TTY status."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig()
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "config.json"

        from unittest.mock import patch

        with patch("arxiv_browser.config.get_config_path", return_value=config_file):
            result = _run_doctor(config, [])

        assert result == 0
        captured = capsys.readouterr()
        assert "TTY" in captured.out or "tty" in captured.out.lower()

    def test_run_doctor_parses_quoted_llm_command_path(self, tmp_path, capsys, monkeypatch):
        """``_run_doctor`` should parse quoted LLM command paths correctly."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig(llm_command='"/Applications/My LLM.app/llm" {prompt}')
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "config.json"

        from unittest.mock import patch

        with (
            patch("arxiv_browser.config.get_config_path", return_value=config_file),
            patch("arxiv_browser.cli.shutil.which", return_value="/Applications/My LLM.app/llm"),
        ):
            result = _run_doctor(config, [])

        assert result == 0
        captured = capsys.readouterr()
        assert "/Applications/My LLM.app/llm found on PATH" in captured.out

    def test_run_doctor_prefers_custom_llm_command_over_preset(self, tmp_path, capsys, monkeypatch):
        """``_run_doctor`` should diagnose the runtime-effective custom LLM command."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig(llm_command="custom-tool {prompt}", llm_preset="claude")
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "config.json"

        from unittest.mock import patch

        def _which(binary: str) -> str | None:
            if binary == "custom-tool":
                return "/usr/local/bin/custom-tool"
            return None

        with (
            patch("arxiv_browser.config.get_config_path", return_value=config_file),
            patch("arxiv_browser.cli.shutil.which", side_effect=_which),
        ):
            result = _run_doctor(config, [])

        assert result == 0
        captured = capsys.readouterr()
        assert "LLM command: custom-tool found on PATH" in captured.out

    def test_run_doctor_warns_when_llm_command_missing_prompt_placeholder(
        self, tmp_path, capsys, monkeypatch
    ):
        """``_run_doctor`` should reject LLM commands without ``{prompt}``."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig(llm_command="custom-tool")
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "config.json"

        from unittest.mock import patch

        with (
            patch("arxiv_browser.config.get_config_path", return_value=config_file),
            patch("arxiv_browser.cli.shutil.which", return_value="/usr/local/bin/custom-tool"),
        ):
            result = _run_doctor(config, [])

        assert result == 1
        captured = capsys.readouterr()
        assert "missing required {prompt} placeholder" in captured.out

    def test_run_doctor_warns_when_shell_fallback_disabled_for_shell_only_command(
        self, tmp_path, capsys, monkeypatch
    ):
        """``_run_doctor`` should flag shell-only LLM commands blocked by policy."""
        from arxiv_browser.cli import _run_doctor

        config = UserConfig(
            llm_command="echo {prompt} | cat",
            allow_llm_shell_fallback=False,
        )
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "config.json"

        from unittest.mock import patch

        with patch("arxiv_browser.config.get_config_path", return_value=config_file):
            result = _run_doctor(config, [])

        assert result == 1
        captured = capsys.readouterr()
        assert "allow_llm_shell_fallback is disabled" in captured.out
