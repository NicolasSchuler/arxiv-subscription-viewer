"""Tests for the keybindings CLI subcommand."""

from __future__ import annotations

import json


class TestKeybindingsCommand:
    """Tests for the keybindings CLI subcommand."""

    def test_keybindings_table_format(self, capsys: object) -> None:
        """Default table format prints categorized bindings."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        output = captured.out
        assert "Getting Started" in output
        assert "Core Actions" in output

    def test_keybindings_table_honors_no_color(self, capsys: object) -> None:
        """--no-color suppresses ANSI escapes for table output."""
        from arxiv_browser.cli import main

        exit_code = main(["--no-color", "keybindings"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "\033[" not in captured.out

    def test_keybindings_table_honors_color_never(self, capsys: object) -> None:
        """--color never suppresses ANSI escapes for table output."""
        from arxiv_browser.cli import main

        exit_code = main(["--color", "never", "keybindings"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "\033[" not in captured.out

    def test_keybindings_table_color_always_overrides_no_color_env(
        self, capsys: object, monkeypatch
    ) -> None:
        """--color always keeps ANSI table output even when NO_COLOR is set."""
        from arxiv_browser.cli import main

        monkeypatch.setenv("NO_COLOR", "1")
        exit_code = main(["--color", "always", "keybindings"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "\033[" in captured.out

    def test_keybindings_json_format(self, capsys: object) -> None:
        """JSON format produces valid JSON with sections."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings", "--format", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert any(s["section"] == "Getting Started" for s in data)
        # Each section has bindings
        for section in data:
            assert "section" in section
            assert "bindings" in section
            for binding in section["bindings"]:
                assert "key" in binding
                assert "description" in binding

    def test_keybindings_markdown_format(self, capsys: object) -> None:
        """Markdown format produces table syntax."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings", "--format", "markdown"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        output = captured.out
        assert "## Getting Started" in output
        assert "| Key | Action |" in output
        assert "|-----|--------|" in output

    def test_keybindings_markdown_escapes_table_cells(self, capsys: object) -> None:
        """Markdown table cells escape pipes in keys and descriptions."""
        from arxiv_browser.cli_keybindings import _render_markdown

        _render_markdown([("Example", [("a|b", "Alpha | beta")])])
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "| `a\\|b` | Alpha \\| beta |" in captured.out

    def test_keybindings_tier_filter_core(self, capsys: object) -> None:
        """Core tier restricts output to core sections only."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings", "--tier", "core"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        output = captured.out
        assert "Core Actions" in output
        assert "Power" not in output

    def test_keybindings_tier_filter_power(self, capsys: object) -> None:
        """Power tier shows only advanced bindings."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings", "--tier", "power"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        output = captured.out
        assert "Power" in output
        assert "Getting Started" not in output

    def test_keybindings_tier_filter_standard(self, capsys: object) -> None:
        """Standard tier shows organize section."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings", "--tier", "standard"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        output = captured.out
        assert "Organize" in output
        assert "Getting Started" not in output

    def test_keybindings_tier_all_shows_everything(self, capsys: object) -> None:
        """All tier shows every section."""
        from arxiv_browser.cli import main

        exit_code = main(["keybindings", "--tier", "all"])
        assert exit_code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        output = captured.out
        assert "Getting Started" in output
        assert "Core Actions" in output

    def test_keybindings_normalize_argv(self) -> None:
        """keybindings command is not rewritten by _normalize_cli_argv."""
        from arxiv_browser.cli import _normalize_cli_argv

        assert _normalize_cli_argv(["keybindings"]) == ["keybindings"]
        assert _normalize_cli_argv(["keybindings", "--format", "json"]) == [
            "keybindings",
            "--format",
            "json",
        ]
