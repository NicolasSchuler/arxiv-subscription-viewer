"""Tests for shell completion script generation."""

from __future__ import annotations

import pytest

from arxiv_browser.completions import SUPPORTED_SHELLS, get_completion_script


class TestGetCompletionScript:
    """Unit tests for get_completion_script()."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_returns_script_containing_arxiv_viewer(self, shell: str) -> None:
        script = get_completion_script(shell)
        assert "arxiv-viewer" in script

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_script_contains_subcommands(self, shell: str) -> None:
        script = get_completion_script(shell)
        for cmd in ("browse", "search", "dates", "completions"):
            assert cmd in script

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_script_contains_search_flags(self, shell: str) -> None:
        script = get_completion_script(shell)
        # Fish uses `-l query` instead of `--query`
        for flag in ("query", "field", "category"):
            assert flag in script

    def test_unsupported_shell_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unsupported shell.*powershell"):
            get_completion_script("powershell")

    def test_supported_shells_constant(self) -> None:
        assert SUPPORTED_SHELLS == ("bash", "zsh", "fish")

    def test_bash_has_complete_command(self) -> None:
        script = get_completion_script("bash")
        assert "complete -F _arxiv_viewer arxiv-viewer" in script

    def test_zsh_has_compdef(self) -> None:
        script = get_completion_script("zsh")
        assert "#compdef arxiv-viewer" in script

    def test_fish_has_complete_command(self) -> None:
        script = get_completion_script("fish")
        assert "complete -c arxiv-viewer" in script


class TestCompletionsCLISubcommand:
    """Integration tests for the completions CLI subcommand."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completions_subcommand_prints_script(
        self, shell: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from arxiv_browser.cli import main

        exit_code = main(["completions", shell])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "arxiv-viewer" in captured.out

    def test_completions_invalid_shell_exits_nonzero(self) -> None:
        from arxiv_browser.cli import main

        with pytest.raises(SystemExit):
            main(["completions", "powershell"])

    def test_completions_no_shell_arg_exits_nonzero(self) -> None:
        from arxiv_browser.cli import main

        with pytest.raises(SystemExit):
            main(["completions"])
