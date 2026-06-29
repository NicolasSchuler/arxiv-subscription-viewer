"""Dispatch CLI subcommands that do not need config or history state."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from arxiv_browser.cache_cli import CACHE_COMMANDS, run_cache_command


def handle_config_free_command(
    args: Namespace,
    *,
    print_config_path: Callable[[], int],
    run_keybindings: Callable[[Namespace], int],
) -> int | None:
    """Run subcommands that do not need config, history, or TTY state."""
    command = getattr(args, "command", None)
    if command == "completions":
        from arxiv_browser.completions import get_completion_script

        print(get_completion_script(args.shell))
        return 0
    if command == "config-path":
        return print_config_path()
    if command == "keybindings":
        return run_keybindings(args)
    if command in CACHE_COMMANDS:
        return run_cache_command(args)
    return None


__all__ = ["handle_config_free_command"]
