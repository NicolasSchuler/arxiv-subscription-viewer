"""LLM provider abstraction — Protocol + CLI subprocess implementation."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from arxiv_browser.llm import _build_llm_shell_command, _resolve_llm_command
from arxiv_browser.models import UserConfig

logger = logging.getLogger(__name__)
_SHELL_META_CHARS = frozenset({"|", "&", ";", "<", ">", "$", "`", "\n", "(", ")"})


@dataclass(slots=True)
class _InvocationPlan:
    use_shell: bool
    argv: list[str] | None = None
    shell_command: str = ""


def _requires_shell_execution(command_template: str) -> bool:
    """Return True when the command template needs shell parsing semantics."""
    return any(char in command_template for char in _SHELL_META_CHARS)


def _strip_wrapping_quotes_windows(args: list[str]) -> list[str]:
    """Normalize shlex(posix=False) output for Windows command arguments."""
    if os.name != "nt":
        return args
    return [
        arg[1:-1] if len(arg) >= 2 and arg.startswith('"') and arg.endswith('"') else arg
        for arg in args
    ]


def _build_invocation_plan(command_template: str, prompt: str) -> _InvocationPlan:
    """Build argv-first invocation plan with controlled shell fallback."""
    if "{prompt}" not in command_template:
        raise ValueError(
            f"LLM command template must contain {{prompt}} placeholder, got: {command_template!r}"
        )

    if _requires_shell_execution(command_template):
        shell_command = _build_llm_shell_command(command_template, prompt)
        return _InvocationPlan(use_shell=True, shell_command=shell_command)

    sentinel = "__ARXIV_BROWSER_PROMPT__"
    templated = command_template.replace("{prompt}", sentinel)
    try:
        argv = shlex.split(templated, posix=os.name != "nt")
    except ValueError:
        shell_command = _build_llm_shell_command(command_template, prompt)
        return _InvocationPlan(use_shell=True, shell_command=shell_command)

    argv = _strip_wrapping_quotes_windows(argv)
    if not argv:
        raise ValueError("LLM command template is empty")

    return _InvocationPlan(
        use_shell=False,
        argv=[arg.replace(sentinel, prompt) for arg in argv],
    )


@dataclass(slots=True)
class LLMResult:
    """Result from an LLM provider call."""

    output: str
    success: bool
    error: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers — any class with this signature is compatible."""

    async def execute(self, prompt: str, timeout: int) -> LLMResult: ...


class CLIProvider:
    """LLM provider that shells out to a CLI tool.

    Wraps the existing subprocess pattern: build shell command from template,
    run with asyncio.create_subprocess_shell, capture stdout/stderr.
    Never raises — returns LLMResult with success=False on any error.
    """

    __slots__ = ("_command_template",)

    def __init__(self, command_template: str) -> None:
        self._command_template = command_template

    @property
    def command_template(self) -> str:
        """The shell command template (contains {prompt} placeholder)."""
        return self._command_template

    async def execute(self, prompt: str, timeout: int) -> LLMResult:
        """Execute the LLM command and return the result."""
        try:
            plan = _build_invocation_plan(self._command_template, prompt)
        except ValueError as e:
            return LLMResult(output="", success=False, error=str(e))

        try:
            if plan.use_shell:
                proc = await asyncio.create_subprocess_shell(
                    plan.shell_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                argv = plan.argv or []
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return LLMResult(output="", success=False, error=f"Timed out after {timeout}s")

            if proc.returncode != 0:
                err_msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                return LLMResult(
                    output="",
                    success=False,
                    error=f"Exit {proc.returncode}: {err_msg[:200]}",
                )

            output = (stdout or b"").decode("utf-8", errors="replace").strip()
            if not output:
                return LLMResult(output="", success=False, error="Empty output")

            return LLMResult(output=output, success=True)

        except Exception as e:
            logger.warning("LLM subprocess failed: %s", e, exc_info=True)
            return LLMResult(output="", success=False, error=str(e))


def resolve_provider(config: UserConfig) -> CLIProvider | None:
    """Create an LLM provider from user config, or None if not configured."""
    template = _resolve_llm_command(config)
    if not template:
        return None
    return CLIProvider(template)


__all__ = [
    "CLIProvider",
    "LLMProvider",
    "LLMResult",
    "resolve_provider",
]
