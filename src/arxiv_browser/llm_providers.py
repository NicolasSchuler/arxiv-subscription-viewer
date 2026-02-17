"""LLM provider abstraction — Protocol + CLI subprocess implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from arxiv_browser.llm import _build_llm_shell_command, _resolve_llm_command
from arxiv_browser.models import UserConfig

logger = logging.getLogger(__name__)


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
            shell_command = _build_llm_shell_command(self._command_template, prompt)
        except ValueError as e:
            return LLMResult(output="", success=False, error=str(e))

        try:
            proc = await asyncio.create_subprocess_shell(
                shell_command,
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
