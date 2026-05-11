"""LLM provider abstraction — Protocol, registry, CLI + HTTP implementations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from arxiv_browser.llm import _build_llm_shell_command, _resolve_llm_command
from arxiv_browser.models import UserConfig

logger = logging.getLogger(__name__)
_SHELL_META_CHARS = frozenset({"|", "&", ";", "<", ">", "$", "`", "\n", "(", ")"})
_POSIX_ENV_ASSIGNMENT_PREFIX = "="


@dataclass(slots=True)
class _InvocationPlan:
    """Internal plan for executing an LLM command.

    Prefer direct argv execution (use_shell=False) when the command template
    contains no shell metacharacters.  Fall back to a full shell string
    (use_shell=True) when piping, environment-variable expansion, or other
    shell semantics are required.
    """

    use_shell: bool
    """True → run via asyncio.create_subprocess_shell(shell_command)."""
    argv: list[str] | None = None
    """Argument vector for direct exec (populated when use_shell=False)."""
    shell_command: str = ""
    """Fully-quoted shell string (populated when use_shell=True)."""


def llm_command_requires_shell(command_template: str) -> bool:
    """Return True when the command template needs shell parsing semantics.

    Args:
        command_template: Raw LLM command string, potentially containing shell
            metacharacters such as ``|``, ``&``, ``$``, or backticks.

    Returns:
        True if the template contains shell metacharacters or starts with one
        or more POSIX-style environment assignments (for example,
        ``OPENAI_API_KEY=... llm {prompt}``), indicating the command must be
        launched via a shell interpreter rather than direct ``execvp``-style
        execution.
    """
    if any(char in command_template for char in _SHELL_META_CHARS):
        return True
    try:
        argv = shlex.split(command_template, posix=os.name != "nt")
    except ValueError:
        return True
    if os.name == "nt" or len(argv) < 2:
        return False
    first = argv[0]
    return (
        _POSIX_ENV_ASSIGNMENT_PREFIX in first
        and not first.startswith(_POSIX_ENV_ASSIGNMENT_PREFIX)
        and first.split(_POSIX_ENV_ASSIGNMENT_PREFIX, 1)[0].isidentifier()
    )


def _strip_wrapping_quotes_windows(args: list[str]) -> list[str]:
    """Normalize shlex(posix=False) output for Windows command arguments.

    On Windows, ``shlex.split(posix=False)`` preserves the surrounding double
    quotes that the shell would strip at runtime.  This helper removes those
    outer quotes so the argument list can be passed directly to
    ``asyncio.create_subprocess_exec``.  On non-Windows platforms this is a
    no-op.

    Args:
        args: Argument list produced by ``shlex.split(posix=False)``.

    Returns:
        The same list with outer double-quote pairs removed from each element
        (Windows only); unchanged on other platforms.
    """
    if os.name != "nt":
        return args
    return [
        arg[1:-1] if len(arg) >= 2 and arg.startswith('"') and arg.endswith('"') else arg
        for arg in args
    ]


def _build_invocation_plan(
    command_template: str, prompt: str, *, allow_shell: bool = True
) -> _InvocationPlan:
    """Build argv-first invocation plan with controlled shell fallback.

    Tries to parse the command template into a safe argv list using
    ``shlex.split`` (preferred — avoids shell injection).  Falls back to a
    shell string only when the template contains shell metacharacters or when
    ``shlex.split`` itself raises ``ValueError``.

    Args:
        command_template: LLM command string containing a ``{prompt}``
            placeholder.  May include shell metacharacters.
        prompt: The resolved prompt text to substitute for ``{prompt}``.
        allow_shell: When False, raise ``ValueError`` instead of falling back
            to shell execution.  Defaults to True.

    Returns:
        An ``_InvocationPlan`` with ``use_shell=False`` and a populated
        ``argv`` list when direct exec is safe, or ``use_shell=True`` and a
        populated ``shell_command`` string when shell fallback is required.

    Raises:
        ValueError: If the template has no ``{prompt}`` placeholder, if the
            template is empty after splitting, or if shell execution is
            required but ``allow_shell=False``.
    """
    if "{prompt}" not in command_template:
        raise ValueError(
            f"LLM command template must contain {{prompt}} placeholder, got: {command_template!r}"
        )

    if llm_command_requires_shell(command_template):
        if not allow_shell:
            raise ValueError(
                "LLM command requires shell execution, but allow_llm_shell_fallback is disabled"
            )
        shell_command = _build_llm_shell_command(command_template, prompt)
        return _InvocationPlan(use_shell=True, shell_command=shell_command)

    sentinel = "__ARXIV_BROWSER_PROMPT__"
    templated = command_template.replace("{prompt}", sentinel)
    try:
        argv = shlex.split(templated, posix=os.name != "nt")
    except ValueError:
        if not allow_shell:
            raise ValueError(
                "LLM command requires shell parsing, but allow_llm_shell_fallback is disabled"
            ) from None
        shell_command = _build_llm_shell_command(command_template, prompt)
        return _InvocationPlan(use_shell=True, shell_command=shell_command)

    argv = _strip_wrapping_quotes_windows(argv)
    if not argv:
        raise ValueError("LLM command template is empty")

    return _InvocationPlan(
        use_shell=False,
        argv=[arg.replace(sentinel, prompt) for arg in argv],
    )


def _shell_session_kwargs(use_shell: bool) -> dict[str, Any]:
    if use_shell and os.name != "nt":
        return {"start_new_session": True}
    return {}


def _terminate_process(proc: Any, *, use_shell: bool) -> None:
    pid = getattr(proc, "pid", None)
    if use_shell and os.name != "nt" and isinstance(pid, int):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGTERM)
            return
    proc.kill()


@dataclass(slots=True)
class LLMResult:
    """Result from an LLM provider call."""

    output: str
    success: bool
    error: str = ""


@dataclass(slots=True)
class LLMChunk:
    """Incremental output from a streaming LLM provider call."""

    delta: str = ""
    error: str = ""
    done: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers — any class with this signature is compatible."""

    async def execute(self, prompt: str, timeout: int) -> LLMResult:
        """Execute a prompt and return the LLM result."""
        ...

    def execute_stream(self, prompt: str, timeout: int) -> AsyncIterator[LLMChunk]:
        """Execute a prompt and yield incremental output chunks."""
        ...


class CLIProvider:
    """LLM provider that shells out to a CLI tool.

    Wraps the existing subprocess pattern: build shell command from template,
    run with asyncio.create_subprocess_shell, capture stdout/stderr.
    Never raises — returns LLMResult with success=False on any error.
    Supports configurable retry with exponential backoff for transient failures.
    """

    __slots__ = ("_allow_shell", "_command_template", "_max_retries")

    _RETRYABLE_PREFIXES = ("Timed out", "Exit ")

    def __init__(
        self,
        command_template: str,
        *,
        allow_shell: bool = True,
        max_retries: int = 0,
    ) -> None:
        """Initialize the CLI provider with a command template and shell policy."""
        self._command_template = command_template
        self._allow_shell = allow_shell
        self._max_retries = max(0, max_retries)

    @property
    def command_template(self) -> str:
        """The shell command template (contains {prompt} placeholder)."""
        return self._command_template

    async def execute(self, prompt: str, timeout: int) -> LLMResult:
        """Execute the LLM command with optional retry for transient failures."""
        last_result: LLMResult | None = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                delay = min(2**attempt, 8)
                logger.info("LLM retry %d/%d after %ds", attempt, self._max_retries, delay)
                await asyncio.sleep(delay)
            result = await self._execute_once(prompt, timeout)
            if result.success:
                return result
            last_result = result
            if not any(result.error.startswith(p) for p in self._RETRYABLE_PREFIXES):
                return result  # Not a transient error, don't retry
        return last_result or LLMResult(output="", success=False, error="No attempts made")

    async def _execute_once(self, prompt: str, timeout: int) -> LLMResult:
        """Execute a single LLM command attempt without retry logic.

        Builds the invocation plan, launches the subprocess, and captures
        stdout/stderr.  Never raises; all error conditions are encoded in the
        returned ``LLMResult``.

        Args:
            prompt: The fully resolved prompt string to pass to the LLM CLI.
            timeout: Maximum seconds to wait for the subprocess to complete
                before killing it and returning a timeout error.

        Returns:
            ``LLMResult(success=True, output=...)`` on success, or
            ``LLMResult(success=False, error=...)`` for any failure (invalid
            template, non-zero exit code, timeout, empty output, or unexpected
            exception).
        """
        try:
            plan = _build_invocation_plan(
                self._command_template,
                prompt,
                allow_shell=self._allow_shell,
            )
        except ValueError as e:
            return LLMResult(output="", success=False, error=str(e))

        try:
            if plan.use_shell:
                proc = await asyncio.create_subprocess_shell(
                    plan.shell_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **_shell_session_kwargs(plan.use_shell),
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
                _terminate_process(proc, use_shell=plan.use_shell)
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

    async def execute_stream(self, prompt: str, timeout: int) -> AsyncIterator[LLMChunk]:
        """Execute the LLM command and yield stdout chunks as they arrive."""
        try:
            plan = _build_invocation_plan(
                self._command_template,
                prompt,
                allow_shell=self._allow_shell,
            )
        except ValueError as e:
            yield LLMChunk(error=str(e), done=True)
            return

        try:
            if plan.use_shell:
                proc = await asyncio.create_subprocess_shell(
                    plan.shell_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **_shell_session_kwargs(plan.use_shell),
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *(plan.argv or []),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        except Exception as e:
            logger.warning("LLM streaming subprocess failed to start: %s", e, exc_info=True)
            yield LLMChunk(error=str(e), done=True)
            return

        emitted = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            stdout = proc.stdout
            if stdout is not None:
                while True:
                    remaining = max(0.01, deadline - loop.time())
                    try:
                        chunk = await asyncio.wait_for(stdout.read(1024), timeout=remaining)
                    except TimeoutError:
                        _terminate_process(proc, use_shell=plan.use_shell)
                        await proc.wait()
                        yield LLMChunk(error=f"Timed out after {timeout}s", done=True)
                        return
                    if not chunk:
                        break
                    emitted = True
                    yield LLMChunk(delta=chunk.decode("utf-8", errors="replace"))

            remaining = max(0.01, deadline - loop.time())
            try:
                await asyncio.wait_for(proc.wait(), timeout=remaining)
            except TimeoutError:
                _terminate_process(proc, use_shell=plan.use_shell)
                await proc.wait()
                yield LLMChunk(error=f"Timed out after {timeout}s", done=True)
                return

            stderr = b""
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    stderr = await proc.stderr.read()
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                yield LLMChunk(error=f"Exit {proc.returncode}: {err_msg[:200]}", done=True)
                return
            if not emitted:
                yield LLMChunk(error="Empty output", done=True)
                return
            yield LLMChunk(done=True)
        except Exception as e:
            logger.warning("LLM streaming subprocess failed: %s", e, exc_info=True)
            yield LLMChunk(error=str(e), done=True)


class HTTPProvider:
    """LLM provider using the OpenAI-compatible chat completions API.

    Works with OpenAI, Ollama, LM Studio, vLLM, and any other server
    exposing the ``/v1/chat/completions`` endpoint.  Uses ``httpx`` (already
    a runtime dependency) for async HTTP.
    """

    __slots__ = ("_api_key", "_base_url", "_max_retries", "_model")

    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        max_retries: int = 1,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_retries = max(0, max_retries)

    @property
    def base_url(self) -> str:
        """The API base URL (without trailing slash)."""
        return self._base_url

    @property
    def model(self) -> str:
        """The model name sent in the request body."""
        return self._model

    async def execute(self, prompt: str, timeout: int) -> LLMResult:
        """Send a chat completion request and return the LLM result."""
        try:
            import httpx
        except ModuleNotFoundError:
            return LLMResult(output="", success=False, error="httpx is not installed")

        url = f"{self._base_url}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }

        last_result: LLMResult | None = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                delay = min(2**attempt, 8)
                logger.info("HTTP retry %d/%d after %ds", attempt, self._max_retries, delay)
                await asyncio.sleep(delay)
            result = await self._request_once(httpx, url, headers, body, timeout)
            if result.success:
                return result
            last_result = result
            if not self._is_retryable(result.error):
                return result
        return last_result or LLMResult(output="", success=False, error="No attempts made")

    def _is_retryable(self, error: str) -> bool:
        """Determine whether an error message indicates a transient failure."""
        if error.startswith("Timed out"):
            return True
        return any(f"HTTP {code}" in error for code in self._RETRYABLE_STATUS_CODES)

    async def _request_once(
        self,
        httpx: object,
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout: int,
    ) -> LLMResult:
        """Execute a single HTTP request without retry logic."""
        httpx_mod: Any = httpx  # Module typed as Any for dynamic import
        try:
            async with httpx_mod.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=body)
        except httpx_mod.TimeoutException:
            return LLMResult(output="", success=False, error=f"Timed out after {timeout}s")
        except Exception as e:
            logger.warning("HTTP request failed: %s", e, exc_info=True)
            return LLMResult(output="", success=False, error=str(e))

        if response.status_code != 200:
            err_body = response.text[:200]
            return LLMResult(
                output="",
                success=False,
                error=f"HTTP {response.status_code}: {err_body}",
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            return LLMResult(output="", success=False, error=f"Invalid JSON: {e}")

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            return LLMResult(
                output="",
                success=False,
                error=f"Unexpected response structure: {e}",
            )

        output = content.strip() if isinstance(content, str) else str(content).strip()
        if not output:
            return LLMResult(output="", success=False, error="Empty response content")
        return LLMResult(output=output, success=True)

    async def execute_stream(self, prompt: str, timeout: int) -> AsyncIterator[LLMChunk]:
        """Send a streaming chat completion request and yield content deltas."""
        try:
            import httpx
        except ModuleNotFoundError:
            yield LLMChunk(error="httpx is not installed", done=True)
            return

        url = f"{self._base_url}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async for chunk in self._stream_request_once(httpx, url, headers, body, timeout):
            yield chunk

    async def _stream_request_once(
        self,
        httpx: object,
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout: int,
    ) -> AsyncIterator[LLMChunk]:
        """Execute a single streaming HTTP request."""
        httpx_mod: Any = httpx
        emitted = False
        try:
            async with (
                httpx_mod.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, headers=headers, json=body) as response,
            ):
                if response.status_code != 200:
                    err_body = (await response.aread()).decode("utf-8", errors="replace")[:200]
                    yield LLMChunk(
                        error=f"HTTP {response.status_code}: {err_body}",
                        done=True,
                    )
                    return
                async for line in response.aiter_lines():
                    parsed = _parse_sse_line(line)
                    if parsed is None:
                        continue
                    if parsed == "[DONE]":
                        yield LLMChunk(done=True)
                        return
                    try:
                        data = json.loads(parsed)
                    except (json.JSONDecodeError, ValueError) as e:
                        yield LLMChunk(error=f"Invalid stream JSON: {e}", done=True)
                        return
                    delta = _extract_stream_delta(data)
                    if delta:
                        emitted = True
                        yield LLMChunk(delta=delta)
                if not emitted:
                    yield LLMChunk(error="Empty response content", done=True)
                    return
                yield LLMChunk(done=True)
        except httpx_mod.TimeoutException:
            yield LLMChunk(error=f"Timed out after {timeout}s", done=True)
        except Exception as e:
            logger.warning("HTTP streaming request failed: %s", e, exc_info=True)
            yield LLMChunk(error=str(e), done=True)


def _parse_sse_line(line: str) -> str | None:
    """Return SSE data payload for a chat-completions stream line."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    return stripped.removeprefix("data:").strip()


def _extract_stream_delta(data: object) -> str:
    """Extract an OpenAI-compatible content delta from a stream payload."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    return ""


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[CLIProvider] | type[HTTPProvider]] = {}


def register_provider(name: str, cls: type[CLIProvider] | type[HTTPProvider]) -> None:
    """Register a provider class under *name* (case-insensitive)."""
    _PROVIDER_REGISTRY[name.lower()] = cls


def get_provider_class(name: str) -> type[CLIProvider] | type[HTTPProvider] | None:
    """Look up a registered provider class by name (case-insensitive)."""
    return _PROVIDER_REGISTRY.get(name.lower())


register_provider("cli", CLIProvider)
register_provider("http", HTTPProvider)


def resolve_provider(config: UserConfig) -> LLMProvider | None:
    """Create an LLM provider from user config, or None if not configured.

    When ``config.llm_provider_type`` is ``"http"``, returns an
    ``HTTPProvider`` (requires ``llm_api_base_url``).  Otherwise falls back
    to the CLI subprocess path (requires ``llm_command`` or ``llm_preset``).
    """
    provider_type = config.llm_provider_type.lower()

    if provider_type == "http":
        if not config.llm_api_base_url:
            return None
        return HTTPProvider(
            base_url=config.llm_api_base_url,
            api_key=config.llm_api_key,
            model=config.llm_api_model,
            max_retries=config.llm_max_retries,
        )

    # Default: CLI provider path
    template = _resolve_llm_command(config)
    if not template:
        return None
    return CLIProvider(
        template,
        allow_shell=config.allow_llm_shell_fallback,
        max_retries=config.llm_max_retries,
    )


__all__ = [
    "CLIProvider",
    "HTTPProvider",
    "LLMChunk",
    "LLMProvider",
    "LLMResult",
    "_extract_stream_delta",
    "_parse_sse_line",
    "get_provider_class",
    "llm_command_requires_shell",
    "register_provider",
    "resolve_provider",
]
