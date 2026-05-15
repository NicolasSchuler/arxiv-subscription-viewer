"""System audio actions for paper triage."""

from __future__ import annotations

import asyncio
import platform
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from arxiv_browser.actions.constants import logger
from arxiv_browser.models import Paper

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser

SeverityLevel = Literal["information", "warning", "error"]


@dataclass(frozen=True, slots=True)
class TtsCommand:
    """Resolved system text-to-speech command."""

    executable: str
    args: tuple[str, ...]
    display_name: str


def _resolve_tts_command(
    system: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> TtsCommand | None:
    """Resolve the first supported system TTS command for the current platform."""
    resolved_system = system or platform.system()
    if resolved_system == "Darwin":
        executable = which("say")
        return TtsCommand(executable, (), "say") if executable else None
    if resolved_system == "Linux":
        for name in ("espeak-ng", "espeak"):
            executable = which(name)
            if executable:
                return TtsCommand(executable, ("--stdin",), name)
    return None


def _tts_missing_message(system: str | None = None) -> str:
    """Return an actionable missing-engine message for the current platform."""
    resolved_system = system or platform.system()
    if resolved_system == "Darwin":
        return "System TTS unavailable: macOS 'say' was not found."
    if resolved_system == "Linux":
        return "System TTS unavailable: install espeak-ng or espeak."
    return "System TTS is supported on macOS (say) and Linux (espeak-ng/espeak)."


def _normalize_speech_text(text: str) -> str:
    return " ".join(text.split())


def _build_abstract_speech_text(paper: Paper, abstract_text: str) -> str:
    """Build the spoken payload for audio-only paper triage."""
    title = _normalize_speech_text(paper.title)
    abstract = _normalize_speech_text(abstract_text)
    if title:
        return f"{title}. Abstract. {abstract}"
    return f"Abstract. {abstract}"


def _is_active_task(task: Any) -> bool:
    done = getattr(task, "done", None)
    return bool(task is not None and (not callable(done) or not done()))


def _terminate_process(process: Any) -> None:
    """Terminate an active TTS process, ignoring already-finished races."""
    if process is None or getattr(process, "returncode", None) is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.debug("Failed to terminate TTS process: %s", exc, exc_info=True)


def _stop_tts_playback(app: ArxivBrowser) -> None:
    """Stop any active TTS playback and clear app-owned TTS state."""
    task = getattr(app, "_tts_task", None)
    process = getattr(app, "_tts_process", None)
    _terminate_process(process)
    cancel = getattr(task, "cancel", None)
    if _is_active_task(task) and callable(cancel):
        cancel()
    app._tts_task = None
    app._tts_process = None
    app._tts_paper_id = None


def action_read_abstract_aloud(app: ArxivBrowser) -> None:
    """Read the current paper title and abstract using system TTS."""
    paper = app._get_current_paper()
    if paper is None:
        app.notify("No paper selected", title="Audio", severity="warning")
        return

    if _same_paper_is_playing(app, paper.arxiv_id):
        _stop_tts_playback(app)
        app.notify("Stopped abstract audio", title="Audio")
        return

    abstract_text = app._get_abstract_text(paper, allow_async=False) or ""
    if not abstract_text.strip():
        app.notify("No abstract available", title="Audio", severity="warning")
        return

    command = _resolve_tts_command()
    if command is None:
        app.notify(_tts_missing_message(), title="Audio", severity="warning", timeout=8)
        return

    _stop_tts_playback(app)
    speech_text = _build_abstract_speech_text(paper, abstract_text)
    mutable_app = cast(Any, app)
    mutable_app._tts_paper_id = paper.arxiv_id
    task = app._track_dataset_task(
        _read_abstract_aloud_async(app, command, speech_text, paper.arxiv_id)
    )
    mutable_app._tts_task = task
    app.notify(f"Reading abstract for {paper.arxiv_id}", title="Audio")


def _same_paper_is_playing(app: ArxivBrowser, arxiv_id: str) -> bool:
    return bool(getattr(app, "_tts_paper_id", None) == arxiv_id) and _is_active_task(
        getattr(app, "_tts_task", None)
    )


async def _read_abstract_aloud_async(
    app: ArxivBrowser,
    command: TtsCommand,
    speech_text: str,
    paper_id: str,
) -> None:
    """Launch and monitor one system TTS process."""
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            command.executable,
            *command.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        cast(Any, app)._tts_process = process
        _, stderr = await process.communicate(speech_text.encode("utf-8"))
        if process.returncode:
            _notify_tts_failure(app, paper_id, stderr)
    except FileNotFoundError:
        _notify_if_current(app, paper_id, _tts_missing_message(), severity="warning")
    except OSError as exc:
        logger.warning("TTS playback failed for %s: %s", paper_id, exc, exc_info=True)
        _notify_if_current(app, paper_id, "TTS playback failed", severity="error")
    except asyncio.CancelledError:
        _terminate_process(process)
        if process is not None:
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=0.5)
        raise
    finally:
        if getattr(app, "_tts_task", None) is asyncio.current_task():
            app._tts_task = None
            app._tts_process = None
            app._tts_paper_id = None


def _notify_tts_failure(app: ArxivBrowser, paper_id: str, stderr: bytes) -> None:
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    suffix = f": {stderr_text[:160]}" if stderr_text else ""
    logger.warning("TTS playback failed for %s%s", paper_id, suffix)
    _notify_if_current(app, paper_id, f"TTS playback failed{suffix}", severity="error")


def _notify_if_current(
    app: ArxivBrowser,
    paper_id: str,
    message: str,
    *,
    severity: SeverityLevel,
) -> None:
    if getattr(app, "_tts_paper_id", None) != paper_id:
        return
    app.notify(message, title="Audio", severity=severity, timeout=8)


class AudioActionMixin:
    """Mixin exposing audio actions to the Textual app."""

    action_read_abstract_aloud = action_read_abstract_aloud
