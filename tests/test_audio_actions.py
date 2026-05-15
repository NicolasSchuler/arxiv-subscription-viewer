"""Tests for system TTS abstract reading actions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arxiv_browser.actions import audio_actions
from arxiv_browser.browser.contracts import COMMAND_PALETTE_COMMANDS
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.help_ui import build_help_sections
from arxiv_browser.ui_constants import APP_BINDINGS


class _TaskStub:
    def __init__(self, *, done: bool = False) -> None:
        self._done = done
        self.cancel = MagicMock()

    def done(self) -> bool:
        return self._done


class _ProcessStub:
    def __init__(self, *, returncode: int | None = None, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.input: bytes | None = None
        self.terminate = MagicMock(side_effect=self._terminate)
        self.waited = False

    def _terminate(self) -> None:
        self.returncode = -15

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.input = input
        return b"", self.stderr

    async def wait(self) -> int | None:
        self.waited = True
        return self.returncode


def _app_for_paper(paper: object, abstract: str = "A useful abstract.") -> SimpleNamespace:
    return SimpleNamespace(
        _get_current_paper=MagicMock(return_value=paper),
        _get_abstract_text=MagicMock(return_value=abstract),
        _tts_task=None,
        _tts_process=None,
        _tts_paper_id=None,
        _track_dataset_task=MagicMock(side_effect=lambda coro: _close_and_return_task(coro)),
        notify=MagicMock(),
    )


def _close_and_return_task(coro: object) -> _TaskStub:
    close = getattr(coro, "close", None)
    if callable(close):
        close()
    return _TaskStub()


def test_resolve_tts_command_uses_macos_say() -> None:
    command = audio_actions._resolve_tts_command(
        "Darwin",
        lambda name: "/usr/bin/say" if name == "say" else None,
    )

    assert command == audio_actions.TtsCommand("/usr/bin/say", (), "say")


def test_resolve_tts_command_prefers_espeak_ng_on_linux() -> None:
    command = audio_actions._resolve_tts_command("Linux", lambda name: f"/usr/bin/{name}")

    assert command == audio_actions.TtsCommand("/usr/bin/espeak-ng", ("--stdin",), "espeak-ng")


def test_resolve_tts_command_falls_back_to_espeak_on_linux() -> None:
    command = audio_actions._resolve_tts_command(
        "Linux",
        lambda name: "/usr/bin/espeak" if name == "espeak" else None,
    )

    assert command == audio_actions.TtsCommand("/usr/bin/espeak", ("--stdin",), "espeak")


def test_resolve_tts_command_rejects_unsupported_platform() -> None:
    assert audio_actions._resolve_tts_command("Windows", lambda _name: "/bin/tts") is None


def test_build_abstract_speech_text_normalizes_title_and_abstract(make_paper) -> None:
    paper = make_paper(title="  A\nCareful   Paper  ")

    assert (
        audio_actions._build_abstract_speech_text(paper, "  This\nis   the abstract. ")
        == "A Careful Paper. Abstract. This is the abstract."
    )


def test_read_abstract_aloud_notifies_when_no_paper() -> None:
    app = SimpleNamespace(_get_current_paper=MagicMock(return_value=None), notify=MagicMock())

    audio_actions.action_read_abstract_aloud(app)

    app.notify.assert_called_once_with("No paper selected", title="Audio", severity="warning")


def test_read_abstract_aloud_notifies_when_no_abstract(make_paper) -> None:
    paper = make_paper()
    app = _app_for_paper(paper, abstract="  ")

    audio_actions.action_read_abstract_aloud(app)

    app.notify.assert_called_once_with("No abstract available", title="Audio", severity="warning")
    app._track_dataset_task.assert_not_called()


def test_read_abstract_aloud_notifies_when_tts_engine_missing(make_paper, monkeypatch) -> None:
    paper = make_paper()
    app = _app_for_paper(paper)
    monkeypatch.setattr(audio_actions, "_resolve_tts_command", lambda: None)
    monkeypatch.setattr(audio_actions.platform, "system", lambda: "Linux")

    audio_actions.action_read_abstract_aloud(app)

    app.notify.assert_called_once_with(
        "System TTS unavailable: install espeak-ng or espeak.",
        title="Audio",
        severity="warning",
        timeout=8,
    )
    app._track_dataset_task.assert_not_called()


def test_read_abstract_aloud_stops_same_paper_playback(make_paper) -> None:
    paper = make_paper()
    app = _app_for_paper(paper)
    task = _TaskStub()
    process = _ProcessStub()
    app._tts_task = task
    app._tts_process = process
    app._tts_paper_id = paper.arxiv_id

    audio_actions.action_read_abstract_aloud(app)

    task.cancel.assert_called_once_with()
    process.terminate.assert_called_once_with()
    assert app._tts_task is None
    assert app._tts_process is None
    assert app._tts_paper_id is None
    app._get_abstract_text.assert_not_called()
    app.notify.assert_called_once_with("Stopped abstract audio", title="Audio")


def test_read_abstract_aloud_replaces_active_playback(make_paper, monkeypatch) -> None:
    old_paper = make_paper(arxiv_id="2401.00001")
    new_paper = make_paper(arxiv_id="2401.00002")
    app = _app_for_paper(new_paper)
    old_task = _TaskStub()
    old_process = _ProcessStub()
    app._tts_task = old_task
    app._tts_process = old_process
    app._tts_paper_id = old_paper.arxiv_id
    monkeypatch.setattr(
        audio_actions,
        "_resolve_tts_command",
        lambda: audio_actions.TtsCommand("/usr/bin/say", (), "say"),
    )

    audio_actions.action_read_abstract_aloud(app)

    old_task.cancel.assert_called_once_with()
    old_process.terminate.assert_called_once_with()
    assert app._tts_paper_id == new_paper.arxiv_id
    assert isinstance(app._tts_task, _TaskStub)
    app._track_dataset_task.assert_called_once()
    app.notify.assert_called_once_with(f"Reading abstract for {new_paper.arxiv_id}", title="Audio")


@pytest.mark.asyncio
async def test_read_abstract_async_sends_stdin_and_clears_state(monkeypatch) -> None:
    process = _ProcessStub(returncode=0)
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(audio_actions.asyncio, "create_subprocess_exec", create)
    app = SimpleNamespace(_tts_task=None, _tts_process=None, _tts_paper_id="2401.00001")

    async def run_current_task() -> None:
        app._tts_task = asyncio.current_task()
        await audio_actions._read_abstract_aloud_async(
            app,
            audio_actions.TtsCommand("/usr/bin/say", (), "say"),
            "hello abstract",
            "2401.00001",
        )

    await run_current_task()

    create.assert_awaited_once()
    assert process.input == b"hello abstract"
    assert app._tts_task is None
    assert app._tts_process is None
    assert app._tts_paper_id is None


@pytest.mark.asyncio
async def test_read_abstract_async_notifies_on_subprocess_failure(monkeypatch) -> None:
    process = _ProcessStub(returncode=2, stderr=b"voice failed")
    monkeypatch.setattr(
        audio_actions.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )
    app = SimpleNamespace(
        _tts_task=None,
        _tts_process=None,
        _tts_paper_id="2401.00001",
        notify=MagicMock(),
    )

    async def run_current_task() -> None:
        app._tts_task = asyncio.current_task()
        await audio_actions._read_abstract_aloud_async(
            app,
            audio_actions.TtsCommand("/usr/bin/say", (), "say"),
            "hello abstract",
            "2401.00001",
        )

    await run_current_task()

    app.notify.assert_called_once_with(
        "TTS playback failed: voice failed",
        title="Audio",
        severity="error",
        timeout=8,
    )
    assert app._tts_task is None


@pytest.mark.asyncio
async def test_read_abstract_async_notifies_when_executable_disappears(monkeypatch) -> None:
    monkeypatch.setattr(
        audio_actions.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError),
    )
    app = SimpleNamespace(
        _tts_task=None,
        _tts_process=None,
        _tts_paper_id="2401.00001",
        notify=MagicMock(),
    )

    async def run_current_task() -> None:
        app._tts_task = asyncio.current_task()
        await audio_actions._read_abstract_aloud_async(
            app,
            audio_actions.TtsCommand("/missing/say", (), "say"),
            "hello abstract",
            "2401.00001",
        )

    await run_current_task()

    assert "System TTS unavailable" in app.notify.call_args.args[0]
    assert app.notify.call_args.kwargs["severity"] == "warning"


def test_read_abstract_aloud_is_registered_in_bindings_help_and_palette() -> None:
    binding_by_action = {binding.action: binding.key for binding in APP_BINDINGS}
    palette_by_action = {command[3]: command for command in COMMAND_PALETTE_COMMANDS}
    sections = build_help_sections(APP_BINDINGS)
    help_entries = {description for _section, entries in sections for _key, description in entries}

    assert binding_by_action["read_abstract_aloud"] == "y"
    assert palette_by_action["read_abstract_aloud"][0] == "Read Abstract Aloud"
    assert "Read abstract aloud" in help_entries
    assert hasattr(ArxivBrowser, "action_read_abstract_aloud")
