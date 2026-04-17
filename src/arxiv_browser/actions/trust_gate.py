# ruff: noqa: UP037
"""Command trust-gate for LLM and PDF-viewer shell commands.

Encapsulates the security boundary between the browser and arbitrary user
commands: hashing, trusted-hash persistence, and confirmation prompts.

Extracted from ``llm_actions.py`` to make the security surface easier to
audit and test in isolation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ScreenStackError

from arxiv_browser.actions.constants import logger
from arxiv_browser.config import save_config
from arxiv_browser.llm import LLM_PRESETS
from arxiv_browser.modals import ConfirmModal
from arxiv_browser.query import truncate_text

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser


_TRUST_HASH_LENGTH = 16
_COMMAND_PREVIEW_MAX_LEN = 120


@dataclass(slots=True)
class CommandTrustRequest:
    """Deferred trust-prompt request for an external command."""

    command_template: str
    title: str
    prompt_heading: str
    trust_button_label: str
    cancel_message: str
    trusted_hashes: list[str]
    on_trusted: Callable[[], None]


def _trust_hash(command_template: str) -> str:
    """Return a stable short hash for trusted command templates."""
    return hashlib.sha256(command_template.encode("utf-8")).hexdigest()[:_TRUST_HASH_LENGTH]


def _remember_trusted_hash(
    app,
    command_template: str,
    trusted_hashes: list[str],
    title: str,
) -> bool:
    """Add a command hash to the trusted list and persist it to disk.

    The command is always trusted for the current session regardless of
    whether the disk write succeeds.  When ``save_config`` fails, the user
    is notified that the trust is session-only; the function still returns
    ``True`` so the calling action can proceed.
    """
    cmd_hash = app._trust_hash(command_template)
    if cmd_hash in trusted_hashes:
        return True
    trusted_hashes.append(cmd_hash)
    if save_config(app._config):
        return True
    app.notify(
        "Could not save trust preference. Command trusted for this session only.",
        title=title,
        severity="warning",
    )
    return True


def _is_llm_command_trusted(app: "ArxivBrowser", command_template: str) -> bool:
    """Return whether an LLM command template is trusted."""
    config = app._config
    if not config.llm_command and not config.llm_preset:
        return True
    if not config.llm_command and command_template == LLM_PRESETS.get(config.llm_preset, ""):
        return True
    return app._trust_hash(command_template) in config.trusted_llm_command_hashes


def _is_pdf_viewer_trusted(app: "ArxivBrowser", viewer_cmd: str) -> bool:
    """Return whether a PDF viewer command is trusted."""
    config = app._config
    return app._trust_hash(viewer_cmd) in config.trusted_pdf_viewer_hashes


def _ensure_command_trusted(
    app,
    request: CommandTrustRequest,
) -> bool:
    """Show a trust confirmation prompt for a custom shell command.

    Pushes a ``ConfirmModal`` asking the user to approve execution.  If the
    user confirms, ``_remember_trusted_hash`` is called to persist the
    approval and then ``request.on_trusted`` is invoked to continue the action.

    Returns:
        Always ``False`` — the action is deferred to the modal callback.
        The caller should not continue inline after this returns.
    """
    command_preview = truncate_text(request.command_template, _COMMAND_PREVIEW_MAX_LEN)

    def _on_decision(confirmed: bool | None) -> None:
        if not confirmed:
            app.notify(request.cancel_message, title=request.title, severity="warning")
            return
        if app._remember_trusted_hash(
            request.command_template,
            request.trusted_hashes,
            request.title,
        ):
            request.on_trusted()

    try:
        app.push_screen(
            ConfirmModal(
                f"{request.prompt_heading}\n"
                f"{command_preview}\n\n"
                "This command executes on your machine.\n"
                f"Confirm to trust and {request.trust_button_label.lower()}."
            ),
            _on_decision,
        )
        return False
    except ScreenStackError:
        logger.debug("Unable to show %s trust prompt", request.title, exc_info=True)
        app.notify(
            f"Could not confirm {request.title.lower()} command trust; action cancelled.",
            title=request.title,
            severity="warning",
        )
        return False


def _ensure_llm_command_trusted(
    app,
    command_template: str,
    on_trusted: Callable[[], None],
) -> bool:
    """Ensure a custom LLM command is trusted before execution.

    Short-circuits to ``True`` (proceed inline) when the command is already
    trusted.  Otherwise delegates to ``_ensure_command_trusted``, which
    pushes a confirmation modal and returns ``False``.
    """
    if app._is_llm_command_trusted(command_template):
        return True
    return app._ensure_command_trusted(
        CommandTrustRequest(
            command_template=command_template,
            title="LLM",
            prompt_heading="Run untrusted custom LLM command?",
            trust_button_label="Run",
            cancel_message="LLM command cancelled",
            trusted_hashes=app._config.trusted_llm_command_hashes,
            on_trusted=on_trusted,
        )
    )


def _ensure_pdf_viewer_trusted(
    app,
    viewer_cmd: str,
    on_trusted: Callable[[], None],
) -> bool:
    """Ensure a custom PDF viewer command is trusted before execution."""
    if app._is_pdf_viewer_trusted(viewer_cmd):
        return True
    return app._ensure_command_trusted(
        CommandTrustRequest(
            command_template=viewer_cmd,
            title="PDF",
            prompt_heading="Run untrusted custom PDF viewer command?",
            trust_button_label="Open",
            cancel_message="PDF open cancelled",
            trusted_hashes=app._config.trusted_pdf_viewer_hashes,
            on_trusted=on_trusted,
        )
    )
