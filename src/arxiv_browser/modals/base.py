"""Shared base class for application modals."""

from __future__ import annotations

from typing import TypeVar

from textual.screen import ModalScreen

T = TypeVar("T")


class ModalBase(ModalScreen[T]):
    """Base class providing shared CSS and helpers for all application modals.

    Subclasses define their own ``CSS`` (widths, heights, modal-specific rules),
    ``BINDINGS``, and ``compose()``. ModalBase centralizes the chrome that used
    to drift between dialogs via ``DEFAULT_CSS`` (combined across the MRO) plus
    reusable classes:

    * The ``ModalBase`` type selector centers every modal's content.
    * ``.modal-dialog`` — canonical dialog box: themed background, ``tall``
      accent border, ``1 2`` padding. Override only width/height (and, for a
      genuinely destructive dialog, the border color) in the subclass.
    * ``.modal-title`` — bold accent title with bottom margin.
    * ``.modal-footer`` — muted key-hint footer line.
    * ``.modal-buttons`` — right-aligned auto-height button row.

    Subclasses should add these classes to the relevant widgets in
    ``compose()`` rather than re-declaring border/background/padding/color.

    Also provides ``action_cancel()`` and ``_focus_widget(selector)``.
    """

    DEFAULT_CSS = """
    ModalBase {
        align: center middle;
    }

    .modal-dialog {
        background: $th-background;
        border: tall $th-accent;
        padding: 1 2;
    }

    .modal-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
    }

    .modal-footer {
        color: $th-muted;
    }

    .modal-buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    def action_cancel(self) -> None:
        """Dismiss the modal with no result (cancel)."""
        self.dismiss(None)

    def _focus_widget(self, selector: str) -> None:
        """Safely focus a widget by CSS selector, ignoring NoMatches."""
        from textual.css.query import NoMatches

        try:
            self.query_one(selector).focus()
        except NoMatches:
            pass
