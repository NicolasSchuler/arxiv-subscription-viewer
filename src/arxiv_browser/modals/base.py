"""Shared base class for application modals."""

from __future__ import annotations

from typing import TypeVar

from textual.screen import ModalScreen

T = TypeVar("T")


class ModalBase(ModalScreen[T]):
    """Base class providing shared CSS and helpers for all application modals.

    Subclasses still define their own CSS, BINDINGS, and compose().
    ModalBase provides:

    * ``BASE_CSS`` — common structural CSS that subclasses can reference
    * ``action_cancel()`` — default dismiss-with-None handler
    * ``_focus_widget(selector)`` — safe focus helper for on_mount
    """

    BASE_CSS = """
    /* Shared structural CSS — subclasses include this via CSS class variable */
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
