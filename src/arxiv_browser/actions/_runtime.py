"""Shared runtime helpers for extracted action modules.

This module intentionally stays independent from ``arxiv_browser.app`` so
action modules do not create circular imports. ``app.py`` wrappers call into
these action modules after app import has completed.
"""

from __future__ import annotations

import sys
from typing import Any


def sync_app_globals(namespace: dict[str, Any]) -> None:
    """Sync patched globals from ``arxiv_browser.app`` into ``namespace``."""
    app_module = sys.modules.get("arxiv_browser.app")
    if app_module is None:
        return
    namespace.update(vars(app_module))
