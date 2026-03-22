# ruff: noqa: F401, F403
"""Static dependency bundle for extracted action modules.

The action layer should depend on canonical modules directly, not on
``arxiv_browser.app`` mutating module globals after import.  This module
centralizes those concrete imports so the action files can stay lightweight
without reaching back into ``app.py`` at runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import platform
import shutil
import sqlite3
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from rapidfuzz import fuzz
from textual.app import ScreenStackError
from textual.css.query import NoMatches

# Keep the wildcard imports in one place: extracted action modules import from
# this runtime bundle instead of assembling their own dependencies or reaching
# back into ``arxiv_browser.app``. That makes the action layer's dependency
# surface explicit while still supporting the temporary compatibility bridge.
from arxiv_browser.action_messages import *
from arxiv_browser.cli import ARXIV_API_MIN_INTERVAL_SECONDS
from arxiv_browser.config import *
from arxiv_browser.config import _coerce_arxiv_api_max_results
from arxiv_browser.enrichment import *
from arxiv_browser.export import *
from arxiv_browser.help_ui import build_help_sections
from arxiv_browser.huggingface import *
from arxiv_browser.io_actions import *
from arxiv_browser.llm import *
from arxiv_browser.llm import (
    LLM_PRESETS,
    SUMMARY_MODES,
    _build_llm_shell_command,
    _compute_command_hash,
    _load_all_relevance_scores,
    _load_relevance_score,
    _load_summary,
    _parse_auto_tag_response,
    _parse_relevance_response,
    _resolve_llm_command,
    _save_relevance_score,
    _save_summary,
)
from arxiv_browser.llm_providers import *
from arxiv_browser.modals import *
from arxiv_browser.models import *
from arxiv_browser.parsing import *
from arxiv_browser.query import *
from arxiv_browser.semantic_scholar import *
from arxiv_browser.services.llm_service import LLMExecutionError as _LLMExecutionError
from arxiv_browser.similarity import *
from arxiv_browser.themes import *

logger = logging.getLogger("arxiv_browser.actions")
SUBPROCESS_TIMEOUT = 5
BATCH_CONFIRM_THRESHOLD = 10
BOOKMARK_NAME_MAX_LEN = 15
ARXIV_API_TIMEOUT = 30
MAX_CONCURRENT_DOWNLOADS = 3
CLIPBOARD_SEPARATOR = "=" * 80


def sync_app_globals(_namespace: dict[str, Any]) -> None:
    """Compatibility bridge for legacy app-module monkeypatching.

    Action modules now import their dependencies statically through this module.
    During the transition away from ``arxiv_browser.app`` as a patch point, keep
    any same-named symbols in sync with the app module so existing compatibility
    tests and one-release shims continue to work.
    """
    app_module = sys.modules.get("arxiv_browser.app")
    if app_module is None:
        return
    for name in list(_namespace):
        if name.startswith("__"):
            continue
        if hasattr(app_module, name):
            _namespace[name] = getattr(app_module, name)


__all__ = [name for name in globals() if not name.startswith("__")]  # pyright: ignore[reportUnsupportedDunderAll]
