"""Canonical import bundle for tests.

This module exists only to keep tests off the public app compatibility shim while the
public compatibility shim remains stable for external callers. Tests may import
helpers and data objects from here, but monkeypatch targets should still point
at the module where the symbol is actually resolved.
"""

from __future__ import annotations

# ruff: noqa: F401, F403
from arxiv_browser.action_messages import *
from arxiv_browser.browser import *

# Private helpers that older tests imported from the public app shim.
from arxiv_browser.browser.content import MAX_PAPER_CONTENT_LENGTH, _fetch_paper_content_async
from arxiv_browser.browser.contracts import *
from arxiv_browser.browser.core import *
from arxiv_browser.cli import *
from arxiv_browser.cli import _configure_logging, _resolve_legacy_fallback, _resolve_papers
from arxiv_browser.config import *
from arxiv_browser.config import _config_to_dict, _dict_to_config, _safe_get
from arxiv_browser.enrichment import *
from arxiv_browser.export import *
from arxiv_browser.help_ui import *
from arxiv_browser.huggingface import *
from arxiv_browser.io_actions import *
from arxiv_browser.llm import *
from arxiv_browser.llm import (
    _build_llm_shell_command,
    _compute_command_hash,
    _init_relevance_db,
    _init_summary_db,
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
from arxiv_browser.query import _HIGHLIGHT_PATTERN_CACHE
from arxiv_browser.semantic_scholar import *
from arxiv_browser.services.llm_service import LLMExecutionError as _LLMExecutionError
from arxiv_browser.similarity import *
from arxiv_browser.similarity import (
    _compute_tf,
    _extract_author_lastnames,
    _extract_keywords,
    _jaccard_similarity,
    _tokenize_for_tfidf,
)
from arxiv_browser.themes import *
from arxiv_browser.themes import _build_textual_theme
from arxiv_browser.ui_constants import *
from arxiv_browser.ui_runtime import *
from arxiv_browser.widgets import *
from arxiv_browser.widgets.chrome import *
from arxiv_browser.widgets.details import *
from arxiv_browser.widgets.details import _detail_cache_key
from arxiv_browser.widgets.listing import *

__all__ = [name for name in globals() if not name.startswith("__")]
