"""Configuration persistence — load, save, export, import."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, overload

from platformdirs import user_config_dir

from arxiv_browser import conference_deadline_config as _deadline_config
from arxiv_browser.authors import dedupe_author_names
from arxiv_browser.config_import_helpers import (
    _import_bookmarks,
    _import_collections,
    _import_tracked_authors,
    _import_watch_entries,
)
from arxiv_browser.models import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    ARXIV_API_MAX_RESULTS_LIMIT,
    CONFIG_APP_NAME,
    DEFAULT_COLLAPSED_SECTIONS,
    DETAIL_MODES,
    DETAIL_SECTION_KEYS,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    WATCH_MATCH_TYPES,
    LineAnnotation,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    SessionState,
    UserConfig,
    WatchListEntry,
)
from arxiv_browser.review import normalize_review_schedule

logger = logging.getLogger(__name__)

# Validation contract — _dict_to_config() guarantees valid output for any input:
#   Field                  Rule                            Handler
#   ─────────────────────  ──────────────────────────────  ──────────────────
#   sort_index             0 ≤ x < len(SORT_OPTIONS)       _parse_session_state
#   arxiv_api_max_results  1 ≤ x ≤ 200                    _parse_session_state
#   watch_list[].match_type  in WATCH_MATCH_TYPES          _parse_watch_list
#   collapsed_sections[]   each in DETAIL_SECTION_KEYS     _parse_collapsed_sections
#   collections length     ≤ MAX_COLLECTIONS               _parse_collections
#   collection.paper_ids   ≤ MAX_PAPERS_PER_COLLECTION     _parse_collections
#   scalar fields          type-checked via _safe_get()    _dict_to_config
#
CONFIG_FILENAME = "config.json"
USER_TCSS_FILENAME = "user.tcss"


def get_config_path() -> Path:
    """Get the path to the configuration file.

    Uses platformdirs for cross-platform config directory:
    - Linux: ~/.config/arxiv-browser/config.json
    - macOS: ~/Library/Application Support/arxiv-browser/config.json
    - Windows: %APPDATA%/arxiv-browser/config.json
    """
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / CONFIG_FILENAME


def get_user_tcss_path() -> Path:
    """Return the path to the optional user Textual CSS override file.

    The file lives next to ``config.json`` in the platform-specific config
    directory. When present, its contents are layered on top of the
    application's embedded CSS at startup, letting users tweak colors,
    spacing, and borders without editing package source.
    """
    return get_config_path().parent / USER_TCSS_FILENAME


def _config_to_dict(config: UserConfig) -> dict[str, Any]:
    """Serialize UserConfig to a JSON-compatible dictionary.

    Args:
        config: The ``UserConfig`` instance to serialize.

    Returns:
        A nested ``dict`` containing only JSON-compatible types (str, int,
        bool, list, dict).  The ``arxiv_api_max_results`` value is clamped
        to the valid range before serialization.
    """
    max_results = max(1, min(config.arxiv_api_max_results, ARXIV_API_MAX_RESULTS_LIMIT))
    return {
        "version": config.version,
        "show_abstract_preview": config.show_abstract_preview,
        "compact_list": config.compact_list,
        "detail_mode": config.detail_mode,
        "bibtex_export_dir": config.bibtex_export_dir,
        "pdf_download_dir": config.pdf_download_dir,
        "prefer_pdf_url": config.prefer_pdf_url,
        "category_colors": config.category_colors,
        "theme": config.theme,
        "theme_name": config.theme_name,
        "llm_command": config.llm_command,
        "llm_prompt_template": config.llm_prompt_template,
        "llm_phd_explainer_field": config.llm_phd_explainer_field,
        "llm_preset": config.llm_preset,
        "allow_llm_shell_fallback": config.allow_llm_shell_fallback,
        "llm_max_retries": config.llm_max_retries,
        "llm_timeout": config.llm_timeout,
        "llm_streaming_enabled": config.llm_streaming_enabled,
        "llm_provider_type": config.llm_provider_type,
        "llm_api_base_url": config.llm_api_base_url,
        "llm_api_key": config.llm_api_key,
        "llm_api_model": config.llm_api_model,
        "paper_content_cache_ttl_days": config.paper_content_cache_ttl_days,
        "paper_content_pdf_fallback": config.paper_content_pdf_fallback,
        "pdf_preview_max_pages": config.pdf_preview_max_pages,
        "arxiv_api_max_results": max_results,
        "s2_enabled": config.s2_enabled,
        "s2_api_key": config.s2_api_key,
        "s2_cache_ttl_days": config.s2_cache_ttl_days,
        "hf_enabled": config.hf_enabled,
        "hf_cache_ttl_hours": config.hf_cache_ttl_hours,
        "semantic_search_backend": config.semantic_search_backend,
        "semantic_search_model": config.semantic_search_model,
        "semantic_search_api_base_url": config.semantic_search_api_base_url,
        "semantic_search_api_key": config.semantic_search_api_key,
        "semantic_search_top_k": max(1, min(config.semantic_search_top_k, 500)),
        "semantic_search_min_score": max(0, min(config.semantic_search_min_score, 100)),
        "conference_deadlines_enabled": config.conference_deadlines_enabled,
        "conference_deadlines_source_url": config.conference_deadlines_source_url,
        "conference_deadlines_cache_ttl_hours": config.conference_deadlines_cache_ttl_hours,
        "research_interests": config.research_interests,
        "collapsed_sections": config.collapsed_sections,
        "pdf_viewer": config.pdf_viewer,
        "trusted_llm_command_hashes": config.trusted_llm_command_hashes,
        "trusted_pdf_viewer_hashes": config.trusted_pdf_viewer_hashes,
        "session": {
            "scroll_index": config.session.scroll_index,
            "current_filter": config.session.current_filter,
            "sort_index": config.session.sort_index,
            "selected_ids": config.session.selected_ids,
            "current_date": config.session.current_date,
        },
        "paper_metadata": {
            arxiv_id: {
                "notes": meta.notes,
                "tags": meta.tags,
                "is_read": meta.is_read,
                "starred": meta.starred,
                "last_checked_version": meta.last_checked_version,
                "next_review_date": meta.next_review_date,
                "review_stage": meta.review_stage,
                "line_annotations": [
                    {"line": annotation.line, "text": annotation.text}
                    for annotation in meta.line_annotations
                ],
            }
            for arxiv_id, meta in config.paper_metadata.items()
        },
        "watch_list": [
            {
                "pattern": entry.pattern,
                "match_type": entry.match_type,
                "case_sensitive": entry.case_sensitive,
            }
            for entry in config.watch_list
        ],
        "tracked_authors": config.tracked_authors,
        "bookmarks": [{"name": b.name, "query": b.query} for b in config.bookmarks],
        "collections": [
            {
                "name": c.name,
                "description": c.description,
                "paper_ids": c.paper_ids,
                "created": c.created,
            }
            for c in config.collections
        ],
        "marks": config.marks,
        "onboarding_seen": config.onboarding_seen,
        "last_seen_whats_new": config.last_seen_whats_new,
    }


@overload
def _safe_get(data: dict, key: str, default: str, expected_type: type[str]) -> str: ...


@overload
def _safe_get(data: dict, key: str, default: bool, expected_type: type[bool]) -> bool: ...


@overload
def _safe_get(data: dict, key: str, default: int, expected_type: type[int]) -> int: ...


@overload
def _safe_get(data: dict, key: str, default: list, expected_type: type[list]) -> list: ...


@overload
def _safe_get(data: dict, key: str, default: dict, expected_type: type[dict]) -> dict: ...


@overload
def _safe_get(data: dict, key: str, default: Any, expected_type: type) -> Any: ...


def _safe_get(data: dict, key: str, default: Any, expected_type: type) -> Any:
    """Safely get a value from dict with type validation.

    Args:
        data: Source dictionary (typically parsed from JSON).
        key: Key to look up.
        default: Value to return when the key is absent or the value has the
            wrong type.
        expected_type: The Python type the value must be an instance of.
            Integers are checked with an extra ``not isinstance(value, bool)``
            guard to prevent JSON booleans from being accepted as ints.

    Returns:
        The value from ``data[key]`` if it exists and has the correct type,
        otherwise ``default``.
    """
    value = data.get(key, default)
    if expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            return default
        return value
    if not isinstance(value, expected_type):
        return default
    return value


def _coerce_arxiv_api_max_results(value: Any) -> int:
    """Validate and clamp the configured max_results for arXiv API queries.

    Args:
        value: Raw value from the config dict (may be any type).

    Returns:
        An integer in the range ``[1, ARXIV_API_MAX_RESULTS_LIMIT]``.
        Returns ``ARXIV_API_DEFAULT_MAX_RESULTS`` if ``value`` is not a valid
        non-bool integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return ARXIV_API_DEFAULT_MAX_RESULTS
    return max(1, min(value, ARXIV_API_MAX_RESULTS_LIMIT))


_SEMANTIC_SEARCH_BACKENDS = frozenset({"auto", "fastembed", "sentence-transformers", "http", "off"})
_DEFAULT_SEMANTIC_SEARCH_MODEL = "BAAI/bge-small-en-v1.5"


def _coerce_semantic_search_backend(value: Any) -> str:
    """Return a supported semantic-search backend name."""
    if not isinstance(value, str):
        return "auto"
    normalized = value.strip().lower()
    return normalized if normalized in _SEMANTIC_SEARCH_BACKENDS else "auto"


def _coerce_semantic_search_model(value: Any) -> str:
    """Return a non-empty embedding model id, or the lightweight default."""
    if not isinstance(value, str):
        return _DEFAULT_SEMANTIC_SEARCH_MODEL
    normalized = value.strip()
    return normalized or _DEFAULT_SEMANTIC_SEARCH_MODEL


def _parse_collapsed_sections(raw: Any) -> list[str]:
    """Parse and validate collapsed_sections from config data.

    Args:
        raw: Raw value from the config dict under the ``"collapsed_sections"``
            key (expected to be a list of strings).

    Returns:
        A list containing only string values that are valid
        ``DETAIL_SECTION_KEYS`` entries.  Returns ``DEFAULT_COLLAPSED_SECTIONS``
        if ``raw`` is not a list.
    """
    if not isinstance(raw, list):
        return list(DEFAULT_COLLAPSED_SECTIONS)
    valid = [s for s in raw if isinstance(s, str) and s in DETAIL_SECTION_KEYS]
    return valid


def _coerce_detail_mode(value: Any) -> str:
    """Validate the persisted detail density mode.

    Args:
        value: Raw value from the config dict (expected to be a string in
            ``DETAIL_MODES``).

    Returns:
        The validated mode string if ``value`` is a recognized ``DETAIL_MODES``
        entry, otherwise ``"scan"`` (the safe default).
    """
    if isinstance(value, str) and value in DETAIL_MODES:
        return value
    return "scan"


def _parse_session_state(data: dict[str, Any]) -> SessionState:
    """Parse the session state section from config data.

    Args:
        data: Top-level config dictionary.  Reads the ``"session"`` sub-dict.

    Returns:
        A ``SessionState`` instance.  Invalid or out-of-range values are
        replaced with safe defaults (``sort_index`` is clamped to
        ``[0, len(SORT_OPTIONS))``, ``selected_ids`` keeps only strings).
    """
    session_data = data.get("session", {})
    if not isinstance(session_data, dict):
        session_data = {}

    current_date_raw = session_data.get("current_date")
    current_date = current_date_raw if isinstance(current_date_raw, str) else None

    sort_index = _safe_get(session_data, "sort_index", 0, int)
    if sort_index < 0 or sort_index >= len(SORT_OPTIONS):
        sort_index = 0
    selected_ids_raw = _safe_get(session_data, "selected_ids", [], list)
    selected_ids = [item for item in selected_ids_raw if isinstance(item, str)]

    return SessionState(
        scroll_index=_safe_get(session_data, "scroll_index", 0, int),
        current_filter=_safe_get(session_data, "current_filter", "", str),
        sort_index=sort_index,
        selected_ids=selected_ids,
        current_date=current_date,
    )


def _parse_paper_metadata_dict(data: dict[str, Any]) -> dict[str, PaperMetadata]:
    """Parse the paper_metadata section from config data.

    Args:
        data: Top-level config dictionary.  Reads the ``"paper_metadata"``
            sub-dict mapping arXiv IDs to metadata objects.

    Returns:
        Mapping from bare arXiv ID to ``PaperMetadata`` instances.  Malformed
        entries are silently skipped.  Tag lists are filtered to strings only.
    """
    result: dict[str, PaperMetadata] = {}
    raw_metadata = data.get("paper_metadata", {})
    if not isinstance(raw_metadata, dict):
        return result
    for arxiv_id, meta_data in raw_metadata.items():
        if not isinstance(meta_data, dict):
            continue
        lcv_raw = meta_data.get("last_checked_version")
        next_review_date, review_stage = normalize_review_schedule(
            meta_data.get("next_review_date"),
            meta_data.get("review_stage"),
        )
        raw_tags = _safe_get(meta_data, "tags", [], list)
        safe_tags = [tag for tag in raw_tags if isinstance(tag, str)]
        result[arxiv_id] = PaperMetadata(
            arxiv_id=arxiv_id,
            notes=_safe_get(meta_data, "notes", "", str),
            tags=safe_tags,
            is_read=_safe_get(meta_data, "is_read", False, bool),
            starred=_safe_get(meta_data, "starred", False, bool),
            last_checked_version=lcv_raw if isinstance(lcv_raw, int) else None,
            next_review_date=next_review_date,
            review_stage=review_stage,
            line_annotations=_parse_line_annotations(meta_data.get("line_annotations")),
        )
    return result


def _parse_line_annotations(raw_annotations: Any) -> list[LineAnnotation]:
    """Parse persisted inline detail-pane annotations, skipping malformed entries."""
    if not isinstance(raw_annotations, list):
        return []
    annotations: list[LineAnnotation] = []
    for item in raw_annotations:
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        text = item.get("text")
        if isinstance(line, int) and line > 0 and isinstance(text, str) and text.strip():
            annotations.append(LineAnnotation(line=line, text=text.strip()))
    return annotations


def _parse_watch_list(data: dict[str, Any]) -> list[WatchListEntry]:
    """Parse the watch_list section from config data.

    Args:
        data: Top-level config dictionary.  Reads the ``"watch_list"`` list.

    Returns:
        A list of ``WatchListEntry`` instances.  Entries with an unrecognised
        ``match_type`` are logged and default to ``"author"``.  Non-dict
        entries are skipped.
    """
    result: list[WatchListEntry] = []
    raw_watch_list = data.get("watch_list", [])
    if not isinstance(raw_watch_list, list):
        return result
    for entry in raw_watch_list:
        if not isinstance(entry, dict):
            continue
        match_type = _safe_get(entry, "match_type", "author", str)
        if match_type not in WATCH_MATCH_TYPES:
            logger.warning(
                "Invalid watch list match_type %r, defaulting to 'author'",
                match_type,
            )
            match_type = "author"
        result.append(
            WatchListEntry(
                pattern=_safe_get(entry, "pattern", "", str),
                match_type=match_type,
                case_sensitive=_safe_get(entry, "case_sensitive", False, bool),
            )
        )
    return result


def _parse_bookmarks(data: dict[str, Any]) -> list[SearchBookmark]:
    """Parse the bookmarks section from config data.

    Args:
        data: Top-level config dictionary.  Reads the ``"bookmarks"`` list.

    Returns:
        A list of ``SearchBookmark`` instances.  Non-dict entries are skipped.
    """
    result: list[SearchBookmark] = []
    raw_bookmarks = data.get("bookmarks", [])
    if not isinstance(raw_bookmarks, list):
        return result
    for b in raw_bookmarks:
        if not isinstance(b, dict):
            continue
        result.append(
            SearchBookmark(
                name=_safe_get(b, "name", "", str),
                query=_safe_get(b, "query", "", str),
            )
        )
    return result


def _parse_collections(data: dict[str, Any]) -> list[PaperCollection]:
    """Parse the collections section from config data.

    Args:
        data: Top-level config dictionary.  Reads the ``"collections"`` list.

    Returns:
        A list of up to ``MAX_COLLECTIONS`` ``PaperCollection`` instances.
        Collections without a name are skipped.  ``paper_ids`` are truncated
        to ``MAX_PAPERS_PER_COLLECTION`` and filtered to strings only.
    """
    result: list[PaperCollection] = []
    raw_collections = data.get("collections", [])
    if not isinstance(raw_collections, list):
        return result
    for c in raw_collections[:MAX_COLLECTIONS]:
        if not isinstance(c, dict):
            continue
        name = _safe_get(c, "name", "", str)
        if not name:
            continue
        paper_ids = _safe_get(c, "paper_ids", [], list)
        safe_ids = [pid for pid in paper_ids if isinstance(pid, str)][:MAX_PAPERS_PER_COLLECTION]
        result.append(
            PaperCollection(
                name=name,
                description=_safe_get(c, "description", "", str),
                paper_ids=safe_ids,
                created=_safe_get(c, "created", "", str),
            )
        )
    return result


def _parse_str_dict(data: dict[str, Any], key: str) -> dict[str, str]:
    """Parse a dict[str, str] field from config data with type validation.

    Args:
        data: Top-level config dictionary.
        key: The config key whose value is expected to be a ``dict[str, str]``.

    Returns:
        A ``dict[str, str]`` containing only entries where both key and value
        are strings.  Returns an empty dict if the value is absent or has the
        wrong type.
    """
    raw = _safe_get(data, key, {}, dict)
    return {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def _parse_str_list(data: dict[str, Any], key: str) -> list[str]:
    """Parse a list[str] field from config data with type validation.

    Args:
        data: Top-level config dictionary.
        key: The config key whose value is expected to be a ``list[str]``.

    Returns:
        A ``list[str]`` containing only string elements.  Returns an empty
        list if the value is absent or has the wrong type.
    """
    raw = _safe_get(data, key, [], list)
    return [item for item in raw if isinstance(item, str)]


def _validate_llm_prompt_template(template: str) -> str:
    """Validate an LLM prompt template, rejecting unapproved placeholders.

    Args:
        template: Raw prompt template string from the config file.  May be
            empty (treated as "use default").

    Returns:
        The template unchanged if all placeholders are in ``_LLM_PROMPT_FIELDS``,
        or ``""`` (signalling "use default") if any invalid placeholders are
        found.  Logs a warning listing the offending placeholder names.
    """
    if not template:
        return ""
    from arxiv_browser.llm import validate_prompt_template

    invalid = validate_prompt_template(template)
    if invalid:
        logger.warning(
            "LLM prompt template contains invalid placeholders: %s — using default",
            ", ".join(f"{{{p}}}" for p in invalid),
        )
        return ""
    return template


def _dict_to_config(data: dict[str, Any]) -> UserConfig:
    """Deserialize a dictionary to UserConfig with type validation.

    All values are validated and coerced via the ``_parse_*`` / ``_safe_get``
    helpers so that the returned ``UserConfig`` is always in a consistent,
    within-bounds state regardless of what the raw JSON contains.

    Args:
        data: Top-level config dictionary parsed from JSON.

    Returns:
        A fully-populated ``UserConfig`` instance with validated fields.
    """
    marks = data.get("marks", {})
    if not isinstance(marks, dict):
        marks = {}

    return UserConfig(
        paper_metadata=_parse_paper_metadata_dict(data),
        watch_list=_parse_watch_list(data),
        tracked_authors=dedupe_author_names(_parse_str_list(data, "tracked_authors")),
        bookmarks=_parse_bookmarks(data),
        collections=_parse_collections(data),
        marks=marks,
        session=_parse_session_state(data),
        show_abstract_preview=_safe_get(data, "show_abstract_preview", False, bool),
        compact_list=_safe_get(data, "compact_list", False, bool),
        detail_mode=_coerce_detail_mode(data.get("detail_mode")),
        bibtex_export_dir=_safe_get(data, "bibtex_export_dir", "", str),
        pdf_download_dir=_safe_get(data, "pdf_download_dir", "", str),
        prefer_pdf_url=_safe_get(data, "prefer_pdf_url", False, bool),
        category_colors=_parse_str_dict(data, "category_colors"),
        theme=_parse_str_dict(data, "theme"),
        theme_name=_safe_get(data, "theme_name", "monokai", str),
        llm_command=_safe_get(data, "llm_command", "", str),
        llm_prompt_template=_validate_llm_prompt_template(
            _safe_get(data, "llm_prompt_template", "", str)
        ),
        llm_phd_explainer_field=_safe_get(data, "llm_phd_explainer_field", "physics", str),
        llm_preset=_safe_get(data, "llm_preset", "", str),
        allow_llm_shell_fallback=_safe_get(data, "allow_llm_shell_fallback", True, bool),
        llm_max_retries=max(0, min(5, _safe_get(data, "llm_max_retries", 1, int))),
        llm_timeout=max(10, min(600, _safe_get(data, "llm_timeout", 120, int))),
        llm_streaming_enabled=_safe_get(data, "llm_streaming_enabled", False, bool),
        llm_provider_type=_safe_get(data, "llm_provider_type", "cli", str),
        llm_api_base_url=_safe_get(data, "llm_api_base_url", "", str),
        llm_api_key=_safe_get(data, "llm_api_key", "", str),
        llm_api_model=_safe_get(data, "llm_api_model", "", str),
        paper_content_cache_ttl_days=max(
            1, min(365, _safe_get(data, "paper_content_cache_ttl_days", 7, int))
        ),
        paper_content_pdf_fallback=_safe_get(data, "paper_content_pdf_fallback", True, bool),
        pdf_preview_max_pages=max(1, min(20, _safe_get(data, "pdf_preview_max_pages", 3, int))),
        arxiv_api_max_results=_coerce_arxiv_api_max_results(
            data.get("arxiv_api_max_results", ARXIV_API_DEFAULT_MAX_RESULTS)
        ),
        s2_enabled=_safe_get(data, "s2_enabled", False, bool),
        s2_api_key=_safe_get(data, "s2_api_key", "", str),
        s2_cache_ttl_days=_safe_get(data, "s2_cache_ttl_days", 7, int),
        hf_enabled=_safe_get(data, "hf_enabled", False, bool),
        hf_cache_ttl_hours=_safe_get(data, "hf_cache_ttl_hours", 6, int),
        semantic_search_backend=_coerce_semantic_search_backend(
            data.get("semantic_search_backend")
        ),
        semantic_search_model=_coerce_semantic_search_model(data.get("semantic_search_model")),
        semantic_search_api_base_url=_safe_get(data, "semantic_search_api_base_url", "", str),
        semantic_search_api_key=_safe_get(data, "semantic_search_api_key", "", str),
        semantic_search_top_k=max(1, min(500, _safe_get(data, "semantic_search_top_k", 100, int))),
        semantic_search_min_score=max(
            0, min(100, _safe_get(data, "semantic_search_min_score", 15, int))
        ),
        **_deadline_config.from_dict(data),
        research_interests=_safe_get(data, "research_interests", "", str),
        collapsed_sections=_parse_collapsed_sections(data.get("collapsed_sections")),
        pdf_viewer=_safe_get(data, "pdf_viewer", "", str),
        trusted_llm_command_hashes=_parse_str_list(data, "trusted_llm_command_hashes"),
        trusted_pdf_viewer_hashes=_parse_str_list(data, "trusted_pdf_viewer_hashes"),
        version=_safe_get(data, "version", 1, int),
        onboarding_seen=_safe_get(data, "onboarding_seen", False, bool),
        last_seen_whats_new=_safe_get(data, "last_seen_whats_new", "", str),
    )


def _backup_corrupt_config(config_path: Path) -> None:
    """Rename a corrupt config file to .corrupt so the next save doesn't overwrite it.

    Args:
        config_path: Path to the current (corrupt) config file.  The backup
            will be written to the same directory with a ``.json.corrupt``
            suffix.
    """
    backup_path = config_path.with_suffix(".json.corrupt")
    try:
        config_path.rename(backup_path)
        logger.warning("Backed up corrupt config to %s", backup_path)
    except OSError:
        logger.warning("Could not back up corrupt config file", exc_info=True)


def load_config() -> UserConfig:
    """Load configuration from disk.

    Returns default config if file doesn't exist or is corrupted.
    When the file is corrupt (invalid JSON or structure), it is backed up
    to config.json.corrupt before returning defaults.
    Logs specific errors to help diagnose config issues.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return UserConfig()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("Config root must be a JSON object")
        return _dict_to_config(data)
    except json.JSONDecodeError as e:
        logger.warning("Config file has invalid JSON, using defaults: %s", e)
        _backup_corrupt_config(config_path)
        return UserConfig(config_defaulted=True)
    except (KeyError, TypeError) as e:
        logger.warning("Config file has invalid structure, using defaults: %s", e)
        _backup_corrupt_config(config_path)
        return UserConfig(config_defaulted=True)
    except OSError as e:
        logger.warning("Could not read config file, using defaults: %s", e)
        return UserConfig()


def save_config(config: UserConfig) -> bool:
    """Save configuration to disk atomically.

    Uses write-to-tempfile + os.replace() to prevent partial writes
    on crash/interrupt from corrupting the config file.

    Creates the config directory if it doesn't exist.
    Returns True on success, False on failure.
    """
    config_path = get_config_path()

    try:
        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory, then atomically replace
        data = _config_to_dict(config)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp", prefix=".config-")
        closed = False
        try:
            os.write(fd, json_str.encode("utf-8"))
            os.close(fd)
            closed = True
            os.replace(tmp_path, config_path)
        except Exception:
            if not closed:
                os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True
    except OSError as e:
        logger.error("Failed to save config: %s", e)
        return False


def export_metadata(config: UserConfig) -> dict[str, Any]:
    """Export user metadata (read/star/notes/tags/reviews/watch list) as a portable dict.

    The exported data can be loaded on another machine via import_metadata().
    """
    return {
        "format": "arxiv-browser-metadata",
        "version": 1,
        "exported_at": datetime.now().isoformat(),
        "paper_metadata": {
            arxiv_id: {
                "notes": meta.notes,
                "tags": meta.tags,
                "is_read": meta.is_read,
                "starred": meta.starred,
                "last_checked_version": meta.last_checked_version,
                "next_review_date": meta.next_review_date,
                "review_stage": meta.review_stage,
                "line_annotations": [
                    {"line": annotation.line, "text": annotation.text}
                    for annotation in meta.line_annotations
                ],
            }
            for arxiv_id, meta in config.paper_metadata.items()
            if meta.notes
            or meta.tags
            or meta.is_read
            or meta.starred
            or meta.next_review_date
            or meta.line_annotations
        },
        "watch_list": [
            {
                "pattern": entry.pattern,
                "match_type": entry.match_type,
                "case_sensitive": entry.case_sensitive,
            }
            for entry in config.watch_list
        ],
        "tracked_authors": config.tracked_authors,
        "bookmarks": [{"name": b.name, "query": b.query} for b in config.bookmarks],
        "collections": [
            {
                "name": c.name,
                "description": c.description,
                "paper_ids": c.paper_ids,
                "created": c.created,
            }
            for c in config.collections
        ],
        "research_interests": config.research_interests,
    }


def _merge_paper_metadata(existing: PaperMetadata, meta_dict: dict[str, Any]) -> None:
    """Merge imported metadata into an existing PaperMetadata entry in-place.

    Merge rules:
    - Tags are union-merged (deduplication via ``dict.fromkeys``).
    - Notes: imported value wins only when the existing entry has no notes.
    - ``is_read`` and ``starred``: set to ``True`` if the imported value is
      truthy; existing ``True`` is never cleared.

    Args:
        existing: The ``PaperMetadata`` instance to mutate.
        meta_dict: Raw import dict for one paper (from the export format).
    """
    import_tags = meta_dict.get("tags")
    if import_tags and isinstance(import_tags, list):
        valid_tags = [t for t in import_tags if isinstance(t, str)]
        existing.tags = list(dict.fromkeys(existing.tags + valid_tags))
    if not existing.notes and meta_dict.get("notes"):
        existing.notes = str(meta_dict["notes"])
    imported_annotations = _parse_line_annotations(meta_dict.get("line_annotations"))
    if imported_annotations:
        seen = {(annotation.line, annotation.text) for annotation in existing.line_annotations}
        for annotation in imported_annotations:
            key = (annotation.line, annotation.text)
            if key not in seen:
                existing.line_annotations.append(annotation)
                seen.add(key)
    next_review_date, review_stage = normalize_review_schedule(
        meta_dict.get("next_review_date"),
        meta_dict.get("review_stage"),
    )
    if next_review_date and not existing.next_review_date:
        existing.next_review_date = next_review_date
        existing.review_stage = review_stage
    if meta_dict.get("is_read"):
        existing.is_read = True
    if meta_dict.get("starred"):
        existing.starred = True


def _create_paper_metadata(arxiv_id: str, meta_dict: dict[str, Any]) -> PaperMetadata:
    """Create a new PaperMetadata from an import dict.

    Args:
        arxiv_id: Bare arXiv identifier for the new entry.
        meta_dict: Raw import dict for one paper (from the export format).
            Expected keys: ``"notes"``, ``"tags"``, ``"is_read"``,
            ``"starred"``, ``"last_checked_version"``.

    Returns:
        A new ``PaperMetadata`` instance populated from the import dict.
        Missing or wrongly-typed values fall back to safe defaults.
    """
    lcv_raw = meta_dict.get("last_checked_version")
    next_review_date, review_stage = normalize_review_schedule(
        meta_dict.get("next_review_date"),
        meta_dict.get("review_stage"),
    )
    return PaperMetadata(
        arxiv_id=arxiv_id,
        notes=str(meta_dict.get("notes", "")),
        tags=[t for t in meta_dict["tags"] if isinstance(t, str)]
        if isinstance(meta_dict.get("tags"), list)
        else [],
        is_read=bool(meta_dict.get("is_read", False)),
        starred=bool(meta_dict.get("starred", False)),
        last_checked_version=lcv_raw if isinstance(lcv_raw, int) else None,
        next_review_date=next_review_date,
        review_stage=review_stage,
        line_annotations=_parse_line_annotations(meta_dict.get("line_annotations")),
    )


def _import_paper_metadata(pm_data: Any, config: UserConfig, merge: bool) -> int:
    """Import paper metadata entries into config.

    Args:
        pm_data: Raw ``"paper_metadata"`` value from the export dict (expected
            to be a ``dict[str, dict]``).
        config: The ``UserConfig`` to modify in-place.
        merge: When ``True``, existing entries are updated via
            ``_merge_paper_metadata``; when ``False``, they are replaced.

    Returns:
        The number of arXiv IDs processed (including those merged into
        existing entries).
    """
    if not isinstance(pm_data, dict):
        return 0
    count = 0
    for arxiv_id, meta_dict in pm_data.items():
        if not isinstance(meta_dict, dict):
            continue
        existing = config.paper_metadata.get(arxiv_id)
        if existing and merge:
            _merge_paper_metadata(existing, meta_dict)
        else:
            config.paper_metadata[arxiv_id] = _create_paper_metadata(arxiv_id, meta_dict)
        count += 1
    return count


def import_metadata(
    data: dict[str, Any], config: UserConfig, merge: bool = True
) -> tuple[int, int, int, int]:
    """Import metadata from a previously exported dict into config.

    When merge=True (default), existing metadata is preserved and new data
    is merged. When merge=False, imported data deterministically replaces
    import-target metadata sections.

    Returns (papers_imported, watch_entries_imported, bookmarks_imported,
    collections_imported).
    """
    if data.get("format") != "arxiv-browser-metadata":
        raise ValueError("Not a valid arxiv-browser metadata export")

    if not merge:
        # Replace mode: clear all import-target sections before applying data.
        config.paper_metadata.clear()
        config.watch_list.clear()
        config.tracked_authors.clear()
        config.bookmarks.clear()
        config.collections.clear()
        config.research_interests = ""

    papers = _import_paper_metadata(data.get("paper_metadata", {}), config, merge)
    watch = _import_watch_entries(data.get("watch_list", []), config)
    _import_tracked_authors(data.get("tracked_authors", []), config)
    # Bookmark/collection import remains deduplicating, but replace mode now
    # imports into the cleared destination collections.
    bookmarks = _import_bookmarks(data.get("bookmarks", []), config, True)
    collections = _import_collections(data.get("collections", []), config, True)

    if not config.research_interests and data.get("research_interests"):
        config.research_interests = str(data["research_interests"])

    return (papers, watch, bookmarks, collections)


__all__ = [
    "CONFIG_APP_NAME",
    "CONFIG_FILENAME",
    "export_metadata",
    "get_config_path",
    "import_metadata",
    "load_config",
    "save_config",
]
