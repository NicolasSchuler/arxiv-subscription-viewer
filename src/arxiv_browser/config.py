"""Configuration persistence — load, save, export, import."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from arxiv_browser.models import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    ARXIV_API_MAX_RESULTS_LIMIT,
    CONFIG_APP_NAME,
    DEFAULT_COLLAPSED_SECTIONS,
    DETAIL_SECTION_KEYS,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    WATCH_MATCH_TYPES,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    SessionState,
    UserConfig,
    WatchListEntry,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Persistence
# ============================================================================
#
# Validation contract — _dict_to_config() guarantees valid output for any input:
#
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
# SessionState.__post_init__ provides defense-in-depth clamping for sort_index,
# ensuring safety even when constructed directly (not via deserialization).
#
CONFIG_FILENAME = "config.json"


def get_config_path() -> Path:
    """Get the path to the configuration file.

    Uses platformdirs for cross-platform config directory:
    - Linux: ~/.config/arxiv-browser/config.json
    - macOS: ~/Library/Application Support/arxiv-browser/config.json
    - Windows: %APPDATA%/arxiv-browser/config.json
    """
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / CONFIG_FILENAME


def _config_to_dict(config: UserConfig) -> dict[str, Any]:
    """Serialize UserConfig to a JSON-compatible dictionary."""
    max_results = max(1, min(config.arxiv_api_max_results, ARXIV_API_MAX_RESULTS_LIMIT))
    return {
        "version": config.version,
        "show_abstract_preview": config.show_abstract_preview,
        "bibtex_export_dir": config.bibtex_export_dir,
        "pdf_download_dir": config.pdf_download_dir,
        "prefer_pdf_url": config.prefer_pdf_url,
        "category_colors": config.category_colors,
        "theme": config.theme,
        "theme_name": config.theme_name,
        "llm_command": config.llm_command,
        "llm_prompt_template": config.llm_prompt_template,
        "llm_preset": config.llm_preset,
        "arxiv_api_max_results": max_results,
        "s2_enabled": config.s2_enabled,
        "s2_api_key": config.s2_api_key,
        "s2_cache_ttl_days": config.s2_cache_ttl_days,
        "hf_enabled": config.hf_enabled,
        "hf_cache_ttl_hours": config.hf_cache_ttl_hours,
        "research_interests": config.research_interests,
        "collapsed_sections": config.collapsed_sections,
        "pdf_viewer": config.pdf_viewer,
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
    }


def _safe_get(data: dict, key: str, default: Any, expected_type: type) -> Any:
    """Safely get a value from dict with type validation.

    Returns the default if key is missing or value has wrong type.
    """
    value = data.get(key, default)
    if not isinstance(value, expected_type):
        return default
    return value


def _coerce_arxiv_api_max_results(value: Any) -> int:
    """Validate and clamp the configured max_results for arXiv API queries."""
    if not isinstance(value, int):
        return ARXIV_API_DEFAULT_MAX_RESULTS
    return max(1, min(value, ARXIV_API_MAX_RESULTS_LIMIT))


def _parse_collapsed_sections(raw: Any) -> list[str]:
    """Parse and validate collapsed_sections from config data."""
    if not isinstance(raw, list):
        return list(DEFAULT_COLLAPSED_SECTIONS)
    valid = [s for s in raw if isinstance(s, str) and s in DETAIL_SECTION_KEYS]
    return valid


def _parse_session_state(data: dict[str, Any]) -> SessionState:
    """Parse the session state section from config data."""
    session_data = data.get("session", {})
    if not isinstance(session_data, dict):
        session_data = {}

    current_date_raw = session_data.get("current_date")
    current_date = current_date_raw if isinstance(current_date_raw, str) else None

    sort_index = _safe_get(session_data, "sort_index", 0, int)
    if sort_index < 0 or sort_index >= len(SORT_OPTIONS):
        sort_index = 0

    return SessionState(
        scroll_index=_safe_get(session_data, "scroll_index", 0, int),
        current_filter=_safe_get(session_data, "current_filter", "", str),
        sort_index=sort_index,
        selected_ids=_safe_get(session_data, "selected_ids", [], list),
        current_date=current_date,
    )


def _parse_paper_metadata_dict(data: dict[str, Any]) -> dict[str, PaperMetadata]:
    """Parse the paper_metadata section from config data."""
    result: dict[str, PaperMetadata] = {}
    raw_metadata = data.get("paper_metadata", {})
    if not isinstance(raw_metadata, dict):
        return result
    for arxiv_id, meta_data in raw_metadata.items():
        if not isinstance(meta_data, dict):
            continue
        lcv_raw = meta_data.get("last_checked_version")
        result[arxiv_id] = PaperMetadata(
            arxiv_id=arxiv_id,
            notes=_safe_get(meta_data, "notes", "", str),
            tags=_safe_get(meta_data, "tags", [], list),
            is_read=_safe_get(meta_data, "is_read", False, bool),
            starred=_safe_get(meta_data, "starred", False, bool),
            last_checked_version=lcv_raw if isinstance(lcv_raw, int) else None,
        )
    return result


def _parse_watch_list(data: dict[str, Any]) -> list[WatchListEntry]:
    """Parse the watch_list section from config data."""
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
    """Parse the bookmarks section from config data."""
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
    """Parse the collections section from config data."""
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
    """Parse a dict[str, str] field from config data with type validation."""
    raw = _safe_get(data, key, {}, dict)
    return {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def _dict_to_config(data: dict[str, Any]) -> UserConfig:
    """Deserialize a dictionary to UserConfig with type validation."""
    marks = data.get("marks", {})
    if not isinstance(marks, dict):
        marks = {}

    return UserConfig(
        paper_metadata=_parse_paper_metadata_dict(data),
        watch_list=_parse_watch_list(data),
        bookmarks=_parse_bookmarks(data),
        collections=_parse_collections(data),
        marks=marks,
        session=_parse_session_state(data),
        show_abstract_preview=_safe_get(data, "show_abstract_preview", False, bool),
        bibtex_export_dir=_safe_get(data, "bibtex_export_dir", "", str),
        pdf_download_dir=_safe_get(data, "pdf_download_dir", "", str),
        prefer_pdf_url=_safe_get(data, "prefer_pdf_url", False, bool),
        category_colors=_parse_str_dict(data, "category_colors"),
        theme=_parse_str_dict(data, "theme"),
        theme_name=_safe_get(data, "theme_name", "monokai", str),
        llm_command=_safe_get(data, "llm_command", "", str),
        llm_prompt_template=_safe_get(data, "llm_prompt_template", "", str),
        llm_preset=_safe_get(data, "llm_preset", "", str),
        arxiv_api_max_results=_coerce_arxiv_api_max_results(
            data.get("arxiv_api_max_results", ARXIV_API_DEFAULT_MAX_RESULTS)
        ),
        s2_enabled=_safe_get(data, "s2_enabled", False, bool),
        s2_api_key=_safe_get(data, "s2_api_key", "", str),
        s2_cache_ttl_days=_safe_get(data, "s2_cache_ttl_days", 7, int),
        hf_enabled=_safe_get(data, "hf_enabled", False, bool),
        hf_cache_ttl_hours=_safe_get(data, "hf_cache_ttl_hours", 6, int),
        research_interests=_safe_get(data, "research_interests", "", str),
        collapsed_sections=_parse_collapsed_sections(data.get("collapsed_sections")),
        pdf_viewer=_safe_get(data, "pdf_viewer", "", str),
        version=_safe_get(data, "version", 1, int),
    )


def load_config() -> UserConfig:
    """Load configuration from disk.

    Returns default config if file doesn't exist or is corrupted.
    Logs specific errors to help diagnose config issues.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return UserConfig()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return _dict_to_config(data)
    except json.JSONDecodeError as e:
        logger.warning("Config file has invalid JSON, using defaults: %s", e)
        return UserConfig()
    except (KeyError, TypeError) as e:
        logger.warning("Config file has invalid structure, using defaults: %s", e)
        return UserConfig()
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
        except BaseException:
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
    """Export user metadata (read/star/notes/tags/watch list) as a portable dict.

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
            }
            for arxiv_id, meta in config.paper_metadata.items()
            if meta.notes or meta.tags or meta.is_read or meta.starred
        },
        "watch_list": [
            {
                "pattern": entry.pattern,
                "match_type": entry.match_type,
                "case_sensitive": entry.case_sensitive,
            }
            for entry in config.watch_list
        ],
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
    """Merge imported metadata into an existing PaperMetadata entry."""
    import_tags = meta_dict.get("tags")
    if import_tags and isinstance(import_tags, list):
        valid_tags = [t for t in import_tags if isinstance(t, str)]
        existing.tags = list(dict.fromkeys(existing.tags + valid_tags))
    if not existing.notes and meta_dict.get("notes"):
        existing.notes = str(meta_dict["notes"])
    if meta_dict.get("is_read"):
        existing.is_read = True
    if meta_dict.get("starred"):
        existing.starred = True


def _create_paper_metadata(arxiv_id: str, meta_dict: dict[str, Any]) -> PaperMetadata:
    """Create a new PaperMetadata from an import dict."""
    lcv_raw = meta_dict.get("last_checked_version")
    return PaperMetadata(
        arxiv_id=arxiv_id,
        notes=str(meta_dict.get("notes", "")),
        tags=[t for t in meta_dict["tags"] if isinstance(t, str)]
        if isinstance(meta_dict.get("tags"), list)
        else [],
        is_read=bool(meta_dict.get("is_read", False)),
        starred=bool(meta_dict.get("starred", False)),
        last_checked_version=lcv_raw if isinstance(lcv_raw, int) else None,
    )


def _import_paper_metadata(pm_data: Any, config: UserConfig, merge: bool) -> int:
    """Import paper metadata entries into config. Returns count imported."""
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


def _import_watch_entries(wl_data: Any, config: UserConfig) -> int:
    """Import watch list entries into config with dedup. Returns count imported."""
    if not isinstance(wl_data, list):
        return 0
    existing_patterns = {(e.pattern, e.match_type) for e in config.watch_list}
    count = 0
    for entry_dict in wl_data:
        if not isinstance(entry_dict, dict):
            continue
        pattern = str(entry_dict.get("pattern", ""))
        match_type = str(entry_dict.get("match_type", "keyword"))
        if match_type not in WATCH_MATCH_TYPES:
            logger.warning(
                "Unknown watch match_type %r for pattern %r, defaulting to 'keyword'",
                match_type,
                pattern,
            )
            match_type = "keyword"
        if not pattern or (pattern, match_type) in existing_patterns:
            continue
        config.watch_list.append(
            WatchListEntry(
                pattern=pattern,
                match_type=match_type,
                case_sensitive=bool(entry_dict.get("case_sensitive", False)),
            )
        )
        count += 1
    return count


def _import_bookmarks(bk_data: Any, config: UserConfig, merge: bool) -> int:
    """Import bookmarks into config with dedup and capacity check. Returns count imported."""
    if not isinstance(bk_data, list) or not merge:
        return 0
    existing_queries = {b.query for b in config.bookmarks}
    count = 0
    for bk_dict in bk_data:
        if not isinstance(bk_dict, dict):
            continue
        query = str(bk_dict.get("query", ""))
        if not query or query in existing_queries:
            continue
        if len(config.bookmarks) >= 9:
            break
        config.bookmarks.append(
            SearchBookmark(name=str(bk_dict.get("name", "Imported")), query=query)
        )
        count += 1
    return count


def _import_collections(col_data: Any, config: UserConfig, merge: bool) -> int:
    """Import collections into config with dedup and capacity check. Returns count imported."""
    if not isinstance(col_data, list) or not merge:
        return 0
    existing_names = {c.name for c in config.collections}
    count = 0
    for col_dict in col_data:
        if not isinstance(col_dict, dict):
            continue
        name = str(col_dict.get("name", ""))
        if not name or name in existing_names:
            continue
        if len(config.collections) >= MAX_COLLECTIONS:
            break
        paper_ids = col_dict.get("paper_ids", [])
        safe_ids = (
            [pid for pid in paper_ids if isinstance(pid, str)][:MAX_PAPERS_PER_COLLECTION]
            if isinstance(paper_ids, list)
            else []
        )
        config.collections.append(
            PaperCollection(
                name=name,
                description=str(col_dict.get("description", "")),
                paper_ids=safe_ids,
                created=str(col_dict.get("created", "")),
            )
        )
        count += 1
    return count


def import_metadata(
    data: dict[str, Any], config: UserConfig, merge: bool = True
) -> tuple[int, int, int, int]:
    """Import metadata from a previously exported dict into config.

    When merge=True (default), existing metadata is preserved and new data
    is merged. When merge=False, imported data replaces existing.

    Returns (papers_imported, watch_entries_imported, bookmarks_imported,
    collections_imported).
    """
    if data.get("format") != "arxiv-browser-metadata":
        raise ValueError("Not a valid arxiv-browser metadata export")

    papers = _import_paper_metadata(data.get("paper_metadata", {}), config, merge)
    watch = _import_watch_entries(data.get("watch_list", []), config)
    bookmarks = _import_bookmarks(data.get("bookmarks", []), config, merge)
    collections = _import_collections(data.get("collections", []), config, merge)

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
