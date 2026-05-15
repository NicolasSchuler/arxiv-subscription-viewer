"""Import helpers for portable metadata snapshots."""

from __future__ import annotations

import logging
from typing import Any

from arxiv_browser.authors import dedupe_author_names
from arxiv_browser.models import (
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    WATCH_MATCH_TYPES,
    PaperCollection,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
)

logger = logging.getLogger(__name__)


def _import_watch_entries(wl_data: Any, config: UserConfig) -> int:
    """Import watch list entries into config with dedup."""
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


def _import_tracked_authors(authors_data: Any, config: UserConfig) -> int:
    """Import tracked authors into config with exact-match deduplication."""
    if not isinstance(authors_data, list):
        return 0
    existing = {author.casefold() for author in dedupe_author_names(config.tracked_authors)}
    count = 0
    for author in dedupe_author_names([item for item in authors_data if isinstance(item, str)]):
        key = author.casefold()
        if key in existing:
            continue
        config.tracked_authors.append(author)
        existing.add(key)
        count += 1
    return count


def _import_bookmarks(bk_data: Any, config: UserConfig, merge: bool) -> int:
    """Import bookmarks into config with dedup and capacity check."""
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
    """Import collections into config with dedup and capacity checks."""
    if not isinstance(col_data, list) or not merge:
        return 0
    existing_names = {c.name for c in config.collections}
    count = 0
    for col_dict in col_data:
        collection = _collection_from_import(col_dict, existing_names)
        if collection is None:
            continue
        if len(config.collections) >= MAX_COLLECTIONS:
            break
        config.collections.append(collection)
        existing_names.add(collection.name)
        count += 1
    return count


def _collection_from_import(col_dict: Any, existing_names: set[str]) -> PaperCollection | None:
    if not isinstance(col_dict, dict):
        return None
    name = str(col_dict.get("name", ""))
    if not name or name in existing_names:
        return None
    return PaperCollection(
        name=name,
        description=str(col_dict.get("description", "")),
        paper_ids=_safe_collection_paper_ids(col_dict.get("paper_ids", [])),
        created=str(col_dict.get("created", "")),
    )


def _safe_collection_paper_ids(paper_ids: Any) -> list[str]:
    if not isinstance(paper_ids, list):
        return []
    return [pid for pid in paper_ids if isinstance(pid, str)][:MAX_PAPERS_PER_COLLECTION]


__all__ = [
    "_collection_from_import",
    "_import_bookmarks",
    "_import_collections",
    "_import_tracked_authors",
    "_import_watch_entries",
    "_safe_collection_paper_ids",
]
