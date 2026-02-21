"""Targeted tests for config import semantics and load hardening."""

from __future__ import annotations

import json

import pytest

from arxiv_browser.app import (
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
    import_metadata,
    load_config,
)


def test_import_metadata_replace_mode_clears_prior_sections() -> None:
    config = UserConfig(
        paper_metadata={"old": PaperMetadata(arxiv_id="old", notes="stale")},
        watch_list=[WatchListEntry(pattern="old", match_type="keyword")],
        bookmarks=[SearchBookmark(name="old", query="cat:old")],
        collections=[PaperCollection(name="old", paper_ids=["old"])],
        research_interests="old interests",
    )
    data = {
        "format": "arxiv-browser-metadata",
        "paper_metadata": {"new": {"notes": "fresh", "starred": True}},
        "watch_list": [{"pattern": "new", "match_type": "keyword"}],
        "bookmarks": [{"name": "new", "query": "cat:new"}],
        "collections": [{"name": "new-col", "paper_ids": ["new"]}],
        "research_interests": "new interests",
    }

    papers_n, watch_n, bookmarks_n, collections_n = import_metadata(data, config, merge=False)

    assert (papers_n, watch_n, bookmarks_n, collections_n) == (1, 1, 1, 1)
    assert list(config.paper_metadata) == ["new"]
    assert [entry.pattern for entry in config.watch_list] == ["new"]
    assert [bookmark.query for bookmark in config.bookmarks] == ["cat:new"]
    assert [collection.name for collection in config.collections] == ["new-col"]
    assert config.research_interests == "new interests"


def test_import_metadata_merge_mode_preserves_existing_sections() -> None:
    config = UserConfig(
        paper_metadata={"paper-1": PaperMetadata(arxiv_id="paper-1", notes="keep me")},
        watch_list=[WatchListEntry(pattern="existing", match_type="keyword")],
        bookmarks=[SearchBookmark(name="existing", query="cat:existing")],
        collections=[PaperCollection(name="existing", paper_ids=["paper-1"])],
    )
    data = {
        "format": "arxiv-browser-metadata",
        "paper_metadata": {"paper-1": {"notes": "replace-attempt", "tags": ["topic:new"]}},
        "watch_list": [{"pattern": "incoming", "match_type": "keyword"}],
        "bookmarks": [{"name": "incoming", "query": "cat:incoming"}],
        "collections": [{"name": "incoming", "paper_ids": ["paper-2"]}],
    }

    import_metadata(data, config, merge=True)

    assert config.paper_metadata["paper-1"].notes == "keep me"
    assert "topic:new" in config.paper_metadata["paper-1"].tags
    assert {entry.pattern for entry in config.watch_list} == {"existing", "incoming"}
    assert {bookmark.query for bookmark in config.bookmarks} == {"cat:existing", "cat:incoming"}
    assert {collection.name for collection in config.collections} == {"existing", "incoming"}


@pytest.mark.parametrize("payload", [[], "oops", 123])
def test_load_config_non_dict_root_returns_default(payload, tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

    loaded = load_config()

    assert isinstance(loaded, UserConfig)
    assert loaded.config_defaulted is True
    assert (tmp_path / "config.json.corrupt").exists()
    assert not config_file.exists()
