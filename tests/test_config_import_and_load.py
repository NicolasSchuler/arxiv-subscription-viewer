"""Targeted tests for config import semantics and load hardening."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import arxiv_browser.config as config_mod
from arxiv_browser.config import (
    import_metadata,
    load_config,
)
from arxiv_browser.models import (
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
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


def test_config_parsing_helpers_cover_validation_and_bounds() -> None:
    valid_section = next(iter(config_mod.DETAIL_SECTION_KEYS))
    with patch("arxiv_browser.llm.validate_prompt_template", return_value=["bad"]):
        config = config_mod._dict_to_config(
            {
                "paper_metadata": {
                    "paper-1": {
                        "notes": 123,
                        "tags": ["keep", 7, "more"],
                        "is_read": True,
                        "starred": False,
                        "last_checked_version": 3,
                    },
                    "paper-2": "skip",
                },
                "watch_list": [
                    {"pattern": "new", "match_type": "invalid", "case_sensitive": "yes"},
                    {"pattern": "author", "match_type": "author", "case_sensitive": True},
                    "skip",
                ],
                "bookmarks": [{"name": "bookmark", "query": "q"}, "skip"],
                "collections": [
                    {"name": "", "paper_ids": ["skip"]},
                    {
                        "name": "col",
                        "description": 5,
                        "paper_ids": [
                            f"p{i}" for i in range(config_mod.MAX_PAPERS_PER_COLLECTION + 3)
                        ],
                    },
                ],
                "marks": ["not", "a", "dict"],
                "session": {
                    "scroll_index": 4,
                    "current_filter": 1,
                    "sort_index": 999,
                    "selected_ids": ["x", 1],
                    "current_date": 123,
                },
                "show_abstract_preview": "yes",
                "detail_mode": "invalid",
                "bibtex_export_dir": 123,
                "pdf_download_dir": "downloads",
                "prefer_pdf_url": True,
                "category_colors": {"cs.AI": "#fff", 1: "#bad"},
                "theme": {"accent": "#000", "ignored": 1},
                "theme_name": 123,
                "llm_command": 123,
                "llm_prompt_template": "{bad}",
                "llm_preset": 123,
                "allow_llm_shell_fallback": "no",
                "llm_max_retries": 9,
                "llm_timeout": 3,
                "arxiv_api_max_results": True,
                "s2_enabled": True,
                "s2_api_key": 123,
                "s2_cache_ttl_days": 8,
                "hf_enabled": True,
                "hf_cache_ttl_hours": 9,
                "research_interests": 123,
                "collapsed_sections": [valid_section, "bad", 1],
                "pdf_viewer": 123,
                "trusted_llm_command_hashes": ["abc", 1],
                "trusted_pdf_viewer_hashes": ["def", 2],
                "version": 2,
                "onboarding_seen": True,
            }
        )

    assert config.paper_metadata["paper-1"].notes == ""
    assert config.paper_metadata["paper-1"].tags == ["keep", "more"]
    assert config.paper_metadata["paper-1"].last_checked_version == 3
    assert [entry.match_type for entry in config.watch_list] == ["author", "author"]
    assert config.bookmarks == [SearchBookmark(name="bookmark", query="q")]
    assert len(config.collections) == 1
    assert config.collections[0].paper_ids == [
        f"p{i}" for i in range(config_mod.MAX_PAPERS_PER_COLLECTION)
    ]
    assert config.marks == {}
    assert config.session.scroll_index == 4
    assert config.session.current_filter == ""
    assert config.session.sort_index == 0
    assert config.session.selected_ids == ["x"]
    assert config.session.current_date is None
    assert config.show_abstract_preview is False
    assert config.detail_mode == "scan"
    assert config.bibtex_export_dir == ""
    assert config.pdf_download_dir == "downloads"
    assert config.prefer_pdf_url is True
    assert config.category_colors == {"cs.AI": "#fff"}
    assert config.theme == {"accent": "#000"}
    assert config.theme_name == "monokai"
    assert config.llm_command == ""
    assert config.llm_prompt_template == ""
    assert config.llm_preset == ""
    assert config.allow_llm_shell_fallback is True
    assert config.llm_max_retries == 5
    assert config.llm_timeout == 10
    assert config.arxiv_api_max_results == config_mod.ARXIV_API_DEFAULT_MAX_RESULTS
    assert config.s2_enabled is True
    assert config.s2_api_key == ""
    assert config.s2_cache_ttl_days == 8
    assert config.hf_enabled is True
    assert config.hf_cache_ttl_hours == 9
    assert config.research_interests == ""
    assert config.collapsed_sections == [valid_section]
    assert config.pdf_viewer == ""
    assert config.trusted_llm_command_hashes == ["abc"]
    assert config.trusted_pdf_viewer_hashes == ["def"]
    assert config.version == 2
    assert config.onboarding_seen is True

    assert config_mod._parse_watch_list({"watch_list": "oops"}) == []
    assert config_mod._parse_bookmarks({"bookmarks": "oops"}) == []
    assert config_mod._parse_collections({"collections": "oops"}) == []
    assert config_mod._parse_str_dict({"bad": {"x": 1}}, "bad") == {}
    assert config_mod._parse_str_list({"bad": [1, "x"]}, "bad") == ["x"]
    assert config_mod._parse_collapsed_sections("oops") == list(
        config_mod.DEFAULT_COLLAPSED_SECTIONS
    )

    with patch("arxiv_browser.llm.validate_prompt_template", return_value=[]):
        assert config_mod._validate_llm_prompt_template("prompt") == "prompt"
    assert config_mod._validate_llm_prompt_template("") == ""


def test_config_import_export_and_disk_error_paths(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError):
        config_mod.import_metadata({"format": "wrong"}, UserConfig())

    existing = PaperMetadata(arxiv_id="paper-1", notes="keep", tags=["keep"], starred=False)
    config_mod._merge_paper_metadata(
        existing,
        {"notes": "incoming", "tags": ["keep", "new", 1], "is_read": True, "starred": True},
    )
    assert existing.notes == "keep"
    assert existing.tags == ["keep", "new"]
    assert existing.is_read is True
    assert existing.starred is True

    created = config_mod._create_paper_metadata(
        "paper-2",
        {
            "notes": 123,
            "tags": ["alpha", 1],
            "is_read": "yes",
            "starred": 1,
            "last_checked_version": "bad",
        },
    )
    assert created.notes == "123"
    assert created.tags == ["alpha"]
    assert created.is_read is True
    assert created.starred is True
    assert created.last_checked_version is None

    config = UserConfig()
    assert config_mod._import_paper_metadata("oops", config, merge=True) == 0
    assert config_mod._import_watch_entries("oops", config) == 0
    assert config_mod._import_bookmarks("oops", config, merge=True) == 0
    assert config_mod._import_collections("oops", config, merge=True) == 0

    watch_config = UserConfig(watch_list=[WatchListEntry(pattern="dup", match_type="keyword")])
    assert (
        config_mod._import_watch_entries(
            [
                {"pattern": "dup", "match_type": "keyword"},
                {"pattern": "new", "match_type": "invalid", "case_sensitive": "yes"},
                {"pattern": "", "match_type": "keyword"},
            ],
            watch_config,
        )
        == 1
    )
    assert watch_config.watch_list[-1].match_type == "keyword"
    assert watch_config.watch_list[-1].case_sensitive is True

    bookmark_config = UserConfig(bookmarks=[SearchBookmark(name="keep", query="q0")])
    bookmarks = [{"name": f"B{i}", "query": f"q{i}"} for i in range(12)]
    assert config_mod._import_bookmarks(bookmarks, bookmark_config, merge=False) == 0
    assert config_mod._import_bookmarks(bookmarks, bookmark_config, merge=True) == 8
    assert len(bookmark_config.bookmarks) == 9

    collection_config = UserConfig()
    collections = [
        {
            "name": f"C{i}",
            "description": i,
            "paper_ids": [f"p{i}", 1, f"p{i + 1}"],
        }
        for i in range(config_mod.MAX_COLLECTIONS + 2)
    ]
    assert config_mod._import_collections(collections, collection_config, merge=False) == 0
    assert (
        config_mod._import_collections(collections, collection_config, merge=True)
        == config_mod.MAX_COLLECTIONS
    )
    assert len(collection_config.collections) == config_mod.MAX_COLLECTIONS
    assert len(collection_config.collections[0].paper_ids) == 2

    exported = config_mod.export_metadata(
        UserConfig(
            paper_metadata={
                "visible": PaperMetadata(
                    arxiv_id="visible",
                    notes="notes",
                    tags=["tag"],
                    is_read=True,
                    starred=True,
                ),
                "hidden": PaperMetadata(arxiv_id="hidden"),
            },
            watch_list=[WatchListEntry(pattern="x", match_type="keyword")],
            bookmarks=[SearchBookmark(name="B", query="q")],
            collections=[PaperCollection(name="C", paper_ids=["p1"])],
            research_interests="interests",
        )
    )
    assert exported["format"] == "arxiv-browser-metadata"
    assert "visible" in exported["paper_metadata"]
    assert "hidden" not in exported["paper_metadata"]

    config_file = tmp_path / "config.json"
    config_file.write_text("{", encoding="utf-8")
    monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)
    loaded = load_config()
    assert loaded.config_defaulted is True
    assert (tmp_path / "config.json.corrupt").exists()

    bad_config_path = tmp_path / "corrupt.json"
    bad_config_path.write_text("{}", encoding="utf-8")
    with patch.object(Path, "rename", side_effect=OSError("boom")):
        config_mod._backup_corrupt_config(bad_config_path)

    with (
        patch("arxiv_browser.config.get_config_path", return_value=tmp_path / "save.json"),
        patch("arxiv_browser.config.os.replace", side_effect=OSError("boom")),
    ):
        assert config_mod.save_config(UserConfig()) is False


def test_load_config_oserror_returns_plain_defaults(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("arxiv_browser.config.get_config_path", lambda: config_file)

    with patch.object(Path, "read_text", side_effect=OSError("boom")):
        loaded = load_config()

    assert isinstance(loaded, UserConfig)
    assert loaded.config_defaulted is False
    assert not (tmp_path / "config.json.corrupt").exists()
