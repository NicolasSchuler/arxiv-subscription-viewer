"""Tests for centralized empty-state copy helpers."""

from __future__ import annotations

from arxiv_browser.empty_state import (
    CITATIONS_CITES_EMPTY,
    CITATIONS_REFS_EMPTY,
    COLLECTION_DETAIL_EMPTY,
    COLLECTIONS_MANAGE_EMPTY,
    COLLECTIONS_PICK_EMPTY,
    build_list_empty_message,
)


def test_main_list_empty_messages_mention_escape_hatches() -> None:
    msg = build_list_empty_message(
        query="", in_arxiv_api_mode=False, watch_filter_active=False, history_mode=False
    )
    assert "A" in msg  # press A to search arXiv

    msg_search = build_list_empty_message(
        query="foo", in_arxiv_api_mode=False, watch_filter_active=False, history_mode=False
    )
    assert "Esc" in msg_search

    msg_api = build_list_empty_message(
        query="", in_arxiv_api_mode=True, watch_filter_active=False, history_mode=False
    )
    assert "Ctrl+e" in msg_api or "Esc" in msg_api

    msg_watch = build_list_empty_message(
        query="", in_arxiv_api_mode=False, watch_filter_active=True, history_mode=False
    )
    assert "w" in msg_watch.lower()

    msg_hist = build_list_empty_message(
        query="", in_arxiv_api_mode=False, watch_filter_active=False, history_mode=True
    )
    assert "[" in msg_hist and "]" in msg_hist


def test_secondary_empty_copy_contains_actionable_verb() -> None:
    for msg in (
        COLLECTIONS_MANAGE_EMPTY,
        COLLECTIONS_PICK_EMPTY,
        COLLECTION_DETAIL_EMPTY,
        CITATIONS_REFS_EMPTY,
        CITATIONS_CITES_EMPTY,
    ):
        assert "Try:" in msg, f"missing Try: guidance in {msg!r}"
