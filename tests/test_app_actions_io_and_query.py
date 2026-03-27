#!/usr/bin/env python3
"""High-impact coverage tests for action-heavy paths in app.py."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.semantic_scholar import (
    CitationEntry,
    S2RecommendationsCacheSnapshot,
    SemanticScholarPaper,
)
from tests.support import canonical_exports as app_mod
from tests.support.app_stubs import (
    _DummyOptionList,
    _make_hf_paper,
    _make_s2_paper,
    _new_app,
)
from tests.support.canonical_exports import (
    ArxivBrowser,
    PaperCollection,
    PaperMetadata,
    SearchBookmark,
    UserConfig,
    _resolve_legacy_fallback,
    _resolve_papers,
)


class TestIoActionHelpers:
    def test_resolve_target_papers_preserves_order_and_includes_hidden(self, make_paper):
        from arxiv_browser.io_actions import resolve_target_papers

        visible = [
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00001"),
        ]
        hidden = make_paper(arxiv_id="2401.99999")

        result = resolve_target_papers(
            filtered_papers=visible,
            selected_ids={"2401.99999", "2401.00001"},
            papers_by_id={p.arxiv_id: p for p in [*visible, hidden]},
            current_paper=None,
        )

        assert [p.arxiv_id for p in result] == ["2401.00001", "2401.99999"]

    def test_filter_papers_needing_download_splits_pending_and_skipped(self, make_paper, tmp_path):
        from arxiv_browser.io_actions import filter_papers_needing_download

        existing_paper = make_paper(arxiv_id="2401.01001")
        pending_paper = make_paper(arxiv_id="2401.01002")
        existing_path = tmp_path / "exists.pdf"
        existing_path.write_bytes(b"ready")
        pending_path = tmp_path / "missing.pdf"

        path_map = {
            existing_paper.arxiv_id: existing_path,
            pending_paper.arxiv_id: pending_path,
        }
        to_download, skipped = filter_papers_needing_download(
            [existing_paper, pending_paper],
            lambda paper: path_map[paper.arxiv_id],
        )

        assert [p.arxiv_id for p in to_download] == [pending_paper.arxiv_id]
        assert skipped == [existing_paper.arxiv_id]

    def test_build_markdown_export_document_renders_expected_structure(self):
        from arxiv_browser.io_actions import build_markdown_export_document

        markdown = build_markdown_export_document(["## Paper A", "## Paper B"])
        assert markdown.startswith("# arXiv Papers Export")
        assert "*Exported 2 paper(s)*" in markdown
        assert markdown.count("\n---\n") == 2

    def test_write_timestamped_export_file_writes_atomically(self, tmp_path):
        from arxiv_browser.io_actions import write_timestamped_export_file

        export_dir = tmp_path / "exports"
        out = write_timestamped_export_file(
            content="hello",
            export_dir=export_dir,
            extension="txt",
            now=datetime(2026, 2, 13, 20, 0, 0),
        )
        assert out.name == "arxiv-2026-02-13_200000.txt"
        assert out.read_text(encoding="utf-8") == "hello"
        assert list(export_dir.glob(".txt-*.tmp")) == []

    def test_build_viewer_args_supports_placeholders_and_appending(self):
        from arxiv_browser.io_actions import build_viewer_args

        assert build_viewer_args("open -a Skim {path}", "/tmp/paper.pdf") == [
            "open",
            "-a",
            "Skim",
            "/tmp/paper.pdf",
        ]
        assert build_viewer_args("zathura", "https://arxiv.org/pdf/1") == [
            "zathura",
            "https://arxiv.org/pdf/1",
        ]

    def test_build_viewer_args_rejects_empty_command(self):
        from arxiv_browser.io_actions import build_viewer_args

        with pytest.raises(ValueError, match="empty"):
            build_viewer_args("   ", "https://arxiv.org/pdf/1")

    def test_build_clipboard_payload_uses_separator_line(self):
        from arxiv_browser.io_actions import build_clipboard_payload

        payload = build_clipboard_payload(["A", "B"], "====")
        assert payload == "A\n\n====\n\nB"

    def test_get_clipboard_command_plan_variants(self):
        from arxiv_browser.io_actions import get_clipboard_command_plan

        assert get_clipboard_command_plan("Darwin") == ([["pbcopy"]], "utf-8")
        assert get_clipboard_command_plan("Linux") == (
            [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]],
            "utf-8",
        )
        assert get_clipboard_command_plan("Windows") == ([["clip"]], "utf-16")
        assert get_clipboard_command_plan("Plan9") is None

    def test_batch_confirmation_and_notification_helpers(self):
        from arxiv_browser.action_messages import (
            build_download_pdfs_confirmation_prompt,
            build_download_start_notification,
            build_open_papers_confirmation_prompt,
            build_open_papers_notification,
            build_open_pdfs_confirmation_prompt,
            build_open_pdfs_notification,
            requires_batch_confirmation,
        )

        assert requires_batch_confirmation(4, 3) is True
        assert requires_batch_confirmation(3, 3) is False
        open_papers_prompt = build_open_papers_confirmation_prompt(4)
        open_pdfs_prompt = build_open_pdfs_confirmation_prompt(5)
        download_prompt = build_download_pdfs_confirmation_prompt(6)

        assert "Open 4 papers in your browser?" in open_papers_prompt
        assert "This may open many tabs." in open_papers_prompt
        assert "[y]" not in open_papers_prompt
        assert "[n/Esc]" not in open_papers_prompt

        assert "Open 5 PDFs in your browser?" in open_pdfs_prompt
        assert "viewer windows" in open_pdfs_prompt
        assert "[y]" not in open_pdfs_prompt
        assert "[n/Esc]" not in open_pdfs_prompt

        assert "Download 6 PDFs?" in download_prompt
        assert "Already-downloaded files will be skipped." in download_prompt
        assert "[y]" not in download_prompt
        assert "[n/Esc]" not in download_prompt
        assert build_open_papers_notification(1) == "Opening 1 paper in your browser..."
        assert build_open_papers_notification(2) == "Opening 2 papers in your browser..."
        assert build_open_pdfs_notification(1) == "Opening 1 PDF..."
        assert build_open_pdfs_notification(2) == "Opening 2 PDFs..."
        assert build_download_start_notification(1) == "Downloading 1 PDF..."
        assert build_download_start_notification(2) == "Downloading 2 PDFs..."


class TestQueryFilterHelpers:
    def test_get_query_tokens_trims_and_handles_empty(self):
        from arxiv_browser.query import get_query_tokens

        assert get_query_tokens("   ") == []
        tokens = get_query_tokens("  cat:cs.AI transformer  ")
        assert [token.value for token in tokens] == ["cs.AI", "transformer"]

    def test_remove_query_token_rebuilds_query_text(self):
        from arxiv_browser.query import remove_query_token

        assert remove_query_token("cat:cs.AI AND transformer", 0) == "transformer"
        assert remove_query_token("cat:cs.AI AND transformer", 2) == "cat:cs.AI"

    def test_execute_query_filter_returns_copy_for_empty_query(self, make_paper):
        from arxiv_browser.query import execute_query_filter

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        fuzzy_search = MagicMock(return_value=[])
        advanced_match = MagicMock(return_value=False)

        filtered, highlight_terms = execute_query_filter(
            "   ",
            papers,
            fuzzy_search=fuzzy_search,
            advanced_match=advanced_match,
        )

        assert [paper.arxiv_id for paper in filtered] == ["2401.00001", "2401.00002"]
        assert filtered is not papers
        assert highlight_terms == {"title": [], "author": [], "abstract": []}
        fuzzy_search.assert_not_called()
        advanced_match.assert_not_called()

    def test_execute_query_filter_uses_fuzzy_for_basic_query(self, make_paper):
        from arxiv_browser.query import execute_query_filter

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        fuzzy_result = [papers[1]]
        fuzzy_search = MagicMock(return_value=fuzzy_result)
        advanced_match = MagicMock(return_value=False)

        filtered, highlight_terms = execute_query_filter(
            " transformer ",
            papers,
            fuzzy_search=fuzzy_search,
            advanced_match=advanced_match,
        )

        assert filtered == fuzzy_result
        assert highlight_terms["title"] == ["transformer"]
        assert highlight_terms["author"] == ["transformer"]
        fuzzy_search.assert_called_once_with("transformer", papers)
        advanced_match.assert_not_called()

    def test_execute_query_filter_passes_input_papers_to_fuzzy_search(self, make_paper):
        from arxiv_browser.query import execute_query_filter

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        seen_scope: list[list[str]] = []

        def fuzzy_search(_query: str, scoped_papers):
            seen_scope.append([paper.arxiv_id for paper in scoped_papers])
            return [scoped_papers[-1]]

        filtered, _ = execute_query_filter(
            "transformer",
            papers,
            fuzzy_search=fuzzy_search,
            advanced_match=MagicMock(return_value=False),
        )

        assert seen_scope == [["2401.00001", "2401.00002"]]
        assert [paper.arxiv_id for paper in filtered] == ["2401.00002"]

    def test_execute_query_filter_uses_advanced_path(self, make_paper):
        from arxiv_browser.query import execute_query_filter

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        fuzzy_search = MagicMock(return_value=[])
        seen_rpn: list[list[str]] = []

        def advanced_match(paper, rpn):
            seen_rpn.append([token.value for token in rpn])
            return paper.arxiv_id == "2401.00001"

        filtered, highlight_terms = execute_query_filter(
            "cat:cs.AI AND transformer",
            papers,
            fuzzy_search=fuzzy_search,
            advanced_match=advanced_match,
        )

        assert [paper.arxiv_id for paper in filtered] == ["2401.00001"]
        assert highlight_terms["title"] == ["transformer"]
        assert highlight_terms["author"] == ["transformer"]
        assert len(seen_rpn) == 2
        assert seen_rpn[0] == ["cs.AI", "transformer", "AND"]
        fuzzy_search.assert_not_called()

    def test_apply_watch_filter_respects_toggle(self, make_paper):
        from arxiv_browser.query import apply_watch_filter

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        watched = {"2401.00002"}

        assert apply_watch_filter(papers, watched, False) is papers
        assert [paper.arxiv_id for paper in apply_watch_filter(papers, watched, True)] == [
            "2401.00002"
        ]

    def test_update_filter_pills_forwards_parsed_tokens(self):
        app = _new_app()
        app._in_arxiv_api_mode = False
        app._watch_filter_active = True
        app._track_task = MagicMock()
        pill_bar = MagicMock()
        app.query_one = MagicMock(return_value=pill_bar)

        app._update_filter_pills("  cat:cs.AI transformer  ")

        pill_bar.update_pills.assert_called_once()
        tokens, watch_filter_active = pill_bar.update_pills.call_args.args
        assert [token.field for token in tokens] == ["cat", None]
        assert [token.value for token in tokens] == ["cs.AI", "transformer"]
        assert watch_filter_active is True
        app._track_task.assert_called_once_with(pill_bar.update_pills.return_value)

    @pytest.mark.parametrize(
        ("query", "token_index", "expected"),
        [
            ("cat:cs.AI AND transformer", 0, "transformer"),
            ("cat:cs.AI AND transformer", 2, "cat:cs.AI"),
            ('title:"deep learning" OR author:Smith', 2, 'title:"deep learning"'),
        ],
    )
    def test_on_remove_filter_rebuilds_query_text(self, query, token_index, expected):
        app = _new_app()
        search_input = SimpleNamespace(value=query)
        app.query_one = MagicMock(return_value=search_input)

        app.on_remove_filter(SimpleNamespace(token_index=token_index))

        assert search_input.value == expected
