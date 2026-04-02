"""Coverage for io_actions helpers: resolve, filter, clipboard, export, platform."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_browser.io_actions import (
    build_clipboard_payload,
    build_markdown_export_document,
    build_viewer_args,
    filter_papers_needing_download,
    get_clipboard_command_plan,
    resolve_target_papers,
    write_timestamped_export_file,
)
from tests.support.canonical_exports import Paper


def _paper(arxiv_id: str = "2401.12345", title: str = "Test") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        date="Mon, 15 Jan 2024",
        title=title,
        authors="Author",
        categories="cs.AI",
        comments=None,
        abstract="abstract",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        abstract_raw="abstract",
    )


class TestResolveTargetPapers:
    def test_returns_selected_papers_in_filtered_order(self):
        a, b, c = _paper("A"), _paper("B"), _paper("C")
        result = resolve_target_papers(
            filtered_papers=[c, a, b],
            selected_ids={"A", "B"},
            papers_by_id={},
            current_paper=None,
        )
        assert [p.arxiv_id for p in result] == ["A", "B"]

    def test_includes_remaining_ids_from_papers_by_id(self):
        a = _paper("A")
        extra = _paper("Z")
        result = resolve_target_papers(
            filtered_papers=[a],
            selected_ids={"A", "Z"},
            papers_by_id={"Z": extra},
            current_paper=None,
        )
        assert [p.arxiv_id for p in result] == ["A", "Z"]

    def test_skips_ids_not_found_in_papers_by_id(self):
        a = _paper("A")
        result = resolve_target_papers(
            filtered_papers=[a],
            selected_ids={"A", "MISSING"},
            papers_by_id={},
            current_paper=None,
        )
        assert [p.arxiv_id for p in result] == ["A"]

    def test_falls_back_to_current_paper_when_no_selection(self):
        paper = _paper("current")
        result = resolve_target_papers(
            filtered_papers=[],
            selected_ids=set(),
            papers_by_id={},
            current_paper=paper,
        )
        assert result == [paper]

    def test_returns_empty_when_nothing_available(self):
        result = resolve_target_papers(
            filtered_papers=[],
            selected_ids=set(),
            papers_by_id={},
            current_paper=None,
        )
        assert result == []

    def test_selected_ids_take_priority_over_current_paper(self):
        a = _paper("A")
        current = _paper("current")
        result = resolve_target_papers(
            filtered_papers=[a],
            selected_ids={"A"},
            papers_by_id={},
            current_paper=current,
        )
        assert len(result) == 1
        assert result[0].arxiv_id == "A"


class TestFilterPapersNeedingDownload:
    def test_splits_into_download_and_skipped(self, tmp_path):
        downloaded = _paper("dl")
        pending = _paper("pend")
        dl_path = tmp_path / "dl.pdf"
        dl_path.write_bytes(b"%PDF content")
        pending_path = tmp_path / "pend.pdf"

        def get_path(p: Paper) -> Path:
            return dl_path if p.arxiv_id == "dl" else pending_path

        to_dl, skipped = filter_papers_needing_download([downloaded, pending], get_path)
        assert [p.arxiv_id for p in to_dl] == ["pend"]
        assert skipped == ["dl"]

    def test_skips_zero_byte_file(self, tmp_path):
        paper = _paper("zero")
        zero_path = tmp_path / "zero.pdf"
        zero_path.write_bytes(b"")

        to_dl, skipped = filter_papers_needing_download([paper], lambda p: zero_path)
        assert len(to_dl) == 1
        assert skipped == []


class TestBuildMarkdownExportDocument:
    def test_builds_document_with_papers(self):
        md1 = "## Paper 1\nAbstract 1"
        md2 = "## Paper 2\nAbstract 2"
        doc = build_markdown_export_document([md1, md2])
        assert "# arXiv Papers Export" in doc
        assert "*Exported 2 paper(s)*" in doc
        assert "## Paper 1" in doc
        assert "---" in doc
        assert "## Paper 2" in doc

    def test_empty_papers(self):
        doc = build_markdown_export_document([])
        assert "*Exported 0 paper(s)*" in doc


class TestBuildClipboardPayload:
    def test_joins_with_separator(self):
        result = build_clipboard_payload(["entry1", "entry2"], "=" * 80)
        assert "entry1" in result
        assert "entry2" in result
        assert ("=" * 80) in result

    def test_single_entry(self):
        result = build_clipboard_payload(["only"], "SEP")
        assert result == "only"


class TestGetClipboardCommandPlan:
    def test_macos(self):
        result = get_clipboard_command_plan("Darwin")
        assert result is not None
        cmds, enc = result
        assert cmds == [["pbcopy"]]
        assert enc == "utf-8"

    def test_linux(self):
        result = get_clipboard_command_plan("Linux")
        assert result is not None
        cmds, enc = result
        assert cmds[0] == ["xclip", "-selection", "clipboard"]
        assert cmds[1] == ["xsel", "--clipboard", "--input"]
        assert enc == "utf-8"

    def test_windows(self):
        result = get_clipboard_command_plan("Windows")
        assert result is not None
        cmds, enc = result
        assert cmds == [["clip"]]
        assert enc == "utf-16"

    def test_unknown_returns_none(self):
        assert get_clipboard_command_plan("FreeBSD") is None


class TestWriteTimestampedExportFile:
    def test_writes_file_with_content(self, tmp_path):
        export_dir = tmp_path / "exports"
        result = write_timestamped_export_file(
            content="test content",
            export_dir=export_dir,
            extension="md",
            now=datetime(2024, 3, 15, 10, 30, 0),
        )
        assert result.exists()
        assert result.read_text() == "test content"
        assert "arxiv-2024-03-15_103000.md" in result.name

    def test_creates_export_dir(self, tmp_path):
        export_dir = tmp_path / "nested" / "dir"
        write_timestamped_export_file(
            content="x",
            export_dir=export_dir,
            extension="txt",
            now=datetime(2024, 1, 1),
        )
        assert export_dir.is_dir()

    def test_uses_now_over_datetime_now(self, tmp_path):
        fixed = datetime(2025, 6, 1, 12, 0, 0)
        result = write_timestamped_export_file(
            content="c",
            export_dir=tmp_path,
            extension="csv",
            now=fixed,
        )
        assert "2025-06-01_120000" in result.name

    def test_cleanup_on_write_failure(self, tmp_path):
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        with (
            patch("os.write", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            write_timestamped_export_file(
                content="fail",
                export_dir=export_dir,
                extension="md",
            )
        tmp_files = list(export_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestBuildViewerArgsPosix:
    def test_appends_url_when_no_placeholder(self, monkeypatch):
        monkeypatch.setattr("arxiv_browser.io_actions.os.name", "posix")
        args = build_viewer_args("firefox", "https://arxiv.org/abs/2401.12345")
        assert args == ["firefox", "https://arxiv.org/abs/2401.12345"]

    def test_replaces_url_placeholder(self, monkeypatch):
        monkeypatch.setattr("arxiv_browser.io_actions.os.name", "posix")
        args = build_viewer_args("firefox {url}", "https://arxiv.org/abs/2401.12345")
        assert args == ["firefox", "https://arxiv.org/abs/2401.12345"]

    def test_replaces_path_placeholder(self, monkeypatch):
        monkeypatch.setattr("arxiv_browser.io_actions.os.name", "posix")
        args = build_viewer_args("zathura {path}", "/tmp/paper.pdf")
        assert args == ["zathura", "/tmp/paper.pdf"]
