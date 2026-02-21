"""Helpers for export/download/clipboard action bookkeeping."""

from __future__ import annotations

import os
import shlex
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from arxiv_browser.models import Paper


def resolve_target_papers(
    *,
    filtered_papers: Sequence[Paper],
    selected_ids: set[str],
    papers_by_id: Mapping[str, Paper],
    current_paper: Paper | None,
) -> list[Paper]:
    """Resolve action target papers from selected IDs or current paper fallback."""
    if selected_ids:
        ordered: list[Paper] = []
        seen: set[str] = set()
        for paper in filtered_papers:
            if paper.arxiv_id in selected_ids:
                ordered.append(paper)
                seen.add(paper.arxiv_id)
        remaining_ids = sorted(arxiv_id for arxiv_id in selected_ids if arxiv_id not in seen)
        for arxiv_id in remaining_ids:
            paper = papers_by_id.get(arxiv_id)
            if paper is not None:
                ordered.append(paper)
        return ordered
    if current_paper is not None:
        return [current_paper]
    return []


def filter_papers_needing_download(
    papers: Sequence[Paper],
    get_download_path: Callable[[Paper], Path],
) -> tuple[list[Paper], list[str]]:
    """Split papers into pending downloads and already-downloaded IDs."""
    to_download: list[Paper] = []
    skipped_ids: list[str] = []
    for paper in papers:
        path = get_download_path(paper)
        if path.exists() and path.stat().st_size > 0:
            skipped_ids.append(paper.arxiv_id)
        else:
            to_download.append(paper)
    return to_download, skipped_ids


def build_markdown_export_document(formatted_papers: Sequence[str]) -> str:
    """Build a markdown export document from per-paper markdown snippets."""
    lines = ["# arXiv Papers Export", "", f"*Exported {len(formatted_papers)} paper(s)*", ""]
    for paper_markdown in formatted_papers:
        lines.extend([paper_markdown, "", "---", ""])
    return "\n".join(lines)


def build_viewer_args(viewer_cmd: str, url_or_path: str) -> list[str]:
    """Build subprocess argument list for a configured external viewer command."""
    args = shlex.split(viewer_cmd, posix=os.name != "nt")
    if os.name == "nt":
        # Windows split keeps wrapping quotes when posix=False.
        args = [
            arg[1:-1] if len(arg) >= 2 and arg.startswith('"') and arg.endswith('"') else arg
            for arg in args
        ]
    if not args:
        raise ValueError("Viewer command is empty")
    if "{url}" in viewer_cmd or "{path}" in viewer_cmd:
        return [arg.replace("{url}", url_or_path).replace("{path}", url_or_path) for arg in args]
    return [*args, url_or_path]


def build_clipboard_payload(entries: Sequence[str], separator_line: str) -> str:
    """Join clipboard entries with a standard visual separator line."""
    separator = f"\n\n{separator_line}\n\n"
    return separator.join(entries)


def get_clipboard_command_plan(system: str) -> tuple[list[list[str]], str] | None:
    """Return clipboard command candidates and input encoding for a platform."""
    if system == "Darwin":
        return ([["pbcopy"]], "utf-8")
    if system == "Linux":
        return ([["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]], "utf-8")
    if system == "Windows":
        return ([["clip"]], "utf-16")
    return None


def write_timestamped_export_file(
    *,
    content: str,
    export_dir: Path,
    extension: str,
    now: datetime | None = None,
) -> Path:
    """Write export content using atomic temp-file replacement."""
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    filename = f"arxiv-{timestamp}.{extension}"
    filepath = export_dir / filename
    export_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=export_dir, suffix=".tmp", prefix=f".{extension}-")
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp_path, filepath)
    except BaseException:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return filepath


__all__ = [
    "build_clipboard_payload",
    "build_markdown_export_document",
    "build_viewer_args",
    "filter_papers_needing_download",
    "get_clipboard_command_plan",
    "resolve_target_papers",
    "write_timestamped_export_file",
]
