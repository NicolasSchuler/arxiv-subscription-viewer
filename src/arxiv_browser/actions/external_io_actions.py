# ruff: noqa: UP037
# pyright: reportUndefinedVariable=false, reportAttributeAccessIssue=false
"""Extracted ArxivBrowser action handlers."""

from __future__ import annotations

import asyncio
import platform
import subprocess
import webbrowser as _webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from arxiv_browser.action_messages import build_actionable_error
from arxiv_browser.actions._runtime import (
    BATCH_CONFIRM_THRESHOLD,
    CLIPBOARD_SEPARATOR,
    DEFAULT_BIBTEX_EXPORT_DIR,
    DEFAULT_PDF_DOWNLOAD_DIR,
    MAX_CONCURRENT_DOWNLOADS,
    SUBPROCESS_TIMEOUT,
    build_clipboard_payload,
    build_download_pdfs_confirmation_prompt,
    build_download_start_notification,
    build_markdown_export_document,
    build_open_papers_confirmation_prompt,
    build_open_papers_notification,
    build_open_pdfs_confirmation_prompt,
    build_open_pdfs_notification,
    build_viewer_args,
    export_metadata,
    filter_papers_needing_download,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_paper_for_clipboard,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    get_clipboard_command_plan,
    get_paper_url,
    get_pdf_download_path,
    get_pdf_url,
    import_metadata,
    logger,
    requires_batch_confirmation,
    save_config,
    write_timestamped_export_file,
)
from arxiv_browser.modals import ConfirmModal, ExportMenuModal, MetadataSnapshotPickerModal
from arxiv_browser.models import Paper

# Compatibility patch surface for tests and callers that monkeypatch browser opening.
webbrowser = _webbrowser

if TYPE_CHECKING:
    from arxiv_browser.app import ArxivBrowser


def action_copy_bibtex(app: "ArxivBrowser") -> None:
    """Copy selected papers as BibTeX entries to clipboard."""
    papers = app._get_target_papers()
    if not papers:
        app.notify("No paper selected", title="BibTeX", severity="warning")
        return

    bibtex_entries = [format_paper_as_bibtex(p) for p in papers]
    bibtex_text = "\n\n".join(bibtex_entries)

    if app._copy_to_clipboard(bibtex_text):
        count = len(papers)
        app.notify(
            f"Copied {count} BibTeX entr{'ies' if count > 1 else 'y'}",
            title="BibTeX",
        )
    else:
        app.notify("Failed to copy to clipboard", title="BibTeX", severity="error")


def action_export_bibtex_file(app: "ArxivBrowser") -> None:
    """Export selected papers to a BibTeX file for Zotero import."""
    papers = app._get_target_papers()
    if not papers:
        app.notify("No paper selected", title="Export", severity="warning")
        return

    bibtex_entries = [format_paper_as_bibtex(p) for p in papers]
    content = "\n\n".join(bibtex_entries)
    app._export_to_file(content, "bib", "BibTeX")


def action_export_markdown(app: "ArxivBrowser") -> None:
    """Export selected papers as Markdown to clipboard."""
    papers = app._get_target_papers()
    if not papers:
        app.notify("No paper selected", title="Markdown", severity="warning")
        return

    markdown_text = build_markdown_export_document(
        [app._format_paper_as_markdown(paper) for paper in papers]
    )

    if app._copy_to_clipboard(markdown_text):
        count = len(papers)
        app.notify(
            f"Copied {count} paper{'s' if count > 1 else ''} as Markdown",
            title="Markdown",
        )
    else:
        app.notify("Failed to copy to clipboard", title="Markdown", severity="error")


def action_export_menu(app: "ArxivBrowser") -> None:
    """Open the unified export menu modal."""
    papers = app._get_target_papers()
    if not papers:
        app.notify("No paper selected", title="Export", severity="warning")
        return
    app.push_screen(
        ExportMenuModal(len(papers)),
        callback=lambda fmt: app._do_export(fmt, papers) if fmt else None,
    )


def _do_export(app: "ArxivBrowser", fmt: str, papers: list[Paper]) -> None:
    """Dispatch export based on format string from ExportMenuModal."""
    dispatch: dict[str, Callable[..., None]] = {
        "clipboard-plain": lambda: app.action_copy_selected(),
        "clipboard-bibtex": lambda: app.action_copy_bibtex(),
        "clipboard-markdown": lambda: app.action_export_markdown(),
        "clipboard-ris": lambda: app._export_clipboard_ris(papers),
        "clipboard-csv": lambda: app._export_clipboard_csv(papers),
        "clipboard-mdtable": lambda: app._export_clipboard_mdtable(papers),
        "file-bibtex": lambda: app.action_export_bibtex_file(),
        "file-ris": lambda: app._export_file_ris(papers),
        "file-csv": lambda: app._export_file_csv(papers),
    }
    handler = dispatch.get(fmt)
    if handler:
        handler()


def _get_export_dir(app: "ArxivBrowser") -> Path:
    """Return the configured export directory path."""
    return Path(
        app._config.bibtex_export_dir or Path.home() / DEFAULT_BIBTEX_EXPORT_DIR
    ).expanduser()


def _export_to_file(app: "ArxivBrowser", content: str, extension: str, format_name: str) -> None:
    """Write content to a timestamped file using atomic write."""
    export_dir = app._get_export_dir()
    try:
        filepath = write_timestamped_export_file(
            content=content,
            export_dir=export_dir,
            extension=extension,
        )
    except OSError as exc:
        app.notify(
            f"Failed to export {format_name}: {exc}",
            title=f"{format_name} Export",
            severity="error",
        )
        return
    app.notify(
        f"Exported to {filepath.resolve()}",
        title=f"{format_name} Export",
    )


def _export_clipboard_ris(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Copy selected papers as RIS entries to clipboard."""
    entries = []
    for paper in papers:
        abstract_text = app._get_abstract_text(paper, allow_async=False) or ""
        entries.append(format_paper_as_ris(paper, abstract_text))
    ris_text = "\n\n".join(entries)
    if app._copy_to_clipboard(ris_text):
        count = len(papers)
        app.notify(
            f"Copied {count} RIS entr{'ies' if count > 1 else 'y'}",
            title="RIS",
        )
    else:
        app.notify("Failed to copy to clipboard", title="RIS", severity="error")


def _export_clipboard_csv(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Copy selected papers as CSV to clipboard."""
    csv_text = format_papers_as_csv(papers, app._config.paper_metadata)
    if app._copy_to_clipboard(csv_text):
        count = len(papers)
        app.notify(
            f"Copied {count} paper{'s' if count > 1 else ''} as CSV",
            title="CSV",
        )
    else:
        app.notify("Failed to copy to clipboard", title="CSV", severity="error")


def _export_clipboard_mdtable(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Copy selected papers as a Markdown table to clipboard."""
    table_text = format_papers_as_markdown_table(papers)
    if app._copy_to_clipboard(table_text):
        count = len(papers)
        app.notify(
            f"Copied {count} paper{'s' if count > 1 else ''} as Markdown table",
            title="Markdown Table",
        )
    else:
        app.notify(
            "Failed to copy to clipboard",
            title="Markdown Table",
            severity="error",
        )


def _export_file_ris(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Export selected papers to an RIS file."""
    entries = []
    for paper in papers:
        abstract_text = app._get_abstract_text(paper, allow_async=False) or ""
        entries.append(format_paper_as_ris(paper, abstract_text))
    content = "\n\n".join(entries)
    app._export_to_file(content, "ris", "RIS")


def _export_file_csv(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Export selected papers to a CSV file."""
    content = format_papers_as_csv(papers, app._config.paper_metadata)
    app._export_to_file(content, "csv", "CSV")


def action_export_metadata(app: "ArxivBrowser") -> None:
    """Export all user metadata to a portable JSON file."""
    import json as _json

    data = export_metadata(app._config)
    content = _json.dumps(data, indent=2, ensure_ascii=False)
    app._export_to_file(content, "json", "Metadata")


def action_import_metadata(app: "ArxivBrowser") -> None:
    """Import metadata from a JSON file in the export directory."""

    export_dir = app._get_export_dir()
    json_files = _list_metadata_snapshots(export_dir)
    if not json_files:
        app.notify(
            f"No metadata snapshots found in {export_dir.resolve()}",
            title="Import",
            severity="warning",
        )
        return
    if len(json_files) == 1:
        _import_metadata_file(app, json_files[0])
        return

    app.push_screen(
        MetadataSnapshotPickerModal(json_files),
        callback=lambda filepath: _import_metadata_file(app, filepath) if filepath else None,
    )


def _list_metadata_snapshots(export_dir: Path) -> list[Path]:
    """Return metadata snapshots newest-first by modified time."""

    def sort_key(path: Path) -> tuple[float, str]:
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = float("-inf")
        return (modified, path.name)

    return sorted(export_dir.glob("arxiv-*.json"), key=sort_key, reverse=True)


def _import_metadata_file(app: "ArxivBrowser", filepath: Path) -> None:
    """Import metadata from a chosen JSON snapshot file."""
    import json as _json

    resolved_path = filepath.resolve()
    try:
        raw = filepath.read_text(encoding="utf-8")
        data = _json.loads(raw)
        papers_n, watch_n, bk_n, col_n = import_metadata(data, app._config)
    except (OSError, ValueError) as exc:
        app.notify(
            f"Import failed from {resolved_path}: {exc}",
            title="Import",
            severity="error",
        )
        return
    if not save_config(app._config):
        app.notify(
            f"Imported {resolved_path} but failed to save changes to disk",
            title="Import",
            severity="warning",
        )
    app._compute_watched_papers()
    app._refresh_list_view()
    parts = []
    if papers_n:
        parts.append(f"{papers_n} papers")
    if watch_n:
        parts.append(f"{watch_n} watch entries")
    if bk_n:
        parts.append(f"{bk_n} bookmarks")
    if col_n:
        parts.append(f"{col_n} collections")
    summary = ", ".join(parts) or "nothing new"
    app.notify(f"Imported {summary} from {resolved_path}", title="Import")


def _start_downloads(app: "ArxivBrowser") -> None:
    """Start download tasks up to the concurrency limit."""
    while app._download_queue and len(app._downloading) < MAX_CONCURRENT_DOWNLOADS:
        paper = app._download_queue.popleft()
        if paper.arxiv_id in app._downloading:
            continue
        app._downloading.add(paper.arxiv_id)
        # Downloads should continue even if the visible dataset changes.
        app._track_task(app._process_single_download(paper))


async def _process_single_download(app: "ArxivBrowser", paper: Paper) -> None:
    """Process a single download and update state."""
    try:
        success = await app._download_pdf_async(paper, app._http_client)
        app._download_results[paper.arxiv_id] = success
    except asyncio.CancelledError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Download failed for %s: %s", paper.arxiv_id, exc, exc_info=True)
        app._download_results[paper.arxiv_id] = False
    except Exception as exc:
        logger.warning(
            "Unexpected download failure for %s: %s",
            paper.arxiv_id,
            exc,
            exc_info=True,
        )
        app._download_results[paper.arxiv_id] = False
    finally:
        app._downloading.discard(paper.arxiv_id)
        if not getattr(app, "_shutting_down", False):
            # Update progress
            completed = len(app._download_results)
            total = app._download_total
            app._update_download_progress(completed, total)

            # Start more downloads if queue has items
            app._start_downloads()

            # Check if batch is complete
            if completed == total:
                app._finish_download_batch()


def _finish_download_batch(app: "ArxivBrowser") -> None:
    """Handle completion of a download batch."""
    if app._download_total <= 0:
        return

    successes = sum(1 for v in app._download_results.values() if v)
    failures = len(app._download_results) - successes

    # Get download directory for notification
    download_dir = app._config.pdf_download_dir or f"~/{DEFAULT_PDF_DOWNLOAD_DIR}"

    if failures == 0:
        app.notify(
            f"Downloaded {successes} PDF{'s' if successes != 1 else ''} to {download_dir}",
            title="Download Complete",
        )
    else:
        app.notify(
            f"Downloaded {successes}/{app._download_total} PDFs ({failures} failed)",
            title="Download Complete",
            severity="warning",
        )

    # Reset state
    app._download_results.clear()
    app._download_total = 0
    app._update_status_bar()


def action_open_url(app: "ArxivBrowser") -> None:
    """Open selected papers' URLs in the default browser."""
    papers = app._get_target_papers()
    if not papers:
        return
    if requires_batch_confirmation(len(papers), BATCH_CONFIRM_THRESHOLD):
        app.push_screen(
            ConfirmModal(build_open_papers_confirmation_prompt(len(papers))),
            lambda confirmed: app._do_open_urls(papers) if confirmed else None,
        )
    else:
        app._do_open_urls(papers)


def _do_open_urls(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Open the given papers' URLs in the browser."""
    for paper in papers:
        app._safe_browser_open(get_paper_url(paper, prefer_pdf=app._config.prefer_pdf_url))
    count = len(papers)
    app.notify(build_open_papers_notification(count), title="Browser")


def action_open_pdf(app: "ArxivBrowser") -> None:
    """Open selected papers' PDF URLs in the default browser."""
    papers = app._get_target_papers()
    if not papers:
        return
    if requires_batch_confirmation(len(papers), BATCH_CONFIRM_THRESHOLD):
        app.push_screen(
            ConfirmModal(build_open_pdfs_confirmation_prompt(len(papers))),
            lambda confirmed: app._do_open_pdfs(papers) if confirmed else None,
        )
    else:
        app._do_open_pdfs(papers)


def _do_open_pdfs(app: "ArxivBrowser", papers: list[Paper]) -> None:
    """Open the given papers' PDF URLs in the browser or configured viewer."""
    viewer = app._config.pdf_viewer.strip()
    if viewer and not app._ensure_pdf_viewer_trusted(
        viewer,
        lambda: app._do_open_pdfs(papers),
    ):
        return
    for paper in papers:
        url = get_pdf_url(paper)
        if viewer:
            app._open_with_viewer(viewer, url)
        else:
            app._safe_browser_open(url)
    count = len(papers)
    app.notify(build_open_pdfs_notification(count), title="PDF")


def _open_with_viewer(app: "ArxivBrowser", viewer_cmd: str, url_or_path: str) -> bool:
    """Open a URL/path with a configured external viewer command.

    The command template can use {url} or {path} as placeholders.
    If no placeholder is found, the URL is appended as an argument.
    """
    try:
        args = build_viewer_args(viewer_cmd, url_or_path)
        # User-configured local viewer command execution is an explicit feature.
        subprocess.Popen(  # nosec B603
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (ValueError, OSError) as e:
        logger.warning("Failed to open with viewer %r: %s", viewer_cmd, e)
        app.notify(
            build_actionable_error(
                "open the configured PDF viewer",
                why="the viewer command failed to launch",
                next_step="check pdf_viewer in config.json or use P to open in browser",
            ),
            title="PDF",
            severity="error",
            timeout=8,
        )
        return False


def action_download_pdf(app: "ArxivBrowser") -> None:
    """Download PDFs for selected papers (or current paper)."""
    if app._is_download_batch_active():
        app.notify("Download already in progress", title="Download", severity="warning")
        return

    papers_to_download = app._get_target_papers()
    if not papers_to_download:
        app.notify("No papers to download", title="Download", severity="warning")
        return

    to_download, skipped_ids = filter_papers_needing_download(
        papers_to_download,
        lambda paper: get_pdf_download_path(paper, app._config),
    )
    for arxiv_id in skipped_ids:
        logger.debug("Skipping %s: already downloaded", arxiv_id)

    if not to_download:
        app.notify("All PDFs already downloaded", title="Download")
        return

    if requires_batch_confirmation(len(to_download), BATCH_CONFIRM_THRESHOLD):
        app.push_screen(
            ConfirmModal(build_download_pdfs_confirmation_prompt(len(to_download))),
            lambda confirmed: app._do_start_downloads(to_download) if confirmed else None,
        )
    else:
        app._do_start_downloads(to_download)


def _do_start_downloads(app: "ArxivBrowser", to_download: list[Paper]) -> None:
    """Initialize and start batch PDF downloads."""
    if app._is_download_batch_active():
        app.notify("Download already in progress", title="Download", severity="warning")
        return

    # Initialize download batch
    app._download_queue.extend(to_download)
    app._download_total = len(to_download)
    app._download_results.clear()

    # Notify and start downloads
    app.notify(build_download_start_notification(len(to_download)), title="Download")
    app._start_downloads()


def _format_paper_for_clipboard(app: "ArxivBrowser", paper: Paper) -> str:
    """Format a paper's metadata for clipboard export."""
    abstract_text = app._get_abstract_text(paper, allow_async=False) or ""
    return format_paper_for_clipboard(paper, abstract_text)


def _copy_to_clipboard(app: "ArxivBrowser", text: str) -> bool:
    """Copy text to system clipboard. Returns True on success.

    Uses platform-specific clipboard tools with timeout protection.
    Logs failures at warning level for troubleshooting.
    """
    try:
        system = platform.system()
        plan = get_clipboard_command_plan(system)
        if plan is None:
            logger.warning("Clipboard copy failed: unsupported platform %s", system)
            return False
        commands, encoding = plan
        payload = text.encode(encoding)
        for index, command in enumerate(commands):
            try:
                subprocess.run(  # nosec B603
                    command,
                    input=payload,
                    check=True,
                    shell=False,
                    timeout=SUBPROCESS_TIMEOUT,
                )
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                if index == len(commands) - 1:
                    raise
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
    ) as e:
        logger.warning("Clipboard copy failed: %s", e)
        return False


def action_copy_selected(app: "ArxivBrowser") -> None:
    """Copy selected papers' metadata to clipboard."""
    papers_to_copy = app._get_target_papers()
    if not papers_to_copy:
        app.notify("No papers to copy", title="Copy", severity="warning")
        return

    formatted = build_clipboard_payload(
        [app._format_paper_for_clipboard(paper) for paper in papers_to_copy],
        CLIPBOARD_SEPARATOR,
    )

    # Copy to clipboard
    if app._copy_to_clipboard(formatted):
        count = len(papers_to_copy)
        app.notify(
            f"Copied {count} paper{'s' if count > 1 else ''} to clipboard",
            title="Copy",
        )
    else:
        app.notify(
            "Failed to copy to clipboard",
            title="Copy",
            severity="error",
        )
