"""UI-facing copy builders for action confirmations and notifications."""

from __future__ import annotations


def requires_batch_confirmation(item_count: int, threshold: int) -> bool:
    """Return whether an action should use a confirmation modal."""
    return item_count > threshold


def build_open_papers_confirmation_prompt(item_count: int) -> str:
    """Build confirmation prompt text for opening paper URLs."""
    return f"Open {item_count} papers in browser?"


def build_open_pdfs_confirmation_prompt(item_count: int) -> str:
    """Build confirmation prompt text for opening PDF URLs."""
    return f"Open {item_count} PDFs in browser?"


def build_download_pdfs_confirmation_prompt(item_count: int) -> str:
    """Build confirmation prompt text for starting PDF downloads."""
    return f"Download {item_count} PDFs?"


def build_open_papers_notification(item_count: int) -> str:
    """Build notification text for opening paper URLs."""
    return f"Opening {item_count} paper{'s' if item_count > 1 else ''}"


def build_open_pdfs_notification(item_count: int) -> str:
    """Build notification text for opening PDF URLs."""
    return f"Opening {item_count} PDF{'s' if item_count > 1 else ''}"


def build_download_start_notification(item_count: int) -> str:
    """Build notification text for starting PDF downloads."""
    return f"Downloading {item_count} PDF{'s' if item_count != 1 else ''}..."


__all__ = [
    "build_download_pdfs_confirmation_prompt",
    "build_download_start_notification",
    "build_open_papers_confirmation_prompt",
    "build_open_papers_notification",
    "build_open_pdfs_confirmation_prompt",
    "build_open_pdfs_notification",
    "requires_batch_confirmation",
]
