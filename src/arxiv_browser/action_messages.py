"""UI-facing copy builders for action confirmations and notifications."""

from __future__ import annotations


def _ensure_sentence(text: str) -> str:
    """Return text with terminal sentence punctuation."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return f"{cleaned}."


def build_actionable_error(
    action: str,
    *,
    next_step: str,
    why: str | None = None,
) -> str:
    """Build a 2-3 line actionable error message."""
    lines = [f"Could not {action.strip()}."]
    if why:
        lines.append(f"Why: {_ensure_sentence(why)}")
    lines.append(f"Next step: {_ensure_sentence(next_step)}")
    return "\n".join(lines)


def requires_batch_confirmation(item_count: int, threshold: int) -> bool:
    """Return whether an action should use a confirmation modal."""
    return item_count > threshold


def build_open_papers_confirmation_prompt(item_count: int) -> str:
    """Build confirmation prompt text for opening paper URLs."""
    return (
        f"Open {item_count} paper{'s' if item_count != 1 else ''} in your browser?\n"
        "This may open many tabs.\n\n"
        "[y] Confirm  [n/Esc] Cancel"
    )


def build_open_pdfs_confirmation_prompt(item_count: int) -> str:
    """Build confirmation prompt text for opening PDF URLs."""
    return (
        f"Open {item_count} PDF{'s' if item_count != 1 else ''} in your browser?\n"
        "This may open many tabs or viewer windows.\n\n"
        "[y] Confirm  [n/Esc] Cancel"
    )


def build_download_pdfs_confirmation_prompt(item_count: int) -> str:
    """Build confirmation prompt text for starting PDF downloads."""
    return (
        f"Download {item_count} PDF{'s' if item_count != 1 else ''}?\n"
        "Already-downloaded files will be skipped.\n\n"
        "[y] Download  [n/Esc] Cancel"
    )


def build_open_papers_notification(item_count: int) -> str:
    """Build notification text for opening paper URLs."""
    return f"Opening {item_count} paper{'s' if item_count > 1 else ''} in your browser..."


def build_open_pdfs_notification(item_count: int) -> str:
    """Build notification text for opening PDF URLs."""
    return f"Opening {item_count} PDF{'s' if item_count > 1 else ''}..."


def build_download_start_notification(item_count: int) -> str:
    """Build notification text for starting PDF downloads."""
    return f"Downloading {item_count} PDF{'s' if item_count != 1 else ''}..."


__all__ = [
    "build_actionable_error",
    "build_download_pdfs_confirmation_prompt",
    "build_download_start_notification",
    "build_open_papers_confirmation_prompt",
    "build_open_papers_notification",
    "build_open_pdfs_confirmation_prompt",
    "build_open_pdfs_notification",
    "requires_batch_confirmation",
]
