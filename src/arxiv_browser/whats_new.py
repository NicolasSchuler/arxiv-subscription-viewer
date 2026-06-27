"""Version-bump 'What's New' content and gating helpers.

The What's New overlay surfaces headline changes once per version to
keep long-time users aware of new capabilities without forcing them
to read CHANGELOG.md. Content is embedded in this module rather than
parsed from CHANGELOG.md so it stays stable across wheel builds and
editable installs alike, and can be curated for readability.

To ship a new release:

1. Update ``WHATS_NEW_VERSION`` to the release tag (e.g. ``"0.2.0"``).
2. Update ``WHATS_NEW_HEADLINE`` and ``WHATS_NEW_ENTRIES`` with the
   most user-visible changes (keep the list to 4-6 bullets).

The modal trusts the strings to contain Rich markup; only package
authors should edit them.
"""

from __future__ import annotations

from typing import Final

WHATS_NEW_VERSION: Final[str] = "0.3.3"
"""Tag of the release whose notes are currently presented.

When a user's ``UserConfig.last_seen_whats_new`` differs from this
value, the modal is shown on the next app start and this value is
stored back to config once they dismiss it.
"""

WHATS_NEW_HEADLINE: Final[str] = "What's New in arXiv Viewer"

WHATS_NEW_ENTRIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "Resizable panes",
        "Use [bold]Alt+Left[/] and [bold]Alt+Right[/] to give more room to "
        "the paper list or detail pane without leaving the TUI.",
    ),
    (
        "Persistent layout",
        "Your chosen list/detail split is saved as [bold]pane_split[/] and restored "
        "on the next launch.",
    ),
    (
        "Narrow-terminal support",
        "Pane resizing also maps cleanly to stacked list/detail heights on narrow terminal widths.",
    ),
    (
        "Updated help surfaces",
        "The README, cheat sheet, config reference, help overlay, and command palette "
        "now all include the pane resize controls.",
    ),
)


def should_show_whats_new(last_seen_version: str) -> bool:
    """Return ``True`` when the user has not yet seen the current notes.

    The check is a simple string inequality rather than semantic version
    comparison — any mismatch (including downgrades and ``""`` defaults)
    surfaces the modal. The modal only ever writes the current version
    back to config on dismiss, so repeat launches stay quiet.
    """
    return last_seen_version != WHATS_NEW_VERSION


__all__ = [
    "WHATS_NEW_ENTRIES",
    "WHATS_NEW_HEADLINE",
    "WHATS_NEW_VERSION",
    "should_show_whats_new",
]
