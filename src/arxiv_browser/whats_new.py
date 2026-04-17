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

WHATS_NEW_VERSION: Final[str] = "0.1.3"
"""Tag of the release whose notes are currently presented.

When a user's ``UserConfig.last_seen_whats_new`` differs from this
value, the modal is shown on the next app start and this value is
stored back to config once they dismiss it.
"""

WHATS_NEW_HEADLINE: Final[str] = "What's New in arXiv Viewer"

WHATS_NEW_ENTRIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "User theme overrides",
        "Drop a [bold]user.tcss[/] next to config.json to layer your own "
        "Textual CSS over the embedded stylesheet.",
    ),
    (
        "Session theme flag",
        "Launch with [bold]--theme high-contrast[/] (or any registered "
        "theme) to override config.json for the session only.",
    ),
    (
        "Actionable empty states",
        "Collections and citation graph list views now explain what to do "
        "when they are empty, instead of showing a blank panel.",
    ),
    (
        "First-run onboarding",
        "A welcome overlay now shows essential shortcuts on first launch. "
        "Press [bold]?[/] anytime for the full keybinding reference.",
    ),
    (
        "doctor & completions",
        "Run [bold]arxiv-viewer doctor[/] for environment checks and "
        "[bold]arxiv-viewer completions bash|zsh|fish[/] to install shell "
        "completions.",
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
