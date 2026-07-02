"""Contracts keeping the command palette, group labels, and help overlay in sync.

These guard three drifts fixed in the UX review:

- Every palette action has an explicit command-group (#20) — no silent "Commands"
  catch-all when a new action is added.
- Every palette action is reachable from the help overlay (#7).
- Keyless palette-only actions render a real description, not a raw identifier (#7).
"""

from __future__ import annotations

from arxiv_browser.browser.contracts import (
    COMMAND_PALETTE_COMMANDS,
    COMMAND_PALETTE_GROUPS,
)
from arxiv_browser.help_ui import (
    HELP_DESCRIPTION_OVERRIDES,
    HELP_PALETTE_ONLY_KEYS,
    HELP_SECTION_ACTIONS,
    _binding_for_help_action,
    build_help_sections,
)
from arxiv_browser.ui_constants import APP_BINDINGS

_PALETTE_ACTIONS = [action for _name, _desc, _key, action in COMMAND_PALETTE_COMMANDS]
_HELP_SECTION_ACTION_SET = {
    action for _section, actions in HELP_SECTION_ACTIONS for action in actions
}


def test_every_palette_action_has_explicit_group():
    """Adding a palette command must also assign it a canonical group (#20)."""
    missing = [a for a in _PALETTE_ACTIONS if a not in COMMAND_PALETTE_GROUPS]
    assert missing == [], f"palette actions missing an explicit group: {missing}"


def test_palette_groups_use_known_labels():
    """Group labels stay within the small canonical vocabulary."""
    from arxiv_browser.widgets.omni_input import GROUP_ORDER

    unknown = {g for g in COMMAND_PALETTE_GROUPS.values() if g not in GROUP_ORDER}
    assert unknown == set(), f"unexpected group labels: {unknown}"


def test_every_palette_action_is_help_reachable():
    """Each palette action appears in a help section and resolves to a row (#7)."""
    unreachable: list[str] = []
    for action in _PALETTE_ACTIONS:
        in_section = action in _HELP_SECTION_ACTION_SET
        renders = (
            _binding_for_help_action(APP_BINDINGS, action) is not None
            or action in HELP_PALETTE_ONLY_KEYS
        )
        if not (in_section and renders):
            unreachable.append(action)
    assert unreachable == [], f"palette actions not help-reachable: {unreachable}"


def test_keyless_palette_only_actions_have_descriptions():
    """Palette-only (keyless) actions must not render as raw identifiers (#7)."""
    for action in HELP_PALETTE_ONLY_KEYS:
        if _binding_for_help_action(APP_BINDINGS, action) is not None:
            continue  # keyed elsewhere; description comes from the binding
        assert action in HELP_DESCRIPTION_OVERRIDES, (
            f"keyless palette-only action {action!r} lacks a description override"
        )


def test_whats_new_is_documented_across_surfaces():
    """F1 / What's New is present in the palette, groups, and a help section (#7)."""
    assert "show_whats_new" in _PALETTE_ACTIONS
    assert "show_whats_new" in COMMAND_PALETTE_GROUPS
    assert "show_whats_new" in _HELP_SECTION_ACTION_SET


def test_settings_and_mark_visible_read_render_expected_keys():
    """Keyed actions newly added to help resolve their real key labels (#7)."""
    sections = build_help_sections(APP_BINDINGS)
    rows = {desc: key for _name, entries in sections for key, desc in entries}
    assert rows.get("Settings") == ","
    assert rows.get("Mark Visible Read") == "Ctrl+r"
    assert rows.get("What's New") == "F1"
