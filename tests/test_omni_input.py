"""Tests for OmniInput widget — mode prefix parsing and message routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App

from arxiv_browser.palette import PaletteCommand
from arxiv_browser.themes import THEME_COLORS, _build_textual_theme
from arxiv_browser.widgets.omni_input import (
    FUZZY_THRESHOLD,
    GROUP_ORDER,
    OMNI_HINT_API,
    OMNI_HINT_COMMAND,
    OMNI_HINT_LOCAL,
    OMNI_HINT_SEMANTIC,
    OMNI_PLACEHOLDER,
    OmniInput,
    OmniMode,
    _group_order_index,
    parse_omni_mode,
)


class _ThemedApp(App):
    """Test app exposing the ``$th-*`` CSS variables OmniInput's CSS references.

    Mirrors what ``ArxivBrowser`` registers at runtime so bare-widget tests
    resolve theme tokens (e.g. ``$th-muted``) instead of erroring on an
    undefined variable.
    """

    def get_css_variables(self) -> dict[str, str]:
        variables = dict(super().get_css_variables())
        variables.update(_build_textual_theme("test", dict(THEME_COLORS)).variables)
        return variables


class TestParseOmniMode:
    """Unit tests for parse_omni_mode."""

    def test_plain_text_is_local(self):
        result = parse_omni_mode("cat:cs.AI")
        assert result == OmniMode(mode="local", query="cat:cs.AI")

    def test_empty_is_local(self):
        result = parse_omni_mode("")
        assert result == OmniMode(mode="local", query="")

    def test_command_prefix(self):
        result = parse_omni_mode(">open")
        assert result == OmniMode(mode="command", query="open")

    def test_command_prefix_with_space(self):
        result = parse_omni_mode("> export")
        assert result == OmniMode(mode="command", query="export")

    def test_command_prefix_only(self):
        result = parse_omni_mode(">")
        assert result == OmniMode(mode="command", query="")

    def test_api_prefix(self):
        result = parse_omni_mode("@transformer attention")
        assert result == OmniMode(mode="api", query="transformer attention")

    def test_api_prefix_with_space(self):
        result = parse_omni_mode("@ neural networks")
        assert result == OmniMode(mode="api", query="neural networks")

    def test_api_prefix_only(self):
        result = parse_omni_mode("@")
        assert result == OmniMode(mode="api", query="")

    def test_semantic_prefix(self):
        result = parse_omni_mode("~papers about RAG hallucination")
        assert result == OmniMode(mode="semantic", query="papers about RAG hallucination")

    def test_semantic_prefix_with_space(self):
        result = parse_omni_mode("~ papers about RAG hallucination")
        assert result == OmniMode(mode="semantic", query="papers about RAG hallucination")

    def test_semantic_prefix_only(self):
        result = parse_omni_mode("~")
        assert result == OmniMode(mode="semantic", query="")

    def test_greater_than_mid_text_is_local(self):
        """A > not at position 0 is just local search."""
        result = parse_omni_mode("foo > bar")
        assert result == OmniMode(mode="local", query="foo > bar")

    def test_at_mid_text_is_local(self):
        """An @ not at position 0 is just local search."""
        result = parse_omni_mode("author@university")
        assert result == OmniMode(mode="local", query="author@university")


class TestOmniModeDataclass:
    """Smoke tests for OmniMode slots and equality."""

    def test_equality(self):
        a = OmniMode(mode="local", query="test")
        b = OmniMode(mode="local", query="test")
        assert a == b

    def test_inequality(self):
        a = OmniMode(mode="local", query="test")
        b = OmniMode(mode="command", query="test")
        assert a != b


class TestConstants:
    """Verify module-level constants exist and are sensible."""

    def test_placeholder_mentions_modes(self):
        assert ">" in OMNI_PLACEHOLDER
        assert "@" in OMNI_PLACEHOLDER

    def test_hints_are_strings(self):
        assert isinstance(OMNI_HINT_LOCAL, str)
        assert isinstance(OMNI_HINT_COMMAND, str)
        assert isinstance(OMNI_HINT_API, str)
        assert isinstance(OMNI_HINT_SEMANTIC, str)

    def test_fuzzy_threshold_is_positive(self):
        assert FUZZY_THRESHOLD > 0


class TestOmniInputWidget:
    """Unit tests for OmniInput widget methods (no TUI)."""

    def _make_commands(self) -> list[PaletteCommand]:
        return [
            PaletteCommand(
                name="Open in Browser",
                description="Open selected paper",
                key_hint="o",
                action="open_url",
                group="Core",
                enabled=True,
            ),
            PaletteCommand(
                name="Toggle Star",
                description="Star or unstar paper",
                key_hint="x",
                action="toggle_star",
                group="Core",
                enabled=True,
            ),
            PaletteCommand(
                name="Export Menu",
                description="Export papers in various formats",
                key_hint="E",
                action="export_menu",
                group="Core",
                enabled=False,
                blocked_reason="No papers loaded",
            ),
        ]

    def test_set_commands(self):
        widget = OmniInput()
        cmds = self._make_commands()
        widget.set_commands(cmds)
        assert widget._commands is cmds

    def test_initial_mode_is_local(self):
        widget = OmniInput()
        assert widget._current_mode == "local"

    def test_command_match_searches_group_key_and_action(self):
        widget = OmniInput()
        command = PaletteCommand(
            name="Open",
            description="Open paper",
            key_hint="Ctrl+k",
            action="collections",
            group="Organize",
        )
        assert widget._command_match("organize", command) is not None
        assert widget._command_match("ctrl+k", command) is not None
        assert widget._command_match("collections", command) is not None

    def _multi_group_commands(self) -> list[PaletteCommand]:
        """Commands whose authored order interleaves groups (as in the real registry)."""

        def cmd(name: str, group: str) -> PaletteCommand:
            return PaletteCommand(
                name=name, description=name, key_hint="", action=name, group=group
            )

        return [
            cmd("a-research", "Research"),
            cmd("b-core", "Core"),
            cmd("c-organize", "Organize"),
            cmd("d-core", "Core"),
            cmd("e-advanced", "Advanced"),
            cmd("f-research", "Research"),
            cmd("g-unknown", "Mystery"),
        ]

    def test_group_order_index_ranks_known_groups(self):
        assert _group_order_index("Core") < _group_order_index("Organize")
        assert _group_order_index("Organize") < _group_order_index("Research")
        assert _group_order_index("Research") < _group_order_index("Advanced")
        # Unknown groups sort after every canonical group.
        assert _group_order_index("Mystery") == len(GROUP_ORDER)

    def test_filter_commands_no_query_sorts_by_group_stably(self):
        widget = OmniInput()
        widget.set_commands(self._multi_group_commands())
        ordered = widget._filter_commands("")
        groups = [c.group for c in ordered]
        # Groups appear contiguously in canonical order, unknown group last.
        assert groups == [
            "Core",
            "Core",
            "Organize",
            "Research",
            "Research",
            "Advanced",
            "Mystery",
        ]
        # Authored within-group order preserved (stable sort).
        core_names = [c.name for c in ordered if c.group == "Core"]
        assert core_names == ["b-core", "d-core"]


@pytest.mark.asyncio
class TestOmniInputTUI:
    """Integration tests requiring a running Textual app."""

    async def test_open_close_visibility(self):

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            assert not omni.is_open
            omni.open()
            assert omni.is_open
            omni.close()
            assert not omni.is_open

    async def test_open_with_initial_text(self):
        from textual.widgets import Input

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open(">export")
            assert omni.is_open
            inp = omni.query_one("#omni-input", Input)
            assert inp.value == ">export"

    async def test_local_search_emits_message(self):
        from textual.widgets import Input

        messages: list[OmniInput.LocalSearch] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_local_search(self, msg: OmniInput.LocalSearch):
                messages.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open()
            inp = omni.query_one("#omni-input", Input)
            inp.value = "cat:cs.AI"
            await pilot.pause()
            assert any(m.query == "cat:cs.AI" for m in messages)

    async def test_command_mode_shows_results(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            cmds = [
                PaletteCommand(
                    name="Open",
                    description="Open paper",
                    key_hint="o",
                    action="open_url",
                    group="Core",
                ),
            ]
            omni.set_commands(cmds)
            omni.open(">")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">"
            await pilot.pause()
            results = omni.query_one("#omni-results", OptionList)
            assert results.has_class("visible")
            assert results.option_count > 0

    async def test_command_mode_no_matches_shows_disabled_empty_result(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Open",
                        description="Open paper",
                        key_hint="o",
                        action="open_url",
                        group="Core",
                    ),
                ]
            )
            omni.open(">zzzzzz")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">zzzzzz"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            assert results.has_class("visible")
            # §7 empty state: the "No matching" line plus Try: and Next: guidance.
            assert results.option_count == 3
            empty_prompt = str(results.get_option_at_index(0).prompt)
            assert "No matching commands" in empty_prompt
            assert "zzzzzz" in empty_prompt
            try_prompt = str(results.get_option_at_index(1).prompt)
            assert "Try:" in try_prompt
            next_prompt = str(results.get_option_at_index(2).prompt)
            assert "Next:" in next_prompt
            assert omni._filtered_commands == []

    @staticmethod
    def _grouped_commands() -> list[PaletteCommand]:
        def cmd(name: str, group: str) -> PaletteCommand:
            return PaletteCommand(
                name=name, description=name, key_hint="", action=name, group=group
            )

        return [
            cmd("alpha", "Research"),
            cmd("bravo", "Core"),
            cmd("charlie", "Organize"),
            cmd("delta", "Core"),
            cmd("echo", "Research"),
        ]

    async def test_no_query_group_headers_appear_at_most_once(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(self._grouped_commands())
            omni.open(">")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            # Header rows are the options not mapped to a command (no empty state here).
            header_indexes = [
                i for i in range(results.option_count) if i not in omni._command_option_indexes
            ]
            distinct_groups = {c.group for c in omni._filtered_commands}
            # Exactly one header per distinct group — no repeats.
            assert len(header_indexes) == len(distinct_groups)

    async def test_query_suppresses_group_headers(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(self._grouped_commands())
            omni.open(">a")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">a"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            assert omni._filtered_commands  # query matched something
            # With a query every option maps to a command — zero header rows.
            header_indexes = [
                i for i in range(results.option_count) if i not in omni._command_option_indexes
            ]
            assert header_indexes == []

    async def test_disabled_command_explains_blocker(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Export",
                        description="Export papers",
                        key_hint="E",
                        action="export_menu",
                        group="Core",
                        enabled=False,
                        blocked_reason="No papers loaded",
                    ),
                ]
            )
            omni.open(">export")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">export"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            prompt = str(results.get_option_at_index(0).prompt)
            assert "Requires: No papers loaded" in prompt

    async def test_suggested_command_label_uses_compact_marker(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Open",
                        description="Open paper",
                        key_hint="o",
                        action="open_url",
                        group="Core",
                        suggested=True,
                    ),
                ]
            )
            omni.open(">open")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">open"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            prompt = str(results.get_option_at_index(0).prompt)
            # Row layout: key hint, then compact "*" suggested marker, then name.
            assert "*" in prompt
            assert "Open" in prompt
            assert "o" in prompt
            # The old trailing "suggested" prose is gone.
            assert "suggested" not in prompt

    async def test_command_row_is_single_line_key_name_desc(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Search Papers",
                        description="Filter papers by text, category, or tag",
                        key_hint="/",
                        action="toggle_search",
                        group="Core",
                    ),
                ]
            )
            omni.open(">search")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">search"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            prompt = str(results.get_option_at_index(0).prompt)
            # One command per line: no newlines, key hint + name + (truncated) desc.
            assert "\n" not in prompt
            assert "/" in prompt
            assert "Search Papers" in prompt
            # Long description is ellipsis-truncated (ASCII "..."), never wrapped.
            assert "Filter papers by text" in prompt
            assert "..." in prompt
            assert "category, or tag" not in prompt

    async def test_command_row_ascii_only(self):
        import re

        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Open in Browser",
                        description="Open selected paper(s) in web browser",
                        key_hint="o",
                        action="open_url",
                        group="Core",
                        suggested=True,
                    ),
                ]
            )
            omni.open(">")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">"
            await pilot.pause()

            results = omni.query_one("#omni-results", OptionList)
            for index in range(results.option_count):
                prompt = str(results.get_option_at_index(index).prompt)
                assert not re.search(r"[^\x00-\x7f]", prompt)

    async def test_command_mode_arrow_keys_move_highlight(self):
        from textual.widgets import Input, OptionList

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Open",
                        description="Open paper",
                        key_hint="o",
                        action="open_url",
                        group="Core",
                    ),
                    PaletteCommand(
                        name="Toggle Star",
                        description="Star paper",
                        key_hint="x",
                        action="toggle_star",
                        group="Core",
                    ),
                ]
            )
            omni.open(">")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">"
            await pilot.pause()
            results = omni.query_one("#omni-results", OptionList)
            results.highlighted = None

            await pilot.press("down")
            assert results.highlighted == 0
            await pilot.press("down")
            assert results.highlighted == 1
            await pilot.press("up")
            assert results.highlighted == 0

    async def test_command_mode_group_headers_are_not_selectable(self):
        from textual.widgets import Input, OptionList

        messages: list[OmniInput.CommandSelected] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_command_selected(self, msg: OmniInput.CommandSelected):
                messages.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.set_commands(
                [
                    PaletteCommand(
                        name="Open",
                        description="Open paper",
                        key_hint="o",
                        action="open_url",
                        group="Core",
                    ),
                    PaletteCommand(
                        name="Edit Tags",
                        description="Tag paper",
                        key_hint="t",
                        action="edit_tags",
                        group="Organize",
                    ),
                ]
            )
            omni.open(">")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">"
            await pilot.pause()
            results = omni.query_one("#omni-results", OptionList)

            assert results.option_count == 4
            assert "Core" in str(results.get_option_at_index(0).prompt)
            assert "Organize" in str(results.get_option_at_index(2).prompt)

            results.highlighted = None
            await pilot.press("down")
            assert results.highlighted == 1
            await pilot.press("down")
            assert results.highlighted == 3

            await inp.action_submit()
            await pilot.pause()
            assert [message.action for message in messages] == ["edit_tags"]

    async def test_api_mode_emits_on_enter(self):
        from textual.widgets import Input

        messages: list[OmniInput.ApiSearch] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_api_search(self, msg: OmniInput.ApiSearch):
                messages.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open("@transformer")
            inp = omni.query_one("#omni-input", Input)
            inp.value = "@transformer"
            await pilot.pause()
            # Simulate Enter
            await inp.action_submit()
            await pilot.pause()
            assert any(m.query == "transformer" for m in messages)

    async def test_semantic_mode_emits_local_search_with_prefix(self):
        from textual.widgets import Input

        messages: list[OmniInput.LocalSearch] = []
        submissions: list[OmniInput.LocalSearchSubmitted] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_local_search(self, msg: OmniInput.LocalSearch):
                messages.append(msg)

            def on_omni_input_local_search_submitted(self, msg: OmniInput.LocalSearchSubmitted):
                submissions.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open("~RAG")
            inp = omni.query_one("#omni-input", Input)
            inp.value = "~RAG"
            await pilot.pause()
            await inp.action_submit()
            await pilot.pause()
            assert any(m.query == "~ RAG" for m in messages)
            assert any(m.query == "~ RAG" for m in submissions)

    async def test_command_select_emits_message(self):
        from textual.widgets import Input

        messages: list[OmniInput.CommandSelected] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_command_selected(self, msg: OmniInput.CommandSelected):
                messages.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            cmds = [
                PaletteCommand(
                    name="Toggle Star",
                    description="Star paper",
                    key_hint="x",
                    action="toggle_star",
                    group="Core",
                ),
            ]
            omni.set_commands(cmds)
            omni.open(">star")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">star"
            await pilot.pause()
            # Simulate Enter to select
            await inp.action_submit()
            await pilot.pause()
            assert any(m.action == "toggle_star" for m in messages)

    async def test_hint_updates_per_mode(self):
        from textual.widgets import Input

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open()
            inp = omni.query_one("#omni-input", Input)

            # Local mode
            inp.value = "test"
            await pilot.pause()
            assert omni._current_mode == "local"

            # Command mode
            inp.value = ">cmd"
            await pilot.pause()
            assert omni._current_mode == "command"

            # API mode
            inp.value = "@query"
            await pilot.pause()
            assert omni._current_mode == "api"

            # Semantic mode
            inp.value = "~query"
            await pilot.pause()
            assert omni._current_mode == "semantic"

            # Back to local
            inp.value = "plain"
            await pilot.pause()
            assert omni._current_mode == "local"

    async def test_ascii_mode_uses_ascii_safe_hints(self):
        from textual.widgets import Input, Static

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        with patch("arxiv_browser._ascii.is_ascii_mode", return_value=True):
            async with TestApp().run_test() as pilot:
                omni = pilot.app.query_one(OmniInput)
                omni.open(">")
                inp = omni.query_one("#omni-input", Input)
                hint = omni.query_one("#omni-hint", Static)

                assert all(ord(ch) < 128 for ch in inp.placeholder)
                assert all(ord(ch) < 128 for ch in str(hint.content))

    async def test_close_clears_input(self):
        from textual.widgets import Input

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open(">test")
            await pilot.pause()
            omni.close()
            inp = omni.query_one("#omni-input", Input)
            assert inp.value == ""
            assert not omni.is_open

    async def test_disabled_command_not_selected(self):
        from textual.widgets import Input

        messages: list[OmniInput.CommandSelected] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_command_selected(self, msg: OmniInput.CommandSelected):
                messages.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            cmds = [
                PaletteCommand(
                    name="Export",
                    description="Export papers",
                    key_hint="E",
                    action="export_menu",
                    group="Core",
                    enabled=False,
                ),
            ]
            omni.set_commands(cmds)
            omni.open(">export")
            inp = omni.query_one("#omni-input", Input)
            inp.value = ">export"
            await pilot.pause()
            # Try to select disabled command
            await inp.action_submit()
            await pilot.pause()
            assert len(messages) == 0

    async def test_empty_api_query_no_emit(self):
        from textual.widgets import Input

        messages: list[OmniInput.ApiSearch] = []

        class TestApp(_ThemedApp):
            def compose(self):
                yield OmniInput()

            def on_omni_input_api_search(self, msg: OmniInput.ApiSearch):
                messages.append(msg)

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open("@")
            inp = omni.query_one("#omni-input", Input)
            inp.value = "@"
            await pilot.pause()
            await inp.action_submit()
            await pilot.pause()
            assert len(messages) == 0
