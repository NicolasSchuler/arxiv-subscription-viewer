"""Tests for OmniInput widget — mode prefix parsing and message routing."""

from __future__ import annotations

import pytest

from arxiv_browser.palette import PaletteCommand
from arxiv_browser.widgets.omni_input import (
    FUZZY_THRESHOLD,
    OMNI_HINT_API,
    OMNI_HINT_COMMAND,
    OMNI_HINT_LOCAL,
    OMNI_PLACEHOLDER,
    OmniInput,
    OmniMode,
    parse_omni_mode,
)


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


@pytest.mark.asyncio
class TestOmniInputTUI:
    """Integration tests requiring a running Textual app."""

    async def test_open_close_visibility(self):
        from textual.app import App

        class TestApp(App):
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
        from textual.app import App
        from textual.widgets import Input

        class TestApp(App):
            def compose(self):
                yield OmniInput()

        async with TestApp().run_test() as pilot:
            omni = pilot.app.query_one(OmniInput)
            omni.open(">export")
            assert omni.is_open
            inp = omni.query_one("#omni-input", Input)
            assert inp.value == ">export"

    async def test_local_search_emits_message(self):
        from textual.app import App
        from textual.widgets import Input

        messages: list[OmniInput.LocalSearch] = []

        class TestApp(App):
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
        from textual.app import App
        from textual.widgets import Input, OptionList

        class TestApp(App):
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
        from textual.app import App
        from textual.widgets import Input, OptionList

        class TestApp(App):
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
            assert results.option_count == 1
            assert omni._filtered_commands == []

    async def test_api_mode_emits_on_enter(self):
        from textual.app import App
        from textual.widgets import Input

        messages: list[OmniInput.ApiSearch] = []

        class TestApp(App):
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

    async def test_command_select_emits_message(self):
        from textual.app import App
        from textual.widgets import Input

        messages: list[OmniInput.CommandSelected] = []

        class TestApp(App):
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
        from textual.app import App
        from textual.widgets import Input

        class TestApp(App):
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

            # Back to local
            inp.value = "plain"
            await pilot.pause()
            assert omni._current_mode == "local"

    async def test_close_clears_input(self):
        from textual.app import App
        from textual.widgets import Input

        class TestApp(App):
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
        from textual.app import App
        from textual.widgets import Input

        messages: list[OmniInput.CommandSelected] = []

        class TestApp(App):
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
        from textual.app import App
        from textual.widgets import Input

        messages: list[OmniInput.ApiSearch] = []

        class TestApp(App):
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
