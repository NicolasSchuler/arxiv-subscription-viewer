"""In-app settings: a settings dialog and an inline LLM preset picker.

These modals let users configure the app without quitting to hand-edit
``config.json``. ``LLMPresetPickerModal`` is shown the moment an LLM action is
invoked while unconfigured (replacing the old "edit config.json" dead-end);
``SettingsModal`` is a general editor reachable from the command palette and
help (LLM preset, theme, Semantic Scholar / HuggingFace toggles, and research
interests).
"""

from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ListItem, ListView, Select, Static, Switch

from arxiv_browser.llm import LLM_PRESETS
from arxiv_browser.modals.base import ModalBase


@dataclass(slots=True)
class SettingsResult:
    """Desired settings returned by ``SettingsModal`` for the caller to apply."""

    llm_preset: str
    theme_name: str
    s2_enabled: bool
    hf_enabled: bool
    research_interests: str


class _PresetItem(ListItem):
    """List row carrying an LLM preset key (empty string = clear/custom)."""

    def __init__(self, preset: str, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.preset = preset


class LLMPresetPickerModal(ModalBase[str | None]):
    """Pick an LLM preset inline when an LLM feature is used unconfigured.

    Dismisses with the chosen preset key (one of ``LLM_PRESETS``) or ``None``
    when cancelled.
    """

    BINDINGS = [
        Binding("enter", "choose", "Select"),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    #llm-preset-dialog {
        width: 64;
        max-width: 90%;
        height: auto;
    }

    #llm-preset-subtitle {
        color: $th-muted;
        margin-bottom: 1;
    }

    #llm-preset-list {
        height: auto;
        max-height: 12;
        background: $th-panel;
        border: none;
    }

    #llm-preset-footer {
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield the preset list with title, hint, and footer."""
        with Vertical(id="llm-preset-dialog", classes="modal-dialog"):
            yield Label("Configure an LLM provider", id="llm-preset-title", classes="modal-title")
            yield Static(
                "AI features need an LLM command. Pick a preset to use its CLI:",
                id="llm-preset-subtitle",
            )
            yield ListView(id="llm-preset-list")
            yield Static(
                "Select: Enter | Cancel: Esc/q", id="llm-preset-footer", classes="modal-footer"
            )

    def on_mount(self) -> None:
        """Populate the preset list (preset name + the command it runs)."""
        from arxiv_browser.themes import theme_colors_for

        muted = theme_colors_for(self)["muted"]
        list_view = self.query_one("#llm-preset-list", ListView)
        list_view.clear()
        for preset in sorted(LLM_PRESETS):
            command = LLM_PRESETS[preset]
            list_view.append(_PresetItem(preset, Label(f"{preset}  [{muted}]{command}[/]")))
        if list_view.children:
            list_view.index = 0
        list_view.focus()

    def action_choose(self) -> None:
        """Dismiss with the highlighted preset key."""
        list_view = self.query_one("#llm-preset-list", ListView)
        if isinstance(list_view.highlighted_child, _PresetItem):
            self.dismiss(list_view.highlighted_child.preset)
        else:
            self.dismiss(None)

    @on(ListView.Selected, "#llm-preset-list")
    def on_selected(self, event: ListView.Selected) -> None:
        """Dismiss with the clicked/selected preset key."""
        if isinstance(event.item, _PresetItem):
            self.dismiss(event.item.preset)


class SettingsModal(ModalBase[SettingsResult | None]):
    """In-app settings editor.

    Dismisses with a :class:`SettingsResult` to apply, or ``None`` on cancel.
    Research-interests editing is delegated to the caller via the returned
    ``research_interests`` value (the caller opens the dedicated editor).
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    #settings-dialog {
        width: 72;
        max-width: 90%;
        height: auto;
    }

    #settings-grid {
        height: auto;
    }

    .settings-row {
        height: 3;
        align-vertical: middle;
    }

    .settings-label {
        width: 24;
        content-align: left middle;
        height: 3;
    }

    .settings-row Select {
        width: 1fr;
    }

    #settings-interests-value {
        width: 1fr;
        color: $th-muted;
        content-align: left middle;
        height: 3;
    }

    #settings-buttons Button {
        margin-left: 1;
    }

    #settings-footer {
        margin-top: 1;
    }
    """

    def __init__(self, current: SettingsResult, theme_names: list[str]) -> None:
        """Initialise from the current settings and the available theme names."""
        super().__init__()
        self._current = current
        self._theme_names = theme_names
        self._interests = current.research_interests

    def compose(self) -> ComposeResult:
        """Yield the settings rows, action buttons, and footer."""
        preset_options = [("(none / custom command)", "")] + [
            (name, name) for name in sorted(LLM_PRESETS)
        ]
        theme_options = [(name, name) for name in self._theme_names]
        with Vertical(id="settings-dialog", classes="modal-dialog"):
            yield Label("Settings", id="settings-title", classes="modal-title")
            with Vertical(id="settings-grid"):
                with Horizontal(classes="settings-row"):
                    yield Label("LLM preset", classes="settings-label")
                    yield Select(
                        preset_options,
                        value=self._current.llm_preset,
                        allow_blank=False,
                        id="settings-llm-preset",
                    )
                with Horizontal(classes="settings-row"):
                    yield Label("Theme", classes="settings-label")
                    yield Select(
                        theme_options,
                        value=self._current.theme_name,
                        allow_blank=False,
                        id="settings-theme",
                    )
                with Horizontal(classes="settings-row"):
                    yield Label("Semantic Scholar", classes="settings-label")
                    yield Switch(value=self._current.s2_enabled, id="settings-s2")
                with Horizontal(classes="settings-row"):
                    yield Label("HuggingFace trending", classes="settings-label")
                    yield Switch(value=self._current.hf_enabled, id="settings-hf")
                with Horizontal(classes="settings-row"):
                    yield Label("Research interests", classes="settings-label")
                    yield Static(self._interests_summary(), id="settings-interests-value")
                    yield Button("Edit…", id="settings-interests-edit")
            with Horizontal(id="settings-buttons", classes="modal-buttons"):
                yield Button("Cancel", variant="default", id="settings-cancel")
                yield Button("Save (Ctrl+S)", variant="primary", id="settings-save")
            yield Static("Ctrl+S save | Esc cancel", id="settings-footer", classes="modal-footer")

    def _interests_summary(self) -> str:
        """Return a short one-line preview of the research interests text."""
        text = " ".join(self._interests.split())
        if not text:
            return "(not set)"
        return text if len(text) <= 40 else f"{text[:39]}…"

    @on(Button.Pressed, "#settings-interests-edit")
    def on_edit_interests(self) -> None:
        """Open the dedicated research-interests editor and capture its result."""
        from arxiv_browser.modals.llm import ResearchInterestsModal

        def _store(result: str | None) -> None:
            if result is not None:
                self._interests = result
                self.query_one("#settings-interests-value", Static).update(
                    self._interests_summary()
                )

        self.app.push_screen(ResearchInterestsModal(self._interests), _store)

    @on(Button.Pressed, "#settings-save")
    def on_save_pressed(self) -> None:
        """Handle the Save button by delegating to action_save."""
        self.action_save()

    @on(Button.Pressed, "#settings-cancel")
    def on_cancel_pressed(self) -> None:
        """Handle the Cancel button by dismissing with no result."""
        self.dismiss(None)

    def action_save(self) -> None:
        """Gather widget state and dismiss with the desired settings."""
        preset = self.query_one("#settings-llm-preset", Select).value
        theme = self.query_one("#settings-theme", Select).value
        self.dismiss(
            SettingsResult(
                llm_preset=str(preset) if preset is not Select.BLANK else "",
                theme_name=str(theme) if theme is not Select.BLANK else self._current.theme_name,
                s2_enabled=self.query_one("#settings-s2", Switch).value,
                hf_enabled=self.query_one("#settings-hf", Switch).value,
                research_interests=self._interests,
            )
        )
