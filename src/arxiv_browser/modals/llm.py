"""LLM-powered modals -- summaries, relevance scoring, paper chat."""

from __future__ import annotations

import logging

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, TextArea

from arxiv_browser.llm import CHAT_SYSTEM_PROMPT, LLM_COMMAND_TIMEOUT
from arxiv_browser.llm_providers import CLIProvider
from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import THEME_COLORS

logger = logging.getLogger(__name__)


class SummaryModeModal(ModalScreen[str]):
    """Modal for selecting AI summary mode (TLDR, methods, results, etc.)."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("d", "mode_default", "Default", show=False),
        Binding("q", "mode_quick", "Quick", show=False),
        Binding("t", "mode_tldr", "TLDR", show=False),
        Binding("m", "mode_methods", "Methods", show=False),
        Binding("r", "mode_results", "Results", show=False),
        Binding("c", "mode_comparison", "Comparison", show=False),
    ]

    CSS = """
    SummaryModeModal {
        align: center middle;
    }

    #summary-mode-dialog {
        width: 52;
        height: auto;
        background: $th-background;
        border: tall $th-purple;
        padding: 0 2;
    }

    #summary-mode-title {
        text-style: bold;
        color: $th-purple;
        margin-bottom: 1;
    }

    .summary-mode-keys {
        padding-left: 2;
        color: $th-text;
    }

    #summary-mode-footer {
        color: $th-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="summary-mode-dialog"):
            yield Label("AI Summary Mode", id="summary-mode-title")
            g = THEME_COLORS["green"]
            yield Static(
                f"  [{g}]d[/]  Default  [dim]— Full summary (Problem / Approach / Results)[/]\n"
                f"  [{g}]q[/]  Quick    [dim]— Fast abstract-only summary[/]\n"
                f"  [{g}]t[/]  TLDR     [dim]— 1-2 sentence summary[/]\n"
                f"  [{g}]m[/]  Methods  [dim]— Technical methodology deep-dive[/]\n"
                f"  [{g}]r[/]  Results  [dim]— Key experimental results with numbers[/]\n"
                f"  [{g}]c[/]  Compare  [dim]— Comparison with related work[/]",
                classes="summary-mode-keys",
            )
            yield Static("[dim]Esc to cancel[/dim]", id="summary-mode-footer")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_mode_default(self) -> None:
        self.dismiss("default")

    def action_mode_quick(self) -> None:
        self.dismiss("quick")

    def action_mode_tldr(self) -> None:
        self.dismiss("tldr")

    def action_mode_methods(self) -> None:
        self.dismiss("methods")

    def action_mode_results(self) -> None:
        self.dismiss("results")

    def action_mode_comparison(self) -> None:
        self.dismiss("comparison")


class ResearchInterestsModal(ModalScreen[str]):
    """Modal dialog for editing research interests used for relevance scoring."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ResearchInterestsModal {
        align: center middle;
    }

    #interests-dialog {
        width: 60%;
        height: 60%;
        min-width: 50;
        min-height: 15;
        background: $th-background;
        border: tall $th-accent-alt;
        padding: 0 2;
    }

    #interests-title {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
    }

    #interests-help {
        color: $th-muted;
        margin-bottom: 1;
    }

    #interests-textarea {
        height: 1fr;
        background: $th-panel;
        border: none;
    }

    #interests-textarea:focus {
        border-left: tall $th-accent;
    }

    #interests-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #interests-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, current_interests: str = "") -> None:
        super().__init__()
        self._current_interests = current_interests

    def compose(self) -> ComposeResult:
        with Vertical(id="interests-dialog"):
            yield Label("Research Interests", id="interests-title")
            yield Static(
                "[dim]Describe your research focus. The LLM will score papers based on this.[/]",
                id="interests-help",
            )
            yield TextArea(self._current_interests, id="interests-textarea")
            with Horizontal(id="interests-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save (Ctrl+S)", variant="primary", id="save-btn")

    def on_mount(self) -> None:
        self.query_one("#interests-textarea", TextArea).focus()

    def action_save(self) -> None:
        text = self.query_one("#interests-textarea", TextArea).text.strip()
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()


class PaperChatScreen(ModalScreen[None]):
    """Interactive chat modal for asking questions about a paper."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    CSS = """
    PaperChatScreen {
        align: center middle;
    }

    #chat-dialog {
        width: 80%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent;
        padding: 0 2;
    }

    #chat-title {
        text-style: bold;
        color: $th-accent;
        margin-bottom: 1;
        height: auto;
    }

    #chat-messages {
        height: 1fr;
        background: $th-panel;
        padding: 1 1;
    }

    .chat-user {
        color: $th-green;
        margin-bottom: 1;
    }

    .chat-assistant {
        color: $th-text;
        margin-bottom: 1;
    }

    .chat-system {
        color: $th-muted;
        margin-bottom: 1;
    }

    #chat-input-row {
        height: auto;
        margin-top: 1;
    }

    #chat-input {
        width: 1fr;
        background: $th-panel;
        border: none;
    }

    #chat-input:focus {
        border-left: tall $th-accent;
    }

    #chat-status {
        height: auto;
        color: $th-muted;
    }
    """

    def __init__(
        self,
        paper: Paper,
        provider: CLIProvider,
        paper_content: str = "",
    ) -> None:
        super().__init__()
        self._paper = paper
        self._provider = provider
        self._paper_content = paper_content
        self._history: list[tuple[str, str]] = []  # (role, text)
        self._waiting = False

    def compose(self) -> ComposeResult:
        title = self._paper.title[:70]
        with Vertical(id="chat-dialog"):
            yield Static(f"Chat: {title}", id="chat-title")
            yield VerticalScroll(id="chat-messages")
            yield Static("", id="chat-status")
            with Horizontal(id="chat-input-row"):
                yield Input(
                    placeholder="Ask a question about this paper... (Enter to send, Esc to close)",
                    id="chat-input",
                )

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        hint = (
            "Paper content loaded. Ask anything!"
            if self._paper_content
            else "Using abstract only (HTML not available). Ask anything!"
        )
        messages = self.query_one("#chat-messages", VerticalScroll)
        messages.mount(Static(f"[dim]{hint}[/]", classes="chat-system"))

    @on(Input.Submitted, "#chat-input")
    def on_question_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self._waiting:
            return
        event.input.value = ""
        self._add_message("user", question)
        self._waiting = True
        self.query_one("#chat-status", Static).update("[dim]Thinking...[/]")
        self.app._track_task(self._ask_llm(question))  # type: ignore[attr-defined]

    def _add_message(self, role: str, text: str, *, markup: bool = False) -> None:
        self._history.append((role, text))
        display = text if markup else escape_rich_text(text)
        messages = self.query_one("#chat-messages", VerticalScroll)
        if role == "user":
            messages.mount(Static(f"[bold green]You:[/] {display}", classes="chat-user"))
        else:
            messages.mount(Static(f"[bold cyan]AI:[/] {display}", classes="chat-assistant"))
        messages.scroll_end(animate=False)

    async def _ask_llm(self, question: str) -> None:
        try:
            # Build context with conversation history
            context = CHAT_SYSTEM_PROMPT.format(
                title=self._paper.title,
                authors=self._paper.authors,
                categories=self._paper.categories,
                paper_content=self._paper_content or self._paper.abstract or "",
            )
            # Append conversation history (exclude current question)
            if self._history[:-1]:
                history_lines = [
                    f"{'User' if role == 'user' else 'Assistant'}: {text}"
                    for role, text in self._history[:-1]
                ]
                context += "\n\nConversation so far:\n" + "\n".join(history_lines)
            context += f"\n\nUser: {question}\nAssistant:"

            result = await self._provider.execute(context, LLM_COMMAND_TIMEOUT)
            if not result.success:
                err = escape_rich_text(result.error[:200])
                self._add_message("assistant", f"[red]Error: {err}[/]", markup=True)
                return
            self._add_message("assistant", result.output)
        except Exception as e:
            logger.warning("Chat LLM call failed: %s", e, exc_info=True)
            self._add_message(
                "assistant", f"[red]Error: {escape_rich_text(str(e))}[/]", markup=True
            )
        finally:
            self._waiting = False
            try:
                self.query_one("#chat-status", Static).update("")
            except NoMatches:
                pass

    def action_close(self) -> None:
        self.dismiss(None)
