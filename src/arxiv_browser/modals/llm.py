"""LLM-powered modals -- summaries, relevance scoring, paper chat."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import cast

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Label, Static, TextArea

from arxiv_browser.app_protocols import TaskTrackingApp
from arxiv_browser.llm import CHAT_SYSTEM_PROMPT, LLM_COMMAND_TIMEOUT, PaperDebateResult
from arxiv_browser.llm_providers import LLMProvider
from arxiv_browser.modals.base import ModalBase
from arxiv_browser.models import Paper
from arxiv_browser.query import escape_rich_text, format_summary_as_rich, truncate_text
from arxiv_browser.themes import theme_colors_for

logger = logging.getLogger(__name__)


class SummaryModeModal(ModalBase[str]):
    """Modal for selecting AI summary mode (TLDR, methods, results, etc.)."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("d", "mode_default", "Default", show=False),
        Binding("q", "mode_quick", "Quick", show=False),
        Binding("t", "mode_tldr", "TLDR", show=False),
        Binding("m", "mode_methods", "Methods", show=False),
        Binding("r", "mode_results", "Results", show=False),
        Binding("c", "mode_comparison", "Comparison", show=False),
        Binding("5", "mode_eli5", "ELI5", show=False),
        Binding("p", "mode_phd", "PhD", show=False),
    ]

    CSS = """
    SummaryModeModal {
        align: center middle;
    }

    #summary-mode-dialog {
        width: 52;
        max-width: 90%;
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
        """Yield the summary mode selection dialog with labeled keyboard shortcuts."""
        from arxiv_browser._ascii import is_ascii_mode

        dash = "--" if is_ascii_mode() else "\u2014"
        g = theme_colors_for(self)["green"]
        with Vertical(id="summary-mode-dialog"):
            yield Label("AI Summary Mode", id="summary-mode-title")
            yield Static(
                f"  [{g}]d[/]  Default  [dim]{dash} Full summary (Problem / Approach / Results)[/]\n"
                f"  [{g}]q[/]  Quick    [dim]{dash} Fast abstract-only summary[/]\n"
                f"  [{g}]t[/]  TLDR     [dim]{dash} 1-2 sentence summary[/]\n"
                f"  [{g}]m[/]  Methods  [dim]{dash} Technical methodology deep-dive[/]\n"
                f"  [{g}]r[/]  Results  [dim]{dash} Key experimental results with numbers[/]\n"
                f"  [{g}]c[/]  Compare  [dim]{dash} Comparison with related work[/]\n"
                f"  [{g}]5[/]  ELI5     [dim]{dash} Jargon-free analogy explanation[/]\n"
                f"  [{g}]p[/]  PhD      [dim]{dash} Explain for another field[/]",
                classes="summary-mode-keys",
            )
            yield Static("[dim]Cancel: Esc[/dim]", id="summary-mode-footer")

    def action_cancel(self) -> None:
        """Dismiss the modal without selecting a summary mode."""
        self.dismiss("")

    def action_mode_default(self) -> None:
        """Dismiss with 'default' for a full structured summary."""
        self.dismiss("default")

    def action_mode_quick(self) -> None:
        """Dismiss with 'quick' for a fast abstract-only summary."""
        self.dismiss("quick")

    def action_mode_tldr(self) -> None:
        """Dismiss with 'tldr' for a 1-2 sentence summary."""
        self.dismiss("tldr")

    def action_mode_methods(self) -> None:
        """Dismiss with 'methods' for a technical methodology deep-dive."""
        self.dismiss("methods")

    def action_mode_results(self) -> None:
        """Dismiss with 'results' for key experimental results."""
        self.dismiss("results")

    def action_mode_comparison(self) -> None:
        """Dismiss with 'comparison' for a related-work comparison."""
        self.dismiss("comparison")

    def action_mode_eli5(self) -> None:
        """Dismiss with 'eli5' for a jargon-free analogy explanation."""
        self.dismiss("eli5")

    def action_mode_phd(self) -> None:
        """Dismiss with 'phd' for a cross-field PhD explanation."""
        self.dismiss("phd")


class ResearchInterestsModal(ModalBase[str]):
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
        width: 70;
        max-width: 90%;
        height: 60%;
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
        """Initialize with the user's current research interests text."""
        super().__init__()
        self._current_interests = current_interests

    def compose(self) -> ComposeResult:
        """Yield the dialog with a text area for interests and Save/Cancel buttons."""
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
            yield Static("[dim]Ctrl+S save · Esc cancel[/dim]", id="interests-keys-help")

    def on_mount(self) -> None:
        """Focus the research interests text area on mount."""
        self._focus_widget("#interests-textarea")

    def action_save(self) -> None:
        """Save the trimmed text area content and dismiss the modal."""
        text = self.query_one("#interests-textarea", TextArea).text.strip()
        self.dismiss(text)

    def action_cancel(self) -> None:
        """Dismiss the modal without saving changes."""
        self.dismiss("")

    @on(Button.Pressed, "#save-btn")
    def on_save_pressed(self) -> None:
        """Handle the Save button press by delegating to action_save."""
        self.action_save()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        """Handle the Cancel button press by delegating to action_cancel."""
        self.action_cancel()


class PaperRemixResultModal(ModalBase[None]):
    """Read-only modal showing an LLM-generated paper-remix idea."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = """
    PaperRemixResultModal {
        align: center middle;
    }

    #paper-remix-dialog {
        width: 82%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-accent-alt;
        padding: 0 2;
    }

    #paper-remix-title {
        text-style: bold;
        color: $th-accent-alt;
        margin-bottom: 1;
        height: auto;
    }

    #paper-remix-sources {
        color: $th-muted;
        margin-bottom: 1;
        height: auto;
    }

    #paper-remix-body {
        height: 1fr;
        background: $th-panel;
        padding: 1;
    }

    #paper-remix-help {
        color: $th-muted;
        margin-top: 1;
        height: auto;
    }
    """

    def __init__(self, papers: list[Paper], result_text: str) -> None:
        """Initialize with the papers synthesized and the generated result."""
        super().__init__()
        self._papers = papers
        self._result_text = result_text

    def compose(self) -> ComposeResult:
        """Yield the paper-remix result dialog."""
        with Vertical(id="paper-remix-dialog"):
            yield Static("Paper Remix", id="paper-remix-title")
            yield Static(self._format_sources(), id="paper-remix-sources", markup=False)
            with VerticalScroll(id="paper-remix-body"):
                yield Static(self._result_text, markup=False)
            yield Static("[dim]Esc/q close[/]", id="paper-remix-help")

    def _format_sources(self) -> str:
        """Return a compact plain-text source list."""
        lines = [f"{index}. {paper.title}" for index, paper in enumerate(self._papers, start=1)]
        return "Sources\n" + "\n".join(lines)

    def action_close(self) -> None:
        """Close the result modal."""
        self.dismiss(None)


class PaperDebateResultModal(ModalBase[None]):
    """Read-only modal showing an advocate-vs-Reviewer-2 debate."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = """
    PaperDebateResultModal {
        align: center middle;
    }

    #paper-debate-dialog {
        width: 82%;
        height: 85%;
        min-width: 60;
        min-height: 20;
        background: $th-background;
        border: tall $th-purple;
        padding: 0 2;
    }

    #paper-debate-title {
        text-style: bold;
        color: $th-purple;
        margin-bottom: 1;
        height: auto;
    }

    #paper-debate-source {
        color: $th-muted;
        margin-bottom: 1;
        height: auto;
    }

    #paper-debate-body {
        height: 1fr;
        background: $th-panel;
        padding: 1;
    }

    #paper-debate-help {
        color: $th-muted;
        margin-top: 1;
        height: auto;
    }
    """

    def __init__(self, paper: Paper, result: PaperDebateResult) -> None:
        """Initialize with the source paper and generated debate result."""
        super().__init__()
        self._paper = paper
        self._result = result

    def compose(self) -> ComposeResult:
        """Yield the paper debate result dialog."""
        with Vertical(id="paper-debate-dialog"):
            yield Static("Debate Paper", id="paper-debate-title")
            yield Static(self._format_source(), id="paper-debate-source", markup=False)
            with VerticalScroll(id="paper-debate-body"):
                yield Static(self._format_debate_body(), id="paper-debate-thread")
            yield Static("[dim]Esc/q close[/]", id="paper-debate-help")

    def _format_source(self) -> str:
        """Return a compact plain-text source label."""
        return f"arXiv:{self._paper.arxiv_id} | {self._paper.title}"

    def _format_debate_body(self) -> str:
        """Return Rich-safe threaded debate markup."""
        colors = theme_colors_for(self)
        advocate_color = colors["green"]
        reviewer_color = colors["pink"]
        advocate = format_summary_as_rich(self._result.advocate, colors)
        reviewer = format_summary_as_rich(self._result.reviewer, colors)
        return (
            f"[bold {advocate_color}]Advocate[/]\n"
            f"{advocate}\n\n"
            f"[bold {reviewer_color}]Reviewer 2[/]\n"
            f"{reviewer}"
        )

    def action_close(self) -> None:
        """Close the result modal."""
        self.dismiss(None)


class PaperComparisonScreen(ModalBase[None]):
    """Read-only side-by-side comparison view for 2-3 papers."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
        Binding("g", "generate_ai", "AI Compare", show=False),
    ]

    CSS = """
    PaperComparisonScreen {
        align: center middle;
    }

    #paper-comparison-dialog {
        width: 92%;
        height: 90%;
        min-width: 70;
        min-height: 24;
        background: $th-background;
        border: tall $th-purple;
        padding: 0 2;
    }

    #paper-comparison-title {
        text-style: bold;
        color: $th-purple;
        margin-bottom: 1;
        height: auto;
    }

    #paper-comparison-columns {
        height: 2fr;
        margin-bottom: 1;
    }

    .paper-comparison-column {
        width: 1fr;
        height: 1fr;
        background: $th-panel;
        padding: 1;
        margin-right: 1;
    }

    #paper-comparison-ai-status {
        height: auto;
        color: $th-muted;
        margin-bottom: 1;
    }

    #paper-comparison-ai-output {
        height: 1fr;
        background: $th-panel;
        padding: 1;
    }

    #paper-comparison-help {
        color: $th-muted;
        margin-top: 1;
        height: auto;
    }
    """

    def __init__(
        self,
        papers: list[Paper],
        abstracts: Mapping[str, str] | None = None,
        on_generate_ai: Callable[[PaperComparisonScreen], None] | None = None,
    ) -> None:
        """Initialize with papers, local abstract text, and optional AI callback."""
        super().__init__()
        self._papers = papers
        self._abstracts = dict(abstracts or {})
        self._on_generate_ai = on_generate_ai
        self._ai_running = False

    @property
    def ai_running(self) -> bool:
        """Return whether an AI comparison request is currently in flight."""
        return self._ai_running

    def compose(self) -> ComposeResult:
        """Yield the comparison dialog with local columns and an AI result panel."""
        with Vertical(id="paper-comparison-dialog"):
            yield Static(f"Compare {len(self._papers)} Papers", id="paper-comparison-title")
            with Horizontal(id="paper-comparison-columns"):
                for index, paper in enumerate(self._papers, start=1):
                    with VerticalScroll(classes="paper-comparison-column"):
                        yield Static(self._format_paper_column(index, paper))
            yield Static(
                "[dim]Press g to generate an optional AI comparison.[/]",
                id="paper-comparison-ai-status",
            )
            with VerticalScroll(id="paper-comparison-ai-output"):
                yield Static("", id="paper-comparison-ai-text")
            yield Static("[dim]g AI compare · Esc/q close[/]", id="paper-comparison-help")

    def _format_paper_column(self, index: int, paper: Paper) -> str:
        """Return Rich-safe metadata and abstract markup for one comparison column."""
        title = escape_rich_text(truncate_text(paper.title, 120))
        authors = escape_rich_text(truncate_text(paper.authors, 120))
        categories = escape_rich_text(paper.categories)
        comments = escape_rich_text(paper.comments or "None")
        abstract = escape_rich_text(self._abstract_for(paper))
        return (
            f"[bold]{index}. {title}[/]\n"
            f"[dim]arXiv:[/] {paper.arxiv_id}\n"
            f"[dim]Date:[/] {escape_rich_text(paper.date)}\n"
            f"[dim]Categories:[/] {categories}\n"
            f"[dim]Authors:[/] {authors}\n"
            f"[dim]Comments:[/] {comments}\n\n"
            f"[bold]Abstract[/]\n{abstract}"
        )

    def _abstract_for(self, paper: Paper) -> str:
        abstract = self._abstracts.get(paper.arxiv_id) or paper.abstract or paper.abstract_raw or ""
        return abstract.strip() or "(no abstract)"

    def action_generate_ai(self) -> None:
        """Request optional AI comparison generation through the app callback."""
        if self._ai_running:
            self._update_ai_status("[dim]AI comparison already generating...[/]")
            return
        if self._on_generate_ai is None:
            self.set_ai_error("AI comparison unavailable")
            return
        self._on_generate_ai(self)

    def set_ai_loading(self) -> None:
        """Mark the AI panel as loading."""
        self._ai_running = True
        self._update_ai_status("[dim]Generating AI comparison...[/]")
        self._update_ai_output("")

    def set_ai_idle(self, message: str) -> None:
        """Mark the AI panel idle with an explanatory status."""
        self._ai_running = False
        self._update_ai_status(f"[dim]{escape_rich_text(message)}[/]")

    def set_ai_result(self, text: str) -> None:
        """Render a completed AI comparison."""
        self._ai_running = False
        self._update_ai_status("[dim]AI comparison generated.[/]")
        self._update_ai_output(format_summary_as_rich(text, theme_colors_for(self)))

    def set_ai_error(self, message: str) -> None:
        """Render an AI comparison error without closing the local comparison."""
        self._ai_running = False
        safe = escape_rich_text(message)
        self._update_ai_status(f"[red]{safe}[/]")
        self._update_ai_output("")

    def _update_ai_status(self, markup: str) -> None:
        try:
            self.query_one("#paper-comparison-ai-status", Static).update(markup)
        except NoMatches:
            pass

    def _update_ai_output(self, markup: str) -> None:
        try:
            self.query_one("#paper-comparison-ai-text", Static).update(markup)
        except NoMatches:
            pass

    def action_close(self) -> None:
        """Close the comparison modal."""
        self.dismiss(None)


class PaperChatScreen(ModalBase[None]):
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
        padding: 1;
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
        provider: LLMProvider,
        paper_content: str = "",
        *,
        timeout: int = 0,
        streaming_enabled: bool = False,
    ) -> None:
        """Initialize with a paper, LLM provider, and optional full paper content."""
        super().__init__()
        self._paper = paper
        self._provider = provider
        self._paper_content = paper_content
        self._history: list[tuple[str, str]] = []  # (role, text)
        self._waiting = False
        self._timeout = timeout or LLM_COMMAND_TIMEOUT
        self._streaming_enabled = streaming_enabled

    def compose(self) -> ComposeResult:
        """Yield the chat dialog with a message scroll area, status bar, and input field."""
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
            yield Static("[dim]Enter send · Esc close[/dim]", id="chat-help")

    def on_mount(self) -> None:
        """Focus the chat input and display a hint about available paper content."""
        self._focus_widget("#chat-input")
        hint = (
            "Paper content loaded. Ask anything!"
            if self._paper_content
            else "Using abstract only (HTML not available). Ask anything!"
        )
        messages = self.query_one("#chat-messages", VerticalScroll)
        messages.mount(Static(f"[dim]{hint}[/]", classes="chat-system"))

    @on(Input.Submitted, "#chat-input")
    def on_question_submitted(self, event: Input.Submitted) -> None:
        """Handle user question submission by displaying it and dispatching to the LLM."""
        question = event.value.strip()
        if not question or self._waiting:
            return
        event.input.value = ""
        self._add_message("user", question)
        self._waiting = True
        self.query_one("#chat-status", Static).update("[dim]Thinking...[/]")
        cast(TaskTrackingApp, self.app)._track_task(self._ask_llm(question))

    def _add_message(self, role: str, text: str, *, markup: bool = False) -> None:
        """Append a message to the conversation history and render it in the chat scroll area."""
        self._history.append((role, text))
        display = text if markup else escape_rich_text(text)
        messages = self.query_one("#chat-messages", VerticalScroll)
        if role == "user":
            messages.mount(Static(f"[bold green]You:[/] {display}", classes="chat-user"))
        else:
            messages.mount(Static(f"[bold cyan]AI:[/] {display}", classes="chat-assistant"))
        messages.scroll_end(animate=False)

    async def _ask_llm(self, question: str) -> None:
        """Build conversation context, send the question to the LLM, and display the response."""
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

            if self._streaming_enabled and callable(
                getattr(self._provider, "execute_stream", None)
            ):
                await self._ask_llm_streaming(context)
            else:
                result = await self._provider.execute(context, self._timeout)
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

    async def _ask_llm_streaming(self, prompt: str) -> None:
        """Stream assistant output into a single message widget."""
        messages = self.query_one("#chat-messages", VerticalScroll)
        assistant_widget = Static("[bold cyan]AI:[/] ", classes="chat-assistant")
        messages.mount(assistant_widget)
        messages.scroll_end(animate=False)
        parts: list[str] = []
        async for chunk in self._provider.execute_stream(prompt, self._timeout):
            if chunk.error:
                err = escape_rich_text(chunk.error[:200])
                assistant_widget.update(f"[bold cyan]AI:[/] [red]Error: {err}[/]")
                self._history.append(("assistant", f"Error: {chunk.error}"))
                return
            if chunk.delta:
                parts.append(chunk.delta)
                assistant_widget.update(f"[bold cyan]AI:[/] {escape_rich_text(''.join(parts))}")
                messages.scroll_end(animate=False)
            if chunk.done:
                break
        output = "".join(parts).strip()
        if not output:
            assistant_widget.update("[bold cyan]AI:[/] [red]Error: Empty response content[/]")
            self._history.append(("assistant", "Error: Empty response content"))
            return
        self._history.append(("assistant", output))

    def action_close(self) -> None:
        """Close the chat screen and return to the previous screen."""
        self.dismiss(None)
