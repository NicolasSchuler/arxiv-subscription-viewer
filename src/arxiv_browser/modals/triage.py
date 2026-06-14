"""Quick triage modal for rapid paper review."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Label, Static

from arxiv_browser.modals.base import ModalBase
from arxiv_browser.modals.collections import CollectionsModal
from arxiv_browser.models import Paper, PaperCollection
from arxiv_browser.query import escape_rich_text
from arxiv_browser.themes import theme_colors_for
from arxiv_browser.triage_model import (
    TRIAGE_BUCKET_LIKELY_SKIP,
    TRIAGE_BUCKET_LIKELY_STAR,
    TRIAGE_BUCKET_UNSURE,
    TriageModelDiagnostics,
    TriagePrediction,
    TriageWeightedTerm,
    format_triage_prediction,
)

TRIAGE_LATER_TAG = "triage:later"
ABSTRACT_WRAP_WIDTH = 86


@dataclass(slots=True)
class QuickTriageItem:
    """Renderable item for one paper in quick triage."""

    paper: Paper
    abstract_text: str
    relevance: tuple[int, str] | None = None
    triage_prediction: TriagePrediction | None = None
    watched: bool = False


@dataclass(slots=True)
class QuickTriageCounts:
    """Session-only decision counts for quick triage."""

    starred: int = 0
    skipped: int = 0
    tagged: int = 0
    saved: int = 0

    @property
    def total(self) -> int:
        """Return the number of reviewed papers."""
        return self.starred + self.skipped + self.tagged + self.saved


@dataclass(slots=True)
class QuickTriageResult:
    """Result returned when quick triage closes."""

    counts: QuickTriageCounts
    reviewed: int
    total: int
    completed: bool


@dataclass(frozen=True, slots=True)
class QuickTriageCallbacks:
    """Mutation callbacks owned by the app action layer."""

    mark_starred_read: Callable[[Paper], bool]
    mark_skipped: Callable[[Paper], bool]
    tag_later: Callable[[Paper], bool]
    save_to_collection: Callable[[Paper, str], bool]


@dataclass(slots=True)
class QuickTriageRequest:
    """Construction request for :class:`QuickTriageScreen`."""

    items: list[QuickTriageItem]
    callbacks: QuickTriageCallbacks
    collections: list[PaperCollection]
    papers_by_id: dict[str, Paper]


class TriageDiagnosticsModal(ModalBase[None]):
    """Read-only diagnostics for the local triage model."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    CSS = """
    #triage-diagnostics-dialog {
        width: 86;
        max-width: 95%;
        height: 82%;
        padding: 0 2;
    }

    #triage-diagnostics-body {
        height: 1fr;
        overflow-y: auto;
    }

    #triage-diagnostics-footer {
        margin-top: 1;
    }
    """

    def __init__(self, diagnostics: TriageModelDiagnostics, papers_by_id: dict[str, Paper]) -> None:
        super().__init__()
        self._diagnostics = diagnostics
        self._papers_by_id = papers_by_id

    def compose(self) -> ComposeResult:
        """Compose the diagnostics dialog."""
        with Vertical(id="triage-diagnostics-dialog", classes="modal-dialog"):
            yield Label(
                "Triage Model Diagnostics",
                id="triage-diagnostics-title",
                classes="modal-title",
            )
            yield Static(
                _render_triage_diagnostics(
                    self._diagnostics, self._papers_by_id, theme_colors_for(self)["accent"]
                ),
                id="triage-diagnostics-body",
            )
            yield Static("Esc/q close", id="triage-diagnostics-footer", classes="modal-footer")

    def action_close(self) -> None:
        """Close the overlay."""
        self.dismiss(None)


def format_quick_triage_summary(result: QuickTriageResult) -> str:
    """Return concise user-facing quick triage summary text."""
    counts = result.counts
    return (
        f"Reviewed {result.reviewed}/{result.total}: "
        f"{counts.starred} starred, {counts.skipped} skipped, "
        f"{counts.tagged} tagged, {counts.saved} saved"
    )


def first_two_abstract_lines(abstract_text: str) -> str:
    """Return a Rich-safe two-line abstract preview."""
    normalized = " ".join(abstract_text.split())
    if not normalized:
        return "[dim italic]No abstract available[/]"

    wrapped = textwrap.wrap(normalized, width=ABSTRACT_WRAP_WIDTH)
    lines = wrapped[:2] or [normalized]
    if len(wrapped) > 2 and lines:
        suffix = "..."
        last = lines[-1]
        if len(last) + len(suffix) > ABSTRACT_WRAP_WIDTH:
            last = last[: max(0, ABSTRACT_WRAP_WIDTH - len(suffix))].rstrip()
        lines[-1] = f"{last}{suffix}"
    escaped = "\n".join(escape_rich_text(line) for line in lines)
    return f"[dim italic]{escaped}[/]"


def _render_triage_diagnostics(
    diagnostics: TriageModelDiagnostics,
    papers_by_id: dict[str, Paper],
    accent: str = "",
) -> str:
    def header(text: str) -> str:
        return f"[bold {accent}]{text}[/]" if accent else f"[bold]{text}[/]"

    lines = [
        f"[bold]Status[/] {escape_rich_text(diagnostics.status)}",
        escape_rich_text(diagnostics.message),
    ]
    info = diagnostics.info
    if info is None:
        lines.extend(["", "[dim]Use Train Triage Model once you have enough decisions.[/]"])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            header("Training"),
            f"  trained: {escape_rich_text(info.trained_at)}",
            f"  labels: {info.total_count} total | {info.positive_count} positive | "
            f"{info.negative_count} negative",
            f"  sklearn: {escape_rich_text(info.sklearn_version)}",
            f"  thresholds: likely-star >= {info.likely_star_threshold:.2f}, "
            f"likely-skip <= {info.likely_skip_threshold:.2f}",
            "",
            header("Current Dataset"),
            f"  predicted: {diagnostics.predicted_count}",
            "  buckets: "
            f"likely-star {diagnostics.bucket_counts.get(TRIAGE_BUCKET_LIKELY_STAR, 0)}, "
            f"unsure {diagnostics.bucket_counts.get(TRIAGE_BUCKET_UNSURE, 0)}, "
            f"likely-skip {diagnostics.bucket_counts.get(TRIAGE_BUCKET_LIKELY_SKIP, 0)}",
            "",
            header("Most Uncertain"),
        ]
    )
    lines.extend(_uncertain_prediction_lines(diagnostics.uncertain_predictions, papers_by_id))
    lines.extend(["", header("Terms Favoring Star")])
    lines.extend(_weighted_term_lines(diagnostics.positive_terms))
    lines.extend(["", header("Terms Favoring Skip")])
    lines.extend(_weighted_term_lines(diagnostics.negative_terms))
    return "\n".join(lines)


def _uncertain_prediction_lines(
    predictions: tuple[TriagePrediction, ...],
    papers_by_id: dict[str, Paper],
) -> list[str]:
    if not predictions:
        return ["  [dim]No predictions for the current dataset.[/]"]
    lines: list[str] = []
    for prediction in predictions:
        paper = papers_by_id.get(prediction.arxiv_id)
        title = paper.title if paper else prediction.arxiv_id
        lines.append(
            f"  {escape_rich_text(format_triage_prediction(prediction))} {escape_rich_text(title)}"
        )
    return lines


def _weighted_term_lines(terms: tuple[TriageWeightedTerm, ...]) -> list[str]:
    if not terms:
        return ["  [dim]Term weights are unavailable for this model shape.[/]"]
    return [f"  {escape_rich_text(term.term)} [dim]{term.weight:+.3f}[/]" for term in terms]


class QuickTriageScreen(ModalBase[QuickTriageResult]):
    """Keyboard-first modal for rapid unread-paper triage."""

    BINDINGS = [
        Binding("y", "star_read", "Star + read", show=False),
        Binding("n", "skip", "Skip", show=False),
        Binding("t", "tag_later", "Tag later", show=False),
        Binding("s", "save", "Save", show=False),
        Binding("escape", "cancel", "Close", show=False),
        Binding("q", "cancel", "Close", show=False),
    ]

    CSS = """
    #triage-dialog {
        width: 96;
        max-width: 90%;
        height: auto;
    }

    #triage-progress {
        color: $th-muted;
        margin-bottom: 1;
    }

    #triage-title {
        text-style: bold;
        color: $th-text;
        margin-bottom: 1;
    }

    #triage-badges {
        color: $th-muted;
        margin-bottom: 1;
    }

    #triage-abstract {
        color: $th-muted;
        /* fixed height (not min-height) so short/empty abstracts don't collapse
           the dialog and jump the help line as the user advances papers. */
        height: 2;
        margin-bottom: 1;
    }

    #triage-help {
        margin-top: 1;
    }
    """

    def __init__(self, request: QuickTriageRequest) -> None:
        """Initialize quick triage from a request object."""
        super().__init__()
        self._items = request.items
        self._triage_callbacks = request.callbacks
        self._collections = request.collections
        self._papers_by_id = request.papers_by_id
        self._index = 0
        self._counts = QuickTriageCounts()
        self._collection_name: str | None = None

    def compose(self) -> ComposeResult:
        """Yield quick triage content widgets."""
        with Vertical(id="triage-dialog", classes="modal-dialog"):
            yield Label("Quick Triage", id="triage-heading", classes="modal-title")
            yield Static("", id="triage-progress")
            yield Static("", id="triage-title")
            yield Static("", id="triage-badges")
            yield Static("", id="triage-abstract")
            yield Static(
                "[bold]y[/] star+read  [bold]n[/] skip  [bold]t[/] tag later  "
                "[bold]s[/] save  [bold]Esc/q[/] close",
                id="triage-help",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        """Render the first paper when the modal mounts."""
        self._refresh()

    def action_star_read(self) -> None:
        """Star and mark the current paper read, then advance."""
        self._apply_current_decision("starred", self._triage_callbacks.mark_starred_read)

    def action_skip(self) -> None:
        """Mark the current paper read without starring, then advance."""
        self._apply_current_decision("skipped", self._triage_callbacks.mark_skipped)

    def action_tag_later(self) -> None:
        """Add the triage-later tag, mark read, then advance."""
        self._apply_current_decision("tagged", self._triage_callbacks.tag_later)

    def action_save(self) -> None:
        """Save the current paper to a collection, prompting once if needed."""
        if self._collection_name:
            self._save_current_to_collection(self._collection_name)
            return
        if not self._collections:
            self.notify(
                "No collections. Press Ctrl+k to create one.",
                title="Quick Triage",
                severity="warning",
            )
            return
        self.app.push_screen(
            CollectionsModal(self._collections, self._papers_by_id, mode="pick"),
            self._on_collection_selected,
        )

    def action_cancel(self) -> None:
        """Close quick triage and return the partial summary."""
        self.dismiss(self._build_result(completed=False))

    def _on_collection_selected(self, name: str | None) -> None:
        """Remember a picked collection and save the current paper."""
        if not name:
            return
        self._collection_name = name
        self._save_current_to_collection(name)

    def _save_current_to_collection(self, name: str) -> None:
        """Save the current paper to the chosen collection."""
        item = self._current_item()
        if item is None:
            return
        if not self._triage_callbacks.save_to_collection(item.paper, name):
            return
        self._counts.saved += 1
        self._advance()

    def _apply_current_decision(
        self,
        count_attr: str,
        callback: Callable[[Paper], bool],
    ) -> None:
        """Apply a decision callback to the current paper and advance on success."""
        item = self._current_item()
        if item is None:
            return
        if not callback(item.paper):
            return
        setattr(self._counts, count_attr, getattr(self._counts, count_attr) + 1)
        self._advance()

    def _advance(self) -> None:
        """Advance to the next paper or close with a completed summary."""
        self._index += 1
        if self._index >= len(self._items):
            self.dismiss(self._build_result(completed=True))
            return
        self._refresh()

    def _current_item(self) -> QuickTriageItem | None:
        """Return the current triage item, if any."""
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return None

    def _build_result(self, *, completed: bool) -> QuickTriageResult:
        """Build a result snapshot for the current session."""
        return QuickTriageResult(
            counts=self._counts,
            reviewed=self._counts.total,
            total=len(self._items),
            completed=completed,
        )

    def _refresh(self) -> None:
        """Refresh the modal for the current paper."""
        item = self._current_item()
        if item is None:
            return
        colors = theme_colors_for(self)
        try:
            self.query_one("#triage-progress", Static).update(
                f"Paper {self._index + 1}/{len(self._items)}"
            )
            self.query_one("#triage-title", Static).update(
                f"[bold {colors['accent']}]{escape_rich_text(item.paper.title)}[/]"
            )
            self.query_one("#triage-badges", Static).update(self._format_badges(item))
            self.query_one("#triage-abstract", Static).update(
                first_two_abstract_lines(item.abstract_text)
            )
        except NoMatches:
            return

    def _format_badges(self, item: QuickTriageItem) -> str:
        """Return compact score/watch badges for the current paper."""
        colors = theme_colors_for(self)
        parts = []
        if item.relevance is None:
            parts.append(f"[{colors['muted']}]\\[Rel: unscored][/]")
        else:
            score, _reason = item.relevance
            parts.append(f"[{colors['green']}]\\[Rel: {score}/10][/]")
        if item.watched:
            parts.append(f"[{colors['orange']}]\\[WATCH][/]")
        if item.triage_prediction is not None:
            color = _triage_badge_color(item.triage_prediction, colors)
            badge = escape_rich_text(format_triage_prediction(item.triage_prediction))
            parts.append(f"[{color}]\\[{badge}][/]")
        # Bracketed pills joined by spaces so distinct badges don't read as a run-on.
        return "  ".join(parts)


def _triage_badge_color(
    prediction: TriagePrediction,
    colors: Mapping[str, str],
) -> str:
    if prediction.bucket == TRIAGE_BUCKET_LIKELY_STAR:
        return colors["green"]
    if prediction.bucket == TRIAGE_BUCKET_LIKELY_SKIP:
        return colors["muted"]
    return colors["yellow"]


__all__ = [
    "TRIAGE_LATER_TAG",
    "QuickTriageCallbacks",
    "QuickTriageCounts",
    "QuickTriageItem",
    "QuickTriageRequest",
    "QuickTriageResult",
    "QuickTriageScreen",
    "first_two_abstract_lines",
    "format_quick_triage_summary",
]
