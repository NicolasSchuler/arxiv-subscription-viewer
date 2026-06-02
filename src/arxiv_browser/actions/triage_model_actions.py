"""Local ML triage model actions for ArxivBrowser."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from textual.css.query import NoMatches

from arxiv_browser.parsing import parse_arxiv_file
from arxiv_browser.triage_model import (
    TRIAGE_INSTALL_HINT,
    InsufficientTriageTrainingDataError,
    MissingTriageModelDependencyError,
    TriageModelInfo,
    TriagePrediction,
    build_triage_model_diagnostics,
    clear_triage_model,
    load_triage_model,
    predict_triage,
    train_and_save_triage_model,
)

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser
    from arxiv_browser.models import Paper

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _TrainingPaperCollection:
    papers: list[Paper]
    skipped_files: int


def action_train_triage_model(app: ArxivBrowser) -> None:
    """Train the local sklearn triage model from historical decisions."""
    if getattr(app, "_triage_training_active", False):
        app.notify("Triage model training is already running", title="Triage Model")
        return
    app._triage_training_active = True
    app.notify("Training from saved triage decisions...", title="Triage Model")
    app._track_task(_train_triage_model_async(app))


def action_clear_triage_model(app: ArxivBrowser) -> None:
    """Clear the persisted local triage model and current predictions."""
    try:
        changed = clear_triage_model()
    except OSError as exc:
        app.notify(f"Could not clear triage model: {exc}", title="Triage Model", severity="error")
        return
    _apply_triage_predictions(app, {}, None, refresh=True)
    message = "Cleared triage model" if changed else "No triage model to clear"
    app.notify(message, title="Triage Model")


def action_triage_model_diagnostics(app: ArxivBrowser) -> None:
    """Open read-only diagnostics for the current local triage model."""
    from arxiv_browser.modals import TriageDiagnosticsModal

    try:
        loaded = load_triage_model()
    except MissingTriageModelDependencyError:
        diagnostics = build_triage_model_diagnostics(
            None,
            None,
            {},
            status="dependency unavailable",
            message=TRIAGE_INSTALL_HINT,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Failed to load triage model diagnostics", exc_info=True)
        diagnostics = build_triage_model_diagnostics(
            None,
            None,
            {},
            status="load error",
            message=f"Could not load triage model: {exc}",
        )
    else:
        if loaded is None:
            diagnostics = build_triage_model_diagnostics(
                None,
                None,
                {},
                status="missing",
                message="No trained model found.",
            )
        else:
            model, info = loaded
            predictions = _diagnostic_predictions(app, model)
            diagnostics = build_triage_model_diagnostics(model, info, predictions)
    app.push_screen(TriageDiagnosticsModal(diagnostics, dict(app._papers_by_id)))


def load_triage_predictions_for_current_dataset(
    app: ArxivBrowser,
    *,
    refresh: bool = False,
    notify_on_error: bool = False,
) -> bool:
    """Load the persisted model and score the current app dataset when possible."""
    try:
        loaded = load_triage_model()
    except MissingTriageModelDependencyError as exc:
        _handle_triage_prediction_error(app, str(exc), notify_on_error)
        return False
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Failed to load triage model", exc_info=True)
        _handle_triage_prediction_error(app, f"Could not load triage model: {exc}", notify_on_error)
        return False

    if loaded is None:
        _apply_triage_predictions(app, {}, None, refresh=refresh)
        return False

    model, info = loaded
    try:
        predictions = predict_triage(list(app.all_papers), model)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Failed to score triage model", exc_info=True)
        _handle_triage_prediction_error(
            app,
            f"Could not score triage model: {exc}",
            notify_on_error,
        )
        return False
    _apply_triage_predictions(app, predictions, info, refresh=refresh)
    return True


async def _train_triage_model_async(app: ArxivBrowser) -> None:
    try:
        loaded_papers = list(app.all_papers)
        history_files = list(getattr(app, "_history_files", []))
        collection = await asyncio.to_thread(
            _collect_training_papers,
            loaded_papers,
            history_files,
        )
        model, info = await asyncio.to_thread(
            train_and_save_triage_model,
            collection.papers,
            app._config.paper_metadata,
            app._config.collections,
        )
        predictions = predict_triage(list(app.all_papers), model)
        _apply_triage_predictions(app, predictions, info, refresh=True)
        app.notify(_training_summary(info, collection.skipped_files), title="Triage Model")
    except MissingTriageModelDependencyError as exc:
        app.notify(str(exc), title="Triage Model", severity="warning", timeout=10)
    except InsufficientTriageTrainingDataError as exc:
        app.notify(str(exc), title="Triage Model", severity="warning", timeout=10)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Triage model training failed", exc_info=True)
        app.notify(f"Triage model training failed: {exc}", title="Triage Model", severity="error")
    finally:
        app._triage_training_active = False
        try:
            app._update_footer()
        except AttributeError:
            pass


def _collect_training_papers(
    loaded_papers: list[Paper],
    history_files: list[tuple[date, Path]],
) -> _TrainingPaperCollection:
    papers_by_id = {paper.arxiv_id: paper for paper in loaded_papers}
    skipped_files = 0
    for _history_date, path in history_files:
        try:
            parsed = parse_arxiv_file(path)
        except OSError:
            skipped_files += 1
            continue
        for paper in parsed:
            papers_by_id.setdefault(paper.arxiv_id, paper)
    return _TrainingPaperCollection(list(papers_by_id.values()), skipped_files)


def _apply_triage_predictions(
    app: ArxivBrowser,
    predictions: dict[str, TriagePrediction],
    info: TriageModelInfo | None,
    *,
    refresh: bool,
) -> None:
    state_app = cast(Any, app)
    state_app._triage_predictions = predictions
    state_app._triage_model_info = info
    if not refresh:
        return
    try:
        app._mark_badges_dirty("triage", immediate=True)
        app._refresh_detail_pane()
    except (AttributeError, NoMatches, RuntimeError):
        pass


def _training_summary(info: TriageModelInfo, skipped_files: int) -> str:
    message = (
        f"Trained on {info.total_count} labels: "
        f"{info.positive_count} positive, {info.negative_count} negative"
    )
    if skipped_files:
        message += f"; skipped {skipped_files} unreadable history file(s)"
    return message


def _handle_triage_prediction_error(
    app: ArxivBrowser,
    message: str,
    notify_on_error: bool,
) -> None:
    _apply_triage_predictions(app, {}, None, refresh=False)
    if notify_on_error:
        app.notify(message, title="Triage Model", severity="warning", timeout=10)


def _diagnostic_predictions(app: ArxivBrowser, model: Any) -> dict[str, TriagePrediction]:
    current = getattr(app, "_triage_predictions", {})
    if current:
        return dict(current)
    try:
        return predict_triage(list(app.all_papers), model)
    except (OSError, ValueError, RuntimeError):
        logger.warning("Failed to score triage model diagnostics", exc_info=True)
        return {}


__all__ = [
    "action_clear_triage_model",
    "action_train_triage_model",
    "action_triage_model_diagnostics",
    "load_triage_predictions_for_current_dataset",
]
