"""Tests for local ML triage model app actions."""

from __future__ import annotations

from datetime import date

import pytest

from arxiv_browser.actions import triage_model_actions
from arxiv_browser.triage_model import (
    InsufficientTriageTrainingDataError,
    MissingTriageModelDependencyError,
    TriageModelInfo,
    TriagePrediction,
)
from tests.support.app_stubs import _new_app_stub


def _info() -> TriageModelInfo:
    return TriageModelInfo(
        model_version=1,
        trained_at="2026-05-15T10:30:00+00:00",
        positive_count=10,
        negative_count=10,
        total_count=20,
        sklearn_version="test",
    )


def test_action_train_triage_model_guards_active_training():
    app = _new_app_stub()
    app._triage_training_active = True

    triage_model_actions.action_train_triage_model(app)

    assert "already running" in app.notify.call_args.args[0]


def test_action_train_triage_model_starts_background_task():
    app = _new_app_stub()

    triage_model_actions.action_train_triage_model(app)

    assert app._triage_training_active is True
    assert app._track_task.called


def test_action_clear_triage_model_updates_state(monkeypatch):
    app = _new_app_stub()
    app._triage_predictions = {"paper": TriagePrediction("paper", 0.9, "likely_star")}
    monkeypatch.setattr(triage_model_actions, "clear_triage_model", lambda: True)

    triage_model_actions.action_clear_triage_model(app)

    assert app._triage_predictions == {}
    assert app._triage_model_info is None
    app._mark_badges_dirty.assert_called_once_with("triage", immediate=True)
    assert "Cleared" in app.notify.call_args.args[0]


def test_action_clear_triage_model_reports_io_error(monkeypatch):
    app = _new_app_stub()

    def raise_os_error():
        raise OSError("locked")

    monkeypatch.setattr(triage_model_actions, "clear_triage_model", raise_os_error)

    triage_model_actions.action_clear_triage_model(app)

    assert "Could not clear" in app.notify.call_args.args[0]


def test_load_triage_predictions_handles_absent_model(monkeypatch, make_paper):
    app = _new_app_stub()
    app.all_papers = [make_paper(arxiv_id="paper")]
    monkeypatch.setattr(triage_model_actions, "load_triage_model", lambda: None)

    loaded = triage_model_actions.load_triage_predictions_for_current_dataset(app)

    assert loaded is False
    assert app._triage_predictions == {}


def test_load_triage_predictions_scores_current_dataset(monkeypatch, make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="paper")
    prediction = TriagePrediction("paper", 0.82, "likely_star")
    app.all_papers = [paper]
    monkeypatch.setattr(triage_model_actions, "load_triage_model", lambda: (object(), _info()))
    monkeypatch.setattr(
        triage_model_actions,
        "predict_triage",
        lambda papers, _model: {papers[0].arxiv_id: prediction},
    )

    loaded = triage_model_actions.load_triage_predictions_for_current_dataset(app, refresh=True)

    assert loaded is True
    assert app._triage_predictions == {"paper": prediction}
    app._mark_badges_dirty.assert_called_once_with("triage", immediate=True)


def test_load_triage_predictions_reports_missing_dependency(monkeypatch):
    app = _new_app_stub()

    def raise_missing():
        raise MissingTriageModelDependencyError("install extras")

    monkeypatch.setattr(triage_model_actions, "load_triage_model", raise_missing)

    loaded = triage_model_actions.load_triage_predictions_for_current_dataset(
        app,
        notify_on_error=True,
    )

    assert loaded is False
    assert "install extras" in app.notify.call_args.args[0]


def test_load_triage_predictions_reports_corrupt_model(monkeypatch):
    app = _new_app_stub()

    def raise_value_error():
        raise ValueError("bad model")

    monkeypatch.setattr(triage_model_actions, "load_triage_model", raise_value_error)

    loaded = triage_model_actions.load_triage_predictions_for_current_dataset(
        app,
        notify_on_error=True,
    )

    assert loaded is False
    assert app._triage_predictions == {}
    assert "Could not load triage model: bad model" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_train_triage_model_async_success(monkeypatch, make_paper):
    app = _new_app_stub()
    paper = make_paper(arxiv_id="paper")
    prediction = TriagePrediction("paper", 0.82, "likely_star")
    app.all_papers = [paper]
    app._history_files = []
    app._triage_training_active = True
    monkeypatch.setattr(
        triage_model_actions,
        "train_and_save_triage_model",
        lambda *_args: (object(), _info()),
    )
    monkeypatch.setattr(
        triage_model_actions, "predict_triage", lambda *_args: {"paper": prediction}
    )

    await triage_model_actions._train_triage_model_async(app)

    assert app._triage_training_active is False
    assert app._triage_predictions == {"paper": prediction}
    assert "Trained on 20 labels" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_train_triage_model_async_reports_missing_dependency(monkeypatch, make_paper):
    app = _new_app_stub()
    app.all_papers = [make_paper()]
    app._history_files = []
    app._triage_training_active = True

    def raise_missing(*_args):
        raise MissingTriageModelDependencyError("install extras")

    monkeypatch.setattr(triage_model_actions, "train_and_save_triage_model", raise_missing)

    await triage_model_actions._train_triage_model_async(app)

    assert app._triage_training_active is False
    assert "install extras" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_train_triage_model_async_reports_insufficient_data(monkeypatch, make_paper):
    app = _new_app_stub()
    app.all_papers = [make_paper()]
    app._history_files = []
    app._triage_training_active = True

    def raise_insufficient(*_args):
        raise InsufficientTriageTrainingDataError("need more labels")

    monkeypatch.setattr(triage_model_actions, "train_and_save_triage_model", raise_insufficient)

    await triage_model_actions._train_triage_model_async(app)

    assert app._triage_training_active is False
    assert "need more labels" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_train_triage_model_async_reports_runtime_error(monkeypatch, make_paper):
    app = _new_app_stub()
    app.all_papers = [make_paper()]
    app._history_files = []
    app._triage_training_active = True
    delattr(app, "_update_footer")

    def raise_runtime(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(triage_model_actions, "train_and_save_triage_model", raise_runtime)

    await triage_model_actions._train_triage_model_async(app)

    assert app._triage_training_active is False
    assert "training failed: boom" in app.notify.call_args.args[0]


def test_apply_triage_predictions_ignores_refresh_errors():
    app = _new_app_stub()
    prediction = TriagePrediction("paper", 0.82, "likely_star")
    app._mark_badges_dirty.side_effect = RuntimeError("not mounted")

    triage_model_actions._apply_triage_predictions(
        app,
        {"paper": prediction},
        _info(),
        refresh=True,
    )

    assert app._triage_predictions == {"paper": prediction}
    assert app._triage_model_info == _info()


def test_training_summary_includes_skipped_history_files():
    assert "skipped 2 unreadable" in triage_model_actions._training_summary(_info(), 2)


def test_collect_training_papers_skips_unreadable_history(monkeypatch, make_paper, tmp_path):
    loaded = [make_paper(arxiv_id="loaded")]
    history_path = tmp_path / "2026-05-15.txt"

    def raise_os_error(_path):
        raise OSError("missing")

    monkeypatch.setattr(triage_model_actions, "parse_arxiv_file", raise_os_error)

    result = triage_model_actions._collect_training_papers(
        loaded,
        [(date(2026, 5, 15), history_path)],
    )

    assert [paper.arxiv_id for paper in result.papers] == ["loaded"]
    assert result.skipped_files == 1


def test_collect_training_papers_adds_unique_history_papers(monkeypatch, make_paper, tmp_path):
    loaded = [make_paper(arxiv_id="loaded")]
    history_path = tmp_path / "2026-05-15.txt"
    parsed = [make_paper(arxiv_id="loaded"), make_paper(arxiv_id="history")]
    monkeypatch.setattr(triage_model_actions, "parse_arxiv_file", lambda _path: parsed)

    result = triage_model_actions._collect_training_papers(
        loaded,
        [(date(2026, 5, 15), history_path)],
    )

    assert [paper.arxiv_id for paper in result.papers] == ["loaded", "history"]
    assert result.skipped_files == 0
