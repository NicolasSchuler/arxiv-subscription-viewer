"""Tests for the local sklearn triage model."""

from __future__ import annotations

import builtins

import pytest

from arxiv_browser.models import PaperCollection, PaperMetadata
from arxiv_browser.triage_model import (
    TRIAGE_BUCKET_LIKELY_SKIP,
    TRIAGE_BUCKET_LIKELY_STAR,
    TRIAGE_BUCKET_UNSURE,
    InsufficientTriageTrainingDataError,
    MissingTriageModelDependencyError,
    TriagePrediction,
    bucket_for_probability,
    build_training_examples,
    clear_triage_model,
    fit_triage_model,
    format_triage_prediction,
    load_triage_model,
    predict_triage,
    save_triage_model,
    train_and_save_triage_model,
)


def _training_corpus(make_paper):
    papers = []
    metadata: dict[str, PaperMetadata] = {}
    for i in range(10):
        arxiv_id = f"pos-{i}"
        papers.append(
            make_paper(
                arxiv_id=arxiv_id,
                title=f"Efficient transformer inference quantization {i}",
                abstract=f"Speculative decoding and low latency serving {i}",
            )
        )
        metadata[arxiv_id] = PaperMetadata(arxiv_id=arxiv_id, is_read=True, starred=True)
    for i in range(10):
        arxiv_id = f"neg-{i}"
        papers.append(
            make_paper(
                arxiv_id=arxiv_id,
                title=f"Classical geometry survey unrelated topic {i}",
                abstract=f"Topological algebra and historical notes {i}",
            )
        )
        metadata[arxiv_id] = PaperMetadata(arxiv_id=arxiv_id, is_read=True)
    return papers, metadata


def test_build_training_examples_uses_conservative_label_rules(make_paper):
    papers = [
        make_paper(arxiv_id="starred"),
        make_paper(arxiv_id="saved"),
        make_paper(arxiv_id="skipped"),
        make_paper(arxiv_id="tagged"),
        make_paper(arxiv_id="unread"),
    ]
    metadata = {
        "starred": PaperMetadata(arxiv_id="starred", starred=True),
        "skipped": PaperMetadata(arxiv_id="skipped", is_read=True),
        "tagged": PaperMetadata(arxiv_id="tagged", is_read=True, tags=["triage:later"]),
        "unread": PaperMetadata(arxiv_id="unread"),
    }
    collections = [PaperCollection(name="Reading", paper_ids=["saved"])]

    examples = build_training_examples(papers, metadata, collections)

    assert {example.arxiv_id: example.label for example in examples} == {
        "starred": 1,
        "saved": 1,
        "skipped": 0,
    }


def test_build_training_examples_deduplicates_papers(make_paper):
    papers = [make_paper(arxiv_id="same"), make_paper(arxiv_id="same")]
    metadata = {"same": PaperMetadata(arxiv_id="same", is_read=True)}

    examples = build_training_examples(papers, metadata, [])

    assert [example.arxiv_id for example in examples] == ["same"]


def test_training_requires_enough_labels_per_class(make_paper):
    papers = [make_paper(arxiv_id=str(i)) for i in range(6)]
    metadata = {
        str(i): PaperMetadata(arxiv_id=str(i), is_read=True, starred=True) for i in range(6)
    }

    with pytest.raises(InsufficientTriageTrainingDataError, match="20"):
        fit_triage_model(papers, metadata, [])


def test_training_requires_minimum_per_class(make_paper):
    papers, metadata = _training_corpus(make_paper)
    for i in range(4, 10):
        metadata[f"neg-{i}"].starred = True

    with pytest.raises(InsufficientTriageTrainingDataError, match="positive and 5 negative"):
        fit_triage_model(papers, metadata, [])


def test_bucket_thresholds_and_display_copy():
    assert bucket_for_probability(0.70) == TRIAGE_BUCKET_LIKELY_STAR
    assert bucket_for_probability(0.20) == TRIAGE_BUCKET_LIKELY_SKIP
    assert bucket_for_probability(0.46) == TRIAGE_BUCKET_UNSURE
    assert format_triage_prediction(
        TriagePrediction("x", 0.82, TRIAGE_BUCKET_LIKELY_STAR)
    ).startswith("ML:")
    assert format_triage_prediction(TriagePrediction("x", 0.12, TRIAGE_BUCKET_LIKELY_SKIP)) == (
        "ML:skip12%"
    )


def test_fit_predict_save_and_load_triage_model(make_paper, tmp_path):
    papers, metadata = _training_corpus(make_paper)
    model, info = fit_triage_model(papers, metadata, [])
    predictions = predict_triage([papers[0], papers[-1]], model)

    assert (
        predictions[papers[0].arxiv_id].probability > predictions[papers[-1].arxiv_id].probability
    )
    assert info.positive_count == 10
    assert info.negative_count == 10

    save_triage_model(model, info, tmp_path)
    loaded = load_triage_model(tmp_path)

    assert loaded is not None
    loaded_model, loaded_info = loaded
    loaded_predictions = predict_triage([papers[0], papers[-1]], loaded_model)
    assert loaded_info.total_count == 20
    assert loaded_predictions[papers[0].arxiv_id].probability == pytest.approx(
        predictions[papers[0].arxiv_id].probability
    )


def test_train_and_save_then_clear_triage_model(make_paper, tmp_path):
    papers, metadata = _training_corpus(make_paper)

    _model, info = train_and_save_triage_model(papers, metadata, [], tmp_path)
    assert info.total_count == 20
    assert load_triage_model(tmp_path) is not None

    assert clear_triage_model(tmp_path) is True
    assert load_triage_model(tmp_path) is None
    assert clear_triage_model(tmp_path) is False


def test_predict_triage_handles_empty_and_one_class_models(make_paper):
    class OneClassModel:
        classes_ = [0]

        def predict_proba(self, texts):
            return [[0.73] for _text in texts]

    assert predict_triage([], OneClassModel()) == {}

    paper = make_paper(arxiv_id="only-class")
    predictions = predict_triage([paper], OneClassModel())

    assert predictions["only-class"].probability == pytest.approx(0.73)


def test_missing_sklearn_dependency_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "sklearn":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(MissingTriageModelDependencyError, match="ML extras"):
        fit_triage_model([], {}, [])
