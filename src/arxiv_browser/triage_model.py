"""Local supervised triage model for paper review buckets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arxiv_browser.database import get_cache_db_path
from arxiv_browser.models import Paper, PaperCollection, PaperMetadata

TRIAGE_MODEL_VERSION = 1
TRIAGE_MODEL_FILENAME = "triage_model.joblib"
TRIAGE_MODEL_INFO_FILENAME = "triage_model.json"
TRIAGE_BUCKET_LIKELY_STAR = "likely_star"
TRIAGE_BUCKET_UNSURE = "unsure"
TRIAGE_BUCKET_LIKELY_SKIP = "likely_skip"
TRIAGE_LIKELY_STAR_THRESHOLD = 0.70
TRIAGE_LIKELY_SKIP_THRESHOLD = 0.20
TRIAGE_MIN_EXAMPLES = 20
TRIAGE_MIN_CLASS_EXAMPLES = 5
TRIAGE_DIAGNOSTIC_TOP_N = 8
TRIAGE_INSTALL_HINT = (
    "Install ML extras with `uv sync --extra ml` or `pip install arxiv-subscription-viewer[ml]`."
)


@dataclass(frozen=True, slots=True)
class TriageTrainingExample:
    """One labeled paper used to train the local triage model."""

    arxiv_id: str
    text: str
    label: int


@dataclass(frozen=True, slots=True)
class TriagePrediction:
    """One probability and bucket assigned by the triage model."""

    arxiv_id: str
    probability: float
    bucket: str


@dataclass(frozen=True, slots=True)
class TriageWeightedTerm:
    """One learned term weight from the triage classifier."""

    term: str
    weight: float


@dataclass(frozen=True, slots=True)
class TriageModelDiagnostics:
    """Read-only diagnostic summary for the local triage model."""

    status: str
    message: str
    info: TriageModelInfo | None = None
    predicted_count: int = 0
    bucket_counts: dict[str, int] = field(default_factory=dict)
    uncertain_predictions: tuple[TriagePrediction, ...] = ()
    positive_terms: tuple[TriageWeightedTerm, ...] = ()
    negative_terms: tuple[TriageWeightedTerm, ...] = ()


@dataclass(frozen=True, slots=True)
class TriageModelInfo:
    """Metadata persisted next to the trained model artifact."""

    model_version: int
    trained_at: str
    positive_count: int
    negative_count: int
    total_count: int
    sklearn_version: str
    likely_star_threshold: float = TRIAGE_LIKELY_STAR_THRESHOLD
    likely_skip_threshold: float = TRIAGE_LIKELY_SKIP_THRESHOLD


class MissingTriageModelDependencyError(RuntimeError):
    """Raised when sklearn/joblib are unavailable."""


class InsufficientTriageTrainingDataError(RuntimeError):
    """Raised when there are not enough labels to train a useful model."""


def triage_model_dir() -> Path:
    """Return the directory that stores triage model artifacts."""
    return get_cache_db_path().parent


def triage_model_paths(base_dir: Path | None = None) -> tuple[Path, Path]:
    """Return ``(model_path, info_path)`` for the triage model artifacts."""
    root = base_dir or triage_model_dir()
    return root / TRIAGE_MODEL_FILENAME, root / TRIAGE_MODEL_INFO_FILENAME


def bucket_for_probability(probability: float) -> str:
    """Map a star-worthiness probability to the user-facing triage bucket."""
    probability = max(0.0, min(1.0, probability))
    if probability >= TRIAGE_LIKELY_STAR_THRESHOLD:
        return TRIAGE_BUCKET_LIKELY_STAR
    if probability <= TRIAGE_LIKELY_SKIP_THRESHOLD:
        return TRIAGE_BUCKET_LIKELY_SKIP
    return TRIAGE_BUCKET_UNSURE


def format_triage_prediction(prediction: TriagePrediction, *, ascii_mode: bool = False) -> str:
    """Return compact plain-text prediction copy for row and triage badges."""
    pct = round(max(0.0, min(1.0, prediction.probability)) * 100)
    if prediction.bucket == TRIAGE_BUCKET_LIKELY_STAR:
        marker = "*" if ascii_mode else "\u2605"
        return f"ML:{marker}{pct}%"
    if prediction.bucket == TRIAGE_BUCKET_LIKELY_SKIP:
        return f"ML:skip{pct}%"
    return f"ML:?{pct}%"


def build_training_examples(
    papers: list[Paper],
    metadata: dict[str, PaperMetadata],
    collections: list[PaperCollection],
) -> list[TriageTrainingExample]:
    """Extract supervised labels from user paper metadata and collections."""
    collection_ids = _collection_paper_ids(collections)
    examples: list[TriageTrainingExample] = []
    seen: set[str] = set()
    for paper in papers:
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        label = _triage_label_for(paper.arxiv_id, metadata, collection_ids)
        if label is None:
            continue
        text = triage_feature_text(paper)
        if text.strip():
            examples.append(TriageTrainingExample(paper.arxiv_id, text, label))
    return examples


def triage_feature_text(paper: Paper) -> str:
    """Return the weighted text representation used by the sklearn pipeline."""
    abstract = paper.abstract_raw or paper.abstract or ""
    return " ".join(
        part
        for part in (
            paper.title,
            paper.title,
            paper.categories,
            paper.categories,
            paper.authors,
            abstract,
        )
        if part
    )


def fit_triage_model(
    papers: list[Paper],
    metadata: dict[str, PaperMetadata],
    collections: list[PaperCollection],
) -> tuple[Any, TriageModelInfo]:
    """Train a sklearn TF-IDF + logistic regression triage model."""
    deps = _load_ml_dependencies()
    examples = build_training_examples(papers, metadata, collections)
    positive_count, negative_count = _validate_training_examples(examples)
    pipeline = deps["Pipeline"](
        [
            (
                "tfidf",
                deps["TfidfVectorizer"](
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                deps["LogisticRegression"](
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipeline.fit([example.text for example in examples], [example.label for example in examples])
    info = TriageModelInfo(
        model_version=TRIAGE_MODEL_VERSION,
        trained_at=datetime.now(UTC).isoformat(),
        positive_count=positive_count,
        negative_count=negative_count,
        total_count=len(examples),
        sklearn_version=str(deps["sklearn_version"]),
    )
    return pipeline, info


def save_triage_model(
    model: Any,
    info: TriageModelInfo,
    base_dir: Path | None = None,
) -> None:
    """Persist the trained model and metadata."""
    deps = _load_ml_dependencies()
    model_path, info_path = triage_model_paths(base_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    deps["joblib_dump"](model, model_path)
    info_path.write_text(json.dumps(asdict(info), indent=2, sort_keys=True) + "\n")


def train_and_save_triage_model(
    papers: list[Paper],
    metadata: dict[str, PaperMetadata],
    collections: list[PaperCollection],
    base_dir: Path | None = None,
) -> tuple[Any, TriageModelInfo]:
    """Train and persist the triage model in one call."""
    model, info = fit_triage_model(papers, metadata, collections)
    save_triage_model(model, info, base_dir)
    return model, info


def load_triage_model(base_dir: Path | None = None) -> tuple[Any, TriageModelInfo] | None:
    """Load the persisted triage model, returning ``None`` when absent."""
    model_path, info_path = triage_model_paths(base_dir)
    if not model_path.exists() or not info_path.exists():
        return None
    deps = _load_ml_dependencies()
    model = deps["joblib_load"](model_path)
    info = _load_model_info(info_path)
    return model, info


def clear_triage_model(base_dir: Path | None = None) -> bool:
    """Delete persisted triage model artifacts. Returns whether anything changed."""
    changed = False
    for path in triage_model_paths(base_dir):
        if path.exists():
            path.unlink()
            changed = True
    return changed


def predict_triage(
    papers: list[Paper],
    model: Any,
) -> dict[str, TriagePrediction]:
    """Predict triage probabilities for the provided papers."""
    if not papers:
        return {}
    texts = [triage_feature_text(paper) for paper in papers]
    probabilities = model.predict_proba(texts)
    positive_index = _positive_class_index(model)
    result: dict[str, TriagePrediction] = {}
    for paper, row in zip(papers, probabilities, strict=True):
        probability = float(row[positive_index])
        result[paper.arxiv_id] = TriagePrediction(
            arxiv_id=paper.arxiv_id,
            probability=max(0.0, min(1.0, probability)),
            bucket=bucket_for_probability(probability),
        )
    return result


def build_triage_model_diagnostics(
    model: Any | None,
    info: TriageModelInfo | None,
    predictions: dict[str, TriagePrediction],
    *,
    status: str = "loaded",
    message: str = "",
    top_n: int = TRIAGE_DIAGNOSTIC_TOP_N,
) -> TriageModelDiagnostics:
    """Build a read-only summary for display in the diagnostics modal."""
    if model is None or info is None:
        return TriageModelDiagnostics(status=status, message=message or "No trained model found.")
    bucket_counts = _prediction_bucket_counts(predictions)
    uncertain = tuple(
        sorted(predictions.values(), key=lambda item: abs(item.probability - 0.5))[:top_n]
    )
    positive_terms, negative_terms = _extract_model_terms(model, top_n)
    return TriageModelDiagnostics(
        status=status,
        message=message or "Loaded triage model.",
        info=info,
        predicted_count=len(predictions),
        bucket_counts=bucket_counts,
        uncertain_predictions=uncertain,
        positive_terms=positive_terms,
        negative_terms=negative_terms,
    )


def _prediction_bucket_counts(predictions: dict[str, TriagePrediction]) -> dict[str, int]:
    counts = {
        TRIAGE_BUCKET_LIKELY_STAR: 0,
        TRIAGE_BUCKET_UNSURE: 0,
        TRIAGE_BUCKET_LIKELY_SKIP: 0,
    }
    for prediction in predictions.values():
        counts[prediction.bucket] = counts.get(prediction.bucket, 0) + 1
    return counts


def _extract_model_terms(
    model: Any,
    top_n: int,
) -> tuple[tuple[TriageWeightedTerm, ...], tuple[TriageWeightedTerm, ...]]:
    try:
        tfidf = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        features = list(tfidf.get_feature_names_out())
        coefficients = _positive_class_coefficients(classifier)
    except (AttributeError, KeyError, TypeError, ValueError):
        return (), ()
    if len(features) != len(coefficients):
        return (), ()
    weighted = [
        TriageWeightedTerm(term, float(weight))
        for term, weight in zip(features, coefficients, strict=True)
    ]
    positive = tuple(sorted(weighted, key=lambda item: item.weight, reverse=True)[:top_n])
    negative = tuple(sorted(weighted, key=lambda item: item.weight)[:top_n])
    return positive, negative


def _positive_class_coefficients(classifier: Any) -> list[float]:
    classes = list(getattr(classifier, "classes_", []))
    coef = getattr(classifier, "coef_", None)
    if not classes or coef is None:
        raise ValueError("missing classifier coefficients")
    rows = coef.tolist() if hasattr(coef, "tolist") else coef
    if len(rows) == 1:
        return [float(value) for value in rows[0]]
    if 1 not in classes:
        raise ValueError("positive class is unavailable")
    return [float(value) for value in rows[classes.index(1)]]


def _load_ml_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise MissingTriageModelDependencyError(TRIAGE_INSTALL_HINT) from exc
    return {
        "Pipeline": Pipeline,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "joblib_dump": joblib.dump,
        "joblib_load": joblib.load,
        "sklearn_version": sklearn.__version__,
    }


def _collection_paper_ids(collections: list[PaperCollection]) -> set[str]:
    paper_ids: set[str] = set()
    for collection in collections:
        paper_ids.update(collection.paper_ids)
    return paper_ids


def _triage_label_for(
    arxiv_id: str,
    metadata: dict[str, PaperMetadata],
    collection_ids: set[str],
) -> int | None:
    entry = metadata.get(arxiv_id)
    if arxiv_id in collection_ids or (entry is not None and entry.starred):
        return 1
    if entry is not None and entry.is_read and not entry.tags:
        return 0
    return None


def _validate_training_examples(examples: list[TriageTrainingExample]) -> tuple[int, int]:
    positive_count = sum(1 for example in examples if example.label == 1)
    negative_count = sum(1 for example in examples if example.label == 0)
    if len(examples) < TRIAGE_MIN_EXAMPLES:
        raise InsufficientTriageTrainingDataError(
            f"Need at least {TRIAGE_MIN_EXAMPLES} labeled papers; found {len(examples)}."
        )
    if positive_count < TRIAGE_MIN_CLASS_EXAMPLES or negative_count < TRIAGE_MIN_CLASS_EXAMPLES:
        raise InsufficientTriageTrainingDataError(
            "Need at least "
            f"{TRIAGE_MIN_CLASS_EXAMPLES} positive and {TRIAGE_MIN_CLASS_EXAMPLES} negative "
            f"examples; found {positive_count} positive and {negative_count} negative."
        )
    return positive_count, negative_count


def _load_model_info(info_path: Path) -> TriageModelInfo:
    data = json.loads(info_path.read_text())
    return TriageModelInfo(
        model_version=int(data.get("model_version", TRIAGE_MODEL_VERSION)),
        trained_at=str(data.get("trained_at", "")),
        positive_count=int(data.get("positive_count", 0)),
        negative_count=int(data.get("negative_count", 0)),
        total_count=int(data.get("total_count", 0)),
        sklearn_version=str(data.get("sklearn_version", "")),
        likely_star_threshold=float(
            data.get("likely_star_threshold", TRIAGE_LIKELY_STAR_THRESHOLD)
        ),
        likely_skip_threshold=float(
            data.get("likely_skip_threshold", TRIAGE_LIKELY_SKIP_THRESHOLD)
        ),
    )


def _positive_class_index(model: Any) -> int:
    classes = list(getattr(model, "classes_", (0, 1)))
    try:
        return classes.index(1)
    except ValueError:
        return min(1, len(classes) - 1)


__all__ = [
    "TRIAGE_BUCKET_LIKELY_SKIP",
    "TRIAGE_BUCKET_LIKELY_STAR",
    "TRIAGE_BUCKET_UNSURE",
    "TRIAGE_DIAGNOSTIC_TOP_N",
    "TRIAGE_INSTALL_HINT",
    "TRIAGE_LIKELY_SKIP_THRESHOLD",
    "TRIAGE_LIKELY_STAR_THRESHOLD",
    "TRIAGE_MIN_CLASS_EXAMPLES",
    "TRIAGE_MIN_EXAMPLES",
    "InsufficientTriageTrainingDataError",
    "MissingTriageModelDependencyError",
    "TriageModelDiagnostics",
    "TriageModelInfo",
    "TriagePrediction",
    "TriageTrainingExample",
    "TriageWeightedTerm",
    "bucket_for_probability",
    "build_training_examples",
    "build_triage_model_diagnostics",
    "clear_triage_model",
    "fit_triage_model",
    "format_triage_prediction",
    "load_triage_model",
    "predict_triage",
    "save_triage_model",
    "train_and_save_triage_model",
    "triage_feature_text",
    "triage_model_paths",
]
