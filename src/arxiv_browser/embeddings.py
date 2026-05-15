"""Semantic-search embedding cache and ranking helpers."""

from __future__ import annotations

import logging
import math
import sqlite3
import struct
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from arxiv_browser.embedding_backends import EmbeddingBackend
from arxiv_browser.models import Paper
from arxiv_browser.parsing import clean_latex

logger = logging.getLogger(__name__)

SEMANTIC_QUERY_PREFIX = "~"
SEMANTIC_SEARCH_BATCH_SIZE = 32
EMBEDDING_DTYPE = "float32"


@dataclass(slots=True, frozen=True)
class SemanticSearchResult:
    """A paper ranked by semantic similarity to a query."""

    paper: Paper
    score: float


@dataclass(slots=True)
class SemanticSearchRequest:
    """Inputs needed to execute one semantic search."""

    db_path: Path
    papers: Sequence[Paper]
    query: str
    backend: EmbeddingBackend
    top_k: int
    min_score_percent: int
    batch_size: int = SEMANTIC_SEARCH_BATCH_SIZE


def is_semantic_search_query(query: str) -> bool:
    """Return whether a local search query requests semantic search."""
    return query.strip().startswith(SEMANTIC_QUERY_PREFIX)


def semantic_query_text(query: str) -> str:
    """Return the user text after the semantic-search prefix."""
    normalized = query.strip()
    if not normalized.startswith(SEMANTIC_QUERY_PREFIX):
        return normalized
    return normalized[len(SEMANTIC_QUERY_PREFIX) :].strip()


def semantic_text_for_paper(paper: Paper) -> str:
    """Build the document text embedded for semantic search."""
    abstract = paper.abstract
    if abstract is None:
        abstract = clean_latex(paper.abstract_raw) if paper.abstract_raw else ""
    return f"Title: {paper.title}\nAbstract: {abstract or ''}"


def semantic_text_hash(text: str) -> str:
    """Return a stable hash for invalidating stale paper embeddings."""
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized vector, preserving zero vectors as zeros."""
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def vector_to_blob(vector: Sequence[float]) -> bytes:
    """Serialize a float vector as a little-endian float32 SQLite BLOB."""
    values = [float(value) for value in vector]
    return struct.pack(f"<{len(values)}f", *values)


def vector_from_blob(blob: bytes, dimensions: int) -> list[float]:
    """Deserialize a float32 SQLite BLOB with the expected dimension count."""
    if dimensions <= 0:
        return []
    expected_size = dimensions * 4
    if len(blob) != expected_size:
        return []
    return [float(value) for value in struct.unpack(f"<{dimensions}f", blob)]


def cosine_for_normalized(query_vector: Sequence[float], document_vector: Sequence[float]) -> float:
    """Return cosine similarity for normalized vectors, clamped at zero."""
    if len(query_vector) != len(document_vector):
        return 0.0
    dot = sum(a * b for a, b in zip(query_vector, document_vector, strict=True))
    return max(0.0, min(1.0, dot))


async def semantic_search_papers(
    request: SemanticSearchRequest,
) -> list[SemanticSearchResult]:
    """Embed/cache papers and return semantic-search results."""
    paper_texts = [(paper, semantic_text_for_paper(paper)) for paper in request.papers]
    cached_vectors, missing = _load_cached_vectors(
        request.db_path,
        paper_texts,
        backend_name=request.backend.backend_name,
        model_id=request.backend.model_id,
    )
    if missing:
        texts = [text for _paper, text, _text_hash in missing]
        embeddings = await request.backend.encode_documents(texts, batch_size=request.batch_size)
        for (paper, _text, text_hash), vector in zip(missing, embeddings, strict=False):
            normalized = normalize_vector(vector)
            if not normalized:
                continue
            cached_vectors[paper.arxiv_id] = normalized
            _save_cached_vector(
                request.db_path,
                paper=paper,
                backend_name=request.backend.backend_name,
                model_id=request.backend.model_id,
                text_hash=text_hash,
                vector=normalized,
            )

    query_vector = normalize_vector(await request.backend.encode_query(request.query))
    if not query_vector:
        return []
    return rank_semantic_results(
        request.papers,
        cached_vectors,
        query_vector=query_vector,
        top_k=request.top_k,
        min_score_percent=request.min_score_percent,
    )


def rank_semantic_results(
    papers: Sequence[Paper],
    vectors_by_id: dict[str, list[float]],
    *,
    query_vector: Sequence[float],
    top_k: int,
    min_score_percent: int,
) -> list[SemanticSearchResult]:
    """Rank papers by normalized-vector cosine similarity."""
    min_score = max(0.0, min(100.0, float(min_score_percent)))
    scored: list[SemanticSearchResult] = []
    for paper in papers:
        vector = vectors_by_id.get(paper.arxiv_id)
        if vector is None:
            continue
        score = cosine_for_normalized(query_vector, vector) * 100.0
        if score >= min_score:
            scored.append(SemanticSearchResult(paper=paper, score=score))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: max(1, top_k)]


def _load_cached_vectors(
    db_path: Path,
    paper_texts: Sequence[tuple[Paper, str]],
    *,
    backend_name: str,
    model_id: str,
) -> tuple[dict[str, list[float]], list[tuple[Paper, str, str]]]:
    vectors: dict[str, list[float]] = {}
    missing: list[tuple[Paper, str, str]] = []
    try:
        with closing(sqlite3.connect(str(db_path))) as conn:
            for paper, text in paper_texts:
                text_hash = semantic_text_hash(text)
                row = conn.execute(
                    "SELECT text_hash, dimensions, dtype, embedding "
                    "FROM semantic_embeddings "
                    "WHERE arxiv_id = ? AND backend = ? AND model_id = ?",
                    (paper.arxiv_id, backend_name, model_id),
                ).fetchone()
                vector = _vector_from_cache_row(row, text_hash)
                if vector is None:
                    missing.append((paper, text, text_hash))
                    continue
                vectors[paper.arxiv_id] = vector
    except sqlite3.Error:
        logger.warning("Failed to load semantic embedding cache", exc_info=True)
        missing = [(paper, text, semantic_text_hash(text)) for paper, text in paper_texts]
    return vectors, missing


def _vector_from_cache_row(row: Any, expected_text_hash: str) -> list[float] | None:
    if row is None:
        return None
    text_hash, dimensions, dtype, blob = row
    if text_hash != expected_text_hash or dtype != EMBEDDING_DTYPE:
        return None
    if not isinstance(dimensions, int) or not isinstance(blob, bytes):
        return None
    vector = vector_from_blob(blob, dimensions)
    return vector or None


def _save_cached_vector(
    db_path: Path,
    *,
    paper: Paper,
    backend_name: str,
    model_id: str,
    text_hash: str,
    vector: Sequence[float],
) -> None:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO semantic_embeddings "
                "(arxiv_id, backend, model_id, text_hash, dimensions, dtype, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper.arxiv_id,
                    backend_name,
                    model_id,
                    text_hash,
                    len(vector),
                    EMBEDDING_DTYPE,
                    vector_to_blob(vector),
                    datetime.now(UTC).isoformat(),
                ),
            )
    except (OSError, sqlite3.Error):
        logger.warning("Failed to save semantic embedding cache", exc_info=True)


__all__ = [
    "EMBEDDING_DTYPE",
    "SEMANTIC_QUERY_PREFIX",
    "SEMANTIC_SEARCH_BATCH_SIZE",
    "SemanticSearchRequest",
    "SemanticSearchResult",
    "cosine_for_normalized",
    "is_semantic_search_query",
    "normalize_vector",
    "rank_semantic_results",
    "semantic_query_text",
    "semantic_search_papers",
    "semantic_text_for_paper",
    "semantic_text_hash",
    "vector_from_blob",
    "vector_to_blob",
]
