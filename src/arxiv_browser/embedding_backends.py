# pyright: reportMissingImports=false
"""Optional embedding backends for semantic search.

All heavyweight libraries are imported lazily inside backend constructors so
installing the base TUI never imports or downloads embedding models.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, cast

import httpx

SEMANTIC_SEARCH_BACKENDS = frozenset({"auto", "fastembed", "sentence-transformers", "http", "off"})


class EmbeddingBackendError(RuntimeError):
    """Base error for embedding backend setup or execution failures."""


class EmbeddingBackendUnavailable(EmbeddingBackendError):
    """Raised when no configured embedding backend can be used."""


class EmbeddingBackend(Protocol):
    """Runtime protocol implemented by semantic-search embedding providers."""

    backend_name: str
    model_id: str

    async def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        """Encode document texts into dense vectors."""
        ...

    async def encode_query(self, text: str) -> list[float]:
        """Encode one user query into a dense vector."""
        ...


class FastEmbedBackend:
    """Embedding backend backed by qdrant/fastembed."""

    backend_name = "fastembed"

    def __init__(self, model_id: str) -> None:
        try:
            TextEmbedding = import_module("fastembed").TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised through resolver tests
            raise EmbeddingBackendUnavailable(
                "Install the semantic-fastembed extra to use FastEmbed semantic search"
            ) from exc
        try:
            self.model_id = model_id
            self._model = TextEmbedding(model_name=model_id)
        except Exception as exc:  # pragma: no cover - depends on optional runtime/model
            raise EmbeddingBackendUnavailable(f"FastEmbed model unavailable: {exc}") from exc

    async def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_documents_sync, list(texts), batch_size)

    async def encode_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode_query_sync, text)

    def _encode_documents_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return [_vector_to_list(vec) for vec in self._model.embed(texts, batch_size=batch_size)]

    def _encode_query_sync(self, text: str) -> list[float]:
        query_embed = getattr(self._model, "query_embed", None)
        if callable(query_embed):
            return _first_vector(query_embed([text]))
        return _first_vector(self._model.embed([text]))


class SentenceTransformersBackend:
    """Embedding backend backed by sentence-transformers."""

    backend_name = "sentence-transformers"

    def __init__(self, model_id: str) -> None:
        try:
            SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised through resolver tests
            raise EmbeddingBackendUnavailable(
                "Install sentence-transformers to use Hugging Face embedding models"
            ) from exc
        try:
            self.model_id = model_id
            self._model = SentenceTransformer(model_id)
        except Exception as exc:  # pragma: no cover - depends on optional runtime/model
            raise EmbeddingBackendUnavailable(
                f"SentenceTransformers model unavailable: {exc}"
            ) from exc

    async def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        return await asyncio.to_thread(
            self._encode_sync,
            list(texts),
            batch_size,
            False,
        )

    async def encode_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._encode_sync, [text], 1, True)
        return vectors[0] if vectors else []

    def _encode_sync(self, texts: list[str], batch_size: int, is_query: bool) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
        }
        if is_query and _model_has_query_prompt(self._model):
            kwargs["prompt_name"] = "query"
        vectors = self._model.encode(texts, **kwargs)
        return [_vector_to_list(vec) for vec in vectors]


class HttpEmbeddingBackend:
    """OpenAI-compatible HTTP embeddings backend."""

    backend_name = "http"

    def __init__(self, model_id: str, base_url: str, api_key: str = "") -> None:
        base = base_url.strip().rstrip("/")
        if not base:
            raise EmbeddingBackendUnavailable("semantic_search_api_base_url is required")
        self.model_id = model_id
        self._base_url = base
        self._api_key = api_key.strip()

    async def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self._request_embeddings(list(texts[start : start + batch_size])))
        return vectors

    async def encode_query(self, text: str) -> list[float]:
        vectors = await self._request_embeddings([text])
        return vectors[0] if vectors else []

    async def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self.model_id, "input": inputs}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/embeddings",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        data = response.json()
        rows = data.get("data")
        if not isinstance(rows, list):
            raise EmbeddingBackendError("Embedding response missing data list")
        sorted_rows = sorted(
            rows, key=lambda row: row.get("index", 0) if isinstance(row, dict) else 0
        )
        vectors: list[list[float]] = []
        for row in sorted_rows:
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise EmbeddingBackendError("Embedding response row is malformed")
            vectors.append([float(value) for value in row["embedding"]])
        return vectors


def resolve_embedding_backend(config: Any) -> EmbeddingBackend:
    """Resolve the configured embedding backend, trying fallbacks for ``auto``."""
    errors: list[str] = []
    for backend_name in _candidate_backend_names(config):
        try:
            return _build_backend(backend_name, config)
        except EmbeddingBackendUnavailable as exc:
            errors.append(f"{backend_name}: {exc}")
    detail = "; ".join(errors) if errors else "semantic search is disabled"
    raise EmbeddingBackendUnavailable(detail)


def _candidate_backend_names(config: Any) -> list[str]:
    backend = str(getattr(config, "semantic_search_backend", "auto")).strip().lower()
    if backend not in SEMANTIC_SEARCH_BACKENDS:
        backend = "auto"
    if backend == "off":
        return []
    if backend != "auto":
        return [backend]

    candidates: list[str] = []
    if str(getattr(config, "semantic_search_api_base_url", "")).strip():
        candidates.append("http")
    candidates.extend(["fastembed", "sentence-transformers"])
    return candidates


def _build_backend(name: str, config: Any) -> EmbeddingBackend:
    model_id = str(getattr(config, "semantic_search_model", "")).strip()
    if not model_id:
        model_id = "BAAI/bge-small-en-v1.5"
    if name == "http":
        return HttpEmbeddingBackend(
            model_id=model_id,
            base_url=str(getattr(config, "semantic_search_api_base_url", "")),
            api_key=str(getattr(config, "semantic_search_api_key", "")),
        )
    if name == "fastembed":
        return FastEmbedBackend(model_id)
    if name == "sentence-transformers":
        return SentenceTransformersBackend(model_id)
    raise EmbeddingBackendUnavailable(f"Unsupported semantic search backend: {name}")


def _model_has_query_prompt(model: object) -> bool:
    prompts = getattr(model, "prompts", None)
    if isinstance(prompts, dict) and "query" in prompts:
        return True
    get_prompts = getattr(model, "get_prompts", None)
    if callable(get_prompts):
        try:
            resolved = get_prompts()
        except Exception:
            return False
        return isinstance(resolved, dict) and "query" in resolved
    return False


def _first_vector(vectors: object) -> list[float]:
    iterator = iter(cast(Any, vectors))
    try:
        return _vector_to_list(next(iterator))
    except StopIteration:
        return []


def _vector_to_list(vector: object) -> list[float]:
    tolist = getattr(vector, "tolist", None)
    if callable(tolist):
        vector = tolist()
    return [float(value) for value in cast(Sequence[float], vector)]


__all__ = [
    "SEMANTIC_SEARCH_BACKENDS",
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "EmbeddingBackendUnavailable",
    "FastEmbedBackend",
    "HttpEmbeddingBackend",
    "SentenceTransformersBackend",
    "resolve_embedding_backend",
]
