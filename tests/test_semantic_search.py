"""Tests for optional semantic search and embedding cache behavior."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.config import _config_to_dict, _dict_to_config
from arxiv_browser.database import init_cache_db
from arxiv_browser.embedding_backends import (
    EmbeddingBackendError,
    EmbeddingBackendUnavailable,
    FastEmbedBackend,
    HttpEmbeddingBackend,
    SentenceTransformersBackend,
    _build_backend,
    _model_has_query_prompt,
    resolve_embedding_backend,
)
from arxiv_browser.embeddings import (
    SemanticSearchRequest,
    cosine_for_normalized,
    is_semantic_search_query,
    normalize_vector,
    rank_semantic_results,
    semantic_query_text,
    semantic_search_papers,
    semantic_text_for_paper,
    semantic_text_hash,
    vector_from_blob,
    vector_to_blob,
)
from arxiv_browser.models import Paper, UserConfig
from tests.support.patch_helpers import patch_save_config


class FakeEmbeddingBackend:
    backend_name = "fake"
    model_id = "fake-model"

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.document_calls = 0
        self.query_calls = 0
        self.encoded_texts: list[str] = []

    async def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        self.document_calls += 1
        self.encoded_texts.extend(texts)
        if self.delay:
            await asyncio.sleep(self.delay)
        return [_vector_for_text(text) for text in texts]

    async def encode_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return _vector_for_text(text)


def _vector_for_text(text: str) -> list[float]:
    lowered = text.lower()
    if "rag" in lowered or "hallucination" in lowered:
        return [1.0, 0.0]
    return [0.0, 1.0]


class FakeArray:
    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def tolist(self) -> list[float]:
        return self._values


def _paper(arxiv_id: str, title: str, abstract: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        date="Mon, 15 Jan 2024",
        title=title,
        authors="Alice Example",
        categories="cs.AI",
        comments=None,
        abstract=abstract,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        abstract_raw=abstract,
    )


def test_config_defaults_and_clamping() -> None:
    config = UserConfig()
    assert config.semantic_search_backend == "auto"
    assert config.semantic_search_model == "BAAI/bge-small-en-v1.5"
    assert config.semantic_search_top_k == 100
    assert config.semantic_search_min_score == 15

    parsed = _dict_to_config(
        {
            "semantic_search_backend": "bad",
            "semantic_search_model": "",
            "semantic_search_top_k": 999,
            "semantic_search_min_score": -5,
        }
    )
    assert parsed.semantic_search_backend == "auto"
    assert parsed.semantic_search_model == "BAAI/bge-small-en-v1.5"
    assert parsed.semantic_search_top_k == 500
    assert parsed.semantic_search_min_score == 0

    serialized = _config_to_dict(parsed)
    assert serialized["semantic_search_top_k"] == 500
    assert serialized["semantic_search_min_score"] == 0


def test_semantic_prefix_helpers_and_hashing(make_paper) -> None:
    paper = make_paper(title="RAG Hallucination Mitigation", abstract="Grounded generation")
    changed = make_paper(title="RAG Hallucination Mitigation", abstract="Citation-aware answers")
    paper_without_abstract = _paper("2401.00042", "Latex cleanup", "")
    paper_without_abstract.abstract = None
    paper_without_abstract.abstract_raw = r"Grounded \(x\)"
    assert is_semantic_search_query(" ~ papers about RAG")
    assert not is_semantic_search_query("papers about RAG")
    assert semantic_query_text("~ papers about RAG") == "papers about RAG"
    assert semantic_query_text("papers about RAG") == "papers about RAG"
    assert semantic_text_for_paper(paper).startswith("Title: RAG Hallucination")
    assert "Grounded" in semantic_text_for_paper(paper_without_abstract)
    assert semantic_text_hash(semantic_text_for_paper(paper)) != semantic_text_hash(
        semantic_text_for_paper(changed)
    )


def test_vector_blob_roundtrip_and_cosine() -> None:
    vector = normalize_vector([3.0, 4.0])
    blob = vector_to_blob(vector)
    restored = vector_from_blob(blob, 2)
    assert restored == pytest.approx(vector)
    assert vector_from_blob(blob, 3) == []
    assert vector_from_blob(blob, 0) == []
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
    assert cosine_for_normalized(vector, vector) == pytest.approx(1.0)
    assert cosine_for_normalized(vector, [-vector[0], -vector[1]]) == 0.0
    assert cosine_for_normalized([1.0], [1.0, 0.0]) == 0.0


def test_rank_semantic_results_skips_missing_and_clamps_limits() -> None:
    match = _paper("2401.00001", "RAG systems", "Hallucination reduction")
    weak = _paper("2401.00002", "Bayesian inference", "Sampling")
    missing = _paper("2401.00003", "No vector", "Not cached")

    results = rank_semantic_results(
        [weak, missing, match],
        {match.arxiv_id: [1.0, 0.0], weak.arxiv_id: [0.0, 1.0]},
        query_vector=[1.0, 0.0],
        top_k=0,
        min_score_percent=-10,
    )

    assert [result.paper.arxiv_id for result in results] == [match.arxiv_id]
    assert results[0].score == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_semantic_search_caches_and_invalidates_changed_text(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    paper = _paper("2401.00001", "RAG hallucination reduction", "Grounded retrieval")
    backend = FakeEmbeddingBackend()

    request = SemanticSearchRequest(
        db_path=db_path,
        papers=[paper],
        query="RAG hallucination",
        backend=backend,
        top_k=10,
        min_score_percent=1,
    )
    first = await semantic_search_papers(request)
    second = await semantic_search_papers(request)
    assert [result.paper.arxiv_id for result in first] == ["2401.00001"]
    assert [result.paper.arxiv_id for result in second] == ["2401.00001"]
    assert backend.document_calls == 1

    changed = _paper("2401.00001", "Bayesian inverse problems", "Sampling")
    changed_request = SemanticSearchRequest(
        db_path=db_path,
        papers=[changed],
        query="Bayesian sampling",
        backend=backend,
        top_k=10,
        min_score_percent=1,
    )
    await semantic_search_papers(changed_request)
    assert backend.document_calls == 2


@pytest.mark.asyncio
async def test_cache_skips_mismatched_dimensions(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    paper = _paper("2401.00001", "RAG systems", "Hallucination reduction")
    text_hash = semantic_text_hash(semantic_text_for_paper(paper))
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "INSERT INTO semantic_embeddings "
            "(arxiv_id, backend, model_id, text_hash, dimensions, dtype, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper.arxiv_id,
                "fake",
                "fake-model",
                text_hash,
                3,
                "float32",
                vector_to_blob([1.0, 0.0]),
                "2026-01-01T00:00:00+00:00",
            ),
        )

    backend = FakeEmbeddingBackend()
    await semantic_search_papers(
        SemanticSearchRequest(
            db_path=db_path,
            papers=[paper],
            query="RAG",
            backend=backend,
            top_k=10,
            min_score_percent=1,
        )
    )
    assert backend.document_calls == 1


@pytest.mark.asyncio
async def test_cache_skips_malformed_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    paper = _paper("2401.00001", "RAG systems", "Hallucination reduction")
    text_hash = semantic_text_hash(semantic_text_for_paper(paper))
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "INSERT INTO semantic_embeddings "
            "(arxiv_id, backend, model_id, text_hash, dimensions, dtype, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper.arxiv_id,
                "fake",
                "fake-model",
                text_hash,
                "bad",
                "float32",
                vector_to_blob([1.0, 0.0]),
                "2026-01-01T00:00:00+00:00",
            ),
        )

    backend = FakeEmbeddingBackend()
    await semantic_search_papers(
        SemanticSearchRequest(
            db_path=db_path,
            papers=[paper],
            query="RAG",
            backend=backend,
            top_k=10,
            min_score_percent=1,
        )
    )
    assert backend.document_calls == 1


@pytest.mark.asyncio
async def test_cache_errors_fall_back_to_reembedding(tmp_path: Path, monkeypatch) -> None:
    paper = _paper("2401.00001", "RAG systems", "Hallucination reduction")
    backend = FakeEmbeddingBackend()

    def raise_sqlite_error(*_args, **_kwargs):
        raise sqlite3.Error("boom")

    monkeypatch.setattr("arxiv_browser.embeddings.sqlite3.connect", raise_sqlite_error)
    results = await semantic_search_papers(
        SemanticSearchRequest(
            db_path=tmp_path / "missing.db",
            papers=[paper],
            query="RAG",
            backend=backend,
            top_k=10,
            min_score_percent=1,
        )
    )
    assert [result.paper.arxiv_id for result in results] == [paper.arxiv_id]
    assert backend.document_calls == 1


@pytest.mark.asyncio
async def test_semantic_search_returns_empty_for_empty_embeddings(tmp_path: Path) -> None:
    class EmptyBackend(FakeEmbeddingBackend):
        async def encode_documents(
            self, texts: Sequence[str], *, batch_size: int
        ) -> list[list[float]]:
            return [[] for _text in texts]

        async def encode_query(self, text: str) -> list[float]:
            return []

    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    results = await semantic_search_papers(
        SemanticSearchRequest(
            db_path=db_path,
            papers=[_paper("2401.00001", "RAG systems", "Hallucination reduction")],
            query="RAG",
            backend=EmptyBackend(),
            top_k=10,
            min_score_percent=1,
        )
    )
    assert results == []


@pytest.mark.asyncio
async def test_fastembed_backend_lazy_import_and_query_paths(monkeypatch) -> None:
    created_models: list[object] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.batch_size = 0
            created_models.append(self)

        def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[FakeArray]:
            self.batch_size = batch_size
            return [FakeArray([float(index + 1), 0.0]) for index, _text in enumerate(texts)]

        def query_embed(self, texts: Sequence[str]) -> list[FakeArray]:
            return [FakeArray([0.0, 1.0]) for _text in texts]

    def fake_import_module(name: str) -> SimpleNamespace:
        assert name == "fastembed"
        return SimpleNamespace(TextEmbedding=FakeTextEmbedding)

    monkeypatch.setattr("arxiv_browser.embedding_backends.import_module", fake_import_module)
    backend = FastEmbedBackend("fake/model")

    assert await backend.encode_documents(["a", "b"], batch_size=7) == [[1.0, 0.0], [2.0, 0.0]]
    assert await backend.encode_query("query") == [0.0, 1.0]
    assert backend.model_id == "fake/model"
    created_model = created_models[0]
    assert isinstance(created_model, FakeTextEmbedding)
    assert created_model.batch_size == 7

    class EmbedOnly:
        def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
            return [[2.0, 3.0] for _text in texts]

    backend._model = EmbedOnly()
    assert backend._encode_query_sync("query") == [2.0, 3.0]


@pytest.mark.asyncio
async def test_sentence_transformers_backend_lazy_import_and_async_paths(monkeypatch) -> None:
    created_models: list[object] = []

    class FakeSentenceTransformer:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.calls: list[dict[str, object]] = []
            created_models.append(self)

        def get_prompts(self) -> dict[str, str]:
            return {"query": "Represent this sentence for searching relevant passages:"}

        def encode(self, texts: Sequence[str], **kwargs: object) -> list[FakeArray]:
            self.calls.append(kwargs)
            return [FakeArray([1.0, 0.0]) for _text in texts]

    def fake_import_module(name: str) -> SimpleNamespace:
        assert name == "sentence_transformers"
        return SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)

    monkeypatch.setattr("arxiv_browser.embedding_backends.import_module", fake_import_module)
    backend = SentenceTransformersBackend("Qwen/Qwen3-Embedding-0.6B")

    assert await backend.encode_documents(["doc"], batch_size=11) == [[1.0, 0.0]]
    assert await backend.encode_query("query") == [1.0, 0.0]
    assert backend.model_id == "Qwen/Qwen3-Embedding-0.6B"
    created_model = created_models[0]
    assert isinstance(created_model, FakeSentenceTransformer)
    assert created_model.calls[0] == {"batch_size": 11, "normalize_embeddings": True}
    assert created_model.calls[1] == {
        "batch_size": 1,
        "normalize_embeddings": True,
        "prompt_name": "query",
    }

    class BrokenPrompts:
        def get_prompts(self) -> dict[str, str]:
            raise RuntimeError("boom")

    assert not _model_has_query_prompt(BrokenPrompts())


@pytest.mark.asyncio
async def test_http_embedding_backend_batches_sorts_and_authorizes(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    class FakeResponse:
        def __init__(self, inputs: Sequence[str]) -> None:
            self._inputs = inputs

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            rows = [
                {"index": index, "embedding": [float(index + 1), float(len(text))]}
                for index, text in reversed(list(enumerate(self._inputs)))
            ]
            return {"data": rows}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> FakeResponse:
            calls.append((url, json, headers, self.timeout))
            return FakeResponse(json["input"])

    monkeypatch.setattr("arxiv_browser.embedding_backends.httpx.AsyncClient", FakeClient)
    backend = HttpEmbeddingBackend("fake-model", "http://localhost:8080/", "secret")

    assert await backend.encode_documents(["aa", "bbb", "c"], batch_size=2) == [
        [1.0, 2.0],
        [2.0, 3.0],
        [1.0, 1.0],
    ]
    assert await backend.encode_query("query") == [1.0, 5.0]
    assert calls[0][0] == "http://localhost:8080/v1/embeddings"
    assert calls[0][1] == {"model": "fake-model", "input": ["aa", "bbb"]}
    assert calls[0][2] == {"Authorization": "Bearer secret"}
    assert calls[0][3] == 60.0


@pytest.mark.asyncio
async def test_http_embedding_backend_validates_responses(monkeypatch) -> None:
    payloads: list[dict[str, object]] = [
        {"data": "bad"},
        {"data": [{"index": 0, "embedding": "bad"}]},
        {"data": []},
    ]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return payloads.pop(0)

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    with pytest.raises(EmbeddingBackendUnavailable):
        HttpEmbeddingBackend("fake-model", "")

    monkeypatch.setattr("arxiv_browser.embedding_backends.httpx.AsyncClient", FakeClient)
    backend = HttpEmbeddingBackend("fake-model", "http://localhost:8080")

    with pytest.raises(EmbeddingBackendError, match="missing data"):
        await backend.encode_documents(["doc"], batch_size=32)
    with pytest.raises(EmbeddingBackendError, match="malformed"):
        await backend.encode_documents(["doc"], batch_size=32)
    assert await backend.encode_query("query") == []


def test_backend_off_and_unsupported_errors() -> None:
    with pytest.raises(EmbeddingBackendUnavailable, match="disabled"):
        resolve_embedding_backend(UserConfig(semantic_search_backend="off"))
    with pytest.raises(EmbeddingBackendUnavailable, match="Unsupported"):
        _build_backend("unknown", UserConfig())


def test_backend_auto_selection_order(monkeypatch) -> None:
    built: list[str] = []

    class FakeFastEmbed:
        backend_name = "fastembed"
        model_id = "model"

        def __init__(self, model_id: str) -> None:
            built.append(f"fastembed:{model_id}")

    class FakeHttp:
        backend_name = "http"
        model_id = "model"

        def __init__(self, model_id: str, base_url: str, api_key: str = "") -> None:
            built.append(f"http:{model_id}:{base_url}:{api_key}")

    monkeypatch.setattr("arxiv_browser.embedding_backends.FastEmbedBackend", FakeFastEmbed)
    monkeypatch.setattr("arxiv_browser.embedding_backends.HttpEmbeddingBackend", FakeHttp)

    config = UserConfig(semantic_search_model="custom/model")
    assert resolve_embedding_backend(config).backend_name == "fastembed"
    assert built == ["fastembed:custom/model"]

    built.clear()
    config.semantic_search_api_base_url = "http://localhost:8080"
    config.semantic_search_api_key = "secret"
    assert resolve_embedding_backend(config).backend_name == "http"
    assert built == ["http:custom/model:http://localhost:8080:secret"]


def test_backend_auto_falls_through_to_sentence_transformers(monkeypatch) -> None:
    built: list[str] = []

    def unavailable(_model_id: str):
        built.append("fastembed")
        raise EmbeddingBackendUnavailable("missing")

    class FakeSentenceTransformers:
        backend_name = "sentence-transformers"
        model_id = "model"

        def __init__(self, model_id: str) -> None:
            built.append(f"sentence:{model_id}")

    monkeypatch.setattr("arxiv_browser.embedding_backends.FastEmbedBackend", unavailable)
    monkeypatch.setattr(
        "arxiv_browser.embedding_backends.SentenceTransformersBackend",
        FakeSentenceTransformers,
    )

    config = UserConfig(semantic_search_model="Qwen/Qwen3-Embedding-0.6B")
    assert resolve_embedding_backend(config).backend_name == "sentence-transformers"
    assert built == ["fastembed", "sentence:Qwen/Qwen3-Embedding-0.6B"]


def test_sentence_transformers_query_prompt_detection() -> None:
    class FakeModel:
        prompts = {"query": "query prompt"}

        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def encode(self, texts, **kwargs):
            self.kwargs = kwargs
            return [[1.0, 0.0] for _text in texts]

    backend = object.__new__(SentenceTransformersBackend)
    backend.model_id = "fake"
    backend._model = FakeModel()
    assert _model_has_query_prompt(backend._model)
    assert backend._encode_sync(["query"], 1, True) == [[1.0, 0.0]]
    assert backend._model.kwargs["prompt_name"] == "query"

    backend._model.prompts = {}
    backend._encode_sync(["query"], 1, True)
    assert "prompt_name" not in backend._model.kwargs


@pytest.mark.asyncio
async def test_browser_semantic_filter_updates_results(
    make_paper, tmp_path: Path, monkeypatch
) -> None:
    match = make_paper(
        arxiv_id="2401.00001",
        title="Reducing hallucinations in RAG systems",
        abstract="Retrieval grounding for generated answers.",
    )
    other = make_paper(
        arxiv_id="2401.00002",
        title="Bayesian inverse problems",
        abstract="Posterior sampling.",
    )
    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    backend = FakeEmbeddingBackend()
    monkeypatch.setattr(
        "arxiv_browser.browser.browse.resolve_embedding_backend", lambda _cfg: backend
    )

    app = ArxivBrowser([other, match], restore_session=False)
    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            app._cache_db_path = db_path
            app._apply_filter("~ hallucination in RAG")
            await pilot.pause()
            await pilot.pause()
            assert [paper.arxiv_id for paper in app.filtered_papers] == [match.arxiv_id]
            assert app._match_scores[match.arxiv_id] > 90


@pytest.mark.asyncio
async def test_browser_semantic_fallback_uses_fuzzy_search(make_paper, monkeypatch) -> None:
    match = make_paper(
        arxiv_id="2401.00001",
        title="Efficient transformer architectures",
        authors="Alice Example",
    )
    other = make_paper(
        arxiv_id="2401.00002",
        title="Bayesian inverse problems",
        authors="Bob Example",
    )

    def unavailable(_config):
        raise EmbeddingBackendUnavailable("missing")

    monkeypatch.setattr("arxiv_browser.browser.browse.resolve_embedding_backend", unavailable)
    app = ArxivBrowser([match, other], restore_session=False)
    with patch_save_config(return_value=True), patch.object(app, "notify", MagicMock()) as notify:
        async with app.run_test() as pilot:
            app._apply_filter("~ transformer")
            await pilot.pause()
            await pilot.pause()
            assert [paper.arxiv_id for paper in app.filtered_papers] == [match.arxiv_id]
            assert any(
                "Semantic search unavailable" in call.args[0] for call in notify.call_args_list
            )


@pytest.mark.asyncio
async def test_stale_semantic_results_do_not_publish(
    make_paper, tmp_path: Path, monkeypatch
) -> None:
    semantic_match = make_paper(
        arxiv_id="2401.00001",
        title="Reducing hallucinations in RAG systems",
        abstract="Retrieval grounding.",
    )
    fuzzy_match = make_paper(
        arxiv_id="2401.00002",
        title="Efficient transformer architectures",
        authors="Alice Example",
    )
    db_path = tmp_path / "cache.db"
    init_cache_db(db_path)
    backend = FakeEmbeddingBackend(delay=0.05)
    monkeypatch.setattr(
        "arxiv_browser.browser.browse.resolve_embedding_backend", lambda _cfg: backend
    )

    app = ArxivBrowser([semantic_match, fuzzy_match], restore_session=False)
    with patch_save_config(return_value=True):
        async with app.run_test() as pilot:
            app._cache_db_path = db_path
            app._apply_filter("~ hallucination RAG")
            app._apply_filter("transformer")
            await pilot.pause()
            await asyncio.sleep(0.08)
            await pilot.pause()
            assert app._get_active_query() == "transformer"
            assert [paper.arxiv_id for paper in app.filtered_papers] == [fuzzy_match.arxiv_id]
