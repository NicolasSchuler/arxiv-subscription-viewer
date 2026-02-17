"""Regression tests for SQLite connection lifecycle/closure behavior."""

from __future__ import annotations

import gc
import sqlite3
import warnings

import arxiv_browser.huggingface as huggingface_module
import arxiv_browser.llm as llm_module
import arxiv_browser.semantic_scholar as semantic_scholar_module


class TrackingConnection:
    """Proxy around sqlite3.Connection that tracks explicit close calls."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._conn.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        self.closed = True
        self._conn.close()


def _patch_tracking_connect(monkeypatch, module):
    real_connect = sqlite3.connect
    opened: list[TrackingConnection] = []

    def _connect(*args, **kwargs):
        conn = TrackingConnection(real_connect(*args, **kwargs))
        opened.append(conn)
        return conn

    monkeypatch.setattr(module.sqlite3, "connect", _connect)
    return opened


def _assert_no_resource_warnings(caught):
    resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert resource_warnings == []


def test_llm_sqlite_connections_are_closed(monkeypatch, tmp_path):
    opened = _patch_tracking_connect(monkeypatch, llm_module)

    summary_db = tmp_path / "summaries.db"
    relevance_db = tmp_path / "relevance.db"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)

        llm_module._save_summary(summary_db, "2401.00001", "summary", "hash")
        assert llm_module._load_summary(summary_db, "2401.00001", "hash") == "summary"

        llm_module._save_relevance_score(relevance_db, "2401.00001", "interests", 8, "reason")
        assert llm_module._load_relevance_score(relevance_db, "2401.00001", "interests") == (
            8,
            "reason",
        )
        assert "2401.00001" in llm_module._load_all_relevance_scores(relevance_db, "interests")

        gc.collect()

    assert opened
    assert all(conn.closed for conn in opened)
    _assert_no_resource_warnings(caught)


def test_huggingface_sqlite_connections_are_closed(monkeypatch, tmp_path):
    opened = _patch_tracking_connect(monkeypatch, huggingface_module)

    db_path = tmp_path / "huggingface.db"
    papers = [
        huggingface_module.HuggingFacePaper(
            arxiv_id="2401.00001",
            title="HF paper",
            upvotes=12,
            num_comments=3,
            ai_summary="",
            ai_keywords=(),
            github_repo="",
            github_stars=0,
        )
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)

        huggingface_module.save_hf_daily_cache(db_path, papers)
        cached = huggingface_module.load_hf_daily_cache(db_path)
        assert cached is not None
        assert "2401.00001" in cached

        gc.collect()

    assert opened
    assert all(conn.closed for conn in opened)
    _assert_no_resource_warnings(caught)


def test_semantic_scholar_sqlite_connections_are_closed(monkeypatch, tmp_path):
    opened = _patch_tracking_connect(monkeypatch, semantic_scholar_module)

    db_path = tmp_path / "semantic_scholar.db"
    paper = semantic_scholar_module.SemanticScholarPaper(
        arxiv_id="2401.00001",
        s2_paper_id="s2id",
        citation_count=10,
        influential_citation_count=2,
        tldr="",
        fields_of_study=("CS",),
        year=2024,
        url="https://example.com",
    )
    refs = [
        semantic_scholar_module.CitationEntry(
            s2_paper_id="ref-1",
            arxiv_id="2401.00002",
            title="Referenced",
            authors="A. Author",
            year=2023,
            citation_count=5,
            url="https://arxiv.org/abs/2401.00002",
        )
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)

        semantic_scholar_module.save_s2_paper(db_path, paper)
        loaded = semantic_scholar_module.load_s2_paper(db_path, "2401.00001")
        assert loaded is not None

        semantic_scholar_module.save_s2_citation_graph(db_path, "s2id", "references", refs)
        loaded_refs = semantic_scholar_module.load_s2_citation_graph(db_path, "s2id", "references")
        assert loaded_refs

        gc.collect()

    assert opened
    assert all(conn.closed for conn in opened)
    _assert_no_resource_warnings(caught)
