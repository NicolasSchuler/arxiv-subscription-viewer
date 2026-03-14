"""Integration tests for the full LLM pipeline using real CLIProvider with echo commands.

Unlike the unit tests in test_services_llm.py (which mock providers), these tests
exercise the real subprocess pipeline: prompt → CLIProvider → subprocess → parse → result.
Cache roundtrip tests verify the SQLite persistence layer end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import arxiv_browser.llm as llm_module
from arxiv_browser.llm_providers import CLIProvider
from arxiv_browser.services.llm_service import (
    generate_summary,
    score_relevance_once,
    suggest_tags_once,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


async def _noop_fetcher(_paper: object) -> str:
    """Dummy paper content fetcher that returns a fixed string."""
    return "Full paper text for testing."


# ── Pipeline: summary via echo ──────────────────────────────────────────────


async def test_summary_pipeline_echo_provider(make_paper) -> None:
    """CLIProvider(echo) → generate_summary → output contains expected text.

    Exercises the full path: prompt building, subprocess exec, stdout capture,
    and result propagation — no mocks.
    """
    paper = make_paper(
        arxiv_id="2401.00001",
        date="2024-01-01",
        title="Test",
        authors="Auth",
        categories="cs.CL",
        abstract="Test abstract",
        url="http://arxiv.org/abs/2401.00001",
    )
    # The #-comment trick makes shlex.split(posix=True) drop {prompt} so the
    # echo output is deterministic regardless of the actual prompt content.
    provider = CLIProvider("echo 'This is a test summary about transformers' #{prompt}")

    summary, error = await generate_summary(
        paper=paper,
        prompt_template="{title}\n{paper_content}",
        provider=provider,
        use_full_paper_content=False,
        summary_timeout_seconds=10,
        fetch_paper_content=_noop_fetcher,
    )

    assert error is None
    assert summary is not None
    assert "test summary" in summary


# ── Pipeline: relevance scoring via echo ────────────────────────────────────


async def test_relevance_pipeline_with_json_echo(make_paper) -> None:
    """CLIProvider(echo JSON) → score_relevance_once → parsed (score, reason).

    The echo command outputs valid JSON; the service layer parses it into a
    (int, str) tuple via _parse_relevance_response.
    """
    paper = make_paper(
        arxiv_id="2401.00001",
        date="2024-01-01",
        title="Test",
        authors="Auth",
        categories="cs.CL",
        abstract="Test abstract",
        url="http://arxiv.org/abs/2401.00001",
    )
    provider = CLIProvider("""echo '{"score": 7, "reason": "relevant to NLP"}' #{prompt}""")

    result = await score_relevance_once(
        paper=paper,
        interests="natural language processing",
        provider=provider,
        timeout_seconds=10,
    )

    assert result is not None
    score, reason = result
    assert score == 7
    assert reason == "relevant to NLP"


# ── Pipeline: auto-tagging via echo ─────────────────────────────────────────


async def test_auto_tag_pipeline_with_json_echo(make_paper) -> None:
    """CLIProvider(echo JSON) → suggest_tags_once → parsed tag list.

    The echo command outputs a JSON object with a "tags" array; the service
    layer parses and normalises it via _parse_auto_tag_response.
    """
    paper = make_paper(
        arxiv_id="2401.00001",
        date="2024-01-01",
        title="Test",
        authors="Auth",
        categories="cs.CL",
        abstract="Test abstract",
        url="http://arxiv.org/abs/2401.00001",
    )
    provider = CLIProvider("""echo '{"tags": ["topic:nlp", "method:attention"]}' #{prompt}""")

    result = await suggest_tags_once(
        paper=paper,
        taxonomy=["topic:nlp"],
        provider=provider,
        timeout_seconds=10,
    )

    assert result is not None
    assert "topic:nlp" in result
    assert "method:attention" in result


# ── Cache roundtrip: summary persistence ────────────────────────────────────


def test_summary_cache_roundtrip(tmp_path: Path) -> None:
    """_save_summary → _load_summary roundtrip through a real SQLite database.

    Uses tmp_path so the DB is created fresh and cleaned up automatically.
    """
    db_path = tmp_path / "summaries.db"
    arxiv_id = "2401.00001"
    command_hash = "abc123"
    summary_text = "A concise summary of transformer architectures."

    # DB does not exist yet — load returns None
    assert llm_module._load_summary(db_path, arxiv_id, command_hash) is None

    # Save and reload
    llm_module._save_summary(db_path, arxiv_id, summary_text, command_hash)
    loaded = llm_module._load_summary(db_path, arxiv_id, command_hash)

    assert loaded == summary_text


def test_summary_cache_different_hash_misses(tmp_path: Path) -> None:
    """Loading with a different command_hash returns None (config-change invalidation)."""
    db_path = tmp_path / "summaries.db"
    arxiv_id = "2401.00001"

    llm_module._save_summary(db_path, arxiv_id, "old summary", "hash_v1")

    assert llm_module._load_summary(db_path, arxiv_id, "hash_v2") is None


# ── Cache roundtrip: relevance score persistence ────────────────────────────


def test_relevance_cache_roundtrip(tmp_path: Path) -> None:
    """_save_relevance_score → _load_relevance_score roundtrip."""
    db_path = tmp_path / "relevance.db"
    arxiv_id = "2401.00001"
    interests_hash = "interests_abc"

    assert llm_module._load_relevance_score(db_path, arxiv_id, interests_hash) is None

    llm_module._save_relevance_score(db_path, arxiv_id, interests_hash, 8, "Strong match")
    loaded = llm_module._load_relevance_score(db_path, arxiv_id, interests_hash)

    assert loaded == (8, "Strong match")


def test_relevance_cache_overwrite(tmp_path: Path) -> None:
    """Saving twice with the same key overwrites (INSERT OR REPLACE)."""
    db_path = tmp_path / "relevance.db"
    arxiv_id = "2401.00001"
    interests_hash = "interests_abc"

    llm_module._save_relevance_score(db_path, arxiv_id, interests_hash, 3, "Weak")
    llm_module._save_relevance_score(db_path, arxiv_id, interests_hash, 9, "Strong")

    loaded = llm_module._load_relevance_score(db_path, arxiv_id, interests_hash)
    assert loaded == (9, "Strong")
