"""Shared test fixtures for arXiv Browser tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import arxiv_browser.browser.constants as browser_constants
import arxiv_browser.browser.core as browser_core
from arxiv_browser.models import (
    Paper,
    PaperMetadata,
    SearchBookmark,
    SessionState,
    UserConfig,
    WatchListEntry,
)
from arxiv_browser.query import (
    _HIGHLIGHT_PATTERN_CACHE,
    format_categories,
)
from arxiv_browser.widgets.chrome import set_ascii_glyphs as set_chrome_ascii_glyphs
from arxiv_browser.widgets.details import set_ascii_glyphs as set_detail_ascii_glyphs
from arxiv_browser.widgets.listing import set_ascii_icons

# ── Cache isolation ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global_dicts():
    """Clear global caches that are intentionally process-wide between tests."""
    yield
    format_categories.cache_clear()
    _HIGHLIGHT_PATTERN_CACHE.clear()
    set_ascii_icons(False)
    set_detail_ascii_glyphs(False)
    set_chrome_ascii_glyphs(False)


@pytest.fixture(autouse=True)
def _shorten_browser_debounce_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce UI timer windows so Textual integration tests are less host-sensitive."""
    monkeypatch.setattr(browser_core, "SEARCH_DEBOUNCE_DELAY", 0.05)
    monkeypatch.setattr(browser_core, "DETAIL_PANE_DEBOUNCE_DELAY", 0.01)
    monkeypatch.setattr(browser_core, "BADGE_COALESCE_DELAY", 0.01)
    monkeypatch.setattr(browser_constants, "BADGE_COALESCE_DELAY", 0.01)


# ── Factories ────────────────────────────────────────────────────────────────


@pytest.fixture
def make_paper():
    """Factory fixture for creating Paper instances with sensible defaults.

    Always sets abstract_raw = abstract to prevent async HTTP fetches during tests.
    """

    def _make(
        arxiv_id: str = "2401.12345",
        date: str = "Mon, 15 Jan 2024",
        title: str = "Test Paper",
        authors: str = "Test Author",
        categories: str = "cs.AI",
        comments: str | None = None,
        abstract: str = "Test abstract content.",
        url: str | None = None,
        abstract_raw: str | None = None,
    ) -> Paper:
        if url is None:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        if abstract_raw is None:
            abstract_raw = abstract
        return Paper(
            arxiv_id=arxiv_id,
            date=date,
            title=title,
            authors=authors,
            categories=categories,
            comments=comments,
            abstract=abstract,
            url=url,
            abstract_raw=abstract_raw,
        )

    return _make


@pytest.fixture
def sample_config():
    """Factory fixture for creating UserConfig with optional overrides."""

    def _make(**kwargs: Any) -> UserConfig:
        return UserConfig(**kwargs)

    return _make


@pytest.fixture
def history_dir(tmp_path):
    """Create a temp history/ directory with a few date files for testing."""
    hdir = tmp_path / "history"
    hdir.mkdir()

    dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
    for d in dates:
        (hdir / f"{d}.txt").write_text(
            f"arXiv:2401.00001\nDate: Mon, 15 Jan 2024\nTitle: Paper {d}\n"
        )

    return tmp_path  # Return base dir (discover_history_files expects parent of history/)
