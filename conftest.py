"""Shared test fixtures for arXiv Browser tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arxiv_browser import (
    CATEGORY_COLORS,
    DEFAULT_CATEGORY_COLORS,
    THEME_COLORS,
    Paper,
    PaperMetadata,
    SearchBookmark,
    SessionState,
    UserConfig,
    WatchListEntry,
    format_categories,
)


# ── Module-level dict isolation ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global_dicts():
    """Restore CATEGORY_COLORS, THEME_COLORS, and clear LRU cache after each test.

    ArxivBrowser.__init__ mutates these module-level dicts. Without this fixture
    tests that instantiate ArxivBrowser would pollute the state for later tests.
    """
    yield
    # Restore module-level dicts
    from arxiv_browser import DEFAULT_THEME

    CATEGORY_COLORS.clear()
    CATEGORY_COLORS.update(DEFAULT_CATEGORY_COLORS)
    THEME_COLORS.clear()
    THEME_COLORS.update(DEFAULT_THEME)
    # Clear LRU cache that captured stale color values
    format_categories.cache_clear()


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
