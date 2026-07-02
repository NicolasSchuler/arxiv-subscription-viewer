"""Regression tests for optional rapidfuzz support."""

from __future__ import annotations

import arxiv_browser.fuzzy as fuzzy
from arxiv_browser.browser.constants import FUZZY_SCORE_CUTOFF
from arxiv_browser.browser.core import ArxivBrowser


def test_fuzzy_fallback_preserves_substring_title_matches(make_paper, monkeypatch) -> None:
    monkeypatch.setattr(fuzzy, "_rapidfuzz_fuzz", None)
    matching = make_paper(
        arxiv_id="2401.00001",
        title="Efficient transformer architectures for long context models",
        authors="Alice Example",
    )
    unrelated = make_paper(
        arxiv_id="2401.00002",
        title="Bayesian sampling for inverse problems",
        authors="Bob Example",
    )
    app = ArxivBrowser([matching, unrelated], restore_session=False)

    result = app._fuzzy_search("transformer")

    assert result == [matching]
    assert app._match_scores[matching.arxiv_id] >= FUZZY_SCORE_CUTOFF
    assert unrelated.arxiv_id not in app._match_scores
