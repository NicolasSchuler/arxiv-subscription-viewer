"""Regression tests for optional rapidfuzz support."""

from __future__ import annotations

from unittest.mock import MagicMock

import arxiv_browser.fuzzy as fuzzy
from arxiv_browser.browser.constants import FUZZY_SCORE_CUTOFF
from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.modals.search import CommandPaletteModal
from arxiv_browser.palette import PaletteCommand


class _PaletteListStub:
    def __init__(self) -> None:
        self.options: list[object] = []
        self.option_count = 0
        self.highlighted: int | None = None

    def clear_options(self) -> None:
        self.options.clear()
        self.option_count = 0

    def add_option(self, option: object) -> None:
        self.options.append(option)
        self.option_count = len(self.options)

    def get_option_at_index(self, index: int) -> object:
        return self.options[index]


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


def test_command_palette_fallback_preserves_partial_matches(monkeypatch) -> None:
    monkeypatch.setattr(fuzzy, "_rapidfuzz_fuzz", None)
    commands = [
        PaletteCommand("Open Paper", "Open selected paper", "o", "open", "Core"),
        PaletteCommand("Toggle Watch", "Toggle watch filter", "w", "watch", "Organize"),
    ]
    palette = CommandPaletteModal(commands)
    palette_list = _PaletteListStub()
    palette.query_one = MagicMock(return_value=palette_list)

    palette._populate_results("watch")

    assert palette._filtered == [commands[1]]
    assert palette_list.option_count == 1
