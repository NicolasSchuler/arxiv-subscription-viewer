"""Focused tests for list rendering helpers/widgets."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.css.query import NoMatches

from arxiv_browser.app import PaperListItem, PaperMetadata, render_paper_option
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.widgets.listing import set_ascii_icons


def test_set_ascii_icons_changes_rendered_selection_marker(make_paper):
    paper = make_paper(title="Selection Test")

    set_ascii_icons(True)
    ascii_text = render_paper_option(paper, selected=True)
    assert "[x]" in ascii_text

    set_ascii_icons(False)
    unicode_text = render_paper_option(paper, selected=True)
    assert "●" in unicode_text


def test_paper_list_item_title_and_authors_text(make_paper):
    paper = make_paper(title="Transformer Study", authors="Alice Author")
    metadata = PaperMetadata(arxiv_id=paper.arxiv_id, starred=True, is_read=True)
    item = PaperListItem(
        paper,
        selected=True,
        watched=True,
        metadata=metadata,
        highlight_terms={"title": ["Transformer"], "author": ["Author"], "abstract": []},
    )

    title_text = item._get_title_text()
    assert "●" in title_text
    assert "👁" in title_text
    assert "⭐" in title_text
    assert "✓" in title_text
    assert "[dim]" in title_text

    authors_text = item._get_authors_text()
    assert "Alice" in authors_text


def test_paper_list_item_meta_badges_with_all_sources(make_paper):
    paper = make_paper()
    paper.source = "api"
    metadata = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["topic:ml"])
    item = PaperListItem(paper, metadata=metadata)

    item._s2_data = SemanticScholarPaper(
        arxiv_id=paper.arxiv_id,
        s2_paper_id="s2id",
        citation_count=42,
        influential_citation_count=5,
        tldr="",
        fields_of_study=(),
        year=2024,
        url="https://s2.example",
    )
    item._hf_data = HuggingFacePaper(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        upvotes=55,
        num_comments=3,
        ai_summary="",
        ai_keywords=(),
        github_repo="",
        github_stars=0,
    )
    item._version_update = (1, 2)
    item._relevance_score = (9, "very relevant")

    meta_text = item._get_meta_text()
    assert "API" in meta_text
    assert "#topic:ml" in meta_text
    assert "C42" in meta_text
    assert "↑55" in meta_text
    assert "v1→v2" in meta_text
    assert "9/10" in meta_text


def test_paper_list_item_preview_text_branches(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)

    item._abstract_text = None
    assert "Loading abstract" in item._get_preview_text()

    item._abstract_text = ""
    assert "No abstract available" in item._get_preview_text()

    item._abstract_text = "Short abstract"
    assert "Short abstract" in item._get_preview_text()

    item._abstract_text = "word " * 80
    preview = item._get_preview_text()
    assert preview.endswith("...[/]")


def test_paper_list_item_setters_and_selection_refresh(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)
    item._update_display = MagicMock()

    item.set_metadata(PaperMetadata(arxiv_id=item.paper.arxiv_id, tags=["status:todo"]))
    item.set_abstract_text("preview")
    item.update_s2_data(None)
    item.update_hf_data(None)
    item.update_version_data((2, 3))
    item.update_relevance_data((7, "good"))

    item.set_selected(True)
    assert item.is_selected is True
    assert item.toggle_selected() is False

    assert item._update_display.call_count >= 8


def test_paper_list_item_update_display_handles_no_matches(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)
    item.query_one = MagicMock(side_effect=NoMatches("missing"))

    item._update_display()


def test_paper_list_item_update_display_updates_widgets(make_paper):
    item = PaperListItem(make_paper(), show_preview=True)

    title_widget = MagicMock()
    authors_widget = MagicMock()
    meta_widget = MagicMock()
    preview_widget = MagicMock()

    def _query(selector, _type=None):
        mapping = {
            ".paper-title": title_widget,
            ".paper-authors": authors_widget,
            ".paper-meta": meta_widget,
            ".paper-preview": preview_widget,
        }
        return mapping[selector]

    item.query_one = MagicMock(side_effect=_query)
    item._update_display()

    title_widget.update.assert_called_once()
    authors_widget.update.assert_called_once()
    meta_widget.update.assert_called_once()
    preview_widget.update.assert_called_once()


def test_paper_list_item_compose_respects_preview_flag(make_paper):
    no_preview = PaperListItem(make_paper(), show_preview=False)
    with_preview = PaperListItem(make_paper(), show_preview=True)

    assert len(list(no_preview.compose())) == 3
    assert len(list(with_preview.compose())) == 4
