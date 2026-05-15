from __future__ import annotations

import pytest

from arxiv_browser.enrichment import get_starred_paper_ids_for_version_check
from arxiv_browser.models import Paper, PaperMetadata
from arxiv_browser.sources import (
    BIORXIV_PROVIDER,
    MEDRXIV_PROVIDER,
    is_arxiv_identifier,
    is_arxiv_paper,
    parse_biorxiv_details_payload,
    provider_display_name,
)
from arxiv_browser.widgets import render_paper_option

BIORXIV_DETAILS_FIXTURE = {
    "messages": [{"count": 1, "cursor": 0}],
    "collection": [
        {
            "doi": "10.1101/2026.01.02.123456",
            "title": "Biological Discovery with Graph Models",
            "authors": "Alice Smith; Bob Jones",
            "date": "2026-01-02",
            "version": "2",
            "category": "cell biology",
            "abstract": "Graph models reveal biological structure.",
            "server": "biorxiv",
        }
    ],
}


def test_biorxiv_details_payload_maps_provider_without_reusing_source() -> None:
    papers = parse_biorxiv_details_payload(BIORXIV_DETAILS_FIXTURE, BIORXIV_PROVIDER)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "10.1101/2026.01.02.123456"
    assert paper.provider == BIORXIV_PROVIDER
    assert paper.source == "api"
    assert paper.authors == "Alice Smith, Bob Jones"
    assert paper.date == "Fri, 02 Jan 2026"
    assert paper.url == "https://www.biorxiv.org/content/10.1101/2026.01.02.123456v2"
    assert not is_arxiv_paper(paper)


def test_medrxiv_parser_and_provider_display_edges() -> None:
    payload = {
        "collection": [
            {
                "doi": "10.1101/2026.02.03.111111",
                "title": "Clinical Signals",
                "authors": "Dana Lee",
                "date": "bad-date",
                "category": "epidemiology",
                "abstract": "",
            },
            "not-a-dict",
            {"title": "missing DOI"},
        ]
    }

    papers = parse_biorxiv_details_payload(payload, MEDRXIV_PROVIDER)

    assert len(papers) == 1
    assert papers[0].provider == MEDRXIV_PROVIDER
    assert papers[0].date == "bad-date"
    assert provider_display_name("unknown-source") == "unknown-source"
    assert parse_biorxiv_details_payload({"collection": "oops"}, MEDRXIV_PROVIDER) == []
    with pytest.raises(ValueError, match="Unsupported"):
        parse_biorxiv_details_payload({}, "ssrn")  # type: ignore[arg-type]


def test_version_check_filters_non_arxiv_identifiers() -> None:
    metadata = {
        "2401.12345": PaperMetadata("2401.12345", starred=True),
        "hep-th/9901001": PaperMetadata("hep-th/9901001", starred=True),
        "10.1101/2026.01.02.123456": PaperMetadata("10.1101/2026.01.02.123456", starred=True),
    }

    assert get_starred_paper_ids_for_version_check(metadata) == {
        "2401.12345",
        "hep-th/9901001",
    }
    assert is_arxiv_identifier("2401.12345")
    assert not is_arxiv_identifier("10.1101/2026.01.02.123456")


def test_provider_labels_do_not_break_existing_api_badge(make_paper) -> None:
    api_paper = make_paper()
    api_paper.source = "api"
    bio_paper = Paper(
        arxiv_id="10.1101/2026.01.02.123456",
        date="Fri, 02 Jan 2026",
        title="Bio Paper",
        authors="Alice",
        categories="cell_biology",
        comments=None,
        abstract="Abstract",
        url="https://www.biorxiv.org/content/10.1101/2026.01.02.123456v1",
        abstract_raw="Abstract",
        source="api",
        provider=BIORXIV_PROVIDER,
    )

    assert "API" in render_paper_option(api_paper)
    rendered = render_paper_option(bio_paper)
    assert "bioRxiv" in rendered
    assert "API" not in rendered
