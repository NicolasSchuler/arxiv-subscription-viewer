"""Preprint provider identity and prototype parsers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from arxiv_browser.models import ARXIV_DATE_FORMAT, Paper
from arxiv_browser.parsing import clean_latex

ARXIV_PROVIDER = "arxiv"
BIORXIV_PROVIDER = "biorxiv"
MEDRXIV_PROVIDER = "medrxiv"
SUPPORTED_PREPRINT_PROVIDERS = (ARXIV_PROVIDER, BIORXIV_PROVIDER, MEDRXIV_PROVIDER)
_ARXIV_IDENTIFIER_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7})$")


@dataclass(slots=True, frozen=True)
class PreprintProvider:
    """Provider identity separated from the local/API source marker."""

    provider_id: str
    display_name: str
    api_base_url: str


PREPRINT_PROVIDERS: dict[str, PreprintProvider] = {
    ARXIV_PROVIDER: PreprintProvider(ARXIV_PROVIDER, "arXiv", "https://export.arxiv.org/api"),
    BIORXIV_PROVIDER: PreprintProvider(BIORXIV_PROVIDER, "bioRxiv", "https://api.biorxiv.org"),
    MEDRXIV_PROVIDER: PreprintProvider(MEDRXIV_PROVIDER, "medRxiv", "https://api.medrxiv.org"),
}


def is_arxiv_identifier(identifier: str) -> bool:
    """Return whether an identifier is syntactically an arXiv identifier."""
    return bool(_ARXIV_IDENTIFIER_RE.match(identifier.strip()))


def is_arxiv_paper(paper: Paper) -> bool:
    """Return whether a paper belongs to the arXiv provider."""
    return getattr(paper, "provider", ARXIV_PROVIDER) == ARXIV_PROVIDER


def provider_display_name(provider_id: str) -> str:
    """Return compact display copy for a provider ID."""
    provider = PREPRINT_PROVIDERS.get(provider_id)
    return provider.display_name if provider else provider_id


def parse_biorxiv_details_payload(
    payload: Mapping[str, Any],
    server: Literal["biorxiv", "medrxiv"],
) -> list[Paper]:
    """Parse a bioRxiv/medRxiv details API JSON payload into ``Paper`` rows."""
    if server not in {BIORXIV_PROVIDER, MEDRXIV_PROVIDER}:
        raise ValueError(f"Unsupported bioRxiv details server: {server}")
    collection = payload.get("collection", [])
    if not isinstance(collection, list):
        return []
    papers = [_parse_biorxiv_entry(entry, server) for entry in collection]
    return [paper for paper in papers if paper is not None]


def _parse_biorxiv_entry(entry: Any, server: str) -> Paper | None:
    if not isinstance(entry, Mapping):
        return None
    doi = _clean_field(entry.get("doi"))
    if not doi:
        return None
    title = clean_latex(_clean_field(entry.get("title")))
    authors = _clean_authors(entry.get("authors"))
    category = _clean_field(entry.get("category")).replace(" ", "_")
    abstract = clean_latex(_clean_field(entry.get("abstract")))
    version = _clean_field(entry.get("version")) or "1"
    return Paper(
        arxiv_id=doi,
        date=_format_provider_date(_clean_field(entry.get("date"))),
        title=title,
        authors=authors,
        categories=category,
        comments=None,
        abstract=abstract,
        abstract_raw=_clean_field(entry.get("abstract")),
        url=f"https://www.{server}.org/content/{doi}v{version}",
        source="api",
        provider=server,
    )


def _clean_authors(value: Any) -> str:
    text = _clean_field(value)
    if ";" in text:
        return ", ".join(part.strip() for part in text.split(";") if part.strip())
    return text


def _clean_field(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _format_provider_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").strftime(ARXIV_DATE_FORMAT)
    except ValueError:
        return raw


__all__ = [
    "ARXIV_PROVIDER",
    "BIORXIV_PROVIDER",
    "MEDRXIV_PROVIDER",
    "PREPRINT_PROVIDERS",
    "SUPPORTED_PREPRINT_PROVIDERS",
    "PreprintProvider",
    "is_arxiv_identifier",
    "is_arxiv_paper",
    "parse_biorxiv_details_payload",
    "provider_display_name",
]
