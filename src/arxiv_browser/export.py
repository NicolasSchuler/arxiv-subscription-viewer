"""Export formatting — BibTeX, RIS, CSV, Markdown, clipboard, PDF paths."""

from __future__ import annotations

import csv
import io
import os
import re
from datetime import datetime
from pathlib import Path

from arxiv_browser.models import (
    STOPWORDS,
    Paper,
    PaperCollection,
    PaperMetadata,
    UserConfig,
)

# BibTeX export settings
DEFAULT_BIBTEX_EXPORT_DIR = "arxiv-exports"  # Default subdirectory in home folder

# PDF download settings
DEFAULT_PDF_DOWNLOAD_DIR = "arxiv-pdfs"  # Relative to home directory

# Matches 4-digit years (2000-2099) for BibTeX export
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def get_pdf_download_path(paper: Paper, config: UserConfig) -> Path:
    """Get the local file path for a downloaded PDF.

    Validates that the resulting path stays within the download directory
    to prevent path traversal attacks via crafted arXiv IDs.

    Args:
        paper: The paper to get the download path for.
        config: User configuration with optional custom download directory.

    Returns:
        Path where the PDF should be saved.

    Raises:
        ValueError: If the arXiv ID would escape the download directory.
    """
    if config.pdf_download_dir:
        base_dir = Path(config.pdf_download_dir).resolve()
    else:
        base_dir = (Path.home() / DEFAULT_PDF_DOWNLOAD_DIR).resolve()
    result = (base_dir / f"{paper.arxiv_id}.pdf").resolve()
    # Ensure the resolved path is still under the base directory
    if not str(result).startswith(str(base_dir) + os.sep) and result.parent != base_dir:
        raise ValueError(f"Invalid arXiv ID for path construction: {paper.arxiv_id!r}")
    return result


# ============================================================================
# BibTeX Formatting Functions (extracted for testability)
# ============================================================================


def escape_bibtex(text: str) -> str:
    """Escape special characters for BibTeX."""
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_authors_bibtex(authors: str) -> str:
    """Escape author string for BibTeX output."""
    return escape_bibtex(authors)


def extract_year(date_str: str) -> str:
    """Extract year from date string, with fallback to current year.

    Args:
        date_str: Date string like "Mon, 15 Jan 2024".

    Returns:
        4-digit year string, or current year if not found.
    """
    current_year = str(datetime.now().year)

    if not date_str or not date_str.strip():
        return current_year

    year_match = _YEAR_PATTERN.search(date_str)
    if year_match:
        return year_match.group(1)

    return current_year


def generate_citation_key(paper: Paper) -> str:
    """Generate a BibTeX citation key like 'smith2024attention'."""
    authors = paper.authors.split(",")[0].strip()
    parts = authors.split()
    last_name = parts[-1].lower() if parts else "unknown"
    last_name = "".join(c for c in last_name if c.isalnum())

    year = extract_year(paper.date)

    title_words = paper.title.lower().split()
    first_word = "paper"
    for word in title_words:
        clean_word = "".join(c for c in word if c.isalnum())
        if clean_word and clean_word not in STOPWORDS:
            first_word = clean_word
            break

    return f"{last_name}{year}{first_word}"


def format_paper_as_bibtex(paper: Paper) -> str:
    """Format a paper as a BibTeX @misc entry."""
    key = generate_citation_key(paper)
    year = extract_year(paper.date)
    categories_list = paper.categories.split()
    primary_class = categories_list[0] if categories_list else "misc"
    lines = [
        f"@misc{{{key},",
        f"  title = {{{escape_bibtex(paper.title)}}},",
        f"  author = {{{format_authors_bibtex(paper.authors)}}},",
        f"  year = {{{year}}},",
        f"  eprint = {{{paper.arxiv_id}}},",
        "  archivePrefix = {arXiv},",
        f"  primaryClass = {{{primary_class}}},",
        f"  url = {{{paper.url}}},",
        "}",
    ]
    return "\n".join(lines)


def format_paper_as_ris(paper: Paper, abstract_text: str = "") -> str:
    """Format a paper as a RIS (Research Information Systems) entry.

    RIS is a standard interchange format supported by reference managers
    such as EndNote, Mendeley, and Zotero.
    """
    lines = [
        "TY  - ELEC",
        f"TI  - {paper.title}",
    ]
    for author in paper.authors.split(","):
        author = author.strip()
        if author:
            lines.append(f"AU  - {author}")
    year = extract_year(paper.date)
    lines.append(f"PY  - {year}")
    lines.append(f"UR  - {paper.url}")
    lines.extend(f"KW  - {cat}" for cat in paper.categories.split() if cat)
    if abstract_text:
        lines.append(f"AB  - {abstract_text}")
    if paper.comments:
        lines.append(f"N2  - {paper.comments}")
    lines.append(f"N1  - arXiv:{paper.arxiv_id}")
    lines.append("ER  - ")
    return "\n".join(lines)


def format_papers_as_csv(
    papers: list[Paper],
    metadata: dict[str, PaperMetadata] | None = None,
) -> str:
    """Format papers as CSV with optional metadata columns.

    Uses csv.writer for proper quoting and escaping. Tags are joined
    with semicolons within a single cell.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "arxiv_id",
        "title",
        "authors",
        "categories",
        "date",
        "url",
        "comments",
    ]
    if metadata is not None:
        header.extend(["starred", "read", "tags", "notes"])
    writer.writerow(header)
    for paper in papers:
        row: list[str] = [
            paper.arxiv_id,
            paper.title,
            paper.authors,
            paper.categories,
            paper.date,
            paper.url,
            paper.comments or "",
        ]
        if metadata is not None:
            meta = metadata.get(paper.arxiv_id)
            if meta:
                row.extend(
                    [
                        str(meta.starred).lower(),
                        str(meta.is_read).lower(),
                        ";".join(meta.tags),
                        meta.notes,
                    ]
                )
            else:
                row.extend(["false", "false", "", ""])
        writer.writerow(row)
    return output.getvalue()


def format_papers_as_markdown_table(papers: list[Paper]) -> str:
    """Format papers as a compact Markdown table.

    Pipe characters in fields are escaped. Authors are truncated to the
    first author + 'et al.' if there are more than 3 authors.
    """
    lines = [
        "| arXiv ID | Title | Authors | Categories | Date |",
        "|----------|-------|---------|------------|------|",
    ]
    for paper in papers:
        # Escape pipe characters in fields
        title = paper.title.replace("|", "\\|")
        categories = paper.categories.replace("|", "\\|")
        date = paper.date.replace("|", "\\|")

        # Truncate authors to first + et al. if >3
        author_list = [a.strip() for a in paper.authors.split(",") if a.strip()]
        authors_str = f"{author_list[0]} et al." if len(author_list) > 3 else ", ".join(author_list)
        authors_str = authors_str.replace("|", "\\|")

        arxiv_link = f"[{paper.arxiv_id}]({paper.url})"
        lines.append(f"| {arxiv_link} | {title} | {authors_str} | {categories} | {date} |")
    return "\n".join(lines)


def get_pdf_url(paper: Paper) -> str:
    """Get the PDF URL for a paper."""
    if "arxiv.org/pdf/" in paper.url:
        return paper.url if paper.url.endswith(".pdf") else f"{paper.url}.pdf"
    return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"


def get_paper_url(paper: Paper, prefer_pdf: bool = False) -> str:
    """Get the preferred URL for a paper (abs or PDF)."""
    if prefer_pdf:
        return get_pdf_url(paper)
    return paper.url


def format_paper_for_clipboard(paper: Paper, abstract_text: str = "") -> str:
    """Format a paper's metadata for clipboard export."""
    lines = [
        f"Title: {paper.title}",
        f"Authors: {paper.authors}",
        f"arXiv: {paper.arxiv_id}",
        f"Date: {paper.date}",
        f"Categories: {paper.categories}",
    ]
    if paper.comments:
        lines.append(f"Comments: {paper.comments}")
    lines.append(f"URL: {paper.url}")
    lines.append("")
    lines.append(f"Abstract: {abstract_text}")
    return "\n".join(lines)


def format_paper_as_markdown(paper: Paper, abstract_text: str = "") -> str:
    """Format a paper as Markdown."""
    lines = [
        f"## {paper.title}",
        "",
        f"**arXiv:** [{paper.arxiv_id}]({paper.url})",
        f"**Date:** {paper.date}",
        f"**Categories:** {paper.categories}",
        f"**Authors:** {paper.authors}",
    ]
    if paper.comments:
        lines.append(f"**Comments:** {paper.comments}")
    lines.extend(
        [
            "",
            "### Abstract",
            "",
            abstract_text,
        ]
    )
    return "\n".join(lines)


def format_collection_as_markdown(
    collection: PaperCollection,
    papers_by_id: dict[str, Paper],
) -> str:
    """Format a paper collection as Markdown."""
    lines = [f"# {collection.name}"]
    if collection.description:
        lines.append(f"\n{collection.description}")
    lines.append(f"\n*{len(collection.paper_ids)} papers*\n")
    for pid in collection.paper_ids:
        paper = papers_by_id.get(pid)
        if paper:
            lines.append(format_paper_as_markdown(paper))
            lines.append("")
        else:
            lines.append(f"- {pid} (not loaded)\n")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BIBTEX_EXPORT_DIR",
    "DEFAULT_PDF_DOWNLOAD_DIR",
    "escape_bibtex",
    "extract_year",
    "format_authors_bibtex",
    "format_collection_as_markdown",
    "format_paper_as_bibtex",
    "format_paper_as_markdown",
    "format_paper_as_ris",
    "format_paper_for_clipboard",
    "format_papers_as_csv",
    "format_papers_as_markdown_table",
    "generate_citation_key",
    "get_paper_url",
    "get_pdf_download_path",
    "get_pdf_url",
]
