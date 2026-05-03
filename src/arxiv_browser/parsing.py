"""arXiv email parsing, HTML text extraction, LaTeX cleaning, and history discovery."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

from arxiv_browser.models import (
    ARXIV_DATE_FORMAT,
    Paper,
    PaperMetadata,
    parse_arxiv_date,
)

logger = logging.getLogger(__name__)

# Placeholder using control characters (cannot appear in academic text)
_ESCAPED_DOLLAR = "\x00ESCAPED_DOLLAR\x00"

# Pre-compiled regex patterns for LaTeX cleaning (performance optimization)
# Each tuple is (pattern, replacement) applied in order
_LATEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Text formatting commands: \textbf{text} -> text, \emph{text} -> text
    (re.compile(r"\\text(?:tt|bf|it|rm|sf)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\emph\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:bf|it|tt|rm|sf)\{([^}]*)\}"), r"\1"),
    # Escaped dollar signs: \$ -> placeholder (restored later)
    (re.compile(r"\\\$"), _ESCAPED_DOLLAR),
    # Math mode: $x^2$ -> x^2 (extracts content, non-greedy)
    (re.compile(r"\$([^$]*)\$"), r"\1"),
    # Restore escaped dollar signs: placeholder -> $
    (re.compile(re.escape(_ESCAPED_DOLLAR)), "$"),
    # Accented characters: \'e -> é, \"a -> ä, \c{c} -> ç, etc.
    (re.compile(r"\\c\{c\}"), "ç"),
    (re.compile(r"\\c\{C\}"), "Ç"),
    (re.compile(r"\\'e"), "é"),
    (re.compile(r"\\'a"), "á"),
    (re.compile(r"\\'o"), "ó"),
    (re.compile(r"\\'i"), "í"),
    (re.compile(r"\\'u"), "ú"),
    (re.compile(r'\\"\{a\}'), "ä"),
    (re.compile(r'\\"\{o\}'), "ö"),
    (re.compile(r'\\"\{u\}'), "ü"),
    (re.compile(r"\\~n"), "ñ"),
    (re.compile(r"\\&"), "&"),
    # Generic command with braces: \foo{content} -> content
    (re.compile(r"\\[a-zA-Z]+\{([^}]*)\}"), r"\1"),
    # Standalone commands: \foo -> (removed)
    (re.compile(r"\\[a-zA-Z]+(?:\s|$)"), " "),
]

# Pre-compiled regex patterns for parsing arXiv entries
# Matches modern IDs (2301.12345) and legacy IDs (hep-th/9901001), with optional version suffix.
_ARXIV_ID_PATTERN = re.compile(
    r"arXiv:((?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?)"
)
# Matches: "Date: Mon, 15 Jan 2024 (v1)" -> captures "Mon, 15 Jan 2024"
_DATE_PATTERN = re.compile(r"Date:\s*(.+?)(?:\s*\(|$)", re.MULTILINE)
# Matches multi-line title up to "Authors:" label
_TITLE_PATTERN = re.compile(r"Title:\s*(.+?)(?=\nAuthors:)", re.DOTALL)
# Matches multi-line authors up to "Categories:" label
_AUTHORS_PATTERN = re.compile(r"Authors:\s*(.+?)(?=\nCategories:)", re.DOTALL)
# Matches: "Categories: cs.AI cs.LG" -> captures "cs.AI cs.LG"
_CATEGORIES_PATTERN = re.compile(r"Categories:\s*(.+?)$", re.MULTILINE)
# Matches: "Comments: 10 pages, 5 figures" -> captures "10 pages, 5 figures"
_COMMENTS_PATTERN = re.compile(r"Comments:\s*(.+?)$", re.MULTILINE)
# Matches abstract text between \\ markers after Categories/Comments line
# Uses .*? to handle multi-line Comments fields (continuation lines)
_ABSTRACT_PATTERN = re.compile(r"(?:Categories|Comments):.*?\n\\\\\n(.+?)\n\\\\", re.DOTALL)
# Matches: "( https://arxiv.org/abs/2301.12345" -> captures the URL
_URL_PATTERN = re.compile(r"\(\s*(https://arxiv\.org/abs/\S+)")
# Matches 70+ dashes used as entry separator
_ENTRY_SEPARATOR = re.compile(r"-{70,}")
# Strip trailing version suffix from IDs (e.g., 2401.12345v2 -> 2401.12345)
_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)

# arXiv API / Atom parsing constants
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_QUERY_FIELDS = {
    "all": "all",
    "title": "ti",
    "author": "au",
    "abstract": "abs",
}

# History file date format (ISO 8601)
HISTORY_DATE_FORMAT = "%Y-%m-%d"


def clean_latex(text: str) -> str:
    """Remove or convert common LaTeX commands to plain text.

    Handles nested braces by iteratively applying patterns until no more
    changes occur (e.g., \\textbf{$O(n^{2})$} -> O(n^{2})).

    Args:
        text: Input text potentially containing LaTeX commands.

    Returns:
        Plain text with LaTeX commands removed or converted.
    """
    # Short-circuit: skip regex processing for text without LaTeX markers
    if "\\" not in text and "$" not in text:
        return " ".join(text.split())

    # Iteratively apply patterns to handle nested structures.
    # Fixed-point iteration: repeat until a full pass produces no changes,
    # which guarantees that nested constructs (e.g. \textbf{\it …}) are
    # fully resolved regardless of nesting depth.
    prev_text = None
    while prev_text != text:
        prev_text = text
        for pattern, replacement in _LATEX_PATTERNS:
            text = pattern.sub(replacement, text)

    # Clean up extra whitespace
    return " ".join(text.split())


def parse_arxiv_file(filepath: Path) -> list[Paper]:
    """Parse arxiv.txt and return a list of Paper objects.

    Duplicate arXiv IDs are deduplicated by bare ID (version suffix stripped).
    When multiple versions of the same ID are present, the highest version wins.

    Args:
        filepath: Path to the arXiv email digest file (typically
            ``arxiv.txt``).  Read with ``errors="replace"`` to handle any
            non-UTF-8 bytes gracefully.

    Returns:
        List of ``Paper`` objects parsed from the file.  Each paper has its
        title, authors, and comments cleaned via ``clean_latex``.  The order
        reflects the order in the file (after deduplication).

    Raises:
        OSError: If the file cannot be opened or read.
    """
    import time

    t0 = time.monotonic()
    # Use errors="replace" to handle any non-UTF-8 characters gracefully
    content = filepath.read_text(encoding="utf-8", errors="replace")
    # Dict value is (version, paper) so we can keep the highest-version entry
    papers_by_id: dict[str, tuple[int, Paper]] = {}

    # Split by paper separator using pre-compiled pattern
    entries = _ENTRY_SEPARATOR.split(content)

    for entry in entries:
        parsed = _parse_arxiv_email_entry(entry)
        if parsed is None:
            continue
        bare_arxiv_id, version, paper = parsed
        existing = papers_by_id.get(paper.arxiv_id)
        if existing is None or version > existing[0]:
            papers_by_id[bare_arxiv_id] = (version, paper)

    elapsed = time.monotonic() - t0
    papers = [paper for _, paper in papers_by_id.values()]
    logger.debug("Parsed %d papers from %s in %.3fs", len(papers), filepath.name, elapsed)
    return papers


def _parse_arxiv_email_entry(entry: str) -> tuple[str, int, Paper] | None:
    """Parse one arXiv email digest entry into its dedupe key and Paper."""
    entry = entry.strip()
    if not entry:
        return None

    id_match = _ARXIV_ID_PATTERN.search(entry)
    if not id_match:
        return None
    arxiv_id_raw = id_match.group(1)
    bare_arxiv_id = normalize_arxiv_id(arxiv_id_raw)
    if not bare_arxiv_id:
        return None

    comments = _entry_match_text(entry, _COMMENTS_PATTERN) or None
    paper = Paper(
        arxiv_id=bare_arxiv_id,
        date=_entry_match_text(entry, _DATE_PATTERN),
        title=clean_latex(_entry_match_text(entry, _TITLE_PATTERN)),
        authors=clean_latex(_entry_match_text(entry, _AUTHORS_PATTERN)),
        categories=_entry_match_text(entry, _CATEGORIES_PATTERN),
        comments=clean_latex(comments) if comments else None,
        abstract=None,
        abstract_raw=_entry_match_text(entry, _ABSTRACT_PATTERN),
        url=_entry_url(entry, bare_arxiv_id),
    )
    return bare_arxiv_id, _arxiv_id_version(arxiv_id_raw), paper


def _entry_match_text(entry: str, pattern: re.Pattern[str]) -> str:
    """Return normalized first capture text for an email entry regex."""
    match = pattern.search(entry)
    return " ".join(match.group(1).split()) if match else ""


def _entry_url(entry: str, arxiv_id: str) -> str:
    """Return explicit arXiv URL from an email entry, or the canonical fallback."""
    match = _URL_PATTERN.search(entry)
    return match.group(1) if match else f"https://arxiv.org/abs/{arxiv_id}"


def _arxiv_id_version(arxiv_id_raw: str) -> int:
    """Return explicit arXiv version, defaulting to v1 when absent."""
    version_match = _ARXIV_VERSION_SUFFIX.search(arxiv_id_raw)
    return int(version_match.group(0)[1:]) if version_match else 1


def normalize_arxiv_id(raw: str) -> str:
    """Normalize arXiv IDs from raw IDs or URLs.

    Examples:
    - https://arxiv.org/abs/2401.12345v2 -> 2401.12345
    - https://arxiv.org/pdf/2401.12345v2.pdf -> 2401.12345
    - hep-th/9901001v1 -> hep-th/9901001
    """
    text = raw.strip()
    if not text:
        return ""

    if "arxiv.org" in text:
        for marker in ("/abs/", "/pdf/"):
            idx = text.find(marker)
            if idx >= 0:
                text = text[idx + len(marker) :]
                break

    text = text.split("?", 1)[0].split("#", 1)[0].strip().strip("/")
    text = text.removesuffix(".pdf")
    return _ARXIV_VERSION_SUFFIX.sub("", text)


def _format_arxiv_api_date(raw_date: str) -> str:
    """Convert Atom timestamp to the app's date format.

    Args:
        raw_date: ISO 8601 datetime string from the Atom feed (e.g.
            ``"2024-01-15T00:00:00Z"`` or ``"2024-01-15T00:00:00+00:00"``).

    Returns:
        Date string formatted as ``ARXIV_DATE_FORMAT`` (e.g.
        ``"Mon, 15 Jan 2024"``), or the original ``raw_date`` if parsing
        fails.  Returns ``""`` for blank input.
    """
    cleaned = raw_date.strip()
    if not cleaned:
        return ""

    normalized = cleaned
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.strftime(ARXIV_DATE_FORMAT)
    except ValueError:
        return cleaned


def _atom_text(node: Any, path: str) -> str:
    """Extract normalized text from an Atom XML node path.

    Args:
        node: An ``ElementTree`` element to search within.
        path: XPath-style path string (using the ``ATOM_NS`` namespace map).

    Returns:
        Whitespace-normalized text of the first matching descendant, or ``""``
        if the path matches nothing or the element has no text.
    """
    found = node.find(path, ATOM_NS)
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())


def build_arxiv_search_query(query: str, field: str = "all", category: str = "") -> str:
    """Build an arXiv API search query string.

    Args:
        query: User search text (can be empty if ``category`` is provided).
            Double-quotes are stripped to avoid breaking the query syntax.
        field: One of: ``all``, ``title``, ``author``, ``abstract``.
        category: Optional category filter such as ``"cs.AI"``.

    Returns:
        A query string ready for the arXiv API ``search_query`` parameter,
        e.g. ``"ti:transformers AND cat:cs.LG"``.

    Raises:
        ValueError: If ``field`` is not a recognised key in
            ``ARXIV_QUERY_FIELDS``, or if both ``query`` and ``category`` are
            empty after stripping.
    """
    field_key = field.strip().lower()
    if field_key not in ARXIV_QUERY_FIELDS:
        raise ValueError(
            f"Unsupported arXiv search field: {field!r}. "
            f"Expected one of: {', '.join(sorted(ARXIV_QUERY_FIELDS))}"
        )

    query_clean = " ".join(query.strip().split()).replace('"', "")
    category_clean = " ".join(category.strip().split()).replace('"', "")
    if not query_clean and not category_clean:
        raise ValueError("Search query or category must be provided")

    parts: list[str] = []
    if query_clean:
        parts.append(f"{ARXIV_QUERY_FIELDS[field_key]}:{query_clean}")
    if category_clean:
        parts.append(f"cat:{category_clean}")

    return " AND ".join(parts)


def parse_arxiv_api_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into Paper objects.

    Args:
        xml_text: Raw XML string returned by the arXiv API.

    Returns:
        List of ``Paper`` objects, one per ``<entry>`` element.  Duplicate
        arXiv IDs within the feed are deduplicated (first occurrence wins).
        Title, authors, and comments are LaTeX-cleaned.  Papers carry
        ``source="api"`` to distinguish them from email-digest papers.

    Raises:
        ValueError: If ``xml_text`` is not valid XML (wraps
            ``ET.ParseError`` or ``DefusedXmlException``).
    """
    root = _parse_arxiv_api_root(xml_text)
    if root is None:
        return []

    papers: list[Paper] = []
    seen_ids: set[str] = set()

    for entry in root.findall("atom:entry", ATOM_NS):
        paper = _parse_arxiv_api_entry(entry)
        if paper is None or paper.arxiv_id in seen_ids:
            continue
        seen_ids.add(paper.arxiv_id)
        papers.append(paper)

    return papers


def _parse_arxiv_api_root(xml_text: str) -> Any | None:
    """Parse an arXiv API XML document, returning None for blank input."""
    if not xml_text.strip():
        return None
    try:
        return ET.fromstring(xml_text)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise ValueError("Invalid arXiv API XML response") from exc


def _parse_arxiv_api_entry(entry: Any) -> Paper | None:
    """Parse one Atom entry into a Paper, or None when no usable ID exists."""
    arxiv_id = normalize_arxiv_id(_atom_text(entry, "atom:id"))
    if not arxiv_id:
        return None

    raw_title = _atom_text(entry, "atom:title")
    raw_summary = _atom_text(entry, "atom:summary")
    raw_published = _atom_text(entry, "atom:published")
    raw_updated = _atom_text(entry, "atom:updated")
    comments_text = _atom_comment_text(entry)

    return Paper(
        arxiv_id=arxiv_id,
        date=_format_arxiv_api_date(raw_published or raw_updated),
        title=clean_latex(raw_title),
        authors=clean_latex(", ".join(_atom_author_names(entry))),
        categories=" ".join(_atom_categories(entry)),
        comments=clean_latex(comments_text) if comments_text else None,
        abstract=clean_latex(raw_summary) if raw_summary else "",
        abstract_raw=raw_summary,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        source="api",
    )


def _atom_author_names(entry: Any) -> list[str]:
    """Return normalized author names for an Atom entry."""
    return [
        " ".join(author.text.split())
        for author in entry.findall("atom:author/atom:name", ATOM_NS)
        if author.text
    ]


def _atom_categories(entry: Any) -> list[str]:
    """Return unique category terms for an Atom entry in feed order."""
    categories: list[str] = []
    for category in entry.findall("atom:category", ATOM_NS):
        term = (category.get("term") or "").strip()
        if term and term not in categories:
            categories.append(term)
    return categories


def _atom_comment_text(entry: Any) -> str:
    """Return normalized arXiv comment text for an Atom entry."""
    comments_node = entry.find("arxiv:comment", ATOM_NS)
    if comments_node is None or not comments_node.text:
        return ""
    return " ".join(comments_node.text.split())


def parse_arxiv_version_map(xml_text: str) -> dict[str, int]:
    """Parse arXiv Atom feed to extract {bare_arxiv_id: version_number} mapping.

    The <id> element contains URLs like ``http://arxiv.org/abs/2401.12345v3``.
    We extract the version suffix with a regex and normalize the bare ID.

    Returns:
        Mapping from bare arXiv ID to integer version number.
        Entries without a version suffix default to version 1.
    """
    if not xml_text.strip():
        return {}

    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, DefusedXmlException):
        logger.warning("Failed to parse arXiv version check response")
        return {}

    result: dict[str, int] = {}
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = _atom_text(entry, "atom:id")
        if not raw_id:
            continue
        # Extract version from the raw ID URL (e.g. http://arxiv.org/abs/2401.12345v3)
        match = _ARXIV_VERSION_SUFFIX.search(raw_id)
        version = int(match.group(0)[1:]) if match else 1
        bare_id = normalize_arxiv_id(raw_id)
        if bare_id:
            result[bare_id] = version

    return result


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from arXiv HTML papers (LaTeXML output)."""

    _SKIP_TAGS = frozenset({"script", "style", "nav", "header", "footer", "math"})
    _BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "section",
            "article",
            "blockquote",
            "figcaption",
        }
    )

    def __init__(self) -> None:
        """Initialize the extractor with empty output and zero skip depth."""
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        """Track entry into tags whose content should be skipped."""
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Track exit from skipped tags and insert newlines after block tags."""
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in self._BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        """Accumulate text content when not inside a skipped tag."""
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        """Return accumulated text with collapsed whitespace and preserved paragraphs."""
        raw = "".join(self._pieces)
        # Collapse whitespace within lines, preserve paragraph breaks
        lines = raw.split("\n")
        cleaned = [" ".join(line.split()) for line in lines]
        return "\n".join(line for line in cleaned if line).strip()


def extract_text_from_html(html: str) -> str:
    """Extract readable text from an arXiv HTML paper page."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def discover_history_files(
    base_dir: Path,
    limit: int | None = None,
) -> list[tuple[date, Path]]:
    """Discover and sort history files by date.

    Looks for files matching YYYY-MM-DD.txt pattern in the history/ subdirectory.

    Args:
        base_dir: Base directory containing the history/ folder.
        limit: Maximum number of history files to return. If None, returns all
               matching files.

    Returns:
        List of (date, path) tuples, sorted newest first.
        Empty list if history/ doesn't exist or has no matching files.
    """
    history_dir = base_dir / "history"
    if not history_dir.is_dir():
        return []

    files: list[tuple[date, Path]] = []
    try:
        paths = list(history_dir.glob("*.txt"))
    except OSError:
        logger.warning("Failed to enumerate history files in %s", history_dir, exc_info=True)
        return []

    for path in paths:
        try:
            d = datetime.strptime(path.stem, HISTORY_DATE_FORMAT).date()
            files.append((d, path))
        except ValueError:
            continue  # Silently skip files whose names don't match YYYY-MM-DD

    # Sort newest first and optionally limit.
    sorted_files = sorted(files, key=lambda x: x[0], reverse=True)
    if limit is None:
        return sorted_files
    return sorted_files[: max(limit, 0)]  # max(..., 0) guards against negative limit


def count_papers_in_file(path: Path) -> int:
    """Count papers in an arXiv email file without full parsing."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return len(_ARXIV_ID_PATTERN.findall(content))
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        return 0


def build_daily_digest(
    papers: list[Paper],
    watched_ids: set[str] | None = None,
    metadata: dict[str, PaperMetadata] | None = None,
) -> str:
    """Build a concise daily digest string summarizing the day's papers.

    Args:
        papers: The full list of papers loaded for the current day.
        watched_ids: Optional set of arXiv IDs that matched the user's watch
            list.  When provided, a ``"N matches your watch list"`` segment is
            included in the output.
        metadata: Optional mapping from arXiv ID to ``PaperMetadata``.  When
            provided, read/starred counts are appended.

    Returns:
        A single-line summary string with segments separated by ``" · "``
        (or ``" | "`` in ASCII mode), e.g.
        ``"42 papers · Top: cs.AI (18) · 3 matches your watch list"``.
        Returns ``"No papers loaded"`` when ``papers`` is empty.
    """
    if not papers:
        return "No papers loaded"

    from arxiv_browser._ascii import is_ascii_mode

    sep = " | " if is_ascii_mode() else " \u00b7 "

    lines = [f"{len(papers)} papers"]
    lines.extend(_daily_digest_category_segments(papers))
    lines.extend(_daily_digest_watch_segments(watched_ids))
    lines.extend(_daily_digest_metadata_segments(papers, metadata))
    return sep.join(lines)


def _daily_digest_category_segments(papers: list[Paper]) -> list[str]:
    """Return category summary segments for a daily digest."""
    cat_counts: dict[str, int] = {}
    for paper in papers:
        primary = paper.categories.split()[0] if paper.categories else "unknown"
        cat_counts[primary] = cat_counts.get(primary, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    cat_parts = [f"{cat} ({n})" for cat, n in top_cats]
    return ["Top: " + ", ".join(cat_parts)] if cat_parts else []


def _daily_digest_watch_segments(watched_ids: set[str] | None) -> list[str]:
    """Return watch-list summary segments for a daily digest."""
    if not watched_ids:
        return []
    n = len(watched_ids)
    return [f"{n} match{'es' if n != 1 else ''} your watch list"]


def _daily_digest_metadata_segments(
    papers: list[Paper], metadata: dict[str, PaperMetadata] | None
) -> list[str]:
    """Return read/starred summary segments for a daily digest."""
    if not metadata:
        return []
    read = sum(1 for p in papers if (m := metadata.get(p.arxiv_id)) and m.is_read)
    starred = sum(1 for p in papers if (m := metadata.get(p.arxiv_id)) and m.starred)
    parts = _daily_digest_count_parts(read, starred)
    return [", ".join(parts)] if parts else []


def _daily_digest_count_parts(read: int, starred: int) -> list[str]:
    """Return non-empty read/starred count labels."""
    parts = []
    if read:
        parts.append(f"{read} read")
    if starred:
        parts.append(f"{starred} starred")
    return parts


__all__ = [
    "ARXIV_DATE_FORMAT",
    "ARXIV_QUERY_FIELDS",
    "ATOM_NS",
    "HISTORY_DATE_FORMAT",
    "_ARXIV_ID_PATTERN",
    "_ARXIV_VERSION_SUFFIX",
    "_ESCAPED_DOLLAR",
    "_LATEX_PATTERNS",
    "_HTMLTextExtractor",
    "build_arxiv_search_query",
    "build_daily_digest",
    "clean_latex",
    "count_papers_in_file",
    "discover_history_files",
    "extract_text_from_html",
    "normalize_arxiv_id",
    "parse_arxiv_api_feed",
    "parse_arxiv_date",
    "parse_arxiv_file",
    "parse_arxiv_version_map",
]
