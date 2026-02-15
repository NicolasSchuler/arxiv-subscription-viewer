"""arXiv email parsing, HTML text extraction, LaTeX cleaning, and history discovery."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

from arxiv_browser.models import Paper, PaperMetadata

logger = logging.getLogger(__name__)

# Date format used in arXiv emails (e.g., "Mon, 15 Jan 2024")
ARXIV_DATE_FORMAT = "%a, %d %b %Y"
# Extract the date prefix when time/zone info is present
_ARXIV_DATE_PREFIX_PATTERN = re.compile(r"([A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})")

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
# Matches: "arXiv:2301.12345" or "arXiv:2301.12345v2" -> captures ID with optional version
_ARXIV_ID_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)")
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


def parse_arxiv_date(date_str: str) -> datetime:
    """Parse arXiv date string to datetime for proper sorting.

    Args:
        date_str: Date string like "Mon, 15 Jan 2024"

    Returns:
        Parsed datetime object, or datetime.min for malformed dates.
    """
    cleaned = date_str.strip()
    if not cleaned:
        return datetime.min

    try:
        return datetime.strptime(cleaned, ARXIV_DATE_FORMAT)
    except ValueError:
        pass

    match = _ARXIV_DATE_PREFIX_PATTERN.search(cleaned)
    if match:
        try:
            return datetime.strptime(match.group(1), ARXIV_DATE_FORMAT)
        except ValueError:
            pass

    return datetime.min  # Fallback for malformed dates


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

    # Iteratively apply patterns to handle nested structures
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
    """
    import time

    t0 = time.monotonic()
    # Use errors="replace" to handle any non-UTF-8 characters gracefully
    content = filepath.read_text(encoding="utf-8", errors="replace")
    papers_by_id: dict[str, tuple[int, Paper]] = {}

    # Split by paper separator using pre-compiled pattern
    entries = _ENTRY_SEPARATOR.split(content)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Extract arXiv ID
        arxiv_match = _ARXIV_ID_PATTERN.search(entry)
        if not arxiv_match:
            continue
        arxiv_id_raw = arxiv_match.group(1)
        bare_arxiv_id = normalize_arxiv_id(arxiv_id_raw)
        if not bare_arxiv_id:
            continue
        version_match = _ARXIV_VERSION_SUFFIX.search(arxiv_id_raw)
        version = int(version_match.group(0)[1:]) if version_match else 1

        # Extract date
        date_match = _DATE_PATTERN.search(entry)
        date_val = date_match.group(1).strip() if date_match else ""

        # Extract title (may span multiple lines)
        title_match = _TITLE_PATTERN.search(entry)
        title = " ".join(title_match.group(1).split()) if title_match else ""

        # Extract authors (may span multiple lines)
        authors_match = _AUTHORS_PATTERN.search(entry)
        authors = " ".join(authors_match.group(1).split()) if authors_match else ""

        # Extract categories
        categories_match = _CATEGORIES_PATTERN.search(entry)
        categories = categories_match.group(1).strip() if categories_match else ""

        # Extract comments (optional)
        comments_match = _COMMENTS_PATTERN.search(entry)
        comments = comments_match.group(1).strip() if comments_match else None

        # Extract abstract (text between \\ markers)
        abstract_match = _ABSTRACT_PATTERN.search(entry)
        abstract_raw = " ".join(abstract_match.group(1).split()) if abstract_match else ""

        # Extract URL
        url_match = _URL_PATTERN.search(entry)
        url = url_match.group(1) if url_match else f"https://arxiv.org/abs/{bare_arxiv_id}"

        paper = Paper(
            arxiv_id=bare_arxiv_id,
            date=date_val,
            title=clean_latex(title),
            authors=clean_latex(authors),
            categories=categories,
            comments=clean_latex(comments) if comments else None,
            abstract=None,
            abstract_raw=abstract_raw,
            url=url,
        )
        existing = papers_by_id.get(bare_arxiv_id)
        if existing is None or version > existing[0]:
            papers_by_id[bare_arxiv_id] = (version, paper)

    elapsed = time.monotonic() - t0
    papers = [paper for _, paper in papers_by_id.values()]
    logger.debug("Parsed %d papers from %s in %.3fs", len(papers), filepath.name, elapsed)
    return papers


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
    """Convert Atom timestamp to the app's date format."""
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


def _atom_text(node: ET.Element, path: str) -> str:
    """Extract normalized text from an Atom XML node path."""
    found = node.find(path, ATOM_NS)
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())


def build_arxiv_search_query(query: str, field: str = "all", category: str = "") -> str:
    """Build an arXiv API search query string.

    Args:
        query: User search text (can be empty if category is provided).
        field: One of: all, title, author, abstract.
        category: Optional category filter like "cs.AI".
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
    """Parse an arXiv Atom feed into Paper objects."""
    if not xml_text.strip():
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError("Invalid arXiv API XML response") from exc

    papers: list[Paper] = []
    seen_ids: set[str] = set()

    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = _atom_text(entry, "atom:id")
        arxiv_id = normalize_arxiv_id(raw_id)
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        raw_title = _atom_text(entry, "atom:title")
        raw_summary = _atom_text(entry, "atom:summary")
        raw_published = _atom_text(entry, "atom:published")
        raw_updated = _atom_text(entry, "atom:updated")

        author_names = [
            " ".join(author.text.split())
            for author in entry.findall("atom:author/atom:name", ATOM_NS)
            if author.text
        ]

        categories: list[str] = []
        for category in entry.findall("atom:category", ATOM_NS):
            term = (category.get("term") or "").strip()
            if term and term not in categories:
                categories.append(term)

        comments_node = entry.find("arxiv:comment", ATOM_NS)
        comments_text = ""
        if comments_node is not None and comments_node.text:
            comments_text = " ".join(comments_node.text.split())

        cleaned_summary = clean_latex(raw_summary) if raw_summary else ""
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                date=_format_arxiv_api_date(raw_published or raw_updated),
                title=clean_latex(raw_title),
                authors=clean_latex(", ".join(author_names)),
                categories=" ".join(categories),
                comments=clean_latex(comments_text) if comments_text else None,
                abstract=cleaned_summary,
                abstract_raw=raw_summary,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source="api",
            )
        )

    return papers


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
    except ET.ParseError:
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
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in self._BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
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
            continue  # Skip files that don't match YYYY-MM-DD pattern

    # Sort newest first and optionally limit.
    sorted_files = sorted(files, key=lambda x: x[0], reverse=True)
    if limit is None:
        return sorted_files
    return sorted_files[: max(limit, 0)]


def count_papers_in_file(path: Path) -> int:
    """Count papers in an arXiv email file without full parsing."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return len(_ARXIV_ID_PATTERN.findall(content))
    except OSError:
        return 0


def build_daily_digest(
    papers: list[Paper],
    watched_ids: set[str] | None = None,
    metadata: dict[str, PaperMetadata] | None = None,
) -> str:
    """Build a concise daily digest string summarizing the day's papers.

    Returns a single-line summary with category breakdown, watch matches, and read stats
    separated by " · ".
    """
    if not papers:
        return "No papers loaded"

    # Category breakdown (top 5)
    cat_counts: dict[str, int] = {}
    for paper in papers:
        primary = paper.categories.split()[0] if paper.categories else "unknown"
        cat_counts[primary] = cat_counts.get(primary, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    cat_parts = [f"{cat} ({n})" for cat, n in top_cats]

    lines = [f"{len(papers)} papers"]

    if cat_parts:
        lines.append("Top: " + ", ".join(cat_parts))

    # Watch list matches
    if watched_ids:
        n = len(watched_ids)
        lines.append(f"{n} match{'es' if n != 1 else ''} your watch list")

    # Read/starred stats
    if metadata:
        read = sum(1 for p in papers if (m := metadata.get(p.arxiv_id)) and m.is_read)
        starred = sum(1 for p in papers if (m := metadata.get(p.arxiv_id)) and m.starred)
        if read or starred:
            parts = []
            if read:
                parts.append(f"{read} read")
            if starred:
                parts.append(f"{starred} starred")
            lines.append(", ".join(parts))

    return " · ".join(lines)


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
