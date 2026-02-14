"""Query parsing, matching, sorting, and text formatting utilities."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup

from arxiv_browser.models import (
    Paper,
    PaperMetadata,
    QueryToken,
    WatchListEntry,
)
from arxiv_browser.themes import CATEGORY_COLORS, DEFAULT_CATEGORY_COLOR, THEME_COLORS

if TYPE_CHECKING:
    from arxiv_browser.huggingface import HuggingFacePaper
    from arxiv_browser.parsing import parse_arxiv_date
    from arxiv_browser.semantic_scholar import SemanticScholarPaper
else:
    # Deferred imports to avoid circular deps at runtime
    def parse_arxiv_date(date_str: str) -> datetime:  # type: ignore[assignment]  # noqa: F821
        from arxiv_browser.parsing import parse_arxiv_date as _parse

        return _parse(date_str)


# ============================================================================
# Text Formatting Utilities
# ============================================================================


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated.

    Args:
        text: The text to truncate.
        max_len: Maximum length before truncation (not including suffix).
        suffix: String to append when truncated.

    Returns:
        Original text if within limit, otherwise truncated with suffix.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix


def escape_rich_text(text: str) -> str:
    """Escape text for safe Rich markup rendering."""
    return escape_markup(text) if text else ""


def render_progress_bar(current: int, total: int, width: int = 10) -> str:
    """Render a Unicode progress bar like ████░░░░░░."""
    if total <= 0:
        return "░" * width
    filled = max(0, min(width, round(current / total * width)))
    return "█" * filled + "░" * (width - filled)


# Pre-compiled patterns for lightweight markdown → Rich markup conversion
_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BULLET_RE = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)


def format_summary_as_rich(text: str) -> str:
    """Convert a markdown-formatted LLM summary to Rich markup.

    Handles headings, bold, inline code, and bullet lists — the typical
    elements produced by the structured prompt.
    """
    if not text:
        return ""
    # Escape first so user content is safe, then layer Rich markup on top
    out = escape_rich_text(text)
    # Headings: ## Foo → colored bold
    heading_color = THEME_COLORS.get("accent", "#66d9ef")

    def _heading_repl(m: re.Match[str]) -> str:
        level = len(m.group(1))
        label = m.group(2).strip()
        if level <= 2:
            return f"[bold {heading_color}]{label}[/]"
        return f"[bold]{label}[/]"

    out = _MD_HEADING_RE.sub(_heading_repl, out)
    # Bold: **text** → [bold]text[/]
    out = _MD_BOLD_RE.sub(r"[bold]\1[/]", out)
    # Inline code: `code` → styled span
    code_color = THEME_COLORS.get("green", "#a6e22e")
    out = _MD_INLINE_CODE_RE.sub(rf"[{code_color}]\1[/]", out)
    # Bullets: - item → • item
    out = _MD_BULLET_RE.sub(r"\1  • ", out)
    # Indent all lines for consistent padding inside the details pane
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in out.split("\n"))
    return indented


_HIGHLIGHT_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def highlight_text(text: str, terms: list[str], color: str) -> str:
    """Highlight terms inside text using Rich markup."""
    if not text:
        return text
    escaped_text = escape_rich_text(text)
    if not terms:
        return escaped_text
    normalized = []
    seen: set[str] = set()
    for term in terms:
        cleaned = term.strip()
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    if not normalized:
        return escaped_text
    normalized.sort(key=len, reverse=True)
    cache_key = tuple(normalized)
    pattern = _HIGHLIGHT_PATTERN_CACHE.get(cache_key)
    if pattern is None:
        escaped_terms = [escape_rich_text(term) for term in normalized]
        pattern = re.compile("|".join(re.escape(term) for term in escaped_terms), re.IGNORECASE)
        _HIGHLIGHT_PATTERN_CACHE[cache_key] = pattern
    return pattern.sub(lambda match: f"[bold {color}]{match.group(0)}[/]", escaped_text)


# ============================================================================
# Query Parser Functions (extracted for testability)
# ============================================================================


_FIELD_NAMES = frozenset({"title", "author", "abstract", "cat", "tag"})


def _parse_quoted_phrase(query: str, i: int, query_len: int) -> tuple[QueryToken, int]:
    """Parse a quoted phrase starting after the opening quote."""
    start = i
    while i < query_len and query[i] != '"':
        i += 1
    value = query[start:i]
    return QueryToken(kind="term", value=value, phrase=True), i + 1


def _parse_field_value(query: str, i: int, query_len: int, field: str) -> tuple[QueryToken, int]:
    """Parse the value after a field:colon, handling both quoted and unquoted."""
    if i < query_len and query[i] == '"':
        i += 1
        value_start = i
        while i < query_len and query[i] != '"':
            i += 1
        value = query[value_start:i]
        return QueryToken(kind="term", value=value, field=field, phrase=True), i + 1
    value_start = i
    while i < query_len and not query[i].isspace():
        i += 1
    value = query[value_start:i]
    return QueryToken(kind="term", value=value, field=field), i


def _parse_plain_term(query: str, start: int, i: int, query_len: int) -> tuple[QueryToken, int]:
    """Parse a plain term or boolean operator, advancing past it."""
    while i < query_len and not query[i].isspace():
        i += 1
    raw = query[start:i]
    upper = raw.upper()
    if upper in {"AND", "OR", "NOT"}:
        return QueryToken(kind="op", value=upper), i
    return QueryToken(kind="term", value=raw), i


def tokenize_query(query: str) -> list[QueryToken]:
    """Tokenize a query string into terms and operators."""
    tokens: list[QueryToken] = []
    i = 0
    query_len = len(query)
    while i < query_len:
        if query[i].isspace():
            i += 1
            continue
        if query[i] == '"':
            token, i = _parse_quoted_phrase(query, i + 1, query_len)
            tokens.append(token)
            continue
        start = i
        while i < query_len and not query[i].isspace() and query[i] != ":":
            i += 1
        if i < query_len and query[i] == ":":
            field = query[start:i].lower()
            if field in _FIELD_NAMES:
                token, i = _parse_field_value(query, i + 1, query_len, field)
                tokens.append(token)
                continue
        token, i = _parse_plain_term(query, start, i, query_len)
        tokens.append(token)
    return tokens


def pill_label_for_token(token: QueryToken) -> str:
    """Return a human-readable label for a query token pill.

    Examples: cat:cs.AI, "exact phrase", author:"John Smith", transformer
    """
    value = token.value
    if token.field and token.phrase:
        return f'{token.field}:"{value}"'
    if token.field:
        return f"{token.field}:{value}"
    if token.phrase:
        return f'"{value}"'
    return value


def reconstruct_query(tokens: list[QueryToken], exclude_index: int) -> str:
    """Rebuild query string omitting the token at exclude_index.

    Cleans up orphaned boolean operators adjacent to the removed term.
    Boolean operators (AND/OR/NOT) are structural and not directly removable.
    """
    if exclude_index < 0 or exclude_index >= len(tokens):
        return " ".join(_token_to_str(t) for t in tokens)

    remaining = [t for i, t in enumerate(tokens) if i != exclude_index]

    # Clean up orphaned operators: strip leading/trailing ops and adjacent ops
    cleaned: list[QueryToken] = []
    for tok in remaining:
        # Skip operators that would be leading or adjacent to another op
        if tok.kind == "op" and (not cleaned or cleaned[-1].kind == "op"):
            continue
        cleaned.append(tok)

    # Remove trailing operator
    if cleaned and cleaned[-1].kind == "op":
        cleaned.pop()

    return " ".join(_token_to_str(t) for t in cleaned)


def get_query_tokens(query: str) -> list[QueryToken]:
    """Tokenize a query after trimming surrounding whitespace."""
    normalized_query = query.strip()
    return tokenize_query(normalized_query) if normalized_query else []


def remove_query_token(query: str, token_index: int) -> str:
    """Remove one token from a query string and rebuild it."""
    tokens = get_query_tokens(query)
    return reconstruct_query(tokens, token_index)


def _token_to_str(token: QueryToken) -> str:
    """Convert a QueryToken back to its query string representation."""
    if token.kind == "op":
        return token.value
    if token.field and token.phrase:
        return f'{token.field}:"{token.value}"'
    if token.field:
        return f"{token.field}:{token.value}"
    if token.phrase:
        return f'"{token.value}"'
    return token.value


def insert_implicit_and(tokens: list[QueryToken]) -> list[QueryToken]:
    """Insert implicit AND operators between adjacent terms."""
    result: list[QueryToken] = []
    prev_was_term = False
    for token in tokens:
        token_is_term_start = token.kind == "term" or token.value == "NOT"
        if prev_was_term and token_is_term_start:
            result.append(QueryToken(kind="op", value="AND"))
        result.append(token)
        prev_was_term = token.kind == "term"
    return result


def to_rpn(tokens: list[QueryToken]) -> list[QueryToken]:
    """Convert tokens to reverse polish notation using operator precedence."""
    output: list[QueryToken] = []
    ops: list[QueryToken] = []
    precedence = {"OR": 1, "AND": 2, "NOT": 3}
    for token in tokens:
        if token.kind == "term":
            output.append(token)
            continue
        while ops and precedence[ops[-1].value] >= precedence[token.value]:
            output.append(ops.pop())
        ops.append(token)
    while ops:
        output.append(ops.pop())
    return output


@functools.lru_cache(maxsize=256)
def format_categories(categories: str) -> str:
    """Format categories with colors. Results are automatically cached via lru_cache."""
    parts = []
    for cat in categories.split():
        color = CATEGORY_COLORS.get(cat, DEFAULT_CATEGORY_COLOR)
        parts.append(f"[{color}]{cat}[/]")
    return " ".join(parts)


# ============================================================================
# Query Matching & Paper Sorting
# ============================================================================


def is_advanced_query(tokens: list[QueryToken]) -> bool:
    """Check if a query uses advanced features (operators, fields, phrases, virtual terms)."""
    return any(
        tok.kind == "op" or tok.field or tok.phrase or tok.value.lower() in {"unread", "starred"}
        for tok in tokens
    )


def build_highlight_terms(tokens: list[QueryToken]) -> dict[str, list[str]]:
    """Build highlight term lists from query tokens by field."""
    highlight: dict[str, list[str]] = {"title": [], "author": [], "abstract": []}
    for token in tokens:
        if token.kind != "term":
            continue
        if token.value.lower() in {"unread", "starred"}:
            continue
        if token.field == "title":
            highlight["title"].append(token.value)
        elif token.field == "author":
            highlight["author"].append(token.value)
        elif token.field == "abstract":
            highlight["abstract"].append(token.value)
        elif token.field is None:
            highlight["title"].append(token.value)
            highlight["author"].append(token.value)
    return highlight


def execute_query_filter(
    query: str,
    papers: list[Paper],
    *,
    fuzzy_search: Callable[[str], list[Paper]],
    advanced_match: Callable[[Paper, list[QueryToken]], bool],
) -> tuple[list[Paper], dict[str, list[str]]]:
    """Filter papers for a query and return (filtered_papers, highlight_terms)."""
    normalized_query = query.strip()
    if not normalized_query:
        return papers.copy(), {"title": [], "author": [], "abstract": []}

    tokens = tokenize_query(normalized_query)
    highlight_terms = build_highlight_terms(tokens)

    if is_advanced_query(tokens):
        advanced_tokens = insert_implicit_and(tokens)
        rpn = to_rpn(advanced_tokens)
        filtered = [paper for paper in papers if advanced_match(paper, rpn)]
        return filtered, highlight_terms

    return fuzzy_search(normalized_query), highlight_terms


def apply_watch_filter(
    papers: list[Paper], watched_paper_ids: set[str], watch_filter_active: bool
) -> list[Paper]:
    """Apply watch-list intersection filtering when enabled."""
    if not watch_filter_active:
        return papers
    return [paper for paper in papers if paper.arxiv_id in watched_paper_ids]


def match_query_term(
    paper: Paper,
    token: QueryToken,
    paper_metadata: PaperMetadata | None,
    abstract_text: str = "",
) -> bool:
    """Check if a paper matches a single query term.

    Args:
        paper: The paper to match against.
        token: The query token to match.
        paper_metadata: The paper's user metadata (for tag/read/star lookups).
        abstract_text: Pre-cleaned abstract text.
    """
    value = token.value.strip()
    if not value:
        return True
    value_lower = value.lower()
    if token.field == "cat":
        return value_lower in paper.categories.lower()
    if token.field == "tag":
        if not paper_metadata:
            return False
        return any(value_lower in tag.lower() for tag in paper_metadata.tags)
    if token.field == "title":
        return value_lower in paper.title.lower()
    if token.field == "author":
        return value_lower in paper.authors.lower()
    if token.field == "abstract":
        return value_lower in abstract_text.lower()
    if value_lower == "unread":
        return not paper_metadata or not paper_metadata.is_read
    if value_lower == "starred":
        return bool(paper_metadata and paper_metadata.starred)
    haystack = f"{paper.title} {paper.authors}".lower()
    return value_lower in haystack


def matches_advanced_query(
    paper: Paper,
    rpn: list[QueryToken],
    paper_metadata: PaperMetadata | None,
    abstract_text: str = "",
) -> bool:
    """Evaluate an RPN query expression against a paper.

    Args:
        paper: The paper to match against.
        rpn: Query in Reverse Polish Notation.
        paper_metadata: The paper's user metadata.
        abstract_text: Pre-cleaned abstract text.
    """
    if not rpn:
        return True
    stack: list[bool] = []
    for token in rpn:
        if token.kind == "term":
            stack.append(match_query_term(paper, token, paper_metadata, abstract_text))
            continue
        if token.value == "NOT":
            value = stack.pop() if stack else False
            stack.append(not value)
        else:
            right = stack.pop() if stack else False
            left = stack.pop() if stack else False
            if token.value == "AND":
                stack.append(left and right)
            else:
                stack.append(left or right)
    return stack[-1] if stack else True


def paper_matches_watch_entry(paper: Paper, entry: WatchListEntry) -> bool:
    """Check if a paper matches a watch list entry."""
    pattern = entry.pattern if entry.case_sensitive else entry.pattern.lower()

    if entry.match_type == "author":
        text = paper.authors if entry.case_sensitive else paper.authors.lower()
        return pattern in text
    elif entry.match_type == "title":
        text = paper.title if entry.case_sensitive else paper.title.lower()
        return pattern in text
    elif entry.match_type == "keyword":
        if entry.case_sensitive:
            return pattern in paper.title or pattern in paper.abstract_raw
        else:
            return pattern in paper.title.lower() or pattern in paper.abstract_raw.lower()
    return False


def sort_papers(
    papers: list[Paper],
    sort_key: str,
    s2_cache: dict[str, SemanticScholarPaper] | None = None,
    hf_cache: dict[str, HuggingFacePaper] | None = None,
    relevance_cache: dict[str, tuple[int, str]] | None = None,
) -> list[Paper]:
    """Sort papers by the given key, returning a new sorted list.

    Args:
        papers: List of papers to sort.
        sort_key: One of "title", "date", "arxiv_id", "citations", "trending", "relevance".
        s2_cache: Optional S2 cache dict, required for "citations" sort.
        hf_cache: Optional HF cache dict, required for "trending" sort.
        relevance_cache: Optional relevance cache dict, required for "relevance" sort.
    """
    if sort_key == "title":
        return sorted(papers, key=lambda p: p.title.lower())
    elif sort_key == "date":
        return sorted(papers, key=lambda p: parse_arxiv_date(p.date), reverse=True)
    elif sort_key == "arxiv_id":
        return sorted(papers, key=lambda p: p.arxiv_id, reverse=True)
    elif sort_key == "citations":

        def _citation_key(p: Paper) -> tuple[int, int]:
            s2 = s2_cache.get(p.arxiv_id) if s2_cache else None
            if s2 is not None:
                return (0, -s2.citation_count)
            return (1, 0)

        return sorted(papers, key=_citation_key)
    elif sort_key == "trending":

        def _trending_key(p: Paper) -> tuple[int, int]:
            hf = hf_cache.get(p.arxiv_id) if hf_cache else None
            if hf is not None:
                return (0, -hf.upvotes)
            return (1, 0)

        return sorted(papers, key=_trending_key)
    elif sort_key == "relevance":

        def _relevance_key(p: Paper) -> tuple[int, int]:
            rel = relevance_cache.get(p.arxiv_id) if relevance_cache else None
            if rel is not None:
                return (0, -rel[0])
            return (1, 0)

        return sorted(papers, key=_relevance_key)
    return list(papers)


__all__ = [
    "_HIGHLIGHT_PATTERN_CACHE",
    "apply_watch_filter",
    "build_highlight_terms",
    "escape_rich_text",
    "execute_query_filter",
    "format_categories",
    "format_summary_as_rich",
    "get_query_tokens",
    "highlight_text",
    "insert_implicit_and",
    "is_advanced_query",
    "match_query_term",
    "matches_advanced_query",
    "paper_matches_watch_entry",
    "pill_label_for_token",
    "reconstruct_query",
    "remove_query_token",
    "render_progress_bar",
    "sort_papers",
    "to_rpn",
    "tokenize_query",
    "truncate_text",
]
