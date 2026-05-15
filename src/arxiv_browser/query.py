"""Query parsing, matching, sorting, and text formatting utilities."""

from __future__ import annotations

import functools
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup

from arxiv_browser.authors import author_matches_exact
from arxiv_browser.models import (
    Paper,
    PaperMetadata,
    QueryToken,
    WatchListEntry,
    parse_arxiv_date,
)
from arxiv_browser.review import is_review_due
from arxiv_browser.themes import DEFAULT_CATEGORY_COLOR, DEFAULT_CATEGORY_COLORS, DEFAULT_THEME
from arxiv_browser.triage_model import (
    TRIAGE_BUCKET_LIKELY_SKIP,
    TRIAGE_BUCKET_LIKELY_STAR,
    TRIAGE_BUCKET_UNSURE,
)

if TYPE_CHECKING:
    from arxiv_browser.huggingface import HuggingFacePaper
    from arxiv_browser.semantic_scholar import SemanticScholarPaper
    from arxiv_browser.triage_model import TriagePrediction


_QUEUE_RELEVANCE_WEIGHT = 0.40
_QUEUE_WATCH_WEIGHT = 0.25
_QUEUE_RECENCY_WEIGHT = 0.15
_QUEUE_HF_WEIGHT = 0.10
_QUEUE_VELOCITY_WEIGHT = 0.10
_QUEUE_RECENCY_DECAY_DAYS = 30.0
_DAYS_PER_YEAR = 365.25
_VIRTUAL_QUERY_TERMS = frozenset({"unread", "starred", "review-due"})


@dataclass(frozen=True, slots=True)
class _QueueScoreContext:
    s2_cache: dict[str, SemanticScholarPaper] | None
    hf_cache: dict[str, HuggingFacePaper] | None
    relevance_cache: dict[str, tuple[int, str]] | None
    watched_paper_ids: set[str]
    today: datetime
    max_hf_upvote_log: float
    max_velocity_log: float


@dataclass(frozen=True, slots=True)
class PaperSortSignals:
    """Optional cached signals used by advanced paper sort modes."""

    s2_cache: dict[str, SemanticScholarPaper] | None = None
    hf_cache: dict[str, HuggingFacePaper] | None = None
    relevance_cache: dict[str, tuple[int, str]] | None = None
    watched_paper_ids: set[str] | None = None
    triage_predictions: dict[str, TriagePrediction] | None = None


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


def truncate_at_word_boundary(text: str, max_length: int, *, ascii_mode: bool = False) -> str:
    """Truncate *text* at a word boundary, appending an ellipsis when shortened.

    The returned string (including ellipsis) never exceeds *max_length* characters.

    Args:
        text: The text to truncate.
        max_length: Maximum total length of the returned string.
        ascii_mode: Use ``"..."`` instead of ``"…"`` when *True*.

    Returns:
        Original text unchanged when it fits, otherwise word-boundary-truncated
        text with an appended ellipsis.
    """
    if len(text) <= max_length:
        return text
    ellipsis = "..." if ascii_mode else "\u2026"
    cutoff = max_length - len(ellipsis)
    if cutoff <= 0:
        return ellipsis[:max_length]
    last_space = text.rfind(" ", 0, cutoff)
    # Only use word boundary if it preserves at least 60% of the budget
    if last_space > cutoff * 0.6:
        return text[:last_space] + ellipsis
    return text[:cutoff] + ellipsis


def escape_rich_text(text: str) -> str:
    """Escape text for safe Rich markup rendering."""
    return escape_markup(text) if text else ""


def render_progress_bar(
    current: int, total: int, width: int = 10, *, ascii_mode: bool | None = None
) -> str:
    """Render a progress bar like ████░░░░░░ (or ###------- in ASCII mode)."""
    if ascii_mode is None:
        from arxiv_browser._ascii import is_ascii_mode

        ascii_mode = is_ascii_mode()
    filled_ch = "#" if ascii_mode else "\u2588"
    empty_ch = "-" if ascii_mode else "\u2591"
    if total <= 0:
        return empty_ch * width
    filled = max(0, min(width, round(current / total * width)))
    return filled_ch * filled + empty_ch * (width - filled)


# Pre-compiled patterns for lightweight markdown → Rich markup conversion
_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BULLET_RE = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)


def format_summary_as_rich(
    text: str,
    theme_colors: Mapping[str, str] | None = None,
) -> str:
    """Convert a markdown-formatted LLM summary to Rich markup.

    Handles headings, bold, inline code, and bullet lists — the typical
    elements produced by the structured prompt.
    """
    if not text:
        return ""
    colors = theme_colors or DEFAULT_THEME
    # Escape first so user content is safe, then layer Rich markup on top
    out = escape_rich_text(text)
    # Headings: ## Foo → colored bold
    heading_color = colors.get("accent", "#66d9ef")

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
    code_color = colors.get("green", "#a6e22e")
    out = _MD_INLINE_CODE_RE.sub(rf"[{code_color}]\1[/]", out)
    # Bullets: - item → • item (or - item in ASCII mode)
    from arxiv_browser._ascii import is_ascii_mode

    bullet = "- " if is_ascii_mode() else "\u2022 "
    out = _MD_BULLET_RE.sub(rf"\1  {bullet}", out)
    # Indent all lines for consistent padding inside the details pane
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in out.split("\n"))
    return indented


_HIGHLIGHT_PATTERN_CACHE_MAX = 256
_HIGHLIGHT_PATTERN_CACHE: OrderedDict[tuple[str, ...], re.Pattern[str]] = OrderedDict()


def _normalize_highlight_terms(terms: list[str]) -> list[str]:
    """Return unique highlight terms, case-insensitive and longest first."""
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
    normalized.sort(key=len, reverse=True)
    return normalized


def _highlight_pattern_for_terms(normalized_terms: list[str]) -> re.Pattern[str]:
    """Return an LRU-cached regex pattern for normalized highlight terms."""
    cache_key = tuple(normalized_terms)
    pattern = _HIGHLIGHT_PATTERN_CACHE.get(cache_key)
    if pattern is not None:
        _HIGHLIGHT_PATTERN_CACHE.move_to_end(cache_key)
        return pattern

    escaped_terms = [escape_rich_text(term) for term in normalized_terms]
    pattern = re.compile("|".join(re.escape(term) for term in escaped_terms), re.IGNORECASE)
    if len(_HIGHLIGHT_PATTERN_CACHE) >= _HIGHLIGHT_PATTERN_CACHE_MAX:
        _HIGHLIGHT_PATTERN_CACHE.popitem(last=False)
    _HIGHLIGHT_PATTERN_CACHE[cache_key] = pattern
    return pattern


def highlight_text(text: str, terms: list[str], color: str) -> str:
    """Highlight terms inside text using Rich markup.

    Terms shorter than 2 characters are ignored. Duplicate terms (case-
    insensitive) are deduplicated, and remaining terms are matched longest-first
    to avoid partial overlaps. Compiled patterns are cached (LRU, max 256).
    """
    if not text:
        return text
    escaped_text = escape_rich_text(text)
    if not terms:
        return escaped_text
    normalized = _normalize_highlight_terms(terms)
    if not normalized:
        return escaped_text
    pattern = _highlight_pattern_for_terms(normalized)
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
    exact_prefix = field == "author" and i < query_len and query[i] == "="
    if exact_prefix:
        i += 1
    if i < query_len and query[i] == '"':
        i += 1
        value_start = i
        while i < query_len and query[i] != '"':
            i += 1
        value = query[value_start:i]
        if exact_prefix:
            value = f"={value}"
        return QueryToken(kind="term", value=value, field=field, phrase=True), i + 1
    value_start = i
    while i < query_len and not query[i].isspace():
        i += 1
    value = query[value_start:i]
    if exact_prefix:
        value = f"={value}"
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
        # Phase 1: advance until whitespace or ':' to check for a field prefix
        while i < query_len and not query[i].isspace() and query[i] != ":":
            i += 1
        if i < query_len and query[i] == ":":
            field = query[start:i].lower()
            if field in _FIELD_NAMES:
                token, i = _parse_field_value(query, i + 1, query_len, field)
                tokens.append(token)
                continue
        # Phase 2: no valid field prefix found; re-use 'start' to parse from
        # the beginning of this chunk as a plain term or boolean operator
        token, i = _parse_plain_term(query, start, i, query_len)
        tokens.append(token)
    return tokens


def pill_label_for_token(token: QueryToken) -> str:
    """Return a human-readable label for a query token pill.

    Examples: cat:cs.AI, "exact phrase", author:"John Smith", transformer
    """
    value = token.value
    if token.field == "author" and value.startswith("="):
        exact_value = value[1:]
        if token.phrase:
            return f'{token.field}:="{exact_value}"'
        return f"{token.field}:={exact_value}"
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
    cleaned = _remove_orphaned_query_ops(remaining)
    return " ".join(_token_to_str(t) for t in cleaned)


def _remove_orphaned_query_ops(tokens: list[QueryToken]) -> list[QueryToken]:
    """Drop leading, adjacent, and trailing Boolean operators after token removal."""

    cleaned: list[QueryToken] = []
    for tok in tokens:
        if tok.kind == "op" and (not cleaned or cleaned[-1].kind == "op"):
            continue
        cleaned.append(tok)

    if cleaned and cleaned[-1].kind == "op":
        cleaned.pop()
    return cleaned


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
    if token.field == "author" and token.value.startswith("="):
        exact_value = token.value[1:]
        if token.phrase:
            return f'{token.field}:="{exact_value}"'
        return f"{token.field}:={exact_value}"
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
        # NOT begins a new term group but is itself an operator, so it triggers
        # an implicit AND insertion without setting prev_was_term afterward.
        token_is_term_start = token.kind == "term" or token.value == "NOT"
        if prev_was_term and token_is_term_start:
            result.append(QueryToken(kind="op", value="AND"))
        result.append(token)
        # Only a resolved term (not an operator) advances the "prev_was_term" flag
        prev_was_term = token.kind == "term"
    return result


def to_rpn(tokens: list[QueryToken]) -> list[QueryToken]:
    """Convert tokens to reverse polish notation using operator precedence."""
    output: list[QueryToken] = []
    ops: list[QueryToken] = []
    # Precedence: OR < AND < NOT (higher value = binds tighter)
    precedence = {"OR": 1, "AND": 2, "NOT": 3}
    for token in tokens:
        if token.kind == "term":
            output.append(token)
            continue
        # Pop operators with >= precedence (left-associative: equal priority flushes first)
        while ops and precedence[ops[-1].value] >= precedence[token.value]:
            output.append(ops.pop())
        ops.append(token)
    # Drain any remaining operators onto the output queue
    while ops:
        output.append(ops.pop())
    return output


@functools.lru_cache(maxsize=256)
def _format_categories_cached(
    categories: str,
    color_items: tuple[tuple[str, str], ...],
) -> str:
    """Format categories with colors for a specific resolved palette."""
    category_colors = dict(color_items)
    parts = []
    for cat in categories.split():
        color = category_colors.get(cat, DEFAULT_CATEGORY_COLOR)
        parts.append(f"[{color}]{cat}[/]")
    return " ".join(parts)


def format_categories(
    categories: str,
    category_colors: Mapping[str, str] | None = None,
) -> str:
    """Format categories with colors using the provided or default palette."""
    palette = category_colors or DEFAULT_CATEGORY_COLORS
    return _format_categories_cached(categories, tuple(sorted(palette.items())))


format_categories.cache_clear = _format_categories_cached.cache_clear  # type: ignore[attr-defined]


# ============================================================================
# Query Matching & Paper Sorting
# ============================================================================


def is_advanced_query(tokens: list[QueryToken]) -> bool:
    """Check if a query uses advanced features (operators, fields, phrases, virtual terms)."""
    return any(
        tok.kind == "op" or tok.field or tok.phrase or tok.value.lower() in _VIRTUAL_QUERY_TERMS
        for tok in tokens
    )


def build_highlight_terms(tokens: list[QueryToken]) -> dict[str, list[str]]:
    """Build highlight term lists from query tokens by field.

    Operator tokens (``AND``, ``OR``, ``NOT``) and virtual terms
    (``unread``, ``starred``, ``review-due``) are excluded — only literal match terms are
    collected.  Unscoped terms appear in both ``"title"`` and ``"author"``
    lists so they are highlighted in both columns.

    Args:
        tokens: Parsed query tokens from ``tokenize_query``.

    Returns:
        Dict keyed by ``"title"``, ``"author"``, ``"abstract"``. Unscoped
        terms appear in both ``"title"`` and ``"author"`` lists.
    """
    highlight: dict[str, list[str]] = {"title": [], "author": [], "abstract": []}
    for token in tokens:
        if token.kind != "term":
            continue
        if token.value.lower() in _VIRTUAL_QUERY_TERMS:
            continue
        if token.field == "title":
            highlight["title"].append(token.value)
        elif token.field == "author":
            highlight["author"].append(token.value.removeprefix("="))
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
    fuzzy_search: Callable[[str, list[Paper]], list[Paper]],
    advanced_match: Callable[[Paper, list[QueryToken]], bool],
) -> tuple[list[Paper], dict[str, list[str]]]:
    """Filter papers by a query string, choosing fuzzy or advanced mode.

    Args:
        query: Raw user query string.
        papers: Full paper list to filter.
        fuzzy_search: Callback for simple (non-advanced) queries; receives the
            normalized query and paper list, returns filtered papers.
        advanced_match: Callback for advanced queries (boolean operators, field
            scopes, phrases); receives a paper and RPN token list, returns
            whether the paper matches.

    Returns:
        Tuple of (filtered_papers, highlight_terms) where highlight_terms is a
        dict keyed by field name (see ``build_highlight_terms``).
    """
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

    return fuzzy_search(normalized_query, papers), highlight_terms


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

    Handles field-scoped tokens (``cat:``, ``tag:``, ``title:``, ``author:``,
    ``abstract:``) as well as virtual terms such as ``unread``, ``starred``,
    and ``review-due``.
    Unscoped tokens match against ``title + authors``.

    Args:
        paper: The paper to match against.
        token: The query token to match.
        paper_metadata: The paper's user metadata (for tag/read/star lookups).
            Pass ``None`` when metadata is unavailable; tag/starred terms will
            return ``False`` in that case.
        abstract_text: Pre-cleaned abstract text used for ``abstract:`` scoped
            queries.

    Returns:
        ``True`` if the paper satisfies the token, ``False`` otherwise.
    """
    value = token.value.strip()
    if not value:
        return True
    value_lower = value.lower()
    if token.field:
        return _match_field_query_term(
            paper, token.field, value_lower, paper_metadata, abstract_text
        )
    virtual_match = _match_virtual_query_term(value_lower, paper_metadata)
    if virtual_match is not None:
        return virtual_match
    return value_lower in _default_query_haystack(paper)


def _match_field_query_term(
    paper: Paper,
    field: str,
    value_lower: str,
    paper_metadata: PaperMetadata | None,
    abstract_text: str,
) -> bool:
    """Match a lower-cased query value against one field scope."""
    if field == "tag":
        return bool(paper_metadata) and any(
            value_lower in tag.lower() for tag in paper_metadata.tags
        )
    if field == "author" and value_lower.startswith("="):
        return author_matches_exact(paper.authors, value_lower[1:])

    haystacks = {
        "cat": paper.categories,
        "title": paper.title,
        "author": paper.authors,
        "abstract": abstract_text,
    }
    haystack = haystacks.get(field)
    return bool(haystack) and value_lower in haystack.lower()


def _match_virtual_query_term(
    value_lower: str, paper_metadata: PaperMetadata | None
) -> bool | None:
    """Return a virtual-term match, or None when the value is not virtual."""
    if value_lower == "unread":
        return not paper_metadata or not paper_metadata.is_read
    if value_lower == "starred":
        return bool(paper_metadata and paper_metadata.starred)
    if value_lower == "review-due":
        return is_review_due(paper_metadata, date.today())
    return None


def _default_query_haystack(paper: Paper) -> str:
    """Return the default lower-cased fields used by unscoped terms."""
    return f"{paper.title} {paper.authors}".lower()


def matches_advanced_query(
    paper: Paper,
    rpn: list[QueryToken],
    paper_metadata: PaperMetadata | None,
    abstract_text: str = "",
) -> bool:
    """Evaluate an RPN query expression against a paper.

    Implements a stack-based evaluator for Boolean RPN produced by
    ``to_rpn(insert_implicit_and(tokenize_query(...)))``.  Operands are pushed
    as ``bool`` values; ``NOT``, ``AND``, and ``OR`` pop and push results.
    Defensive pops default to ``False`` for malformed RPN; an empty stack
    after evaluation returns ``True`` (fail-open — match everything).

    Args:
        paper: The paper to match against.
        rpn: Query in Reverse Polish Notation, as returned by ``to_rpn``.
        paper_metadata: The paper's user metadata.
        abstract_text: Pre-cleaned abstract text passed through to
            ``match_query_term``.

    Returns:
        ``True`` if the paper satisfies the expression, ``False`` otherwise.
        Returns ``True`` for an empty *rpn* list (no filter applied).
    """
    if not rpn:
        return True
    stack: list[bool] = []
    for token in rpn:
        if token.kind == "term":
            stack.append(match_query_term(paper, token, paper_metadata, abstract_text))
            continue
        _apply_rpn_operator(stack, token.value)
    return _final_rpn_result(stack)


def _pop_rpn_value(stack: list[bool]) -> bool:
    """Pop a stack value, using False for malformed RPN."""
    return stack.pop() if stack else False


def _apply_rpn_operator(stack: list[bool], operator: str) -> None:
    """Apply one Boolean RPN operator in place."""
    if operator == "NOT":
        stack.append(not _pop_rpn_value(stack))
        return

    right = _pop_rpn_value(stack)
    left = _pop_rpn_value(stack)
    if operator == "AND":
        stack.append(left and right)
        return
    stack.append(left or right)


def _final_rpn_result(stack: list[bool]) -> bool:
    """Return the expression result; empty expressions fail open."""
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


def _build_queue_score_context(
    papers: list[Paper],
    s2_cache: dict[str, SemanticScholarPaper] | None,
    hf_cache: dict[str, HuggingFacePaper] | None,
    relevance_cache: dict[str, tuple[int, str]] | None,
    watched_paper_ids: set[str] | None,
    today: datetime | None = None,
) -> _QueueScoreContext:
    resolved_today = today or datetime.now()
    context = _QueueScoreContext(
        s2_cache=s2_cache,
        hf_cache=hf_cache,
        relevance_cache=relevance_cache,
        watched_paper_ids=watched_paper_ids or set(),
        today=resolved_today,
        max_hf_upvote_log=0.0,
        max_velocity_log=0.0,
    )
    return _QueueScoreContext(
        s2_cache=s2_cache,
        hf_cache=hf_cache,
        relevance_cache=relevance_cache,
        watched_paper_ids=context.watched_paper_ids,
        today=resolved_today,
        max_hf_upvote_log=max((_queue_hf_upvote_log(p, context) for p in papers), default=0.0),
        max_velocity_log=max((_queue_velocity_log(p, context) for p in papers), default=0.0),
    )


def _queue_sort_key(paper: Paper, context: _QueueScoreContext) -> float:
    """Return a descending sort key for the smart reading queue."""
    return -_queue_score(paper, context)


def _queue_score(paper: Paper, context: _QueueScoreContext) -> float:
    """Return the composite smart-reading priority score for a paper."""
    return (
        _QUEUE_RELEVANCE_WEIGHT * _queue_relevance_score(paper, context)
        + _QUEUE_WATCH_WEIGHT * _queue_watch_score(paper, context)
        + _QUEUE_RECENCY_WEIGHT * _queue_recency_score(paper, context)
        + _QUEUE_HF_WEIGHT
        * _normalized_log(_queue_hf_upvote_log(paper, context), context.max_hf_upvote_log)
        + _QUEUE_VELOCITY_WEIGHT
        * _normalized_log(_queue_velocity_log(paper, context), context.max_velocity_log)
    )


def _queue_relevance_score(paper: Paper, context: _QueueScoreContext) -> float:
    rel = context.relevance_cache.get(paper.arxiv_id) if context.relevance_cache else None
    if rel is None:
        return 0.0
    return max(0.0, min(float(rel[0]), 10.0)) / 10.0


def _queue_watch_score(paper: Paper, context: _QueueScoreContext) -> float:
    return 1.0 if paper.arxiv_id in context.watched_paper_ids else 0.0


def _queue_recency_score(paper: Paper, context: _QueueScoreContext) -> float:
    age_days = _queue_paper_age_days(paper, context.today)
    if age_days is None:
        return 0.0
    return 1.0 / (1.0 + age_days / _QUEUE_RECENCY_DECAY_DAYS)


def _queue_hf_upvote_log(paper: Paper, context: _QueueScoreContext) -> float:
    hf = context.hf_cache.get(paper.arxiv_id) if context.hf_cache else None
    if hf is None or hf.upvotes <= 0:
        return 0.0
    return math.log1p(hf.upvotes)


def _queue_velocity_log(paper: Paper, context: _QueueScoreContext) -> float:
    s2 = context.s2_cache.get(paper.arxiv_id) if context.s2_cache else None
    if s2 is None or s2.citation_count <= 0:
        return 0.0
    age_days = _queue_paper_age_days(paper, context.today)
    if age_days is None:
        return 0.0
    age_years = max(age_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR)
    return math.log1p(s2.citation_count / age_years)


def _queue_paper_age_days(paper: Paper, today: datetime) -> float | None:
    parsed = parse_arxiv_date(paper.date)
    if parsed == datetime.min:
        return None
    return max((today.date() - parsed.date()).days, 0)


def _normalized_log(value: float, max_value: float) -> float:
    if value <= 0.0 or max_value <= 0.0:
        return 0.0
    return min(value / max_value, 1.0)


def _resolve_sort_signals(
    signals: PaperSortSignals | None,
    legacy_signals: dict[str, object],
) -> PaperSortSignals:
    allowed = {
        "s2_cache",
        "hf_cache",
        "relevance_cache",
        "watched_paper_ids",
        "triage_predictions",
    }
    unknown = sorted(set(legacy_signals) - allowed)
    if unknown:
        raise TypeError(f"Unknown sort signal(s): {', '.join(unknown)}")
    base = signals or PaperSortSignals()
    if not legacy_signals:
        return base
    return PaperSortSignals(
        s2_cache=legacy_signals.get("s2_cache", base.s2_cache),  # type: ignore[arg-type]
        hf_cache=legacy_signals.get("hf_cache", base.hf_cache),  # type: ignore[arg-type]
        relevance_cache=legacy_signals.get("relevance_cache", base.relevance_cache),  # type: ignore[arg-type]
        watched_paper_ids=legacy_signals.get("watched_paper_ids", base.watched_paper_ids),  # type: ignore[arg-type]
        triage_predictions=legacy_signals.get(
            "triage_predictions",
            base.triage_predictions,
        ),  # type: ignore[arg-type]
    )


def _triage_sort_key(
    paper: Paper,
    predictions: dict[str, TriagePrediction] | None,
) -> tuple[int, float, float]:
    prediction = predictions.get(paper.arxiv_id) if predictions else None
    if prediction is None:
        return (3, 0.0, 0.0)
    probability = max(0.0, min(1.0, prediction.probability))
    if prediction.bucket == TRIAGE_BUCKET_LIKELY_STAR:
        return (0, -probability, 0.0)
    if prediction.bucket == TRIAGE_BUCKET_UNSURE:
        return (1, abs(probability - 0.5), -probability)
    if prediction.bucket == TRIAGE_BUCKET_LIKELY_SKIP:
        return (2, -probability, 0.0)
    return (3, 0.0, 0.0)


def sort_papers(
    papers: list[Paper],
    sort_key: str,
    signals: PaperSortSignals | None = None,
    **legacy_signals: object,
) -> list[Paper]:
    """Sort papers by the given key, returning a new sorted list.

    For cache-backed sort modes (``citations``, ``trending``, ``relevance``),
    papers with a cache entry sort before papers without one using a two-tuple
    key ``(0, -value)`` < ``(1, 0)``.  Within the first group the value is
    negated to produce a descending order. The ``queue`` mode combines cached
    relevance, watch-list matches, recency, HF upvotes, and a local S2 citation
    velocity proxy into one stable priority rank.

    Args:
        papers: List of papers to sort.
        sort_key: One of ``"title"``, ``"date"``, ``"arxiv_id"``,
            ``"citations"``, ``"trending"``, ``"relevance"``, ``"queue"``,
            ``"triage"``.
        signals: Optional grouped cache inputs used by cache-backed sort modes.
        legacy_signals: Backward-compatible keyword inputs such as
            ``relevance_cache`` and ``watched_paper_ids``.

    Returns:
        A new sorted list.  Unknown *sort_key* values return a shallow copy
        of the input list in its original order.
    """
    signals = _resolve_sort_signals(signals, legacy_signals)
    if sort_key == "title":
        return sorted(papers, key=lambda p: p.title.lower())
    elif sort_key == "date":
        return sorted(papers, key=lambda p: parse_arxiv_date(p.date), reverse=True)
    elif sort_key == "arxiv_id":
        return sorted(papers, key=lambda p: p.arxiv_id, reverse=True)
    elif sort_key == "citations":

        def _citation_key(p: Paper) -> tuple[int, int]:
            s2 = signals.s2_cache.get(p.arxiv_id) if signals.s2_cache else None
            if s2 is not None:
                # (0, -count) sorts before (1, 0); negated count gives descending order
                return (0, -s2.citation_count)
            return (1, 0)  # Papers without S2 data sort last

        return sorted(papers, key=_citation_key)
    elif sort_key == "trending":

        def _trending_key(p: Paper) -> tuple[int, int]:
            hf = signals.hf_cache.get(p.arxiv_id) if signals.hf_cache else None
            if hf is not None:
                return (0, -hf.upvotes)
            return (1, 0)  # Papers without HF data sort last

        return sorted(papers, key=_trending_key)
    elif sort_key == "relevance":

        def _relevance_key(p: Paper) -> tuple[int, int]:
            rel = signals.relevance_cache.get(p.arxiv_id) if signals.relevance_cache else None
            if rel is not None:
                return (0, -rel[0])
            return (1, 0)  # Papers without a relevance score sort last

        return sorted(papers, key=_relevance_key)
    elif sort_key == "queue":
        context = _build_queue_score_context(
            papers,
            signals.s2_cache,
            signals.hf_cache,
            signals.relevance_cache,
            signals.watched_paper_ids,
        )
        return sorted(papers, key=lambda p: _queue_sort_key(p, context))
    elif sort_key == "triage":
        return sorted(papers, key=lambda p: _triage_sort_key(p, signals.triage_predictions))
    return list(papers)


__all__ = [
    "_HIGHLIGHT_PATTERN_CACHE",
    "PaperSortSignals",
    "build_highlight_terms",
    "escape_rich_text",
    "format_categories",
    "format_summary_as_rich",
    "highlight_text",
    "insert_implicit_and",
    "is_advanced_query",
    "match_query_term",
    "matches_advanced_query",
    "paper_matches_watch_entry",
    "pill_label_for_token",
    "reconstruct_query",
    "render_progress_bar",
    "sort_papers",
    "to_rpn",
    "tokenize_query",
    "truncate_at_word_boundary",
    "truncate_text",
]
