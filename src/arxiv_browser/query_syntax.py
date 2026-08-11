"""Token grammar and query-string reconstruction helpers."""

from __future__ import annotations

from arxiv_browser.models import QueryToken

_FIELD_NAMES = frozenset({"title", "author", "abstract", "cat", "tag"})


def _scan_quoted_value(query: str, i: int, query_len: int) -> tuple[str, int]:
    """Scan through a closing quote and return the value and next index."""
    start = i
    while i < query_len and query[i] != '"':
        i += 1
    return query[start:i], i + 1


def _scan_plain_value(query: str, i: int, query_len: int) -> tuple[str, int]:
    """Scan one whitespace-delimited value and return it with the next index."""
    start = i
    while i < query_len and not query[i].isspace():
        i += 1
    return query[start:i], i


def _parse_quoted_phrase(query: str, i: int, query_len: int) -> tuple[QueryToken, int]:
    """Parse a quoted phrase starting after the opening quote."""
    value, next_index = _scan_quoted_value(query, i, query_len)
    return QueryToken(kind="term", value=value, phrase=True), next_index


def _parse_field_value(query: str, i: int, query_len: int, field: str) -> tuple[QueryToken, int]:
    """Parse the value after a field:colon, handling both quoted and unquoted."""
    exact_prefix = field == "author" and i < query_len and query[i] == "="
    if exact_prefix:
        i += 1
    phrase = i < query_len and query[i] == '"'
    value, next_index = (
        _scan_quoted_value(query, i + 1, query_len)
        if phrase
        else _scan_plain_value(query, i, query_len)
    )
    if exact_prefix:
        value = f"={value}"
    return QueryToken(kind="term", value=value, field=field, phrase=phrase), next_index


def _parse_plain_term(query: str, start: int, i: int, query_len: int) -> tuple[QueryToken, int]:
    """Parse a plain term or boolean operator, advancing past it."""
    while i < query_len and not query[i].isspace():
        i += 1
    raw = query[start:i]
    upper = raw.upper()
    if upper in {"AND", "OR", "NOT"}:
        return QueryToken(kind="op", value=upper), i
    return QueryToken(kind="term", value=raw), i


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
