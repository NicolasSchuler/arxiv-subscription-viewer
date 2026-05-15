"""Local-history analytics for the Trend Radar overlay."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

from arxiv_browser.authors import AuthorCount, split_author_names
from arxiv_browser.models import STOPWORDS, Paper
from arxiv_browser.parsing import clean_latex, parse_arxiv_file

RECENT_WINDOW_DATES = 10
TREND_RADAR_TOP_N = 8
UNKNOWN_CATEGORY = "unknown"
SPARKLINE_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
ASCII_SPARKLINE_CHARS = "12345678"
_CATEGORY_RE = re.compile(r"^[A-Za-z-]+(?:\.[A-Za-z-]+)?$")
_BIGRAM_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")


@dataclass(slots=True, frozen=True)
class CategoryTrend:
    """Trend row for one primary category."""

    category: str
    counts: tuple[int, ...]
    recent_count: int
    previous_count: int
    delta: int


@dataclass(slots=True, frozen=True)
class BigramCount:
    """Counted topic bigram from recent abstracts."""

    bigram: str
    count: int


@dataclass(slots=True, frozen=True)
class TrendRadarReport:
    """Complete analytics payload for the Trend Radar modal."""

    dates: tuple[date, ...]
    category_trends: tuple[CategoryTrend, ...]
    top_authors: tuple[AuthorCount, ...]
    hot_bigrams: tuple[BigramCount, ...]
    total_papers: int
    history_file_count: int
    recent_file_count: int
    previous_file_count: int


def render_sparkline(values: Sequence[int], *, ascii_mode: bool | None = None) -> str:
    """Render a sequence as a Unicode or ASCII sparkline."""
    if ascii_mode is None:
        from arxiv_browser._ascii import is_ascii_mode

        ascii_mode = is_ascii_mode()
    chars = ASCII_SPARKLINE_CHARS if ascii_mode else SPARKLINE_CHARS
    if not values:
        return ""
    safe_values = [max(0, value) for value in values]
    low = min(safe_values)
    high = max(safe_values)
    if high == low:
        return chars[0] * len(safe_values)
    scale = len(chars) - 1
    return "".join(chars[round((value - low) / (high - low) * scale)] for value in safe_values)


def primary_category(categories: str) -> str:
    """Return a paper's primary category, or ``unknown`` for missing/malformed values."""
    first = categories.split()[0] if categories else ""
    return first if first and _CATEGORY_RE.match(first) else UNKNOWN_CATEGORY


def build_trend_report(
    history_files: Sequence[tuple[date, Path]],
    *,
    recent_window: int = RECENT_WINDOW_DATES,
    top_n: int = TREND_RADAR_TOP_N,
) -> TrendRadarReport:
    """Parse local history files and build a Trend Radar report."""
    dated_papers: list[tuple[date, list[Paper]]] = []
    for day, path in history_files:
        try:
            dated_papers.append((day, parse_arxiv_file(path)))
        except OSError:
            dated_papers.append((day, []))
    return build_trend_report_from_papers(dated_papers, recent_window=recent_window, top_n=top_n)


def build_trend_report_from_papers(
    dated_papers: Sequence[tuple[date, Sequence[Paper]]],
    *,
    recent_window: int = RECENT_WINDOW_DATES,
    top_n: int = TREND_RADAR_TOP_N,
) -> TrendRadarReport:
    """Build a Trend Radar report from already parsed dated paper groups."""
    normalized = _dedupe_dated_papers(dated_papers)
    if not normalized:
        return _empty_report()

    newest_first = sorted(normalized, key=lambda item: item[0], reverse=True)
    recent_count = max(0, recent_window)
    recent = newest_first[:recent_count]
    previous = newest_first[recent_count : recent_count * 2]
    display = sorted([*recent, *previous], key=lambda item: item[0])
    dates = tuple(day for day, _papers in display)
    recent_dates = {day for day, _papers in recent}
    previous_dates = {day for day, _papers in previous}
    category_trends = _category_trends(display, recent_dates, previous_dates, top_n)
    all_papers = [paper for _day, papers in normalized for paper in papers]
    recent_papers = [paper for _day, papers in recent for paper in papers]
    return TrendRadarReport(
        dates=dates,
        category_trends=category_trends,
        top_authors=_top_authors(all_papers, top_n),
        hot_bigrams=_hot_bigrams(recent_papers, top_n),
        total_papers=len(all_papers),
        history_file_count=len(normalized),
        recent_file_count=len(recent),
        previous_file_count=len(previous),
    )


def _empty_report() -> TrendRadarReport:
    return TrendRadarReport((), (), (), (), 0, 0, 0, 0)


def _dedupe_dated_papers(
    dated_papers: Sequence[tuple[date, Sequence[Paper]]],
) -> list[tuple[date, list[Paper]]]:
    result: list[tuple[date, list[Paper]]] = []
    for day, papers in dated_papers:
        seen: set[str] = set()
        deduped: list[Paper] = []
        for paper in papers:
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            deduped.append(paper)
        result.append((day, deduped))
    return result


def _category_trends(
    display: Sequence[tuple[date, Sequence[Paper]]],
    recent_dates: set[date],
    previous_dates: set[date],
    top_n: int,
) -> tuple[CategoryTrend, ...]:
    categories = sorted(
        {primary_category(paper.categories) for _day, papers in display for paper in papers}
    )
    rows: list[CategoryTrend] = []
    for category in categories:
        counts = tuple(
            sum(1 for paper in papers if primary_category(paper.categories) == category)
            for _day, papers in display
        )
        recent_total = sum(
            count
            for (day, _papers), count in zip(display, counts, strict=True)
            if day in recent_dates
        )
        previous_total = sum(
            count
            for (day, _papers), count in zip(display, counts, strict=True)
            if day in previous_dates
        )
        rows.append(
            CategoryTrend(
                category=category,
                counts=counts,
                recent_count=recent_total,
                previous_count=previous_total,
                delta=recent_total - previous_total,
            )
        )
    rows.sort(key=lambda row: (-row.delta, -row.recent_count, row.category))
    return tuple(rows[:top_n])


def _top_authors(papers: Sequence[Paper], top_n: int) -> tuple[AuthorCount, ...]:
    counts: Counter[str] = Counter()
    for paper in papers:
        counts.update(split_author_names(paper.authors))
    return tuple(
        AuthorCount(name, count)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    )


def _hot_bigrams(papers: Sequence[Paper], top_n: int) -> tuple[BigramCount, ...]:
    counts: Counter[str] = Counter()
    for paper in papers:
        abstract = paper.abstract if paper.abstract is not None else clean_latex(paper.abstract_raw)
        tokens = [
            token
            for token in _BIGRAM_TOKEN_RE.findall(abstract.casefold())
            if len(token) >= 3 and token not in STOPWORDS
        ]
        counts.update(f"{left} {right}" for left, right in pairwise(tokens))
    return tuple(
        BigramCount(bigram, count)
        for bigram, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    )


__all__ = [
    "ASCII_SPARKLINE_CHARS",
    "RECENT_WINDOW_DATES",
    "SPARKLINE_CHARS",
    "TREND_RADAR_TOP_N",
    "UNKNOWN_CATEGORY",
    "BigramCount",
    "CategoryTrend",
    "TrendRadarReport",
    "build_trend_report",
    "build_trend_report_from_papers",
    "primary_category",
    "render_sparkline",
]
