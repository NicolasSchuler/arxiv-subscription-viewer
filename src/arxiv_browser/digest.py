"""Headless Markdown digest generation for cron-friendly CLI use."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from arxiv_browser.config import save_config
from arxiv_browser.enrichment import apply_version_updates, get_starred_paper_ids_for_version_check
from arxiv_browser.huggingface import (
    HF_DEFAULT_CACHE_TTL_HOURS,
    HuggingFacePaper,
    fetch_hf_daily_papers,
    get_hf_db_path,
    load_hf_daily_cache_snapshot,
    save_hf_daily_cache,
)
from arxiv_browser.llm import (
    _compute_command_hash,
    _load_all_relevance_scores,
    _resolve_llm_command,
    _save_relevance_score,
    get_relevance_db_path,
)
from arxiv_browser.llm_providers import LLMProvider, resolve_provider
from arxiv_browser.models import ArxivSearchRequest, Paper, UserConfig
from arxiv_browser.parsing import build_daily_digest, parse_arxiv_file, parse_arxiv_version_map
from arxiv_browser.query import paper_matches_watch_entry
from arxiv_browser.services.arxiv_api_service import (
    ARXIV_API_MIN_INTERVAL_SECONDS,
    ARXIV_API_TIMEOUT,
    ARXIV_API_URL,
    ARXIV_API_USER_AGENT,
    fetch_recent_digest,
    format_query_label,
)
from arxiv_browser.services.llm_service import score_relevance_once
from arxiv_browser.triage_model import (
    TRIAGE_BUCKET_LIKELY_STAR,
    TRIAGE_BUCKET_UNSURE,
    MissingTriageModelDependencyError,
    TriageModelInfo,
    TriagePrediction,
    format_triage_prediction,
    load_triage_model,
    predict_triage,
)

logger = logging.getLogger(__name__)

DigestPeriod = Literal["daily", "weekly"]
DigestHfMode = Literal["config", "include", "off"]
DigestRelevanceMode = Literal["score", "cached", "off"]

_VERSION_BATCH_SIZE = 40
_RELEVANCE_CONCURRENCY = 3
_DEFAULT_LIMIT = 10
_DEFAULT_MIN_RELEVANCE = 7
_MARKDOWN_SPECIALS = "\\`*_{}[]()#+!|"


class DigestError(RuntimeError):
    """Raised when a digest cannot be generated from the requested source."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class DigestOptions:
    """Options for one Markdown digest generation run."""

    input_path: Path | None = None
    request: ArxivSearchRequest = field(default_factory=lambda: ArxivSearchRequest(query=""))
    period: DigestPeriod = "daily"
    max_results: int = 50
    limit: int = _DEFAULT_LIMIT
    min_relevance: int = _DEFAULT_MIN_RELEVANCE
    hf_mode: DigestHfMode = "config"
    relevance_mode: DigestRelevanceMode = "score"
    versions_enabled: bool = True
    include_triage: bool = False


@dataclass(slots=True)
class DigestResult:
    """Generated digest payload plus non-fatal enrichment warnings."""

    markdown: str
    warnings: list[str]
    paper_count: int
    source_label: str


@dataclass(slots=True)
class VersionFetchResult:
    """Version API result with partial-failure accounting."""

    versions: dict[str, int]
    failed_batches: int = 0


@dataclass(slots=True)
class DigestDependencies:
    """Injectable collaborators for deterministic tests and CLI wrappers."""

    clock: Callable[[], datetime] = _utc_now
    parse_input_file: Callable[[Path], list[Paper]] = parse_arxiv_file
    fetch_recent_papers: Callable[..., list[Paper]] = fetch_recent_digest
    load_relevance_scores: Callable[[Path, str], dict[str, tuple[int, str]]] = (
        _load_all_relevance_scores
    )
    save_relevance_score: Callable[..., None] = _save_relevance_score
    resolve_provider_fn: Callable[[UserConfig], LLMProvider | None] = resolve_provider
    score_relevance_once_fn: Callable[..., Awaitable[tuple[int, str] | None]] = score_relevance_once
    relevance_db_path_fn: Callable[[], Path] = get_relevance_db_path
    hf_db_path_fn: Callable[[], Path] = get_hf_db_path
    load_hf_snapshot_fn: Callable[..., object] = load_hf_daily_cache_snapshot
    save_hf_cache_fn: Callable[[Path, list[HuggingFacePaper]], None] = save_hf_daily_cache
    fetch_hf_papers_fn: Callable[[int], tuple[list[HuggingFacePaper], bool]] | None = None
    fetch_versions_fn: Callable[[set[str]], VersionFetchResult] | None = None
    save_config_fn: Callable[[UserConfig], bool] = save_config
    load_triage_model_fn: Callable[[], tuple[Any, TriageModelInfo] | None] = load_triage_model
    predict_triage_fn: Callable[[list[Paper], Any], dict[str, TriagePrediction]] = predict_triage


@dataclass(slots=True)
class _DigestData:
    papers: list[Paper]
    source_label: str
    watched_ids: set[str]
    relevance_scores: dict[str, tuple[int, str]]
    triage_predictions: dict[str, TriagePrediction]
    hf_matches: dict[str, HuggingFacePaper]
    version_updates: dict[str, tuple[int, int]]


@dataclass(slots=True)
class _RelevanceContext:
    config: UserConfig
    deps: DigestDependencies
    interests: str
    interests_hash: str
    provider: LLMProvider
    warnings: list[str]


def generate_digest(
    options: DigestOptions,
    config: UserConfig,
    deps: DigestDependencies | None = None,
) -> DigestResult:
    """Generate a Markdown digest from live arXiv search or a local input file."""
    resolved_deps = deps or DigestDependencies()
    warnings: list[str] = []
    papers, source_label = _load_digest_source(options, resolved_deps)
    papers_by_id = {paper.arxiv_id: paper for paper in papers}
    watched_ids = _collect_watched_ids(papers, config)
    relevance_scores = _load_relevance(options, config, papers, warnings, resolved_deps)
    triage_predictions = _load_triage_predictions(options, papers, warnings, resolved_deps)
    hf_matches = _load_hf_matches(options, config, papers_by_id, warnings, resolved_deps)
    version_updates = (
        _load_version_updates(config, warnings, resolved_deps) if options.versions_enabled else {}
    )

    data = _DigestData(
        papers=papers,
        source_label=source_label,
        watched_ids=watched_ids,
        relevance_scores=relevance_scores,
        triage_predictions=triage_predictions,
        hf_matches=hf_matches,
        version_updates=version_updates,
    )
    markdown = render_digest_markdown(data, options, config, generated_at=resolved_deps.clock())
    return DigestResult(
        markdown=markdown,
        warnings=warnings,
        paper_count=len(papers),
        source_label=source_label,
    )


def _load_digest_source(
    options: DigestOptions,
    deps: DigestDependencies,
) -> tuple[list[Paper], str]:
    if options.input_path is not None:
        return _load_input_source(options.input_path, deps)
    try:
        papers = deps.fetch_recent_papers(
            request=options.request,
            period=options.period,
            max_results=options.max_results,
        )
    except ValueError as exc:
        raise DigestError(str(exc)) from exc
    except (httpx.HTTPError, OSError) as exc:
        raise DigestError("Failed to fetch papers from arXiv API") from exc
    return papers, _live_source_label(options)


def _load_input_source(path: Path, deps: DigestDependencies) -> tuple[list[Paper], str]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise DigestError(f"{resolved} not found")
    if resolved.is_dir():
        raise DigestError(f"{resolved} is a directory, not a file")
    try:
        papers = deps.parse_input_file(resolved)
    except OSError as exc:
        raise DigestError(f"Failed to read {resolved}: {exc}") from exc
    return papers, f"input {resolved.name}"


def _live_source_label(options: DigestOptions) -> str:
    try:
        query_label = format_query_label(options.request)
    except ValueError:
        query_label = options.request.query or f"cat:{options.request.category}"
    return f"arXiv {options.period} search ({query_label})"


def _collect_watched_ids(papers: list[Paper], config: UserConfig) -> set[str]:
    watched: set[str] = set()
    if not config.watch_list:
        return watched
    for paper in papers:
        if any(paper_matches_watch_entry(paper, entry) for entry in config.watch_list):
            watched.add(paper.arxiv_id)
    return watched


def _load_relevance(
    options: DigestOptions,
    config: UserConfig,
    papers: list[Paper],
    warnings: list[str],
    deps: DigestDependencies,
) -> dict[str, tuple[int, str]]:
    if options.relevance_mode == "off":
        return {}
    if not config.research_interests:
        warnings.append("Relevance skipped: research_interests is not configured.")
        return {}

    cache_key = _relevance_cache_key(config)
    if not cache_key:
        warnings.append("Relevance skipped: no LLM command or HTTP provider is configured.")
        return {}

    interests_hash = _compute_command_hash(cache_key, config.research_interests)
    scores = _load_cached_relevance(deps, interests_hash, warnings)
    missing = [paper for paper in papers if paper.arxiv_id not in scores]
    if options.relevance_mode == "cached" or not missing:
        return scores
    if _fresh_relevance_blocked(config, cache_key, warnings):
        return scores

    provider = deps.resolve_provider_fn(config)
    if provider is None:
        warnings.append("Relevance skipped: configured LLM provider could not be resolved.")
        return scores

    context = _RelevanceContext(
        config=config,
        deps=deps,
        interests=config.research_interests,
        interests_hash=interests_hash,
        provider=provider,
        warnings=warnings,
    )
    asyncio.run(_score_missing_relevance(missing, scores, context))
    return scores


def _relevance_cache_key(config: UserConfig) -> str:
    if config.llm_provider_type.lower() == "http":
        if not config.llm_api_base_url:
            return ""
        return f"http:{config.llm_api_base_url}|{config.llm_api_model}"
    return _resolve_llm_command(config)


def _load_cached_relevance(
    deps: DigestDependencies,
    interests_hash: str,
    warnings: list[str],
) -> dict[str, tuple[int, str]]:
    try:
        return deps.load_relevance_scores(deps.relevance_db_path_fn(), interests_hash)
    except OSError:
        warnings.append("Relevance cache could not be read.")
        return {}


def _fresh_relevance_blocked(
    config: UserConfig,
    cache_key: str,
    warnings: list[str],
) -> bool:
    if config.llm_provider_type.lower() == "http":
        return False
    if not config.llm_command:
        return False
    command_hash = _command_trust_hash(cache_key)
    if command_hash in config.trusted_llm_command_hashes:
        return False
    warnings.append("Fresh relevance scoring skipped: custom LLM command is not trusted.")
    return True


def _command_trust_hash(command_template: str) -> str:
    import hashlib

    return hashlib.sha256(command_template.encode("utf-8")).hexdigest()[:16]


async def _score_missing_relevance(
    papers: list[Paper],
    scores: dict[str, tuple[int, str]],
    context: _RelevanceContext,
) -> None:
    sem = asyncio.Semaphore(_RELEVANCE_CONCURRENCY)
    failures = 0
    results = await asyncio.gather(*(_score_one_relevance(paper, context, sem) for paper in papers))
    for paper, result in zip(papers, results, strict=True):
        if result is None:
            failures += 1
            continue
        score, reason = result
        scores[paper.arxiv_id] = (score, reason)
        await asyncio.to_thread(
            context.deps.save_relevance_score,
            context.deps.relevance_db_path_fn(),
            paper.arxiv_id,
            context.interests_hash,
            score,
            reason,
        )
    if failures:
        context.warnings.append(f"Relevance scoring failed for {failures} paper(s).")


async def _score_one_relevance(
    paper: Paper,
    context: _RelevanceContext,
    sem: asyncio.Semaphore,
) -> tuple[int, str] | None:
    async with sem:
        try:
            result = await context.deps.score_relevance_once_fn(
                paper=paper,
                interests=context.interests,
                provider=context.provider,
                timeout_seconds=context.config.llm_timeout,
            )
        except Exception:
            logger.warning("Digest relevance scoring failed for %s", paper.arxiv_id, exc_info=True)
            return None
        return result if isinstance(result, tuple) else None


def _load_triage_predictions(
    options: DigestOptions,
    papers: list[Paper],
    warnings: list[str],
    deps: DigestDependencies,
) -> dict[str, TriagePrediction]:
    if not options.include_triage:
        return {}
    try:
        loaded = deps.load_triage_model_fn()
    except MissingTriageModelDependencyError:
        warnings.append(
            "Triage model skipped: install ML extras with `pip install arxiv-subscription-viewer[ml]`."
        )
        return {}
    except (OSError, ValueError, RuntimeError):
        warnings.append("Triage model could not be loaded.")
        return {}
    if loaded is None:
        warnings.append("Triage model skipped: no trained model found.")
        return {}
    model, _info = loaded
    try:
        return deps.predict_triage_fn(papers, model)
    except (ValueError, RuntimeError):
        warnings.append("Triage model prediction failed.")
        return {}


def _load_hf_matches(
    options: DigestOptions,
    config: UserConfig,
    papers_by_id: dict[str, Paper],
    warnings: list[str],
    deps: DigestDependencies,
) -> dict[str, HuggingFacePaper]:
    if options.hf_mode == "off" or (options.hf_mode == "config" and not config.hf_enabled):
        return {}
    db_path = deps.hf_db_path_fn()
    ttl = config.hf_cache_ttl_hours or HF_DEFAULT_CACHE_TTL_HOURS
    snapshot = deps.load_hf_snapshot_fn(db_path, ttl)
    status = getattr(snapshot, "status", "miss")
    papers = getattr(snapshot, "papers", {})
    if status == "found" and isinstance(papers, dict):
        return _filter_hf_matches(papers, papers_by_id)
    if status == "empty":
        return {}

    fetch_fn = deps.fetch_hf_papers_fn or _fetch_hf_papers_sync
    try:
        fetched, complete = fetch_fn(ttl)
    except (httpx.HTTPError, OSError, RuntimeError):
        warnings.append("Hugging Face trending fetch failed.")
        return {}
    if not complete:
        warnings.append("Hugging Face trending fetch failed.")
        return {}
    deps.save_hf_cache_fn(db_path, fetched)
    return _filter_hf_matches({paper.arxiv_id: paper for paper in fetched}, papers_by_id)


def _filter_hf_matches(
    hf_papers: dict[str, HuggingFacePaper],
    papers_by_id: dict[str, Paper],
) -> dict[str, HuggingFacePaper]:
    return {arxiv_id: paper for arxiv_id, paper in hf_papers.items() if arxiv_id in papers_by_id}


def _fetch_hf_papers_sync(timeout: int) -> tuple[list[HuggingFacePaper], bool]:
    async def _fetch() -> tuple[list[HuggingFacePaper], bool]:
        async with httpx.AsyncClient() as client:
            return await fetch_hf_daily_papers(client, timeout=timeout, include_status=True)

    return asyncio.run(_fetch())


def _load_version_updates(
    config: UserConfig,
    warnings: list[str],
    deps: DigestDependencies,
) -> dict[str, tuple[int, int]]:
    arxiv_ids = get_starred_paper_ids_for_version_check(config.paper_metadata)
    if not arxiv_ids:
        return {}
    fetch_fn = deps.fetch_versions_fn or fetch_arxiv_versions
    try:
        result = fetch_fn(arxiv_ids)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError):
        warnings.append("Version check failed.")
        return {}
    if result.failed_batches:
        warnings.append(f"Version check skipped {result.failed_batches} batch(es).")
    updates: dict[str, tuple[int, int]] = {}
    if result.versions:
        apply_version_updates(result.versions, config.paper_metadata, updates)
        if not deps.save_config_fn(config):
            warnings.append("Version baselines updated in memory but could not be saved.")
    return updates


def fetch_arxiv_versions(arxiv_ids: set[str]) -> VersionFetchResult:
    """Fetch current arXiv versions for IDs, continuing past failed batches."""
    versions: dict[str, int] = {}
    failed_batches = 0
    ids = sorted(arxiv_ids)
    for start in range(0, len(ids), _VERSION_BATCH_SIZE):
        if start:
            time.sleep(ARXIV_API_MIN_INTERVAL_SECONDS)
        batch = ids[start : start + _VERSION_BATCH_SIZE]
        try:
            response = httpx.get(
                ARXIV_API_URL,
                params={"id_list": ",".join(batch), "max_results": len(batch) + 10},
                headers={"User-Agent": ARXIV_API_USER_AGENT},
                timeout=ARXIV_API_TIMEOUT,
            )
            response.raise_for_status()
            versions.update(parse_arxiv_version_map(response.text))
        except (httpx.HTTPError, ValueError, OSError):
            logger.warning("Version check batch failed", exc_info=True)
            failed_batches += 1
    return VersionFetchResult(versions=versions, failed_batches=failed_batches)


def render_digest_markdown(
    data: _DigestData,
    options: DigestOptions,
    config: UserConfig,
    *,
    generated_at: datetime,
) -> str:
    """Render resolved digest data as Markdown."""
    lines = [
        "# arXiv Digest",
        "",
        f"Generated: {_format_generated_at(generated_at)}",
        f"Source: {_escape_markdown(data.source_label)}",
        "",
    ]
    if not data.papers:
        lines.append("No matching papers.")
        return "\n".join(lines).rstrip() + "\n"

    lines.append(build_daily_digest(data.papers, data.watched_ids, config.paper_metadata))
    lines.append("")
    _append_paper_section(lines, "Watch List Matches", _watched_papers(data), options.limit)
    _append_relevance_section(lines, data, options)
    _append_triage_sections(lines, data, options.limit)
    _append_hf_section(lines, data, options.limit)
    _append_version_section(lines, data, options.limit)
    _append_paper_section(lines, "New Papers", data.papers, options.limit)
    return "\n".join(lines).rstrip() + "\n"


def _format_generated_at(generated_at: datetime) -> str:
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    return generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _watched_papers(data: _DigestData) -> list[Paper]:
    return [paper for paper in data.papers if paper.arxiv_id in data.watched_ids]


def _append_paper_section(
    lines: list[str],
    title: str,
    papers: list[Paper],
    limit: int,
) -> None:
    if not papers:
        return
    lines.extend([f"## {title}", ""])
    lines.extend(_render_paper_item(paper) for paper in papers[:limit])
    _append_omitted_count(lines, len(papers), limit)
    lines.append("")


def _append_relevance_section(
    lines: list[str],
    data: _DigestData,
    options: DigestOptions,
) -> None:
    if not data.relevance_scores:
        return
    scored = [
        paper
        for paper in data.papers
        if (score := data.relevance_scores.get(paper.arxiv_id))
        and score[0] >= options.min_relevance
    ]
    scored.sort(key=lambda paper: data.relevance_scores[paper.arxiv_id][0], reverse=True)
    if not scored:
        return
    lines.extend(["## High Relevance", ""])
    for paper in scored[: options.limit]:
        score, reason = data.relevance_scores[paper.arxiv_id]
        suffix = f"relevance {score}/10"
        if reason:
            suffix += f" - {_escape_markdown(reason)}"
        lines.append(_render_paper_item(paper, suffix=suffix))
    _append_omitted_count(lines, len(scored), options.limit)
    lines.append("")


def _append_triage_sections(lines: list[str], data: _DigestData, limit: int) -> None:
    if not data.triage_predictions:
        return
    likely_star = _triage_bucket_papers(data, TRIAGE_BUCKET_LIKELY_STAR)
    likely_star.sort(
        key=lambda paper: data.triage_predictions[paper.arxiv_id].probability,
        reverse=True,
    )
    unsure = _triage_bucket_papers(data, TRIAGE_BUCKET_UNSURE)
    unsure.sort(key=lambda paper: abs(data.triage_predictions[paper.arxiv_id].probability - 0.5))
    _append_triage_section(lines, "Likely Star", likely_star, data, limit)
    _append_triage_section(lines, "Unsure Review Queue", unsure, data, limit)


def _triage_bucket_papers(data: _DigestData, bucket: str) -> list[Paper]:
    return [
        paper
        for paper in data.papers
        if (prediction := data.triage_predictions.get(paper.arxiv_id))
        and prediction.bucket == bucket
    ]


def _append_triage_section(
    lines: list[str],
    title: str,
    papers: list[Paper],
    data: _DigestData,
    limit: int,
) -> None:
    if not papers:
        return
    lines.extend([f"## {title}", ""])
    for paper in papers[:limit]:
        prediction = data.triage_predictions[paper.arxiv_id]
        lines.append(_render_paper_item(paper, suffix=format_triage_prediction(prediction)))
    _append_omitted_count(lines, len(papers), limit)
    lines.append("")


def _append_hf_section(lines: list[str], data: _DigestData, limit: int) -> None:
    if not data.hf_matches:
        return
    papers = [paper for paper in data.papers if paper.arxiv_id in data.hf_matches]
    papers.sort(key=lambda paper: data.hf_matches[paper.arxiv_id].upvotes, reverse=True)
    lines.extend(["## Trending on Hugging Face", ""])
    for paper in papers[:limit]:
        hf = data.hf_matches[paper.arxiv_id]
        suffix = f"{hf.upvotes} upvotes"
        if hf.num_comments:
            suffix += f", {hf.num_comments} comments"
        if hf.github_repo:
            suffix += f", GitHub: {_escape_markdown(hf.github_repo)}"
        lines.append(_render_paper_item(paper, suffix=suffix))
    _append_omitted_count(lines, len(papers), limit)
    lines.append("")


def _append_version_section(lines: list[str], data: _DigestData, limit: int) -> None:
    if not data.version_updates:
        return
    papers_by_id = {paper.arxiv_id: paper for paper in data.papers}
    updates = sorted(data.version_updates.items())
    lines.extend(["## Version Updates", ""])
    for arxiv_id, (old_version, new_version) in updates[:limit]:
        paper = papers_by_id.get(arxiv_id)
        label = _escape_markdown(paper.title) if paper else _escape_markdown(arxiv_id)
        url = _paper_url(paper) if paper else f"https://arxiv.org/abs/{arxiv_id}"
        lines.append(f"- [{label}]({url}) ({arxiv_id}) - v{old_version} -> v{new_version}")
    _append_omitted_count(lines, len(updates), limit)
    lines.append("")


def _append_omitted_count(lines: list[str], total: int, limit: int) -> None:
    if total > limit:
        lines.append(f"- ... {total - limit} more")


def _render_paper_item(paper: Paper, *, suffix: str = "") -> str:
    parts = [
        f"[{_escape_markdown(paper.title)}]({_paper_url(paper)})",
        _escape_markdown(paper.arxiv_id),
    ]
    if paper.authors:
        parts.append(_escape_markdown(paper.authors))
    if paper.date:
        parts.append(_escape_markdown(paper.date))
    if paper.categories:
        parts.append(_escape_markdown(paper.categories))
    if suffix:
        parts.append(suffix)
    return "- " + " - ".join(parts)


def _paper_url(paper: Paper | None) -> str:
    if paper is not None and paper.url:
        return paper.url
    arxiv_id = paper.arxiv_id if paper is not None else ""
    return f"https://arxiv.org/abs/{arxiv_id}"


def _escape_markdown(text: str) -> str:
    normalized = " ".join(str(text).split())
    return "".join(f"\\{char}" if char in _MARKDOWN_SPECIALS else char for char in normalized)


__all__ = [
    "DigestDependencies",
    "DigestError",
    "DigestOptions",
    "DigestResult",
    "VersionFetchResult",
    "fetch_arxiv_versions",
    "generate_digest",
    "render_digest_markdown",
]
