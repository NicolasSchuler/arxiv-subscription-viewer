"""Conference deadline import, caching, and submission-target matching."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone
from datetime import date as date_type
from pathlib import Path
from typing import Any, Literal, overload
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import yaml

from arxiv_browser.http_retry import retry_with_backoff
from arxiv_browser.models import Paper, PaperMetadata

logger = logging.getLogger(__name__)

DEFAULT_CONFERENCE_DEADLINES_SOURCE_URL = (
    "https://raw.githubusercontent.com/paperswithcode/ai-deadlines/gh-pages/_data/conferences.yml"
)
CONFERENCE_DEADLINES_DB_FILENAME = "conference_deadlines.db"
CONFERENCE_DEADLINES_CACHE_TTL_HOURS = 24
CONFERENCE_DEADLINES_REQUEST_TIMEOUT = 15
CONFERENCE_DEADLINES_MAX_RETRIES = 3
CONFERENCE_DEADLINES_INITIAL_BACKOFF = 1.0
MAX_SUBMISSION_TARGETS = 3

_FETCH_CACHE_KEY = "ai-deadlines"
_OFFSET_TZ_RE = re.compile(r"^(?:UTC|GMT)([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

ARXIV_CATEGORY_SUBJECTS: dict[str, tuple[str, ...]] = {
    "cs.AI": ("ML", "KR"),
    "cs.CL": ("NLP",),
    "cs.CV": ("CV",),
    "cs.DB": ("DM",),
    "cs.HC": ("HCI",),
    "cs.IR": ("DM",),
    "cs.LG": ("ML",),
    "cs.RO": ("RO",),
    "cs.SD": ("SP",),
    "eess.AS": ("SP",),
    "stat.ML": ("ML",),
}

SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "CV": ("computer vision", "vision", "image", "video", "visual", "multimodal"),
    "DM": ("data mining", "retrieval", "ranking", "search", "recommender", "graph mining"),
    "HCI": ("hci", "human computer interaction", "human-ai", "user study", "interface"),
    "KR": ("knowledge representation", "logic", "reasoning", "planning", "symbolic"),
    "ML": (
        "machine learning",
        "learning",
        "neural",
        "deep learning",
        "reinforcement learning",
        "optimization",
        "bayesian",
    ),
    "NLP": (
        "nlp",
        "language",
        "llm",
        "large language model",
        "translation",
        "dialogue",
        "text",
        "linguistic",
    ),
    "RO": ("robotics", "robot", "manipulation", "navigation", "autonomous"),
    "SP": ("speech", "audio", "signal processing", "acoustic", "music"),
}


@dataclass(frozen=True, slots=True)
class ConferenceDeadline:
    """Normalized conference deadline imported from AI Deadlines."""

    conference_id: str
    title: str
    year: int
    subjects: tuple[str, ...]
    deadline_at: datetime
    timezone_name: str
    abstract_deadline_at: datetime | None = None
    full_name: str = ""
    link: str = ""
    place: str = ""
    event_dates: str = ""
    start: str = ""
    end: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class ConferenceDeadlineCacheSnapshot:
    """Resolved cache state for imported conference deadlines."""

    status: Literal["miss", "empty", "found"]
    deadlines: list[ConferenceDeadline]


@dataclass(frozen=True, slots=True)
class SubmissionTarget:
    """A ranked paper-to-conference match for detail-pane display."""

    deadline: ConferenceDeadline
    score: int
    matching_subjects: tuple[str, ...]
    matching_terms: tuple[str, ...]
    next_deadline_kind: str
    next_deadline_at: datetime


def parse_ai_deadlines_yaml(text: str) -> list[ConferenceDeadline]:
    """Parse AI Deadlines YAML into normalized conference deadlines."""
    if not text.strip():
        return []
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        logger.warning("AI Deadlines source returned invalid YAML", exc_info=True)
        return []
    return parse_ai_deadline_rows(raw)


def parse_ai_deadline_rows(raw: Any) -> list[ConferenceDeadline]:
    """Normalize a decoded AI Deadlines row list."""
    if not isinstance(raw, list):
        return []
    deadlines: list[ConferenceDeadline] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        deadline = parse_ai_deadline_row(row)
        if deadline is not None:
            deadlines.append(deadline)
    return deadlines


def parse_ai_deadline_row(row: dict[str, Any]) -> ConferenceDeadline | None:
    """Normalize one decoded AI Deadlines row, skipping malformed rows."""
    title = _coerce_str(row.get("title")).strip()
    year = _coerce_int(row.get("year"))
    timezone_name = _coerce_str(row.get("timezone"), "UTC").strip() or "UTC"
    deadline_at = parse_deadline_datetime(row.get("deadline"), timezone_name)
    if not title or year <= 0 or deadline_at is None:
        return None

    conference_id = _coerce_str(row.get("id")).strip()
    if not conference_id:
        conference_id = f"{normalize_venue_key(title)}{year}"

    return ConferenceDeadline(
        conference_id=conference_id,
        title=title,
        year=year,
        subjects=_normalize_subjects(row.get("sub")),
        deadline_at=deadline_at,
        timezone_name=timezone_name,
        abstract_deadline_at=parse_deadline_datetime(row.get("abstract_deadline"), timezone_name),
        full_name=_coerce_str(row.get("full_name")).strip(),
        link=_coerce_str(row.get("link")).strip(),
        place=_coerce_str(row.get("place")).strip(),
        event_dates=_coerce_str(row.get("date")).strip(),
        start=_coerce_str(row.get("start")).strip(),
        end=_coerce_str(row.get("end")).strip(),
        note=_coerce_str(row.get("note") or row.get("Note")).strip(),
    )


def parse_deadline_datetime(value: Any, timezone_name: str) -> datetime | None:
    """Parse an AI Deadlines timestamp and return an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date_type):
        parsed = datetime.combine(value, time.max)
    elif isinstance(value, str):
        parsed = _parse_datetime_string(value.strip())
    else:
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=parse_timezone(timezone_name))
    return parsed.astimezone(UTC)


def parse_timezone(value: str) -> timezone | ZoneInfo:
    """Parse common AI Deadlines timezone spellings."""
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"UTC", "GMT"}:
        return UTC
    match = _OFFSET_TZ_RE.match(cleaned)
    if match:
        sign, hours_raw, minutes_raw = match.groups()
        hours = int(hours_raw)
        minutes = int(minutes_raw or "0")
        delta = timedelta(hours=hours, minutes=minutes)
        if sign == "-":
            delta = -delta
        return timezone(delta, cleaned.upper())
    try:
        return ZoneInfo(cleaned)
    except ZoneInfoNotFoundError:
        logger.info("Unknown conference deadline timezone %r; using UTC", value)
        return UTC


def next_deadline_moment(
    deadline: ConferenceDeadline, now: datetime | None = None
) -> tuple[str, datetime] | None:
    """Return the next future deadline moment for a conference, if any."""
    current = _coerce_utc(now)
    moments = []
    if deadline.abstract_deadline_at is not None:
        moments.append(("abstract", deadline.abstract_deadline_at))
    moments.append(("paper", deadline.deadline_at))
    future = [(kind, at) for kind, at in moments if at > current]
    if not future:
        return None
    return min(future, key=lambda item: item[1])


def format_countdown(target: datetime, now: datetime | None = None) -> str:
    """Format a compact countdown for detail-pane display."""
    current = _coerce_utc(now)
    target_utc = target.astimezone(UTC)
    seconds = int((target_utc - current).total_seconds())
    if seconds <= 0:
        return "passed"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = max(1, remainder // 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_deadline_time(target: datetime) -> str:
    """Format a deadline timestamp in UTC for compact terminal display."""
    return target.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def match_paper_to_deadlines(
    paper: Paper,
    deadlines: list[ConferenceDeadline],
    metadata: PaperMetadata | None = None,
    now: datetime | None = None,
    top_n: int = MAX_SUBMISSION_TARGETS,
) -> list[SubmissionTarget]:
    """Rank future conference deadlines by deterministic topic overlap."""
    current = _coerce_utc(now)
    scored: list[SubmissionTarget] = []
    features = _paper_subject_features(paper, metadata)
    tags = tuple(metadata.tags if metadata else ())
    text_terms = _subject_terms_from_text(_paper_text(paper))
    for deadline in deadlines:
        next_moment = next_deadline_moment(deadline, current)
        if next_moment is None:
            continue
        score, subjects, terms = _score_deadline_match(deadline, features, tags, text_terms)
        if score <= 0:
            continue
        kind, at = next_moment
        scored.append(
            SubmissionTarget(
                deadline=deadline,
                score=score,
                matching_subjects=subjects,
                matching_terms=terms,
                next_deadline_kind=kind,
                next_deadline_at=at,
            )
        )
    scored.sort(key=lambda match: (-match.score, match.next_deadline_at, match.deadline.title))
    return scored[:top_n]


def get_conference_deadlines_db_path() -> Path:
    """Return the conference-deadline cache database path."""
    from arxiv_browser.database import resolve_db_path

    return resolve_db_path(CONFERENCE_DEADLINES_DB_FILENAME)


def init_conference_deadlines_db(db_path: Path) -> None:
    """Create conference deadline cache tables."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise sqlite3.OperationalError(f"Cannot create DB directory: {exc}") from exc
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conference_deadlines ("
            "  conference_id TEXT PRIMARY KEY,"
            "  payload_json TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conference_deadlines_fetch_state ("
            "  cache_key TEXT PRIMARY KEY,"
            "  source_url TEXT NOT NULL,"
            "  status TEXT NOT NULL,"
            "  fetched_at TEXT NOT NULL"
            ")"
        )


def load_conference_deadlines_cache_snapshot(
    db_path: Path,
    ttl_hours: int = CONFERENCE_DEADLINES_CACHE_TTL_HOURS,
    source_url: str = DEFAULT_CONFERENCE_DEADLINES_SOURCE_URL,
) -> ConferenceDeadlineCacheSnapshot:
    """Load a fresh cached deadline snapshot, preserving explicit empty state."""
    if not db_path.exists():
        return ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[])
    try:
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            status = _load_conference_fetch_state(conn, ttl_hours, source_url)
            if status == "empty":
                return ConferenceDeadlineCacheSnapshot(status="empty", deadlines=[])
            if status is None:
                return ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[])

            rows = conn.execute(
                "SELECT payload_json, fetched_at FROM conference_deadlines"
            ).fetchall()
            if not rows:
                return ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[])
            _, fetched_at = rows[0]
            if not _is_fresh(fetched_at, ttl_hours):
                return ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[])

            deadlines = [_json_to_deadline(payload) for payload, _ in rows]
            parsed = [deadline for deadline in deadlines if deadline is not None]
            if parsed:
                return ConferenceDeadlineCacheSnapshot(status="found", deadlines=parsed)
            return ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[])
    except sqlite3.Error:
        logger.warning("Failed to load conference deadline cache", exc_info=True)
        return ConferenceDeadlineCacheSnapshot(status="miss", deadlines=[])


def save_conference_deadlines_cache(
    db_path: Path,
    deadlines: list[ConferenceDeadline],
    source_url: str = DEFAULT_CONFERENCE_DEADLINES_SOURCE_URL,
) -> None:
    """Persist imported conference deadlines to SQLite."""
    try:
        init_conference_deadlines_db(db_path)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(str(db_path))) as conn, conn:
            conn.execute("DELETE FROM conference_deadlines")
            status = "empty"
            for deadline in deadlines:
                conn.execute(
                    "INSERT OR REPLACE INTO conference_deadlines "
                    "(conference_id, payload_json, fetched_at) VALUES (?, ?, ?)",
                    (deadline.conference_id, _deadline_to_json(deadline), now),
                )
                status = "found"
            conn.execute(
                "INSERT OR REPLACE INTO conference_deadlines_fetch_state "
                "(cache_key, source_url, status, fetched_at) VALUES (?, ?, ?, ?)",
                (_FETCH_CACHE_KEY, source_url, status, now),
            )
    except sqlite3.Error:
        logger.warning("Failed to save conference deadline cache", exc_info=True)


@overload
async def fetch_conference_deadlines(
    client: httpx.AsyncClient,
    source_url: str = ...,
    timeout: int = ...,
    include_status: Literal[False] = ...,
) -> list[ConferenceDeadline]: ...


@overload
async def fetch_conference_deadlines(
    client: httpx.AsyncClient,
    source_url: str = ...,
    timeout: int = ...,
    include_status: Literal[True] = ...,
) -> tuple[list[ConferenceDeadline], bool]: ...


async def fetch_conference_deadlines(
    client: httpx.AsyncClient,
    source_url: str = DEFAULT_CONFERENCE_DEADLINES_SOURCE_URL,
    timeout: int = CONFERENCE_DEADLINES_REQUEST_TIMEOUT,
    include_status: bool = False,
) -> list[ConferenceDeadline] | tuple[list[ConferenceDeadline], bool]:
    """Fetch and parse conference deadlines. Never raises to callers."""

    async def _do_request() -> httpx.Response:
        response = await client.get(source_url, timeout=timeout)
        response.raise_for_status()
        return response

    try:
        response = await retry_with_backoff(
            _do_request,
            max_retries=CONFERENCE_DEADLINES_MAX_RETRIES - 1,
            backoff_base=CONFERENCE_DEADLINES_INITIAL_BACKOFF,
            operation="conference deadlines",
        )
    except httpx.HTTPStatusError as exc:
        logger.warning("Conference deadline source returned %d", exc.response.status_code)
        return _deadline_fetch_failure(include_status)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
        logger.warning("Conference deadline source timeout/connection error after retries")
        return _deadline_fetch_failure(include_status)
    except httpx.HTTPError:
        logger.warning("Conference deadline source HTTP error", exc_info=True)
        return _deadline_fetch_failure(include_status)

    deadlines = parse_ai_deadlines_yaml(response.text)
    if include_status:
        return deadlines, True
    return deadlines


def normalize_venue_key(value: str) -> str:
    """Normalize a venue/tag string for direct target matching."""
    return "".join(_TOKEN_RE.findall(value.lower()))


def _parse_datetime_string(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(normalized, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            return datetime.combine(parsed.date(), time.max)
        return parsed
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _deadline_fetch_failure(
    include_status: bool,
) -> list[ConferenceDeadline] | tuple[list[ConferenceDeadline], bool]:
    return ([], False) if include_status else []


def _deadline_to_json(deadline: ConferenceDeadline) -> str:
    return json.dumps(
        {
            "conference_id": deadline.conference_id,
            "title": deadline.title,
            "year": deadline.year,
            "subjects": list(deadline.subjects),
            "deadline_at": deadline.deadline_at.isoformat(),
            "timezone_name": deadline.timezone_name,
            "abstract_deadline_at": deadline.abstract_deadline_at.isoformat()
            if deadline.abstract_deadline_at is not None
            else None,
            "full_name": deadline.full_name,
            "link": deadline.link,
            "place": deadline.place,
            "event_dates": deadline.event_dates,
            "start": deadline.start,
            "end": deadline.end,
            "note": deadline.note,
        },
        ensure_ascii=False,
    )


def _json_to_deadline(payload: str) -> ConferenceDeadline | None:
    try:
        data = json.loads(payload)
        if not isinstance(data, dict):
            return None
        deadline_at = _iso_to_utc(data.get("deadline_at"))
        if deadline_at is None:
            return None
        raw_subjects = data.get("subjects", [])
        subjects = _normalize_subjects(raw_subjects)
        return ConferenceDeadline(
            conference_id=_coerce_str(data.get("conference_id")),
            title=_coerce_str(data.get("title")),
            year=_coerce_int(data.get("year")),
            subjects=subjects,
            deadline_at=deadline_at,
            timezone_name=_coerce_str(data.get("timezone_name"), "UTC"),
            abstract_deadline_at=_iso_to_utc(data.get("abstract_deadline_at")),
            full_name=_coerce_str(data.get("full_name")),
            link=_coerce_str(data.get("link")),
            place=_coerce_str(data.get("place")),
            event_dates=_coerce_str(data.get("event_dates")),
            start=_coerce_str(data.get("start")),
            end=_coerce_str(data.get("end")),
            note=_coerce_str(data.get("note")),
        )
    except (TypeError, json.JSONDecodeError):
        logger.warning("Failed to deserialize conference deadline cache row", exc_info=True)
        return None


def _iso_to_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_conference_fetch_state(
    conn: sqlite3.Connection,
    ttl_hours: int,
    source_url: str,
) -> Literal["empty", "found"] | None:
    row = conn.execute(
        "SELECT source_url, status, fetched_at "
        "FROM conference_deadlines_fetch_state WHERE cache_key = ?",
        (_FETCH_CACHE_KEY,),
    ).fetchone()
    if row is None:
        return None
    cached_source, status, fetched_at = row
    if cached_source != source_url or status not in {"empty", "found"}:
        return None
    if not isinstance(fetched_at, str) or not _is_fresh(fetched_at, ttl_hours):
        return None
    return status


def _is_fresh(fetched_at_str: str, ttl_hours: int) -> bool:
    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        age_hours = (datetime.now(UTC) - fetched_at).total_seconds() / 3600
        return age_hours < ttl_hours
    except (TypeError, ValueError):
        return False


def _normalize_subjects(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list | tuple):
        values = [item for item in raw if isinstance(item, str)]
    else:
        values = []
    normalized = [value.strip().upper() for value in values if value.strip()]
    return tuple(dict.fromkeys(normalized))


def _paper_subject_features(
    paper: Paper,
    metadata: PaperMetadata | None,
) -> dict[str, set[str]]:
    tags = tuple(metadata.tags if metadata else ())
    return {
        "category": _subjects_from_categories(paper.categories),
        "tag": _subjects_from_tags(tags),
        "text": _subjects_from_text(_paper_text(paper)),
        "venue": _venue_keys_from_tags(tags),
    }


def _subjects_from_categories(categories: str) -> set[str]:
    subjects: set[str] = set()
    for category in categories.split():
        subjects.update(ARXIV_CATEGORY_SUBJECTS.get(category, ()))
    return subjects


def _subjects_from_tags(tags: tuple[str, ...]) -> set[str]:
    return _subjects_from_text(" ".join(_tag_value(tag) for tag in tags))


def _subjects_from_text(text: str) -> set[str]:
    haystack = f" {text.lower().replace('_', ' ').replace('-', ' ')} "
    subjects: set[str] = set()
    for subject, aliases in SUBJECT_ALIASES.items():
        if any(_alias_in_text(alias, haystack) for alias in aliases):
            subjects.add(subject)
    return subjects


def _subject_terms_from_text(text: str) -> dict[str, tuple[str, ...]]:
    haystack = f" {text.lower().replace('_', ' ').replace('-', ' ')} "
    result: dict[str, tuple[str, ...]] = {}
    for subject, aliases in SUBJECT_ALIASES.items():
        matches = tuple(alias for alias in aliases if _alias_in_text(alias, haystack))
        if matches:
            result[subject] = matches
    return result


def _alias_in_text(alias: str, haystack: str) -> bool:
    cleaned = alias.lower().replace("-", " ")
    if " " in cleaned:
        return cleaned in haystack
    return bool(re.search(rf"\b{re.escape(cleaned)}\b", haystack))


def _venue_keys_from_tags(tags: tuple[str, ...]) -> set[str]:
    keys: set[str] = set()
    for tag in tags:
        keys.add(normalize_venue_key(tag))
        keys.add(normalize_venue_key(_tag_value(tag)))
    return {key for key in keys if key}


def _score_deadline_match(
    deadline: ConferenceDeadline,
    features: dict[str, set[str]],
    tags: tuple[str, ...],
    text_terms: dict[str, tuple[str, ...]],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    deadline_subjects = set(deadline.subjects)
    category_overlap = features["category"] & deadline_subjects
    tag_overlap = features["tag"] & deadline_subjects
    text_overlap = features["text"] & deadline_subjects
    venue_match = _matches_venue(deadline, features["venue"])

    score = 0
    if venue_match:
        score += 100
    score += 30 * len(category_overlap)
    score += 25 * len(tag_overlap)
    score += 10 * len(text_overlap)

    subjects = tuple(sorted(category_overlap | tag_overlap | text_overlap))
    terms = _matching_terms(subjects, text_terms, tags, venue_match, deadline)
    return score, subjects, terms


def _matches_venue(deadline: ConferenceDeadline, venue_keys: set[str]) -> bool:
    keys = {
        normalize_venue_key(deadline.title),
        normalize_venue_key(deadline.full_name),
        normalize_venue_key(deadline.conference_id),
        normalize_venue_key(re.sub(r"\d+$", "", deadline.conference_id)),
    }
    keys.discard("")
    return any(key in venue_keys for key in keys)


def _matching_terms(
    subjects: tuple[str, ...],
    text_terms: dict[str, tuple[str, ...]],
    tags: tuple[str, ...],
    venue_match: bool,
    deadline: ConferenceDeadline,
) -> tuple[str, ...]:
    terms: list[str] = []
    if venue_match:
        terms.append(deadline.title)
    for subject in subjects:
        terms.extend(text_terms.get(subject, ())[:2])
    for tag in tags:
        tag_subjects = _subjects_from_text(_tag_value(tag))
        if tag_subjects & set(subjects):
            terms.append(tag)
    return tuple(dict.fromkeys(term for term in terms if term))


def _paper_text(paper: Paper) -> str:
    return " ".join(part for part in (paper.title, paper.abstract, paper.abstract_raw) if part)


def _tag_value(tag: str) -> str:
    return tag.split(":", 1)[1] if ":" in tag else tag


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _coerce_int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


__all__ = [
    "CONFERENCE_DEADLINES_CACHE_TTL_HOURS",
    "DEFAULT_CONFERENCE_DEADLINES_SOURCE_URL",
    "MAX_SUBMISSION_TARGETS",
    "ConferenceDeadline",
    "ConferenceDeadlineCacheSnapshot",
    "SubmissionTarget",
    "fetch_conference_deadlines",
    "format_countdown",
    "format_deadline_time",
    "get_conference_deadlines_db_path",
    "init_conference_deadlines_db",
    "load_conference_deadlines_cache_snapshot",
    "match_paper_to_deadlines",
    "next_deadline_moment",
    "normalize_venue_key",
    "parse_ai_deadline_row",
    "parse_ai_deadline_rows",
    "parse_ai_deadlines_yaml",
    "parse_deadline_datetime",
    "parse_timezone",
    "save_conference_deadlines_cache",
]
