"""Conference deadline import, matching, cache, and detail rendering tests."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing, suppress
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import arxiv_browser.conference_deadline_config as deadline_config
import arxiv_browser.conference_deadline_ui as deadline_ui
import arxiv_browser.conference_deadlines as deadlines_mod
import arxiv_browser.config as config_mod
from arxiv_browser.conference_deadlines import (
    ConferenceDeadline,
    SubmissionTarget,
    format_countdown,
    format_deadline_time,
    load_conference_deadlines_cache_snapshot,
    match_paper_to_deadlines,
    next_deadline_moment,
    parse_ai_deadlines_yaml,
    parse_deadline_datetime,
    save_conference_deadlines_cache,
)
from arxiv_browser.models import PaperMetadata, UserConfig
from arxiv_browser.services import enrichment_service
from arxiv_browser.services.enrichment_service import ConferenceDeadlinesFetchResult
from arxiv_browser.widgets.details import DetailRenderState, PaperDetails
from tests.support.app_stubs import _new_app_stub

NOW = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)


def _deadline(**kwargs) -> ConferenceDeadline:
    defaults = {
        "conference_id": "iclr27",
        "title": "ICLR",
        "year": 2027,
        "subjects": ("ML",),
        "deadline_at": datetime(2026, 9, 20, 23, 59, tzinfo=UTC),
        "timezone_name": "UTC",
        "abstract_deadline_at": datetime(2026, 9, 10, 23, 59, tzinfo=UTC),
        "link": "https://iclr.cc",
    }
    defaults.update(kwargs)
    return ConferenceDeadline(**defaults)


def test_parse_ai_deadlines_yaml_normalizes_subjects_and_timezones() -> None:
    yaml_text = """
    - title: ICLR
      year: 2027
      id: iclr27
      link: https://iclr.cc/Conferences/2027
      deadline: '2026-09-20 23:59:59'
      abstract_deadline: '2026-09-10 23:59:59'
      timezone: UTC-12
      sub: ['ML', 'NLP']
      place: Kigali, Rwanda
      date: Apr 2027
    - title: ACL
      year: 2027
      id: acl27
      deadline: '2026-05-21 12:00'
      timezone: GMT
      sub: NLP
    - title: ICRA
      year: 2027
      id: icra27
      deadline: '2026-01-10 23:00:00'
      timezone: America/Los_Angeles
      sub: RO
    - title: Broken
      year: 2027
      timezone: UTC
      sub: ML
    """

    parsed = parse_ai_deadlines_yaml(yaml_text)

    assert [item.title for item in parsed] == ["ICLR", "ACL", "ICRA"]
    assert parsed[0].subjects == ("ML", "NLP")
    assert parsed[0].deadline_at == datetime(2026, 9, 21, 11, 59, 59, tzinfo=UTC)
    assert parsed[0].abstract_deadline_at == datetime(2026, 9, 11, 11, 59, 59, tzinfo=UTC)
    assert parsed[1].subjects == ("NLP",)
    assert parsed[1].deadline_at == datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    assert parsed[2].deadline_at == datetime(2026, 1, 11, 7, 0, tzinfo=UTC)


def test_parse_deadline_datetime_handles_malformed_dates_and_date_only() -> None:
    assert parse_ai_deadlines_yaml("   ") == []
    assert parse_ai_deadlines_yaml("not: a list") == []
    assert parse_ai_deadlines_yaml("- just-a-string") == []
    assert parse_ai_deadlines_yaml("[unclosed") == []
    assert parse_deadline_datetime("not a date", "UTC") is None
    assert parse_deadline_datetime(123, "UTC") is None
    assert parse_deadline_datetime("2026-05-14", "UTC") == datetime(
        2026, 5, 14, 23, 59, 59, 999999, tzinfo=UTC
    )


def test_parse_ai_deadline_row_defaults_and_python_date_values() -> None:
    parsed = deadlines_mod.parse_ai_deadline_row(
        {
            "title": "Test Conf",
            "year": 2027,
            "deadline": date(2026, 5, 20),
            "abstract_deadline": datetime(2026, 5, 10, 12, 0, tzinfo=timezone(timedelta(hours=2))),
            "timezone": "Mars/Base",
            "sub": ("ml", "ML", "NLP"),
            "Note": "community-maintained",
        }
    )

    assert parsed is not None
    assert parsed.conference_id == "testconf2027"
    assert parsed.deadline_at == datetime(2026, 5, 20, 23, 59, 59, 999999, tzinfo=UTC)
    assert parsed.abstract_deadline_at == datetime(2026, 5, 10, 10, 0, tzinfo=UTC)
    assert parsed.subjects == ("ML", "NLP")
    assert parsed.note == "community-maintained"


def test_conference_deadline_cache_roundtrip_empty_stale_and_source_mismatch(tmp_path) -> None:
    db_path = tmp_path / "deadlines.db"
    deadline = _deadline()

    save_conference_deadlines_cache(db_path, [deadline], source_url="https://source/a.yml")
    snapshot = load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/a.yml")
    assert snapshot.status == "found"
    assert snapshot.deadlines == [deadline]
    assert load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/b.yml").status == (
        "miss"
    )

    old_time = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute("UPDATE conference_deadlines SET fetched_at = ?", (old_time,))
        conn.execute("UPDATE conference_deadlines_fetch_state SET fetched_at = ?", (old_time,))
    assert load_conference_deadlines_cache_snapshot(db_path, 1, "https://source/a.yml").status == (
        "miss"
    )

    save_conference_deadlines_cache(db_path, [], source_url="https://source/a.yml")
    empty = load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/a.yml")
    assert empty.status == "empty"
    assert empty.deadlines == []


def test_conference_deadline_cache_handles_missing_and_corrupt_rows(tmp_path) -> None:
    db_path = tmp_path / "deadlines.db"
    assert load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/a.yml").status == (
        "miss"
    )

    deadlines_mod.init_conference_deadlines_db(db_path)
    now = datetime.now(UTC).isoformat()
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO conference_deadlines_fetch_state "
            "(cache_key, source_url, status, fetched_at) VALUES (?, ?, ?, ?)",
            (deadlines_mod._FETCH_CACHE_KEY, "https://source/a.yml", "found", now),
        )
    assert load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/a.yml").status == (
        "miss"
    )

    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "INSERT INTO conference_deadlines VALUES (?, ?, ?)",
            ("bad-json", "{", now),
        )
    assert load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/a.yml").status == (
        "miss"
    )

    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "UPDATE conference_deadlines_fetch_state SET fetched_at = ?",
            ("not-a-date",),
        )
    assert load_conference_deadlines_cache_snapshot(db_path, 24, "https://source/a.yml").status == (
        "miss"
    )


@pytest.mark.asyncio
async def test_fetch_conference_deadlines_reports_source_failures() -> None:
    request = httpx.Request("GET", "https://example.test/deadlines.yml")
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(500, request=request)

    assert await deadlines_mod.fetch_conference_deadlines(client, include_status=True) == (
        [],
        False,
    )

    with patch(
        "arxiv_browser.conference_deadlines.retry_with_backoff",
        AsyncMock(side_effect=httpx.ConnectError("offline")),
    ):
        assert await deadlines_mod.fetch_conference_deadlines(client) == []

    with patch(
        "arxiv_browser.conference_deadlines.retry_with_backoff",
        AsyncMock(side_effect=httpx.HTTPError("broken")),
    ):
        assert await deadlines_mod.fetch_conference_deadlines(client, include_status=True) == (
            [],
            False,
        )


@pytest.mark.asyncio
async def test_service_load_or_fetch_conference_deadlines_handles_unavailable_source(
    tmp_path,
) -> None:
    request = httpx.Request("GET", "https://example.test/deadlines.yml")
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(404, request=request)

    result = await enrichment_service.load_or_fetch_conference_deadlines_result(
        db_path=tmp_path / "deadlines.db",
        cache_ttl_hours=24,
        client=client,
        source_url="https://example.test/deadlines.yml",
    )

    assert result.state == "unavailable"
    assert result.complete is False
    assert result.deadlines == []


@pytest.mark.asyncio
async def test_service_load_or_fetch_conference_deadlines_fetches_and_caches(tmp_path) -> None:
    yaml_text = """
    - title: NeurIPS
      year: 2027
      id: neurips27
      deadline: '2026-05-20 19:59:59'
      timezone: UTC
      sub: ML
    """
    request = httpx.Request("GET", "https://example.test/deadlines.yml")
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(200, text=yaml_text, request=request)
    db_path = tmp_path / "deadlines.db"

    result = await enrichment_service.load_or_fetch_conference_deadlines_result(
        db_path=db_path,
        cache_ttl_hours=24,
        client=client,
        source_url="https://example.test/deadlines.yml",
    )
    cached = await enrichment_service.load_or_fetch_conference_deadlines_result(
        db_path=db_path,
        cache_ttl_hours=24,
        client=client,
        source_url="https://example.test/deadlines.yml",
    )

    assert result.state == "found"
    assert result.complete is True
    assert result.from_cache is False
    assert [deadline.title for deadline in result.deadlines] == ["NeurIPS"]
    assert cached.from_cache is True
    assert client.get.await_count == 1


def test_match_paper_to_deadlines_uses_categories_tags_venue_and_future_filter(make_paper) -> None:
    nlp = _deadline(title="ACL", conference_id="acl27", subjects=("NLP",))
    cv = _deadline(title="CVPR", conference_id="cvpr27", subjects=("CV",))
    venue = _deadline(title="NeurIPS", conference_id="neurips27", subjects=("DM",))
    past = _deadline(
        title="ICML",
        conference_id="icml26",
        subjects=("ML",),
        deadline_at=datetime(2026, 1, 1, tzinfo=UTC),
        abstract_deadline_at=None,
    )
    paper = make_paper(
        categories="cs.CL",
        title="Language models for retrieval",
        abstract="A text retrieval method.",
    )
    metadata = PaperMetadata(arxiv_id=paper.arxiv_id, tags=["topic:vision", "venue:neurips"])

    matches = match_paper_to_deadlines(paper, [nlp, cv, venue, past], metadata, NOW)

    assert [match.deadline.title for match in matches] == ["NeurIPS", "ACL", "CVPR"]
    assert matches[0].score > matches[1].score
    assert "NLP" in matches[1].matching_subjects
    assert "CV" in matches[2].matching_subjects
    assert all(match.deadline.title != "ICML" for match in matches)


def test_match_paper_to_deadlines_returns_empty_without_overlap(make_paper) -> None:
    paper = make_paper(categories="math.OC", title="Convex proofs", abstract="Optimization proof.")
    deadline = _deadline(title="CVPR", conference_id="cvpr27", subjects=("CV",))

    assert match_paper_to_deadlines(paper, [deadline], None, NOW) == []


def test_match_paper_to_deadlines_sorts_equal_scores_by_soonest(make_paper) -> None:
    early = _deadline(
        title="ACL",
        conference_id="acl27",
        subjects=("NLP",),
        deadline_at=datetime(2026, 6, 1, tzinfo=UTC),
        abstract_deadline_at=None,
    )
    late = _deadline(
        title="EMNLP",
        conference_id="emnlp27",
        subjects=("NLP",),
        deadline_at=datetime(2026, 7, 1, tzinfo=UTC),
        abstract_deadline_at=None,
    )
    paper = make_paper(categories="cs.CL", title="Parsing", abstract="Language parsing.")

    matches = match_paper_to_deadlines(paper, [late, early], None, NOW)

    assert [match.deadline.title for match in matches] == ["ACL", "EMNLP"]


def test_countdown_and_deadline_time_formatting() -> None:
    assert format_countdown(NOW + timedelta(days=2, hours=3), NOW) == "2d 3h"
    assert format_countdown(NOW + timedelta(hours=4, minutes=5), NOW) == "4h 5m"
    assert format_countdown(NOW + timedelta(minutes=3), NOW) == "3m"
    assert format_countdown(NOW - timedelta(seconds=1), NOW) == "passed"
    assert format_deadline_time(NOW) == "2026-05-14 12:00 UTC"


def test_deadline_private_helpers_cover_json_time_and_path_edges(tmp_path) -> None:
    assert deadlines_mod._json_to_deadline("[]") is None
    assert deadlines_mod._json_to_deadline('{"deadline_at": "not-a-date"}') is None
    assert deadlines_mod._iso_to_utc("") is None
    assert deadlines_mod._iso_to_utc("not-a-date") is None
    assert deadlines_mod._coerce_utc(None).tzinfo is UTC
    assert deadlines_mod._coerce_utc(datetime(2026, 5, 14)) == datetime(2026, 5, 14, tzinfo=UTC)
    assert (
        next_deadline_moment(
            _deadline(
                deadline_at=NOW - timedelta(days=1),
                abstract_deadline_at=NOW - timedelta(days=2),
            ),
            NOW,
        )
        is None
    )

    payload = deadlines_mod._deadline_to_json(_deadline(deadline_at=datetime(2026, 5, 20, 12, 0)))
    restored = deadlines_mod._json_to_deadline(payload)
    assert restored is not None
    assert restored.deadline_at == datetime(2026, 5, 20, 12, 0, tzinfo=UTC)

    with patch("arxiv_browser.database.resolve_db_path", return_value=tmp_path / "deadlines.db"):
        assert deadlines_mod.get_conference_deadlines_db_path() == tmp_path / "deadlines.db"


def test_detail_pane_renders_submission_targets_and_omits_empty_section(make_paper) -> None:
    paper = make_paper(title="Efficient language modeling")
    target = SubmissionTarget(
        deadline=_deadline(title="ICLR", year=2027),
        score=30,
        matching_subjects=("ML",),
        matching_terms=("language",),
        next_deadline_kind="abstract",
        next_deadline_at=datetime(2099, 1, 1, tzinfo=UTC),
    )
    details = PaperDetails()

    details.update_state(
        DetailRenderState(
            paper=paper,
            abstract_text="abstract",
            submission_targets=(target,),
            deadline_countdown_key="209901010000",
        )
    )
    content = details.content
    assert "Submission Targets" in content
    assert "ICLR 2027" in content
    assert "abstract:" in content
    assert "https://iclr.cc" in content

    details.update_state(DetailRenderState(paper=paper, abstract_text="abstract"))
    assert "Submission Targets" not in details.content


def test_detail_pane_deadline_countdown_key_invalidates_cache(make_paper) -> None:
    paper = make_paper()
    target = SubmissionTarget(
        deadline=_deadline(),
        score=30,
        matching_subjects=("ML",),
        matching_terms=(),
        next_deadline_kind="paper",
        next_deadline_at=datetime(2099, 1, 1, tzinfo=UTC),
    )
    details = PaperDetails()
    state = DetailRenderState(
        paper=paper,
        abstract_text="abstract",
        submission_targets=(target,),
        deadline_countdown_key="minute-1",
    )

    details.update_state(state)
    details.update_state(replace(state, deadline_countdown_key="minute-2"))

    assert len(details._detail_cache) == 2


def test_conference_deadline_config_roundtrip_and_validation() -> None:
    config = UserConfig(
        conference_deadlines_enabled=True,
        conference_deadlines_source_url="https://example.test/deadlines.yml",
        conference_deadlines_cache_ttl_hours=12,
    )

    data = config_mod._config_to_dict(config)
    loaded = config_mod._dict_to_config(data)
    clamped = config_mod._dict_to_config(
        {
            "conference_deadlines_enabled": "yes",
            "conference_deadlines_source_url": "",
            "conference_deadlines_cache_ttl_hours": 999,
        }
    )

    assert data["conference_deadlines_enabled"] is True
    assert deadline_config.to_dict(config)["conference_deadlines_enabled"] is True
    assert data["conference_deadlines_source_url"] == "https://example.test/deadlines.yml"
    assert data["conference_deadlines_cache_ttl_hours"] == 12
    assert loaded.conference_deadlines_enabled is True
    assert loaded.conference_deadlines_source_url == "https://example.test/deadlines.yml"
    assert loaded.conference_deadlines_cache_ttl_hours == 12
    assert clamped.conference_deadlines_enabled is False
    assert clamped.conference_deadlines_source_url == UserConfig().conference_deadlines_source_url
    assert clamped.conference_deadlines_cache_ttl_hours == 168


def test_deadline_runtime_helpers_initialize_start_and_match(make_paper) -> None:
    app = _new_app_stub()
    app._cache_db_path = "cache.db"
    app._config.conference_deadlines_enabled = True
    app._conference_deadlines = [_deadline(subjects=("NLP",))]
    app._conference_deadlines_active = True
    paper = make_paper(categories="cs.CL", title="Language agents", abstract="LLM systems")

    deadline_ui.init_conference_deadline_state(app)
    assert app._conference_deadlines_db_path == "cache.db"
    assert app._deadline_countdown_task is None

    app._config.conference_deadlines_enabled = True
    app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
    with patch("arxiv_browser.conference_deadline_ui.ensure_deadline_countdown_timer") as ensure:
        deadline_ui.start_conference_deadlines_if_enabled(app)
    assert app._conference_deadlines_active is True
    ensure.assert_called_once_with(app)
    app._track_dataset_task.assert_called_once()

    app._conference_deadlines = [_deadline(subjects=("NLP",))]
    targets, cache_key = deadline_ui.build_detail_submission_targets(app, paper, None)
    assert targets[0].deadline.title == "ICLR"
    assert cache_key.isdigit()

    app._conference_deadlines_active = False
    assert deadline_ui.build_detail_submission_targets(app, paper, None)[0] == ()


@pytest.mark.asyncio
async def test_deadline_countdown_timer_is_singleton() -> None:
    app = _new_app_stub()
    created = MagicMock(name="created-task")
    app._track_dataset_task = MagicMock(side_effect=lambda coro: (coro.close(), created)[1])

    deadline_ui.ensure_deadline_countdown_timer(app)
    assert app._deadline_countdown_task is created

    async def _sleep_forever() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(_sleep_forever())
    app._deadline_countdown_task = task
    try:
        deadline_ui.ensure_deadline_countdown_timer(app)
        app._track_dataset_task.assert_called_once()
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def test_apply_conference_deadline_result_updates_detail_state() -> None:
    app = _new_app_stub()
    result = ConferenceDeadlinesFetchResult(
        state="found",
        deadlines=[_deadline()],
        complete=True,
        from_cache=False,
    )

    with patch("arxiv_browser.conference_deadline_ui.ensure_deadline_countdown_timer") as ensure:
        deadline_ui.apply_conference_deadlines_result(app, result)

    assert app._conference_deadlines == result.deadlines
    assert app._conference_deadlines_api_error is False
    ensure.assert_called_once_with(app)
    app._refresh_detail_pane.assert_called_once()
    assert "Conference deadlines loaded" in app.notify.call_args.args[0]


def test_apply_conference_deadline_result_handles_empty_and_unavailable() -> None:
    app = _new_app_stub()
    empty = ConferenceDeadlinesFetchResult(
        state="empty",
        deadlines=[],
        complete=True,
        from_cache=False,
    )
    unavailable = ConferenceDeadlinesFetchResult(
        state="unavailable",
        deadlines=[],
        complete=False,
        from_cache=False,
    )

    with patch("arxiv_browser.conference_deadline_ui.ensure_deadline_countdown_timer"):
        deadline_ui.apply_conference_deadlines_result(app, empty)
    assert app._conference_deadlines == []
    assert "No conference deadlines were returned" in app.notify.call_args.args[0]

    deadline_ui.apply_conference_deadlines_result(app, unavailable)
    assert app._conference_deadlines_api_error is True
    assert "refresh conference deadlines" in app.notify.call_args.args[0]

    app.notify.reset_mock()
    cached_empty = ConferenceDeadlinesFetchResult(
        state="empty",
        deadlines=[],
        complete=True,
        from_cache=True,
    )
    with patch("arxiv_browser.conference_deadline_ui.ensure_deadline_countdown_timer"):
        deadline_ui.apply_conference_deadlines_result(app, cached_empty)
    assert app.notify.call_count == 0


@pytest.mark.asyncio
async def test_refresh_conference_deadlines_action_enables_and_schedules() -> None:
    app = _new_app_stub()

    with (
        patch("arxiv_browser.conference_deadline_ui.save_config", return_value=True),
        patch("arxiv_browser.conference_deadline_ui.ensure_deadline_countdown_timer") as ensure,
        patch(
            "arxiv_browser.conference_deadline_ui.fetch_conference_deadlines", AsyncMock()
        ) as fetch,
    ):
        await deadline_ui.action_refresh_conference_deadlines(app)

    assert app._conference_deadlines_active is True
    assert app._config.conference_deadlines_enabled is True
    ensure.assert_called_once_with(app)
    fetch.assert_awaited_once_with(app)

    app._conference_deadlines_active = True
    with (
        patch("arxiv_browser.conference_deadline_ui.save_config") as save,
        patch("arxiv_browser.conference_deadline_ui.ensure_deadline_countdown_timer"),
        patch("arxiv_browser.conference_deadline_ui.fetch_conference_deadlines", AsyncMock()),
    ):
        await deadline_ui.action_refresh_conference_deadlines(app)
    save.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_conference_deadlines_action_rolls_back_when_save_fails() -> None:
    app = _new_app_stub()

    with (
        patch("arxiv_browser.conference_deadline_ui.save_config", return_value=False),
        patch(
            "arxiv_browser.conference_deadline_ui.fetch_conference_deadlines", AsyncMock()
        ) as fetch,
    ):
        await deadline_ui.action_refresh_conference_deadlines(app)

    assert app._conference_deadlines_active is False
    assert app._config.conference_deadlines_enabled is False
    fetch.assert_not_called()
    assert "Failed to save" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_fetch_conference_deadlines_handles_loading_and_schedule_errors() -> None:
    app = _new_app_stub()
    app._conference_deadlines_loading = True
    await deadline_ui.fetch_conference_deadlines(app)
    assert "already refreshing" in app.notify.call_args.args[0]

    app = _new_app_stub()

    def fail(coro) -> None:
        coro.close()
        raise OSError("cache closed")

    app._track_dataset_task = MagicMock(side_effect=fail)
    await deadline_ui.fetch_conference_deadlines(app)

    assert app._conference_deadlines_loading is False
    assert app._conference_deadlines_api_error is True
    assert "refresh conference deadlines" in app.notify.call_args.args[0]

    app = _new_app_stub()

    def fail_unexpected(coro) -> None:
        coro.close()
        raise LookupError("unexpected")

    app._track_dataset_task = MagicMock(side_effect=fail_unexpected)
    with pytest.raises(LookupError):
        await deadline_ui.fetch_conference_deadlines(app)
    assert app._conference_deadlines_loading is False


@pytest.mark.asyncio
async def test_fetch_conference_deadlines_async_applies_service_result() -> None:
    app = _new_app_stub()
    app._http_client = MagicMock()
    app._conference_deadlines_db_path = "cache.db"
    app._conference_deadlines_loading = True
    app._capture_dataset_epoch = MagicMock(return_value=3)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
    result = ConferenceDeadlinesFetchResult(
        state="found",
        deadlines=[_deadline()],
        complete=True,
        from_cache=True,
    )
    app._get_services = MagicMock(
        return_value=SimpleNamespace(
            enrichment=SimpleNamespace(
                load_or_fetch_conference_deadlines=AsyncMock(return_value=result)
            )
        )
    )

    await deadline_ui.fetch_conference_deadlines_async(app)

    assert app._conference_deadlines == result.deadlines
    assert app._conference_deadlines_loading is False
    app._refresh_detail_pane.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_conference_deadlines_async_handles_missing_client_and_errors() -> None:
    app = _new_app_stub()
    app._http_client = None
    app._capture_dataset_epoch = MagicMock(return_value=1)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    app._conference_deadlines_loading = True

    await deadline_ui.fetch_conference_deadlines_async(app)
    assert app._conference_deadlines_api_error is True
    assert app._conference_deadlines_loading is False

    app = _new_app_stub()
    app._http_client = MagicMock()
    app._capture_dataset_epoch = MagicMock(return_value=1)
    app._is_current_dataset_epoch = MagicMock(return_value=True)
    app._conference_deadlines_loading = True
    app._get_services = MagicMock(
        return_value=SimpleNamespace(
            enrichment=SimpleNamespace(
                load_or_fetch_conference_deadlines=AsyncMock(side_effect=httpx.HTTPError("boom"))
            )
        )
    )

    await deadline_ui.fetch_conference_deadlines_async(app)

    assert app._conference_deadlines_api_error is True
    assert app._conference_deadlines_loading is False
    assert "refresh conference deadlines" in app.notify.call_args.args[0]


def test_deadline_exception_and_finish_skip_stale_epochs() -> None:
    app = _new_app_stub()
    app._is_current_dataset_epoch = MagicMock(return_value=False)
    app._conference_deadlines_loading = True

    deadline_ui.handle_conference_deadlines_exception(app, 1, RuntimeError("old"))
    deadline_ui.finish_conference_deadlines_fetch(app, 1)

    app.notify.assert_not_called()
    assert app._conference_deadlines_loading is True
