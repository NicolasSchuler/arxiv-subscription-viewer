"""Tests for headless Markdown digest generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from arxiv_browser import cli as cli_module
from arxiv_browser import digest as digest_module
from arxiv_browser.cli import CliDependencies, main
from arxiv_browser.digest import (
    DigestDependencies,
    DigestError,
    DigestOptions,
    DigestResult,
    DigestSection,
    DigestSectionItem,
    VersionFetchResult,
    fetch_arxiv_versions,
    generate_digest,
)
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import ArxivSearchRequest, PaperMetadata, UserConfig, WatchListEntry
from arxiv_browser.triage_model import (
    TRIAGE_BUCKET_LIKELY_STAR,
    TRIAGE_BUCKET_UNSURE,
    MissingTriageModelDependencyError,
    TriagePrediction,
)


def _clock() -> datetime:
    return datetime(2026, 5, 15, 10, 30, tzinfo=UTC)


def _deps_for(papers, **overrides) -> DigestDependencies:
    values = {
        "clock": _clock,
        "fetch_recent_papers": lambda **_kwargs: list(papers),
        "load_relevance_scores": lambda *_args: {},
        "hf_db_path_fn": lambda: Path("hf.db"),
        "load_hf_snapshot_fn": lambda *_args: SimpleNamespace(status="empty", papers={}),
        "save_hf_cache_fn": lambda *_args: None,
        "fetch_versions_fn": lambda _ids: VersionFetchResult({}),
        "save_config_fn": lambda _config: True,
    }
    values.update(overrides)
    return DigestDependencies(**values)


def _options(**overrides) -> DigestOptions:
    values = {
        "request": ArxivSearchRequest(query="agents", category="cs.AI"),
        "relevance_mode": "off",
        "hf_mode": "off",
        "versions_enabled": False,
    }
    values.update(overrides)
    return DigestOptions(**values)


def test_digest_empty_source_renders_valid_markdown(make_paper) -> None:
    del make_paper
    result = generate_digest(_options(), UserConfig(), _deps_for([]))

    assert result.paper_count == 0
    assert "Generated: 2026-05-15 10:30 UTC" in result.markdown
    assert "Source: arXiv daily search" in result.markdown
    assert "No matching papers." in result.markdown


def test_digest_defaults_and_naive_clock_branches(make_paper) -> None:
    paper = make_paper(authors="", date="", categories="", url="")
    options = DigestOptions(relevance_mode="off", hf_mode="off", versions_enabled=True)
    deps = _deps_for([paper], clock=lambda: datetime(2026, 5, 15, 10, 30))

    result = generate_digest(options, UserConfig(), deps)

    assert options.request.query == ""
    assert DigestDependencies().clock().tzinfo is not None
    assert "Generated: 2026-05-15 10:30 UTC" in result.markdown
    assert "https://arxiv.org/abs/2401.12345" in result.markdown


def test_digest_renders_sections_escapes_markdown_and_caps_limits(make_paper) -> None:
    papers = [
        make_paper(
            arxiv_id="2605.00001",
            title="A [Sharp]_Paper | One",
            authors="A. Author\nB. Writer",
        ),
        make_paper(arxiv_id="2605.00002", title="Second Paper", authors="Other Author"),
        make_paper(arxiv_id="2605.00003", title="Third Paper", authors="Third Author"),
    ]
    config = UserConfig(
        watch_list=[WatchListEntry(pattern="sharp", match_type="title")],
        hf_enabled=True,
        research_interests="agents",
        llm_provider_type="http",
        llm_api_base_url="https://llm.example/v1",
    )
    hf = HuggingFacePaper("2605.00002", "Second", 42, 3, "", (), "org/repo", 10)
    deps = _deps_for(
        papers,
        load_relevance_scores=lambda *_args: {
            "2605.00001": (9, "Strong [fit]_reason"),
            "2605.00003": (8, "Useful"),
        },
        load_hf_snapshot_fn=lambda *_args: SimpleNamespace(
            status="found", papers={hf.arxiv_id: hf}
        ),
    )
    result = generate_digest(
        _options(relevance_mode="cached", hf_mode="include", limit=1),
        config,
        deps,
    )

    markdown = result.markdown
    assert markdown.index("## Watch List Matches") < markdown.index("## High Relevance")
    assert markdown.index("## High Relevance") < markdown.index("## Trending on Hugging Face")
    assert markdown.index("## Trending on Hugging Face") < markdown.index("## New Papers")
    assert "A \\[Sharp\\]\\_Paper \\| One" in markdown
    assert "Strong \\[fit\\]\\_reason" in markdown
    assert "GitHub: org/repo" in markdown
    assert "- ... 2 more" in markdown
    assert [section.title for section in result.sections] == [
        "Watch List Matches",
        "High Relevance",
        "Trending on Hugging Face",
        "New Papers",
    ]
    assert result.sections[1].items[0].suffix == "relevance 9/10 - Strong \\[fit\\]\\_reason"


def test_digest_ascii_overview_uses_ascii_separator(make_paper) -> None:
    from arxiv_browser._ascii import set_ascii_mode

    set_ascii_mode(True)
    try:
        result = generate_digest(_options(), UserConfig(), _deps_for([make_paper()]))
    finally:
        set_ascii_mode(False)

    assert "1 papers | Top: cs.AI (1)" in result.markdown


def test_digest_input_source_parses_file(make_paper, tmp_path) -> None:
    input_path = tmp_path / "papers.txt"
    input_path.write_text("placeholder", encoding="utf-8")
    paper = make_paper(arxiv_id="2605.10001")
    deps = _deps_for([], parse_input_file=lambda path: [paper] if path == input_path else [])

    result = generate_digest(
        _options(input_path=input_path, request=ArxivSearchRequest(query="")),
        UserConfig(),
        deps,
    )

    assert result.source_label == "input papers.txt"
    assert "2605.10001" in result.markdown


def test_digest_input_source_errors(tmp_path) -> None:
    with pytest.raises(DigestError, match="not found"):
        generate_digest(_options(input_path=tmp_path / "missing.txt"), UserConfig(), _deps_for([]))

    with pytest.raises(DigestError, match="directory"):
        generate_digest(_options(input_path=tmp_path), UserConfig(), _deps_for([]))

    existing = tmp_path / "bad.txt"
    existing.write_text("x", encoding="utf-8")
    deps = _deps_for([], parse_input_file=MagicMock(side_effect=OSError("boom")))
    with pytest.raises(DigestError, match="Failed to read"):
        generate_digest(_options(input_path=existing), UserConfig(), deps)


def test_digest_live_source_errors_are_digest_errors() -> None:
    deps = _deps_for([], fetch_recent_papers=MagicMock(side_effect=ValueError("bad query")))
    with pytest.raises(DigestError, match="bad query"):
        generate_digest(_options(), UserConfig(), deps)

    deps = _deps_for([], fetch_recent_papers=MagicMock(side_effect=httpx.ConnectError("down")))
    with pytest.raises(DigestError, match="Failed to fetch"):
        generate_digest(_options(), UserConfig(), deps)


def test_digest_live_source_label_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "arxiv_browser.digest.format_query_label",
        MagicMock(side_effect=ValueError("bad")),
    )
    options = _options(request=ArxivSearchRequest(query="", category="cs.CL"))

    result = generate_digest(options, UserConfig(), _deps_for([]))

    assert "cat:cs.CL" in result.source_label


def test_relevance_scores_missing_papers_and_saves(make_paper, tmp_path) -> None:
    papers = [
        make_paper(arxiv_id="2605.00001"),
        make_paper(arxiv_id="2605.00002"),
        make_paper(arxiv_id="2605.00003"),
    ]
    calls: list[str] = []
    saved: list[tuple[str, int, str]] = []

    async def fake_score(**kwargs):
        arxiv_id = kwargs["paper"].arxiv_id
        calls.append(arxiv_id)
        if arxiv_id == "2605.00003":
            return None
        return (8, "fresh")

    deps = _deps_for(
        papers,
        load_relevance_scores=lambda *_args: {"2605.00001": (10, "cached")},
        resolve_provider_fn=lambda _config: object(),
        score_relevance_once_fn=fake_score,
        relevance_db_path_fn=lambda: tmp_path / "relevance.db",
        save_relevance_score=lambda _path, arxiv_id, _hash, score, reason: saved.append(
            (arxiv_id, score, reason)
        ),
    )
    config = UserConfig(
        research_interests="debugging agents",
        llm_provider_type="http",
        llm_api_base_url="https://llm.example/v1",
        llm_api_model="model",
    )
    result = generate_digest(_options(relevance_mode="score"), config, deps)

    assert calls == ["2605.00002", "2605.00003"]
    assert saved == [("2605.00002", 8, "fresh")]
    assert "Relevance scoring failed for 1 paper(s)." in result.warnings
    assert "2605.00001" in result.markdown
    assert "2605.00002" in result.markdown


def test_relevance_provider_none_cache_error_trusted_and_exception_paths(make_paper) -> None:
    paper = make_paper(arxiv_id="2605.00001")
    cache_error = generate_digest(
        _options(relevance_mode="score"),
        UserConfig(
            research_interests="agents",
            llm_provider_type="http",
            llm_api_base_url="https://llm.example/v1",
        ),
        _deps_for([paper], load_relevance_scores=MagicMock(side_effect=OSError("db"))),
    )
    assert "Relevance cache could not be read." in cache_error.warnings

    http_missing_base = generate_digest(
        _options(relevance_mode="score"),
        UserConfig(research_interests="agents", llm_provider_type="http"),
        _deps_for([paper]),
    )
    assert "no LLM command or HTTP provider" in http_missing_base.warnings[0]

    provider_none = generate_digest(
        _options(relevance_mode="score"),
        UserConfig(research_interests="agents", llm_preset="llm"),
        _deps_for([paper], resolve_provider_fn=lambda _config: None),
    )
    assert "provider could not be resolved" in provider_none.warnings[0]

    trusted_config = UserConfig(research_interests="agents", llm_command="custom {prompt}")
    trusted_config.trusted_llm_command_hashes = [
        digest_module._command_trust_hash("custom {prompt}")
    ]

    async def raising_score(**_kwargs):
        raise RuntimeError("boom")

    trusted = generate_digest(
        _options(relevance_mode="score"),
        trusted_config,
        _deps_for(
            [paper],
            resolve_provider_fn=lambda _config: object(),
            score_relevance_once_fn=raising_score,
        ),
    )
    assert "Relevance scoring failed for 1 paper(s)." in trusted.warnings


def test_relevance_cached_only_and_off_modes_do_not_call_provider(make_paper) -> None:
    paper = make_paper()
    provider = MagicMock(side_effect=AssertionError("provider should not resolve"))
    config = UserConfig(research_interests="systems", llm_command="custom {prompt}")
    deps = _deps_for(
        [paper],
        load_relevance_scores=lambda *_args: {paper.arxiv_id: (8, "cached")},
        resolve_provider_fn=provider,
    )

    cached = generate_digest(_options(relevance_mode="cached"), config, deps)
    assert "relevance 8/10" in cached.markdown
    provider.assert_not_called()

    no_relevance_deps = _deps_for(
        [paper],
        load_relevance_scores=MagicMock(side_effect=AssertionError("cache should not load")),
        resolve_provider_fn=provider,
    )
    off = generate_digest(_options(relevance_mode="off"), config, no_relevance_deps)
    assert "High Relevance" not in off.markdown


def test_digest_include_triage_renders_likely_star_and_unsure_sections(make_paper) -> None:
    star = make_paper(arxiv_id="star", title="Star Paper")
    unsure = make_paper(arxiv_id="unsure", title="Unsure Paper")
    deps = _deps_for(
        [star, unsure],
        load_triage_model_fn=lambda: (object(), object()),
        predict_triage_fn=lambda _papers, _model: {
            "star": TriagePrediction("star", 0.82, TRIAGE_BUCKET_LIKELY_STAR),
            "unsure": TriagePrediction("unsure", 0.46, TRIAGE_BUCKET_UNSURE),
        },
    )

    result = generate_digest(_options(include_triage=True), UserConfig(), deps)

    assert "## Likely Star" in result.markdown
    assert "Star Paper" in result.markdown
    assert "ML:★82%" in result.markdown
    assert "## Unsure Review Queue" in result.markdown
    assert "Unsure Paper" in result.markdown
    assert "## New Papers" in result.markdown
    assert [section.title for section in result.sections] == [
        "Likely Star",
        "Unsure Review Queue",
        "New Papers",
    ]


def test_digest_include_triage_warns_without_model_or_sklearn(make_paper) -> None:
    paper = make_paper()
    no_model = generate_digest(
        _options(include_triage=True),
        UserConfig(),
        _deps_for([paper], load_triage_model_fn=lambda: None),
    )

    def missing_model():
        raise MissingTriageModelDependencyError("missing")

    missing = generate_digest(
        _options(include_triage=True),
        UserConfig(),
        _deps_for([paper], load_triage_model_fn=missing_model),
    )

    assert any("no trained model" in warning for warning in no_model.warnings)
    assert "Likely Star" not in no_model.markdown
    assert any("install ML extras" in warning for warning in missing.warnings)


def test_relevance_skips_unconfigured_and_untrusted_custom_commands(make_paper) -> None:
    paper = make_paper()
    no_interests = generate_digest(
        _options(relevance_mode="score"), UserConfig(), _deps_for([paper])
    )
    assert no_interests.warnings == ["Relevance skipped: research_interests is not configured."]

    no_provider_config = UserConfig(research_interests="agents")
    no_provider = generate_digest(
        _options(relevance_mode="score"),
        no_provider_config,
        _deps_for([paper]),
    )
    assert "no LLM command or HTTP provider" in no_provider.warnings[0]

    untrusted_config = UserConfig(research_interests="agents", llm_command="custom {prompt}")
    deps = _deps_for([paper], resolve_provider_fn=MagicMock(side_effect=AssertionError))
    untrusted = generate_digest(_options(relevance_mode="score"), untrusted_config, deps)
    assert "custom LLM command is not trusted" in untrusted.warnings[0]


def test_relevance_section_omitted_when_scores_below_threshold(make_paper) -> None:
    paper = make_paper()
    config = UserConfig(
        research_interests="agents",
        llm_provider_type="http",
        llm_api_base_url="https://llm.example/v1",
    )
    result = generate_digest(
        _options(relevance_mode="cached", min_relevance=9),
        config,
        _deps_for([paper], load_relevance_scores=lambda *_args: {paper.arxiv_id: (8, "")}),
    )

    assert "High Relevance" not in result.markdown


def test_hf_cache_hit_empty_fetch_success_and_failure(make_paper) -> None:
    paper = make_paper(arxiv_id="2605.00001")
    config = UserConfig(hf_enabled=True)
    hf = HuggingFacePaper(paper.arxiv_id, "HF", 5, 0, "", (), "", 0)

    cache_hit = generate_digest(
        _options(hf_mode="config"),
        config,
        _deps_for(
            [paper],
            load_hf_snapshot_fn=lambda *_args: SimpleNamespace(
                status="found", papers={paper.arxiv_id: hf}
            ),
            fetch_hf_papers_fn=MagicMock(side_effect=AssertionError),
        ),
    )
    assert "Trending on Hugging Face" in cache_hit.markdown

    empty = generate_digest(
        _options(hf_mode="include"),
        config,
        _deps_for(
            [paper],
            load_hf_snapshot_fn=lambda *_args: SimpleNamespace(status="empty", papers={}),
            fetch_hf_papers_fn=MagicMock(side_effect=AssertionError),
        ),
    )
    assert "Trending on Hugging Face" not in empty.markdown

    saved: list[list[HuggingFacePaper]] = []
    fetched = generate_digest(
        _options(hf_mode="include"),
        config,
        _deps_for(
            [paper],
            load_hf_snapshot_fn=lambda *_args: SimpleNamespace(status="miss", papers={}),
            fetch_hf_papers_fn=lambda _timeout: ([hf], True),
            save_hf_cache_fn=lambda _path, papers: saved.append(papers),
        ),
    )
    assert saved == [[hf]]
    assert "5 upvotes" in fetched.markdown

    incomplete = generate_digest(
        _options(hf_mode="include"),
        config,
        _deps_for(
            [paper],
            load_hf_snapshot_fn=lambda *_args: SimpleNamespace(status="miss", papers={}),
            fetch_hf_papers_fn=lambda _timeout: ([], False),
        ),
    )
    assert incomplete.warnings == ["Hugging Face trending fetch failed."]

    failed = generate_digest(
        _options(hf_mode="include"),
        config,
        _deps_for(
            [paper],
            load_hf_snapshot_fn=lambda *_args: SimpleNamespace(status="miss", papers={}),
            fetch_hf_papers_fn=MagicMock(side_effect=httpx.ConnectError("down")),
        ),
    )
    assert failed.warnings == ["Hugging Face trending fetch failed."]


def test_fetch_hf_papers_sync(monkeypatch) -> None:
    expected = HuggingFacePaper("2605.00001", "HF", 1, 0, "", (), "", 0)

    async def fake_fetch(_client, timeout, include_status):
        assert timeout == 12
        assert include_status is True
        return [expected], True

    monkeypatch.setattr("arxiv_browser.digest.fetch_hf_daily_papers", fake_fetch)

    assert digest_module._fetch_hf_papers_sync(12) == ([expected], True)


def test_version_updates_starred_papers_and_persist_baselines(make_paper) -> None:
    paper = make_paper(arxiv_id="2605.00001", title="Updated Paper")
    config = UserConfig(
        paper_metadata={
            paper.arxiv_id: PaperMetadata(paper.arxiv_id, starred=True, last_checked_version=1),
            "2605.00002": PaperMetadata("2605.00002", starred=False, last_checked_version=1),
        }
    )
    result = generate_digest(
        _options(versions_enabled=True),
        config,
        _deps_for(
            [paper],
            fetch_versions_fn=lambda ids: VersionFetchResult(
                dict.fromkeys(ids, 3) | {"2605.00002": 4},
                failed_batches=1,
            ),
        ),
    )

    assert "## Version Updates" in result.markdown
    assert "v1 -> v3" in result.markdown
    assert config.paper_metadata[paper.arxiv_id].last_checked_version == 3
    assert "Version check skipped 1 batch(es)." in result.warnings


def test_version_fetch_failure_and_save_failure_warn(make_paper) -> None:
    paper = make_paper(arxiv_id="2605.00001")
    config = UserConfig(
        paper_metadata={
            paper.arxiv_id: PaperMetadata(paper.arxiv_id, starred=True, last_checked_version=1)
        }
    )
    failed_fetch = generate_digest(
        _options(versions_enabled=True),
        config,
        _deps_for([paper], fetch_versions_fn=MagicMock(side_effect=httpx.ConnectError("down"))),
    )
    assert failed_fetch.warnings == ["Version check failed."]

    save_failed = generate_digest(
        _options(versions_enabled=True),
        config,
        _deps_for(
            [paper],
            fetch_versions_fn=lambda _ids: VersionFetchResult({paper.arxiv_id: 4}),
            save_config_fn=lambda _config: False,
        ),
    )
    assert "could not be saved" in save_failed.warnings[0]

    no_versions = generate_digest(
        _options(versions_enabled=True),
        UserConfig(paper_metadata={"x": PaperMetadata("x", starred=True)}),
        _deps_for([paper], fetch_versions_fn=lambda _ids: VersionFetchResult({})),
    )
    assert no_versions.warnings == []


def test_fetch_arxiv_versions_success_and_failed_batch(monkeypatch) -> None:
    first = MagicMock()
    first.text = "<feed/>"
    first.raise_for_status = MagicMock()
    second = MagicMock()
    second.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad",
        request=httpx.Request("GET", "https://example.com"),
        response=httpx.Response(500),
    )
    get = MagicMock(side_effect=[first, second])
    monkeypatch.setattr("arxiv_browser.digest.httpx.get", get)
    monkeypatch.setattr(
        "arxiv_browser.digest.parse_arxiv_version_map",
        lambda _text: {"2605.00000": 2},
    )
    monkeypatch.setattr("arxiv_browser.digest.time.sleep", MagicMock())

    ids = {f"2605.{i:05d}" for i in range(41)}
    result = fetch_arxiv_versions(ids)

    assert result.versions == {"2605.00000": 2}
    assert result.failed_batches == 1
    assert get.call_count == 2


def test_digest_cli_stdout_stderr_and_no_tty_requirement(capsys) -> None:
    validate = MagicMock(side_effect=AssertionError("TTY check should not run"))
    deps = CliDependencies(
        load_config_fn=UserConfig,
        discover_history_files_fn=lambda _base: [],
        validate_interactive_tty_fn=validate,
    )
    generated = DigestResult("# Digest\n", ["side warning"], 1, "source")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("arxiv_browser.digest.generate_digest", lambda _options, _config: generated)
        exit_code = main(["digest", "--category", "cs.AI"], deps=deps)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "# Digest\n"
    assert captured.err == "Warning: side warning\n"
    validate.assert_not_called()


def test_digest_cli_output_file_and_conflicting_source_flags(tmp_path, capsys) -> None:
    deps = CliDependencies(load_config_fn=UserConfig, discover_history_files_fn=lambda _base: [])
    generated = DigestResult("# Digest\n", [], 1, "source")
    output = tmp_path / "nested" / "digest.md"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("arxiv_browser.digest.generate_digest", lambda _options, _config: generated)
        exit_code = main(["digest", "--category", "cs.AI", "--output", str(output)], deps=deps)

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "# Digest\n"

    input_path = tmp_path / "input.txt"
    input_path.write_text("", encoding="utf-8")
    conflict = main(
        ["digest", "--input", str(input_path), "--query", "agents"],
        deps=deps,
    )
    captured = capsys.readouterr()
    assert conflict == 2
    assert "--input cannot be combined" in captured.err


def test_digest_cli_tui_launches_app_with_inbox_context(make_paper, capsys) -> None:
    paper = make_paper(arxiv_id="2605.00001")
    run = MagicMock()
    app_factory = MagicMock(return_value=SimpleNamespace(run=run))
    deps = CliDependencies(
        load_config_fn=UserConfig,
        discover_history_files_fn=lambda _base: [],
        validate_interactive_tty_fn=lambda: True,
        app_factory=app_factory,
        app_factory_supports_options=True,
    )
    generated = DigestResult(
        "# Digest\n",
        ["side warning"],
        1,
        "source",
        papers=[paper],
        sections=[
            DigestSection(
                "High Relevance",
                (DigestSectionItem(paper.arxiv_id, "relevance 9/10"),),
            ),
            DigestSection("New Papers", (DigestSectionItem(paper.arxiv_id),)),
        ],
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("arxiv_browser.digest.generate_digest", lambda _options, _config: generated)
        exit_code = main(["digest", "--category", "cs.AI", "--tui"], deps=deps)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Warning: side warning" in captured.err
    app_factory.assert_called_once()
    assert app_factory.call_args.args[0] == [paper]
    options = app_factory.call_args.kwargs["options"]
    assert options.restore_session is False
    assert options.digest_inbox_context.source_label == "source"
    assert options.digest_inbox_context.section_labels_by_id[paper.arxiv_id] == [
        "High Relevance",
        "New Papers",
    ]
    run.assert_called_once()


def test_digest_cli_tui_rejects_output_and_non_tty(capsys) -> None:
    deps = CliDependencies(
        load_config_fn=UserConfig,
        discover_history_files_fn=lambda _base: [],
        validate_interactive_tty_fn=lambda: False,
    )

    assert main(["digest", "--category", "cs.AI", "--tui", "--output", "x.md"], deps=deps) == 2
    assert "--tui cannot be combined" in capsys.readouterr().err

    assert main(["digest", "--category", "cs.AI", "--tui"], deps=deps) == 2
    assert "requires an interactive TTY" in capsys.readouterr().err


def test_digest_cli_source_and_output_errors(tmp_path, capsys) -> None:
    deps = CliDependencies(load_config_fn=UserConfig, discover_history_files_fn=lambda _base: [])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "arxiv_browser.digest.generate_digest", MagicMock(side_effect=DigestError("bad"))
        )
        assert main(["digest", "--category", "cs.AI"], deps=deps) == 1
    assert "Error: bad" in capsys.readouterr().err

    output_dir = tmp_path / "as-dir"
    output_dir.mkdir()
    generated = DigestResult("# Digest\n", [], 1, "source")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("arxiv_browser.digest.generate_digest", lambda _options, _config: generated)
        assert main(["digest", "--category", "cs.AI", "--output", str(output_dir)], deps=deps) == 1
    assert "Failed to write digest" in capsys.readouterr().err


def test_digest_cli_validation_helpers(tmp_path) -> None:
    assert cli_module._positive_int("2") == 2
    with pytest.raises(Exception, match="expected an integer"):
        cli_module._positive_int("nope")
    with pytest.raises(Exception, match="positive"):
        cli_module._positive_int("0")
    with pytest.raises(Exception, match="1 to 10"):
        cli_module._relevance_score_int("11")

    args = SimpleNamespace(input=None)
    assert cli_module._digest_input_source_conflict(args) == ""

    args = SimpleNamespace(
        input=tmp_path / "in.txt",
        query="q",
        category="cs.AI",
        field="title",
        period="weekly",
        max_results=10,
    )
    conflict = cli_module._digest_input_source_conflict(args)
    assert "--category" in conflict
    assert "--field" in conflict
    assert "--period" in conflict
    assert "--max-results" in conflict

    assert cli_module._digest_relevance_mode(SimpleNamespace(no_relevance=True)) == "off"
    assert (
        cli_module._digest_relevance_mode(
            SimpleNamespace(no_relevance=False, cached_relevance_only=True)
        )
        == "cached"
    )


def test_digest_help_and_completions_expose_flags(capsys) -> None:
    from arxiv_browser.completions import get_completion_script

    with pytest.raises(SystemExit):
        main(["digest", "--help"])
    help_text = capsys.readouterr().out
    for flag in (
        "--period",
        "--output",
        "--tui",
        "--include-triage",
        "--include-hf",
        "--cached-relevance-only",
    ):
        assert flag in help_text

    for shell in ("bash", "zsh", "fish"):
        script = get_completion_script(shell)
        assert "digest" in script
        assert "tui" in script
        assert "include-triage" in script
        assert "cached-relevance-only" in script
