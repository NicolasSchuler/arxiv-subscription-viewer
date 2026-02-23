# ruff: noqa: F403, F405, UP037
# pyright: reportUndefinedVariable=false, reportAttributeAccessIssue=false
"""Extracted ArxivBrowser action handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arxiv_browser.actions._runtime import *

if TYPE_CHECKING:
    from arxiv_browser.app import ArxivBrowser


def _sync_app_globals() -> None:
    """Sync patched globals from arxiv_browser.app without importing it."""
    sync_app_globals(globals())


def _collect_all_tags(app: "ArxivBrowser") -> list[str]:
    """Collect all unique tags across all paper metadata."""
    _sync_app_globals()
    return list(
        dict.fromkeys(tag for meta in app._config.paper_metadata.values() for tag in meta.tags)
    )


def _trust_hash(command_template: str) -> str:
    """Return a stable short hash for trusted command templates."""
    _sync_app_globals()
    return hashlib.sha256(command_template.encode("utf-8")).hexdigest()[:16]


def _remember_trusted_hash(
    app,
    command_template: str,
    trusted_hashes: list[str],
    title: str,
) -> bool:
    """Persist command trust hash. Returns True when trusted in-memory."""
    _sync_app_globals()
    cmd_hash = app._trust_hash(command_template)
    if cmd_hash in trusted_hashes:
        return True
    trusted_hashes.append(cmd_hash)
    if save_config(app._config):
        return True
    app.notify(
        "Could not save trust preference. Command trusted for this session only.",
        title=title,
        severity="warning",
    )
    return True


def _is_llm_command_trusted(app: "ArxivBrowser", command_template: str) -> bool:
    """Return whether an LLM command template is trusted."""
    _sync_app_globals()
    config = app._config
    if not config.llm_command and not config.llm_preset:
        return True
    if not config.llm_command and command_template == LLM_PRESETS.get(config.llm_preset, ""):
        return True
    return app._trust_hash(command_template) in config.trusted_llm_command_hashes


def _is_pdf_viewer_trusted(app: "ArxivBrowser", viewer_cmd: str) -> bool:
    """Return whether a PDF viewer command is trusted."""
    _sync_app_globals()
    config = app._config
    return app._trust_hash(viewer_cmd) in config.trusted_pdf_viewer_hashes


def _ensure_command_trusted(
    app,
    *,
    command_template: str,
    title: str,
    prompt_heading: str,
    trust_button_label: str,
    cancel_message: str,
    trusted_hashes: list[str],
    on_trusted: Callable[[], None],
) -> bool:
    """Show trust prompt for a custom command and persist approval on confirm."""
    _sync_app_globals()
    command_preview = truncate_text(command_template, 120)

    def _on_decision(confirmed: bool | None) -> None:
        if not confirmed:
            app.notify(cancel_message, title=title, severity="warning")
            return
        if app._remember_trusted_hash(command_template, trusted_hashes, title):
            on_trusted()

    try:
        app.push_screen(
            ConfirmModal(
                f"{prompt_heading}\n"
                f"{command_preview}\n\n"
                "This command executes on your machine.\n"
                f"Confirm to trust and {trust_button_label.lower()}."
            ),
            _on_decision,
        )
        return False
    except ScreenStackError:
        logger.debug("Unable to show %s trust prompt", title, exc_info=True)
        app.notify(
            f"Could not confirm {title.lower()} command trust; action cancelled.",
            title=title,
            severity="warning",
        )
        return False


def _ensure_llm_command_trusted(
    app,
    command_template: str,
    on_trusted: Callable[[], None],
) -> bool:
    """Ensure a custom LLM command is trusted before execution."""
    _sync_app_globals()
    if app._is_llm_command_trusted(command_template):
        return True
    return app._ensure_command_trusted(
        command_template=command_template,
        title="LLM",
        prompt_heading="Run untrusted custom LLM command?",
        trust_button_label="Run",
        cancel_message="LLM command cancelled",
        trusted_hashes=app._config.trusted_llm_command_hashes,
        on_trusted=on_trusted,
    )


def _ensure_pdf_viewer_trusted(
    app,
    viewer_cmd: str,
    on_trusted: Callable[[], None],
) -> bool:
    """Ensure a custom PDF viewer command is trusted before execution."""
    _sync_app_globals()
    if app._is_pdf_viewer_trusted(viewer_cmd):
        return True
    return app._ensure_command_trusted(
        command_template=viewer_cmd,
        title="PDF",
        prompt_heading="Run untrusted custom PDF viewer command?",
        trust_button_label="Open",
        cancel_message="PDF open cancelled",
        trusted_hashes=app._config.trusted_pdf_viewer_hashes,
        on_trusted=on_trusted,
    )


def _require_llm_command(app: "ArxivBrowser") -> str | None:
    """Resolve LLM command, showing a notification if not configured.

    Also refreshes app._llm_provider so it stays in sync with config.
    Returns the command template string (needed for cache hashing).
    """
    _sync_app_globals()
    command_template = _resolve_llm_command(app._config)
    if not command_template:
        preset = app._config.llm_preset
        if preset and preset not in LLM_PRESETS:
            valid = ", ".join(sorted(LLM_PRESETS))
            msg = f"Unknown preset '{preset}'. Valid: {valid}"
        else:
            msg = f"Set llm_command or llm_preset in config.json ({get_config_path()})"
        app.notify(msg, title="LLM not configured", severity="warning", timeout=8)
        return None
    app._llm_provider = CLIProvider(command_template)
    return command_template


def action_generate_summary(app: "ArxivBrowser") -> None:
    """Generate an AI summary for the currently highlighted paper."""
    _sync_app_globals()
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_summary_flow(command_template),
    ):
        return
    app._start_summary_flow(command_template)


def _start_summary_flow(app: "ArxivBrowser", command_template: str) -> None:
    """Start the summary mode flow after command trust checks pass."""
    _sync_app_globals()

    paper = app._get_current_paper()
    if not paper:
        app.notify("No paper selected", title="AI Summary", severity="warning")
        return

    if paper.arxiv_id in app._summary_loading:
        app.notify("Summary already generating...", title="AI Summary")
        return

    app.push_screen(
        SummaryModeModal(),
        lambda mode: app._on_summary_mode_selected(mode, paper, command_template),
    )


def _on_summary_mode_selected(app, mode: str | None, paper: Paper, command_template: str) -> None:
    """Handle the mode chosen from SummaryModeModal."""
    _sync_app_globals()
    if not mode:
        return
    if mode not in SUMMARY_MODES:
        app.notify(f"Unknown summary mode: {mode}", title="AI Summary", severity="error")
        return

    arxiv_id = paper.arxiv_id
    if arxiv_id in app._summary_loading:
        return

    # Resolve prompt template for this mode
    if mode == "default" and app._config.llm_prompt_template:
        prompt_template = app._config.llm_prompt_template
    else:
        prompt_template = SUMMARY_MODES[mode][1]

    cmd_hash = _compute_command_hash(command_template, prompt_template)
    mode_label = mode.upper() if mode != "default" else ""
    use_full_paper_content = mode != "quick"

    # Check SQLite cache first
    cached = _load_summary(app._summary_db_path, arxiv_id, cmd_hash)
    if cached:
        app._paper_summaries[arxiv_id] = cached
        app._summary_mode_label[arxiv_id] = mode_label
        app._summary_command_hash[arxiv_id] = cmd_hash
        app._update_abstract_display(arxiv_id)
        app.notify("Summary loaded from cache", title="AI Summary")
        return

    # Avoid showing stale content under a newly selected mode.
    if app._summary_command_hash.get(arxiv_id) != cmd_hash:
        app._paper_summaries.pop(arxiv_id, None)
        app._summary_command_hash.pop(arxiv_id, None)

    # Start async generation
    app._summary_loading.add(arxiv_id)
    app._summary_mode_label[arxiv_id] = mode_label
    app._update_abstract_display(arxiv_id)
    app._track_task(
        app._generate_summary_async(
            paper,
            prompt_template,
            cmd_hash,
            mode_label=mode_label,
            use_full_paper_content=use_full_paper_content,
        )
    )


async def _generate_summary_async(
    app,
    paper: Paper,
    prompt_template: str,
    cmd_hash: str,
    mode_label: str = "",
    use_full_paper_content: bool = True,
) -> None:
    """Run the LLM CLI tool asynchronously and update the UI."""
    _sync_app_globals()
    arxiv_id = paper.arxiv_id
    generated_summary = False
    try:
        if app._llm_provider is None:
            logger.warning("LLM provider unexpectedly None in _generate_summary_async")
            return
        if use_full_paper_content:
            app.notify("Fetching paper content...", title="AI Summary")

        summary, error = await app._get_services().llm.generate_summary(
            paper=paper,
            prompt_template=prompt_template,
            provider=app._llm_provider,
            use_full_paper_content=use_full_paper_content,
            summary_timeout_seconds=LLM_COMMAND_TIMEOUT,
            fetch_paper_content=lambda selected_paper: _fetch_paper_content_async(
                selected_paper,
                app._http_client,
                timeout=SUMMARY_HTML_TIMEOUT,
            ),
        )
        if summary is None:
            app.notify(
                (error or "LLM command failed")[:200],
                title="AI Summary",
                severity="error",
                timeout=8,
            )
            return

        # Cache in memory and persist to SQLite
        app._paper_summaries[arxiv_id] = summary
        app._summary_mode_label[arxiv_id] = mode_label
        app._summary_command_hash[arxiv_id] = cmd_hash
        await asyncio.to_thread(_save_summary, app._summary_db_path, arxiv_id, summary, cmd_hash)
        generated_summary = True
        app.notify("Summary generated", title="AI Summary")

    except ValueError as e:
        # Config/template errors — show the descriptive message directly
        logger.warning("Summary config error for %s: %s", arxiv_id, e)
        app.notify(str(e), title="AI Summary", severity="error", timeout=10)
    except (OSError, RuntimeError) as e:
        logger.warning("Summary generation runtime failure for %s: %s", arxiv_id, e, exc_info=True)
        app.notify("Summary failed", title="AI Summary", severity="error")
    except Exception as e:
        logger.warning(
            "Unexpected summary generation failure for %s: %s", arxiv_id, e, exc_info=True
        )
        app.notify("Summary failed", title="AI Summary", severity="error")
    finally:
        app._summary_loading.discard(arxiv_id)
        if not generated_summary and arxiv_id not in app._paper_summaries:
            app._summary_mode_label.pop(arxiv_id, None)
            app._summary_command_hash.pop(arxiv_id, None)
        app._update_abstract_display(arxiv_id)


def action_chat_with_paper(app: "ArxivBrowser") -> None:
    """Open an interactive chat session about the current paper."""
    _sync_app_globals()
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_chat_with_paper(),
    ):
        return
    app._start_chat_with_paper()


def _start_chat_with_paper(app: "ArxivBrowser") -> None:
    """Start the chat flow after command trust checks pass."""
    _sync_app_globals()
    paper = app._get_current_paper()
    if not paper:
        app.notify("No paper selected", title="Chat", severity="warning")
        return
    if app._llm_provider is None:
        logger.warning("LLM provider unexpectedly None in _start_chat_with_paper")
        return
    app.notify("Fetching paper content...", title="Chat")
    app._track_task(app._open_chat_screen(paper, app._llm_provider))


async def _open_chat_screen(app: "ArxivBrowser", paper: Paper, provider: CLIProvider) -> None:
    """Fetch paper content and open the chat modal."""
    _sync_app_globals()
    paper_content = await _fetch_paper_content_async(paper, app._http_client)
    app.push_screen(PaperChatScreen(paper, provider, paper_content))


def action_score_relevance(app: "ArxivBrowser") -> None:
    """Score all loaded papers for relevance using the configured LLM."""
    _sync_app_globals()
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_score_relevance_flow(command_template),
    ):
        return
    app._start_score_relevance_flow(command_template)


def _start_score_relevance_flow(app: "ArxivBrowser", command_template: str) -> None:
    """Start relevance scoring after command trust checks pass."""
    _sync_app_globals()

    if app._relevance_scoring_active:
        app.notify("Relevance scoring already in progress", title="Relevance")
        return

    interests = app._config.research_interests
    if not interests:
        app.push_screen(
            ResearchInterestsModal(),
            lambda text: app._on_interests_saved_then_score(text, command_template),
        )
        return

    app._start_relevance_scoring(command_template, interests)


def _on_interests_saved_then_score(
    app: "ArxivBrowser", interests: str | None, command_template: str
) -> None:
    """Callback after ResearchInterestsModal: save interests then start scoring."""
    _sync_app_globals()
    if not interests:
        return
    if app._relevance_scoring_active:
        app.notify("Relevance scoring already in progress", title="Relevance")
        return
    app._config.research_interests = interests
    app._save_config_or_warn("research interests")
    app.notify("Research interests saved", title="Relevance")
    app._start_relevance_scoring(command_template, interests)


def _start_relevance_scoring(app: "ArxivBrowser", command_template: str, interests: str) -> None:
    """Begin batch relevance scoring for all loaded papers."""
    _sync_app_globals()
    if app._relevance_scoring_active:
        app.notify("Relevance scoring already in progress", title="Relevance")
        return
    app._relevance_scoring_active = True
    app._update_footer()
    papers = list(app.all_papers)
    app._track_task(app._score_relevance_batch_async(papers, command_template, interests))


def action_edit_interests(app: "ArxivBrowser") -> None:
    """Edit research interests and clear relevance cache."""
    _sync_app_globals()
    app.push_screen(
        ResearchInterestsModal(app._config.research_interests),
        app._on_interests_edited,
    )


def _on_interests_edited(app: "ArxivBrowser", interests: str | None) -> None:
    """Callback after editing interests: save and clear cache."""
    _sync_app_globals()
    if not interests or interests == app._config.research_interests:
        return
    app._config.research_interests = interests
    app._save_config_or_warn("research interests")
    app._relevance_scores.clear()
    app._mark_badges_dirty("relevance", immediate=True)
    app._refresh_detail_pane()
    if interests:
        app.notify("Research interests updated — press L to re-score", title="Relevance")
    else:
        app.notify("Research interests cleared", title="Relevance")


async def _score_relevance_batch_async(
    app,
    papers: list[Paper],
    command_template: str,
    interests: str,
) -> None:
    """Background task: batch-score papers for relevance."""
    _sync_app_globals()
    try:
        interests_hash = _compute_command_hash(command_template, interests)

        # Bulk-load existing scores from SQLite
        cached_scores = await asyncio.to_thread(
            _load_all_relevance_scores, app._relevance_db_path, interests_hash
        )

        # Populate in-memory cache with DB-cached scores
        for aid, score_data in cached_scores.items():
            app._relevance_scores[aid] = score_data

        # Refresh badges for cached papers
        app._mark_badges_dirty("relevance")
        app._refresh_detail_pane()

        # Filter to uncached papers
        uncached = [p for p in papers if p.arxiv_id not in cached_scores]

        if not uncached:
            app.notify(
                f"All {len(papers)} papers already scored",
                title="Relevance",
            )
            return

        total = len(uncached)
        scored = 0
        failed = 0

        if app._llm_provider is None:
            logger.warning("LLM provider unexpectedly None in _score_relevance_batch")
            return
        for i, paper in enumerate(uncached):
            app._scoring_progress = (i + 1, total)
            app._update_footer()

            try:
                parsed = await app._get_services().llm.score_relevance_once(
                    paper=paper,
                    interests=interests,
                    provider=app._llm_provider,
                    timeout_seconds=RELEVANCE_SCORE_TIMEOUT,
                )
                if parsed is None:
                    failed += 1
                    continue

                score, reason = parsed
                app._relevance_scores[paper.arxiv_id] = (score, reason)

                # Persist to SQLite
                await asyncio.to_thread(
                    _save_relevance_score,
                    app._relevance_db_path,
                    paper.arxiv_id,
                    interests_hash,
                    score,
                    reason,
                )

                # Update list item badge
                app._update_relevance_badge(paper.arxiv_id)
                scored += 1

            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Relevance scoring error for %s: %s",
                    paper.arxiv_id,
                    exc,
                    exc_info=True,
                )
                failed += 1
            except Exception as exc:
                logger.warning(
                    "Unexpected relevance scoring error for %s: %s",
                    paper.arxiv_id,
                    exc,
                    exc_info=True,
                )
                failed += 1

            # Progress notification every 5 papers
            done = i + 1
            if done % 5 == 0:
                app.notify(
                    f"Scoring relevance {done}/{total}...",
                    title="Relevance",
                )

        # Final notification
        msg = f"Relevance scoring complete: {scored} scored"
        if failed:
            msg += f", {failed} failed"
        cached_count = len(papers) - total
        if cached_count:
            msg += f", {cached_count} cached"
        app.notify(msg, title="Relevance")

        # Refresh display
        app._mark_badges_dirty("relevance")
        app._refresh_detail_pane()

    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Relevance batch scoring failed: %s", exc, exc_info=True)
        app.notify("Relevance scoring failed", title="Relevance", severity="error")
    except Exception as exc:
        logger.warning("Unexpected relevance batch scoring failure: %s", exc, exc_info=True)
        app.notify("Relevance scoring failed", title="Relevance", severity="error")
    finally:
        app._relevance_scoring_active = False
        app._scoring_progress = None
        app._update_footer()


def _update_relevance_badge(app: "ArxivBrowser", arxiv_id: str) -> None:
    """Update a single list item's relevance badge."""
    _sync_app_globals()
    app._update_option_for_paper(arxiv_id)


def action_auto_tag(app: "ArxivBrowser") -> None:
    """Auto-tag current or selected papers using the configured LLM."""
    _sync_app_globals()
    command_template = app._require_llm_command()
    if not command_template:
        return
    if not app._ensure_llm_command_trusted(
        command_template,
        lambda: app._start_auto_tag_flow(),
    ):
        return
    app._start_auto_tag_flow()


def _start_auto_tag_flow(app: "ArxivBrowser") -> None:
    """Start auto-tagging after command trust checks pass."""
    _sync_app_globals()

    if app._auto_tag_active:
        app.notify("Auto-tagging already in progress", title="Auto-Tag")
        return

    taxonomy = app._collect_all_tags()
    app._auto_tag_active = True

    if app.selected_ids:
        papers = [p for p in app.all_papers if p.arxiv_id in app.selected_ids]
        if not papers:
            app._auto_tag_active = False
            app.notify("No selected papers found", title="Auto-Tag", severity="warning")
            return
        app._update_footer()
        app._track_task(app._auto_tag_batch_async(papers, taxonomy))
    else:
        paper = app._get_current_paper()
        if not paper:
            app._auto_tag_active = False
            app.notify("No paper selected", title="Auto-Tag", severity="warning")
            return
        current_tags = (app._tags_for(paper.arxiv_id) or [])[:]
        app._track_task(app._auto_tag_single_async(paper, taxonomy, current_tags))


async def _auto_tag_single_async(
    app,
    paper: Paper,
    taxonomy: list[str],
    current_tags: list[str],
) -> None:
    """Auto-tag a single paper: call LLM, show suggestion modal."""
    _sync_app_globals()
    try:
        suggested = await app._call_auto_tag_llm(paper, taxonomy)
        if suggested is None:
            app.notify("Auto-tagging failed", title="Auto-Tag", severity="warning")
            return

        # Show modal for user to accept/modify
        app.push_screen(
            AutoTagSuggestModal(paper.title, suggested, current_tags),
            lambda tags: app._on_auto_tag_accepted(tags, paper.arxiv_id),
        )
    except Exception:
        logger.warning("Auto-tag single failed for %s", paper.arxiv_id, exc_info=True)
        app.notify("Auto-tagging failed", title="Auto-Tag", severity="error")
    finally:
        app._auto_tag_active = False
        app._update_footer()


async def _auto_tag_batch_async(
    app,
    papers: list[Paper],
    taxonomy: list[str],
) -> None:
    """Batch auto-tag: call LLM for each paper, apply directly."""
    _sync_app_globals()
    try:
        total = len(papers)
        tagged = 0
        failed = 0

        for i, paper in enumerate(papers):
            app._auto_tag_progress = (i + 1, total)
            app._update_footer()

            suggested = await app._call_auto_tag_llm(paper, taxonomy)
            if suggested is None:
                failed += 1
                continue

            # Apply tags directly in batch mode (merge with existing)
            meta = app._get_or_create_metadata(paper.arxiv_id)
            merged = list(dict.fromkeys(meta.tags + suggested))
            meta.tags = merged
            tagged += 1

            # Update taxonomy for subsequent papers
            for tag in suggested:
                if tag not in taxonomy:
                    taxonomy.append(tag)

        app._save_config_or_warn("auto-tag results")
        app._mark_badges_dirty("tags", immediate=True)
        app._refresh_detail_pane()

        msg = f"Auto-tagged {tagged} paper{'s' if tagged != 1 else ''}"
        if failed:
            msg += f" ({failed} failed)"
        app.notify(msg, title="Auto-Tag")

    except Exception:
        logger.error("Auto-tag batch failed after tagging %d papers", tagged, exc_info=True)
        if tagged > 0:
            app._save_config_or_warn("partial auto-tag results")
        app.notify(
            f"Auto-tagging failed ({tagged} tagged before error)"
            if tagged
            else "Auto-tagging failed",
            title="Auto-Tag",
            severity="error",
        )
    finally:
        app._auto_tag_active = False
        app._auto_tag_progress = None
        app._update_footer()
