# pyright: reportAttributeAccessIssue=false, reportUndefinedVariable=false
"""Discovery, recommendation, and version-tracking mixin for ArxivBrowser."""

from __future__ import annotations

import asyncio

import httpx

from arxiv_browser.action_messages import build_actionable_error, build_actionable_warning
from arxiv_browser.browser._runtime import ARXIV_API_URL, logger
from arxiv_browser.enrichment import apply_version_updates
from arxiv_browser.modals.citations import CitationGraphScreen, RecommendationsScreen
from arxiv_browser.models import Paper
from arxiv_browser.parsing import clean_latex, parse_arxiv_version_map
from arxiv_browser.semantic_scholar import SemanticScholarPaper
from arxiv_browser.services.arxiv_api_service import ARXIV_API_TIMEOUT
from arxiv_browser.services.llm_service import LLMExecutionError as _LLMExecutionError
from arxiv_browser.similarity import (
    TfidfIndex,
    build_similarity_corpus_key,
    find_similar_papers,
)


class DiscoveryMixin:
    """Async discovery workflows for versions, similarity, and recommendations."""

    VERSION_CHECK_BATCH_SIZE = 40  # IDs per API request (URL length safe)

    async def _check_versions_async(self, arxiv_ids: set[str]) -> None:
        """Check starred papers for newer arXiv versions against the live API.
        The task captures the current dataset epoch and abandons publication if
        the user changes datasets mid-flight. That keeps version notifications,
        badge updates, and progress state from leaking across local/API mode or
        filter-scope transitions.
        """
        task_epoch = self._capture_dataset_epoch()
        try:
            client = self._http_client
            if client is None:
                return
            # Batch IDs into groups
            id_list = sorted(arxiv_ids)
            version_map: dict[str, int] = {}
            total_batches = max(1, -(-len(id_list) // self.VERSION_CHECK_BATCH_SIZE))
            for i in range(0, len(id_list), self.VERSION_CHECK_BATCH_SIZE):
                if not self._is_current_dataset_epoch(task_epoch):
                    return
                batch_num = i // self.VERSION_CHECK_BATCH_SIZE + 1
                self._version_progress = (batch_num, total_batches)
                self._update_footer()
                batch = id_list[i : i + self.VERSION_CHECK_BATCH_SIZE]
                await self._apply_arxiv_rate_limit()
                try:
                    response = await client.get(
                        ARXIV_API_URL,
                        params={
                            "id_list": ",".join(batch),
                            "max_results": len(batch) + 10,
                        },
                        headers={"User-Agent": "arxiv-subscription-viewer/1.0"},
                        timeout=ARXIV_API_TIMEOUT,
                    )
                    response.raise_for_status()
                    batch_map = parse_arxiv_version_map(response.text)
                    version_map.update(batch_map)
                except (httpx.HTTPError, ValueError, OSError):
                    logger.warning(
                        "Version check batch failed (IDs %d-%d)",
                        i,
                        i + len(batch),
                        exc_info=True,
                    )
            # Compare with stored versions
            if not self._is_current_dataset_epoch(task_epoch):
                return
            updates_found = apply_version_updates(
                version_map,
                self._config.paper_metadata,
                self._version_updates,
            )
            # Persist updated metadata
            self._save_config_or_warn("version tracking data")
            # Refresh UI
            self._mark_badges_dirty("version")
            self._get_ui_refresh_coordinator().refresh_detail_pane()
            if updates_found > 0:
                self.notify(
                    f"{updates_found} paper(s) have new versions",
                    title="Versions",
                )
            else:
                self.notify("All starred papers are up to date", title="Versions")
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Version check failed (%s): %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self.notify(
                build_actionable_error(
                    "check paper versions",
                    why="an API or network error occurred",
                    next_step="retry with V after a short delay",
                ),
                title="Versions",
                severity="error",
            )
        finally:
            if self._is_current_dataset_epoch(task_epoch):
                self._version_checking = False
                self._version_progress = None
                self._update_status_bar()

    def _version_update_for(self, arxiv_id: str) -> tuple[int, int] | None:
        """Return version update tuple if paper has an update, else None."""
        return self._version_updates.get(arxiv_id)

    async def _change_arxiv_page(self, direction: int) -> None:
        """Move to the previous or next arXiv API results page."""
        state = self._arxiv_search_state
        if not self._in_arxiv_api_mode or state is None:
            return
        if self._arxiv_api_fetch_inflight:
            self.notify("Search already in progress", title="arXiv Search")
            return
        if direction < 0 and state.start <= 0:
            self.notify("Already at first API page", title="arXiv Search")
            return
        target_start = max(0, state.start + (direction * state.max_results))
        await self._run_arxiv_search(state.request, start=target_start)

    def _show_recommendations(self, paper: Paper, source: str | None) -> None:
        """Dispatcher for local or S2 recommendations."""
        if not source:  # User cancelled the source modal
            return
        if source == "s2":
            self._track_dataset_task(self._show_s2_recommendations(paper))
        else:
            self._show_local_recommendations(paper)

    def _show_local_recommendations(self, paper: Paper) -> None:
        """Show local recommendations, building the TF-IDF index lazily if needed.
        Similarity indexing is intentionally corpus-scoped to ``self.all_papers``
        rather than the currently filtered subset. When the corpus changes, the
        previous index is treated as stale and rebuilt in the background before
        the recommendation modal is shown.
        """
        corpus_key = build_similarity_corpus_key(self.all_papers)
        tfidf_index = getattr(self, "_tfidf_index", None)
        tfidf_corpus_key = getattr(self, "_tfidf_corpus_key", None)
        if tfidf_index is None or tfidf_corpus_key != corpus_key:
            self._pending_similarity_paper_id = paper.arxiv_id
            build_task = getattr(self, "_tfidf_build_task", None)
            if build_task is not None and not build_task.done():
                self.notify("Similarity indexing in progress...", title="Similar")
                return
            self.notify("Indexing papers for similarity...", title="Similar")
            self._tfidf_build_task = self._track_dataset_task(
                self._build_tfidf_index_async(corpus_key)
            )
            return
        similar_papers = find_similar_papers(
            paper,
            self.all_papers,
            metadata=self._config.paper_metadata,
            abstract_lookup=lambda _paper: "",
            tfidf_index=tfidf_index,
        )
        if not similar_papers:
            self.notify(
                build_actionable_warning(
                    "No similar papers were found",
                    next_step="try another paper, or broaden your search with /",
                ),
                title="Similar",
                severity="warning",
            )
            return
        self.push_screen(
            RecommendationsScreen(paper, similar_papers),
            self._on_recommendation_selected,
        )

    @staticmethod
    def _build_tfidf_index_for_similarity(papers: list[Paper]) -> TfidfIndex:
        """Build a TF-IDF index using cleaned abstract text."""
        abstract_cache: dict[str, str] = {}

        def _text_for(paper: Paper) -> str:
            abstract = paper.abstract
            if abstract is None:
                abstract = abstract_cache.get(paper.arxiv_id)
                if abstract is None:
                    abstract = clean_latex(paper.abstract_raw) if paper.abstract_raw else ""
                    abstract_cache[paper.arxiv_id] = abstract
            return f"{paper.title} {abstract}"

        return TfidfIndex.build(papers, text_fn=_text_for)

    async def _build_tfidf_index_async(self, corpus_key: str) -> None:
        """Build the TF-IDF index off the UI thread and publish it only if fresh.
        The build uses a snapshot of the current paper corpus plus the captured
        dataset epoch. If either the epoch or corpus key changes before
        publication, the completed index is discarded rather than attached to a
        newer dataset.
        """
        task_epoch = self._capture_dataset_epoch()
        papers_snapshot = list(self.all_papers)
        try:
            index = await asyncio.to_thread(self._build_tfidf_index_for_similarity, papers_snapshot)
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            if not self._is_current_dataset_epoch(task_epoch):
                return
            logger.warning(
                "Failed to build similarity index (%s): %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self.notify(
                build_actionable_error(
                    "build the similarity index",
                    why="an indexing error occurred",
                    next_step="retry with R after changing paper or filter scope",
                ),
                title="Similar",
                severity="error",
            )
            return
        finally:
            if self._is_current_dataset_epoch(task_epoch):
                self._tfidf_build_task = None
        if not self._is_current_dataset_epoch(task_epoch):
            return
        if corpus_key != build_similarity_corpus_key(self.all_papers):
            logger.debug("Discarded stale similarity index for corpus key %s", corpus_key)
            return
        self._tfidf_index = index
        self._tfidf_corpus_key = corpus_key
        pending_id = self._pending_similarity_paper_id
        self._pending_similarity_paper_id = None
        if pending_id is None:
            self.notify("Similarity index ready", title="Similar")
            return
        current_paper = self._get_current_paper()
        if current_paper is None or current_paper.arxiv_id != pending_id:
            self.notify("Similarity index ready", title="Similar")
            return
        self._show_local_recommendations(current_paper)

    async def _show_s2_recommendations(self, paper: Paper) -> None:
        """Fetch S2 recommendations and show them in the modal."""
        task_epoch = self._capture_dataset_epoch()
        try:
            self.notify("Fetching S2 recommendations...", title="S2")
            recs = await self._fetch_s2_recommendations_async(paper.arxiv_id)
            if not self._is_current_dataset_epoch(task_epoch):
                return
            if not recs:
                self.notify(
                    build_actionable_warning(
                        "No Semantic Scholar recommendations were found",
                        next_step="press R and choose local recommendations, or retry later",
                    ),
                    title="S2",
                    severity="warning",
                )
                return
            similar = self._s2_recs_to_paper_tuples(recs)
            self.push_screen(
                RecommendationsScreen(paper, similar),
                self._on_recommendation_selected,
            )
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            if not self._is_current_dataset_epoch(task_epoch):
                return
            logger.warning(
                "Failed to show S2 recommendations for %s (%s): %s",
                paper.arxiv_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self.notify(
                build_actionable_error(
                    "fetch Semantic Scholar recommendations",
                    why="an API or network error occurred",
                    next_step="retry with R, or switch to local recommendations",
                ),
                title="S2",
                severity="error",
            )

    def _on_recommendation_selected(self, arxiv_id: str | None) -> None:
        """Handle selection from the recommendations modal."""
        if not arxiv_id:
            return
        option_list = self._get_paper_list_widget()
        visible_index = self._resolve_visible_index(arxiv_id)
        if visible_index is not None:
            option_list.highlighted = visible_index
            return
        self.notify(
            build_actionable_warning(
                "That paper is not in the current filtered view",
                next_step="clear or adjust the filter with /, then try again",
            ),
            title="Similar",
            severity="warning",
        )

    @staticmethod
    def _s2_recs_to_paper_tuples(
        recs: list[SemanticScholarPaper],
    ) -> list[tuple[Paper, float]]:
        """Convert S2 recommendations to (Paper, score) tuples for RecommendationsScreen."""
        max_cites = max((r.citation_count for r in recs), default=1) or 1
        results = []
        for r in recs:
            paper = Paper(
                arxiv_id=r.arxiv_id or r.s2_paper_id,
                date="",
                title=r.title or "Unknown Title",
                authors="",
                categories="",
                comments=None,
                abstract=r.abstract or r.tldr or None,
                url=r.url or (f"https://arxiv.org/abs/{r.arxiv_id}" if r.arxiv_id else ""),
                source="s2",
            )
            score = r.citation_count / max_cites
            results.append((paper, score))
        return results

    async def _show_citation_graph(self, paper_id: str, title: str) -> None:
        """Fetch citation graph data and push the CitationGraphScreen."""
        task_epoch = self._capture_dataset_epoch()
        try:
            refs, cites = await self._fetch_citation_graph(paper_id)
            if not self._is_current_dataset_epoch(task_epoch):
                return
            if not refs and not cites:
                self.notify(
                    build_actionable_warning(
                        "No citation graph data was found",
                        next_step="press G again later, or press Ctrl+e to toggle S2",
                    ),
                    title="Citations",
                    severity="warning",
                )
                return
            local_ids = frozenset(self._papers_by_id.keys())
            self.push_screen(
                CitationGraphScreen(
                    root_title=title,
                    root_paper_id=paper_id,
                    references=refs,
                    citations=cites,
                    fetch_callback=self._fetch_citation_graph,
                    local_arxiv_ids=local_ids,
                ),
                self._on_citation_graph_selected,
            )
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            if not self._is_current_dataset_epoch(task_epoch):
                return
            logger.warning(
                "Failed to show citation graph for %s (%s): %s",
                paper_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self.notify(
                build_actionable_error(
                    "load the citation graph",
                    why="an API or network error occurred",
                    next_step="retry with G after a moment",
                ),
                title="Citations",
                severity="error",
            )

    def _on_citation_graph_selected(self, arxiv_id: str | None) -> None:
        """Handle selection from the citation graph modal (jump to local paper)."""
        self._on_recommendation_selected(arxiv_id)

    async def _call_auto_tag_llm(self, paper: Paper, taxonomy: list[str]) -> list[str] | None:
        """Call the LLM to get tag suggestions for a paper. Returns tags or None on failure."""
        if self._llm_provider is None:
            logger.warning("LLM provider unexpectedly None in _call_auto_tag_llm")
            return None
        try:
            tags = await self._get_services().llm.suggest_tags_once(
                paper=paper,
                taxonomy=taxonomy,
                provider=self._llm_provider,
                timeout_seconds=max(15, self._config.llm_timeout // 4),
            )
        except _LLMExecutionError as exc:
            logger.warning("Auto-tag failed for %s: %s", paper.arxiv_id, str(exc)[:200])
            return None
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Auto-tag runtime failure for %s: %s", paper.arxiv_id, exc, exc_info=True
            )
            return None
        if tags is None:
            logger.warning("Failed to parse auto-tag response for %s", paper.arxiv_id)
            self.notify("Could not parse LLM response", title="Auto-Tag", severity="warning")
            return None
        return tags

    def _on_auto_tag_accepted(self, tags: list[str] | None, arxiv_id: str) -> None:
        """Callback when user accepts auto-tag suggestions."""
        if tags is None:
            return
        meta = self._get_or_create_metadata(arxiv_id)
        meta.tags = tags
        self._save_config_or_warn("tag changes")
        self._update_option_for_paper(arxiv_id)
        self._refresh_detail_pane()
        self.notify(f"Tags updated: {', '.join(tags)}", title="Auto-Tag")
