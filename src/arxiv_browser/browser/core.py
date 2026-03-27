# ruff: noqa: F403, F405
# pyright: reportAssignmentType=false, reportUndefinedVariable=false
"""Core ArxivBrowser implementation and compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arxiv_browser.actions import external_io_actions as _external_io_actions
from arxiv_browser.actions import library_actions as _library_actions
from arxiv_browser.actions import llm_actions as _llm_actions
from arxiv_browser.actions import search_api_actions as _search_api_actions
from arxiv_browser.actions import ui_actions as _ui_actions
from arxiv_browser.browser._runtime import *
from arxiv_browser.browser.browse import BrowseMixin
from arxiv_browser.browser.chrome import ChromeMixin
from arxiv_browser.browser.content import (
    _fetch_paper_content_async,
)
from arxiv_browser.browser.discovery import DiscoveryMixin
from arxiv_browser.cli import (
    CliDependencies,
    _configure_color_mode,
    _configure_logging,
    _resolve_papers,
    _validate_interactive_tty,
)
from arxiv_browser.cli import main as _cli_main

logger = logging.getLogger(__name__)
# ============================================================================
# Constants
# ============================================================================
# UI Layout constants
MIN_LIST_WIDTH = 50
MAX_LIST_WIDTH = 100
CLIPBOARD_SEPARATOR = "=" * 80


@dataclass(slots=True)
class ArxivBrowserOptions:
    """Normalized constructor inputs for ``ArxivBrowser``.
    This is the forward-looking constructor shape. The browser still accepts a
    legacy positional/keyword argument form, and those calls are coerced into
    this dataclass before app initialization continues.
    """

    config: UserConfig | None = None
    restore_session: bool = True
    history_files: list[tuple[date, Path]] | None = None
    current_date_index: int = 0
    ascii_icons: bool = False
    services: AppServices | None = None


_LEGACY_BROWSER_OPTION_FIELDS = (
    "config",
    "restore_session",
    "history_files",
    "current_date_index",
    "ascii_icons",
    "services",
)


def _coerce_browser_options(
    options: Any,
    legacy_args: tuple[Any, ...],
    legacy_kwargs: dict[str, Any],
) -> ArxivBrowserOptions:
    """Normalize new-style options plus the legacy constructor calling convention.
    The compatibility goal is that existing callers can continue passing the
    older positional/keyword shape while new code can pass one options object.
    This helper rejects ambiguous mixed usage and always returns a fresh
    ``ArxivBrowserOptions`` instance for downstream initialization.
    """
    if options is not None:
        if isinstance(options, ArxivBrowserOptions):
            if legacy_args or legacy_kwargs:
                raise TypeError("ArxivBrowserOptions cannot be combined with legacy arguments")
            return ArxivBrowserOptions(
                config=options.config,
                restore_session=options.restore_session,
                history_files=list(options.history_files)
                if options.history_files is not None
                else None,
                current_date_index=options.current_date_index,
                ascii_icons=options.ascii_icons,
                services=options.services,
            )
        legacy_args = (options, *legacy_args)
    if legacy_args:
        if len(legacy_args) > len(_LEGACY_BROWSER_OPTION_FIELDS):
            raise TypeError(
                "ArxivBrowser() accepts at most "
                f"{len(_LEGACY_BROWSER_OPTION_FIELDS) + 1} positional arguments"
            )
        for field_name in _LEGACY_BROWSER_OPTION_FIELDS[: len(legacy_args)]:
            if field_name in legacy_kwargs:
                raise TypeError(f"ArxivBrowser() got multiple values for argument '{field_name}'")
        legacy_kwargs = {
            **dict(zip(_LEGACY_BROWSER_OPTION_FIELDS, legacy_args, strict=False)),
            **legacy_kwargs,
        }
    return ArxivBrowserOptions(**legacy_kwargs)


# Subprocess timeout in seconds
SUBPROCESS_TIMEOUT = 5
# Fuzzy search settings
FUZZY_SCORE_CUTOFF = 60  # Minimum score (0-100) to include in results
FUZZY_LIMIT = 100  # Maximum number of results to return
# Paper similarity settings
# UI truncation limits
BOOKMARK_NAME_MAX_LEN = 15  # Max bookmark name display length
MAX_ABSTRACT_LOADS = 32  # Maximum concurrent abstract loads
# History file discovery cap retained for custom callers/tests.
MAX_HISTORY_FILES = 365  # Compatibility constant for callers that want capped discovery.
# arXiv API search settings
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_API_MIN_INTERVAL_SECONDS = 3.0  # arXiv guidance: max 1 request / 3 seconds
ARXIV_API_TIMEOUT = 30  # Seconds to wait for arXiv API responses
PDF_DOWNLOAD_TIMEOUT = 60  # Seconds per download
MAX_CONCURRENT_DOWNLOADS = 3  # Limit parallel downloads
BATCH_CONFIRM_THRESHOLD = (
    10  # Ask for confirmation when batch operating on more than this many papers
)
# LLM summary settings
SUMMARY_DB_FILENAME = "summaries.db"
SUMMARY_HTML_TIMEOUT = 10  # Faster timeout for summary generation path
RELEVANCE_SCORE_TIMEOUT = 30  # Seconds to wait for relevance scoring LLM response
RELEVANCE_DB_FILENAME = "relevance.db"
AUTO_TAG_TIMEOUT = 30  # Seconds to wait for auto-tag LLM response
# Search debounce delay in seconds
SEARCH_DEBOUNCE_DELAY = 0.3
# Detail pane update debounce delay in seconds (shorter — must feel responsive)
DETAIL_PANE_DEBOUNCE_DELAY = 0.1
# Badge refresh coalesce delay — multiple badge sources within this window
# are merged into a single list iteration (50ms is imperceptible)
BADGE_COALESCE_DELAY = 0.05


def build_list_empty_message(
    *,
    query: str,
    in_arxiv_api_mode: bool,
    watch_filter_active: bool,
    history_mode: bool,
) -> str:
    """Build actionable empty-state copy for the paper list."""
    if query:
        return (
            "[dim italic]No papers match your search.[/]\n"
            "[dim]Try: edit the query or press [bold]Esc[/bold] to clear search.[/]\n"
            "[dim]Next: press [bold]?[/bold] for shortcuts or [bold]Ctrl+p[/bold] for commands.[/]"
        )
    if in_arxiv_api_mode:
        return (
            "[dim italic]No API results on this page.[/]\n"
            "[dim]Try: [bold]][/bold] next page, [bold][[/bold] previous page, "
            "or [bold]A[/bold] for a new query.[/]\n"
            "[dim]Next: press [bold]Esc[/bold] or [bold]Ctrl+e[/bold] to exit API mode.[/]"
        )
    if watch_filter_active:
        return (
            "[dim italic]No watched papers found.[/]\n"
            "[dim]Try: press [bold]w[/bold] to show all papers.[/]\n"
            "[dim]Next: press [bold]W[/bold] to manage watch list patterns.[/]"
        )
    if history_mode:
        return (
            "[dim italic]No papers available for this date.[/]\n"
            "[dim]Try: press [bold][[/bold] or [bold]][/bold] to change dates.[/]\n"
            "[dim]Next: press [bold]A[/bold] to search arXiv.[/]"
        )
    return (
        "[dim italic]No papers available.[/]\n"
        "[dim]Try: press [bold]A[/bold] to search arXiv.[/]\n"
        "[dim]Next: load a history file or run with [bold]-i[/bold] <file>.[/]"
    )


class ArxivBrowser(ChromeMixin, BrowseMixin, DiscoveryMixin, App):
    """A TUI application to browse arXiv papers."""

    TITLE = "arXiv Paper Browser"
    CSS = APP_CSS
    BINDINGS = APP_BINDINGS
    VERSION_CHECK_BATCH_SIZE = 40
    action_toggle_search = _search_api_actions.action_toggle_search
    action_cancel_search = _search_api_actions.action_cancel_search
    action_ctrl_e_dispatch = _ui_actions.action_ctrl_e_dispatch
    action_toggle_s2 = _ui_actions.action_toggle_s2
    action_fetch_s2 = _ui_actions.action_fetch_s2
    _fetch_s2_paper_async = _ui_actions._fetch_s2_paper_async
    action_toggle_hf = _ui_actions.action_toggle_hf
    _fetch_hf_daily = _ui_actions._fetch_hf_daily
    _fetch_hf_daily_async = _ui_actions._fetch_hf_daily_async
    action_check_versions = _ui_actions.action_check_versions
    action_exit_arxiv_search_mode = _search_api_actions.action_exit_arxiv_search_mode
    action_arxiv_search = _search_api_actions.action_arxiv_search
    _format_arxiv_search_label = _search_api_actions._format_arxiv_search_label
    _apply_arxiv_rate_limit = _search_api_actions._apply_arxiv_rate_limit
    _fetch_arxiv_api_page = _search_api_actions._fetch_arxiv_api_page
    _apply_arxiv_search_results = _search_api_actions._apply_arxiv_search_results
    _run_arxiv_search = _search_api_actions._run_arxiv_search
    action_cursor_down = _library_actions.action_cursor_down
    action_cursor_up = _library_actions.action_cursor_up
    action_toggle_select = _library_actions.action_toggle_select
    action_select_all = _library_actions.action_select_all
    action_clear_selection = _library_actions.action_clear_selection
    action_cycle_sort = _library_actions.action_cycle_sort
    action_toggle_read = _library_actions.action_toggle_read
    action_toggle_star = _library_actions.action_toggle_star
    action_edit_notes = _library_actions.action_edit_notes
    action_edit_tags = _library_actions.action_edit_tags
    _collect_all_tags = _llm_actions._collect_all_tags
    action_toggle_watch_filter = _library_actions.action_toggle_watch_filter
    action_manage_watch_list = _library_actions.action_manage_watch_list
    action_goto_bookmark = _search_api_actions.action_goto_bookmark
    action_add_bookmark = _search_api_actions.action_add_bookmark
    action_remove_bookmark = _search_api_actions.action_remove_bookmark
    action_copy_bibtex = _external_io_actions.action_copy_bibtex
    action_export_bibtex_file = _external_io_actions.action_export_bibtex_file
    action_export_markdown = _external_io_actions.action_export_markdown
    action_export_menu = _external_io_actions.action_export_menu
    _do_export = _external_io_actions._do_export
    _get_export_dir = _external_io_actions._get_export_dir
    _export_to_file = _external_io_actions._export_to_file
    _export_clipboard_ris = _external_io_actions._export_clipboard_ris
    _export_clipboard_csv = _external_io_actions._export_clipboard_csv
    _export_clipboard_mdtable = _external_io_actions._export_clipboard_mdtable
    _export_file_ris = _external_io_actions._export_file_ris
    _export_file_csv = _external_io_actions._export_file_csv
    action_export_metadata = _external_io_actions.action_export_metadata
    action_import_metadata = _external_io_actions.action_import_metadata
    action_show_similar = _ui_actions.action_show_similar
    _fetch_s2_recommendations_async = _ui_actions._fetch_s2_recommendations_async
    action_citation_graph = _ui_actions.action_citation_graph
    _fetch_citation_graph = _ui_actions._fetch_citation_graph
    action_cycle_theme = _ui_actions.action_cycle_theme
    action_toggle_sections = _ui_actions.action_toggle_sections
    action_show_help = _ui_actions.action_show_help
    action_command_palette = _ui_actions.action_command_palette
    action_collections = _ui_actions.action_collections
    action_add_to_collection = _ui_actions.action_add_to_collection
    _trust_hash = staticmethod(_llm_actions._trust_hash)
    _remember_trusted_hash = _llm_actions._remember_trusted_hash
    _is_llm_command_trusted = _llm_actions._is_llm_command_trusted
    _is_pdf_viewer_trusted = _llm_actions._is_pdf_viewer_trusted
    _ensure_command_trusted = _llm_actions._ensure_command_trusted
    _ensure_llm_command_trusted = _llm_actions._ensure_llm_command_trusted
    _ensure_pdf_viewer_trusted = _llm_actions._ensure_pdf_viewer_trusted
    _require_llm_command = _llm_actions._require_llm_command
    action_generate_summary = _llm_actions.action_generate_summary
    _start_summary_flow = _llm_actions._start_summary_flow
    _on_summary_mode_selected = _llm_actions._on_summary_mode_selected
    _generate_summary_async = _llm_actions._generate_summary_async
    action_chat_with_paper = _llm_actions.action_chat_with_paper
    _start_chat_with_paper = _llm_actions._start_chat_with_paper
    _open_chat_screen = _llm_actions._open_chat_screen
    action_score_relevance = _llm_actions.action_score_relevance
    _start_score_relevance_flow = _llm_actions._start_score_relevance_flow
    _on_interests_saved_then_score = _llm_actions._on_interests_saved_then_score
    _start_relevance_scoring = _llm_actions._start_relevance_scoring
    action_edit_interests = _llm_actions.action_edit_interests
    _on_interests_edited = _llm_actions._on_interests_edited
    _score_relevance_batch_async = _llm_actions._score_relevance_batch_async
    _update_relevance_badge = _llm_actions._update_relevance_badge
    action_auto_tag = _llm_actions.action_auto_tag
    _start_auto_tag_flow = _llm_actions._start_auto_tag_flow
    _auto_tag_single_async = _llm_actions._auto_tag_single_async
    _auto_tag_batch_async = _llm_actions._auto_tag_batch_async
    _start_downloads = _external_io_actions._start_downloads
    _process_single_download = _external_io_actions._process_single_download
    _finish_download_batch = _external_io_actions._finish_download_batch
    action_open_url = _external_io_actions.action_open_url
    _do_open_urls = _external_io_actions._do_open_urls
    action_open_pdf = _external_io_actions.action_open_pdf
    _do_open_pdfs = _external_io_actions._do_open_pdfs
    _open_with_viewer = _external_io_actions._open_with_viewer
    action_download_pdf = _external_io_actions.action_download_pdf
    _do_start_downloads = _external_io_actions._do_start_downloads
    _format_paper_for_clipboard = _external_io_actions._format_paper_for_clipboard
    _copy_to_clipboard = _external_io_actions._copy_to_clipboard
    action_copy_selected = _external_io_actions.action_copy_selected
    action_prev_date = _search_api_actions.action_prev_date
    action_next_date = _search_api_actions.action_next_date

    def __init__(
        self,
        papers: list[Paper],
        options: ArxivBrowserOptions | UserConfig | None = None,
        *legacy_args: Any,
        **legacy_kwargs: Any,
    ) -> None:
        """Initialize the app with papers, config, and optional history/service overrides."""
        super().__init__()
        resolved_options = _coerce_browser_options(options, legacy_args, legacy_kwargs)
        self._register_textual_themes()
        self._init_dataset_state(papers, resolved_options.services)
        self._init_config_state(resolved_options)
        self._init_history_and_watch_state(resolved_options)
        self._init_preview_and_search_state()
        self._init_task_and_io_state()
        self._init_llm_and_api_state()
        self._init_enrichment_and_scoring_state()
        self._configure_ascii_mode(resolved_options.ascii_icons)
        self._init_ui_runtime()

    def _register_textual_themes(self) -> None:
        """Register Textual themes before compose() resolves CSS variables."""
        for textual_theme in TEXTUAL_THEMES.values():
            self.register_theme(textual_theme)

    def _init_dataset_state(
        self,
        papers: list[Paper],
        services: AppServices | None,
    ) -> None:
        """Initialize paper collections, timers, and core dataset-local state."""
        self.all_papers = papers
        self.filtered_papers = papers.copy()
        self._papers_by_id = {p.arxiv_id: p for p in papers}
        self._visible_index_by_id = {
            paper.arxiv_id: idx for idx, paper in enumerate(self.filtered_papers)
        }
        self.selected_ids: set[str] = set()
        self._search_timer: Timer | None = None
        self._pending_query = ""
        self._applied_query = ""
        self._detail_timer: Timer | None = None
        self._pending_detail_paper: Paper | None = None
        self._pending_detail_started_at: float | None = None
        self._badges_dirty: set[str] = set()
        self._badge_timer: Timer | None = None
        self._sort_refresh_dirty: set[str] = set()
        self._sort_refresh_timer: Timer | None = None
        self._sort_index = 0
        self._services = services or build_default_app_services()
        self._shutting_down = False
        self._dataset_epoch = 0
        self._theme_runtime = build_theme_runtime("monokai")

    def _init_config_state(self, options: ArxivBrowserOptions) -> None:
        """Initialize persisted config state and theme overrides."""
        self._config = options.config or UserConfig()
        self._config.arxiv_api_max_results = _coerce_arxiv_api_max_results(
            self._config.arxiv_api_max_results
        )
        self._restore_session = options.restore_session
        self._apply_category_overrides()
        self._apply_theme_overrides()

    def _init_history_and_watch_state(self, options: ArxivBrowserOptions) -> None:
        """Initialize history navigation and watch-list derived state."""
        self._history_files = options.history_files or []
        self._current_date_index = options.current_date_index
        self._watched_paper_ids: set[str] = set()
        self._watch_filter_active = False
        self._compute_watched_papers()

    def _init_preview_and_search_state(self) -> None:
        """Initialize preview, search, and abstract-loading state."""
        self._show_abstract_preview = self._config.show_abstract_preview
        self._detail_mode = (
            self._config.detail_mode if self._config.detail_mode in DETAIL_MODES else "scan"
        )
        self._abstract_cache: dict[str, str] = {}
        self._abstract_loading: set[str] = set()
        self._abstract_queue: deque[Paper] = deque()
        self._abstract_pending_ids: set[str] = set()
        self._active_bookmark_index = -1
        self._match_scores: dict[str, float] = {}
        self._highlight_terms = {"title": [], "author": [], "abstract": []}
        self._pending_mark_action: str | None = None

    def _init_task_and_io_state(self) -> None:
        """Initialize background task tracking and download queues."""
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._dataset_tasks: set[asyncio.Task[None]] = set()
        self._download_queue: deque[Paper] = deque()
        self._downloading: set[str] = set()
        self._download_results: dict[str, bool] = {}
        self._download_total = 0

    def _init_llm_and_api_state(self) -> None:
        """Initialize LLM summary state, API browsing state, and shared clients."""
        self._paper_summaries: dict[str, str] = {}
        self._summary_loading: set[str] = set()
        self._summary_db_path = get_summary_db_path()
        self._summary_mode_label: dict[str, str] = {}
        self._summary_command_hash: dict[str, str] = {}
        self._in_arxiv_api_mode = False
        self._arxiv_search_state: ArxivSearchModeState | None = None
        self._local_browse_snapshot: LocalBrowseSnapshot | None = None
        self._arxiv_api_fetch_inflight = False
        self._arxiv_api_loading = False
        self._last_arxiv_api_request_at = 0.0
        self._arxiv_api_request_token = 0
        self._http_client: httpx.AsyncClient | None = None
        self._llm_provider = resolve_provider(self._config)

    def _init_enrichment_and_scoring_state(self) -> None:
        """Initialize enrichment caches, scoring state, and similarity state."""
        self._s2_active = False
        self._s2_cache: dict[str, SemanticScholarPaper] = {}
        self._s2_loading: set[str] = set()
        self._s2_db_path = get_s2_db_path()
        self._s2_api_error = False
        self._hf_active = False
        self._hf_cache: dict[str, HuggingFacePaper] = {}
        self._hf_loading = False
        self._hf_db_path = get_hf_db_path()
        self._hf_api_error = False
        self._version_updates: dict[str, tuple[int, int]] = {}
        self._version_checking = False
        self._version_progress: tuple[int, int] | None = None
        self._relevance_scores: dict[str, tuple[int, str]] = {}
        self._relevance_scoring_active = False
        self._scoring_progress: tuple[int, int] | None = None
        self._relevance_db_path = get_relevance_db_path()
        self._auto_tag_active = False
        self._auto_tag_progress: tuple[int, int] | None = None
        self._cancel_batch_requested = False
        self._tfidf_index: TfidfIndex | None = None
        self._tfidf_corpus_key: str | None = None
        self._tfidf_build_task: asyncio.Task[None] | None = None
        self._pending_similarity_paper_id: str | None = None

    def _configure_ascii_mode(self, ascii_icons: bool) -> None:
        """Apply ASCII/Unicode rendering mode across the UI."""
        from arxiv_browser._ascii import set_ascii_mode

        set_ascii_mode(ascii_icons)
        _widget_chrome.set_ascii_glyphs(ascii_icons)
        _widget_listing.set_ascii_icons(ascii_icons)
        _widget_details.set_ascii_glyphs(ascii_icons)

    def _init_ui_runtime(self) -> None:
        """Initialize cached widget refs and common refresh orchestration."""
        self._ui_refs = UiRefs()
        self._ui_refresh = UiRefreshCoordinator(
            refresh_list_view=self._refresh_list_view,
            update_list_header=self._update_list_header,
            update_status_bar=self._update_status_bar,
            update_filter_pills=self._update_filter_pills,
            refresh_detail_pane=self._refresh_detail_pane,
            refresh_current_list_item=self._refresh_current_list_item,
        )

    def _get_services(self) -> AppServices:
        """Return app service interfaces, lazily creating defaults for test doubles."""
        services = getattr(self, "_services", None)
        if services is None:
            services = build_default_app_services()
            self._services = services
        return services

    def _resolved_theme_runtime(self) -> ThemeRuntime:
        """Return app-owned runtime theme state, rebuilding a default when absent."""
        runtime = getattr(self, "_theme_runtime", None)
        if isinstance(runtime, ThemeRuntime):
            return runtime
        config = getattr(self, "_config", UserConfig())
        runtime = build_theme_runtime(
            config.theme_name,
            theme_overrides=config.theme,
            category_overrides=config.category_colors,
        )
        self._theme_runtime = runtime
        return runtime

    def compose(self) -> ComposeResult:
        """Build the main UI layout: header, split panes for list/detail, and footer."""
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-pane"):
                yield Label(f" Papers ({len(self.all_papers)} total)", id="list-header")
                yield DateNavigator(self._history_files, self._current_date_index)
                yield BookmarkTabBar(
                    self._config.bookmarks,
                    self._active_bookmark_index,
                    active_search=bool(self._config.session.current_filter.strip()),
                )
                yield FilterPillBar()
                with Vertical(id="search-container"):
                    yield Input(
                        placeholder=' Search papers (e.g., cat:cs.AI or "large language")',
                        id="search-input",
                    )
                    yield Static(
                        'Examples: cat:cs.AI  author:hinton  unread  "large language"  ? help  Ctrl+p commands',
                        id="search-hint",
                    )
                yield OptionList(id="paper-list")
                yield Label("", id="status-bar")
            with Vertical(id="right-pane"):
                yield Label(" Paper Details", id="details-header")
                with VerticalScroll(id="details-scroll"):
                    yield PaperDetails()
        yield ContextFooter()

    def on_mount(self) -> None:
        """Called when app is mounted. Restores session state if enabled."""
        # Create shared HTTP client for connection pooling
        self._http_client = httpx.AsyncClient()
        # Warn if config was corrupt and defaults were used
        if self._config.config_defaulted:
            self.notify(
                "Config file was corrupt and has been backed up. Using defaults.",
                severity="warning",
                timeout=8,
            )
        # Initialize S2 runtime state from config
        self._s2_active = self._config.s2_enabled
        # Initialize HF runtime state from config
        self._hf_active = self._config.hf_enabled
        if self._hf_active:
            self._track_dataset_task(self._fetch_hf_daily())
        # Initialize date navigator if in history mode
        if self._is_history_mode() and len(self._history_files) > 1:
            self.call_after_refresh(self._refresh_date_navigator)
        # Restore session state if enabled
        if self._restore_session and self._config.session:
            session = self._config.session
            self._sort_index = session.sort_index
            self.selected_ids = set(session.selected_ids)
            # Apply saved filter if any
            if session.current_filter:
                search_input = self._get_search_input_widget()
                search_input.value = session.current_filter
                self._apply_filter(session.current_filter)  # calls _refresh_list_view
            else:
                self._refresh_list_view()  # populate without filter
            # Restore scroll position
            option_list = self._get_paper_list_widget()
            if option_list.option_count > 0:
                # Clamp index to valid range
                index = min(session.scroll_index, option_list.option_count - 1)
                option_list.highlighted = max(0, index)
        else:
            # Populate list (deferred from compose for faster first paint)
            self._refresh_list_view()
            # Default: select first item if available
            option_list = self._get_paper_list_widget()
            if option_list.option_count > 0:
                option_list.highlighted = 0
        self._update_header()
        self._update_subtitle()
        self._update_details_header()
        self._track_task(self._update_bookmark_bar())
        self._notify_watch_list_matches()
        logger.debug(
            "App mounted: %d papers, history_mode=%s, s2=%s, hf=%s",
            len(self.all_papers),
            self._is_history_mode(),
            self._s2_active,
            self._hf_active,
        )
        self._prime_ui_refs()
        # Focus the paper list so key bindings work
        try:
            self._get_paper_list_widget().focus()
        except NoMatches:
            pass

    async def on_unmount(self) -> None:
        """Called when app is unmounted. Saves session state and cleans up timers.
        Uses atomic swap pattern to avoid race conditions with timer callbacks.
        """
        self._shutting_down = True
        # Clean up timers
        timer = self._search_timer
        self._search_timer = None
        if timer is not None:
            timer.stop()
        detail_timer = self._detail_timer
        self._detail_timer = None
        if detail_timer is not None:
            detail_timer.stop()
        badge_timer = self._badge_timer
        self._badge_timer = None
        if badge_timer is not None:
            badge_timer.stop()
        sort_timer = getattr(self, "_sort_refresh_timer", None)
        self._sort_refresh_timer = None
        if sort_timer is not None:
            sort_timer.stop()
        # Save after timers are cancelled so a pending debounce cannot overwrite
        # the last applied filter during teardown.
        self._save_session_state()
        # Cancel tracked background tasks to avoid leaks during teardown.
        self._cancel_dataset_tasks()
        background_tasks = getattr(self, "_background_tasks", set())
        pending = [task for task in background_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=0.5)
            for task in still_pending:
                logger.debug("Background task did not cancel before shutdown: %r", task)
        if hasattr(self, "_background_tasks"):
            self._background_tasks.clear()
        self._tfidf_build_task = None
        # Close shared HTTP client
        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception as e:
                logger.debug(
                    "Failed to close shared HTTP client during shutdown: %s", e, exc_info=True
                )
        ui_refs = getattr(self, "_ui_refs", None)
        if ui_refs is not None:
            ui_refs.reset()

    @staticmethod
    def _is_live_widget(widget: Any) -> bool:
        """Return True for mounted/attached widgets safe to reuse."""
        return bool(widget is not None and getattr(widget, "is_attached", False))

    def _get_cached_widget(self, ref_name: str, resolver: Callable[[], Any]) -> Any:
        """Resolve and cache a widget reference by UiRefs attribute name."""
        ui_refs = getattr(self, "_ui_refs", None)
        if ui_refs is None:
            return resolver()
        widget = getattr(ui_refs, ref_name)
        if self._is_live_widget(widget):
            return widget
        widget = resolver()
        setattr(ui_refs, ref_name, widget)
        return widget

    def _get_search_input_widget(self) -> Input:
        """Return the cached search input widget."""
        return self._get_cached_widget(
            "search_input", lambda: self.query_one("#search-input", Input)
        )

    def _get_search_container_widget(self) -> Any:
        """Return the cached search container widget."""
        return self._get_cached_widget(
            "search_container", lambda: self.query_one("#search-container")
        )

    def _get_paper_list_widget(self) -> OptionList:
        """Return the cached paper list OptionList widget."""
        return self._get_cached_widget(
            "paper_list", lambda: self.query_one("#paper-list", OptionList)
        )

    def _get_list_header_widget(self) -> Label:
        """Return the cached left-pane header label."""
        return self._get_cached_widget("list_header", lambda: self.query_one("#list-header", Label))

    def _get_details_header_widget(self) -> Label:
        """Return the cached right-pane header label."""
        return self._get_cached_widget(
            "details_header", lambda: self.query_one("#details-header", Label)
        )

    def _get_status_bar_widget(self) -> Label:
        """Return the cached status bar label."""
        return self._get_cached_widget("status_bar", lambda: self.query_one("#status-bar", Label))

    def _get_footer_widget(self) -> ContextFooter:
        """Return the cached context-sensitive footer widget."""
        return self._get_cached_widget("footer", lambda: self.query_one(ContextFooter))

    def _get_date_navigator_widget(self) -> DateNavigator:
        """Return the cached date navigator widget."""
        return self._get_cached_widget("date_navigator", lambda: self.query_one(DateNavigator))

    def _get_filter_pill_bar_widget(self) -> FilterPillBar:
        """Return the cached filter pill bar widget."""
        return self._get_cached_widget("filter_pill_bar", lambda: self.query_one(FilterPillBar))

    def _get_bookmark_bar_widget(self) -> BookmarkTabBar:
        """Return the cached bookmark tab bar widget."""
        return self._get_cached_widget("bookmark_bar", lambda: self.query_one(BookmarkTabBar))

    def _get_paper_details_widget(self) -> PaperDetails:
        """Return the cached paper details widget."""
        return self._get_cached_widget("paper_details", lambda: self.query_one(PaperDetails))

    def _prime_ui_refs(self) -> None:
        """Warm caches for frequently queried widgets once the DOM is mounted."""
        for ref_name, getter in (
            ("search_input", self._get_search_input_widget),
            ("search_container", self._get_search_container_widget),
            ("paper_list", self._get_paper_list_widget),
            ("list_header", self._get_list_header_widget),
            ("details_header", self._get_details_header_widget),
            ("status_bar", self._get_status_bar_widget),
            ("footer", self._get_footer_widget),
            ("date_navigator", self._get_date_navigator_widget),
            ("filter_pill_bar", self._get_filter_pill_bar_widget),
            ("bookmark_bar", self._get_bookmark_bar_widget),
            ("paper_details", self._get_paper_details_widget),
        ):
            try:
                getattr(self._ui_refs, ref_name)
                getter()
            except NoMatches:
                continue

    def _get_ui_refresh_coordinator(self) -> UiRefreshCoordinator:
        """Return the UI refresh coordinator, lazily creating it for __new__ tests."""
        coordinator = getattr(self, "_ui_refresh", None)
        if coordinator is None:
            coordinator = UiRefreshCoordinator(
                refresh_list_view=self._refresh_list_view,
                update_list_header=self._update_list_header,
                update_status_bar=self._update_status_bar,
                update_filter_pills=self._update_filter_pills,
                refresh_detail_pane=self._refresh_detail_pane,
                refresh_current_list_item=self._refresh_current_list_item,
            )
            self._ui_refresh = coordinator
        return coordinator

    def _refresh_date_navigator(self) -> None:
        """Refresh date navigator labels after DOM updates settle."""
        try:
            date_nav = self._get_date_navigator_widget()
        except NoMatches:
            return
        self._track_task(date_nav.update_dates(self._history_files, self._current_date_index))

    def _capture_dataset_epoch(self) -> int:
        """Capture the current dataset epoch for stale-task guards."""
        return getattr(self, "_dataset_epoch", 0)

    def _is_current_dataset_epoch(self, epoch: int) -> bool:
        """Return whether a task epoch still matches the live dataset."""
        return not getattr(self, "_shutting_down", False) and epoch == getattr(
            self, "_dataset_epoch", 0
        )

    def _advance_dataset_epoch(self) -> int:
        """Invalidate dataset-bound async work and return the new epoch."""
        self._dataset_epoch = getattr(self, "_dataset_epoch", 0) + 1
        self._cancel_dataset_tasks()
        return self._dataset_epoch

    def _cancel_dataset_tasks(self) -> None:
        """Cancel in-flight async work whose results belong to the prior dataset."""
        dataset_tasks = list(getattr(self, "_dataset_tasks", set()))
        for task in dataset_tasks:
            if not task.done():
                task.cancel()
        if hasattr(self, "_dataset_tasks"):
            self._dataset_tasks.clear()

    def _track_task(self, coro: Any, *, dataset_bound: bool = False) -> asyncio.Task[None]:
        """Create an asyncio task and track it to prevent garbage collection."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        if dataset_bound:
            self._dataset_tasks.add(task)
            task.add_done_callback(self._dataset_tasks.discard)
        task.add_done_callback(self._on_task_done)
        return task

    def _track_dataset_task(self, coro: Any) -> asyncio.Task[None]:
        """Track background work that must be cancelled on dataset swaps."""
        tracker = self._track_task
        if getattr(tracker, "__func__", None) is ArxivBrowser._track_task:
            return tracker(coro, dataset_bound=True)
        task = tracker(coro)
        if isinstance(task, asyncio.Task):
            dataset_tasks = getattr(self, "_dataset_tasks", None)
            if dataset_tasks is not None:
                dataset_tasks.add(task)
                task.add_done_callback(dataset_tasks.discard)
        return task

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Log unhandled exceptions from background tasks and notify user."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Unhandled exception in background task: %s", exc, exc_info=exc)
            if getattr(self, "_shutting_down", False):
                return
            try:
                self.notify(str(exc)[:200], severity="error", title="Background task error")
            except Exception:
                pass  # App may be shutting down  # nosec B110

    def _s2_state_for(self, arxiv_id: str) -> tuple[SemanticScholarPaper | None, bool]:
        """Return (s2_data, s2_loading) for a paper, respecting the active toggle."""
        if not self._s2_active:
            return None, False
        return self._s2_cache.get(arxiv_id), arxiv_id in self._s2_loading

    def _hf_state_for(self, arxiv_id: str) -> HuggingFacePaper | None:
        """Return HF data for a paper if HF is active, else None."""
        if not self._hf_active:
            return None
        return self._hf_cache.get(arxiv_id)

    @on(OptionList.OptionSelected, "#paper-list")
    def on_paper_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle paper selection (Enter key)."""
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            paper = self.filtered_papers[idx]
            details = self._get_paper_details_widget()
            aid = paper.arxiv_id
            abstract_text = self._get_abstract_text(paper, allow_async=True)
            details.update_state(self._build_detail_state(aid, paper, abstract_text))

    @on(OptionList.OptionHighlighted, "#paper-list")
    def on_paper_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Handle paper highlight (keyboard navigation) with debouncing."""
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            self._pending_detail_paper = self.filtered_papers[idx]
            self._pending_detail_started_at = (
                time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
            )
            # Atomic swap pattern (same as search debounce)
            old_timer = self._detail_timer
            self._detail_timer = None
            if old_timer is not None:
                old_timer.stop()
            self._detail_timer = self.set_timer(
                DETAIL_PANE_DEBOUNCE_DELAY,
                self._debounced_detail_update,
            )

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        self._apply_filter(event.value)
        # Hide search after submission
        self._get_search_container_widget().remove_class("visible")
        # Focus the list
        self._get_paper_list_widget().focus()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Handle search input change with debouncing.
        Uses atomic swap pattern to avoid race conditions with timer callbacks.
        """
        self._pending_query = event.value
        # Atomic swap pattern: capture and clear before stopping
        old_timer = self._search_timer
        self._search_timer = None
        if old_timer is not None:
            old_timer.stop()
        # Set new timer for debounced filter
        self._search_timer = self.set_timer(
            SEARCH_DEBOUNCE_DELAY,
            self._debounced_filter,
        )

    @on(DateNavigator.JumpToDate)
    def on_date_jump(self, event: DateNavigator.JumpToDate) -> None:
        """Handle click on a date in the navigator."""
        if 0 <= event.index < len(self._history_files):
            self._set_history_index(event.index)

    @on(DateNavigator.NavigateDate)
    def on_date_navigate(self, event: DateNavigator.NavigateDate) -> None:
        """Handle click on prev/next arrows in the date navigator."""
        if event.direction > 0:
            self.action_prev_date()
        else:
            self.action_next_date()

    @on(FilterPillBar.RemoveFilter)
    def on_remove_filter(self, event: FilterPillBar.RemoveFilter) -> None:
        """Handle removal of a filter pill by reconstructing the query."""
        search_input = self._get_search_input_widget()
        search_input.value = remove_query_token(search_input.value, event.token_index)

    @on(FilterPillBar.RemoveWatchFilter)
    def on_remove_watch_filter(self, event: FilterPillBar.RemoveWatchFilter) -> None:
        """Handle removal of the watch filter pill."""
        self._watch_filter_active = False
        self._apply_filter(self._pending_query)

    def on_key(self, event: Key) -> None:
        """Handle key events for vim-style mark capture."""
        if self._pending_mark_action is None:
            return
        key = event.key
        # Only accept single lowercase letters
        if len(key) == 1 and key.isalpha() and key.islower():
            if self._pending_mark_action == "set":
                self._set_mark(key)
            elif self._pending_mark_action == "goto":
                self._goto_mark(key)
            event.prevent_default()
            event.stop()
        # Cancel mark mode on any other key
        self._pending_mark_action = None

    async def _fetch_paper_content_async(self, paper: Paper) -> str:
        """Fetch canonical paper content for LLM workflows."""
        return await _fetch_paper_content_async(
            paper,
            client=self._http_client,
            timeout=SUMMARY_HTML_TIMEOUT,
        )

    def action_show_search_syntax(self) -> None:
        """Open the help overlay with search syntax prioritized."""
        self.push_screen(HelpScreen(sections=self._build_help_sections(search_first=True)))


def main() -> int:
    """Main entry point wrapper for CLI/bootstrap logic."""
    return _cli_main(
        deps=CliDependencies(
            load_config_fn=load_config,
            discover_history_files_fn=discover_history_files,
            resolve_papers_fn=_resolve_papers,
            configure_logging_fn=_configure_logging,
            configure_color_mode_fn=_configure_color_mode,
            validate_interactive_tty_fn=_validate_interactive_tty,
            app_factory=ArxivBrowser,
            app_factory_supports_options=True,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
