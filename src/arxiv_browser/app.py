"""Backward-compatible compatibility shim for the public arxiv_browser.app surface."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys

import httpx

logger = logging.getLogger(__name__)

_PUBLIC_EXPORTS = [
    "ARXIV_API_DEFAULT_MAX_RESULTS",
    "ARXIV_API_MAX_RESULTS_LIMIT",
    "ARXIV_DATE_FORMAT",
    "ARXIV_QUERY_FIELDS",
    "ATOM_NS",
    "AUTO_TAG_PROMPT_TEMPLATE",
    "CATEGORY_COLORS",
    "CATPPUCCIN_MOCHA_THEME",
    "CHAT_SYSTEM_PROMPT",
    "COMMAND_PALETTE_COMMANDS",
    "CONFIG_APP_NAME",
    "CONFIG_FILENAME",
    "DEFAULT_BIBTEX_EXPORT_DIR",
    "DEFAULT_CATEGORY_COLOR",
    "DEFAULT_CATEGORY_COLORS",
    "DEFAULT_COLLAPSED_SECTIONS",
    "DEFAULT_LLM_PROMPT",
    "DEFAULT_PDF_DOWNLOAD_DIR",
    "DEFAULT_THEME",
    "DETAIL_SECTION_KEYS",
    "DETAIL_SECTION_NAMES",
    "HISTORY_DATE_FORMAT",
    "LLM_PRESETS",
    "MAX_COLLECTIONS",
    "MAX_PAPERS_PER_COLLECTION",
    "RELEVANCE_PROMPT_TEMPLATE",
    "SIMILARITY_READ_PENALTY",
    "SIMILARITY_RECENCY_DAYS",
    "SIMILARITY_RECENCY_WEIGHT",
    "SIMILARITY_STARRED_BOOST",
    "SIMILARITY_TOP_N",
    "SIMILARITY_UNREAD_BOOST",
    "SIMILARITY_WEIGHT_AUTHOR",
    "SIMILARITY_WEIGHT_CATEGORY",
    "SIMILARITY_WEIGHT_TEXT",
    "SOLARIZED_DARK_THEME",
    "SORT_OPTIONS",
    "STOPWORDS",
    "SUMMARY_MODES",
    "TAG_NAMESPACE_COLORS",
    "TEXTUAL_THEMES",
    "THEMES",
    "THEME_CATEGORY_COLORS",
    "THEME_COLORS",
    "THEME_NAMES",
    "THEME_TAG_NAMESPACE_COLORS",
    "WATCH_MATCH_TYPES",
    "AddToCollectionModal",
    "ArxivBrowser",
    "ArxivBrowserOptions",
    "ArxivSearchModeState",
    "ArxivSearchRequest",
    "CLIProvider",
    "CliDependencies",
    "CollectionViewModal",
    "CollectionsModal",
    "CommandPaletteModal",
    "DetailRenderState",
    "FilterPillBar",
    "LLMResult",
    "LocalBrowseSnapshot",
    "MetadataSnapshotPickerModal",
    "Paper",
    "PaperChatScreen",
    "PaperCollection",
    "PaperDetails",
    "PaperHighlightTerms",
    "PaperListItem",
    "PaperMetadata",
    "PaperRowRenderState",
    "QueryToken",
    "RecommendationSourceModal",
    "SearchBookmark",
    "SectionToggleModal",
    "SessionState",
    "StatusBarState",
    "TfidfIndex",
    "UserConfig",
    "WatchListEntry",
    "_configure_logging",
    "build_arxiv_search_query",
    "build_auto_tag_prompt",
    "build_daily_digest",
    "build_highlight_terms",
    "build_llm_prompt",
    "build_relevance_prompt",
    "clean_latex",
    "compute_paper_similarity",
    "count_papers_in_file",
    "discover_history_files",
    "escape_bibtex",
    "escape_rich_text",
    "export_metadata",
    "extract_text_from_html",
    "extract_year",
    "find_similar_papers",
    "format_authors_bibtex",
    "format_categories",
    "format_collection_as_markdown",
    "format_paper_as_bibtex",
    "format_paper_as_markdown",
    "format_paper_as_ris",
    "format_paper_for_clipboard",
    "format_papers_as_csv",
    "format_papers_as_markdown_table",
    "format_summary_as_rich",
    "generate_citation_key",
    "get_config_path",
    "get_paper_url",
    "get_pdf_download_path",
    "get_pdf_url",
    "get_relevance_db_path",
    "get_summary_db_path",
    "get_tag_color",
    "highlight_text",
    "import_metadata",
    "insert_implicit_and",
    "is_advanced_query",
    "load_config",
    "main",
    "match_query_term",
    "matches_advanced_query",
    "normalize_arxiv_id",
    "paper_matches_watch_entry",
    "parse_arxiv_api_feed",
    "parse_arxiv_date",
    "parse_arxiv_file",
    "parse_arxiv_version_map",
    "parse_tag_namespace",
    "pill_label_for_token",
    "reconstruct_query",
    "render_paper_option",
    "render_progress_bar",
    "resolve_provider",
    "save_config",
    "sort_papers",
    "to_rpn",
    "tokenize_query",
    "truncate_text",
]
__all__ = list(_PUBLIC_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]

_EXPORT_MODULES = (
    "arxiv_browser.browser.core",
    "arxiv_browser.action_messages",
    "arxiv_browser.cli",
    "arxiv_browser.config",
    "arxiv_browser.enrichment",
    "arxiv_browser.export",
    "arxiv_browser.help_ui",
    "arxiv_browser.huggingface",
    "arxiv_browser.io_actions",
    "arxiv_browser.llm",
    "arxiv_browser.llm_providers",
    "arxiv_browser.modals",
    "arxiv_browser.models",
    "arxiv_browser.parsing",
    "arxiv_browser.query",
    "arxiv_browser.semantic_scholar",
    "arxiv_browser.services.interfaces",
    "arxiv_browser.services.llm_service",
    "arxiv_browser.similarity",
    "arxiv_browser.themes",
    "arxiv_browser.ui_constants",
    "arxiv_browser.ui_runtime",
    "arxiv_browser.widgets",
    "arxiv_browser.widgets.chrome",
    "arxiv_browser.widgets.details",
    "arxiv_browser.widgets.listing",
)


async def _fetch_paper_content_async(
    paper, client: httpx.AsyncClient | None = None, timeout: int | None = None
) -> str:
    """Fetch full paper text through the compatibility module patch surface."""
    html_url = f"https://arxiv.org/html/{paper.arxiv_id}"
    request_timeout = __getattr__("ARXIV_HTML_TIMEOUT") if timeout is None else timeout
    max_length = __getattr__("MAX_PAPER_CONTENT_LENGTH")
    extract_html = globals().get("extract_text_from_html")
    if extract_html is None:
        extract_html = __getattr__("extract_text_from_html")

    try:
        if client is not None:
            response = await client.get(html_url, timeout=request_timeout, follow_redirects=True)
        else:
            async with httpx.AsyncClient() as temp_client:
                response = await temp_client.get(
                    html_url,
                    timeout=request_timeout,
                    follow_redirects=True,
                )
        if response.status_code == 200:
            text = await asyncio.to_thread(extract_html, response.text)
            if text:
                return text[:max_length]
        else:
            logger.warning(
                "arXiv HTML fetch returned %d for %s",
                response.status_code,
                paper.arxiv_id,
            )
    except (httpx.HTTPError, OSError):
        logger.warning("Failed to fetch HTML for %s", paper.arxiv_id, exc_info=True)

    abstract = paper.abstract or paper.abstract_raw or ""
    return f"Abstract:\n{abstract}" if abstract else ""


def __getattr__(name: str):
    for module_name in _EXPORT_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'arxiv_browser.app' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def main() -> int:
    """CLI entry point that preserves the legacy arxiv_browser.app patch surface."""
    cli_module = importlib.import_module("arxiv_browser.cli")
    return cli_module.main(
        deps=cli_module.CliDependencies(
            load_config_fn=globals().get("load_config") or __getattr__("load_config"),
            discover_history_files_fn=globals().get("discover_history_files")
            or __getattr__("discover_history_files"),
            resolve_papers_fn=globals().get("_resolve_papers") or __getattr__("_resolve_papers"),
            configure_logging_fn=globals().get("_configure_logging")
            or __getattr__("_configure_logging"),
            configure_color_mode_fn=globals().get("_configure_color_mode")
            or __getattr__("_configure_color_mode"),
            validate_interactive_tty_fn=globals().get("_validate_interactive_tty")
            or __getattr__("_validate_interactive_tty"),
            app_factory=globals().get("ArxivBrowser") or __getattr__("ArxivBrowser"),
        )
    )


if __name__ == "__main__":
    sys.exit(main())
