#!/usr/bin/env python3
"""Tests for arXiv Paper Browser TUI."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from arxiv_browser.browser.core import SUBPROCESS_TIMEOUT
from arxiv_browser.config import (
    export_metadata,
    import_metadata,
    load_config,
    save_config,
)
from arxiv_browser.export import (
    escape_bibtex,
    extract_year,
    format_collection_as_markdown,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
    format_papers_as_markdown_table,
    generate_citation_key,
    get_pdf_download_path,
)
from arxiv_browser.llm import (
    DEFAULT_LLM_PROMPT,
    LLM_PRESETS,
    SUMMARY_MODES,
    build_llm_prompt,
    get_summary_db_path,
)
from arxiv_browser.models import (
    ARXIV_API_DEFAULT_MAX_RESULTS,
    MAX_COLLECTIONS,
    MAX_PAPERS_PER_COLLECTION,
    SORT_OPTIONS,
    Paper,
    PaperCollection,
    PaperMetadata,
    QueryToken,
    SearchBookmark,
    UserConfig,
    WatchListEntry,
)
from arxiv_browser.parsing import (
    ARXIV_DATE_FORMAT,
    build_arxiv_search_query,
    clean_latex,
    extract_text_from_html,
    normalize_arxiv_id,
    parse_arxiv_api_feed,
    parse_arxiv_date,
    parse_arxiv_file,
    parse_arxiv_version_map,
)
from arxiv_browser.query import (
    format_categories,
    format_summary_as_rich,
    insert_implicit_and,
    pill_label_for_token,
    reconstruct_query,
    to_rpn,
    tokenize_query,
)
from arxiv_browser.themes import (
    DEFAULT_CATEGORY_COLOR,
    TAG_NAMESPACE_COLORS,
    THEME_NAMES,
    THEMES,
    get_tag_color,
    parse_tag_namespace,
)
from tests.support.patch_helpers import patch_save_config

# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestCountPapersInFile:
    """Tests for count_papers_in_file utility."""

    def test_counts_ids(self, tmp_path):
        from arxiv_browser.parsing import count_papers_in_file

        f = tmp_path / "test.txt"
        f.write_text(
            "arXiv:2401.12345 some paper\narXiv:2401.67890v2 another\n",
            encoding="utf-8",
        )
        assert count_papers_in_file(f) == 2

    def test_missing_file(self, tmp_path):
        from arxiv_browser.parsing import count_papers_in_file

        assert count_papers_in_file(tmp_path / "nonexistent.txt") == 0

    def test_empty_file(self, tmp_path):
        from arxiv_browser.parsing import count_papers_in_file

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert count_papers_in_file(f) == 0


class TestDateNavigator:
    """Tests for DateNavigator widget."""

    def test_window_centers_on_current(self):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import _compute_window_bounds

        # Create 10 fake history entries
        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        start, end = _compute_window_bounds(len(files), 5, 5)
        assert (start, end) == (3, 8)

    def test_window_clamps_at_edges(self):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import _compute_window_bounds

        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        assert _compute_window_bounds(len(files), 0, 5) == (0, 5)
        assert _compute_window_bounds(len(files), len(files) - 1, 5) == (5, 10)

    def test_small_file_list(self):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import _compute_window_bounds

        # Only 3 files — window should show all of them
        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(3)]
        assert _compute_window_bounds(len(files), 1, 5) == (0, 3)

    def test_responsive_plan_keeps_counts_when_wide(self):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import (
            DATE_NAV_LABEL_WITH_COUNTS,
            _compute_responsive_date_plan,
        )

        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        start, end, label_mode = _compute_responsive_date_plan(
            files,
            5,
            width=90,
            get_count=lambda _index: 382,
        )

        assert label_mode == DATE_NAV_LABEL_WITH_COUNTS
        assert (start, end) == (3, 8)

    def test_responsive_plan_drops_counts_before_dropping_dates(self):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import (
            DATE_NAV_LABEL_MONTH_DAY,
            _compute_responsive_date_plan,
        )

        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        start, end, label_mode = _compute_responsive_date_plan(
            files,
            5,
            width=60,
            get_count=lambda _index: 382,
        )

        assert label_mode == DATE_NAV_LABEL_MONTH_DAY
        assert end - start == 5
        assert start <= 5 < end

    def test_responsive_plan_compacts_window_and_labels_when_narrow(self):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import (
            DATE_NAV_LABEL_NUMERIC,
            _compute_responsive_date_plan,
        )

        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        start, end, label_mode = _compute_responsive_date_plan(
            files,
            5,
            width=30,
            get_count=lambda _index: 382,
        )

        assert label_mode == DATE_NAV_LABEL_NUMERIC
        assert end - start < 5
        assert start <= 5 < end

    def test_count_cache_tracks_file_path_not_index(self, tmp_path):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import DateNavigator

        first = tmp_path / "2026-01-01.txt"
        second = tmp_path / "2026-01-02.txt"
        replacement = tmp_path / "2026-01-03.txt"

        first.write_text("arXiv:2401.00001\n", encoding="utf-8")
        second.write_text("arXiv:2401.00002\narXiv:2401.00003\n", encoding="utf-8")
        replacement.write_text(
            "arXiv:2401.00004\narXiv:2401.00005\narXiv:2401.00006\n",
            encoding="utf-8",
        )

        nav = DateNavigator([(dt_date(2026, 1, 1), first), (dt_date(2026, 1, 2), second)])
        assert nav._get_paper_count(0) == 1

        # Replace index 0 with a different file; count should update accordingly.
        nav._history_files = [(dt_date(2026, 1, 3), replacement), (dt_date(2026, 1, 2), second)]
        assert nav._get_paper_count(0) == 3

    @pytest.mark.asyncio
    async def test_update_dates_does_not_create_duplicate_ids(self, tmp_path, make_paper):
        from datetime import date as dt_date
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = tmp_path / "2026-01-01.txt"
        f2 = tmp_path / "2026-01-02.txt"
        f3 = tmp_path / "2026-01-03.txt"
        f1.write_text("arXiv:2401.00001\n", encoding="utf-8")
        f2.write_text("arXiv:2401.00002\n", encoding="utf-8")
        f3.write_text("arXiv:2401.00003\n", encoding="utf-8")

        first = [
            (dt_date(2026, 1, 3), f3),
            (dt_date(2026, 1, 2), f2),
            (dt_date(2026, 1, 1), f1),
        ]
        second = [
            (dt_date(2026, 1, 4), f3),
            (dt_date(2026, 1, 3), f2),
            (dt_date(2026, 1, 2), f1),
        ]

        app = ArxivBrowser(
            [make_paper(arxiv_id="2401.00003", title="Paper for 2026-01-03")],
            restore_session=False,
            history_files=first,
            current_date_index=0,
        )
        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                await pilot.pause()
                nav = app.query_one(DateNavigator)
                await nav.update_dates(first, 0)
                await pilot.pause()
                await nav.update_dates(second, 1)
                await pilot.pause()
                date_item_ids = [
                    child.id
                    for child in nav.children
                    if "date-nav-item" in child.classes and child.id is not None
                ]
                assert len(date_item_ids) == len(set(date_item_ids))

    @pytest.mark.asyncio
    async def test_update_dates_hides_and_clears_items_for_short_history(
        self, tmp_path, make_paper
    ):
        from datetime import date as dt_date
        from unittest.mock import patch

        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.widgets.chrome import DateNavigator

        f1 = tmp_path / "2026-01-01.txt"
        f2 = tmp_path / "2026-01-02.txt"
        f1.write_text("arXiv:2401.00001\n", encoding="utf-8")
        f2.write_text("arXiv:2401.00002\n", encoding="utf-8")

        history = [
            (dt_date(2026, 1, 2), f2),
            (dt_date(2026, 1, 1), f1),
        ]

        app = ArxivBrowser(
            [make_paper(arxiv_id="2401.00002", title="Paper for 2026-01-02")],
            restore_session=False,
            history_files=history,
            current_date_index=0,
        )

        with patch_save_config(return_value=True):
            async with app.run_test() as pilot:
                await pilot.pause()
                nav = app.query_one(DateNavigator)

                await nav.update_dates(history, 0)
                await pilot.pause()
                assert "visible" in nav.classes
                assert any("date-nav-item" in child.classes for child in nav.children)

                await nav.update_dates([history[0]], 0)
                await pilot.pause()
                assert "visible" not in nav.classes
                assert not any("date-nav-item" in child.classes for child in nav.children)

                await nav.update_dates([], 0)
                await pilot.pause()
                assert "visible" not in nav.classes
                assert not any("date-nav-item" in child.classes for child in nav.children)

    @pytest.mark.asyncio
    async def test_update_dates_prunes_stale_count_cache(self, tmp_path):
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import DateNavigator

        keep = tmp_path / "2026-01-01.txt"
        stale = tmp_path / "2026-01-02.txt"
        keep.write_text("arXiv:2401.00001\n", encoding="utf-8")
        stale.write_text("arXiv:2401.00002\narXiv:2401.00003\n", encoding="utf-8")

        nav = DateNavigator([(dt_date(2026, 1, 2), stale), (dt_date(2026, 1, 1), keep)])
        nav._paper_counts = {keep: 1, stale: 2}

        await nav.update_dates([(dt_date(2026, 1, 1), keep)], 0)

        assert nav._paper_counts == {keep: 1}


class TestChromeWidgetBranches:
    """Extra branch coverage for chrome.py widget helpers."""

    # ── _compute_responsive_date_plan edge paths ─────────────────────────────

    def test_responsive_plan_empty_history_returns_zero_range(self):
        """Line 136: total=0 returns (0, 0, WITH_COUNTS) immediately."""
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import (
            DATE_NAV_LABEL_WITH_COUNTS,
            _compute_responsive_date_plan,
        )

        start, end, mode = _compute_responsive_date_plan([], 0, width=90, get_count=lambda _: 0)
        assert (start, end) == (0, 0)
        assert mode == DATE_NAV_LABEL_WITH_COUNTS

    def test_responsive_plan_very_narrow_width_falls_back_to_numeric(self):
        """Lines 158-159: no window/mode fits → fallback single-window NUMERIC."""
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import (
            DATE_NAV_LABEL_NUMERIC,
            _compute_responsive_date_plan,
        )

        files = [(dt_date(2026, 1, 1), Path("/tmp/f.txt"))]
        # width=1 is > 0 (avoids line 138-140) but too small for any label
        start, end, mode = _compute_responsive_date_plan(files, 0, width=1, get_count=lambda _: 0)
        assert mode == DATE_NAV_LABEL_NUMERIC
        assert end - start == 1

    # ── DateNavigator unit helpers ────────────────────────────────────────────

    def test_compute_window_method_delegates(self):
        """Line 260: _compute_window calls _compute_window_bounds correctly."""
        from datetime import date as dt_date

        from arxiv_browser.widgets.chrome import DateNavigator

        files = [(dt_date(2026, 1, i + 1), Path(f"/tmp/{i}.txt")) for i in range(10)]
        nav = DateNavigator(files, current_index=5)
        result = nav._compute_window(10, 5)
        assert result == (3, 8)

    def test_patch_items_in_place_skips_missing_child(self):
        """Line 312: continue when desired item_id not found in existing_by_id."""
        from datetime import date as dt_date
        from unittest.mock import MagicMock

        from arxiv_browser.widgets.chrome import DateNavigator

        nav = DateNavigator([])
        existing_label = MagicMock()
        existing_label.id = "date-nav-0"
        # desired has an ID absent from existing_by_id → child is None → continue
        nav._patch_items_in_place([existing_label], [("date-nav-99", "text", True)])
        # existing_label.update should NOT have been called (child was None, continue hit)
        existing_label.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_rebuild_items_removes_existing_before_mounting(self):
        """Line 326: await child.remove() called for each existing item."""
        from unittest.mock import AsyncMock, MagicMock

        from arxiv_browser.widgets.chrome import DateNavigator

        nav = DateNavigator([])
        mock_label = MagicMock()
        mock_label.remove = AsyncMock()
        nav.query_one = MagicMock(return_value=MagicMock())
        nav.mount = MagicMock()
        await nav._rebuild_items([mock_label], [("date-nav-1", "Jan 02", True)])
        mock_label.remove.assert_awaited_once()
        nav.mount.assert_called_once()

    # ── DateNavigator.on_click edge paths ────────────────────────────────────

    def test_on_click_non_click_event_returns_early(self):
        """Line 378: on_click with a non-Click event does nothing."""
        from unittest.mock import MagicMock

        from arxiv_browser.widgets.chrome import DateNavigator

        nav = DateNavigator([])
        nav.post_message = MagicMock()
        nav.on_click("not-a-click-event")
        nav.post_message.assert_not_called()

    def test_on_click_none_widget_returns_early(self):
        """Line 381: on_click with Click(widget=None) does nothing."""
        from unittest.mock import MagicMock

        from textual.events import Click

        from arxiv_browser.widgets.chrome import DateNavigator

        nav = DateNavigator([])
        nav.post_message = MagicMock()
        click = Click(
            widget=None,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
        )
        nav.on_click(click)
        nav.post_message.assert_not_called()

    def test_on_click_unrecognized_widget_id_is_noop(self):
        """387->exit: widget_id matches none of the known IDs → silent no-op."""
        from unittest.mock import MagicMock

        from textual.events import Click
        from textual.widgets import Label

        from arxiv_browser.widgets.chrome import DateNavigator

        nav = DateNavigator([])
        nav.post_message = MagicMock()
        widget = Label("x", id="some-unrelated-button")
        nav.on_click(
            Click(
                widget=widget,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        nav.post_message.assert_not_called()

    def test_on_click_invalid_date_nav_id_raises_value_error(self):
        """Lines 391-392: int('abc') raises ValueError → except pass, no message."""
        from unittest.mock import MagicMock

        from textual.events import Click
        from textual.widgets import Label

        from arxiv_browser.widgets.chrome import DateNavigator

        nav = DateNavigator([])
        nav.post_message = MagicMock()
        widget = Label("x", id="date-nav-abc")  # removeprefix → 'abc', int() raises
        nav.on_click(
            Click(
                widget=widget,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        nav.post_message.assert_not_called()

    # ── BookmarkTabBar branches ───────────────────────────────────────────────

    def test_bookmark_tab_bar_init_adds_visible_class(self):
        """Line 459: __init__ calls add_class('visible') when bookmarks present."""
        from arxiv_browser.widgets.chrome import BookmarkTabBar

        bm = SearchBookmark(name="Test", query="test")
        bar = BookmarkTabBar(bookmarks=[bm], active_index=0)
        assert "visible" in bar.classes

        bar2 = BookmarkTabBar(bookmarks=[], active_search=True)
        assert "visible" in bar2.classes

    def test_bookmark_tab_bar_compose_renders_tabs(self):
        """Lines 465-468: compose yields tab labels when bookmarks non-empty."""
        from arxiv_browser.widgets.chrome import BookmarkTabBar

        bm = SearchBookmark(name="MLPapers", query="cat:cs.LG")
        bar = BookmarkTabBar(bookmarks=[bm], active_index=0)
        children = list(bar.compose())
        ids = [getattr(c, "id", None) for c in children]
        assert "bookmark-0" in ids
        assert "bookmark-add" in ids

    def test_bookmark_tab_bar_compose_active_search_hint(self):
        """Line 470: compose yields save-hint label when active_search with no bookmarks."""
        from arxiv_browser.widgets.chrome import BookmarkTabBar

        bar = BookmarkTabBar(bookmarks=[], active_search=True)
        children = list(bar.compose())
        ids = [getattr(c, "id", None) for c in children]
        assert "bookmark-hint" in ids

    @pytest.mark.asyncio
    async def test_bookmark_tab_bar_update_bookmarks_mounts_tabs(self):
        """Lines 490-493: update_bookmarks mounts tab labels when bookmarks given."""
        from unittest.mock import AsyncMock, MagicMock

        from arxiv_browser.widgets.chrome import BookmarkTabBar

        bar = BookmarkTabBar(bookmarks=[])
        bar.remove_children = AsyncMock()
        bar.mount = MagicMock()
        bar.add_class = MagicMock()
        bar.remove_class = MagicMock()

        bm = SearchBookmark(name="Saved", query="graph")
        await bar.update_bookmarks(bookmarks=[bm], active_index=0)
        # At least 3 mounts: "Saved searches" label + tab label + "Ctrl+b save" label
        assert bar.mount.call_count >= 3

    # ── ContextFooter.render_bindings branch ─────────────────────────────────

    def test_context_footer_render_bindings_label_only_entry(self):
        """Line 83: elif label: path hit when key is empty but label is non-empty."""
        from unittest.mock import MagicMock, patch

        from arxiv_browser.widgets.chrome import ContextFooter

        with patch(
            "arxiv_browser.widgets.chrome.theme_colors_for",
            return_value={"accent": "blue", "muted": "gray"},
        ):
            footer = ContextFooter()
            footer.update = MagicMock()
            # ("", "progress text") → key empty, label non-empty → line 83
            footer.render_bindings([("", "progress: 50%")])
            footer.update.assert_called_once()
            rendered = footer.update.call_args[0][0]
            assert "progress: 50%" in rendered

    # ── FilterPillBar.on_click edge paths ─────────────────────────────────────

    def test_filter_pill_bar_on_click_non_click_returns_early(self):
        """Line 609: non-Click event → return immediately."""
        from unittest.mock import MagicMock

        from arxiv_browser.widgets.chrome import FilterPillBar

        bar = FilterPillBar()
        bar.post_message = MagicMock()
        bar.on_click("not-a-click-event")
        bar.post_message.assert_not_called()

    def test_filter_pill_bar_on_click_non_label_widget_returns_early(self):
        """Line 612: Click with non-Label widget → return immediately."""
        from unittest.mock import MagicMock

        from textual.events import Click
        from textual.widgets import Static

        from arxiv_browser.widgets.chrome import FilterPillBar

        bar = FilterPillBar()
        bar.post_message = MagicMock()
        non_label = Static("hello", id="some-id")
        bar.on_click(
            Click(
                widget=non_label,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        bar.post_message.assert_not_called()

    def test_filter_pill_bar_on_click_unrecognized_id_is_noop(self):
        """616->exit: widget_id neither 'pill-watch' nor starts with 'pill-' → no-op."""
        from unittest.mock import MagicMock

        from textual.events import Click
        from textual.widgets import Label

        from arxiv_browser.widgets.chrome import FilterPillBar

        bar = FilterPillBar()
        bar.post_message = MagicMock()
        unrelated = Label("other", id="bookmark-0")
        bar.on_click(
            Click(
                widget=unrelated,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        bar.post_message.assert_not_called()


class TestThemeSwitcher:
    """Tests for U7: Color theme switcher."""

    def test_themes_have_matching_keys(self):
        from arxiv_browser.themes import (
            CATPPUCCIN_MOCHA_THEME,
            DEFAULT_THEME,
        )

        assert set(DEFAULT_THEME.keys()) == set(CATPPUCCIN_MOCHA_THEME.keys())

    def test_theme_name_roundtrip(self):
        from arxiv_browser.config import (
            _config_to_dict,
            _dict_to_config,
        )

        config = UserConfig(theme_name="catppuccin-mocha")
        data = _config_to_dict(config)
        assert data["theme_name"] == "catppuccin-mocha"
        restored = _dict_to_config(data)
        assert restored.theme_name == "catppuccin-mocha"

    def test_theme_name_defaults_to_monokai(self):
        from arxiv_browser.config import _dict_to_config

        config = _dict_to_config({})
        assert config.theme_name == "monokai"

    def test_apply_uses_named_theme(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import CATPPUCCIN_MOCHA_THEME

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="catppuccin-mocha")
        app._http_client = None
        app._apply_theme_overrides()
        assert app._theme_runtime.colors["accent"] == CATPPUCCIN_MOCHA_THEME["accent"]
        assert app._theme_runtime.colors["green"] == CATPPUCCIN_MOCHA_THEME["green"]

    def test_per_key_override_layers(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import CATPPUCCIN_MOCHA_THEME

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(
            theme_name="catppuccin-mocha",
            theme={"accent": "#ff0000"},
        )
        app._http_client = None
        app._apply_theme_overrides()
        # Per-key override wins over base theme
        assert app._theme_runtime.colors["accent"] == "#ff0000"
        # Other keys come from the base theme
        assert app._theme_runtime.colors["green"] == CATPPUCCIN_MOCHA_THEME["green"]

    def test_unknown_theme_falls_back_to_default(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import DEFAULT_THEME

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="nonexistent-theme")
        app._http_client = None
        app._apply_theme_overrides()
        assert app._theme_runtime.colors["accent"] == DEFAULT_THEME["accent"]

    def test_category_overrides_rebuild_current_theme(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import SOLARIZED_DARK_THEME

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="catppuccin-mocha")
        app._http_client = None
        app._apply_theme_overrides()
        app._config.theme_name = "solarized-dark"
        app._apply_category_overrides()

        assert app._theme_runtime.colors["accent"] == SOLARIZED_DARK_THEME["accent"]


class TestProgressIndicators:
    """Tests for U4: X/Y counter progress indicators in footer."""

    def _make_app(self):
        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig()
        app._relevance_scoring_active = False
        app._version_checking = False
        app._scoring_progress = None
        app._version_progress = None
        app._in_arxiv_api_mode = False
        app.selected_ids = set()
        app._s2_active = False
        app._history_files = []
        app._download_queue = []
        app._downloading = set()
        app._download_total = 0
        app._download_results = {}
        app._auto_tag_active = False
        app._auto_tag_progress = None
        return app

    def test_scoring_progress_in_footer(self):
        app = self._make_app()
        app._scoring_progress = (3, 50)
        bindings = app._get_footer_bindings()
        assert any("Scoring" in label and "3/50" in label for _, label in bindings)

    def test_version_progress_in_footer(self):
        app = self._make_app()
        app._version_progress = (2, 5)
        bindings = app._get_footer_bindings()
        assert any("Versions" in label and "2/5" in label for _, label in bindings)

    def test_scoring_progress_overrides_boolean(self):
        """Tuple progress takes priority over boolean flag."""
        app = self._make_app()
        app._relevance_scoring_active = True
        app._scoring_progress = (7, 20)
        bindings = app._get_footer_bindings()
        # Should show X/Y with progress bar, not static text
        labels = [label for _, label in bindings]
        assert any("Scoring" in lbl and "7/20" in lbl for lbl in labels)
        assert not any("Scoring papers" in lbl for lbl in labels)

    def test_version_progress_overrides_boolean(self):
        """Tuple progress takes priority over boolean flag."""
        app = self._make_app()
        app._version_checking = True
        app._version_progress = (1, 3)
        bindings = app._get_footer_bindings()
        labels = [label for _, label in bindings]
        assert any("Versions" in lbl and "1/3" in lbl for lbl in labels)
        assert not any("Checking versions" in lbl for lbl in labels)

    def test_scoring_progress_cleared_is_none(self):
        """When progress is None, boolean fallback is used."""
        app = self._make_app()
        app._relevance_scoring_active = True
        app._scoring_progress = None
        bindings = app._get_footer_bindings()
        # Falls back to boolean text
        labels = [label for _, label in bindings]
        assert any("Scoring papers" in lbl for lbl in labels)


class TestDailyDigest:
    """Tests for build_daily_digest function."""

    def test_empty_papers(self):
        from arxiv_browser.parsing import build_daily_digest

        assert build_daily_digest([]) == "No papers loaded"

    def test_basic_digest(self, make_paper):
        from arxiv_browser.parsing import build_daily_digest

        papers = [
            make_paper(categories="cs.AI cs.LG"),
            make_paper(categories="cs.AI"),
            make_paper(categories="cs.CL"),
        ]
        digest = build_daily_digest(papers)
        assert "3 papers" in digest
        assert "cs.AI (2)" in digest

    def test_digest_with_watch_matches(self, make_paper):
        from arxiv_browser.parsing import build_daily_digest

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        digest = build_daily_digest(papers, watched_ids={"2401.00001"})
        assert "1 match" in digest
        assert "watch list" in digest

    def test_digest_with_metadata(self, make_paper):
        from arxiv_browser.parsing import build_daily_digest

        papers = [make_paper(arxiv_id="2401.00001"), make_paper(arxiv_id="2401.00002")]
        meta = {
            "2401.00001": PaperMetadata(arxiv_id="2401.00001", is_read=True),
            "2401.00002": PaperMetadata(arxiv_id="2401.00002", starred=True),
        }
        digest = build_daily_digest(papers, metadata=meta)
        assert "1 read" in digest
        assert "1 starred" in digest

    def test_digest_top_categories_capped_at_5(self, make_paper):
        from arxiv_browser.parsing import build_daily_digest

        papers = [make_paper(categories=f"cs.{chr(65 + i)}") for i in range(10)]
        digest = build_daily_digest(papers)
        # Should only show top 5
        assert digest.count("(1)") <= 5


class TestAutoTagPrompt:
    """Tests for build_auto_tag_prompt and _parse_auto_tag_response."""

    def test_build_prompt_with_taxonomy(self, make_paper):
        from arxiv_browser.llm import build_auto_tag_prompt

        paper = make_paper(
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            categories="cs.CL, cs.LG",
            abstract="We propose transformer architecture...",
        )
        taxonomy = ["topic:nlp", "method:attention", "status:to-read"]
        prompt = build_auto_tag_prompt(paper, taxonomy)
        assert "topic:nlp" in prompt
        assert "method:attention" in prompt
        assert "Attention Is All You Need" in prompt
        assert "Vaswani et al." in prompt

    def test_build_prompt_empty_taxonomy(self, make_paper):
        from arxiv_browser.llm import build_auto_tag_prompt

        paper = make_paper(title="Test Paper")
        prompt = build_auto_tag_prompt(paper, [])
        assert "no existing tags" in prompt

    def test_parse_response_json(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response('{"tags": ["topic:nlp", "method:transformer"]}')
        assert result == ["topic:nlp", "method:transformer"]

    def test_parse_response_markdown_fence(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response('```json\n{"tags": ["topic:cv"]}\n```')
        assert result == ["topic:cv"]

    def test_parse_response_regex_fallback(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response(
            'Here are my suggestions:\n"tags": ["topic:ml", "status:important"]'
        )
        assert result == ["topic:ml", "status:important"]

    def test_parse_response_lowercases(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response('{"tags": ["Topic:NLP", "METHOD:CNN"]}')
        assert result == ["topic:nlp", "method:cnn"]

    def test_parse_response_strips_whitespace(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response('{"tags": ["  topic:ml  ", " status:done "]}')
        assert result == ["topic:ml", "status:done"]

    def test_parse_response_invalid_returns_none(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        assert _parse_auto_tag_response("I don't understand") is None

    def test_parse_response_empty_tags_list(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response('{"tags": []}')
        assert result == []

    def test_parse_response_filters_empty_strings(self):
        from arxiv_browser.llm import _parse_auto_tag_response

        result = _parse_auto_tag_response('{"tags": ["topic:ml", "", "  "]}')
        assert result == ["topic:ml"]


class TestAutoTagFooterProgress:
    """Tests for auto-tag progress in footer."""

    def _make_app(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.models import UserConfig

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig()
        app._relevance_scoring_active = False
        app._version_checking = False
        app._scoring_progress = None
        app._version_progress = None
        app._in_arxiv_api_mode = False
        app.selected_ids = set()
        app._s2_active = False
        app._history_files = []
        app._download_queue = []
        app._downloading = set()
        app._download_total = 0
        app._download_results = {}
        app._auto_tag_active = False
        app._auto_tag_progress = None
        return app

    def test_auto_tag_progress_in_footer(self):
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        app = self._make_app()
        app._auto_tag_progress = (3, 10)
        app._auto_tag_active = True
        app.query_one = MagicMock(side_effect=NoMatches())
        bindings = app._get_footer_bindings()
        labels = [label for _, label in bindings]
        assert any("Auto-tagging" in lbl and "3/10" in lbl for lbl in labels)

    def test_auto_tag_active_without_progress(self):
        from unittest.mock import MagicMock

        from textual.css.query import NoMatches

        app = self._make_app()
        app._auto_tag_active = True
        app.query_one = MagicMock(side_effect=NoMatches())
        bindings = app._get_footer_bindings()
        labels = [label for _, label in bindings]
        assert any("Auto-tagging" in lbl for lbl in labels)


class TestSolarizedDarkTheme:
    """Tests for U7: Solarized Dark theme expansion."""

    def test_solarized_theme_exists(self):
        from arxiv_browser.themes import SOLARIZED_DARK_THEME
        from arxiv_browser.themes import THEMES as APP_THEMES

        assert "solarized-dark" in APP_THEMES
        assert APP_THEMES["solarized-dark"] is SOLARIZED_DARK_THEME

    def test_solarized_has_all_keys(self):
        from arxiv_browser.themes import (
            DEFAULT_THEME,
            SOLARIZED_DARK_THEME,
        )

        assert set(SOLARIZED_DARK_THEME.keys()) == set(DEFAULT_THEME.keys())

    def test_solarized_palette_spot_check(self):
        from arxiv_browser.themes import SOLARIZED_DARK_THEME

        assert SOLARIZED_DARK_THEME["background"] == "#002b36"
        assert SOLARIZED_DARK_THEME["accent"] == "#3c9be2"  # WCAG AA adjusted
        assert SOLARIZED_DARK_THEME["green"] == "#859900"
        assert SOLARIZED_DARK_THEME["pink"] == "#e85da0"  # WCAG AA adjusted

    def test_four_themes_in_cycle(self):
        from arxiv_browser.themes import THEME_NAMES as APP_THEME_NAMES

        assert len(APP_THEME_NAMES) == 4
        assert "monokai" in APP_THEME_NAMES
        assert "catppuccin-mocha" in APP_THEME_NAMES
        assert "solarized-dark" in APP_THEME_NAMES
        assert "high-contrast" in APP_THEME_NAMES

    def test_solarized_config_roundtrip(self):
        from arxiv_browser.config import (
            _config_to_dict,
            _dict_to_config,
        )

        config = UserConfig(theme_name="solarized-dark")
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.theme_name == "solarized-dark"

    def test_category_colors_update_on_theme(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import THEME_CATEGORY_COLORS

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="solarized-dark")
        app._http_client = None
        app._apply_category_overrides()
        expected = THEME_CATEGORY_COLORS["solarized-dark"]
        for cat, color in expected.items():
            assert app._theme_runtime.category_colors[cat] == color

    def test_tag_namespace_colors_update_on_theme(self):
        from arxiv_browser.browser.core import ArxivBrowser
        from arxiv_browser.themes import THEME_TAG_NAMESPACE_COLORS

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(theme_name="solarized-dark")
        app._http_client = None
        app._apply_theme_overrides()
        expected = THEME_TAG_NAMESPACE_COLORS["solarized-dark"]
        for ns, color in expected.items():
            assert app._theme_runtime.tag_namespace_colors[ns] == color


class TestCollapsibleSections:
    """Tests for U3: Collapsible detail pane sections."""

    def test_default_collapsed_sections(self):
        from arxiv_browser.models import DEFAULT_COLLAPSED_SECTIONS

        config = UserConfig()
        assert config.collapsed_sections == DEFAULT_COLLAPSED_SECTIONS
        assert "tags" in config.collapsed_sections
        assert "authors" not in config.collapsed_sections

    def test_collapsed_sections_roundtrip(self):
        from arxiv_browser.config import (
            _config_to_dict,
            _dict_to_config,
        )

        config = UserConfig(collapsed_sections=["authors", "abstract"])
        data = _config_to_dict(config)
        assert data["collapsed_sections"] == ["authors", "abstract"]
        restored = _dict_to_config(data)
        assert restored.collapsed_sections == ["authors", "abstract"]

    def test_invalid_sections_filtered(self):
        from arxiv_browser.config import _dict_to_config

        data = {"collapsed_sections": ["authors", "invalid_key", "abstract", 42]}
        config = _dict_to_config(data)
        assert config.collapsed_sections == ["authors", "abstract"]

    def test_missing_collapsed_sections_uses_defaults(self):
        from arxiv_browser.config import _dict_to_config
        from arxiv_browser.models import DEFAULT_COLLAPSED_SECTIONS

        config = _dict_to_config({})
        assert config.collapsed_sections == DEFAULT_COLLAPSED_SECTIONS

    def test_expanded_section_shows_content(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(authors="John Doe")
        details.update_paper(paper, collapsed_sections=[])
        rendered = str(details.content)
        assert "John Doe" in rendered
        assert "\u25be Authors" in rendered  # expanded indicator

    def test_collapsed_section_hides_content(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper(authors="John Doe")
        details.update_paper(paper, collapsed_sections=["authors"])
        rendered = str(details.content)
        assert "John Doe" not in rendered
        assert "\u25b8 Authors" in rendered  # collapsed indicator

    def test_collapsed_s2_shows_citation_hint(self, make_paper):
        from arxiv_browser.semantic_scholar import SemanticScholarPaper
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        s2_data = SemanticScholarPaper(
            arxiv_id="2401.12345",
            s2_paper_id="abc",
            citation_count=42,
            influential_citation_count=5,
            tldr="A test paper.",
            fields_of_study=("Computer Science",),
            year=2024,
            url="https://api.semanticscholar.org/abc",
            title="Test",
        )
        details.update_paper(paper, s2_data=s2_data, collapsed_sections=["s2"])
        rendered = str(details.content)
        assert "42 cites" in rendered
        assert "\u25b8 Semantic Scholar" in rendered

    def test_collapsed_hf_shows_upvote_hint(self, make_paper):
        from arxiv_browser.huggingface import HuggingFacePaper
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        hf_data = HuggingFacePaper(
            arxiv_id="2401.12345",
            title="Test",
            upvotes=15,
            num_comments=3,
            github_repo="",
            github_stars=0,
            ai_summary="",
            ai_keywords=(),
        )
        details.update_paper(paper, hf_data=hf_data, collapsed_sections=["hf"])
        rendered = str(details.content)
        assert "\u219115" in rendered  # ↑15
        assert "\u25b8 HuggingFace" in rendered

    def test_collapsed_tags_shows_count(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(
            paper, tags=["ml", "topic:transformers", "status:read"], collapsed_sections=["tags"]
        )
        rendered = str(details.content)
        assert "Tags (3)" in rendered
        assert "\u25b8 Tags" in rendered

    def test_collapsed_relevance_shows_score(self, make_paper):
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        details.update_paper(paper, relevance=(8, "High quality"), collapsed_sections=["relevance"])
        rendered = str(details.content)
        assert "Relevance (\u26058/10)" in rendered
        assert "High quality" not in rendered

    def test_url_always_visible_despite_collapsed(self, make_paper):
        """URL section is always visible — not collapsible."""
        from arxiv_browser.widgets.details import PaperDetails

        details = PaperDetails()
        paper = make_paper()
        # Even if "url" appears in collapsed list (from old config), URL should show
        details.update_paper(paper, collapsed_sections=["url"])
        rendered = str(details.content)
        assert "URL" in rendered
        assert "arxiv.org" in rendered

    def test_detail_section_keys_complete(self):
        from arxiv_browser.models import (
            DETAIL_SECTION_KEYS,
            DETAIL_SECTION_NAMES,
        )

        assert len(DETAIL_SECTION_KEYS) == 8
        for key in DETAIL_SECTION_KEYS:
            assert key in DETAIL_SECTION_NAMES
