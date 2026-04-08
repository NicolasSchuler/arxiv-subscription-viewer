"""Shared stub builders for split action and coverage tests."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.huggingface import HuggingFacePaper
from arxiv_browser.models import UserConfig
from arxiv_browser.semantic_scholar import SemanticScholarPaper


def _new_app() -> ArxivBrowser:
    app = ArxivBrowser.__new__(ArxivBrowser)
    app._http_client = None
    app._visible_index_by_id = {}
    app._config = UserConfig()
    return app


def _make_s2_paper(arxiv_id: str) -> SemanticScholarPaper:
    return SemanticScholarPaper(
        arxiv_id=arxiv_id,
        s2_paper_id=f"s2:{arxiv_id}",
        citation_count=10,
        influential_citation_count=1,
        tldr="",
        fields_of_study=(),
        year=2024,
        url=f"https://api.semanticscholar.org/{arxiv_id}",
    )


def _make_hf_paper(arxiv_id: str) -> HuggingFacePaper:
    return HuggingFacePaper(
        arxiv_id=arxiv_id,
        title=f"HF {arxiv_id}",
        upvotes=5,
        num_comments=1,
        ai_summary="",
        ai_keywords=(),
        github_repo="",
        github_stars=0,
    )


class _DummyOptionList:
    def __init__(self) -> None:
        self.highlighted: int | None = None
        self.option_count = 0

    def clear_options(self) -> None:
        self.option_count = 0

    def add_options(self, options: list[object]) -> None:
        self.option_count = len(options)

    def add_option(self, _option: object) -> None:
        self.option_count += 1


class _OptionListStub:
    def __init__(self, highlighted: int | None = None, option_count: int = 0) -> None:
        self.highlighted = highlighted
        self.highlighted_child = None
        self.option_count = option_count
        self.focused = False
        self.options: list[object] = []
        self.replaced: list[tuple[int, str]] = []
        self.classes: set[str] = set()

    def clear_options(self) -> None:
        self.options.clear()
        self.option_count = 0

    def add_options(self, options: list[object]) -> None:
        self.options.extend(options)
        self.option_count = len(self.options)

    def add_option(self, option: object) -> None:
        self.options.append(option)
        self.option_count = len(self.options)

    def replace_option_prompt_at_index(self, index: int, markup: str) -> None:
        self.replaced.append((index, markup))

    def focus(self) -> None:
        self.focused = True

    def remove_class(self, class_name: str) -> None:
        self.classes.discard(class_name)

    def add_class(self, class_name: str) -> None:
        self.classes.add(class_name)

    def has_class(self, class_name: str) -> bool:
        return class_name in self.classes


class _DummyInput:
    def __init__(self, value: str = "") -> None:
        self.value = value
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _DummyLabel:
    def __init__(self, content: str = "") -> None:
        self.content = content

    def update(self, value: str) -> None:
        self.content = value


class _DummyListView:
    def __init__(self, index: int | None = None) -> None:
        self.index = index
        self.children: list[object] = []
        self.mounted: list[object] = []
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1
        self.children.clear()
        self.mounted.clear()

    def mount(self, item: object) -> None:
        self.children.append(item)
        self.mounted.append(item)


class _DummyTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _paper(arxiv_id: str = "2401.12345", **kwargs):
    from arxiv_browser.models import Paper

    defaults = {
        "arxiv_id": arxiv_id,
        "date": "Mon, 15 Jan 2024",
        "title": f"Paper {arxiv_id}",
        "authors": "A. Author",
        "categories": "cs.AI",
        "comments": None,
        "abstract": "Abstract text.",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "abstract_raw": "Abstract text.",
    }
    defaults.update(kwargs)
    return Paper(**defaults)


def _make_app_config(**kwargs):
    config = UserConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def _new_app_stub():
    app = ArxivBrowser.__new__(ArxivBrowser)
    app.notify = MagicMock()
    app._config = UserConfig()
    app._detail_timer = None
    app._background_tasks = set()
    app._dataset_tasks = set()
    app._ui_refs = SimpleNamespace()
    app._update_status_bar = MagicMock()
    app._update_footer = MagicMock()
    app._update_header = MagicMock()
    app._update_subtitle = MagicMock()
    app._mark_badges_dirty = MagicMock()
    app._refresh_detail_pane = MagicMock()
    app._refresh_list_view = MagicMock()
    app._save_config_or_warn = MagicMock()
    app._get_ui_refresh_coordinator = MagicMock(
        return_value=SimpleNamespace(refresh_detail_pane=MagicMock())
    )
    app._track_task = MagicMock(side_effect=lambda coro: coro.close())
    app._track_dataset_task = MagicMock(side_effect=lambda coro: coro.close())
    app._abstract_cache = {}
    app._abstract_loading = set()
    app._abstract_queue = deque()
    app._abstract_pending_ids = set()
    app._s2_active = False
    app._hf_active = False
    app._s2_cache = {}
    app._hf_cache = {}
    app._s2_loading = set()
    app._hf_loading = False
    app._badges_dirty = set()
    app._badge_timer = None
    app._sort_refresh_dirty = set()
    app._sort_refresh_timer = None
    app._download_queue = deque()
    app._downloading = set()
    app._download_results = {}
    app._download_total = 0
    app._papers_by_id = {}
    app.filtered_papers = []
    app.all_papers = []
    app.selected_ids = set()
    app._highlight_terms = {"abstract": []}
    app._version_updates = {}
    app._relevance_scores = {}
    app._paper_summaries = {}
    app._summary_loading = set()
    app._summary_mode_label = {}
    app._match_scores = {}
    app._highlight_terms = {"abstract": []}
    app._local_browse_snapshot = None
    app._pending_detail_paper = None
    app._pending_detail_started_at = None
    app._show_abstract_preview = False
    app._detail_mode = "scan"
    app._current_date_index = 0
    app._history_files = []
    app._in_arxiv_api_mode = False
    app._active_bookmark_index = 0
    app._sort_index = 0
    return app
