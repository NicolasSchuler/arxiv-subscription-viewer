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

# ============================================================================
# Tests for clean_latex function
# ============================================================================


class TestExportMenuModal:
    """Tests for ExportMenuModal action methods and structure."""

    def test_bindings_cover_all_formats(self):
        from arxiv_browser.modals import ExportMenuModal

        binding_keys = {b.key for b in ExportMenuModal.BINDINGS}
        expected = {"escape", "q", "c", "b", "m", "r", "v", "t", "B", "R", "C"}
        assert expected <= binding_keys
        binding_descriptions = {b.key: b.description for b in ExportMenuModal.BINDINGS}
        assert binding_descriptions["t"] == "Markdown table"

    def test_action_cancel_dismisses_empty_string(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=3)
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with("")

    def test_action_do_clipboard_plain(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_plain()
        modal.dismiss.assert_called_once_with("clipboard-plain")

    def test_action_do_clipboard_bibtex(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_bibtex()
        modal.dismiss.assert_called_once_with("clipboard-bibtex")

    def test_action_do_clipboard_markdown(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_markdown()
        modal.dismiss.assert_called_once_with("clipboard-markdown")

    def test_action_do_clipboard_ris(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_ris()
        modal.dismiss.assert_called_once_with("clipboard-ris")

    def test_action_do_clipboard_csv(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_csv()
        modal.dismiss.assert_called_once_with("clipboard-csv")

    def test_action_do_clipboard_mdtable(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_clipboard_mdtable()
        modal.dismiss.assert_called_once_with("clipboard-mdtable")

    def test_action_do_file_bibtex(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_file_bibtex()
        modal.dismiss.assert_called_once_with("file-bibtex")

    def test_action_do_file_ris(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_file_ris()
        modal.dismiss.assert_called_once_with("file-ris")

    def test_action_do_file_csv(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()
        modal.action_do_file_csv()
        modal.dismiss.assert_called_once_with("file-csv")

    def test_paper_count_stored(self):
        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=5)
        assert modal._paper_count == 5

    def test_paper_count_plural_suffix(self):
        from arxiv_browser.modals import ExportMenuModal

        modal_one = ExportMenuModal(paper_count=1)
        assert modal_one._paper_count == 1

        modal_many = ExportMenuModal(paper_count=3)
        assert modal_many._paper_count == 3

    def test_all_action_dismiss_values_are_unique(self):
        """Ensure each export action produces a distinct format string."""
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ExportMenuModal

        modal = ExportMenuModal(paper_count=1)
        modal.dismiss = MagicMock()

        actions = [
            modal.action_do_clipboard_plain,
            modal.action_do_clipboard_bibtex,
            modal.action_do_clipboard_markdown,
            modal.action_do_clipboard_ris,
            modal.action_do_clipboard_csv,
            modal.action_do_clipboard_mdtable,
            modal.action_do_file_bibtex,
            modal.action_do_file_ris,
            modal.action_do_file_csv,
        ]
        values = []
        for action in actions:
            modal.dismiss.reset_mock()
            action()
            values.append(modal.dismiss.call_args[0][0])

        assert len(set(values)) == 9

    def test_compose_footer_uses_cancel_esc_copy(self):
        import inspect

        from arxiv_browser.modals import ExportMenuModal

        assert "Cancel: Esc/q" in inspect.getsource(ExportMenuModal.compose)

    def test_compose_uses_markdown_table_copy(self):
        import inspect

        from arxiv_browser.modals import ExportMenuModal

        source = inspect.getsource(ExportMenuModal.compose)
        assert "Markdown table" in source
        assert "Md table" not in source


class TestMetadataSnapshotPickerModal:
    """Tests for metadata snapshot selection modal behavior."""

    def test_action_choose_dismisses_highlighted_snapshot(self, tmp_path):
        from unittest.mock import MagicMock

        from textual.widgets import Label

        from arxiv_browser.modals.common import (
            MetadataSnapshotItem,
            MetadataSnapshotPickerModal,
        )

        snapshot = tmp_path / "arxiv-2026-03-07.json"
        modal = MetadataSnapshotPickerModal([snapshot])
        modal.dismiss = MagicMock()
        modal.query_one = MagicMock(
            return_value=type(
                "ListViewStub",
                (),
                {"highlighted_child": MetadataSnapshotItem(snapshot, Label("x"))},
            )()
        )

        modal.action_choose()

        modal.dismiss.assert_called_once_with(snapshot)

    def test_action_cancel_dismisses_none(self, tmp_path):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import MetadataSnapshotPickerModal

        modal = MetadataSnapshotPickerModal([tmp_path / "arxiv-2026-03-07.json"])
        modal.dismiss = MagicMock()

        modal.action_cancel()

        modal.dismiss.assert_called_once_with(None)

    def test_format_snapshot_label_includes_name_and_modified_time(self, tmp_path):
        from arxiv_browser.modals.common import MetadataSnapshotPickerModal

        snapshot = tmp_path / "arxiv-2026-03-07.json"
        snapshot.write_text("{}", encoding="utf-8")

        label = MetadataSnapshotPickerModal._format_snapshot_label(snapshot)

        assert snapshot.name in label
        assert "modified" in label

    def test_format_snapshot_label_handles_stat_errors(self, tmp_path, monkeypatch):
        from arxiv_browser.modals.common import MetadataSnapshotPickerModal

        snapshot = tmp_path / "arxiv-2026-03-07.json"
        snapshot.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(Path, "stat", lambda _self: (_ for _ in ()).throw(OSError("boom")))

        label = MetadataSnapshotPickerModal._format_snapshot_label(snapshot)

        assert snapshot.name in label
        assert "unknown time" in label

    def test_action_choose_empty_selection_dismisses_none(self, tmp_path):
        from unittest.mock import MagicMock

        from arxiv_browser.modals.common import MetadataSnapshotPickerModal

        modal = MetadataSnapshotPickerModal([tmp_path / "arxiv-2026-03-07.json"])
        modal.dismiss = MagicMock()
        modal.query_one = MagicMock(
            return_value=type("ListViewStub", (), {"highlighted_child": None})()
        )

        modal.action_choose()

        modal.dismiss.assert_called_once_with(None)


class TestSummaryModeModalDismiss:
    """Tests that each SummaryModeModal action dismisses with the correct mode name."""

    def test_action_mode_default_dismisses_default(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_default()
        modal.dismiss.assert_called_once_with("default")

    def test_action_mode_tldr_dismisses_tldr(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_tldr()
        modal.dismiss.assert_called_once_with("tldr")

    def test_action_mode_quick_dismisses_quick(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_quick()
        modal.dismiss.assert_called_once_with("quick")

    def test_action_mode_methods_dismisses_methods(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_methods()
        modal.dismiss.assert_called_once_with("methods")

    def test_action_mode_results_dismisses_results(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_results()
        modal.dismiss.assert_called_once_with("results")

    def test_action_mode_comparison_dismisses_comparison(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_mode_comparison()
        modal.dismiss.assert_called_once_with("comparison")

    def test_action_cancel_dismisses_empty(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with("")

    def test_all_modes_match_summary_modes_dict(self):
        """Verify each modal mode corresponds to a key in SUMMARY_MODES."""
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SummaryModeModal

        modal = SummaryModeModal()
        modal.dismiss = MagicMock()

        mode_actions = {
            "default": modal.action_mode_default,
            "quick": modal.action_mode_quick,
            "tldr": modal.action_mode_tldr,
            "methods": modal.action_mode_methods,
            "results": modal.action_mode_results,
            "comparison": modal.action_mode_comparison,
        }
        for mode_name, action in mode_actions.items():
            modal.dismiss.reset_mock()
            action()
            dismissed_value = modal.dismiss.call_args[0][0]
            assert dismissed_value == mode_name
            assert mode_name in SUMMARY_MODES


class TestResearchInterestsModalActions:
    """Tests for ResearchInterestsModal save/cancel dismiss behavior."""

    def test_action_cancel_dismisses_empty_string(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import ResearchInterestsModal

        modal = ResearchInterestsModal("some interests")
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with("")

    def test_initial_interests_stored(self):
        from arxiv_browser.modals import ResearchInterestsModal

        modal = ResearchInterestsModal("LLM quantization, speculative decoding")
        assert modal._current_interests == "LLM quantization, speculative decoding"

    def test_default_interests_empty(self):
        from arxiv_browser.modals import ResearchInterestsModal

        modal = ResearchInterestsModal()
        assert modal._current_interests == ""


class TestSectionToggleModal:
    """Tests for SectionToggleModal toggle and save/cancel behavior."""

    def test_init_stores_collapsed_as_set(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal(["authors", "abstract"])
        assert modal._collapsed == {"authors", "abstract"}

    def test_init_empty_collapsed(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        assert modal._collapsed == set()

    def test_toggle_adds_section_when_not_collapsed(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal._toggle("a")
        assert "authors" in modal._collapsed

    def test_toggle_removes_section_when_collapsed(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal(["authors"])
        modal._toggle("a")
        assert "authors" not in modal._collapsed

    def test_toggle_idempotent_double_toggle(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal._toggle("a")
        assert "authors" in modal._collapsed
        modal._toggle("a")
        assert "authors" not in modal._collapsed

    def test_toggle_invalid_key_ignored(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal._toggle("z")
        assert modal._collapsed == set()

    def test_action_toggle_a_toggles_authors(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_a()
        assert "authors" in modal._collapsed

    def test_action_toggle_b_toggles_abstract(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_b()
        assert "abstract" in modal._collapsed

    def test_action_toggle_t_toggles_tags(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_t()
        assert "tags" in modal._collapsed

    def test_action_toggle_r_toggles_relevance(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_r()
        assert "relevance" in modal._collapsed

    def test_action_toggle_s_toggles_summary(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_s()
        assert "summary" in modal._collapsed

    def test_action_toggle_e_toggles_s2(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_e()
        assert "s2" in modal._collapsed

    def test_action_toggle_h_toggles_hf(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_h()
        assert "hf" in modal._collapsed

    def test_action_toggle_v_toggles_version(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        modal.action_toggle_v()
        assert "version" in modal._collapsed

    def test_action_save_returns_sorted_collapsed(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal(["url", "authors", "hf"])
        modal.dismiss = MagicMock()
        modal.action_save()
        modal.dismiss.assert_called_once_with(sorted(["url", "authors", "hf"]))

    def test_action_save_after_toggle_reflects_changes(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal(["authors"])
        modal.action_toggle_a()
        modal.action_toggle_b()
        modal.dismiss = MagicMock()
        modal.action_save()
        result = modal.dismiss.call_args[0][0]
        assert "abstract" in result
        assert "authors" not in result

    def test_action_cancel_returns_none(self):
        from unittest.mock import MagicMock

        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal(["authors"])
        modal.dismiss = MagicMock()
        modal.action_cancel()
        modal.dismiss.assert_called_once_with(None)

    def test_render_list_shows_all_sections(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal([])
        rendered = modal._render_list()
        assert "Authors" in rendered
        assert "Abstract" in rendered
        assert "Tags" in rendered
        assert "Relevance" in rendered
        assert "AI Summary" in rendered
        assert "Semantic Scholar" in rendered
        assert "HuggingFace" in rendered
        assert "Version Update" in rendered
        # URL is no longer collapsible — should NOT be listed
        assert "URL" not in rendered

    def test_render_list_indicates_collapsed_state(self):
        from arxiv_browser.modals import SectionToggleModal

        modal = SectionToggleModal(["authors", "version"])
        rendered = modal._render_list()
        lines = rendered.split("\n")
        authors_line = next(line for line in lines if "Authors" in line)
        version_line = next(line for line in lines if "Version" in line)
        abstract_line = next(line for line in lines if "Abstract" in line)
        assert "\u25b8" in authors_line
        assert "collapsed" in authors_line
        assert "\u25b8" in version_line
        assert "\u25be" in abstract_line
        assert "expanded" in abstract_line

    def test_bindings_have_all_toggle_keys(self):
        from arxiv_browser.modals import SectionToggleModal

        binding_keys = {b.key for b in SectionToggleModal.BINDINGS}
        expected = {"escape", "q", "enter", "a", "b", "t", "r", "s", "e", "h", "v"}
        assert expected <= binding_keys

    def test_compose_footer_uses_cancel_esc_copy(self):
        import inspect

        from arxiv_browser.modals import SectionToggleModal

        assert "Cancel: Esc/q" in inspect.getsource(SectionToggleModal.compose)


class TestGetTargetPapers:
    """Tests for ArxivBrowser._get_target_papers selection logic."""

    def _make_mock_app(self, make_paper, papers=None, selected_ids=None):
        from unittest.mock import MagicMock

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None

        if papers is None:
            papers = [make_paper(arxiv_id=f"2401.{i:05d}") for i in range(3)]
        app.filtered_papers = papers
        app._papers_by_id = {p.arxiv_id: p for p in papers}
        app.selected_ids = selected_ids or set()
        app._get_current_paper = MagicMock(return_value=papers[0] if papers else None)

        return app

    def test_no_selection_returns_current_paper(self, make_paper):
        papers = [make_paper(arxiv_id="2401.00001", title="Paper 1")]
        app = self._make_mock_app(make_paper, papers=papers, selected_ids=set())

        result = app._get_target_papers()
        assert len(result) == 1
        assert result[0].arxiv_id == "2401.00001"

    def test_with_selection_returns_selected_papers(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00002"),
            make_paper(arxiv_id="2401.00003"),
        ]
        app = self._make_mock_app(
            make_paper,
            papers=papers,
            selected_ids={"2401.00001", "2401.00003"},
        )

        result = app._get_target_papers()
        assert len(result) == 2
        ids = [p.arxiv_id for p in result]
        assert "2401.00001" in ids
        assert "2401.00003" in ids

    def test_selection_preserves_list_order(self, make_paper):
        papers = [
            make_paper(arxiv_id="2401.00003"),
            make_paper(arxiv_id="2401.00001"),
            make_paper(arxiv_id="2401.00002"),
        ]
        app = self._make_mock_app(
            make_paper,
            papers=papers,
            selected_ids={"2401.00001", "2401.00002", "2401.00003"},
        )

        result = app._get_target_papers()
        result_ids = [p.arxiv_id for p in result]
        assert result_ids == ["2401.00003", "2401.00001", "2401.00002"]

    def test_no_selection_no_current_paper_returns_empty(self, make_paper):
        from unittest.mock import MagicMock

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app.filtered_papers = []
        app._papers_by_id = {}
        app.selected_ids = set()
        app._get_current_paper = MagicMock(return_value=None)

        result = app._get_target_papers()
        assert result == []

    def test_selected_id_not_in_filtered_still_included(self, make_paper):
        """Papers selected but then filtered out should still be returned."""
        visible = [make_paper(arxiv_id="2401.00001")]
        hidden = make_paper(arxiv_id="2401.00099")

        app = self._make_mock_app(
            make_paper,
            papers=visible,
            selected_ids={"2401.00001", "2401.00099"},
        )
        app._papers_by_id["2401.00099"] = hidden

        result = app._get_target_papers()
        result_ids = [p.arxiv_id for p in result]
        assert "2401.00001" in result_ids
        assert "2401.00099" in result_ids


class TestToggleReadStar:
    """Tests for action_toggle_read and action_toggle_star via mock app."""

    def _make_mock_app(self, make_paper, papers=None):
        from unittest.mock import MagicMock

        from arxiv_browser.browser.core import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig()
        app.selected_ids = set()

        if papers is None:
            papers = [make_paper()]
        app.filtered_papers = papers
        app._papers_by_id = {p.arxiv_id: p for p in papers}

        app._get_current_paper = MagicMock(return_value=papers[0] if papers else None)
        app._get_current_index = MagicMock(return_value=0 if papers else None)
        app._update_option_at_index = MagicMock()
        app.notify = MagicMock()

        return app

    def test_toggle_read_creates_metadata_and_sets_read(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_read()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.is_read is True
        app.notify.assert_called_once()
        assert "read" in app.notify.call_args[0][0]

    def test_toggle_read_twice_unsets_read(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_read()
        app.action_toggle_read()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.is_read is False

    def test_toggle_read_calls_update_option(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_read()
        app._update_option_at_index.assert_called_once_with(0)

    def test_toggle_read_no_paper_does_nothing(self, make_paper):
        app = self._make_mock_app(make_paper, papers=[make_paper()])
        app._get_current_paper = lambda: None

        app.action_toggle_read()
        assert app._config.paper_metadata == {}

    def test_toggle_star_creates_metadata_and_sets_starred(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_star()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.starred is True
        app.notify.assert_called_once()
        assert "starred" in app.notify.call_args[0][0]

    def test_toggle_star_twice_unsets_star(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_star()
        app.action_toggle_star()

        meta = app._config.paper_metadata["2401.00001"]
        assert meta.starred is False

    def test_toggle_star_calls_update_option(self, make_paper):
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])

        app.action_toggle_star()
        app._update_option_at_index.assert_called_once_with(0)

    def test_toggle_star_no_paper_does_nothing(self, make_paper):
        app = self._make_mock_app(make_paper, papers=[make_paper()])
        app._get_current_paper = lambda: None

        app.action_toggle_star()
        assert app._config.paper_metadata == {}

    def test_toggle_read_with_none_index_skips_option_update(self, make_paper):
        """When _get_current_index returns None, _update_option_at_index is not called."""
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])
        app._get_current_index = lambda: None

        app.action_toggle_read()
        app._update_option_at_index.assert_not_called()
        assert app._config.paper_metadata["2401.00001"].is_read is True

    def test_toggle_star_with_none_index_skips_option_update(self, make_paper):
        """When _get_current_index returns None, _update_option_at_index is not called."""
        paper = make_paper(arxiv_id="2401.00001")
        app = self._make_mock_app(make_paper, papers=[paper])
        app._get_current_index = lambda: None

        app.action_toggle_star()
        app._update_option_at_index.assert_not_called()
        assert app._config.paper_metadata["2401.00001"].starred is True


class TestFormatPaperForClipboardExtended:
    """Extended tests for format_paper_for_clipboard output formatting."""

    def test_includes_all_metadata_fields(self, make_paper):
        from arxiv_browser.export import format_paper_for_clipboard

        paper = make_paper(
            arxiv_id="2401.12345",
            title="A Great Paper",
            authors="Alice, Bob",
            date="Mon, 15 Jan 2024",
            categories="cs.AI cs.LG",
            comments="10 pages, 5 figures",
        )
        result = format_paper_for_clipboard(paper, abstract_text="The abstract text.")
        assert "Title: A Great Paper" in result
        assert "Authors: Alice, Bob" in result
        assert "arXiv: 2401.12345" in result
        assert "Date: Mon, 15 Jan 2024" in result
        assert "Categories: cs.AI cs.LG" in result
        assert "Comments: 10 pages, 5 figures" in result
        assert "URL: https://arxiv.org/abs/2401.12345" in result
        assert "Abstract: The abstract text." in result

    def test_omits_comments_when_none(self, make_paper):
        from arxiv_browser.export import format_paper_for_clipboard

        paper = make_paper(comments=None)
        result = format_paper_for_clipboard(paper)
        assert "Comments:" not in result

    def test_empty_abstract_still_has_label(self, make_paper):
        from arxiv_browser.export import format_paper_for_clipboard

        paper = make_paper()
        result = format_paper_for_clipboard(paper, abstract_text="")
        assert "Abstract: " in result


class TestFormatPaperAsMarkdownExtended:
    """Extended tests for format_paper_as_markdown output formatting."""

    def test_output_has_markdown_structure(self, make_paper):
        from arxiv_browser.export import format_paper_as_markdown

        paper = make_paper(
            arxiv_id="2401.12345",
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            date="Mon, 15 Jan 2024",
            categories="cs.CL",
        )
        result = format_paper_as_markdown(paper, abstract_text="We propose Transformer.")
        assert result.startswith("## Attention Is All You Need")
        assert "**arXiv:** [2401.12345](https://arxiv.org/abs/2401.12345)" in result
        assert "**Date:** Mon, 15 Jan 2024" in result
        assert "**Categories:** cs.CL" in result
        assert "**Authors:** Vaswani et al." in result
        assert "### Abstract" in result
        assert "We propose Transformer." in result

    def test_includes_comments_when_present(self, make_paper):
        from arxiv_browser.export import format_paper_as_markdown

        paper = make_paper(comments="Accepted at NeurIPS 2024")
        result = format_paper_as_markdown(paper)
        assert "**Comments:** Accepted at NeurIPS 2024" in result

    def test_omits_comments_when_none(self, make_paper):
        from arxiv_browser.export import format_paper_as_markdown

        paper = make_paper(comments=None)
        result = format_paper_as_markdown(paper)
        assert "**Comments:**" not in result


class TestFormatPaperAsBibtexExtended:
    """Extended tests for format_paper_as_bibtex output formatting."""

    def test_bibtex_uses_misc_type(self, make_paper):

        paper = make_paper()
        result = format_paper_as_bibtex(paper)
        assert result.startswith("@misc{")

    def test_bibtex_contains_required_fields(self, make_paper):

        paper = make_paper(
            arxiv_id="2401.12345",
            title="Deep Learning",
            authors="John Smith",
            date="Mon, 15 Jan 2024",
            categories="cs.AI cs.LG",
        )
        result = format_paper_as_bibtex(paper)
        assert "title = {Deep Learning}" in result
        assert "author = {John Smith}" in result
        assert "year = {2024}" in result
        assert "eprint = {2401.12345}" in result
        assert "archivePrefix = {arXiv}" in result
        assert "primaryClass = {cs.AI}" in result
        assert "url = {https://arxiv.org/abs/2401.12345}" in result

    def test_bibtex_ends_with_closing_brace(self, make_paper):

        paper = make_paper()
        result = format_paper_as_bibtex(paper)
        assert result.strip().endswith("}")

    def test_bibtex_escapes_special_chars(self, make_paper):

        paper = make_paper(title="NLP & Transformers: 100% Better")
        result = format_paper_as_bibtex(paper)
        assert r"NLP \& Transformers: 100\% Better" in result

    def test_bibtex_primary_class_from_first_category(self, make_paper):

        paper = make_paper(categories="stat.ML cs.LG")
        result = format_paper_as_bibtex(paper)
        assert "primaryClass = {stat.ML}" in result


class TestFormatPaperAsRisExtended:
    """Extended tests for format_paper_as_ris output formatting."""

    def test_ris_has_correct_record_type(self, make_paper):

        paper = make_paper()
        result = format_paper_as_ris(paper)
        assert result.startswith("TY  - ELEC")

    def test_ris_ends_with_end_record(self, make_paper):

        paper = make_paper()
        result = format_paper_as_ris(paper)
        assert result.strip().endswith("ER  -")

    def test_ris_includes_title_and_url(self, make_paper):

        paper = make_paper(arxiv_id="2401.12345", title="My Paper")
        result = format_paper_as_ris(paper)
        assert "TI  - My Paper" in result
        assert "UR  - https://arxiv.org/abs/2401.12345" in result

    def test_ris_splits_authors_by_comma(self, make_paper):

        paper = make_paper(authors="Alice Smith, Bob Jones, Charlie Brown")
        result = format_paper_as_ris(paper)
        assert "AU  - Alice Smith" in result
        assert "AU  - Bob Jones" in result
        assert "AU  - Charlie Brown" in result

    def test_ris_includes_abstract_when_provided(self, make_paper):

        paper = make_paper()
        result = format_paper_as_ris(paper, abstract_text="This is the abstract.")
        assert "AB  - This is the abstract." in result

    def test_ris_omits_abstract_when_empty(self, make_paper):

        paper = make_paper()
        result = format_paper_as_ris(paper)
        assert "AB  -" not in result

    def test_ris_includes_comments_as_note(self, make_paper):

        paper = make_paper(comments="Accepted at ICML")
        result = format_paper_as_ris(paper)
        assert "N2  - Accepted at ICML" in result

    def test_ris_omits_comments_note_when_none(self, make_paper):

        paper = make_paper(comments=None)
        result = format_paper_as_ris(paper)
        assert "N2  -" not in result

    def test_ris_includes_categories_as_keywords(self, make_paper):

        paper = make_paper(categories="cs.AI cs.LG stat.ML")
        result = format_paper_as_ris(paper)
        assert "KW  - cs.AI" in result
        assert "KW  - cs.LG" in result
        assert "KW  - stat.ML" in result

    def test_ris_includes_arxiv_note(self, make_paper):

        paper = make_paper(arxiv_id="2401.12345")
        result = format_paper_as_ris(paper)
        assert "N1  - arXiv:2401.12345" in result


class TestFormatPapersAsCsvExtended:
    """Extended tests for format_papers_as_csv output formatting."""

    def test_csv_header_without_metadata(self, make_paper):

        papers = [make_paper()]
        result = format_papers_as_csv(papers)
        lines = result.strip().split("\n")
        header = lines[0]
        assert "arxiv_id" in header
        assert "title" in header
        assert "authors" in header
        assert "categories" in header
        assert "date" in header
        assert "url" in header
        assert "comments" in header
        assert "starred" not in header
        assert "read" not in header
        assert "tags" not in header
        assert "notes" not in header

    def test_csv_header_with_metadata(self, make_paper):

        papers = [make_paper()]
        result = format_papers_as_csv(papers, metadata={})
        lines = result.strip().split("\n")
        header = lines[0]
        assert "starred" in header
        assert "read" in header
        assert "tags" in header
        assert "notes" in header

    def test_csv_multiple_papers(self, make_paper):

        papers = [
            make_paper(arxiv_id="2401.00001", title="Paper A"),
            make_paper(arxiv_id="2401.00002", title="Paper B"),
        ]
        result = format_papers_as_csv(papers)
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_csv_with_metadata_values(self, make_paper):

        paper = make_paper(arxiv_id="2401.00001")
        meta = PaperMetadata(
            arxiv_id="2401.00001",
            starred=True,
            is_read=True,
            tags=["ml", "topic:transformers"],
            notes="Important paper",
        )
        result = format_papers_as_csv([paper], metadata={"2401.00001": meta})
        assert "true" in result
        assert "ml;topic:transformers" in result
        assert "Important paper" in result

    def test_csv_with_metadata_no_match(self, make_paper):

        paper = make_paper(arxiv_id="2401.00001")
        result = format_papers_as_csv([paper], metadata={})
        lines = result.strip().split("\n")
        data_line = lines[1]
        assert "false" in data_line

    def test_csv_escapes_commas_in_fields(self, make_paper):
        import csv
        import io

        paper = make_paper(title='Title with "quotes" and, commas')
        result = format_papers_as_csv([paper])
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[1][1] == 'Title with "quotes" and, commas'


class TestFormatPapersAsMarkdownTableExtended:
    """Extended tests for format_papers_as_markdown_table output."""

    def test_markdown_table_header(self, make_paper):

        papers = [make_paper()]
        result = format_papers_as_markdown_table(papers)
        lines = result.strip().split("\n")
        assert "| arXiv ID | Title | Authors | Categories | Date |" in lines[0]
        assert lines[1].startswith("|---")

    def test_markdown_table_arxiv_link(self, make_paper):

        paper = make_paper(arxiv_id="2401.12345")
        result = format_papers_as_markdown_table([paper])
        assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in result

    def test_markdown_table_truncates_many_authors(self, make_paper):

        paper = make_paper(authors="Alice, Bob, Charlie, Diana, Eve")
        result = format_papers_as_markdown_table([paper])
        assert "Alice et al." in result
        assert "Eve" not in result

    def test_markdown_table_shows_few_authors(self, make_paper):

        paper = make_paper(authors="Alice, Bob, Charlie")
        result = format_papers_as_markdown_table([paper])
        assert "Alice, Bob, Charlie" in result

    def test_markdown_table_escapes_pipes(self, make_paper):

        paper = make_paper(title="A | B | C")
        result = format_papers_as_markdown_table([paper])
        assert "A \\| B \\| C" in result

    def test_markdown_table_empty_list(self, make_paper):

        result = format_papers_as_markdown_table([])
        lines = result.strip().split("\n")
        assert len(lines) == 2
