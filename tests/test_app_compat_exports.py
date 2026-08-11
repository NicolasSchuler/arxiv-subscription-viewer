"""Dedicated compatibility tests for the public ``arxiv_browser.app`` surface."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest


class TestAppCompatibilityExports:
    def test_app_exports_are_explicit_and_importable(self) -> None:
        import arxiv_browser.app as app_module
        from arxiv_browser.app import __all__

        assert __all__ == [
            "ArxivBrowser",
            "ArxivBrowserOptions",
            "_configure_color_mode",
            "_configure_logging",
            "discover_history_files",
            "load_config",
            "main",
            "_fetch_paper_content_async",
            "_resolve_papers",
            "_validate_interactive_tty",
        ]
        for name in __all__:
            assert hasattr(app_module, name), f"{name} not found in arxiv_browser.app"

    @pytest.mark.parametrize(
        ("name", "module_name", "attr_name"),
        [
            ("ArxivBrowser", "arxiv_browser.browser.core", "ArxivBrowser"),
            ("ArxivBrowserOptions", "arxiv_browser.browser.core", "ArxivBrowserOptions"),
            ("_configure_color_mode", "arxiv_browser.cli", "_configure_color_mode"),
            ("_configure_logging", "arxiv_browser.cli", "_configure_logging"),
            ("discover_history_files", "arxiv_browser.parsing", "discover_history_files"),
            ("load_config", "arxiv_browser.config", "load_config"),
            ("_resolve_papers", "arxiv_browser.cli", "_resolve_papers"),
            ("_validate_interactive_tty", "arxiv_browser.cli", "_validate_interactive_tty"),
        ],
    )
    def test_app_exports_resolve_to_canonical_symbols(
        self,
        name: str,
        module_name: str,
        attr_name: str,
    ) -> None:
        import arxiv_browser.app as app_module

        canonical = getattr(importlib.import_module(module_name), attr_name)

        resolved = app_module.__getattr__(name)

        assert resolved is canonical
        assert getattr(app_module, name) is canonical

    @pytest.mark.parametrize(
        ("name", "module_name", "attr_name"),
        [
            ("Paper", "arxiv_browser.models", "Paper"),
            ("ArxivBrowser", "arxiv_browser.browser.core", "ArxivBrowser"),
            ("UserConfig", "arxiv_browser.models", "UserConfig"),
            ("main", "arxiv_browser.cli", "main"),
        ],
    )
    def test_root_package_exports_resolve_to_canonical_symbols(
        self,
        name: str,
        module_name: str,
        attr_name: str,
    ) -> None:
        import arxiv_browser

        canonical = getattr(importlib.import_module(module_name), attr_name)

        assert getattr(arxiv_browser, name) is canonical

    def test_root_package_removed_legacy_extras_fail_cleanly(self) -> None:
        package = importlib.import_module("arxiv_browser")

        with pytest.raises(AttributeError):
            package.__getattr__("DEFAULT_THEME")

        with pytest.raises(AttributeError):
            package.__getattr__("highlight_text")

    def test_app_fetch_paper_content_async_uses_compat_patch_surface(self, monkeypatch) -> None:
        import arxiv_browser.app as app_module

        class _Response:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

        class _Client:
            def __init__(self, response: _Response) -> None:
                self.response = response
                self.calls: list[tuple[str, int, bool]] = []

            async def get(self, url: str, *, timeout: int, follow_redirects: bool):
                self.calls.append((url, timeout, follow_redirects))
                return self.response

        class _TempClient:
            def __init__(self, response: _Response) -> None:
                self.response = response
                self.calls: list[tuple[str, int, bool]] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url: str, *, timeout: int, follow_redirects: bool):
                self.calls.append((url, timeout, follow_redirects))
                return self.response

        paper = SimpleNamespace(
            arxiv_id="2401.99991",
            abstract="Fallback abstract.",
            abstract_raw="Fallback abstract.",
        )

        client = _Client(_Response(200, "<p>x</p>"))
        fetch_paper_content = cast(Any, app_module._fetch_paper_content_async)
        monkeypatch.setattr(
            app_module, "extract_text_from_html", lambda _html: "abcdef", raising=False
        )
        text = asyncio.run(fetch_paper_content(paper, client=client, timeout=4))
        assert text == "abcdef"
        assert client.calls == [("https://arxiv.org/html/2401.99991", 4, True)]

        compat_logger = MagicMock()
        temp_client = _TempClient(_Response(404, ""))
        monkeypatch.delattr(app_module, "extract_text_from_html", raising=False)
        monkeypatch.setattr(app_module, "logger", compat_logger)
        with patch(
            "arxiv_browser.app.httpx.AsyncClient",
            return_value=temp_client,
        ):
            text = asyncio.run(fetch_paper_content(paper))
        assert text == "Abstract:\nFallback abstract."
        assert temp_client.calls == [
            (
                "https://arxiv.org/html/2401.99991",
                app_module.ARXIV_HTML_TIMEOUT,
                True,
            )
        ]
        compat_logger.warning.assert_called_once_with(
            "arXiv HTML fetch returned %d for %s",
            404,
            "2401.99991",
        )

    def test_app_fetch_paper_content_async_preserves_error_and_parser_fallbacks(
        self, monkeypatch
    ) -> None:
        import arxiv_browser.app as app_module

        class _Response:
            status_code = 200
            text = "<p>x</p>"

        class _Client:
            def __init__(self, *, error: Exception | None = None) -> None:
                self.error = error

            async def get(self, *_args, **_kwargs):
                if self.error is not None:
                    raise self.error
                return _Response()

        fetch_paper_content = cast(Any, app_module._fetch_paper_content_async)
        paper = SimpleNamespace(
            arxiv_id="2401.99993",
            abstract=None,
            abstract_raw="Raw fallback.",
        )
        compat_logger = MagicMock()
        monkeypatch.setattr(app_module, "logger", compat_logger)

        text = asyncio.run(
            fetch_paper_content(paper, client=_Client(error=httpx.HTTPError("boom")))
        )
        assert text == "Abstract:\nRaw fallback."
        compat_logger.warning.assert_called_once_with(
            "Failed to fetch HTML for %s",
            "2401.99993",
            exc_info=True,
        )

        monkeypatch.delattr(app_module, "extract_text_from_html", raising=False)
        monkeypatch.setattr(app_module, "_extract_text_from_html", lambda _html: "abcdefgh")
        monkeypatch.setattr(app_module, "MAX_PAPER_CONTENT_LENGTH", 4)
        text = asyncio.run(fetch_paper_content(paper, client=_Client()))
        assert text == "abcd"

    def test_app_fetch_paper_content_async_delegates_to_canonical_owner(self, monkeypatch) -> None:
        import arxiv_browser.app as app_module
        import arxiv_browser.browser.content as browser_content

        captured_requests: list[Any] = []

        async def fetch_legacy(request: Any) -> str:
            captured_requests.append(request)
            return "delegated"

        parser = MagicMock()
        client_factory = MagicMock()
        compat_logger = MagicMock()
        to_thread = MagicMock()
        client = MagicMock()
        paper = SimpleNamespace(arxiv_id="2401.99992", abstract="", abstract_raw="")

        monkeypatch.setattr(browser_content, "_fetch_legacy_paper_content", fetch_legacy)
        monkeypatch.setattr(app_module, "extract_text_from_html", parser, raising=False)
        monkeypatch.setattr(app_module.httpx, "AsyncClient", client_factory)
        monkeypatch.setattr(app_module, "logger", compat_logger)
        monkeypatch.setattr(app_module, "asyncio", SimpleNamespace(to_thread=to_thread))
        monkeypatch.setattr(app_module, "ARXIV_HTML_TIMEOUT", 13)
        monkeypatch.setattr(app_module, "MAX_PAPER_CONTENT_LENGTH", 17)

        result = asyncio.run(cast(Any, app_module._fetch_paper_content_async)(paper, client=client))

        assert result == "delegated"
        assert len(captured_requests) == 1
        request = captured_requests[0]
        assert request.paper is paper
        assert request.client is client
        assert request.timeout == 13
        assert request.max_content_length == 17
        assert request.extract_html is parser
        assert request.client_factory is client_factory
        assert request.to_thread is to_thread
        assert request.log is compat_logger

    def test_app_getattr_dir_and_missing_attr(self) -> None:
        import arxiv_browser.app as app_module

        assert "load_config" in app_module.__dir__()
        assert "highlight_text" not in app_module.__dir__()

        with pytest.raises(AttributeError):
            app_module.__getattr__("highlight_text")

    def test_app_main_uses_compatibility_resolved_dependencies(self, monkeypatch) -> None:
        import arxiv_browser.app as app_module
        import arxiv_browser.cli as cli_module

        load_config = MagicMock()
        discover_history_files = MagicMock()
        resolve_papers = MagicMock()
        configure_logging = MagicMock()
        configure_color_mode = MagicMock()
        validate_interactive_tty = MagicMock()
        app_factory = MagicMock()

        monkeypatch.setattr(app_module, "load_config", load_config, raising=False)
        monkeypatch.setattr(
            app_module,
            "discover_history_files",
            discover_history_files,
            raising=False,
        )
        monkeypatch.setattr(app_module, "_resolve_papers", resolve_papers, raising=False)
        monkeypatch.setattr(app_module, "_configure_logging", configure_logging, raising=False)
        monkeypatch.setattr(
            app_module, "_configure_color_mode", configure_color_mode, raising=False
        )
        monkeypatch.setattr(
            app_module,
            "_validate_interactive_tty",
            validate_interactive_tty,
            raising=False,
        )
        monkeypatch.setattr(app_module, "ArxivBrowser", app_factory, raising=False)

        with patch.object(cli_module, "main", return_value=7) as main_mock:
            result = app_module.main()

        assert result == 7
        deps = main_mock.call_args.kwargs["deps"]
        assert deps.load_config_fn is load_config
        assert deps.discover_history_files_fn is discover_history_files
        assert deps.resolve_papers_fn is resolve_papers
        assert deps.configure_logging_fn is configure_logging
        assert deps.configure_color_mode_fn is configure_color_mode
        assert deps.validate_interactive_tty_fn is validate_interactive_tty
        assert deps.app_factory is app_factory
