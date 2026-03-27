"""Focused tests for LLM command trust, shell policy, and config serialization."""

from __future__ import annotations

import pytest

from tests.support.canonical_exports import LLM_PRESETS, UserConfig


class TestRequireLlmCommand:
    """Verify _require_llm_command helper notifies when LLM is not configured."""

    def _make_mock_app(self, **config_kwargs):
        from unittest.mock import MagicMock

        from tests.support.canonical_exports import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._http_client = None
        app._config = UserConfig(**config_kwargs)
        app.notify = MagicMock()
        return app

    def test_returns_command_when_configured(self):
        app = self._make_mock_app(llm_command="my-tool {prompt}")
        result = app._require_llm_command()
        assert result == "my-tool {prompt}"
        app.notify.assert_not_called()

    def test_refreshes_provider_with_configured_retry_policy(self):
        app = self._make_mock_app(
            llm_command="my-tool {prompt}",
            allow_llm_shell_fallback=False,
            llm_max_retries=3,
        )

        result = app._require_llm_command()

        assert result == "my-tool {prompt}"
        assert app._llm_provider is not None
        assert app._llm_provider._allow_shell is False
        assert app._llm_provider._max_retries == 3

    def test_returns_none_and_notifies_when_not_configured(self):
        app = self._make_mock_app()
        result = app._require_llm_command()
        assert result is None
        app.notify.assert_called_once()
        assert "LLM not configured" in str(app.notify.call_args)

    def test_returns_none_with_unknown_preset(self):
        app = self._make_mock_app(llm_preset="nonexistent")
        result = app._require_llm_command()
        assert result is None
        call_args_str = str(app.notify.call_args)
        assert "Unknown preset" in call_args_str

    def test_blocks_shell_template_when_shell_fallback_disabled(self):
        app = self._make_mock_app(
            llm_command="echo {prompt} | cat",
            allow_llm_shell_fallback=False,
        )
        result = app._require_llm_command()
        assert result is None
        call_args_str = str(app.notify.call_args)
        assert "allow_llm_shell_fallback" in call_args_str
        assert "LLM command blocked" in call_args_str


class TestCommandTrustGuardrails:
    """Tests for one-time trust prompts for external commands."""

    def _make_mock_app(self, **config_kwargs):
        from unittest.mock import MagicMock

        from tests.support.canonical_exports import ArxivBrowser

        app = ArxivBrowser.__new__(ArxivBrowser)
        app._config = UserConfig(**config_kwargs)
        app.notify = MagicMock()
        return app

    def test_llm_preset_is_auto_trusted(self):
        app = self._make_mock_app(llm_preset="copilot")
        command_template = LLM_PRESETS["copilot"]
        assert app._is_llm_command_trusted(command_template) is True

    def test_custom_llm_command_trusted_by_hash(self):
        app = self._make_mock_app(llm_command="custom-tool {prompt}")
        command_template = "custom-tool {prompt}"
        cmd_hash = app._trust_hash(command_template)
        app._config.trusted_llm_command_hashes = [cmd_hash]
        assert app._is_llm_command_trusted(command_template) is True

    def test_arxiv_browser_uses_shared_ui_constants(self):
        from arxiv_browser.ui_constants import APP_BINDINGS, APP_CSS
        from tests.support.canonical_exports import ArxivBrowser

        assert ArxivBrowser.BINDINGS is APP_BINDINGS
        assert ArxivBrowser.CSS == APP_CSS

    def test_ensure_llm_trusted_prompts_and_persists(self, monkeypatch):
        from unittest.mock import MagicMock

        app = self._make_mock_app(llm_command="custom-tool {prompt}")
        on_trusted = MagicMock()
        command_template = "custom-tool {prompt}"

        monkeypatch.setattr("arxiv_browser.actions.llm_actions.save_config", lambda _config: True)

        def fake_push_screen(_modal, callback):
            callback(True)

        app.push_screen = fake_push_screen

        trusted_now = app._ensure_llm_command_trusted(command_template, on_trusted)
        assert trusted_now is False
        on_trusted.assert_called_once()
        cmd_hash = app._trust_hash(command_template)
        assert cmd_hash in app._config.trusted_llm_command_hashes

    def test_ensure_pdf_viewer_trusted_prompts_and_persists(self, monkeypatch):
        from unittest.mock import MagicMock

        app = self._make_mock_app()
        viewer_cmd = "open -a Preview {path}"
        on_trusted = MagicMock()

        monkeypatch.setattr("arxiv_browser.actions.llm_actions.save_config", lambda _config: True)

        def fake_push_screen(_modal, callback):
            callback(True)

        app.push_screen = fake_push_screen

        trusted_now = app._ensure_pdf_viewer_trusted(viewer_cmd, on_trusted)
        assert trusted_now is False
        on_trusted.assert_called_once()
        cmd_hash = app._trust_hash(viewer_cmd)
        assert cmd_hash in app._config.trusted_pdf_viewer_hashes

    def test_ensure_llm_trusted_cancel_callback_does_not_persist_or_run(self, monkeypatch):
        from unittest.mock import MagicMock

        app = self._make_mock_app(llm_command="custom-tool {prompt}")
        on_trusted = MagicMock()
        command_template = "custom-tool {prompt}"

        monkeypatch.setattr("arxiv_browser.actions.llm_actions.save_config", lambda _config: True)

        def fake_push_screen(_modal, callback):
            callback(False)

        app.push_screen = fake_push_screen

        trusted_now = app._ensure_llm_command_trusted(command_template, on_trusted)
        assert trusted_now is False
        on_trusted.assert_not_called()
        assert app._config.trusted_llm_command_hashes == []
        assert "LLM command cancelled" in str(app.notify.call_args)

    def test_ensure_llm_trusted_warns_when_trust_persistence_fails(self, monkeypatch):
        from unittest.mock import MagicMock

        app = self._make_mock_app(llm_command="custom-tool {prompt}")
        on_trusted = MagicMock()
        command_template = "custom-tool {prompt}"

        monkeypatch.setattr("arxiv_browser.actions.llm_actions.save_config", lambda _config: False)

        def fake_push_screen(_modal, callback):
            callback(True)

        app.push_screen = fake_push_screen

        trusted_now = app._ensure_llm_command_trusted(command_template, on_trusted)
        assert trusted_now is False
        on_trusted.assert_called_once()
        assert app._config.trusted_llm_command_hashes == [app._trust_hash(command_template)]
        assert "session only" in str(app.notify.call_args)

    def test_ensure_llm_trusted_bypasses_prompt_when_already_trusted(self):
        from unittest.mock import MagicMock

        command_template = "custom-tool {prompt}"
        trusted_hash = self._make_mock_app()._trust_hash(command_template)
        app = self._make_mock_app(
            llm_command=command_template,
            trusted_llm_command_hashes=[trusted_hash],
        )
        app.push_screen = MagicMock()

        trusted_now = app._ensure_llm_command_trusted(command_template, MagicMock())

        assert trusted_now is True
        app.push_screen.assert_not_called()

    def test_ensure_llm_trusted_cancels_when_prompt_unavailable(self):
        from unittest.mock import MagicMock

        from textual.app import ScreenStackError

        app = self._make_mock_app(llm_command="custom-tool {prompt}")
        app.push_screen = MagicMock(side_effect=ScreenStackError("no screen"))
        on_trusted = MagicMock()

        trusted_now = app._ensure_llm_command_trusted("custom-tool {prompt}", on_trusted)

        assert trusted_now is False
        on_trusted.assert_not_called()
        assert "action cancelled" in str(app.notify.call_args)

    def test_ensure_pdf_viewer_trusted_cancels_when_prompt_unavailable(self):
        from unittest.mock import MagicMock

        from textual.app import ScreenStackError

        app = self._make_mock_app()
        app.push_screen = MagicMock(side_effect=ScreenStackError("no screen"))
        on_trusted = MagicMock()

        trusted_now = app._ensure_pdf_viewer_trusted("open -a Preview {path}", on_trusted)

        assert trusted_now is False
        on_trusted.assert_not_called()
        assert "action cancelled" in str(app.notify.call_args)


class TestBuildLlmShellCommand:
    """Tests for shell command building with proper escaping."""

    def test_basic(self):
        from tests.support.canonical_exports import _build_llm_shell_command

        result = _build_llm_shell_command("claude -p {prompt}", "hello world")
        assert "claude -p" in result
        assert "hello world" in result

    def test_prompt_with_quotes(self):
        from tests.support.canonical_exports import _build_llm_shell_command

        result = _build_llm_shell_command("claude -p {prompt}", 'text with "quotes"')
        assert "claude -p" in result
        assert "quotes" in result

    def test_prompt_with_shell_chars(self):
        import shlex

        from tests.support.canonical_exports import _build_llm_shell_command

        dangerous = "text; rm -rf /"
        result = _build_llm_shell_command("llm {prompt}", dangerous)
        assert shlex.quote(dangerous) in result
        assert result == f"llm {shlex.quote(dangerous)}"

    def test_missing_prompt_placeholder(self):
        from tests.support.canonical_exports import _build_llm_shell_command

        with pytest.raises(ValueError, match=r"must contain.*\{prompt\}"):
            _build_llm_shell_command("claude", "hello")

    def test_windows_prompt_uses_list2cmdline(self):
        from unittest.mock import patch

        from tests.support.canonical_exports import _build_llm_shell_command

        with (
            patch("arxiv_browser.llm.os.name", "nt"),
            patch(
                "arxiv_browser.llm.subprocess.list2cmdline", return_value='"safe ^& prompt"'
            ) as quote_mock,
        ):
            result = _build_llm_shell_command("llm {prompt}", "safe ^& prompt")

        quote_mock.assert_called_once_with(["safe ^& prompt"])
        assert result == 'llm "safe ^& prompt"'


class TestCommandHash:
    """Tests for command hash computation."""

    def test_same_input_same_hash(self):
        from tests.support.canonical_exports import _compute_command_hash

        h1 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        h2 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        assert h1 == h2

    def test_different_command_different_hash(self):
        from tests.support.canonical_exports import _compute_command_hash

        h1 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        h2 = _compute_command_hash("llm {prompt}", "Summarize: {title}")
        assert h1 != h2

    def test_different_prompt_different_hash(self):
        from tests.support.canonical_exports import _compute_command_hash

        h1 = _compute_command_hash("claude -p {prompt}", "Summarize: {title}")
        h2 = _compute_command_hash("claude -p {prompt}", "Explain: {title}")
        assert h1 != h2

    def test_hash_length(self):
        from tests.support.canonical_exports import _compute_command_hash

        h = _compute_command_hash("cmd", "prompt")
        assert len(h) == 16


class TestLlmPresets:
    """Tests for LLM preset definitions."""

    def test_presets_exist(self):
        assert "claude" in LLM_PRESETS
        assert "codex" in LLM_PRESETS
        assert "llm" in LLM_PRESETS
        assert "copilot" in LLM_PRESETS

    def test_presets_have_prompt_placeholder(self):
        for name, cmd in LLM_PRESETS.items():
            assert "{prompt}" in cmd, f"Preset {name!r} missing {{prompt}} placeholder"


class TestLlmConfigSerialization:
    """Tests for LLM config fields roundtrip."""

    def test_roundtrip(self):
        from tests.support.canonical_exports import _config_to_dict, _dict_to_config

        config = UserConfig(
            llm_command="claude -p {prompt}",
            llm_prompt_template="Summarize: {title}",
            llm_preset="claude",
            allow_llm_shell_fallback=False,
            trusted_llm_command_hashes=["abc123"],
            trusted_pdf_viewer_hashes=["def456"],
        )
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.llm_command == "claude -p {prompt}"
        assert restored.llm_prompt_template == "Summarize: {title}"
        assert restored.llm_preset == "claude"
        assert restored.allow_llm_shell_fallback is False
        assert restored.trusted_llm_command_hashes == ["abc123"]
        assert restored.trusted_pdf_viewer_hashes == ["def456"]

    def test_missing_fields_default(self):
        from tests.support.canonical_exports import _dict_to_config

        config = _dict_to_config({})
        assert config.llm_command == ""
        assert config.llm_prompt_template == ""
        assert config.llm_preset == ""
        assert config.allow_llm_shell_fallback is True
        assert config.trusted_llm_command_hashes == []
        assert config.trusted_pdf_viewer_hashes == []
