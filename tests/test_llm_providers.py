"""Tests for the LLM provider abstraction layer."""

from __future__ import annotations

from arxiv_browser.llm_providers import CLIProvider, LLMProvider, LLMResult, resolve_provider

# ============================================================================
# LLMResult
# ============================================================================


class TestLLMResult:
    """Tests for the LLMResult dataclass."""

    def test_default_error_is_empty(self):
        result = LLMResult(output="hello", success=True)
        assert result.error == ""

    def test_fields_accessible(self):
        result = LLMResult(output="text", success=False, error="boom")
        assert result.output == "text"
        assert result.success is False
        assert result.error == "boom"


# ============================================================================
# CLIProvider
# ============================================================================


class TestCLIProvider:
    """Tests for the CLIProvider subprocess wrapper."""

    def test_command_template_property(self):
        provider = CLIProvider("echo {prompt}")
        assert provider.command_template == "echo {prompt}"

    async def test_success(self):
        provider = CLIProvider("echo {prompt}")
        result = await provider.execute("hello world", timeout=10)
        assert result.success is True
        assert result.output != ""

    async def test_timeout(self):
        provider = CLIProvider("sleep 60 && echo {prompt}")
        result = await provider.execute("test", timeout=1)
        assert result.success is False
        assert "Timed out" in result.error

    async def test_nonzero_exit(self):
        provider = CLIProvider("bash -c 'echo {prompt} >/dev/null; exit 1'")
        result = await provider.execute("test", timeout=5)
        assert result.success is False
        assert "Exit 1" in result.error

    async def test_empty_output(self):
        provider = CLIProvider("bash -c 'echo -n {prompt} >/dev/null'")
        result = await provider.execute("test", timeout=5)
        assert result.success is False
        assert "Empty output" in result.error

    async def test_invalid_template_returns_error(self):
        provider = CLIProvider("echo {missing_placeholder}")
        result = await provider.execute("test", timeout=5)
        assert result.success is False
        assert result.error != ""

    async def test_never_raises(self):
        from unittest.mock import AsyncMock, patch

        provider = CLIProvider("echo {prompt}")
        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=OSError("exec failed"),
        ):
            result = await provider.execute("test", timeout=5)

        assert result.success is False
        assert "exec failed" in result.error

    async def test_stderr_truncated_in_error(self):
        from unittest.mock import AsyncMock, patch

        provider = CLIProvider("echo {prompt}")
        proc = AsyncMock()
        proc.communicate.return_value = (b"", b"x" * 300)
        proc.returncode = 1

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=proc,
        ):
            result = await provider.execute("test", timeout=5)

        assert result.success is False
        # Error message includes stderr truncated to 200 chars
        assert len(result.error) < 250

    async def test_shell_fallback_for_shell_syntax(self):
        from unittest.mock import AsyncMock, patch

        provider = CLIProvider("cat <<'EOF'\n{prompt}\nEOF")
        proc = AsyncMock()
        proc.communicate.return_value = (b"ok\n", b"")
        proc.returncode = 0

        with patch(
            "arxiv_browser.llm_providers.asyncio.create_subprocess_shell",
            new_callable=AsyncMock,
            return_value=proc,
        ) as shell_mock:
            result = await provider.execute("hello", timeout=5)

        assert result.success is True
        shell_mock.assert_called_once()


# ============================================================================
# resolve_provider
# ============================================================================


class TestResolveProvider:
    """Tests for the resolve_provider factory function."""

    def test_none_when_unconfigured(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig()
        result = resolve_provider(config)
        assert result is None

    def test_preset_returns_provider(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_preset="copilot")
        result = resolve_provider(config)
        assert result is not None
        assert isinstance(result, CLIProvider)

    def test_custom_command_returns_provider(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_command="my-tool -p {prompt}")
        result = resolve_provider(config)
        assert result is not None
        assert isinstance(result, CLIProvider)
        assert "{prompt}" in result.command_template

    def test_custom_command_takes_precedence(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_preset="copilot", llm_command="custom {prompt}")
        result = resolve_provider(config)
        assert result is not None
        assert "custom" in result.command_template


# ============================================================================
# Protocol compliance
# ============================================================================


class TestLLMProviderProtocol:
    """Verify CLIProvider satisfies the LLMProvider protocol."""

    def test_cli_provider_is_llm_provider(self):
        provider = CLIProvider("echo {prompt}")
        assert isinstance(provider, LLMProvider)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(LLMProvider, "__protocol_attrs__") or callable(
            getattr(LLMProvider, "__instancecheck__", None)
        )
