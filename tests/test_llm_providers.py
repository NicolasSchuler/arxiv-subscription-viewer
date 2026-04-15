"""Tests for the LLM provider abstraction layer."""

from __future__ import annotations

from arxiv_browser.llm_providers import (
    CLIProvider,
    HTTPProvider,
    LLMProvider,
    LLMResult,
    get_provider_class,
    llm_command_requires_shell,
    register_provider,
    resolve_provider,
)

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

    async def test_shell_fallback_blocked_when_disabled(self):
        provider = CLIProvider("echo {prompt} | cat", allow_shell=False)
        result = await provider.execute("hello", timeout=5)
        assert result.success is False
        assert "allow_llm_shell_fallback" in result.error

    async def test_shell_fallback_for_env_prefixed_command(self):
        from unittest.mock import AsyncMock, patch

        provider = CLIProvider("OPENAI_API_KEY=test-key llm {prompt}")
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

    def test_env_prefixed_command_requires_shell(self):
        assert llm_command_requires_shell("OPENAI_API_KEY=test-key llm {prompt}") is True

    async def test_shell_fallback_uses_windows_prompt_quoting(self):
        from unittest.mock import AsyncMock, patch

        provider = CLIProvider("echo {prompt} | cat")
        proc = AsyncMock()
        proc.communicate.return_value = (b"ok\n", b"")
        proc.returncode = 0

        with (
            patch("arxiv_browser.llm.os.name", "nt"),
            patch(
                "arxiv_browser.llm.subprocess.list2cmdline", return_value='"safe ^& prompt"'
            ) as quote_mock,
            patch(
                "arxiv_browser.llm_providers.asyncio.create_subprocess_shell",
                new_callable=AsyncMock,
                return_value=proc,
            ) as shell_mock,
        ):
            result = await provider.execute("safe ^& prompt", timeout=5)

        assert result.success is True
        quote_mock.assert_called_once_with(["safe ^& prompt"])
        shell_mock.assert_called_once()
        assert shell_mock.call_args.args[0] == 'echo "safe ^& prompt" | cat'


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

    async def test_resolve_provider_honors_shell_policy(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_command="echo {prompt} | cat", allow_llm_shell_fallback=False)
        result = resolve_provider(config)
        assert result is not None
        blocked = await result.execute("hello", timeout=5)
        assert blocked.success is False
        assert "allow_llm_shell_fallback" in blocked.error


# ============================================================================
# Protocol compliance
# ============================================================================


class TestLLMProviderProtocol:
    """Verify CLIProvider satisfies the LLMProvider protocol."""

    def test_cli_provider_is_llm_provider(self):
        provider = CLIProvider("echo {prompt}")
        assert isinstance(provider, LLMProvider)

    def test_http_provider_is_llm_provider(self):
        provider = HTTPProvider("http://localhost:8000", "", "test-model")
        assert isinstance(provider, LLMProvider)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(LLMProvider, "__protocol_attrs__") or callable(
            getattr(LLMProvider, "__instancecheck__", None)
        )


# ============================================================================
# HTTPProvider
# ============================================================================


class TestHTTPProvider:
    """Tests for the HTTPProvider using the OpenAI-compatible chat completions API."""

    def test_properties(self):
        provider = HTTPProvider("http://localhost:8000/", "sk-test", "gpt-4o")
        assert provider.base_url == "http://localhost:8000"
        assert provider.model == "gpt-4o"

    def test_trailing_slash_stripped(self):
        provider = HTTPProvider("http://localhost:8000/", "", "m")
        assert provider.base_url == "http://localhost:8000"

    async def test_success(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "sk-key", "model-1")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}],
        }
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("Say hello", timeout=30)

        assert result.success is True
        assert result.output == "Hello world"
        client.post.assert_called_once()
        call_args = client.post.call_args
        assert "/v1/chat/completions" in call_args.args[0]
        assert call_args.kwargs["json"]["model"] == "model-1"

    async def test_auth_header_set_when_api_key_present(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "sk-secret", "model-1")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            await provider.execute("test", timeout=30)

        headers = client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-secret"

    async def test_no_auth_header_when_key_empty(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://localhost:11434", "", "llama3")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            await provider.execute("test", timeout=30)

        headers = client.post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    async def test_timeout(self):
        from unittest.mock import AsyncMock, patch

        import httpx

        provider = HTTPProvider("http://api.example.com", "", "model-1")
        client = AsyncMock()
        client.post.side_effect = httpx.TimeoutException("timed out")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=5)

        assert result.success is False
        assert "Timed out" in result.error

    async def test_http_error_status(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "", "model-1")
        response = MagicMock()
        response.status_code = 429
        response.text = "Rate limit exceeded"
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=30)

        assert result.success is False
        assert "HTTP 429" in result.error

    async def test_invalid_json_response(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "", "model-1")
        response = MagicMock()
        response.status_code = 200
        response.json.side_effect = ValueError("bad json")
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=30)

        assert result.success is False
        assert "Invalid JSON" in result.error

    async def test_unexpected_response_structure(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "", "model-1")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"choices": []}  # Missing message
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=30)

        assert result.success is False
        assert "Unexpected response structure" in result.error

    async def test_empty_content(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "", "model-1")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "   "}}],
        }
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=30)

        assert result.success is False
        assert "Empty response content" in result.error

    async def test_retry_on_server_error(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "", "m", max_retries=1)
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.text = "Internal Server Error"
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{"message": {"content": "recovered"}}],
        }

        client = AsyncMock()
        client.post.side_effect = [fail_response, ok_response]
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("arxiv_browser.llm_providers.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await provider.execute("test", timeout=30)

        assert result.success is True
        assert result.output == "recovered"
        assert client.post.call_count == 2

    async def test_no_retry_on_client_error(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = HTTPProvider("http://api.example.com", "", "m", max_retries=2)
        response = MagicMock()
        response.status_code = 401
        response.text = "Unauthorized"
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=30)

        assert result.success is False
        assert "HTTP 401" in result.error
        # Should NOT retry on 401
        assert client.post.call_count == 1

    async def test_connection_error(self):
        from unittest.mock import AsyncMock, patch

        provider = HTTPProvider("http://api.example.com", "", "model-1")
        client = AsyncMock()
        client.post.side_effect = Exception("Connection refused")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            result = await provider.execute("test", timeout=30)

        assert result.success is False
        assert "Connection refused" in result.error


# ============================================================================
# Provider Registry
# ============================================================================


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_cli_registered(self):
        assert get_provider_class("cli") is CLIProvider

    def test_http_registered(self):
        assert get_provider_class("http") is HTTPProvider

    def test_case_insensitive(self):
        assert get_provider_class("CLI") is CLIProvider
        assert get_provider_class("Http") is HTTPProvider

    def test_unknown_returns_none(self):
        assert get_provider_class("nonexistent") is None

    def test_register_custom_provider(self):
        class DummyProvider:
            async def execute(self, prompt: str, timeout: int) -> LLMResult:
                return LLMResult(output="dummy", success=True)

        register_provider("dummy", DummyProvider)  # type: ignore[arg-type]
        assert get_provider_class("dummy") is DummyProvider  # type: ignore[comparison-overlap]


# ============================================================================
# resolve_provider with HTTP
# ============================================================================


class TestResolveProviderHTTP:
    """Tests for resolve_provider with HTTP provider type."""

    def test_http_provider_when_configured(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(
            llm_provider_type="http",
            llm_api_base_url="http://localhost:8000",
            llm_api_model="llama3",
        )
        result = resolve_provider(config)
        assert result is not None
        assert isinstance(result, HTTPProvider)
        assert result.base_url == "http://localhost:8000"
        assert result.model == "llama3"

    def test_http_provider_none_when_no_url(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_provider_type="http")
        result = resolve_provider(config)
        assert result is None

    def test_http_provider_passes_retries(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(
            llm_provider_type="http",
            llm_api_base_url="http://localhost:8000",
            llm_api_model="m",
            llm_max_retries=3,
        )
        provider = resolve_provider(config)
        assert provider is not None
        assert isinstance(provider, HTTPProvider)

    def test_cli_provider_still_default(self):
        from arxiv_browser.models import UserConfig

        config = UserConfig(llm_preset="copilot")
        result = resolve_provider(config)
        assert result is not None
        assert isinstance(result, CLIProvider)
