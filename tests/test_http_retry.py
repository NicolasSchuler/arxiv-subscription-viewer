"""Tests for the HTTP retry utility with exponential backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from arxiv_browser.http_retry import (
    BACKOFF_BASE,
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    retry_with_backoff,
)


def _make_response(status_code: int) -> httpx.Response:
    """Build a minimal httpx.Response with the given status code."""
    request = httpx.Request("GET", "https://example.com")
    return httpx.Response(status_code=status_code, request=request)


def _raise_status_error(status_code: int) -> None:
    """Raise an HTTPStatusError for *status_code*."""
    resp = _make_response(status_code)
    resp.request = httpx.Request("GET", "https://example.com")
    raise httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=resp.request,
        response=resp,
    )


# ── Success (no retry) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_success_no_retry() -> None:
    """A successful call should return the result without retrying."""
    fn = AsyncMock(return_value="ok")
    result = await retry_with_backoff(fn, operation="test")
    assert result == "ok"
    fn.assert_awaited_once()


# ── Retry on 429 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_429() -> None:
    """A 429 on the first call should be retried, succeeding on the second."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _raise_status_error(429)
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await retry_with_backoff(_fn, operation="test")

    assert result == "ok"
    assert call_count == 2
    mock_sleep.assert_awaited_once_with(BACKOFF_BASE * (2**0))


# ── Retry on 503 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_503() -> None:
    """A 503 should be retried."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _raise_status_error(503)
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = await retry_with_backoff(_fn, operation="test")

    assert result == "ok"
    assert call_count == 2


# ── Retry on connection error ───────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_connect_error() -> None:
    """A ConnectError should trigger a retry."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("connection refused")
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = await retry_with_backoff(_fn, operation="test")

    assert result == "ok"
    assert call_count == 2


# ── Retry on timeout ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_timeout() -> None:
    """A TimeoutException should trigger a retry."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("timed out")
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = await retry_with_backoff(_fn, operation="test")

    assert result == "ok"
    assert call_count == 2


# ── Max retries exceeded ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_retries_exceeded() -> None:
    """After max_retries failures, the last exception is re-raised."""

    async def _fn() -> str:
        _raise_status_error(503)
        return "unreachable"  # pragma: no cover

    with (
        patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(httpx.HTTPStatusError) as exc_info,
    ):
        await retry_with_backoff(_fn, max_retries=2, operation="test")

    assert exc_info.value.response.status_code == 503


@pytest.mark.asyncio
async def test_max_retries_exceeded_connection_error() -> None:
    """After max_retries ConnectErrors, the last exception is re-raised."""

    async def _fn() -> str:
        raise httpx.ConnectError("connection refused")

    with (
        patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(httpx.ConnectError),
    ):
        await retry_with_backoff(_fn, max_retries=2, operation="test")


# ── Non-retryable status codes ──────────────────────────────────────


@pytest.mark.asyncio
async def test_non_retryable_404_raises_immediately() -> None:
    """A 404 is not retryable and should be raised without retry."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        _raise_status_error(404)
        return "unreachable"  # pragma: no cover

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await retry_with_backoff(_fn, operation="test")

    assert exc_info.value.response.status_code == 404
    assert call_count == 1  # No retry


@pytest.mark.asyncio
async def test_non_retryable_400_raises_immediately() -> None:
    """A 400 is not retryable and should be raised without retry."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        _raise_status_error(400)
        return "unreachable"  # pragma: no cover

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await retry_with_backoff(_fn, operation="test")

    assert exc_info.value.response.status_code == 400
    assert call_count == 1


# ── Exponential backoff delays ──────────────────────────────────────


@pytest.mark.asyncio
async def test_backoff_delays_increase_exponentially() -> None:
    """Sleep delays should follow 1s, 2s, 4s pattern (backoff_base * 2^attempt)."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_RETRIES:
            _raise_status_error(502)
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await retry_with_backoff(_fn, operation="test")

    assert result == "ok"
    assert call_count == MAX_RETRIES + 1

    # Verify exponential delays: 1.0, 2.0, 4.0
    expected_delays = [BACKOFF_BASE * (2**i) for i in range(MAX_RETRIES)]
    actual_delays = [call.args[0] for call in mock_sleep.await_args_list]
    assert actual_delays == expected_delays


# ── All retryable status codes ──────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
async def test_all_retryable_codes_trigger_retry(status_code: int) -> None:
    """Every code in RETRYABLE_STATUS_CODES should be retried."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _raise_status_error(status_code)
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = await retry_with_backoff(_fn, operation="test")

    assert result == "ok"
    assert call_count == 2


# ── Custom configuration ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_max_retries_and_backoff() -> None:
    """Custom max_retries and backoff_base should be respected."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            _raise_status_error(500)
        return "ok"

    with patch("arxiv_browser.http_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await retry_with_backoff(
            _fn, max_retries=1, backoff_base=0.5, operation="custom"
        )

    assert result == "ok"
    assert call_count == 2
    mock_sleep.assert_awaited_once_with(0.5)
