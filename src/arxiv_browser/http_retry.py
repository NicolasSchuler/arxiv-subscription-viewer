"""HTTP retry utility with exponential backoff."""

from __future__ import annotations

__all__ = [
    "BACKOFF_BASE",
    "MAX_RETRIES",
    "RETRYABLE_STATUS_CODES",
    "retry_with_backoff",
]

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


async def retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
    operation: str = "HTTP request",
) -> T:
    """Execute an async function with exponential backoff on transient failures.

    Retries on:
    - ``httpx.HTTPStatusError`` with a status code in *RETRYABLE_STATUS_CODES*
    - ``httpx.ConnectError``, ``httpx.TimeoutException``, ``httpx.ReadError``

    Non-retryable ``HTTPStatusError`` codes (e.g. 404) are raised immediately.
    After *max_retries* failed attempts the last exception is re-raised.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_exc = exc
            if attempt < max_retries:
                delay = backoff_base * (2**attempt)
                logger.warning(
                    "%s: HTTP %d, retrying in %.1fs (attempt %d/%d)",
                    operation,
                    exc.response.status_code,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = backoff_base * (2**attempt)
                logger.warning(
                    "%s: %s, retrying in %.1fs (attempt %d/%d)",
                    operation,
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]
