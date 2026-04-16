"""Small runtime-neutral protocols shared across UI modules."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class TaskTrackingApp(Protocol):
    """Protocol for app objects that can track background coroutines."""

    def _track_task(self, coro: Any, *, dataset_bound: bool = False) -> asyncio.Task[None]: ...


__all__ = ["TaskTrackingApp"]
