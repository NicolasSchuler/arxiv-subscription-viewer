"""Config serialization helpers for the conference deadline tracker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arxiv_browser.models import UserConfig


def to_dict(config: UserConfig) -> dict[str, Any]:
    """Return deadline-tracker fields for config persistence."""
    return {
        "conference_deadlines_enabled": config.conference_deadlines_enabled,
        "conference_deadlines_source_url": config.conference_deadlines_source_url,
        "conference_deadlines_cache_ttl_hours": config.conference_deadlines_cache_ttl_hours,
    }


def from_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Parse deadline-tracker config fields with conservative bounds."""
    default_source = UserConfig().conference_deadlines_source_url
    enabled = data.get("conference_deadlines_enabled")
    source_url = data.get("conference_deadlines_source_url")
    ttl_hours = data.get("conference_deadlines_cache_ttl_hours")
    return {
        "conference_deadlines_enabled": enabled if isinstance(enabled, bool) else False,
        "conference_deadlines_source_url": source_url
        if isinstance(source_url, str) and source_url
        else default_source,
        "conference_deadlines_cache_ttl_hours": max(
            1, min(168, ttl_hours if isinstance(ttl_hours, int) else 24)
        ),
    }
