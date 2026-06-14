"""One-time onboarding hint helpers for the browser app."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arxiv_browser.browser.core import ArxivBrowser
    from arxiv_browser.models import Paper


def maybe_hint_shortcuts(app: ArxivBrowser) -> None:
    """Show the one-time nudge toward full keybinding help."""
    if app._config.shortcuts_hint_seen or not app._config.onboarding_seen:
        return
    if len(app.screen_stack) > 1:
        return
    app._config.shortcuts_hint_seen = True
    app._save_config_or_warn("shortcuts hint")
    app.notify("Press ? for all shortcuts", title="Tip", timeout=6)


def paper_has_enrichment(app: ArxivBrowser, paper: Paper) -> bool:
    """Return True if a paper carries any meta-line enrichment badge."""
    aid = paper.arxiv_id
    if aid in app._s2_cache or aid in app._hf_cache:
        return True
    if aid in app._relevance_scores or aid in app._version_updates:
        return True
    ctx = getattr(app, "_digest_inbox_context", None)
    return bool(ctx and aid in getattr(ctx, "section_labels_by_id", {}))


def maybe_hint_badge_legend(app: ArxivBrowser, paper: Paper) -> None:
    """Show the one-time badge legend nudge once enrichment appears."""
    if app._config.badge_legend_hint_seen or len(app.screen_stack) > 1:
        return
    if not app._paper_has_enrichment(paper):
        return
    app._config.badge_legend_hint_seen = True
    app._save_config_or_warn("badge legend hint")
    app.notify("Badge meanings are in ? -> Badge Legend", title="Tip", timeout=6)
