"""Render deterministic Textual SVG snapshots for documentation and review."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from arxiv_browser.browser.core import ArxivBrowser
from arxiv_browser.browser.options import ArxivBrowserOptions
from arxiv_browser.models import Paper, PaperMetadata, SearchBookmark, UserConfig
from arxiv_browser.whats_new import WHATS_NEW_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "docs" / "snapshots"
HERO_SOURCE = REPO_ROOT / "docs" / "screenshot.svg"


@dataclass(frozen=True, slots=True)
class SnapshotCase:
    """One deterministic app state to capture as an SVG snapshot."""

    name: str
    size: tuple[int, int]
    setup: Callable[[object], Awaitable[None]]
    theme_name: str = "monokai"
    ascii_icons: bool = False


async def _idle(_: object) -> None:
    """Leave the app in its default mounted browse state."""


async def _focus_details(pilot: object) -> None:
    """Move focus from the paper list into the details pane."""
    await pilot.press("tab")  # type: ignore[attr-defined]
    await pilot.pause(0.1)  # type: ignore[attr-defined]


async def _open_command_palette(pilot: object) -> None:
    """Open the inline command palette through the public keybinding."""
    await pilot.press("ctrl+p")  # type: ignore[attr-defined]
    await pilot.pause(0.1)  # type: ignore[attr-defined]


SNAPSHOT_CASES = (
    SnapshotCase("default-browse", (100, 30), _idle),
    SnapshotCase("breakpoint-browse", (96, 30), _idle),
    SnapshotCase("narrow-browse", (80, 24), _idle),
    SnapshotCase("detail-focus", (100, 30), _focus_details),
    SnapshotCase("command-palette", (100, 30), _open_command_palette),
    SnapshotCase("light-theme-browse", (100, 30), _idle, theme_name="github-light"),
    SnapshotCase(
        "ascii-high-contrast",
        (80, 24),
        _idle,
        theme_name="high-contrast",
        ascii_icons=True,
    ),
)


def _paper(
    arxiv_id: str,
    title: str,
    authors: str,
    categories: str,
    abstract: str,
) -> Paper:
    """Build a fixture paper without triggering network-backed abstract loads."""
    return Paper(
        arxiv_id=arxiv_id,
        date="Mon, 15 Jan 2024",
        title=title,
        authors=authors,
        categories=categories,
        comments=None,
        abstract=abstract,
        abstract_raw=abstract,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _papers() -> list[Paper]:
    """Return representative papers with realistic titles and category density."""
    return [
        _paper(
            "2401.01001",
            "Retrieval-Augmented Agents for Long-Horizon Literature Review",
            "Ada Lovelace, Alan Turing, Grace Hopper",
            "cs.AI cs.CL cs.IR",
            "We study retrieval-augmented agents that organize papers, notes, "
            "and citations across repeated literature-review sessions.",
        ),
        _paper(
            "2401.01002",
            "Sparse Diffusion Transformers with Calibrated Uncertainty",
            "Katherine Johnson, Claude Shannon",
            "cs.LG stat.ML",
            "This paper introduces sparse attention patterns for diffusion "
            "transformers and reports calibrated uncertainty estimates.",
        ),
        _paper(
            "2401.01003",
            "Benchmarking Semantic Search Backends for Scientific Triage",
            "Barbara Liskov, Donald Knuth",
            "cs.DL cs.HC",
            "We compare embedding backends for ranking research papers under "
            "interactive terminal latency constraints.",
        ),
    ]


def _config(theme_name: str = "monokai") -> UserConfig:
    """Return persisted-state fixtures that make snapshots content-rich."""
    return UserConfig(
        onboarding_seen=True,
        last_seen_whats_new=WHATS_NEW_VERSION,
        show_abstract_preview=True,
        theme_name=theme_name,
        bookmarks=[
            SearchBookmark(name="agents", query="agent"),
            SearchBookmark(name="unread AI", query="cat:cs.AI unread"),
        ],
        paper_metadata={
            "2401.01001": PaperMetadata(
                arxiv_id="2401.01001",
                tags=["topic:agents", "status:to-read", "project:lit-review"],
                starred=True,
            ),
            "2401.01002": PaperMetadata(
                arxiv_id="2401.01002",
                tags=["topic:diffusion", "priority:high"],
                is_read=True,
            ),
        },
    )


async def _capture(case: SnapshotCase) -> str:
    """Render one snapshot case and return its SVG payload."""
    app = ArxivBrowser(
        _papers(),
        ArxivBrowserOptions(
            config=_config(theme_name=case.theme_name),
            restore_session=False,
            ascii_icons=case.ascii_icons,
        ),
    )
    async with app.run_test(size=case.size) as pilot:
        await pilot.pause(0.2)
        await case.setup(pilot)
        await pilot.pause(0.1)
        return _normalize_svg(app.export_screenshot(title=case.name, simplify=True))


def _normalize_svg(svg: str) -> str:
    """Return SVG text with deterministic line endings and no trailing blanks."""
    return "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"


async def _render_all() -> dict[str, str]:
    """Render every configured snapshot case."""
    rendered: dict[str, str] = {}
    for case in SNAPSHOT_CASES:
        rendered[case.name] = await _capture(case)
    return rendered


def _snapshot_path(name: str) -> Path:
    """Return the committed SVG path for a snapshot case name."""
    return SNAPSHOT_DIR / f"{name}.svg"


def _diff(expected: str, actual: str, path: Path) -> str:
    """Build a concise unified diff for a mismatched snapshot."""
    return "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=str(path),
            tofile=f"generated:{path.name}",
            lineterm="",
            n=2,
        )
    )


def _write_snapshots(rendered: dict[str, str]) -> None:
    """Persist rendered snapshots and refresh the hero SVG source."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for name, svg in rendered.items():
        _snapshot_path(name).write_text(svg, encoding="utf-8")
    HERO_SOURCE.write_text(rendered["default-browse"], encoding="utf-8")


def _check_snapshots(rendered: dict[str, str]) -> int:
    """Compare rendered snapshots with committed baselines."""
    failures: list[str] = []
    for name, svg in rendered.items():
        path = _snapshot_path(name)
        if not path.exists():
            failures.append(f"Missing snapshot baseline: {path}")
            continue
        expected = path.read_text(encoding="utf-8")
        if expected != svg:
            failures.append(_diff(expected, svg, path))
    if HERO_SOURCE.exists():
        expected_hero = HERO_SOURCE.read_text(encoding="utf-8")
        if expected_hero != rendered["default-browse"]:
            failures.append(_diff(expected_hero, rendered["default-browse"], HERO_SOURCE))
    else:
        failures.append(f"Missing hero screenshot source: {HERO_SOURCE}")

    if not failures:
        print(f"OK: {len(rendered)} TUI snapshots match committed baselines")
        return 0
    print("\n\n".join(failures), file=sys.stderr)
    print(
        "\nSnapshots changed. Run `just snapshots-update` after visually inspecting them.",
        file=sys.stderr,
    )
    return 1


async def _main(argv: list[str] | None = None) -> int:
    """Render snapshots and either update or compare baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite committed SVG baselines instead of checking them.",
    )
    args = parser.parse_args(argv)

    rendered = await _render_all()
    if args.update:
        _write_snapshots(rendered)
        print(f"Updated {len(rendered)} TUI snapshots in {SNAPSHOT_DIR}")
        print(f"Updated hero screenshot source at {HERO_SOURCE}")
        return 0
    return _check_snapshots(rendered)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
