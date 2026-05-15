"""Bounded citation-genealogy tree construction.

The genealogy view is citation-derived only: it follows Semantic Scholar
references/citations and does not try to infer intellectual influence beyond
those graph edges.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from arxiv_browser.semantic_scholar import CitationEntry

GenealogyDirection = Literal["ancestors", "descendants"]
CitationGraphFetcher = Callable[[str], Awaitable[tuple[list[CitationEntry], list[CitationEntry]]]]


@dataclass(frozen=True, slots=True)
class GenealogyOptions:
    """Bounds for automatic genealogy expansion."""

    max_depth: int = 3
    branch_factor: int = 3
    max_nodes: int = 25


@dataclass(frozen=True, slots=True)
class GenealogyPaper:
    """Paper payload stored on each rendered genealogy node."""

    paper_id: str
    arxiv_id: str
    title: str
    year: int | None
    citation_count: int
    url: str


@dataclass(slots=True)
class GenealogyNode:
    """One node in the bounded citation-genealogy tree."""

    paper: GenealogyPaper
    children: list[GenealogyNode] = field(default_factory=list)
    repeated: bool = False
    truncated: bool = False
    is_target: bool = False
    is_local: bool = False
    is_starred: bool = False


@dataclass(frozen=True, slots=True)
class GenealogyRoot:
    """Root paper metadata for genealogy construction."""

    paper_id: str
    title: str
    arxiv_id: str = ""
    year: int | None = None
    citation_count: int = 0
    url: str = ""


@dataclass(frozen=True, slots=True)
class GenealogyContext:
    """Local/user-specific flags for genealogy nodes."""

    local_arxiv_ids: frozenset[str] = frozenset()
    starred_arxiv_ids: frozenset[str] = frozenset()


@dataclass(slots=True)
class _BuildState:
    options: GenealogyOptions
    context: GenealogyContext
    expanded_ids: set[str] = field(default_factory=set)
    node_count: int = 0


@dataclass(frozen=True, slots=True)
class _ExpansionRequest:
    paper: GenealogyPaper
    depth: int
    path_ids: frozenset[str]
    target_id: str


async def build_genealogy_tree(
    root: GenealogyRoot,
    direction: GenealogyDirection,
    fetch_callback: CitationGraphFetcher,
    options: GenealogyOptions | None = None,
    context: GenealogyContext | None = None,
) -> GenealogyNode:
    """Build a bounded citation-genealogy tree for *root*."""
    state = _BuildState(
        options=_normalize_options(options or GenealogyOptions()),
        context=context or GenealogyContext(),
    )
    root_paper = _paper_from_root(root)
    root_node = await _expand_node(
        _ExpansionRequest(
            paper=root_paper,
            depth=0,
            path_ids=frozenset(),
            target_id=_node_id(root_paper),
        ),
        direction,
        fetch_callback,
        state,
    )
    if direction == "ancestors":
        return _invert_ancestor_tree(root_node)
    return root_node


def _normalize_options(options: GenealogyOptions) -> GenealogyOptions:
    return GenealogyOptions(
        max_depth=max(0, options.max_depth),
        branch_factor=max(0, options.branch_factor),
        max_nodes=max(1, options.max_nodes),
    )


async def _expand_node(
    request: _ExpansionRequest,
    direction: GenealogyDirection,
    fetch_callback: CitationGraphFetcher,
    state: _BuildState,
) -> GenealogyNode:
    node = _make_node(request.paper, state.context, request.target_id)
    _consume_budget(state)

    paper_id = _node_id(request.paper)
    if paper_id in request.path_ids:
        node.repeated = True
        return node
    if paper_id in state.expanded_ids:
        node.repeated = True
        return node
    if request.depth >= state.options.max_depth:
        node.truncated = True
        return node

    state.expanded_ids.add(paper_id)
    references, citations = await fetch_callback(request.paper.paper_id)
    candidates = references if direction == "ancestors" else citations
    await _expand_children(node, candidates, direction, fetch_callback, state, request)
    return node


async def _expand_children(
    node: GenealogyNode,
    entries: list[CitationEntry],
    direction: GenealogyDirection,
    fetch_callback: CitationGraphFetcher,
    state: _BuildState,
    request: _ExpansionRequest,
) -> None:
    candidates = _rank_entries(entries)
    limited = candidates[: state.options.branch_factor]
    node.truncated = len(candidates) > len(limited)
    for entry in limited:
        if state.node_count >= state.options.max_nodes:
            node.truncated = True
            break
        paper_id = _node_id(request.paper)
        child = await _expand_node(
            _ExpansionRequest(
                paper=_paper_from_entry(entry),
                depth=request.depth + 1,
                path_ids=frozenset((*request.path_ids, paper_id)),
                target_id=request.target_id,
            ),
            direction,
            fetch_callback,
            state,
        )
        node.children.append(child)


def _consume_budget(state: _BuildState) -> None:
    state.node_count += 1


def _rank_entries(entries: list[CitationEntry]) -> list[CitationEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            -entry.citation_count,
            entry.year if entry.year is not None else 9999,
            entry.title.casefold(),
            entry.s2_paper_id,
        ),
    )


def _make_node(
    paper: GenealogyPaper,
    context: GenealogyContext,
    target_id: str,
) -> GenealogyNode:
    return GenealogyNode(
        paper=paper,
        is_target=_node_id(paper) == target_id,
        is_local=bool(paper.arxiv_id and paper.arxiv_id in context.local_arxiv_ids),
        is_starred=bool(paper.arxiv_id and paper.arxiv_id in context.starred_arxiv_ids),
    )


def _invert_ancestor_tree(target_root: GenealogyNode) -> GenealogyNode:
    synthetic = GenealogyNode(
        paper=GenealogyPaper(
            paper_id="",
            arxiv_id="",
            title=f"Ancestors of {target_root.paper.title}",
            year=None,
            citation_count=0,
            url="",
        ),
        is_target=False,
    )
    for path in _collect_paths(target_root, ()):
        _merge_reversed_path(synthetic, reversed(path))
    return synthetic


def _collect_paths(
    node: GenealogyNode,
    prefix: tuple[GenealogyNode, ...],
) -> list[tuple[GenealogyNode, ...]]:
    path = (*prefix, node)
    if not node.children:
        return [path]
    paths: list[tuple[GenealogyNode, ...]] = []
    for child in node.children:
        paths.extend(_collect_paths(child, path))
    return paths


def _merge_reversed_path(
    root: GenealogyNode,
    reversed_path: Iterable[GenealogyNode],
) -> None:
    current = root
    for source in reversed_path:
        child = _find_child(current, _node_id(source.paper))
        if child is None:
            child = _copy_node_without_children(source)
            current.children.append(child)
            current.children.sort(key=_display_sort_key)
        else:
            child.repeated = child.repeated or source.repeated
            child.truncated = child.truncated or source.truncated
        current = child


def _find_child(parent: GenealogyNode, paper_id: str) -> GenealogyNode | None:
    for child in parent.children:
        if _node_id(child.paper) == paper_id:
            return child
    return None


def _copy_node_without_children(node: GenealogyNode) -> GenealogyNode:
    return GenealogyNode(
        paper=node.paper,
        repeated=node.repeated,
        truncated=node.truncated,
        is_target=node.is_target,
        is_local=node.is_local,
        is_starred=node.is_starred,
    )


def _display_sort_key(node: GenealogyNode) -> tuple[int, int, str, str]:
    year = node.paper.year if node.paper.year is not None else 9999
    return (year, -node.paper.citation_count, node.paper.title.casefold(), node.paper.paper_id)


def _paper_from_root(root: GenealogyRoot) -> GenealogyPaper:
    arxiv_id = root.arxiv_id or _arxiv_id_from_paper_id(root.paper_id)
    return GenealogyPaper(
        paper_id=root.paper_id or _fallback_paper_id(arxiv_id),
        arxiv_id=arxiv_id,
        title=root.title or "Unknown Title",
        year=root.year,
        citation_count=root.citation_count,
        url=root.url or _url_for(arxiv_id, root.paper_id),
    )


def _paper_from_entry(entry: CitationEntry) -> GenealogyPaper:
    return GenealogyPaper(
        paper_id=entry.s2_paper_id or _fallback_paper_id(entry.arxiv_id),
        arxiv_id=entry.arxiv_id,
        title=entry.title or "Unknown Title",
        year=entry.year,
        citation_count=entry.citation_count,
        url=entry.url or _url_for(entry.arxiv_id, entry.s2_paper_id),
    )


def _node_id(paper: GenealogyPaper) -> str:
    return paper.paper_id or _fallback_paper_id(paper.arxiv_id) or paper.title.casefold()


def _fallback_paper_id(arxiv_id: str) -> str:
    return f"ARXIV:{arxiv_id}" if arxiv_id else ""


def _arxiv_id_from_paper_id(paper_id: str) -> str:
    return paper_id.removeprefix("ARXIV:") if paper_id.startswith("ARXIV:") else ""


def _url_for(arxiv_id: str, paper_id: str) -> str:
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    if paper_id and not paper_id.startswith("ARXIV:"):
        return f"https://www.semanticscholar.org/paper/{paper_id}"
    return ""


__all__ = [
    "CitationGraphFetcher",
    "GenealogyContext",
    "GenealogyDirection",
    "GenealogyNode",
    "GenealogyOptions",
    "GenealogyPaper",
    "GenealogyRoot",
    "build_genealogy_tree",
]
