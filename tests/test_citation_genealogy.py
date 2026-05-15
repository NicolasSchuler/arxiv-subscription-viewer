"""Tests for bounded citation-genealogy construction."""

from __future__ import annotations

import pytest

from arxiv_browser.citation_genealogy import (
    GenealogyContext,
    GenealogyOptions,
    GenealogyRoot,
    build_genealogy_tree,
)
from arxiv_browser.semantic_scholar import CitationEntry


def _entry(
    paper_id: str,
    title: str,
    *,
    arxiv_id: str = "",
    year: int | None = 2020,
    citations: int = 10,
) -> CitationEntry:
    return CitationEntry(
        s2_paper_id=paper_id,
        arxiv_id=arxiv_id,
        title=title,
        authors="A. Author",
        year=year,
        citation_count=citations,
        url=f"https://example.test/{paper_id}",
    )


def _titles_at(node) -> list[str]:
    return [child.paper.title for child in node.children]


@pytest.mark.asyncio
async def test_ancestor_tree_inverts_paths_to_target_leaf() -> None:
    graph = {
        "target": ([_entry("gpt2", "GPT-2", year=2019, citations=200)], []),
        "gpt2": ([_entry("bert", "BERT", year=2018, citations=300)], []),
        "bert": (
            [_entry("transformer", "Attention Is All You Need", year=2017, citations=500)],
            [],
        ),
        "transformer": ([], []),
    }
    calls: list[str] = []

    async def fetch(paper_id: str):
        calls.append(paper_id)
        return graph.get(paper_id, ([], []))

    tree = await build_genealogy_tree(
        GenealogyRoot("target", "Starred Paper", arxiv_id="2401.00001"),
        "ancestors",
        fetch,
        GenealogyOptions(max_depth=4, branch_factor=3, max_nodes=25),
        GenealogyContext(
            local_arxiv_ids=frozenset({"2401.00001"}),
            starred_arxiv_ids=frozenset({"2401.00001"}),
        ),
    )

    attention = tree.children[0]
    bert = attention.children[0]
    gpt2 = bert.children[0]
    target = gpt2.children[0]

    assert [attention.paper.title, bert.paper.title, gpt2.paper.title, target.paper.title] == [
        "Attention Is All You Need",
        "BERT",
        "GPT-2",
        "Starred Paper",
    ]
    assert target.is_target is True
    assert target.is_local is True
    assert target.is_starred is True
    assert calls == ["target", "gpt2", "bert", "transformer"]


@pytest.mark.asyncio
async def test_descendant_tree_sorts_and_truncates_by_branch_factor() -> None:
    graph = {
        "root": (
            [],
            [
                _entry("low", "Low", year=2019, citations=1),
                _entry("top", "Top", year=2021, citations=50),
                _entry("tie-a", "Alpha", year=None, citations=20),
                _entry("tie-b", "Beta", year=2018, citations=20),
            ],
        )
    }

    async def fetch(paper_id: str):
        return graph.get(paper_id, ([], []))

    tree = await build_genealogy_tree(
        GenealogyRoot("root", "Root"),
        "descendants",
        fetch,
        GenealogyOptions(max_depth=1, branch_factor=3, max_nodes=25),
    )

    assert tree.paper.title == "Root"
    assert _titles_at(tree) == ["Top", "Beta", "Alpha"]
    assert tree.truncated is True
    assert all(child.truncated for child in tree.children)


@pytest.mark.asyncio
async def test_descendant_tree_marks_cycles_as_repeats() -> None:
    graph = {
        "root": ([], [_entry("child", "Child", citations=10)]),
        "child": ([], [_entry("root", "Root Again", citations=5)]),
    }

    async def fetch(paper_id: str):
        return graph.get(paper_id, ([], []))

    tree = await build_genealogy_tree(
        GenealogyRoot("root", "Root"),
        "descendants",
        fetch,
        GenealogyOptions(max_depth=3, branch_factor=3, max_nodes=25),
    )

    repeated_root = tree.children[0].children[0]
    assert repeated_root.repeated is True
    assert repeated_root.children == []
    assert repeated_root.is_target is True


@pytest.mark.asyncio
async def test_shared_descendant_is_marked_repeat_without_expanding_twice() -> None:
    graph = {
        "root": (
            [],
            [
                _entry("branch-a", "Branch A", citations=30),
                _entry("branch-b", "Branch B", citations=20),
            ],
        ),
        "branch-a": ([], [_entry("shared", "Shared", citations=10)]),
        "branch-b": ([], [_entry("shared", "Shared", citations=10)]),
        "shared": ([], [_entry("leaf", "Leaf", citations=1)]),
    }
    calls: list[str] = []

    async def fetch(paper_id: str):
        calls.append(paper_id)
        return graph.get(paper_id, ([], []))

    tree = await build_genealogy_tree(
        GenealogyRoot("root", "Root"),
        "descendants",
        fetch,
        GenealogyOptions(max_depth=4, branch_factor=3, max_nodes=25),
    )

    first_shared = tree.children[0].children[0]
    second_shared = tree.children[1].children[0]
    assert first_shared.repeated is False
    assert second_shared.repeated is True
    assert second_shared.children == []
    assert calls.count("shared") == 1


@pytest.mark.asyncio
async def test_node_budget_stops_expansion_before_extra_fetches() -> None:
    graph = {
        "root": (
            [],
            [
                _entry("first", "First", citations=30),
                _entry("second", "Second", citations=20),
            ],
        ),
        "first": ([], [_entry("grandchild", "Grandchild", citations=10)]),
    }
    calls: list[str] = []

    async def fetch(paper_id: str):
        calls.append(paper_id)
        return graph.get(paper_id, ([], []))

    tree = await build_genealogy_tree(
        GenealogyRoot("root", "Root"),
        "descendants",
        fetch,
        GenealogyOptions(max_depth=3, branch_factor=3, max_nodes=2),
    )

    assert _titles_at(tree) == ["First"]
    assert tree.truncated is True
    assert tree.children[0].truncated is True
    assert calls == ["root", "first"]


@pytest.mark.asyncio
async def test_zero_depth_and_missing_year_use_safe_fallbacks() -> None:
    calls: list[str] = []

    async def fetch(paper_id: str):
        calls.append(paper_id)
        return ([_entry("ignored", "Ignored", year=None)], [])

    tree = await build_genealogy_tree(
        GenealogyRoot("ARXIV:2401.00001", ""),
        "descendants",
        fetch,
        GenealogyOptions(max_depth=-1, branch_factor=-1, max_nodes=0),
    )

    assert tree.paper.title == "Unknown Title"
    assert tree.paper.arxiv_id == "2401.00001"
    assert tree.paper.url == "https://arxiv.org/abs/2401.00001"
    assert tree.truncated is True
    assert tree.children == []
    assert calls == []
