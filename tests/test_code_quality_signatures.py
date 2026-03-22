"""Structural quality gates for production function signatures."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "arxiv_browser"
MAX_EFFECTIVE_NAMED_PARAMS = 6


@dataclass(frozen=True, slots=True)
class SignatureOffender:
    """One source function or method that violates the signature limit."""

    path: Path
    line: int
    qualname: str
    effective_named_params: int
    params: tuple[str, ...]


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function is an overload stub."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "overload":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "overload":
            return True
    return False


def _effective_named_params(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, tuple[str, ...]]:
    """Count named parameters, excluding `self` and `cls`."""
    params = tuple(
        arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    )
    effective = len(params)
    if node.args.args and node.args.args[0].arg in {"self", "cls"}:
        effective -= 1
    return effective, params


def _scan_signature_offenders() -> list[SignatureOffender]:
    """Collect non-overload functions in src/ that exceed the parameter limit."""
    offenders: list[SignatureOffender] = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))

        class Visitor(ast.NodeVisitor):
            def __init__(self, file_path: Path) -> None:
                self.file_path = file_path
                self.scope: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._handle(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._handle(node)

            def _handle(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                if _is_overload(node):
                    return
                effective, params = _effective_named_params(node)
                if effective > MAX_EFFECTIVE_NAMED_PARAMS:
                    offenders.append(
                        SignatureOffender(
                            path=self.file_path,
                            line=node.lineno,
                            qualname=".".join([*self.scope, node.name]),
                            effective_named_params=effective,
                            params=params,
                        )
                    )
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

        Visitor(path).visit(tree)

    return offenders


def test_no_function_exceeds_six_effective_named_parameters() -> None:
    """Production code should group related inputs into state/request objects."""
    offenders = _scan_signature_offenders()
    assert not offenders, (
        "Found functions or methods with more than 6 effective named parameters:\n"
        + "\n".join(
            f"- {offender.path.relative_to(SRC_ROOT.parent)}:{offender.line} "
            f"{offender.qualname}({', '.join(offender.params)}) -> "
            f"{offender.effective_named_params} effective parameters"
            for offender in offenders
        )
    )
