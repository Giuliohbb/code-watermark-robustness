"""Edits whose purpose is to remove a watermark while preserving behaviour."""

from __future__ import annotations

import random

import libcst as cst
import python_minifier
from libcst.metadata import (
    Assignment,
    ComprehensionScope,
    FunctionScope,
    MetadataWrapper,
    PositionProvider,
    ScopeProvider,
)

from cwm.transforms.base import Transform, TransformError, fresh_name

LOCAL_NAMES = (
    "value", "item", "result", "current", "total", "count", "index", "element", "data",
    "acc", "output", "entry", "node", "key", "temp", "buffer", "state", "number", "part",
    "chunk", "piece", "record", "token", "candidate", "accumulator", "collected", "found",
)  # fmt: skip
DEAD_NAMES = (
    "cache", "flag", "limit", "offset", "ratio", "marker", "step", "width", "height",
    "label", "score", "level", "weight", "size", "threshold", "counter", "checked", "spare",
)  # fmt: skip


def _parse(code: str) -> cst.Module:
    try:
        return cst.parse_module(code)
    except cst.ParserSyntaxError as error:
        raise TransformError(f"libcst could not parse the code: {error}") from error


def _all_names(module: cst.Module) -> set[str]:
    names: set[str] = set()

    class Collector(cst.CSTVisitor):
        def visit_Name(self, node: cst.Name) -> None:
            names.add(node.value)

    module.visit(Collector())
    return names


def _declared_global_or_nonlocal(module: cst.Module) -> set[str]:
    names: set[str] = set()

    class Collector(cst.CSTVisitor):
        def visit_Global(self, node: cst.Global) -> None:
            names.update(item.name.value for item in node.names)

        def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
            names.update(item.name.value for item in node.names)

    module.visit(Collector())
    return names


class _Renamer(cst.CSTTransformer):
    def __init__(self, new_names: dict[int, str]) -> None:
        self.new_names = new_names

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        new = self.new_names.get(id(original_node))
        return updated_node.with_changes(value=new) if new else updated_node


class RenameLocals(Transform):
    """Rename a fraction of local variables (never parameters, functions or imports).

    Parameters keep their names because callers may pass them by keyword. Each chosen
    variable is renamed consistently at every definition and use, using scope
    analysis, so behaviour is unchanged.
    """

    name = "rename"
    family = "attack"

    def __init__(self, fraction: float = 1.0) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction

    def apply(self, code: str, rng: random.Random) -> str:
        wrapper = MetadataWrapper(_parse(code))
        scopes = wrapper.resolve(ScopeProvider)
        positions = wrapper.resolve(PositionProvider)
        excluded = _declared_global_or_nonlocal(wrapper.module)
        parameters = _parameter_name_ids(wrapper.module)

        candidates: dict[tuple[int, str], list[Assignment]] = {}
        for scope in {s for s in scopes.values() if s is not None}:
            if not isinstance(scope, (FunctionScope, ComprehensionScope)):
                continue
            for assignment in scope.assignments:
                name = assignment.name
                if name in excluded or (id(scope), name) in candidates:
                    continue
                assigns = list(scope.assignments[name])
                renamable = all(
                    isinstance(a, Assignment)
                    and isinstance(a.node, cst.Name)
                    and id(a.node) not in parameters
                    for a in assigns
                )
                if renamable:
                    candidates[(id(scope), name)] = assigns
        if not candidates:
            return code

        def first_position(assigns: list[Assignment]) -> tuple[int, int]:
            return min(
                (positions[a.node].start.line, positions[a.node].start.column) for a in assigns
            )

        ordered = sorted(candidates.items(), key=lambda kv: (first_position(kv[1]), kv[0][1]))
        k = max(1, round(self.fraction * len(ordered)))
        chosen = rng.sample(ordered, k)

        taken = _all_names(wrapper.module)
        new_names: dict[int, str] = {}
        for (_, _name), assigns in chosen:
            new = fresh_name(list(LOCAL_NAMES), taken, rng)
            for assignment in assigns:
                new_names[id(assignment.node)] = new
                for access in assignment.references:
                    if isinstance(access.node, cst.Name):
                        new_names[id(access.node)] = new
        return wrapper.module.visit(_Renamer(new_names)).code


def _parameter_name_ids(module: cst.Module) -> set[int]:
    """Identity of every ``Name`` node that declares a function parameter."""
    ids: set[int] = set()

    class Collector(cst.CSTVisitor):
        def visit_Param(self, node: cst.Param) -> None:
            ids.add(id(node.name))

    module.visit(Collector())
    return ids


class _Inserter(cst.CSTTransformer):
    def __init__(self, target: cst.FunctionDef, index: int, statements: list) -> None:
        self.target = target
        self.index = index
        self.statements = statements

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if original_node is not self.target:
            return updated_node
        body = list(updated_node.body.body)
        body[self.index : self.index] = self.statements
        return updated_node.with_changes(body=updated_node.body.with_changes(body=body))


class InsertDeadCode(Transform):
    """Insert statements that never change the result, as in Suresh et al. (2024)."""

    name = "dead_code"
    family = "attack"

    TEMPLATES = (
        "if False:\n    {a} = {b}\n",
        "{a} = {n}\nif {a} != {a}:\n    {b} = 0\n",
    )

    def __init__(self, n_insertions: int = 3) -> None:
        if n_insertions < 1:
            raise ValueError("n_insertions must be >= 1")
        self.n_insertions = n_insertions

    def apply(self, code: str, rng: random.Random) -> str:
        module = _parse(code)
        taken = _all_names(module)
        for _ in range(self.n_insertions):
            functions = _function_bodies(module)
            if not functions:
                raise TransformError("no function body to insert dead code into")
            target = rng.choice(functions)
            body = list(target.body.body)
            lowest = 1 if body and _starts_with_docstring(body) else 0
            index = rng.randint(lowest, len(body))
            snippet = rng.choice(self.TEMPLATES).format(
                a=fresh_name(list(DEAD_NAMES), taken, rng),
                b=fresh_name(list(DEAD_NAMES), taken, rng),
                n=rng.randint(0, 99),
            )
            statements = list(cst.parse_module(snippet).body)
            module = module.visit(_Inserter(target, index, statements))
        return module.code


def _function_bodies(module: cst.Module) -> list[cst.FunctionDef]:
    """Every function with an indented body, in source order."""
    found: list[cst.FunctionDef] = []

    class Collector(cst.CSTVisitor):
        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if isinstance(node.body, cst.IndentedBlock):
                found.append(node)

    module.visit(Collector())
    return found


def _starts_with_docstring(body: list) -> bool:
    first = body[0]
    return (
        isinstance(first, cst.SimpleStatementLine)
        and len(first.body) == 1
        and isinstance(first.body[0], cst.Expr)
        and isinstance(first.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    )


class Minify(Transform):
    """Minify with python-minifier, as in the Disappearing Ink experiments.

    Local names are shortened and comments, docstrings and layout are compacted;
    module-level names (the entry point) are kept so the tests can still call it.
    """

    name = "minify"
    family = "attack"

    def apply(self, code: str, rng: random.Random) -> str:
        try:
            return python_minifier.minify(code, rename_locals=True, rename_globals=False)
        except (SyntaxError, ValueError) as error:
            raise TransformError(f"python-minifier failed: {error}") from error
