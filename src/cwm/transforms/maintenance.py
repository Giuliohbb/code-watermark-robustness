"""Routine maintenance edits: formatting and removing comments/docstrings."""

from __future__ import annotations

import random

import black
import libcst as cst

from cwm.transforms.base import Transform, TransformError


class BlackFormat(Transform):
    """Reformat with Black's default style, as a formatter or pre-commit hook would."""

    name = "black"
    family = "maintenance"

    def apply(self, code: str, rng: random.Random) -> str:
        try:
            return black.format_str(code, mode=black.Mode())
        except black.InvalidInput as error:
            raise TransformError(f"black could not parse the code: {error}") from error


def _is_docstring(statement: cst.BaseStatement) -> bool:
    return (
        isinstance(statement, cst.SimpleStatementLine)
        and len(statement.body) == 1
        and isinstance(statement.body[0], cst.Expr)
        and isinstance(statement.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    )


def _without_docstring(body: list[cst.BaseStatement]) -> list[cst.BaseStatement]:
    if body and _is_docstring(body[0]):
        return body[1:]
    return body


class _CommentAndDocstringRemover(cst.CSTTransformer):
    def leave_EmptyLine(
        self, original_node: cst.EmptyLine, updated_node: cst.EmptyLine
    ) -> cst.EmptyLine | cst.RemovalSentinel:
        if updated_node.comment is not None:
            return cst.RemoveFromParent()
        return updated_node

    def leave_TrailingWhitespace(
        self, original_node: cst.TrailingWhitespace, updated_node: cst.TrailingWhitespace
    ) -> cst.TrailingWhitespace:
        if updated_node.comment is not None:
            return updated_node.with_changes(comment=None, whitespace=cst.SimpleWhitespace(""))
        return updated_node

    def _strip_block(self, node: cst.FunctionDef | cst.ClassDef) -> cst.FunctionDef | cst.ClassDef:
        if not isinstance(node.body, cst.IndentedBlock):
            return node
        body = _without_docstring(list(node.body.body))
        if not body:
            body = [cst.SimpleStatementLine(body=[cst.Pass()])]
        return node.with_changes(body=node.body.with_changes(body=body))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return self._strip_block(updated_node)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        return self._strip_block(updated_node)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        return updated_node.with_changes(body=_without_docstring(list(updated_node.body)))


class StripComments(Transform):
    """Remove comments and docstrings while keeping the rest of the layout intact."""

    name = "strip_comments"
    family = "maintenance"

    def apply(self, code: str, rng: random.Random) -> str:
        try:
            module = cst.parse_module(code)
        except cst.ParserSyntaxError as error:
            raise TransformError(f"libcst could not parse the code: {error}") from error
        return module.visit(_CommentAndDocstringRemover()).code
