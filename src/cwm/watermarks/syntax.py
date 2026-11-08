"""Syntax-token classification used by STONE.

The token sets are copied from the official STONE implementation
(``watermark/stone/stone.py`` in github.com/inistory/STONE-watermarking), which
matches Table 9 of the paper.

Two classification rules are available:

``official``
    Exactly what the official code does: a model token is syntactic if its decoded
    text, or that text stripped of surrounding whitespace, is one of the listed
    elements. Merged BPE tokens such as ``"):"`` or pure indentation such as
    ``"    "`` are therefore *not* syntactic and get watermarked.
``strict``
    Also treats whitespace-only tokens and tokens made only of operator/delimiter
    characters (``"):"``, ``"]]"``) as syntactic. This is closer to the intent
    described in the paper and is offered for ablations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

SyntaxRule = Literal["official", "strict"]

PYTHON_KEYWORDS = frozenset(
    {
        "True", "False", "None", "and", "as", "assert", "async", "await", "break",
        "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
        "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
        "pass", "raise", "return", "try", "while", "with", "yield",
    }
)  # fmt: skip
PYTHON_OPERATORS = frozenset(
    {
        "+", "-", "*", "/", "%", "**", "//", "=", "==", "!=", ">", "<", ">=", "<=",
        "+=", "-=", "*=", "/=", "%=", "//=", "**=", "&", "|", "<<", ">>", "^", "~",
    }
)  # fmt: skip
PYTHON_DELIMITERS = frozenset({"(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "@", "->", "..."})
PYTHON_WHITESPACE = frozenset({" ", "\t", "\n"})
PYTHON_TYPES = frozenset(
    {"int", "float", "complex", "str", "bytes", "bool", "list", "tuple", "set", "dict", "NoneType"}
)

PYTHON_SYNTAX = (
    PYTHON_KEYWORDS | PYTHON_OPERATORS | PYTHON_DELIMITERS | PYTHON_WHITESPACE | PYTHON_TYPES
)

_SYMBOL_CHARS = frozenset("()[]{},:.;@+-*/%=<>!&|^~")


def is_syntax_token(text: str, rule: SyntaxRule = "official") -> bool:
    """Whether the decoded text of one model token counts as syntactic for STONE."""
    if text in PYTHON_SYNTAX or text.strip() in PYTHON_SYNTAX:
        return True
    if rule == "official":
        return False
    if rule == "strict":
        stripped = text.strip()
        return stripped == "" or all(char in _SYMBOL_CHARS for char in stripped)
    raise ValueError(f"Unknown syntax rule: {rule!r}")


def build_syntax_mask(
    decode: Callable[[int], str], vocab_size: int, rule: SyntaxRule = "official"
) -> torch.Tensor:
    """Boolean mask over the vocabulary, ``True`` where the token is syntactic.

    ``decode`` maps one token id to its text. Ids beyond the tokenizer (padding rows of
    the embedding matrix) decode to ``""`` and are treated as non-syntactic, as in the
    official code; the model never produces them in practice.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for token_id in range(vocab_size):
        mask[token_id] = is_syntax_token(decode(token_id), rule)
    return mask
