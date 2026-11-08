"""Where does the watermark signal live? Lexical category of every scored token.

For count-based detectors the z-score is a sum over tokens, so it splits exactly
into one contribution per lexical category::

    z = sum_c (green_c - gamma * n_c) / sqrt(gamma * (1 - gamma) * N)

where ``n_c`` is the number of scored tokens of category ``c`` and ``N`` the total.
"""

from __future__ import annotations

import io
import keyword
import tokenize
from collections import defaultdict

from cwm.watermarks.syntax import PYTHON_OPERATORS, PYTHON_TYPES

CATEGORIES = (
    "identifier", "keyword", "type", "operator", "delimiter", "literal",
    "docstring", "comment", "whitespace", "unknown",
)  # fmt: skip

_STATEMENT_START = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
_LITERAL_TYPES = {tokenize.NUMBER, tokenize.STRING}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    if hasattr(tokenize, _name):
        _LITERAL_TYPES.add(getattr(tokenize, _name))


def _line_starts(code: str) -> list[int]:
    starts = [0]
    for line in code.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def lexical_spans(code: str) -> list[tuple[int, int, str]]:
    """Character spans of Python lexical tokens with their category."""
    starts = _line_starts(code)
    try:
        tokens = [
            t for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type not in (tokenize.ENDMARKER,)
        ]  # fmt: skip
    except (tokenize.TokenError, SyntaxError, UnicodeError, ValueError):
        return []

    spans: list[tuple[int, int, str]] = []
    previous_type: int | None = None
    for index, tok in enumerate(tokens):
        begin = starts[tok.start[0] - 1] + tok.start[1]
        end = starts[tok.end[0] - 1] + tok.end[1]
        if tok.type == tokenize.NAME:
            if keyword.iskeyword(tok.string):
                category = "keyword"
            elif tok.string in PYTHON_TYPES:
                category = "type"
            else:
                category = "identifier"
        elif tok.type == tokenize.OP:
            category = "operator" if tok.string in PYTHON_OPERATORS else "delimiter"
        elif tok.type == tokenize.STRING:
            following = next(
                (t.type for t in tokens[index + 1 :] if t.type != tokenize.COMMENT), None
            )
            standalone = previous_type in _STATEMENT_START | {None} and following in (
                tokenize.NEWLINE, None,
            )  # fmt: skip
            category = "docstring" if standalone else "literal"
        elif tok.type in _LITERAL_TYPES:
            category = "literal"
        elif tok.type == tokenize.COMMENT:
            category = "comment"
        else:
            category = "whitespace"
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            previous_type = tok.type
        if end > begin:
            spans.append((begin, end, category))
    return spans


def token_categories(code: str, offsets: list[tuple[int, int]]) -> list[str]:
    """Category of each model token, from the lexical token it overlaps the most."""
    spans = lexical_spans(code)
    categories = []
    for start, end in offsets:
        text = code[start:end]
        if text.strip() == "":
            categories.append("whitespace")
            continue
        best, best_overlap = "unknown", 0
        for span_start, span_end, category in spans:
            if span_end <= start:
                continue
            if span_start >= end:
                break
            overlap = min(end, span_end) - max(start, span_start)
            if overlap > best_overlap:
                best, best_overlap = category, overlap
        categories.append(best)
    return categories


def z_contributions(
    categories: list[str], scored: list[bool], green: list[bool], gamma: float
) -> dict[str, float]:
    """Split a count-based z-score into additive contributions per category."""
    total = sum(scored)
    if total == 0:
        return {}
    denominator = (gamma * (1.0 - gamma) * total) ** 0.5
    green_by: dict[str, int] = defaultdict(int)
    count_by: dict[str, int] = defaultdict(int)
    for category, s, g in zip(categories, scored, green, strict=True):
        if s:
            count_by[category] += 1
            green_by[category] += int(g)
    return {c: (green_by[c] - gamma * count_by[c]) / denominator for c in count_by}
