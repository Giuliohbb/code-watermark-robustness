import pytest

from cwm.evaluation.signal import lexical_spans, token_categories, z_contributions

CODE = 'def f(x):\n    """Doc."""\n    total = x + 1  # note\n    return total\n'


def _char_offsets(code, pieces):
    offsets, position = [], 0
    for piece in pieces:
        start = code.index(piece, position)
        offsets.append((start, start + len(piece)))
        position = start + len(piece)
    return offsets


def test_lexical_categories():
    categories = {code: cat for (s, e, cat) in lexical_spans(CODE) for code in [CODE[s:e]]}
    assert categories["def"] == "keyword"
    assert categories["total"] == "identifier"
    assert categories['"""Doc."""'] == "docstring"
    assert categories["# note"] == "comment"
    assert categories["+"] == "operator"
    assert categories["("] == "delimiter"
    assert categories["1"] == "literal"


def test_token_categories_follow_overlap():
    pieces = ["def", " f", "(", "x", "):", "\n    ", '"""Doc."""', " total", " +", " 1", "  # note"]
    categories = token_categories(CODE, _char_offsets(CODE, pieces))
    assert categories == [
        "keyword", "identifier", "delimiter", "identifier", "delimiter", "whitespace",
        "docstring", "identifier", "operator", "literal", "comment",
    ]  # fmt: skip


def test_invalid_code_is_unknown():
    assert token_categories("def (", [(0, 3)]) in (["keyword"], ["unknown"])


def test_contributions_sum_to_z():
    categories = ["identifier", "identifier", "keyword", "literal"]
    scored = [True, True, False, True]
    green = [True, True, True, False]
    parts = z_contributions(categories, scored, green, gamma=0.5)
    # z = (2 - 0.5 * 3) / sqrt(0.25 * 3)
    assert sum(parts.values()) == pytest.approx((2 - 1.5) / (0.75**0.5))
    assert "keyword" not in parts
