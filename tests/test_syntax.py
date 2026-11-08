import pytest

from cwm.watermarks.syntax import build_syntax_mask, is_syntax_token


@pytest.mark.parametrize(
    "text", ["return", " return", "for", " (", ")", ":", "+=", "int", "\n", " "]
)
def test_syntax_tokens(text):
    assert is_syntax_token(text, "official")
    assert is_syntax_token(text, "strict")


@pytest.mark.parametrize("text", [" total", "result", "_x", " 42", '"abc"', "# note"])
def test_non_syntax_tokens(text):
    assert not is_syntax_token(text, "official")
    assert not is_syntax_token(text, "strict")


@pytest.mark.parametrize("text", ["):", "    ", "\n    ", "]]", "):\n"])
def test_merged_tokens_differ_between_rules(text):
    """The official code leaves merged punctuation and indentation unprotected."""
    assert not is_syntax_token(text, "official")
    assert is_syntax_token(text, "strict")


def test_unknown_rule():
    with pytest.raises(ValueError):
        is_syntax_token("x", "nope")  # type: ignore[arg-type]


def test_build_mask():
    vocab = ["def", " foo", "(", "):", ""]
    mask = build_syntax_mask(lambda i: vocab[i], len(vocab), "official")
    assert mask.tolist() == [True, False, True, False, False]
