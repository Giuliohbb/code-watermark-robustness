"""Every transformation must preserve behaviour and actually change the code."""

import random

import pytest

from cwm.config import TransformSpec
from cwm.transforms import (
    BlackFormat,
    InsertDeadCode,
    Minify,
    RenameLocals,
    StripComments,
    TransformError,
    build_transform,
)

PROGRAMS = {
    "loops": (
        '''from typing import List


def below_threshold(l: List[int], t: int) -> bool:
    """Return True if all numbers in l are below t."""
    # check every element
    for e in l:  # loop
        if e >= t:
            return False
    squares = [x * x for x in l]
    total = 0
    for s in squares:
        total += s
    return True
''',
        "below_threshold",
        [([1, 2, 4, 10], 100), ([1, 200], 100), ([], 0)],
    ),
    "nested": (
        """def counts(words):
    seen = {}
    def bump(key):
        seen[key] = seen.get(key, 0) + 1
    for word in words:
        bump(word.lower())
    pairs = sorted((k, v) for k, v in seen.items())
    return pairs
""",
        "counts",
        [(["a", "B", "b"],), ([],)],
    ),
    "keyword_call_and_class": (
        '''class Box:
    """A box."""

    def __init__(self, size):
        self.size = size


def area(width, height=2):
    box = Box(width)
    result = box.size * height
    return result
''',
        "area",
        [(3,), (2, 5)],
    ),
    "global_state": (
        """COUNTER = 0


def step(n):
    global COUNTER
    COUNTER += n
    value = COUNTER * 2
    return value
""",
        "step",
        [(1,), (4,)],
    ),
}

TRANSFORMS = [
    BlackFormat(),
    StripComments(),
    RenameLocals(0.5),
    RenameLocals(1.0),
    InsertDeadCode(3),
    Minify(),
]


def _run(code, entry_point, args):
    namespace = {}
    exec(code, namespace)
    return namespace[entry_point](*args)


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: f"{t.name}")
@pytest.mark.parametrize("program", list(PROGRAMS))
def test_behaviour_is_preserved(transform, program):
    code, entry_point, cases = PROGRAMS[program]
    for seed in range(3):
        new_code = transform(code, random.Random(seed))
        compile(new_code, "<transformed>", "exec")
        for args in cases:
            assert _run(new_code, entry_point, args) == _run(code, entry_point, args)


def test_strip_comments_removes_comments_and_docstrings_only():
    code = PROGRAMS["loops"][0]
    out = StripComments()(code)
    assert "#" not in out and '"""' not in out
    assert "for e in l:" in out and "squares = [x * x for x in l]" in out


def test_docstring_only_function_gets_pass():
    out = StripComments()('def f():\n    """Only doc."""\n')
    assert "pass" in out


def test_rename_keeps_parameters_and_functions():
    code = PROGRAMS["keyword_call_and_class"][0]
    out = RenameLocals(1.0)(code, random.Random(0))
    assert "def area(width, height=2):" in out
    assert "box =" not in out and "result =" not in out


def test_rename_respects_global_declarations():
    out = RenameLocals(1.0)(PROGRAMS["global_state"][0], random.Random(0))
    assert "global COUNTER" in out and "value =" not in out


def test_rename_fraction_changes_fewer_names():
    code = PROGRAMS["loops"][0]
    half = RenameLocals(0.5)(code, random.Random(0))
    full = RenameLocals(1.0)(code, random.Random(0))
    changed = lambda out: sum(n not in out for n in ("e in l", "squares", "total", "s in"))  # noqa: E731
    assert changed(half) < changed(full)


def test_dead_code_adds_statements_after_docstring():
    code = PROGRAMS["loops"][0]
    out = InsertDeadCode(3)(code, random.Random(1))
    assert len(out.splitlines()) > len(code.splitlines())
    body = out.split("def below_threshold")[1].splitlines()
    assert '"""' in body[1]


def test_minify_shrinks_code():
    code = PROGRAMS["loops"][0]
    assert len(Minify()(code)) < len(code)


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: t.name)
def test_invalid_code_raises_transform_error(transform):
    with pytest.raises(TransformError):
        transform("def broken(:\n    return", random.Random(0))


def test_registry_builds_from_spec():
    transform = build_transform(TransformSpec("rename", {"fraction": 0.5}))
    assert isinstance(transform, RenameLocals) and transform.fraction == 0.5
    assert TransformSpec("rename", {"fraction": 0.5}).label == "rename_0.5"
    with pytest.raises(ValueError):
        build_transform(TransformSpec("unknown"))
