"""Common interface for semantics-preserving code transformations."""

from __future__ import annotations

import builtins
import keyword
import random
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

Family = Literal["maintenance", "attack"]

RESERVED_NAMES = (
    frozenset(keyword.kwlist) | frozenset(keyword.softkwlist) | frozenset(dir(builtins))
)


class TransformError(ValueError):
    """Raised when a transformation cannot be applied (e.g. the code does not parse)."""


class Transform(ABC):
    """Maps Python source to an equivalent program.

    ``family`` separates routine maintenance (what any code goes through) from edits
    whose only purpose is to remove a watermark.
    """

    name: ClassVar[str]
    family: ClassVar[Family]

    @abstractmethod
    def apply(self, code: str, rng: random.Random) -> str: ...

    def __call__(self, code: str, rng: random.Random | None = None) -> str:
        return self.apply(code, rng or random.Random(0))


def fresh_name(candidates: list[str], taken: set[str], rng: random.Random) -> str:
    """A realistic identifier not already used in the program."""
    pool = list(candidates)
    rng.shuffle(pool)
    for name in pool:
        if name not in taken and name not in RESERVED_NAMES:
            taken.add(name)
            return name
    suffix = 2
    while True:
        for name in pool:
            candidate = f"{name}{suffix}"
            if candidate not in taken:
                taken.add(candidate)
                return candidate
        suffix += 1
