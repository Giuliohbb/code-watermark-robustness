"""Context-seeded vocabulary partition shared by every watermark in this project.

All schemes (KGW, SWEET, STONE and the EWD detector) use the same partition so that
differences between methods come from *where* they watermark and score, never from
the hashing itself. The scheme follows the "time" seeding of MarkLLM with a context
width of one token, which is also what the official STONE code uses::

    seed = hash_key * (prev_token % vocab_size)
    green = randperm(vocab_size, seed)[: int(gamma * vocab_size)]

The random generator always runs on CPU. ``torch.randperm`` returns different
permutations on CPU and CUDA for the same seed, so seeding on the model device would
make GPU generation and CPU detection disagree.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


class GreenList:
    """Deterministic green/red split of the vocabulary, keyed by the previous token."""

    def __init__(
        self, vocab_size: int, gamma: float, hash_key: int, cache_size: int = 1024
    ) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key
        self.green_size = int(vocab_size * gamma)
        self._cached_mask = lru_cache(maxsize=cache_size)(self._compute_mask)

    def _compute_mask(self, prev_token: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.hash_key * (prev_token % self.vocab_size))
        permutation = torch.randperm(self.vocab_size, generator=generator)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        mask[permutation[: self.green_size]] = True
        return mask

    def mask(self, prev_token: int) -> torch.Tensor:
        """Boolean mask over the vocabulary, ``True`` for green tokens (CPU tensor)."""
        return self._cached_mask(int(prev_token))

    def is_green(self, prev_token: int, token: int) -> bool:
        return bool(self.mask(prev_token)[int(token)])


def z_score(green: float, total: float, gamma: float) -> float:
    """One-proportion z-test used by KGW, SWEET and STONE.

    Returns ``nan`` when nothing was scored, so empty samples are never mistaken for
    unwatermarked ones.
    """
    if total <= 0:
        return float("nan")
    return (green - gamma * total) / math.sqrt(total * gamma * (1.0 - gamma))
