"""Shared building blocks for watermark generation and detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import LogitsProcessor

from cwm.watermarks.greenlist import GreenList, z_score


@dataclass
class DetectionResult:
    """Outcome of scoring one code sample.

    Token-level fields are aligned with the *code* tokens (the context is never
    scored). ``weights`` is only set by weighted detectors such as EWD.
    """

    z: float
    n_scored: float
    n_green: float
    token_ids: list[int]
    scored: list[bool]
    green: list[bool]
    weights: list[float] | None = None

    def summary(self) -> dict[str, Any]:
        return {"z": self.z, "n_scored": self.n_scored, "n_green": self.n_green}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GreenListProcessor(LogitsProcessor):
    """Adds ``delta`` to the logits of green tokens on the rows a subclass selects.

    Subclasses decide *where* to watermark by overriding :meth:`rows_to_mark`, which
    always receives the raw logits of the model. Sampling filters such as top-k must
    therefore be placed after this processor (see :mod:`cwm.generation`).
    """

    def __init__(self, greenlist: GreenList, delta: float) -> None:
        self.greenlist = greenlist
        self.delta = delta

    def rows_to_mark(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        return torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] != self.greenlist.vocab_size:
            raise ValueError(
                f"Logits have {scores.shape[-1]} entries but the green list was built for "
                f"{self.greenlist.vocab_size}. Use model.config.vocab_size for both."
            )
        rows = self.rows_to_mark(input_ids, scores)
        if not bool(rows.any()):
            return scores
        bias = torch.zeros_like(scores)
        for row in torch.nonzero(rows).flatten().tolist():
            green = self.greenlist.mask(int(input_ids[row, -1])).to(scores.device)
            bias[row, green] = self.delta
        return scores + bias


class Detector(ABC):
    """Scores code tokens given the token context that preceded them at generation."""

    name: str = "base"
    requires_model: bool = False

    def __init__(self, greenlist: GreenList) -> None:
        self.greenlist = greenlist

    @property
    def gamma(self) -> float:
        return self.greenlist.gamma

    def green_flags(self, ids: list[int], start: int) -> list[bool]:
        """Green membership of ``ids[start:]``, each seeded by the token before it."""
        return [self.greenlist.is_green(ids[pos - 1], ids[pos]) for pos in range(start, len(ids))]

    @staticmethod
    def _join(context_ids: list[int], code_ids: list[int]) -> tuple[list[int], int]:
        """Concatenate context and code; return ids and the index of the first scorable token.

        Without context the first code token has no predecessor and cannot be scored,
        exactly as in the reference implementations (``prefix_length = 1``).
        """
        ids = list(context_ids) + list(code_ids)
        start = len(context_ids) if context_ids else 1
        return ids, start

    @abstractmethod
    def detect(self, context_ids: list[int], code_ids: list[int]) -> DetectionResult: ...

    def _unweighted_result(
        self,
        code_ids: list[int],
        start_offset: int,
        green: list[bool],
        scored: list[bool],
        denominator: float | None = None,
    ) -> DetectionResult:
        """Build a result for the count-based detectors (KGW, SWEET, STONE)."""
        pad = [False] * start_offset
        green_full = pad + green
        scored_full = pad + scored
        n_green = float(sum(g and s for g, s in zip(green_full, scored_full, strict=True)))
        n_scored = float(sum(scored_full)) if denominator is None else float(denominator)
        return DetectionResult(
            z=z_score(n_green, n_scored, self.gamma),
            n_scored=n_scored,
            n_green=n_green,
            token_ids=list(code_ids),
            scored=scored_full,
            green=green_full,
        )


@torch.no_grad()
def next_token_entropies(
    model: Any, ids: list[int], start: int, spike_modulus: float | None = None, chunk: int = 256
) -> tuple[list[float], list[float] | None]:
    """Entropy of the model's next-token distribution at every position ``>= start``.

    Position ``t`` uses the logits computed at ``t - 1``, i.e. the distribution the
    model sampled ``ids[t]`` from. Returns Shannon entropies and, if ``spike_modulus``
    is given, the spike entropy used by EWD.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids=input_ids).logits[0]
    shannon: list[float] = []
    spike: list[float] | None = [] if spike_modulus is not None else None
    for begin in range(start - 1, len(ids) - 1, chunk):
        end = min(begin + chunk, len(ids) - 1)
        probs = torch.softmax(logits[begin:end].float(), dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-30))
        shannon.extend((-(probs * log_probs).sum(dim=-1)).tolist())
        if spike is not None:
            spike.extend((probs / (1.0 + spike_modulus * probs)).sum(dim=-1).tolist())
    return shannon, spike
