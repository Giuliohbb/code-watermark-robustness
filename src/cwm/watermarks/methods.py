"""KGW, SWEET, STONE and EWD.

Each generation method is a :class:`GreenListProcessor` that decides where to add the
bias; each detector decides which code tokens to score and how to weight them.

References
----------
KGW    Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.
SWEET  Lee et al., "Who Wrote this Code? Watermarking for Code Generation", ACL 2024.
STONE  Kim, Park and Han, "Marking Code Without Breaking It", Findings of EACL 2026.
EWD    Lu et al., "An Entropy-based Text Watermarking Detection Method", ACL 2024.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from cwm.watermarks.base import DetectionResult, Detector, GreenListProcessor, next_token_entropies
from cwm.watermarks.greenlist import GreenList

StoneCandidate = Literal["argmax", "sample"]
StoneDenominator = Literal["paper", "official"]


# --------------------------------------------------------------------------- KGW
class KGWProcessor(GreenListProcessor):
    """Biases green tokens at every generation step."""


class KGWDetector(Detector):
    """Counts green tokens over every code token, each with the same weight."""

    name = "kgw"

    def detect(self, context_ids: list[int], code_ids: list[int]) -> DetectionResult:
        ids, start = self._join(context_ids, code_ids)
        green = self.green_flags(ids, start)
        offset = len(code_ids) - len(green)
        return self._unweighted_result(code_ids, offset, green, [True] * len(green))


# ------------------------------------------------------------------------- SWEET
def shannon_entropy(scores: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(scores.float(), dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)


class SWEETProcessor(GreenListProcessor):
    """Biases green tokens only where the next-token entropy exceeds a threshold."""

    def __init__(self, greenlist: GreenList, delta: float, entropy_threshold: float) -> None:
        super().__init__(greenlist, delta)
        self.entropy_threshold = entropy_threshold

    def rows_to_mark(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        return shannon_entropy(scores) > self.entropy_threshold


class SWEETDetector(Detector):
    """Scores only tokens generated under high entropy; needs the generation model."""

    name = "sweet"
    requires_model = True

    def __init__(self, greenlist: GreenList, model: Any, entropy_threshold: float) -> None:
        super().__init__(greenlist)
        self.model = model
        self.entropy_threshold = entropy_threshold

    def detect(self, context_ids: list[int], code_ids: list[int]) -> DetectionResult:
        ids, start = self._join(context_ids, code_ids)
        green = self.green_flags(ids, start)
        entropy, _ = next_token_entropies(self.model, ids, start)
        scored = [h > self.entropy_threshold for h in entropy]
        offset = len(code_ids) - len(green)
        return self._unweighted_result(code_ids, offset, green, scored)


# ------------------------------------------------------------------------- STONE
class STONEProcessor(GreenListProcessor):
    """Skips the bias when the token the model is about to produce is syntactic.

    The paper (Algorithm 1) samples a candidate token; the official code uses the
    argmax of the raw distribution. ``candidate`` selects either behaviour and
    defaults to the official one, which produced the published numbers.
    """

    def __init__(
        self,
        greenlist: GreenList,
        delta: float,
        syntax_mask: torch.Tensor,
        candidate: StoneCandidate = "argmax",
    ) -> None:
        super().__init__(greenlist, delta)
        if syntax_mask.shape[0] != greenlist.vocab_size:
            raise ValueError("syntax_mask must cover the whole vocabulary")
        self.syntax_mask = syntax_mask
        self.candidate = candidate

    def rows_to_mark(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if self.candidate == "argmax":
            candidates = scores.argmax(dim=-1)
        elif self.candidate == "sample":
            probs = torch.softmax(scores.float(), dim=-1)
            candidates = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            raise ValueError(f"Unknown STONE candidate rule: {self.candidate!r}")
        if self.syntax_mask.device != scores.device:
            self.syntax_mask = self.syntax_mask.to(scores.device)  # moved once, not per step
        is_syntax = self.syntax_mask[candidates]
        return ~is_syntax


class STONEDetector(Detector):
    """Scores only non-syntactic tokens; needs no model.

    ``denominator="paper"`` uses the number of scored (non-syntactic) tokens, as in
    Algorithm 2 of the paper. ``"official"`` reproduces the official code, whose
    denominator is the number of *syntactic* tokens (see docs/IMPLEMENTATION_NOTES.md).
    """

    name = "stone"

    def __init__(
        self,
        greenlist: GreenList,
        syntax_mask: torch.Tensor,
        denominator: StoneDenominator = "paper",
    ) -> None:
        super().__init__(greenlist)
        self.syntax_mask = syntax_mask
        self.denominator = denominator

    def detect(self, context_ids: list[int], code_ids: list[int]) -> DetectionResult:
        ids, start = self._join(context_ids, code_ids)
        green = self.green_flags(ids, start)
        scored = [not bool(self.syntax_mask[token]) for token in ids[start:]]
        offset = len(code_ids) - len(green)
        if self.denominator == "paper":
            return self._unweighted_result(code_ids, offset, green, scored)
        if self.denominator == "official":
            n_syntax = len(scored) - sum(scored)
            return self._unweighted_result(code_ids, offset, green, scored, denominator=n_syntax)
        raise ValueError(f"Unknown STONE denominator: {self.denominator!r}")


# --------------------------------------------------------------------------- EWD
def ewd_spike_modulus(gamma: float, delta: float) -> float:
    alpha = math.exp(delta)
    return ((1.0 - gamma) * (alpha - 1.0)) / (1.0 - gamma + alpha * gamma)


class EWDDetector(Detector):
    """Detects a KGW watermark with entropy-dependent token weights.

    Weights are the spike entropy of each position minus the minimum over the
    sample, and the statistic is the weighted z-score of the official code.
    """

    name = "ewd"
    requires_model = True

    def __init__(self, greenlist: GreenList, model: Any, delta: float) -> None:
        super().__init__(greenlist)
        self.model = model
        self.spike_modulus = ewd_spike_modulus(greenlist.gamma, delta)

    def detect(self, context_ids: list[int], code_ids: list[int]) -> DetectionResult:
        ids, start = self._join(context_ids, code_ids)
        green = self.green_flags(ids, start)
        _, spike = next_token_entropies(self.model, ids, start, spike_modulus=self.spike_modulus)
        assert spike is not None
        offset = len(code_ids) - len(green)
        if not spike:
            return self._unweighted_result(code_ids, offset, green, [])
        floor = min(spike)
        weights = [s - floor for s in spike]
        sum_w = sum(weights)
        sum_w2 = sum(w * w for w in weights)
        green_w = sum(w for w, g in zip(weights, green, strict=True) if g)
        gamma = self.gamma
        z = (
            (green_w - gamma * sum_w) / math.sqrt(sum_w2 * gamma * (1.0 - gamma))
            if sum_w2 > 0
            else float("nan")
        )
        pad = [False] * offset
        return DetectionResult(
            z=z,
            n_scored=float(len(weights)),
            n_green=float(sum(green)),
            token_ids=list(code_ids),
            scored=pad + [True] * len(weights),
            green=pad + green,
            weights=[0.0] * offset + weights,
        )
