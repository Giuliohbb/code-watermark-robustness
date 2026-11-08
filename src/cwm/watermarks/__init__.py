"""Watermark methods and a factory that builds them consistently."""

from __future__ import annotations

from typing import Any

import torch
from transformers import LogitsProcessor

from cwm.config import WatermarkConfig
from cwm.watermarks.base import DetectionResult, Detector, GreenListProcessor
from cwm.watermarks.greenlist import GreenList, z_score
from cwm.watermarks.methods import (
    EWDDetector,
    KGWDetector,
    KGWProcessor,
    STONEDetector,
    STONEProcessor,
    SWEETDetector,
    SWEETProcessor,
)
from cwm.watermarks.syntax import build_syntax_mask, is_syntax_token

__all__ = [
    "DetectionResult",
    "Detector",
    "EWDDetector",
    "GreenList",
    "GreenListProcessor",
    "KGWDetector",
    "KGWProcessor",
    "STONEDetector",
    "STONEProcessor",
    "SWEETDetector",
    "SWEETProcessor",
    "WatermarkFactory",
    "build_syntax_mask",
    "is_syntax_token",
    "z_score",
]


class WatermarkFactory:
    """Builds processors and detectors that share one green list and one syntax mask.

    ``vocab_size`` must be the size of the model's output layer
    (``model.config.vocab_size``), which can be larger than ``len(tokenizer)``.
    """

    def __init__(
        self, settings: WatermarkConfig, tokenizer: Any, vocab_size: int, model: Any = None
    ) -> None:
        self.settings = settings
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.model = model
        self.greenlist = GreenList(
            vocab_size, settings.gamma, settings.hash_key, settings.greenlist_cache_size
        )
        self._syntax_mask: torch.Tensor | None = None

    @property
    def syntax_mask(self) -> torch.Tensor:
        if self._syntax_mask is None:
            self._syntax_mask = build_syntax_mask(
                self._decode_one, self.vocab_size, self.settings.stone_syntax_rule
            )
        return self._syntax_mask

    def _decode_one(self, token_id: int) -> str:
        # The output layer is usually padded beyond the tokenizer's vocabulary.
        if token_id >= len(self.tokenizer):
            return ""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def processor(self, condition: str) -> LogitsProcessor | None:
        s = self.settings
        if condition == "unwatermarked":
            return None
        if condition == "kgw":
            return KGWProcessor(self.greenlist, s.delta)
        if condition == "sweet":
            return SWEETProcessor(self.greenlist, s.delta, s.sweet_entropy_threshold)
        if condition == "stone":
            return STONEProcessor(self.greenlist, s.delta, self.syntax_mask, s.stone_candidate)
        raise ValueError(f"Unknown generation condition: {condition!r}")

    def detector(self, name: str) -> Detector:
        s = self.settings
        if name == "kgw":
            return KGWDetector(self.greenlist)
        if name == "stone":
            return STONEDetector(self.greenlist, self.syntax_mask, s.stone_denominator)
        if name in ("sweet", "ewd") and self.model is None:
            raise ValueError(f"The {name} detector needs the generation model")
        if name == "sweet":
            return SWEETDetector(self.greenlist, self.model, s.sweet_entropy_threshold)
        if name == "ewd":
            return EWDDetector(self.greenlist, self.model, s.delta)
        raise ValueError(f"Unknown detector: {name!r}")
