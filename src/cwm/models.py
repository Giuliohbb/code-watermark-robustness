"""Loading the model and tokenizer once, with explicit precision and device."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cwm.config import ModelConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; set device: cpu")
    return device


def load_tokenizer(cfg: ModelConfig) -> Any:
    return AutoTokenizer.from_pretrained(cfg.name, revision=cfg.revision)


def output_vocab_size(cfg: ModelConfig) -> int:
    """Size of the output layer, read from the config so no weights are loaded."""
    return int(AutoConfig.from_pretrained(cfg.name, revision=cfg.revision).vocab_size)


def load_model(cfg: ModelConfig) -> tuple[Any, Any]:
    if cfg.dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    device = resolve_device(cfg.device)
    tokenizer = load_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name, revision=cfg.revision, torch_dtype=_DTYPES[cfg.dtype]
    ).to(device)
    model.eval()
    return model, tokenizer
