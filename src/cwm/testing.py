"""A tiny randomly initialised causal LM for tests and offline smoke runs.

It needs no downloads: the tokenizer is a byte-level BPE trained on a few modules of
the Python standard library and the model is a two-layer GPT-2. Its outputs are
meaningless, but the logits are close to uniform, which makes every watermark easy
to embed and detect, exactly what the tests need.
"""

from __future__ import annotations

import inspect
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}<|end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _corpus() -> list[str]:
    import heapq
    import json.encoder
    import string
    import textwrap

    return [inspect.getsource(module) for module in (heapq, json.encoder, string, textwrap)]


def build_tiny_model(path: str | Path, vocab_size: int = 600, seed: int = 0) -> Path:
    """Create (once) and save a tiny tokenizer + model under ``path``."""
    path = Path(path)
    if (path / "config.json").exists():
        return path

    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    special = ["<|endoftext|>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(_corpus(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=special[1:],
    )
    fast.chat_template = CHAT_TEMPLATE

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(fast),
        n_positions=2048,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=fast.eos_token_id,
        eos_token_id=fast.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    path.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(path)
    model.save_pretrained(path)
    return path
