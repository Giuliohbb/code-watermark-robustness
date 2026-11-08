"""Watermarks embedded by the processors must be found by the matching detectors."""

from __future__ import annotations

import pytest
import torch

from cwm.config import GenerationConfig, WatermarkConfig
from cwm.generation import sample_responses
from cwm.watermarks import WatermarkFactory
from cwm.watermarks.methods import STONEProcessor

pytestmark = pytest.mark.slow
CONTEXT = "<|user|>\nwrite code<|end|>\n<|assistant|>\n```python\n"


def _factory(tiny_model, **overrides):
    model, tokenizer = tiny_model
    settings = WatermarkConfig(delta=4.0, **overrides)
    return WatermarkFactory(settings, tokenizer, int(model.config.vocab_size), model=model)


def _generate(tiny_model, factory, condition, seed=0):
    model, tokenizer = tiny_model
    sampling = GenerationConfig(n_samples=4, max_new_tokens=60, top_k=50, seed=seed)
    outputs = sample_responses(
        model, tokenizer, CONTEXT, factory.processor(condition), sampling, seed
    )
    context_ids = tokenizer(CONTEXT, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    # Score the sampled ids directly: a random model rarely samples the canonical
    # tokenization, so re-tokenizing its text would blur what is being tested.
    return context_ids, [[t for t in ids if t != eos] for _, ids in outputs]


@pytest.mark.parametrize(
    ("condition", "detector"),
    [("kgw", "kgw"), ("kgw", "ewd"), ("sweet", "sweet"), ("stone", "stone")],
)
def test_watermarked_scores_above_unwatermarked(tiny_model, condition, detector):
    factory = _factory(tiny_model)
    context, marked = _generate(tiny_model, factory, condition)
    _, plain = _generate(tiny_model, factory, "unwatermarked")
    det = factory.detector(detector)
    z_marked = [det.detect(context, ids).z for ids in marked]
    z_plain = [det.detect(context, ids).z for ids in plain]
    assert min(z_marked) > 4.0
    assert max(z_plain) < 4.0


def test_detection_without_context_skips_first_token(tiny_model):
    factory = _factory(tiny_model)
    result = factory.detector("kgw").detect([], [5, 6, 7, 8])
    assert result.n_scored == 3 and result.scored[0] is False


def test_stone_scores_only_non_syntax_tokens(tiny_model):
    factory = _factory(tiny_model)
    _, tokenizer = tiny_model
    ids = tokenizer("def total(x):\n    return x + 1\n", add_special_tokens=False)["input_ids"]
    result = factory.detector("stone").detect([0], ids)
    for token, scored in zip(ids, result.scored, strict=True):
        assert scored == (not bool(factory.syntax_mask[token]))


def test_stone_official_denominator_counts_syntax_tokens(tiny_model):
    paper = _factory(tiny_model).detector("stone")
    official = _factory(tiny_model, stone_denominator="official").detector("stone")
    _, tokenizer = tiny_model
    ids = tokenizer("def total(x):\n    return x + 1\n", add_special_tokens=False)["input_ids"]
    a, b = paper.detect([0], ids), official.detect([0], ids)
    assert a.n_green == b.n_green
    assert b.n_scored == len(ids) - a.n_scored


def test_sweet_threshold_controls_scored_tokens(tiny_model):
    low = _factory(tiny_model, sweet_entropy_threshold=0.0).detector("sweet")
    high = _factory(tiny_model, sweet_entropy_threshold=100.0).detector("sweet")
    ids = list(range(10, 30))
    assert low.detect([1], ids).n_scored == len(ids)
    assert high.detect([1], ids).n_scored == 0


def test_stone_processor_skips_rows_whose_candidate_is_syntax(tiny_model):
    factory = _factory(tiny_model)
    processor = factory.processor("stone")
    assert isinstance(processor, STONEProcessor)
    vocab = factory.vocab_size
    syntax_id = int(torch.nonzero(factory.syntax_mask).flatten()[0])
    other_id = int(torch.nonzero(~factory.syntax_mask).flatten()[0])
    scores = torch.zeros(2, vocab)
    scores[0, syntax_id] = 10.0
    scores[1, other_id] = 10.0
    out = processor(torch.tensor([[1], [1]]), scores.clone())
    assert torch.equal(out[0], scores[0])
    assert not torch.equal(out[1], scores[1])


def test_processor_rejects_wrong_vocab_size(tiny_model):
    processor = _factory(tiny_model).processor("kgw")
    with pytest.raises(ValueError):
        processor(torch.tensor([[1]]), torch.zeros(1, 7))


def test_detectors_needing_a_model(tiny_model):
    _, tokenizer = tiny_model
    factory = WatermarkFactory(WatermarkConfig(), tokenizer, len(tokenizer), model=None)
    with pytest.raises(ValueError):
        factory.detector("sweet")
    assert factory.detector("stone").name == "stone"
