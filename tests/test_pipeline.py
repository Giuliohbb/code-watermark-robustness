"""End-to-end run of every stage on the tiny model and a three-task local dataset."""

from __future__ import annotations

import pytest

from cwm.config import load_config
from cwm.detection import run_detection
from cwm.evaluation.functional import run_functional
from cwm.evaluation.report import run_report
from cwm.generation import run_generation
from cwm.io import read_jsonl
from cwm.prompts import build_chat_prompt, extract_code
from cwm.transforms import run_transforms

pytestmark = pytest.mark.slow


def test_extract_code():
    assert extract_code("x = 1\n```\nexplanation") == ("x = 1\n", True)
    assert extract_code("x = 1") == ("x = 1", False)


def test_chat_prompt_ends_inside_code_block(tiny_model):
    _, tokenizer = tiny_model
    prompt = build_chat_prompt("def f():\n    pass", tokenizer)
    assert prompt.endswith("```python\n")
    assert "def f():" in prompt


def test_full_pipeline(tmp_path, tiny_model, tiny_model_dir, mini_dataset):
    config = load_config(
        "configs/smoke.yaml",
        {
            "output_dir": str(tmp_path),
            "dataset_path": str(mini_dataset),
            "model": {"name": str(tiny_model_dir)},
            "generation": {"n_samples": 2, "max_new_tokens": 24},
        },
    )
    model, tokenizer = tiny_model
    run_generation(config, model, tokenizer)
    run_generation(config, model, tokenizer)  # resuming must not duplicate work
    generations = read_jsonl(config.run_dir / "generations.jsonl")
    assert len(generations) == 3 * len(config.conditions) * 2

    run_transforms(config)
    variants = read_jsonl(config.run_dir / "variants.jsonl")
    human = [v for v in variants if v["source"] == "human"]
    assert len(human) == 3 * (1 + len(config.transforms))
    assert all(v["ok"] for v in human), "every transform must apply to the reference solutions"

    run_functional(config)
    functional = read_jsonl(config.run_dir / "functional.jsonl")
    human_results = [r for r in functional if r["source"] == "human"]
    assert human_results and all(r["passed"] for r in human_results)

    run_detection(config, tokenizer, int(model.config.vocab_size), model=model)
    scores = read_jsonl(config.run_dir / "scores.jsonl")
    assert {s["detector"] for s in scores} == set(config.detectors)

    report = run_report(config, tokenizer=tokenizer)
    text = report.read_text()
    assert "## Detection" in text and "## Where the signal lives" in text
