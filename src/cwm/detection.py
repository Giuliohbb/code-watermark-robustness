"""Scoring every code variant with every detector.

Detection always re-tokenizes the code text, as a real detector only sees text. The
chat prompt (ending at the opening ```python fence) is used as context, which is how
SWEET recovers the generation-time entropy; human solutions get the same context.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from tqdm import tqdm

from cwm.config import DETECTOR_SOURCE, ExperimentConfig
from cwm.data import load_problems, select_tasks
from cwm.io import append_jsonl, environment_info, read_jsonl
from cwm.prompts import build_chat_prompt
from cwm.watermarks import WatermarkFactory


def detection_targets(detector: str, negatives: list[str]) -> set[str]:
    """Sources a detector is run on: its own watermarked samples plus the negatives."""
    return {DETECTOR_SOURCE[detector], *negatives}


def run_detection(
    config: ExperimentConfig, tokenizer: Any, vocab_size: int, model: Any = None
) -> None:
    run_dir = config.run_dir
    variants = [v for v in read_jsonl(run_dir / "variants.jsonl") if v["ok"] and v["code"]]
    scores_path = run_dir / "scores.jsonl"
    tokens_path = run_dir / "tokens.jsonl"
    done = {(r["detector"], r["variant_id"]) for r in read_jsonl(scores_path)}

    problems = load_problems(config.dataset, config.dataset_path)
    tasks = select_tasks(problems, config.task_limit, config.task_selection_seed)
    context: dict[str, list[int]] = defaultdict(list)
    if config.detection.use_prompt_context:
        for task in tasks:
            prompt = build_chat_prompt(task.prompt, tokenizer)
            context[task.task_id] = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    factory = WatermarkFactory(config.watermark, tokenizer, vocab_size, model=model)
    for name in config.detectors:
        detector = factory.detector(name)
        target = DETECTOR_SOURCE[name]
        sources = detection_targets(name, config.evaluation.negatives)
        todo = [
            v for v in variants if v["source"] in sources and (name, v["variant_id"]) not in done
        ]
        score_rows: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        for variant in tqdm(todo, desc=f"detect[{name}]"):
            code_ids = tokenizer(variant["code"], add_special_tokens=False)["input_ids"]
            result = detector.detect(context[variant["task_id"]], code_ids)
            score_rows.append(
                {
                    "detector": name,
                    "variant_id": variant["variant_id"],
                    "sample_id": variant["sample_id"],
                    "source": variant["source"],
                    "task_id": variant["task_id"],
                    "sample_idx": variant["sample_idx"],
                    "transform": variant["transform"],
                    "family": variant["family"],
                    "label": int(variant["source"] == target),
                    "n_tokens": len(code_ids),
                    **result.summary(),
                }
            )
            if config.detection.save_token_details:
                token_rows.append(
                    {"detector": name, "variant_id": variant["variant_id"], **result.to_dict()}
                )
            if len(score_rows) >= 200:
                append_jsonl(scores_path, score_rows)
                append_jsonl(tokens_path, token_rows)
                score_rows, token_rows = [], []
        append_jsonl(scores_path, score_rows)
        append_jsonl(tokens_path, token_rows)
    append_jsonl(run_dir / "environment.jsonl", [{"stage": "detect", **environment_info()}])
