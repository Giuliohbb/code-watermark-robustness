"""Sampling code from the model under each watermarking condition.

Sampling settings are passed explicitly through a fresh ``GenerationConfig`` so
nothing is inherited from the model's own defaults (Qwen ships with top-k 20,
top-p 0.8, temperature 0.7 and a repetition penalty).

The watermark processor runs on the raw logits and the top-k / top-p filters run
after it. This keeps SWEET's entropy test and STONE's candidate test on the same
distribution the detectors later recompute from the model.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import asdict, dataclass
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from cwm.config import ExperimentConfig
from cwm.config import GenerationConfig as SamplingConfig
from cwm.data import Task, load_problems, select_tasks
from cwm.io import append_jsonl, environment_info, read_jsonl
from cwm.prompts import CODE_FENCE, build_chat_prompt, extract_code
from cwm.watermarks import WatermarkFactory


@dataclass
class GenerationRecord:
    sample_id: str
    task_id: str
    condition: str
    sample_idx: int
    seed: int
    response: str
    code: str
    fence_closed: bool
    hit_token_limit: bool
    n_new_tokens: int
    seconds: float
    # Token ids as sampled (before decoding); detection re-tokenizes the text, and the
    # report measures how often the two agree.
    response_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_id(condition: str, task_id: str, sample_idx: int) -> str:
    return f"{condition}::{task_id}::{sample_idx}"


def task_seed(base_seed: int, task_id: str) -> int:
    """Same seed for a task under every condition (common random numbers)."""
    return base_seed + zlib.crc32(task_id.encode()) % 1_000_000


class FenceStoppingCriteria(StoppingCriteria):
    """Stops each sequence once its generated part contains the closing code fence."""

    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> torch.BoolTensor:
        done = [
            CODE_FENCE in self.tokenizer.decode(row[self.prompt_length :], skip_special_tokens=True)
            for row in input_ids
        ]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def build_processors(
    watermark: LogitsProcessor | None, sampling: SamplingConfig
) -> LogitsProcessorList:
    processors: list[LogitsProcessor] = []
    if sampling.temperature != 1.0:
        processors.append(TemperatureLogitsWarper(sampling.temperature))
    if watermark is not None:
        processors.append(watermark)
    if sampling.top_k > 0:
        processors.append(TopKLogitsWarper(top_k=sampling.top_k))
    if sampling.top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p=sampling.top_p))
    return LogitsProcessorList(processors)


def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
    eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


@torch.no_grad()
def sample_responses(
    model: Any,
    tokenizer: Any,
    prompt: str,
    watermark: LogitsProcessor | None,
    sampling: SamplingConfig,
    seed: int,
) -> list[tuple[str, list[int]]]:
    """Draw ``n_samples`` responses for one prompt; returns (text, sampled ids w/o padding)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=sampling.max_new_tokens,
        num_return_sequences=sampling.n_samples,
        eos_token_id=eos_ids,
        pad_token_id=pad_id,
    )
    torch.manual_seed(seed)
    output = model.generate(
        **inputs,
        generation_config=config,
        logits_processor=build_processors(watermark, sampling),
        stopping_criteria=StoppingCriteriaList([FenceStoppingCriteria(tokenizer, prompt_length)]),
    )
    results = []
    for row in output[:, prompt_length:].tolist():
        n_new = len(row)
        for index, token in enumerate(row):
            if token in eos_ids:
                n_new = index + 1
                break
        results.append((tokenizer.decode(row[:n_new], skip_special_tokens=True), row[:n_new]))
    return results


def run_generation(config: ExperimentConfig, model: Any, tokenizer: Any) -> None:
    """Generate every (task, condition) pair, resuming from what is already on disk."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "generations.jsonl"
    done = {(r["task_id"], r["condition"]) for r in read_jsonl(out_path)}

    problems = load_problems(config.dataset, config.dataset_path)
    tasks: list[Task] = select_tasks(problems, config.task_limit, config.task_selection_seed)
    factory = WatermarkFactory(
        config.watermark, tokenizer, int(model.config.vocab_size), model=model
    )
    jobs = [(t, c) for t in tasks for c in config.conditions if (t.task_id, c) not in done]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for task, condition in tqdm(jobs, desc="generate"):
        prompt = build_chat_prompt(task.prompt, tokenizer)
        seed = task_seed(config.generation.seed, task.task_id)
        start = time.perf_counter()
        responses = sample_responses(
            model, tokenizer, prompt, factory.processor(condition), config.generation, seed
        )
        seconds = (time.perf_counter() - start) / len(responses)
        records = []
        for index, (response, ids) in enumerate(responses):
            code, closed = extract_code(response)
            records.append(
                GenerationRecord(
                    sample_id=sample_id(condition, task.task_id, index),
                    task_id=task.task_id,
                    condition=condition,
                    sample_idx=index,
                    seed=seed,
                    response=response,
                    code=code,
                    fence_closed=closed,
                    hit_token_limit=(not closed and len(ids) >= config.generation.max_new_tokens),
                    n_new_tokens=len(ids),
                    seconds=round(seconds, 3),
                    response_ids=ids,
                ).to_dict()
            )
        append_jsonl(out_path, records)

    info = environment_info()
    if torch.cuda.is_available():
        info["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    append_jsonl(run_dir / "environment.jsonl", [{"stage": "generate", **info}])
