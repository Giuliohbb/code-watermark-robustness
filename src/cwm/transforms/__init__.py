"""Semantics-preserving transformations, grouped into maintenance and attack families."""

from __future__ import annotations

import random
import zlib
from typing import Any

from tqdm import tqdm

from cwm.config import ExperimentConfig, TransformSpec
from cwm.data import load_problems, select_tasks
from cwm.io import read_jsonl, write_jsonl
from cwm.transforms.attacks import InsertDeadCode, Minify, RenameLocals
from cwm.transforms.base import Transform, TransformError
from cwm.transforms.maintenance import BlackFormat, StripComments

__all__ = [
    "REGISTRY",
    "BlackFormat",
    "InsertDeadCode",
    "Minify",
    "RenameLocals",
    "StripComments",
    "Transform",
    "TransformError",
    "build_transform",
    "run_transforms",
]

REGISTRY: dict[str, type[Transform]] = {
    cls.name: cls for cls in (BlackFormat, StripComments, RenameLocals, InsertDeadCode, Minify)
}
ORIGINAL = "none"


def build_transform(spec: TransformSpec) -> Transform:
    if spec.name not in REGISTRY:
        raise ValueError(f"Unknown transform {spec.name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[spec.name](**spec.params)


def _rng(seed: int, key: str) -> random.Random:
    return random.Random(seed * 1_000_003 + zlib.crc32(key.encode()))


def human_sample_id(task_id: str) -> str:
    return f"human::{task_id}::0"


def run_transforms(config: ExperimentConfig) -> None:
    """Write ``variants.jsonl``: every sample in its original form and after each transform.

    Samples are the generated code of every condition plus the human reference
    solution of every selected task (the negative set of the detectors).
    """
    run_dir = config.run_dir
    generations = read_jsonl(run_dir / "generations.jsonl")
    problems = load_problems(config.dataset, config.dataset_path)
    tasks = select_tasks(problems, config.task_limit, config.task_selection_seed)

    samples: list[dict[str, Any]] = [
        {"sample_id": g["sample_id"], "source": g["condition"], "task_id": g["task_id"],
         "sample_idx": g["sample_idx"], "code": g["code"]}
        for g in generations
    ]  # fmt: skip
    samples += [
        {"sample_id": human_sample_id(t.task_id), "source": "human", "task_id": t.task_id,
         "sample_idx": 0, "code": t.human_solution}
        for t in tasks
    ]  # fmt: skip

    transforms = [(spec.label, build_transform(spec)) for spec in config.transforms]
    variants: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="transform"):
        base = {k: sample[k] for k in ("sample_id", "source", "task_id", "sample_idx")}
        variants.append(
            {**base, "variant_id": f"{sample['sample_id']}::{ORIGINAL}", "transform": ORIGINAL,
             "family": ORIGINAL, "code": sample["code"], "ok": True, "error": None}
        )  # fmt: skip
        for label, transform in transforms:
            variant_id = f"{sample['sample_id']}::{label}"
            try:
                code = transform(sample["code"], _rng(config.transform_seed, variant_id))
                ok, error = True, None
            except TransformError as exc:
                code, ok, error = None, False, str(exc)[:300]
            variants.append(
                {**base, "variant_id": variant_id, "transform": label,
                 "family": transform.family, "code": code, "ok": ok, "error": error}
            )  # fmt: skip
    write_jsonl(run_dir / "variants.jsonl", variants)
