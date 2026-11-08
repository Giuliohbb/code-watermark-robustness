"""Benchmark tasks (HumanEval+ / MBPP+ through EvalPlus) and human reference solutions."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    entry_point: str
    human_solution: str


def load_problems(dataset: str, dataset_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load EvalPlus problems, or a local JSONL in the same format (used by tests)."""
    if dataset_path is not None:
        with open(dataset_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return {row["task_id"]: row for row in rows}
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus

        return get_human_eval_plus()
    if dataset == "mbpp":
        from evalplus.data import get_mbpp_plus

        return get_mbpp_plus()
    raise ValueError(f"Unknown dataset: {dataset}")


def problems_hash(dataset: str, dataset_path: str | None = None) -> str:
    """Key under which EvalPlus caches the expected outputs."""
    if dataset_path is not None:
        digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()[:16]
        return f"local-{digest}"
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus_hash

        return get_human_eval_plus_hash()
    from evalplus.data import get_mbpp_plus_hash

    return get_mbpp_plus_hash()


def select_tasks(problems: dict[str, dict[str, Any]], limit: int | None, seed: int) -> list[Task]:
    """Deterministic subset of tasks, kept in the dataset's own order."""
    task_ids = list(problems)
    if limit is not None and limit < len(task_ids):
        chosen = set(random.Random(seed).sample(task_ids, limit))
        task_ids = [task_id for task_id in task_ids if task_id in chosen]
    return [
        Task(
            task_id=task_id,
            prompt=problems[task_id]["prompt"],
            entry_point=problems[task_id]["entry_point"],
            # EvalPlus builds its own oracle from prompt + canonical_solution, for both datasets.
            human_solution=problems[task_id]["prompt"] + problems[task_id]["canonical_solution"],
        )
        for task_id in task_ids
    ]
