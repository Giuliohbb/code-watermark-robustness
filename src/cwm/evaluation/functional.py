"""Functional correctness with the EvalPlus test harness.

``evalplus.evaluate`` insists on samples for every task of the dataset, so the pilot
(a subset) calls its per-sample checker directly. Expected outputs are computed once
from the reference solutions and cached by EvalPlus.
"""

from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from cwm.config import ExperimentConfig
from cwm.data import load_problems, problems_hash, select_tasks
from cwm.io import read_jsonl, write_jsonl

PASS = "pass"


def _expected_outputs(config: ExperimentConfig, problems: dict[str, Any]) -> dict[str, Any]:
    from evalplus.evaluate import get_groundtruth

    only_not_none: list[str] = []
    if config.dataset == "mbpp":
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

        only_not_none = list(MBPP_OUTPUT_NOT_NONE_TASKS)
    subset_key = zlib.crc32(",".join(sorted(problems)).encode())
    cache_key = f"{problems_hash(config.dataset, config.dataset_path)}-{subset_key:08x}"
    return get_groundtruth(problems, cache_key, only_not_none)


def _check(dataset: str, problem: dict[str, Any], solution: str, expected: Any, ident: str) -> dict:
    from evalplus.evaluate import check_correctness

    result = check_correctness(dataset, 0, problem, solution, expected, identifier=ident)
    return {
        "variant_id": ident,
        "base_status": result["base"][0],
        "plus_status": result["plus"][0],
    }


def run_functional(config: ExperimentConfig) -> None:
    """Write ``functional.jsonl`` with base/plus test status of every valid variant."""
    run_dir = config.run_dir
    out_path = run_dir / "functional.jsonl"
    variants = [v for v in read_jsonl(run_dir / "variants.jsonl") if v["ok"] and v["code"]]
    done = {r["variant_id"] for r in read_jsonl(out_path)}
    todo = [v for v in variants if v["variant_id"] not in done]
    if not todo:
        return

    all_problems = load_problems(config.dataset, config.dataset_path)
    tasks = select_tasks(all_problems, config.task_limit, config.task_selection_seed)
    problems = {t.task_id: all_problems[t.task_id] for t in tasks}
    expected = _expected_outputs(config, problems)

    meta = {v["variant_id"]: v for v in todo}
    rows = read_jsonl(out_path)
    with ProcessPoolExecutor(max_workers=config.evaluation.functional_workers) as pool:
        futures = [
            pool.submit(
                _check,
                config.dataset,
                problems[v["task_id"]],
                v["code"],
                expected[v["task_id"]],
                v["variant_id"],
            )
            for v in todo
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="functional"):
            row = future.result()
            variant = meta[row["variant_id"]]
            rows.append(
                {
                    **row,
                    "sample_id": variant["sample_id"],
                    "source": variant["source"],
                    "task_id": variant["task_id"],
                    "transform": variant["transform"],
                    "passed": row["base_status"] == PASS and row["plus_status"] == PASS,
                }
            )
    rows.sort(key=lambda r: r["variant_id"])
    write_jsonl(out_path, rows)
