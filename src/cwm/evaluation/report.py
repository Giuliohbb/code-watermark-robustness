"""Aggregate a run into CSV tables, a Markdown report and (optionally) figures."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from cwm.config import DETECTOR_SOURCE, ExperimentConfig
from cwm.evaluation.metrics import (
    auroc,
    bootstrap_ci,
    pass_at_1,
    rate_above,
    threshold_at_fpr,
)
from cwm.evaluation.signal import CATEGORIES, token_categories, z_contributions
from cwm.io import iter_jsonl, read_jsonl
from cwm.transforms import ORIGINAL

COUNT_BASED = {"kgw", "sweet", "stone"}


def retokenization_agreement(code: str, response_ids: list[int], tokenizer: Any) -> float:
    """Share of re-tokenized code tokens that equal the sampled tokens, position by position.

    Detectors only see text, so a token the model sampled in a non-canonical way
    (e.g. ``"de" + "f"`` instead of ``"def"``) is scored as something else.
    """
    code_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    if not code_ids:
        return float("nan")
    matching = 0
    for retokenized, sampled in zip(code_ids, response_ids, strict=False):
        if retokenized != sampled:
            break
        matching += 1
    return matching / len(code_ids)


def generation_table(generations: list[dict[str, Any]], tokenizer: Any = None) -> pd.DataFrame:
    if not generations:
        return pd.DataFrame()
    df = pd.DataFrame(generations)
    aggregations: dict[str, tuple[str, str]] = {
        "samples": ("sample_id", "count"),
        "mean_new_tokens": ("n_new_tokens", "mean"),
        "fence_closed": ("fence_closed", "mean"),
        "hit_token_limit": ("hit_token_limit", "mean"),
        "seconds_per_sample": ("seconds", "mean"),
    }
    if tokenizer is not None and "response_ids" in df:
        df["retokenization_agreement"] = [
            retokenization_agreement(code, ids, tokenizer)
            for code, ids in zip(df["code"], df["response_ids"], strict=True)
        ]
        aggregations["retokenization_agreement"] = ("retokenization_agreement", "mean")
    return df.groupby("condition").agg(**aggregations).reset_index()


def functional_tables(functional: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pass@1 per (source, transform), and how often a transform changed a test outcome."""
    if not functional:
        return pd.DataFrame(), pd.DataFrame()
    by_group: dict[tuple[str, str], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    original: dict[str, bool] = {}
    for row in functional:
        by_group[(row["source"], row["transform"])][row["task_id"]].append(row["passed"])
        if row["transform"] == ORIGINAL:
            original[row["sample_id"]] = row["passed"]
    pass_rows = [
        {"source": s, "transform": t, "pass@1": pass_at_1(tasks), "tasks": len(tasks)}
        for (s, t), tasks in sorted(by_group.items())
    ]
    changed: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in functional:
        if row["transform"] != ORIGINAL and row["sample_id"] in original:
            changed[(row["source"], row["transform"])].append(
                row["passed"] != original[row["sample_id"]]
            )
    preservation = [
        {"source": s, "transform": t, "outcome_changed": sum(v) / len(v), "n": len(v)}
        for (s, t), v in sorted(changed.items())
    ]
    return pd.DataFrame(pass_rows), pd.DataFrame(preservation)


def _auroc_by_task(groups: list[object]) -> float:
    pos = [z for g in groups for z in g[0]]  # type: ignore[index]
    neg = [z for g in groups for z in g[1]]  # type: ignore[index]
    return auroc(pos, neg)


def detection_table(
    scores: list[dict[str, Any]],
    negatives: list[str],
    fpr_targets: list[float],
    n_bootstrap: int = 0,
) -> pd.DataFrame:
    """AUROC, TPR/FPR at fixed clean-negative thresholds and z retention per transform."""
    if not scores:
        return pd.DataFrame()
    df = pd.DataFrame(scores)
    rows = []
    for detector, d in df.groupby("detector"):
        target = DETECTOR_SOURCE[detector]
        positives = d[d["source"] == target]
        clean_pos = positives[positives["transform"] == ORIGINAL].set_index("sample_id")["z"]
        for negative in negatives:
            neg = d[d["source"] == negative]
            if neg.empty:
                continue
            clean_neg = neg[neg["transform"] == ORIGINAL]["z"].tolist()
            thresholds = {f: threshold_at_fpr(clean_neg, f) for f in fpr_targets}
            for transform in sorted(d["transform"].unique(), key=lambda t: (t != ORIGINAL, t)):
                pos_t = positives[positives["transform"] == transform]
                neg_t = neg[neg["transform"] == transform]["z"].tolist()
                paired = pos_t.set_index("sample_id")["z"].dropna()
                common = paired.index.intersection(clean_pos.dropna().index)
                before = clean_pos.loc[common].mean() if len(common) else math.nan
                after = paired.loc[common].mean() if len(common) else math.nan
                if pos_t.empty and not neg_t:
                    continue
                row = {
                    "detector": detector,
                    "negatives": negative,
                    "transform": transform,
                    "family": d.loc[d["transform"] == transform, "family"].iloc[0],
                    "n_pos": int(pos_t["z"].notna().sum()),
                    "n_neg": len(neg_t),
                    "auroc": auroc(pos_t["z"].tolist(), neg_t),
                    "mean_z": float(pos_t["z"].mean()) if len(pos_t) else math.nan,
                    "z_retention": after / before
                    if before and not math.isnan(before)
                    else math.nan,
                }
                if n_bootstrap > 0:
                    neg_frame = neg[neg["transform"] == transform]
                    groups = {
                        task: (
                            pos_t.loc[pos_t["task_id"] == task, "z"].tolist(),
                            neg_frame.loc[neg_frame["task_id"] == task, "z"].tolist(),
                        )
                        for task in sorted(set(pos_t["task_id"]) | set(neg_frame["task_id"]))
                    }
                    row["auroc_ci_low"], row["auroc_ci_high"] = bootstrap_ci(
                        groups, _auroc_by_task, n_resamples=n_bootstrap
                    )
                for fpr, threshold in thresholds.items():
                    row[f"tpr@{fpr:g}"] = rate_above(pos_t["z"].tolist(), threshold)
                    row[f"fpr@{fpr:g}"] = rate_above(neg_t, threshold)
                rows.append(row)
    return pd.DataFrame(rows)


def signal_table(
    tokens_path: Path,
    variants: dict[str, dict[str, Any]],
    tokenizer: Any,
    gamma: float,
) -> pd.DataFrame:
    """Mean z contribution and share of scored tokens per lexical category.

    Only original (untransformed) watermarked samples of count-based detectors are used.
    """
    if not tokens_path.exists():
        return pd.DataFrame()
    contributions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    shares: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in iter_jsonl(tokens_path):
        variant = variants.get(row["variant_id"])
        detector = row["detector"]
        if (
            variant is None
            or detector not in COUNT_BASED
            or variant["transform"] != ORIGINAL
            or variant["source"] != DETECTOR_SOURCE[detector]
        ):
            continue
        encoded = tokenizer(variant["code"], add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != row["token_ids"]:
            continue
        categories = token_categories(variant["code"], encoded["offset_mapping"])
        parts = z_contributions(categories, row["scored"], row["green"], gamma)
        n_scored = sum(row["scored"]) or 1
        for category in CATEGORIES:
            contributions[detector][category].append(parts.get(category, 0.0))
            count = sum(
                1 for c, s in zip(categories, row["scored"], strict=True) if s and c == category
            )
            shares[detector][category].append(count / n_scored)
    rows = []
    for detector in sorted(contributions):
        for category in CATEGORIES:
            values = contributions[detector][category]
            if not any(shares[detector][category]):
                continue
            rows.append(
                {
                    "detector": detector,
                    "category": category,
                    "mean_z_contribution": sum(values) / len(values),
                    "share_of_scored_tokens": sum(shares[detector][category]) / len(values),
                    "samples": len(values),
                }
            )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_no data_\n"

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return "nan" if math.isnan(value) else format(value, floatfmt)
        return str(value)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body]) + "\n"


def plot_detection(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Optional figures; skipped silently when matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if table.empty:
        return []
    paths = []
    subset = table[table["negatives"] == table["negatives"].iloc[0]]
    for metric in ("auroc", "z_retention"):
        pivot = subset.pivot(index="transform", columns="detector", values=metric)
        order = [ORIGINAL, *sorted(t for t in pivot.index if t != ORIGINAL)]
        pivot = pivot.reindex(order)
        ax = pivot.plot(kind="bar", figsize=(8, 4), width=0.8)
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.set_title(f"{metric} by transformation")
        ax.legend(title="detector", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = out_dir / f"{metric}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def run_report(config: ExperimentConfig, tokenizer: Any = None) -> Path:
    run_dir = config.run_dir
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    generations = read_jsonl(run_dir / "generations.jsonl")
    functional = read_jsonl(run_dir / "functional.jsonl")
    scores = read_jsonl(run_dir / "scores.jsonl")
    variants = {v["variant_id"]: v for v in read_jsonl(run_dir / "variants.jsonl")}

    tables = {"generation": generation_table(generations, tokenizer)}
    tables["pass_at_1"], tables["outcome_changes"] = functional_tables(functional)
    tables["detection"] = detection_table(
        scores,
        config.evaluation.negatives,
        config.evaluation.fpr_targets,
        n_bootstrap=config.evaluation.bootstrap_samples,
    )
    if tokenizer is not None:
        tables["signal"] = signal_table(
            run_dir / "tokens.jsonl", variants, tokenizer, config.watermark.gamma
        )
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    figures = plot_detection(tables["detection"], out_dir)

    sections = [f"# Report: {config.run_name}\n", f"Dataset: `{config.dataset}`\n"]
    titles = {
        "generation": "Generation",
        "pass_at_1": "Functional correctness (pass@1)",
        "outcome_changes": "Share of variants whose test outcome changed after the transform",
        "detection": "Detection",
        "signal": "Where the signal lives (original watermarked samples)",
    }
    for name, table in tables.items():
        sections.append(f"\n## {titles[name]}\n\n{to_markdown(table)}")
    for figure in figures:
        sections.append(f"\n![{figure.stem}]({figure.name})\n")
    environment = read_jsonl(run_dir / "environment.jsonl")
    if environment:
        sections.append(f"\n## Environment\n\n{to_markdown(pd.DataFrame(environment))}")
    report_path = out_dir / "report.md"
    report_path.write_text("".join(sections), encoding="utf-8")
    return report_path
