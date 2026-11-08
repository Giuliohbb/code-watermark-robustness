"""Detection and correctness metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _clean(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Area under the ROC curve of the score, ``nan`` if a class is empty."""
    pos, neg = _clean(positives), _clean(negatives)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, np.concatenate([pos, neg])))


def threshold_at_fpr(negatives: Sequence[float], fpr: float) -> float:
    """Smallest threshold such that at most ``fpr`` of the negatives score above it.

    A sample is flagged when its score is strictly greater than the threshold. The
    threshold is chosen on *untransformed* negatives and then kept fixed, which is
    how a deployed detector would behave.
    """
    neg = np.sort(_clean(negatives))[::-1]
    if len(neg) == 0:
        return float("nan")
    allowed = math.floor(fpr * len(neg))
    if allowed >= len(neg):
        return float("-inf")
    return float(neg[allowed])


def rate_above(values: Sequence[float], threshold: float) -> float:
    array = _clean(values)
    if len(array) == 0 or math.isnan(threshold):
        return float("nan")
    return float(np.mean(array > threshold))


def pass_at_1(passed_by_task: dict[str, list[bool]]) -> float:
    """Unbiased pass@1 with n samples per task: the mean, over tasks, of c / n."""
    if not passed_by_task:
        return float("nan")
    return float(np.mean([np.mean(v) for v in passed_by_task.values() if v]))


def bootstrap_ci(
    groups: dict[str, object],
    statistic: Callable[[list[object]], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval from resampling whole tasks, which keeps samples of a task together."""
    keys = list(groups)
    if not keys or n_resamples <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        value = statistic([groups[keys[i]] for i in chosen])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return (float(low), float(high))
