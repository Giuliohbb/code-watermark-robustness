import math

import pytest

from cwm.evaluation.metrics import auroc, bootstrap_ci, pass_at_1, rate_above, threshold_at_fpr


def test_auroc_perfect_and_chance():
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert math.isnan(auroc([], [1.0]))


def test_auroc_ignores_nan():
    assert auroc([3, float("nan")], [0]) == 1.0


def test_threshold_controls_false_positive_rate():
    negatives = list(range(100))  # 0..99
    threshold = threshold_at_fpr(negatives, 0.05)
    assert rate_above(negatives, threshold) <= 0.05
    assert threshold == 94
    assert threshold_at_fpr(negatives, 1.0) == float("-inf")


def test_threshold_with_few_negatives_flags_nothing_above_max():
    threshold = threshold_at_fpr([0.1, 0.5, 2.0], 0.01)
    assert threshold == 2.0
    assert rate_above([2.0, 2.1], threshold) == 0.5


def test_pass_at_1_is_mean_of_per_task_rates():
    assert pass_at_1({"a": [True, False], "b": [True, True]}) == pytest.approx(0.75)
    assert math.isnan(pass_at_1({}))


def test_bootstrap_interval_contains_point_estimate():
    groups = {str(i): [i % 2 == 0] for i in range(40)}
    low, high = bootstrap_ci(groups, lambda gs: sum(g[0] for g in gs) / len(gs), n_resamples=200)
    assert low <= 0.5 <= high
