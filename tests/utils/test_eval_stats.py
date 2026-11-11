# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Statistics attached to evaluation success rates: Wilson intervals, Fisher's exact test."""

import math

import pytest

from lerobot.utils.eval_stats import fisher_exact, success_summary, wilson_interval

# Reference values: Wilson score interval, z = 1.959964 (95%), no continuity correction.
# 5/10 -> [0.2366, 0.7634]; 9/10 -> [0.5958, 0.9821]; 0/10 -> [0.0, 0.2775]; 10/10 -> [0.7225, 1.0].
# 70/100 -> [0.6042, 0.7811]. Same figures as statsmodels' proportion_confint(method="wilson").


@pytest.mark.parametrize(
    ("n_success", "n_trials", "expected"),
    [
        (5, 10, (0.2366, 0.7634)),
        (9, 10, (0.5958, 0.9821)),
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (70, 100, (0.6042, 0.7811)),
    ],
)
def test_wilson_interval_matches_reference_values(n_success, n_trials, expected):
    low, high = wilson_interval(n_success, n_trials)
    assert low == pytest.approx(expected[0], abs=1e-4)
    assert high == pytest.approx(expected[1], abs=1e-4)


def test_wilson_interval_is_nan_for_zero_trials():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_wilson_interval_narrows_with_more_trials():
    low_10, high_10 = wilson_interval(5, 10)
    low_100, high_100 = wilson_interval(50, 100)
    assert high_100 - low_100 < high_10 - low_10


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def test_wilson_interval_confidence_level_changes_width():
    low_95, high_95 = wilson_interval(5, 10, confidence=0.95)
    low_90, high_90 = wilson_interval(5, 10, confidence=0.90)
    assert high_90 - low_90 < high_95 - low_95


def test_success_summary_reports_counts_rate_and_interval_in_percent():
    summary = success_summary([True, False, True, True, False, True, True, True, True, False])
    assert summary["n_episodes"] == 10
    assert summary["n_success"] == 7
    assert summary["pc_success"] == pytest.approx(70.0)
    low, high = summary["pc_success_ci95"]
    assert low == pytest.approx(39.68, abs=0.05)
    assert high == pytest.approx(89.22, abs=0.05)


def test_success_summary_of_nothing_is_nan_not_an_error():
    summary = success_summary([])
    assert summary["n_episodes"] == 0
    assert summary["n_success"] == 0
    assert math.isnan(summary["pc_success"])
    assert all(math.isnan(x) for x in summary["pc_success_ci95"])


def test_success_summary_accepts_numeric_flags():
    # eval_policy yields Python bools today; accept 0/1 ints too so a future change to the
    # reduction cannot silently change what gets counted.
    summary = success_summary([1, 0, 1, 1])
    assert summary["n_success"] == 3
    assert summary["pc_success"] == pytest.approx(75.0)


# Reference values: two-sided Fisher exact p-values (scipy.stats.fisher_exact on [[k_a, n_a-k_a], [k_b, n_b-k_b]]).
# 9/10 vs 5/10 -> 0.1409; 7/10 vs 7/10 -> 1.0; 10/10 vs 0/10 -> 1.083e-05; 80/100 vs 60/100 -> 0.003192.


@pytest.mark.parametrize(
    ("k_a", "n_a", "k_b", "n_b", "expected"),
    [
        (9, 10, 5, 10, 0.1409),
        (7, 10, 7, 10, 1.0),
        (10, 10, 0, 10, 1.083e-05),
        (80, 100, 60, 100, 0.003192),
    ],
)
def test_fisher_exact_matches_reference_values(k_a, n_a, k_b, n_b, expected):
    assert fisher_exact(k_a, n_a, k_b, n_b) == pytest.approx(expected, rel=1e-3)


def test_fisher_exact_is_symmetric():
    assert fisher_exact(9, 10, 5, 10) == pytest.approx(fisher_exact(5, 10, 9, 10))


def test_fisher_exact_rejects_impossible_counts():
    with pytest.raises(ValueError):
        fisher_exact(11, 10, 5, 10)
    with pytest.raises(ValueError):
        fisher_exact(5, 10, 5, 0)
