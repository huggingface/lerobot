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
"""Comparing two success counts: Newcombe interval on the difference, and the regression verdict."""

import math

import pytest

from lerobot.utils.eval_stats import compare_success_counts, newcombe_interval

# Reference values: Newcombe (1998), "Interval estimation for the difference between independent
# proportions: comparison of eleven methods", Table II, method 10 (hybrid Wilson score, no continuity
# correction). Differences are p_a - p_b.
#   (a) 56/70 vs 48/80 -> 0.2000, [0.0524, 0.3339]
#   (b)  9/10 vs  3/10 -> 0.6000, [0.1705, 0.8090]
#   (c)  6/7  vs  2/7  -> 0.5714, [0.0582, 0.8062]
#   (d)  5/56 vs  0/29 -> 0.0893, [-0.0381, 0.1926]


@pytest.mark.parametrize(
    ("k_a", "n_a", "k_b", "n_b", "expected"),
    [
        (56, 70, 48, 80, (0.0524, 0.3339)),
        (9, 10, 3, 10, (0.1705, 0.8090)),
        (6, 7, 2, 7, (0.0582, 0.8062)),
        (5, 56, 0, 29, (-0.0381, 0.1926)),
    ],
)
def test_newcombe_interval_matches_published_examples(k_a, n_a, k_b, n_b, expected):
    low, high = newcombe_interval(k_a, n_a, k_b, n_b)
    assert low == pytest.approx(expected[0], abs=1e-4)
    assert high == pytest.approx(expected[1], abs=1e-4)


def test_newcombe_interval_contains_the_point_difference():
    low, high = newcombe_interval(7, 10, 9, 10)
    assert low < -0.2 < high


def test_newcombe_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        newcombe_interval(11, 10, 5, 10)
    with pytest.raises(ValueError):
        newcombe_interval(5, 10, 5, 0)


def test_compare_flags_a_regression_when_the_whole_interval_clears_the_floor():
    # 95/100 -> 60/100: drop of 35 pp, interval far below -5 pp.
    c = compare_success_counts(95, 100, 60, 100, min_drop_pp=5.0)
    assert c["verdict"] == "REGRESSED"
    assert c["delta_pp"] == pytest.approx(-35.0)
    assert c["ci95_pp"][1] < -5.0
    assert c["p_value"] < 0.001


def test_compare_calls_a_visible_drop_suspect_when_noise_could_explain_it():
    # 92/100 -> 82/100: -10 pp, but the interval reaches above -5 pp.
    c = compare_success_counts(92, 100, 82, 100, min_drop_pp=5.0)
    assert c["verdict"] == "SUSPECT"
    assert c["ci95_pp"][1] > -5.0


def test_compare_holds_small_changes():
    c = compare_success_counts(85, 100, 83, 100, min_drop_pp=5.0)
    assert c["verdict"] == "HELD"


def test_compare_reports_an_improvement_only_when_the_interval_clears_zero():
    assert compare_success_counts(37, 100, 77, 100)["verdict"] == "IMPROVED"
    assert compare_success_counts(70, 100, 74, 100)["verdict"] == "HELD"


def test_compare_refuses_to_judge_below_min_episodes():
    c = compare_success_counts(9, 9, 5, 9, min_episodes=10)
    assert c["verdict"] == "UNDERPOWERED"
    assert math.isnan(c["p_value"]) or c["p_value"] >= 0.0


def test_compare_reports_the_drop_it_could_have_called():
    c = compare_success_counts(90, 100, 90, 100, min_drop_pp=5.0)
    # With no change, the smallest callable drop is the floor plus the interval half-width.
    assert c["resolvable_drop_pp"] > 5.0
    assert c["resolvable_drop_pp"] == pytest.approx(5.0 + (c["ci95_pp"][1] - c["delta_pp"]), abs=1e-6)
