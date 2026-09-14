#!/usr/bin/env python

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

import numpy as np
import pytest

# lerobot.eval.multi_seed imports eval_policy from lerobot.scripts.lerobot_eval, which needs the dataset
# extra at import time.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.eval.multi_seed as multi_seed  # noqa: E402
from lerobot.eval import MultiSeedResult, run_multi_seed_eval, wilson_ci  # noqa: E402


def _fake_eval_policy_factory(success_by_seed):
    """Return a stub for `eval_policy` that emits a fixed pattern of successes per seed.

    `success_by_seed` maps a `start_seed` to the list of per-episode booleans to report. This lets the
    orchestration be tested without a simulator, GPU, or a real policy.
    """

    def _fake_eval_policy(env, policy, *, n_episodes, start_seed, **kwargs):
        successes = success_by_seed[start_seed]
        assert len(successes) == n_episodes
        return {
            "per_episode": [{"success": s, "seed": start_seed + i} for i, s in enumerate(successes)],
            "aggregated": {
                "avg_sum_reward": float(np.mean(successes)),
                "avg_max_reward": float(np.max(successes)),
                "pc_success": float(np.mean(successes) * 100),
            },
        }

    return _fake_eval_policy


@pytest.fixture
def patched_run(monkeypatch):
    """`run_multi_seed_eval` with `eval_policy` stubbed and the processor args set to None."""

    def _run(success_by_seed, **kwargs):
        monkeypatch.setattr(multi_seed, "eval_policy", _fake_eval_policy_factory(success_by_seed))
        return run_multi_seed_eval(
            env=None,
            policy=None,
            env_preprocessor=None,
            env_postprocessor=None,
            preprocessor=None,
            postprocessor=None,
            seeds=list(success_by_seed),
            episodes_per_seed=len(next(iter(success_by_seed.values()))),
            **kwargs,
        )

    return _run


# ---------------------------------------------------------------------- #
# wilson_ci                                                              #
# ---------------------------------------------------------------------- #


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(7, 10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_wilson_ci_stays_in_unit_interval_at_extremes():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_ci(20, 20)
    assert 0.0 < lo < 1.0 and hi == 1.0


def test_wilson_ci_narrows_with_more_trials():
    def width(n):
        lo, hi = wilson_ci(n // 2, n)
        return hi - lo

    assert width(20) > width(200)


def test_wilson_ci_matches_known_reference():
    # 50 of 100 -> the 0.95 Wilson interval is approximately (0.404, 0.596).
    lo, hi = wilson_ci(50, 100, ci_level=0.95)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_ci_level_monotonicity():
    # A higher confidence level must give a wider interval.
    lo90, hi90 = wilson_ci(30, 50, ci_level=0.90)
    lo99, hi99 = wilson_ci(30, 50, ci_level=0.99)
    assert lo99 < lo90 and hi90 < hi99


@pytest.mark.parametrize(
    ("n_successes", "n_trials"),
    [(-1, 10), (11, 10), (5, 0)],
)
def test_wilson_ci_rejects_invalid_counts(n_successes, n_trials):
    with pytest.raises(ValueError):
        wilson_ci(n_successes, n_trials)


def test_wilson_ci_rejects_invalid_ci_level():
    with pytest.raises(ValueError):
        wilson_ci(5, 10, ci_level=1.0)


# ---------------------------------------------------------------------- #
# run_multi_seed_eval                                                    #
# ---------------------------------------------------------------------- #


def test_pools_outcomes_across_seeds(patched_run):
    result = patched_run({0: [True, False], 1: [True, True]})
    assert isinstance(result, MultiSeedResult)
    assert result.seeds == (0, 1)
    assert result.episodes_per_seed == 2
    assert result.n_episodes == 4
    assert result.n_successes == 3
    assert result.success_rate == pytest.approx(0.75)


def test_per_seed_breakdown_is_preserved(patched_run):
    result = patched_run({0: [True, False, False], 7: [True, True, False]})
    assert tuple(s.seed for s in result.per_seed) == (0, 7)
    assert result.per_seed[0].n_successes == 1
    assert result.per_seed[1].n_successes == 2
    assert result.per_seed[0].successes == (True, False, False)


def test_all_successes_is_flat_and_ordered(patched_run):
    result = patched_run({0: [True, False], 1: [False, True]})
    np.testing.assert_array_equal(result.all_successes, np.array([True, False, False, True]))


def test_ci_brackets_pooled_rate(patched_run):
    result = patched_run({0: [True] * 5, 1: [False] * 5})
    assert result.success_rate == pytest.approx(0.5)
    assert result.ci_low < 0.5 < result.ci_high


def test_pooled_ci_matches_wilson_over_flat_episodes(patched_run):
    result = patched_run({0: [True] * 4 + [False], 1: [True] * 3 + [False] * 2})
    lo, hi = wilson_ci(result.n_successes, result.n_episodes, ci_level=result.ci_level)
    assert result.ci_low == pytest.approx(lo)
    assert result.ci_high == pytest.approx(hi)


def test_seeding_contract_applies_set_seed_per_seed(patched_run, monkeypatch):
    seen = []
    monkeypatch.setattr(multi_seed, "set_seed", lambda s: seen.append(s))
    patched_run({3: [True], 9: [False], 1: [True]})
    assert seen == [3, 9, 1]


def test_rejects_empty_seeds(monkeypatch):
    monkeypatch.setattr(multi_seed, "eval_policy", _fake_eval_policy_factory({}))
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        run_multi_seed_eval(
            env=None,
            policy=None,
            env_preprocessor=None,
            env_postprocessor=None,
            preprocessor=None,
            postprocessor=None,
            seeds=[],
            episodes_per_seed=5,
        )


def test_rejects_nonpositive_episodes(monkeypatch):
    monkeypatch.setattr(multi_seed, "eval_policy", _fake_eval_policy_factory({0: [True]}))
    with pytest.raises(ValueError, match="episodes_per_seed must be positive"):
        run_multi_seed_eval(
            env=None,
            policy=None,
            env_preprocessor=None,
            env_postprocessor=None,
            preprocessor=None,
            postprocessor=None,
            seeds=[0],
            episodes_per_seed=0,
        )
