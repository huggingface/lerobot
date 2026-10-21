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
"""Statistics for evaluation success rates.

A success rate measured over `n` episodes is a binomial estimate, and at the episode counts robot
evaluations typically use (10 to 50 per task) its sampling uncertainty is large: 9 successes out of 10
episodes is compatible with a true rate anywhere between roughly 60% and 98%. The helpers here attach that
uncertainty to the numbers `lerobot-eval` reports, so that two success rates can be compared honestly.

Everything is implemented with the standard library only: the functions must stay importable in the
minimal install and in CI without adding a dependency.

The intervals describe *evaluation sampling* noise only, i.e. which episodes happened to be drawn. They do
not cover variation between training runs (seeds) of the same recipe, which has to be measured by training
more than once.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist


def _z_from_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def _check_counts(n_success: int, n_trials: int) -> None:
    if n_trials < 0 or n_success < 0 or n_success > n_trials:
        raise ValueError(f"impossible counts: {n_success} successes out of {n_trials} trials")


def wilson_interval(n_success: int, n_trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal-approximation ("Wald") interval because it stays inside [0, 1] and keeps
    its nominal coverage at small `n_trials` and at rates near 0% or 100%, which is where robot
    evaluations live. No continuity correction is applied.

    Args:
        n_success (`int`):
            Number of successful episodes.
        n_trials (`int`):
            Number of evaluated episodes.
        confidence (`float`, *optional*, defaults to `0.95`):
            Coverage of the interval, strictly between 0 and 1.

    Returns:
        `tuple[float, float]`: Lower and upper bound of the interval as proportions in [0, 1]. Both are
        `nan` when `n_trials` is 0.

    Raises:
        ValueError: If the counts are impossible or `confidence` is outside (0, 1).

    Example:
        ```python
        >>> from lerobot.utils.eval_stats import wilson_interval
        >>> low, high = wilson_interval(9, 10)
        >>> round(low, 3), round(high, 3)
        (0.596, 0.982)
        ```
    """
    _check_counts(n_success, n_trials)
    z = _z_from_confidence(confidence)
    if n_trials == 0:
        return (math.nan, math.nan)
    p = n_success / n_trials
    z2 = z * z
    denominator = 1.0 + z2 / n_trials
    centre = (p + z2 / (2.0 * n_trials)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / n_trials + z2 / (4.0 * n_trials * n_trials)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def success_summary(successes: Sequence[bool | int | float]) -> dict:
    """Summarise a list of per-episode success flags.

    Args:
        successes (`Sequence[bool | int | float]`):
            One truthy/falsy flag per evaluated episode.

    Returns:
        `dict`: `n_episodes`, `n_success`, `pc_success` (percent) and `pc_success_ci95` (a two-element
        list with the 95% Wilson interval bounds in percent). The coverage is fixed at 95% so the
        key name always matches the number; `wilson_interval` takes a `confidence` argument for
        callers that want a different one. When there are no episodes the rate and the bounds are
        `nan`, matching how the other aggregated metrics behave.

    Example:
        ```python
        >>> from lerobot.utils.eval_stats import success_summary
        >>> summary = success_summary([True, True, False, True])
        >>> summary["n_success"], summary["pc_success"]
        (3, 75.0)
        ```
    """
    n_episodes = len(successes)
    n_success = int(sum(bool(s) for s in successes))
    if n_episodes == 0:
        pc_success = math.nan
        low, high = math.nan, math.nan
    else:
        pc_success = 100.0 * n_success / n_episodes
        low, high = wilson_interval(n_success, n_episodes)
        low, high = 100.0 * low, 100.0 * high
    return {
        "n_episodes": n_episodes,
        "n_success": n_success,
        "pc_success": pc_success,
        "pc_success_ci95": [low, high],
    }


def _log_hypergeom_pmf(k: int, n_a: int, n_b: int, k_total: int) -> float:
    return (
        math.lgamma(n_a + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n_a - k + 1)
        + math.lgamma(n_b + 1)
        - math.lgamma(k_total - k + 1)
        - math.lgamma(n_b - k_total + k + 1)
        - math.lgamma(n_a + n_b + 1)
        + math.lgamma(k_total + 1)
        + math.lgamma(n_a + n_b - k_total + 1)
    )


def fisher_exact(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """Two-sided Fisher's exact test on two success counts.

    Answers "could these two success rates come from the same underlying rate?" without any normal
    approximation, which matters at 10 to 50 episodes per task. The two-sided p-value sums every table
    at least as unlikely as the observed one, the same convention as `scipy.stats.fisher_exact`.

    Args:
        k_a (`int`):
            Successes of policy A.
        n_a (`int`):
            Episodes evaluated for policy A.
        k_b (`int`):
            Successes of policy B.
        n_b (`int`):
            Episodes evaluated for policy B.

    Returns:
        `float`: The two-sided p-value in [0, 1].

    Raises:
        ValueError: If either pair of counts is impossible or either `n` is 0.

    Example:
        ```python
        >>> from lerobot.utils.eval_stats import fisher_exact
        >>> round(fisher_exact(9, 10, 5, 10), 3)
        0.141
        ```
    """
    _check_counts(k_a, n_a)
    _check_counts(k_b, n_b)
    if n_a == 0 or n_b == 0:
        raise ValueError("both policies need at least one evaluated episode")
    k_total = k_a + k_b
    k_min = max(0, k_total - n_b)
    k_max = min(k_total, n_a)
    observed = _log_hypergeom_pmf(k_a, n_a, n_b, k_total)
    # Relative tolerance so tables with (floating-point-)equal probability count as "as extreme".
    threshold = observed + 1e-7
    p_value = 0.0
    for k in range(k_min, k_max + 1):
        log_p = _log_hypergeom_pmf(k, n_a, n_b, k_total)
        if log_p <= threshold:
            p_value += math.exp(log_p)
    return min(1.0, p_value)
