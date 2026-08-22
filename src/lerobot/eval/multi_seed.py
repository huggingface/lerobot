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

"""Run a policy across several seeds and report a success rate with a confidence interval.

`eval_policy` (in `lerobot.scripts.lerobot_eval`) evaluates a policy over a batch of episodes seeded from a
single `start_seed`. A single-seed point estimate is fragile: simulator initial conditions, GPU
non-determinism, and the policy's own stochasticity all move the number, and a reported `pc_success`
without an interval can hide a 10+ point swing between seeds.

This module adds the missing orchestration layer: evaluate over a list of seeds, collect the per-episode
success outcomes, and aggregate them into a single success rate with a Wilson score confidence interval. It
is a thin loop over the existing `eval_policy` — it does not reimplement the rollout.

Seeding contract: for each seed `s` in `seeds`, `set_seed(s)` is called once before that seed's episodes
run (fixing the Python / NumPy / Torch global RNGs, which is what makes the policy's own stochasticity
reproducible), and the environments are reset from `start_seed=s` so episode `e` of that seed uses
environment seed `s + e`. The torch generator is *not* re-seeded between episodes within a seed, so only
the `(policy, env, seed)` cell as a whole is reproducible, not individual episodes within it.

The confidence interval is the textbook Wilson score interval over the flat list of per-episode binary
outcomes — the unit of resampling is the *episode*, not the seed. Resampling over the handful of seeds
instead would inflate the interval and is the common mistake this helper is meant to avoid. Richer
machinery (bootstrap and paired-bootstrap intervals, paired policy-vs-policy ranking) lives in the
Embodimetry benchmark (https://github.com/thrmnn/embodimetry) where this module originated, and is out of
scope here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.policies import PreTrainedPolicy
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

if TYPE_CHECKING:
    import gymnasium as gym

    from lerobot.lerobot_types import PolicyAction
    from lerobot.processor import PolicyProcessorPipeline


@dataclass(frozen=True)
class SeedResult:
    """Outcome of evaluating one seed.

    Args:
        seed (`int`):
            The seed this cell was evaluated under.
        n_episodes (`int`):
            Number of episodes run for this seed.
        n_successes (`int`):
            Number of successful episodes.
        success_rate (`float`):
            `n_successes / n_episodes`.
        avg_sum_reward (`float`):
            Mean per-episode summed reward, as reported by `eval_policy`.
        avg_max_reward (`float`):
            Mean per-episode maximum reward, as reported by `eval_policy`.
        successes (`tuple[bool, ...]`):
            Per-episode success outcomes, in episode order.
    """

    seed: int
    n_episodes: int
    n_successes: int
    success_rate: float
    avg_sum_reward: float
    avg_max_reward: float
    successes: tuple[bool, ...]


@dataclass(frozen=True)
class MultiSeedResult:
    """Aggregated outcome across all seeds.

    Args:
        seeds (`tuple[int, ...]`):
            The seeds that were evaluated, in order.
        episodes_per_seed (`int`):
            Number of episodes run for each seed.
        n_episodes (`int`):
            Total number of episodes, i.e. `len(seeds) * episodes_per_seed`.
        n_successes (`int`):
            Total number of successful episodes across all seeds.
        success_rate (`float`):
            Pooled success rate over all episodes.
        ci_low (`float`):
            Lower bound of the Wilson confidence interval.
        ci_high (`float`):
            Upper bound of the Wilson confidence interval.
        ci_level (`float`):
            Confidence level of the interval, e.g. `0.95`.
        per_seed (`tuple[SeedResult, ...]`):
            The individual [`SeedResult`] for each seed, in order.
    """

    seeds: tuple[int, ...]
    episodes_per_seed: int
    n_episodes: int
    n_successes: int
    success_rate: float
    ci_low: float
    ci_high: float
    ci_level: float
    per_seed: tuple[SeedResult, ...]

    @property
    def all_successes(self) -> np.ndarray:
        """The flat `(n_episodes,)` boolean array of per-episode outcomes, in seed then episode order."""
        return np.concatenate([np.asarray(s.successes, dtype=bool) for s in self.per_seed])


def wilson_ci(n_successes: int, n_trials: int, *, ci_level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli success proportion.

    The Wilson interval (Wilson, 1927) is a closed-form binomial confidence interval that, unlike the
    normal (Wald) approximation, stays inside `[0, 1]` and behaves well for small `n` and for proportions
    near 0 or 1 — exactly the regime success-rate evaluation lives in.

    Args:
        n_successes (`int`):
            Number of successes; must satisfy `0 <= n_successes <= n_trials`.
        n_trials (`int`):
            Number of trials; must be positive.
        ci_level (`float`, *optional*, defaults to 0.95):
            Confidence level, in `(0, 1)`.

    Returns:
        `tuple[float, float]`: The `(low, high)` bounds of the interval, both clamped to `[0, 1]`.

    Raises:
        ValueError: If `n_trials` is not positive, `n_successes` is out of range, or `ci_level` is not in
            `(0, 1)`.

    Example:
        ```python
        >>> from lerobot.eval import wilson_ci
        >>> low, high = wilson_ci(50, 100)
        >>> (round(low, 3), round(high, 3))
        (0.404, 0.596)
        ```
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if not 0 <= n_successes <= n_trials:
        raise ValueError(f"n_successes={n_successes} not in [0, {n_trials}]")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    z = NormalDist().inv_cdf((1.0 + ci_level) / 2.0)
    p_hat = n_successes / n_trials
    z2_n = z * z / n_trials
    center = (p_hat + z2_n / 2.0) / (1.0 + z2_n)
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) + z2_n / 4.0) / n_trials)) / (1.0 + z2_n)
    # At the boundaries the Wilson bounds are exactly 0 and 1; pin them so floating-point residue from
    # `center - half` cannot leak a bound like 1e-17.
    low = 0.0 if n_successes == 0 else max(0.0, center - half)
    high = 1.0 if n_successes == n_trials else min(1.0, center + half)
    return low, high


def run_multi_seed_eval(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    *,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: Sequence[int],
    episodes_per_seed: int,
    ci_level: float = 0.95,
    videos_dir: Path | None = None,
    max_episodes_rendered: int = 0,
) -> MultiSeedResult:
    """Evaluate `policy` on `env` across `seeds` and aggregate with a confidence interval.

    For each seed, `set_seed(seed)` is applied and then `eval_policy` runs `episodes_per_seed` episodes
    with `start_seed=seed`, so episode `e` of that seed uses environment seed `seed + e`. The per-episode
    success outcomes from every seed are pooled, and a Wilson score interval is computed over the pooled
    outcomes. See the module docstring for the full seeding contract and for what statistics intentionally
    live elsewhere.

    Args:
        env (`gym.vector.VectorEnv`):
            The batch of environments, as built by `lerobot.envs.make_env`.
        policy (`PreTrainedPolicy`):
            The policy to evaluate.
        env_preprocessor (`PolicyProcessorPipeline`):
            Environment-side preprocessor pipeline, passed through to `eval_policy`.
        env_postprocessor (`PolicyProcessorPipeline`):
            Environment-side postprocessor pipeline, passed through to `eval_policy`.
        preprocessor (`PolicyProcessorPipeline`):
            Policy input preprocessor pipeline, passed through to `eval_policy`.
        postprocessor (`PolicyProcessorPipeline`):
            Policy action postprocessor pipeline, passed through to `eval_policy`.
        seeds (`Sequence[int]`):
            The seeds to evaluate. Must be non-empty.
        episodes_per_seed (`int`):
            Episodes to run for each seed. Must be positive.
        ci_level (`float`, *optional*, defaults to 0.95):
            Confidence level for the Wilson interval.
        videos_dir (`Path`, *optional*):
            Where to save rendered videos, if any.
        max_episodes_rendered (`int`, *optional*, defaults to 0):
            Maximum number of episodes to render into videos, per seed.

    Returns:
        [`MultiSeedResult`]: The pooled success rate, its Wilson interval, and the per-seed breakdown.

    Raises:
        ValueError: If `seeds` is empty, `episodes_per_seed` is not positive, or `ci_level` is not in
            `(0, 1)`.

    Example:
        ```python
        >>> from lerobot.eval import run_multi_seed_eval
        >>> result = run_multi_seed_eval(  # doctest: +SKIP
        ...     env,
        ...     policy,
        ...     env_preprocessor=env_preprocessor,
        ...     env_postprocessor=env_postprocessor,
        ...     preprocessor=preprocessor,
        ...     postprocessor=postprocessor,
        ...     seeds=[1000, 2000, 3000],
        ...     episodes_per_seed=50,
        ... )
        >>> print(f"{result.success_rate:.2f} [{result.ci_low:.2f}, {result.ci_high:.2f}]")  # doctest: +SKIP
        0.62 [0.54, 0.69]
        ```
    """
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")
    if episodes_per_seed <= 0:
        raise ValueError(f"episodes_per_seed must be positive, got {episodes_per_seed}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    per_seed: list[SeedResult] = []
    for seed in seeds:
        set_seed(seed)
        info = eval_policy(
            env,
            policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=episodes_per_seed,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=seed,
        )
        successes = tuple(bool(ep["success"]) for ep in info["per_episode"])
        per_seed.append(
            SeedResult(
                seed=seed,
                n_episodes=len(successes),
                n_successes=sum(successes),
                success_rate=sum(successes) / len(successes),
                avg_sum_reward=float(info["aggregated"]["avg_sum_reward"]),
                avg_max_reward=float(info["aggregated"]["avg_max_reward"]),
                successes=successes,
            )
        )

    n_episodes = sum(s.n_episodes for s in per_seed)
    n_successes = sum(s.n_successes for s in per_seed)
    ci_low, ci_high = wilson_ci(n_successes, n_episodes, ci_level=ci_level)
    return MultiSeedResult(
        seeds=tuple(seeds),
        episodes_per_seed=episodes_per_seed,
        n_episodes=n_episodes,
        n_successes=n_successes,
        success_rate=n_successes / n_episodes,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        per_seed=tuple(per_seed),
    )
