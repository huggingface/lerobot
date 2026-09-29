# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Numerical coverage for the raw-space normalization feeding the canonical layout.

Normalization now runs through the standard ``NormalizerProcessorStep`` in raw
feature space (before the slot-mapping step). Since the stats are per-dim
independent, slicing the raw stats with the slot offsets reproduces the released
checkpoints' per-slot statistics exactly; these tests pin that equivalence and
the upstream→standard norm-mode mapping (``meanstd`` → MEAN_STD,
``bounds_99_woclip`` → QUANTILES, i.e. q01/q99 without clipping).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (  # noqa: E402
    _raw_stats_from_slot_stats,
    _resolve_norm_map,
)
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

ARM = f"{OBS_STATE}.arm.position"
ACT_ARM = f"{ACTION}.arm.position"


def _normalizer(stats, mode=NormalizationMode.MEAN_STD):
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return NormalizerProcessorStep(
        features=features,
        norm_map={"VISUAL": NormalizationMode.IDENTITY, "STATE": mode, "ACTION": mode},
        stats={OBS_STATE: stats, ACTION: stats},
    )


def _transition(state=None, action=None):
    transition = {}
    if state is not None:
        transition[TransitionKey.OBSERVATION] = {OBS_STATE: state}
    if action is not None:
        transition[TransitionKey.ACTION] = action
    return transition


def test_meanstd_normalization_is_exact():
    stats = {
        "mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "std": np.full(6, 0.5),
    }
    step = _normalizer(stats)
    x = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]])
    out = step(_transition(state=x))[TransitionKey.OBSERVATION][OBS_STATE]
    expected = (x - torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])) / (0.5 + 1e-8)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_quantiles_maps_bounds_99_woclip_without_clipping():
    """QUANTILES is the standard equivalent of upstream bounds_99_woclip: q01/q99,
    mapped to [-1, 1], with no clipping of out-of-range values."""
    stats = {
        "q01": np.full(6, -1.0),
        "q99": np.full(6, 1.0),
        "min": np.full(6, -100.0),  # must be ignored by the quantile mode
        "max": np.full(6, 100.0),
    }
    step = _normalizer(stats, mode=NormalizationMode.QUANTILES)
    x = torch.tensor([[0.0, 1.0, -1.0, 0.5, -0.5, 2.0]])
    out = step(_transition(action=x))[TransitionKey.ACTION]
    expected = 2.0 * (x - (-1.0)) / (1.0 - (-1.0)) - 1.0
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    # 2.0 stays outside [-1, 1] (no clip) — a bounds-with-clip mode would clamp it.
    assert out[0, -1].item() == pytest.approx(2.0, abs=1e-5)


def test_normalize_unnormalize_roundtrip_recovers_input():
    stats = {"mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "std": np.full(6, 0.5)}
    normalizer = _normalizer(stats)
    unnormalizer = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        norm_map={"ACTION": NormalizationMode.MEAN_STD},
        stats={ACTION: stats},
    )
    x = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    normalized = normalizer(_transition(action=x))[TransitionKey.ACTION]
    back = unnormalizer(_transition(action=normalized))[TransitionKey.ACTION]
    torch.testing.assert_close(back, x, atol=1e-4, rtol=1e-4)


def test_raw_space_normalization_matches_per_slot_slicing():
    """Normalizing the raw vector then slicing equals slicing the stats per slot —
    the equivalence that lets the standard step replace the per-slot normalizer."""
    raw_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    raw_std = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    # Slot layout: arm = raw[0:6], effector = raw[6:7].
    robot_config = {
        "states": [
            {ARM: {"origin_keys": [{OBS_STATE: {"start": 0, "end": 6}}]}},
            {f"{OBS_STATE}.effector.position": {"origin_keys": [{OBS_STATE: {"start": 6, "end": 7}}]}},
        ],
        "actions": [
            {ACT_ARM: {"origin_keys": [{ACTION: {"start": 0, "end": 6}}]}},
            {f"{ACTION}.effector.position": {"origin_keys": [{ACTION: {"start": 6, "end": 7}}]}},
        ],
    }
    slot_stats = {
        "norm_stats": {
            ARM: {"mean": raw_mean[0:6].tolist(), "std": raw_std[0:6].tolist()},
            f"{OBS_STATE}.effector.position": {"mean": raw_mean[6:7].tolist(), "std": raw_std[6:7].tolist()},
            ACT_ARM: {"mean": raw_mean[0:6].tolist(), "std": raw_std[0:6].tolist()},
            f"{ACTION}.effector.position": {"mean": raw_mean[6:7].tolist(), "std": raw_std[6:7].tolist()},
        }
    }
    raw_stats = _raw_stats_from_slot_stats(robot_config, slot_stats)
    step = _normalizer(raw_stats[OBS_STATE])
    x = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]])
    out = step(_transition(state=x))[TransitionKey.OBSERVATION][OBS_STATE]
    # Per-slot reference: each slot normalized with its own sliced stats.
    arm = (x[:, 0:6] - torch.tensor(raw_mean[0:6])).float() / (torch.tensor(raw_std[0:6]) + 1e-8).float()
    effector = (x[:, 6:7] - torch.tensor(raw_mean[6:7])).float() / (torch.tensor(raw_std[6:7]) + 1e-8).float()
    torch.testing.assert_close(out[:, 0:6], arm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, 6:7], effector, atol=1e-5, rtol=1e-5)


def test_horizon_indexed_action_stats_broadcast_per_timestep():
    """Action stats are [chunk_size, dim]: each denoising step has its own mean/std;
    the standard step broadcasts them against (B, chunk, dim) chunks."""
    chunk, dim = 4, 3
    mean = np.arange(chunk * dim, dtype=np.float64).reshape(chunk, dim)
    std = np.ones((chunk, dim))
    step = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(dim,))},
        norm_map={"ACTION": NormalizationMode.MEAN_STD},
        stats={ACTION: {"mean": mean, "std": std}},
    )
    x = torch.zeros(2, chunk, dim)
    out = step(_transition(action=x))[TransitionKey.ACTION]
    # out[t] = (0 - mean[t]) / 1 == -mean[t]; per-timestep means preserved.
    torch.testing.assert_close(out[0], -torch.from_numpy(mean).float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1], out[0], atol=1e-6, rtol=1e-6)


def test_norm_mode_resolution_maps_upstream_modes():
    uniform_meanstd = _resolve_norm_map({"arm.position": "meanstd", "effector.position": "meanstd"})
    assert uniform_meanstd["STATE"] is NormalizationMode.MEAN_STD
    assert uniform_meanstd["ACTION"] is NormalizationMode.MEAN_STD

    quantiles = _resolve_norm_map({"arm.position": "bounds_99_woclip"})
    assert quantiles["STATE"] is NormalizationMode.QUANTILES
    assert quantiles["ACTION"] is NormalizationMode.QUANTILES

    # Mixed per-slot modes collapse to the first joint's mode (standard step is
    # per-type); the resolution must at least stay deterministic and standard.
    mixed = _resolve_norm_map({"arm.position": "bounds_99_woclip", "hand.position": "meanstd"})
    assert mixed["ACTION"] is NormalizationMode.QUANTILES

    identity = _resolve_norm_map({"reserved.slots": "identity"})
    assert identity["STATE"] is NormalizationMode.IDENTITY
