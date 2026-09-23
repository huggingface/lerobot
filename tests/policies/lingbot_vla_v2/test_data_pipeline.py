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

"""Numerical coverage for the canonical-space normalization and slot mapping.

These tests pin the exact transforms the released checkpoints were trained with —
the per-slot norm modes, the horizon-indexed action stats, and the raw->canonical
slot slicing. They are the executable companion to the design discussion on why
``dataset.meta.stats`` (flat per-dim stats, no q02/q98, raw key space) cannot drive
this path: every assertion here would change value if the standard
``NormalizerProcessorStep`` were substituted for the canonical one.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lerobot.policies.lingbot_vla_v2.preprocessing.data_transform import Normalizer  # noqa: E402

ARM = "observation.state.arm.position"
ACT_ARM = "action.arm.position"


def _meanstd_stats():
    return {
        ARM: {"mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "std": np.full(6, 0.5)},
        ACT_ARM: {"mean": np.zeros(6), "std": np.ones(6)},
    }


def test_meanstd_normalization_is_exact():
    n = Normalizer(norm_stats=_meanstd_stats(), norm_type={ARM: "meanstd"})
    x = {ARM: torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])}
    out = n.normalize(x)
    # (x - mean) / (std + 1e-6) with mean=[1..6], std=0.5 -> [2,4,6,8,10,12]
    expected = (x[ARM] - torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])) / 0.5
    torch.testing.assert_close(out[ARM], expected, atol=1e-4, rtol=1e-4)


def test_normalize_unnormalize_roundtrip_recovers_input():
    n = Normalizer(norm_stats=_meanstd_stats(), norm_type={ARM: "meanstd", ACT_ARM: "meanstd"})
    x = {
        ARM: torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        ACT_ARM: torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
    }
    back = n.unnormalize(n.normalize(x))
    torch.testing.assert_close(back[ARM], x[ARM], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(back[ACT_ARM], x[ACT_ARM], atol=1e-4, rtol=1e-4)


def test_bounds_99_woclip_uses_quantile_bounds_not_minmax():
    """bounds_99_woclip reads q01/q99 (no clipping) — the robotwin profile's mode.
    With q01/q99 set far inside min/max, the output range differs from a min/max
    mapping, proving the quantile keys (not min/max) drive it."""
    stats = {
        ACT_ARM: {
            "q01": np.full(6, -1.0),
            "q99": np.full(6, 1.0),
            "min": np.full(6, -100.0),  # must be ignored by bounds_99
            "max": np.full(6, 100.0),
        }
    }
    n = Normalizer(norm_stats=stats, norm_type={ACT_ARM: "bounds_99_woclip"})
    x = {ACT_ARM: torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 2.0])}
    out = n.normalize(x)[ACT_ARM]
    # (x - q01) / (q99 - q01) * 2 - 1  == x here since q01=-1, q99=1; and NO clamp.
    expected = (x[ACT_ARM] - (-1.0)) / (1.0 - (-1.0)) * 2.0 - 1.0
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    # 2.0 stays 2.0 (no clip) — a bounds-with-clip mode would have clamped it.
    assert out[-1].item() == pytest.approx(2.0, abs=1e-4)


def test_bounds_98_requires_q02_q98_keys():
    """bounds_98* modes read q02/q98, which are absent from LeRobot's default
    quantile set — the structural reason dataset.meta.stats cannot feed this mode."""
    stats = {ACT_ARM: {"q02": np.full(6, -2.0), "q98": np.full(6, 2.0)}}
    n = Normalizer(norm_stats=stats, norm_type={ACT_ARM: "bounds_98_woclip"})
    x = {ACT_ARM: torch.zeros(6)}
    out = n.normalize(x)[ACT_ARM]
    # (0 - (-2)) / (2 - (-2)) * 2 - 1 = 0
    torch.testing.assert_close(out, torch.zeros(6), atol=1e-4, rtol=1e-4)


def test_horizon_indexed_action_stats_are_per_timestep():
    """Action stats are [chunk_size, dim]: each denoising step has its own mean/std.
    This is the horizon dimension that flat per-dim dataset stats cannot express."""
    chunk, dim = 4, 3
    # Distinct mean per timestep — a flat stat would collapse these to one row.
    mean = np.arange(chunk * dim, dtype=np.float64).reshape(chunk, dim)
    std = np.ones((chunk, dim))
    stats = {ACT_ARM: {"mean": mean, "std": std}}
    n = Normalizer(norm_stats=stats, norm_type={ACT_ARM: "meanstd"})
    x = {ACT_ARM: torch.zeros(chunk, dim)}
    out = n.normalize(x)[ACT_ARM]
    # out[t] = (0 - mean[t]) / 1 == -mean[t]; per-timestep means preserved.
    torch.testing.assert_close(out, -torch.from_numpy(mean).float(), atol=1e-4, rtol=1e-4)


def test_horizon_stats_sliced_to_shorter_value_horizon():
    """A value with a shorter horizon than the stats must reuse the leading slice."""
    stats = {ACT_ARM: {"mean": np.arange(12, dtype=np.float64).reshape(4, 3), "std": np.ones((4, 3))}}
    n = Normalizer(norm_stats=stats, norm_type={ACT_ARM: "meanstd"})
    x = {ACT_ARM: torch.zeros(2, 3)}  # shorter horizon than the stats' 4
    out = n.normalize(x)[ACT_ARM]
    expected = -torch.from_numpy(np.arange(12, dtype=np.float64).reshape(4, 3)[:2]).float()
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_horizon_mismatch_raises_not_broadcasts():
    """A value horizon longer than the stats must error — never silently broadcast."""
    stats = {ACT_ARM: {"mean": np.zeros((2, 3)), "std": np.ones((2, 3))}}
    n = Normalizer(norm_stats=stats, norm_type={ACT_ARM: "meanstd"})
    with pytest.raises(ValueError, match="horizon mismatch"):
        n.normalize({ACT_ARM: torch.zeros(5, 3)})


def test_dim_mismatch_raises():
    stats = {ACT_ARM: {"mean": np.zeros(3), "std": np.ones(3)}}
    n = Normalizer(norm_stats=stats, norm_type={ACT_ARM: "meanstd"})
    # 5-dim value vs 3-dim stats: the elementwise op cannot broadcast -> RuntimeError.
    with pytest.raises((ValueError, KeyError, IndexError, RuntimeError)):
        n.normalize({ACT_ARM: torch.zeros(5)})


def test_sincos_only_allowed_for_state_keys():
    stats = {"observation.state.joint": {"mean": np.zeros(2), "std": np.ones(2)}}
    n = Normalizer(norm_stats=stats, norm_type={"observation.state.joint": "sincos"})
    out = n.normalize({"observation.state.joint": torch.zeros(2)})
    # sincos concatenates cos and sin -> doubles the last dim.
    assert out["observation.state.joint"].shape[-1] == 4
    # sincos on a non-state key must raise.
    n_bad = Normalizer(norm_stats={ACT_ARM: {"mean": np.zeros(2), "std": np.ones(2)}}, norm_type={ACT_ARM: "sincos"})
    with pytest.raises(ValueError, match="sincos"):
        n_bad.normalize({ACT_ARM: torch.zeros(2)})
