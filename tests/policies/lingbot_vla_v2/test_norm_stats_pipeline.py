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

"""End-to-end norm-stats pipeline comparison.

Simulates the upstream ``scripts/compute_norm_stats.py`` data flow —
feature items out of a LeRobot dataset → FeatureTransform (slot mapping,
subtract_state) with normalization and padding disabled → RunningStats →
save — and checks:

1. The synthetic dataset round-trips through ``FeatureTransform.apply`` with
   ``do_normalize=False, return_item_before_padding=True, processor=None``
   without touching the tokenizer / image processor (the stats path must be
   image-free, matching upstream ``disabled_image_features=True``).
2. RunningStats over the transformed items matches ground-truth numpy
   statistics computed from the raw arrays.
3. subtract_state (relative action) semantics: the stats are taken over
   ``action - state``, so the action-mean stats equal ``mean(action) −
   mean(state)`` — this is the case where the old "slice the raw stats"
   shortcut is numerically wrong.
4. The saved norm_stats.json can be consumed by ``Normalizer`` (the exact
   class the training preprocessor builds) and round-trips: normalize →
   unnormalize recovers the input.
"""

from __future__ import annotations

import json
import sys

# Bypass ``lerobot.policies.lingbot_vla_v2.__init__`` (pulls transformers): load
# the preprocessing modules under their real dotted names so the package-relative
# imports inside them (``from .data_transform import ...``) resolve — while never
# executing the heavy ``lingbot_vla_v2/__init__.py``.
import types as _types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_PP = Path(__file__).resolve().parents[3] / "src/lerobot/policies/lingbot_vla_v2/preprocessing"
_PKG = "lerobot.policies.lingbot_vla_v2.preprocessing"

# Stub out the two ancestors whose __init__ would pull heavy deps; register the
# real preprocessing package with __path__ so submodule imports work.
for _stub_name in ("lerobot.policies", "lerobot.policies.lingbot_vla_v2"):
    if _stub_name not in sys.modules:
        _stub = _types.ModuleType(_stub_name)
        _stub.__path__ = []
        sys.modules[_stub_name] = _stub
_pkg_mod = _types.ModuleType(_PKG)
_pkg_mod.__path__ = [str(_PP)]
sys.modules[_PKG] = _pkg_mod

from lerobot.policies.lingbot_vla_v2.preprocessing.data_transform import Normalizer  # noqa: E402
from lerobot.policies.lingbot_vla_v2.preprocessing.feature_transform import FeatureTransform  # noqa: E402
from lerobot.policies.lingbot_vla_v2.preprocessing.norm_stats import RunningStats, save  # noqa: E402

CHUNK = 4  # synthetic chunk; the real config is 50
DIM_ARM = 2  # synthetic arm joints
DIM_EFF = 1

ROBOT_CONFIG = {
    "states": [
        {
            "observation.state.arm.position": {
                "origin_keys": [{"observation.state": {"start": 0, "end": DIM_ARM}}]
            }
        },
        {
            "observation.state.effector.position": {
                "origin_keys": [{"observation.state": {"start": DIM_ARM, "end": DIM_ARM + DIM_EFF}}]
            }
        },
    ],
    "actions": [
        {
            "action.arm.position": {
                "origin_keys": [{"action": {"start": 0, "end": DIM_ARM}}],
                "subtract_state": False,
            }
        },
        {
            "action.effector.position": {
                "origin_keys": [{"action": {"start": DIM_ARM, "end": DIM_ARM + DIM_EFF}}],
                "subtract_state": False,
            }
        },
    ],
    "images": [],
}

ROBOT_CONFIG_REL = json.loads(json.dumps(ROBOT_CONFIG))
ROBOT_CONFIG_REL["actions"][0]["action.arm.position"]["subtract_state"] = True


def _make_data_config(norm_type="meanstd"):
    joints = {
        "arm.position": 14,
        "effector.position": 2,
    }
    return SimpleNamespace(
        joints=[f"{{'{k}': {v}}}" for k, v in joints.items()],
        norm_type=[f"{{'{k}': '{norm_type}'}}" for k in joints],
        cameras=[],
        img_size=224,
        chat_template="default",
        text_keys="task",
    )


def _make_model_config():
    return SimpleNamespace(
        max_state_dim=55,
        max_action_dim=55,
        chunk_size=CHUNK,
        tokenizer_max_length=72,
        use_qwen3_chat_template=True,
        return_image_grid_thw=True,
        qwen3vl_use_vision_boundaries=True,
        resize_imgs_with_padding=(224, 224),
    )


def _make_items(n: int, seed: int = 0, state_scale=(0.7, 1.3), action_scale=(0.5, 2.0)):
    """Synthetic dataset items: raw state (DIM_ARM+DIM_EFF) and action chunk
    (CHUNK, DIM_ARM+DIM_EFF), no images — the stats path is image-free."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        state = rng.normal(0.5, state_scale[0], size=(DIM_ARM + DIM_EFF,)).astype(np.float64)
        state[-1] = np.clip(state[-1], 0.0, 1.0)  # gripper
        action = (
            rng.normal(0.2, action_scale[0], size=(CHUNK, DIM_ARM + DIM_EFF)).astype(np.float64)
        )
        action[:, -1] = np.clip(action[:, -1], 0.0, 1.0)
        items.append(
            {
                "observation.state": torch.from_numpy(state),
                "action": torch.from_numpy(action),
                "task": "pick the cube",
                "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
            }
        )
    return items


def _run_stats(robot_config, items):
    ft = FeatureTransform(
        robot_config=robot_config,
        data_config=_make_data_config(),
        model_config=_make_model_config(),
        processor=None,
        do_normalize=False,
        return_item_before_padding=True,
        disabled_image_features=True,
        chunk_size=CHUNK,
    )
    state_keys = list(ft.states)
    action_keys = list(ft.actions)
    accum = {k: RunningStats() for k in state_keys + action_keys}
    for item in items:
        out = ft.apply(item, policy_eval=False)
        for k in state_keys:
            v = out[k]
            accum[k].update(np.asarray(v).reshape(-1, v.shape[-1]))
        for k in action_keys:
            v = out[k]
            accum[k].update(np.asarray(v).reshape(-1, v.shape[-1]))
    return ft, {k: s.get_statistics() for k, s in accum.items()}


def test_transformed_state_stats_match_ground_truth():
    items = _make_items(64)
    _, stats = _run_stats(ROBOT_CONFIG, items)
    raw_state = np.stack([np.asarray(i["observation.state"]) for i in items])  # [N, 3]
    arm = stats["observation.state.arm.position"]
    np.testing.assert_allclose(arm.mean, raw_state[:, :DIM_ARM].mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(arm.std, raw_state[:, :DIM_ARM].std(axis=0), rtol=1e-9)
    eff = stats["observation.state.effector.position"]
    np.testing.assert_allclose(eff.mean, raw_state[:, DIM_ARM:].mean(axis=0), rtol=1e-12)


def test_absolute_action_stats_match_ground_truth():
    items = _make_items(64)
    _, stats = _run_stats(ROBOT_CONFIG, items)
    raw_action = np.stack([np.asarray(i["action"]) for i in items]).reshape(-1, DIM_ARM + DIM_EFF)
    arm = stats["action.arm.position"]
    np.testing.assert_allclose(arm.mean, raw_action[:, :DIM_ARM].mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(arm.std, raw_action[:, :DIM_ARM].std(axis=0), rtol=1e-9)


def test_subtract_state_action_stats_are_relative():
    """The case the old slice-the-raw-stats shortcut gets wrong: with
    subtract_state=True the stats describe ``action − state``, not ``action``."""
    items = _make_items(64, state_scale=(0.7, 0.0), action_scale=(0.5, 0.0))
    _, stats = _run_stats(ROBOT_CONFIG_REL, items)
    raw_action = np.stack([np.asarray(i["action"]) for i in items]).reshape(-1, DIM_ARM + DIM_EFF)
    raw_state = np.stack([np.asarray(i["observation.state"]) for i in items])  # [N, 3]
    # state broadcasts over the chunk dim (each action row subtracts that sample's state).
    states_expanded = np.repeat(raw_state[:, None, :], CHUNK, axis=1).reshape(-1, DIM_ARM + DIM_EFF)
    rel = raw_action - states_expanded
    arm = stats["action.arm.position"]
    np.testing.assert_allclose(arm.mean, rel[:, :DIM_ARM].mean(axis=0), rtol=1e-10)
    # And it must NOT match the absolute stats — that is the failure mode being guarded.
    assert not np.allclose(arm.mean, raw_action[:, :DIM_ARM].mean(axis=0), rtol=1e-3)


def test_norm_stats_file_consumed_by_normalizer_roundtrip(tmp_path):
    items = _make_items(64)
    _, stats = _run_stats(ROBOT_CONFIG, items)
    out = tmp_path / "norm_stats.json"
    save(out, stats, count=64 * (1 + CHUNK))
    saved = json.loads(out.read_text())

    norm_type = {
        "observation.state.arm.position": "meanstd",
        "observation.state.effector.position": "meanstd",
        "action.arm.position": "meanstd",
        "action.effector.position": "meanstd",
    }
    normalizer = Normalizer(norm_stats=saved["norm_stats"], norm_type=norm_type)
    probe = {
        "observation.state.arm.position": np.array([0.5, 0.5]),
        "action.arm.position": np.array([[0.1, 0.2]] * CHUNK),
    }
    normed = normalizer.normalize(probe)
    denormed = normalizer.unnormalize(normed)
    for k, v in probe.items():
        np.testing.assert_allclose(denormed[k], v, rtol=1e-10, atol=1e-12)
