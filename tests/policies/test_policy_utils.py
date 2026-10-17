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

"""Correctness tests for `prepare_observation_for_inference`'s reordered image pipeline
(2026-08-05): device transfer moved before the uint8->float32 conversion and channel permute,
instead of after, for real measured performance reasons (see the function's own docstring and
`Experiments/engineering/Engineering.md` in the parent research repo). These tests pin that the
reordering is a pure performance change, not a behavior change: a local copy of the *original*
step order is kept below as a ground-truth oracle, and every test compares the real function's
output against it, rather than against hand-picked expected values that could drift.
"""

import numpy as np
import pytest
import torch

from lerobot.policies.utils import prepare_observation_for_inference


def _reference_prepare_observation_for_inference(observation, device, task=None, robot_type=None):
    """The pre-2026-08-05 implementation, kept verbatim as a ground-truth oracle. Deliberately
    duplicated rather than imported (there's nothing left to import it from) or reconstructed from
    the new code (that would make the test circular)."""
    observation = dict(observation)
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            if observation[name].dtype == torch.uint8:
                observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)
    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""
    return observation


def _make_observation():
    rng = np.random.default_rng(0)
    return {
        "observation.images.front": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation.state": rng.uniform(-1, 1, size=(6,)).astype(np.float32),
    }


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device_str", DEVICES)
def test_matches_reference_implementation_exactly(device_str):
    device = torch.device(device_str)
    obs_a = _make_observation()
    obs_b = {k: v.copy() for k, v in obs_a.items()}

    actual = prepare_observation_for_inference(
        obs_a, device, task="pick up the screwdriver", robot_type="so101"
    )
    expected = _reference_prepare_observation_for_inference(
        obs_b, device, task="pick up the screwdriver", robot_type="so101"
    )

    for key in ("observation.images.front", "observation.images.wrist", "observation.state"):
        # Exact (rtol=0, atol=0) on CPU: both implementations do the /255 divide on the same
        # device with the same op, so there's no source of divergence at all. On CUDA, the divide
        # now happens on-GPU instead of on-CPU (that's the whole point of the reordering) --
        # CPU and GPU floating-point division aren't guaranteed bit-identical (different hardware
        # implementations of the same IEEE754 operation), so a tiny, real, expected tolerance is
        # correct here, not a sign of a bug. Observed magnitude on this hardware: ~6e-8, i.e.
        # utterly negligible against a uint8 image's own ~0.0039 quantization step.
        tol = {"rtol": 0, "atol": 0} if device.type == "cpu" else {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(actual[key], expected[key], **tol)
        assert actual[key].device.type == device.type
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype

    assert actual["task"] == expected["task"] == "pick up the screwdriver"
    assert actual["robot_type"] == expected["robot_type"] == "so101"


def test_image_normalized_to_unit_range_and_chw():
    obs = _make_observation()
    result = prepare_observation_for_inference(obs, torch.device("cpu"))
    img = result["observation.images.front"]
    assert img.shape == (1, 3, 480, 640)  # (batch, C, H, W)
    assert img.dtype == torch.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_image_tensor_is_contiguous():
    """The permute alone would leave a non-contiguous view; `.contiguous()` must still run after
    the reordering, not get lost in the process of moving the device transfer earlier."""
    obs = _make_observation()
    result = prepare_observation_for_inference(obs, torch.device("cpu"))
    assert result["observation.images.front"].is_contiguous()


def test_non_image_key_gets_batch_dim_and_device_but_no_normalization():
    obs = _make_observation()
    original_state = obs["observation.state"].copy()
    result = prepare_observation_for_inference(obs, torch.device("cpu"))
    state = result["observation.state"]
    assert state.shape == (1, 6)
    torch.testing.assert_close(state.squeeze(0), torch.from_numpy(original_state), rtol=0, atol=0)


def test_missing_task_and_robot_type_default_to_empty_string():
    obs = _make_observation()
    result = prepare_observation_for_inference(obs, torch.device("cpu"))
    assert result["task"] == ""
    assert result["robot_type"] == ""


def test_already_float_image_is_not_renormalized():
    """An already-float image (dtype != uint8) must skip the /255 step entirely, same as the
    original implementation's `if observation[name].dtype == torch.uint8` guard."""
    obs = {
        "observation.images.front": np.full((4, 4, 3), 0.5, dtype=np.float32),
    }
    result = prepare_observation_for_inference(obs, torch.device("cpu"))
    torch.testing.assert_close(
        result["observation.images.front"], torch.full((1, 3, 4, 4), 0.5), rtol=0, atol=0
    )
