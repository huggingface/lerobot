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

import math

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    ActionSafetyProcessorStep,
    DataProcessorPipeline,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION


def test_registry_registration():
    assert ProcessorStepRegistry.get("action_safety_processor") is ActionSafetyProcessorStep


def test_passthrough_when_action_is_none():
    step = ActionSafetyProcessorStep(action_min=-1.0, action_max=1.0, max_delta=0.1)
    transition = create_transition(observation={"state": torch.zeros(3)}, action=None)
    out = step(transition)

    assert out[TransitionKey.ACTION] is None
    assert out is not transition


def test_unsupported_action_type_raises():
    step = ActionSafetyProcessorStep()
    transition = create_transition(action="invalid_action")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="expects action to be"):
        step(transition)


# ---------------------------------------------------------------------------
# Tensor actions
# ---------------------------------------------------------------------------


def test_tensor_bounds_clamping_scalar_and_per_dim():
    raw = torch.tensor([[-2.0, 0.5, 3.0], [0.0, -5.0, 1.5]], dtype=torch.float32)
    raw_copy = raw.clone()

    step_scalar = ActionSafetyProcessorStep(action_min=-1.0, action_max=1.0)
    out_scalar = step_scalar(create_transition(action=raw))[TransitionKey.ACTION]
    torch.testing.assert_close(out_scalar, torch.tensor([[-1.0, 0.5, 1.0], [0.0, -1.0, 1.0]]))
    torch.testing.assert_close(raw, raw_copy)  # no in-place mutation

    step_per_dim = ActionSafetyProcessorStep(action_min=[-1.5, -2.0, 0.0], action_max=[0.5, 1.0, 2.0])
    out_per_dim = step_per_dim(create_transition(action=raw))[TransitionKey.ACTION]
    torch.testing.assert_close(out_per_dim, torch.tensor([[-1.5, 0.5, 2.0], [0.0, -2.0, 1.5]]))
    assert out_per_dim.dtype == torch.float32


def test_tensor_one_sided_bounds_and_dtype_preservation():
    raw = torch.tensor([-3.0, 0.0, 4.0], dtype=torch.float64)
    step_min = ActionSafetyProcessorStep(action_min=-1.0)
    step_max = ActionSafetyProcessorStep(action_max=2.0)

    out_min = step_min(create_transition(action=raw))[TransitionKey.ACTION]
    out_max = step_max(create_transition(action=raw))[TransitionKey.ACTION]

    torch.testing.assert_close(out_min, torch.tensor([-1.0, 0.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(out_max, torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float64))
    assert out_min.dtype == torch.float64
    assert out_max.dtype == torch.float64


def test_tensor_non_finite_policies():
    bad = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])

    step_raise = ActionSafetyProcessorStep(on_non_finite="raise")
    with pytest.raises(ValueError, match="Non-finite action values detected"):
        step_raise(create_transition(action=bad))

    step_zero = ActionSafetyProcessorStep(on_non_finite="zero")
    out_zero = step_zero(create_transition(action=bad))[TransitionKey.ACTION]
    torch.testing.assert_close(out_zero, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    step_prev = ActionSafetyProcessorStep(on_non_finite="previous")
    # First step has no history -> falls back to 0.0
    out_first = step_prev(create_transition(action=bad))[TransitionKey.ACTION]
    torch.testing.assert_close(out_first, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Establish a safe previous action, then feed non-finite values
    step_prev(create_transition(action=torch.tensor([2.0, 3.0, 4.0, 5.0])))
    out_next = step_prev(create_transition(action=bad))[TransitionKey.ACTION]
    torch.testing.assert_close(out_next, torch.tensor([1.0, 3.0, 4.0, 5.0]))


def test_tensor_check_finite_disabled():
    step = ActionSafetyProcessorStep(check_finite=False)
    out = step(create_transition(action=torch.tensor([1.0, float("nan")])))[TransitionKey.ACTION]
    assert torch.isnan(out[1])


def test_tensor_slew_rate_limiting_and_reset():
    step = ActionSafetyProcessorStep(max_delta=[0.5, 1.0], action_min=-2.0, action_max=2.0)

    # Step 0: no previous action, only bounds apply
    a0 = step(create_transition(action=torch.tensor([0.0, 0.0])))[TransitionKey.ACTION]
    torch.testing.assert_close(a0, torch.tensor([0.0, 0.0]))

    # Step 1: large jump is clamped by max_delta
    a1 = step(create_transition(action=torch.tensor([3.0, -3.0])))[TransitionKey.ACTION]
    torch.testing.assert_close(a1, torch.tensor([0.5, -1.0]))

    # Step 2: continues smoothly from a1
    a2 = step(create_transition(action=torch.tensor([3.0, -3.0])))[TransitionKey.ACTION]
    torch.testing.assert_close(a2, torch.tensor([1.0, -2.0]))

    # Step 3: hits action_max / action_min bounds
    a3 = step(create_transition(action=torch.tensor([3.0, -3.0])))[TransitionKey.ACTION]
    torch.testing.assert_close(a3, torch.tensor([1.5, -2.0]))

    # Reset clears history so next step can jump freely within bounds
    step.reset()
    a_new = step(create_transition(action=torch.tensor([-1.5, 1.5])))[TransitionKey.ACTION]
    torch.testing.assert_close(a_new, torch.tensor([-1.5, 1.5]))


def test_tensor_action_chunk_3d_support():
    step = ActionSafetyProcessorStep(action_min=-1.0, action_max=1.0, max_delta=0.25)
    chunk0 = torch.zeros(2, 4, 3)
    chunk1 = torch.ones(2, 4, 3)

    out0 = step(create_transition(action=chunk0))[TransitionKey.ACTION]
    out1 = step(create_transition(action=chunk1))[TransitionKey.ACTION]

    torch.testing.assert_close(out0, torch.zeros(2, 4, 3))
    torch.testing.assert_close(out1, torch.full((2, 4, 3), 0.25))


def test_tensor_max_delta_zero_holds_initial_action():
    step = ActionSafetyProcessorStep(max_delta=0.0)
    a0 = step(create_transition(action=torch.tensor([0.4, -0.2])))[TransitionKey.ACTION]
    a1 = step(create_transition(action=torch.tensor([5.0, -5.0])))[TransitionKey.ACTION]

    torch.testing.assert_close(a0, torch.tensor([0.4, -0.2]))
    torch.testing.assert_close(a1, torch.tensor([0.4, -0.2]))


def test_tensor_shape_and_dim_mismatch_errors():
    step_dim = ActionSafetyProcessorStep(action_min=[-1.0, -1.0])
    with pytest.raises(ValueError, match="does not match action last dimension"):
        step_dim(create_transition(action=torch.zeros(3)))

    step_dict = ActionSafetyProcessorStep(action_min={"j0": -1.0})
    with pytest.raises(ValueError, match="only supported for dict robot actions"):
        step_dict(create_transition(action=torch.zeros(2)))

    step_state = ActionSafetyProcessorStep(max_delta=0.1)
    step_state(create_transition(action=torch.zeros(2, 3)))
    with pytest.raises(ValueError, match="Action shape changed"):
        step_state(create_transition(action=torch.zeros(1, 3)))


# ---------------------------------------------------------------------------
# NumPy actions
# ---------------------------------------------------------------------------


def test_numpy_finite_bounds_and_slew_rate():
    step = ActionSafetyProcessorStep(
        on_non_finite="previous",
        action_min=[-1.0, -2.0],
        action_max=[1.0, 2.0],
        max_delta=0.5,
    )
    a0 = np.array([0.0, 0.0], dtype=np.float32)
    a0_copy = a0.copy()
    out0 = step(create_transition(action=a0))[TransitionKey.ACTION]
    np.testing.assert_allclose(out0, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(a0, a0_copy)
    assert out0.dtype == np.float32

    out1 = step(create_transition(action=np.array([np.nan, 10.0], dtype=np.float32)))[TransitionKey.ACTION]
    np.testing.assert_allclose(out1, np.array([0.0, 0.5], dtype=np.float32))
    assert out1.dtype == np.float32


def test_numpy_non_finite_raises():
    step = ActionSafetyProcessorStep(on_non_finite="raise")
    with pytest.raises(ValueError, match="Non-finite action values detected"):
        step(create_transition(action=np.array([0.1, np.inf], dtype=np.float32)))


# ---------------------------------------------------------------------------
# Dict RobotAction
# ---------------------------------------------------------------------------


def test_dict_robot_action_bounds_slew_rate_and_non_finite():
    step = ActionSafetyProcessorStep(
        on_non_finite="previous",
        action_min={"shoulder.pos": -1.0, "gripper.pos": 0.0},
        action_max={"shoulder.pos": 1.0, "gripper.pos": 1.0},
        max_delta={"shoulder.pos": 0.2, "gripper.pos": 0.5},
    )
    raw0 = {"shoulder.pos": 0.0, "gripper.pos": 0.2, "unbounded.pos": 50.0, "mode": "teleop"}
    out0 = step(create_transition(action=raw0))[TransitionKey.ACTION]
    assert out0 == {"shoulder.pos": 0.0, "gripper.pos": 0.2, "unbounded.pos": 50.0, "mode": "teleop"}
    assert raw0["shoulder.pos"] == 0.0

    out1 = step(
        create_transition(
            action={"shoulder.pos": 5.0, "gripper.pos": math.nan, "unbounded.pos": 100.0, "mode": "auto"}
        )
    )[TransitionKey.ACTION]
    assert out1["shoulder.pos"] == pytest.approx(0.2)
    assert out1["gripper.pos"] == pytest.approx(0.2)
    assert out1["unbounded.pos"] == pytest.approx(100.0)
    assert out1["mode"] == "auto"

    step.reset()
    out2 = step(create_transition(action={"shoulder.pos": 5.0, "gripper.pos": -1.0}))[TransitionKey.ACTION]
    assert out2 == {"shoulder.pos": 1.0, "gripper.pos": 0.0}


def test_dict_non_finite_raises_and_list_bound_rejected():
    step = ActionSafetyProcessorStep(on_non_finite="raise")
    with pytest.raises(ValueError, match="Non-finite action value detected for key 'j0'"):
        step(create_transition(action={"j0": float("nan")}))

    step_list = ActionSafetyProcessorStep(action_min=[-1.0, -1.0])
    with pytest.raises(ValueError, match="only supported for tensor/array actions"):
        step_list(create_transition(action={"j0": 0.5}))


def test_action_type_change_between_steps_raises():
    step = ActionSafetyProcessorStep(max_delta=0.1)
    step(create_transition(action=torch.zeros(2)))
    with pytest.raises(ValueError, match="Action type changed"):
        step(create_transition(action={"j0": 0.0}))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,exc_type,match",
    [
        ({"on_non_finite": "clip"}, ValueError, "on_non_finite"),
        ({"action_min": float("nan")}, ValueError, "must be finite"),
        ({"action_max": float("inf")}, ValueError, "must be finite"),
        ({"max_delta": -0.1}, ValueError, "must be >= 0"),
        ({"max_delta": [0.1, -0.2]}, ValueError, "must be >= 0"),
        ({"max_delta": {"j0": -1.0}}, ValueError, "must be >= 0"),
        ({"action_min": 1.0, "action_max": 0.0}, ValueError, "must be <= action_max"),
        ({"action_min": [0.0, 2.0], "action_max": [1.0, 1.0]}, ValueError, "must be <= action_max"),
        ({"action_min": [0.0, 1.0], "action_max": [1.0]}, ValueError, "must match action_max length"),
        ({"action_min": []}, ValueError, "must not be empty"),
        ({"action_min": [0.0], "action_max": {"j0": 1.0}}, ValueError, "cannot mix"),
        ({"action_min": {"j0": 2.0}, "action_max": {"j0": 1.0}}, ValueError, "must be <= action_max"),
        ({"action_min": True}, TypeError, "got bool"),
    ],
)
def test_invalid_config_raises(kwargs, exc_type, match):
    with pytest.raises(exc_type, match=match):
        ActionSafetyProcessorStep(**kwargs)


# ---------------------------------------------------------------------------
# Feature contract & Pipeline serialization round-trip
# ---------------------------------------------------------------------------


def test_transform_features_preserves_contract():
    step = ActionSafetyProcessorStep(action_min=-1.0, action_max=1.0, max_delta=0.2)
    features = {
        PipelineFeatureType.OBSERVATION: {"q": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        PipelineFeatureType.ACTION: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    }
    assert step.transform_features(features) == features


def test_pipeline_save_load_and_reset(tmp_path):
    step = ActionSafetyProcessorStep(
        check_finite=True,
        on_non_finite="previous",
        action_min=[-1.0, -2.0],
        action_max=[1.0, 2.0],
        max_delta=0.25,
    )
    pipeline = DataProcessorPipeline(steps=[step], name="post")
    pipeline.save_pretrained(tmp_path)

    loaded_pipeline = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="post.json")
    loaded_step = loaded_pipeline.steps[0]

    assert isinstance(loaded_step, ActionSafetyProcessorStep)
    assert loaded_step.get_config() == step.get_config()

    # Verify pipeline.reset() propagates to the safety step
    a0 = loaded_pipeline.process_action(torch.tensor([0.0, 0.0]))
    a1 = loaded_pipeline.process_action(torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(a0, torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(a1, torch.tensor([0.25, 0.25]))

    loaded_pipeline.reset()
    a_reset = loaded_pipeline.process_action(torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(a_reset, torch.tensor([1.0, 1.0]))
