#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
from pathlib import Path

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType
from lerobot.processor import (
    DataProcessorPipeline,
    PolicyActionToRobotActionProcessorStep,
    ProcessorStepRegistry,
    RobotActionToPolicyActionProcessorStep,
)
from lerobot.processor.converters import identity_transition
from lerobot.utils.constants import ACTION
from tests.conftest import assert_contract_is_typed


def test_robot_to_policy_basic_action_conversion():
    """Test basic robot action to policy action conversion."""
    motor_names = ["joint1", "joint2", "joint3"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    robot_action = {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
    }

    policy_action = processor.action(robot_action)

    assert isinstance(policy_action, torch.Tensor)
    assert policy_action.shape == (3,)
    torch.testing.assert_close(policy_action, torch.tensor([1.0, 2.0, 3.0]))


def test_robot_to_policy_action_conversion_preserves_order():
    """Test that motor names order is preserved in conversion."""
    motor_names = ["gripper", "arm", "wrist"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    robot_action = {
        "arm.pos": 10.0,
        "gripper.pos": 5.0,
        "wrist.pos": 15.0,
    }

    policy_action = processor.action(robot_action)

    expected = torch.tensor([5.0, 10.0, 15.0])
    torch.testing.assert_close(policy_action, expected)


def test_robot_to_policy_action_conversion_with_floats_and_tensors():
    """Test conversion with mixed float and tensor values."""
    motor_names = ["joint1", "joint2"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    robot_action = {
        "joint1.pos": torch.tensor(1.5),
        "joint2.pos": 2.5,  # Regular float
    }

    policy_action = processor.action(robot_action)

    assert isinstance(policy_action, torch.Tensor)
    torch.testing.assert_close(policy_action, torch.tensor([1.5, 2.5]))


def test_robot_to_policy_action_length_mismatch_error():
    """Test error when robot action length doesn't match motor names."""
    motor_names = ["joint1", "joint2", "joint3"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    # Too few actions
    robot_action = {"joint1.pos": 1.0, "joint2.pos": 2.0}

    with pytest.raises(ValueError, match="Action must have 3 elements, got 2"):
        processor.action(robot_action)

    robot_action = {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
        "extra.pos": 4.0,
    }

    with pytest.raises(ValueError, match="Action must have 3 elements, got 4"):
        processor.action(robot_action)


def test_robot_to_policy_missing_motor_key_error():
    """Test error when robot action is missing expected motor keys."""
    motor_names = ["joint1", "joint2"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    robot_action = {
        "joint1.pos": 1.0,
        "wrong_key.pos": 2.0,
    }

    with pytest.raises(KeyError):
        processor.action(robot_action)


def test_robot_to_policy_transform_features():
    """Test feature transformation for robot to policy action processor."""
    motor_names = ["joint1", "joint2", "joint3"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    features = {
        PipelineFeatureType.ACTION: {
            "joint1.pos": {"type": FeatureType.ACTION, "shape": (1,)},
            "joint2.pos": {"type": FeatureType.ACTION, "shape": (1,)},
            "joint3.pos": {"type": FeatureType.ACTION, "shape": (1,)},
            "other_data": {"type": FeatureType.ENV, "shape": (1,)},
        }
    }

    transformed = processor.transform_features(features)

    assert ACTION in transformed[PipelineFeatureType.ACTION]
    action_feature = transformed[PipelineFeatureType.ACTION][ACTION]
    assert action_feature.type == FeatureType.ACTION
    assert action_feature.shape == (3,)

    assert "joint1.pos" in transformed[PipelineFeatureType.ACTION]
    assert "joint2.pos" in transformed[PipelineFeatureType.ACTION]
    assert "joint3.pos" in transformed[PipelineFeatureType.ACTION]

    assert "other_data" in transformed[PipelineFeatureType.ACTION]


def test_robot_to_policy_get_config():
    """Test configuration serialization."""
    motor_names = ["motor1", "motor2"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    config = processor.get_config()
    assert config == {"motor_names": motor_names}


def test_robot_to_policy_state_dict():
    """Test state dict operations."""
    processor = RobotActionToPolicyActionProcessorStep(motor_names=["joint1"])

    state = processor.state_dict()
    assert state == {}

    processor.load_state_dict({})


def test_robot_to_policy_single_motor():
    """Test with single motor."""
    processor = RobotActionToPolicyActionProcessorStep(motor_names=["single_joint"])

    robot_action = {"single_joint.pos": 42.0}
    policy_action = processor.action(robot_action)

    assert policy_action.shape == (1,)
    torch.testing.assert_close(policy_action, torch.tensor([42.0]))


def test_policy_to_robot_basic_action_conversion():
    """Test basic policy action to robot action conversion."""
    motor_names = ["joint1", "joint2", "joint3"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    policy_action = torch.tensor([1.0, 2.0, 3.0])
    robot_action = processor.action(policy_action)

    assert isinstance(robot_action, dict)
    assert len(robot_action) == 3

    expected = {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
    }

    for key, expected_value in expected.items():
        assert key in robot_action
        actual_value = robot_action[key]
        if isinstance(actual_value, torch.Tensor):
            actual_value = actual_value.item()
        assert actual_value == pytest.approx(expected_value)


def test_policy_to_robot_action_conversion_preserves_order():
    """Test that motor names order corresponds to tensor indices."""
    motor_names = ["gripper", "arm", "wrist"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    policy_action = torch.tensor([5.0, 10.0, 15.0])
    robot_action = processor.action(policy_action)

    assert robot_action["gripper.pos"] == pytest.approx(5.0)
    assert robot_action["arm.pos"] == pytest.approx(10.0)
    assert robot_action["wrist.pos"] == pytest.approx(15.0)


def test_policy_to_robot_action_conversion_with_numpy_input():
    """Test conversion with numpy array input."""
    import numpy as np

    motor_names = ["joint1", "joint2"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    policy_action = np.array([1.5, 2.5])
    robot_action = processor.action(policy_action)

    assert robot_action["joint1.pos"] == pytest.approx(1.5)
    assert robot_action["joint2.pos"] == pytest.approx(2.5)


def test_policy_to_robot_action_length_mismatch_error():
    """Test error when policy action length doesn't match motor names."""
    motor_names = ["joint1", "joint2", "joint3"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    policy_action = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="Action must have 3 elements, got 2"):
        processor.action(policy_action)

    policy_action = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Action must have 3 elements, got 4"):
        processor.action(policy_action)


def test_policy_to_robot_transform_features():
    """Test feature transformation for policy to robot action processor."""
    motor_names = ["joint1", "joint2"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    features = {
        PipelineFeatureType.ACTION: {
            ACTION: {"type": FeatureType.ACTION, "shape": (2,)},
            "other_data": {"type": FeatureType.ENV, "shape": (1,)},
        }
    }

    transformed = processor.transform_features(features)

    assert "joint1.pos" in transformed[PipelineFeatureType.ACTION]
    assert "joint2.pos" in transformed[PipelineFeatureType.ACTION]

    for motor in motor_names:
        motor_feature = transformed[PipelineFeatureType.ACTION][f"{motor}.pos"]
        assert motor_feature.type == FeatureType.ACTION
        assert motor_feature.shape == (1,)

    assert ACTION in transformed[PipelineFeatureType.ACTION]

    assert "other_data" in transformed[PipelineFeatureType.ACTION]


def test_policy_to_robot_get_config():
    """Test configuration serialization."""
    motor_names = ["motor1", "motor2"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    config = processor.get_config()
    assert config == {"motor_names": motor_names}


def test_policy_to_robot_state_dict():
    """Test state dict operations."""
    processor = PolicyActionToRobotActionProcessorStep(motor_names=["joint1"])

    state = processor.state_dict()
    assert state == {}

    processor.load_state_dict({})


def test_policy_to_robot_single_motor():
    """Test with single motor."""
    processor = PolicyActionToRobotActionProcessorStep(motor_names=["single_joint"])

    policy_action = torch.tensor([42.0])
    robot_action = processor.action(policy_action)

    assert len(robot_action) == 1
    assert robot_action["single_joint.pos"] == pytest.approx(42.0)


def test_robot_to_policy_registry():
    """Test RobotActionToPolicyActionProcessorStep registry."""
    assert "robot_action_to_policy_action_processor" in ProcessorStepRegistry.list()

    retrieved_class = ProcessorStepRegistry.get("robot_action_to_policy_action_processor")
    assert retrieved_class is RobotActionToPolicyActionProcessorStep

    instance = retrieved_class(motor_names=["test"])
    assert isinstance(instance, RobotActionToPolicyActionProcessorStep)
    assert instance.motor_names == ["test"]


def test_policy_to_robot_registry():
    """Test PolicyActionToRobotActionProcessorStep registry."""
    assert "policy_action_to_robot_action_processor" in ProcessorStepRegistry.list()

    retrieved_class = ProcessorStepRegistry.get("policy_action_to_robot_action_processor")
    assert retrieved_class is PolicyActionToRobotActionProcessorStep

    instance = retrieved_class(motor_names=["test"])
    assert isinstance(instance, PolicyActionToRobotActionProcessorStep)
    assert instance.motor_names == ["test"]


def test_save_and_load_robot_to_policy():
    """Test saving and loading RobotActionToPolicyActionProcessorStep."""
    motor_names = ["joint1", "joint2", "joint3"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)
    pipeline = DataProcessorPipeline([processor], name="TestRobotToPolicy")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)

        # Check config file exists
        config_path = Path(tmp_dir) / "testrobottopolicy.json"
        assert config_path.exists()

        # Load pipeline
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            "testrobottopolicy.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        assert loaded_pipeline.name == "TestRobotToPolicy"
        assert len(loaded_pipeline) == 1

        # Check loaded processor
        loaded_processor = loaded_pipeline.steps[0]
        assert isinstance(loaded_processor, RobotActionToPolicyActionProcessorStep)
        assert loaded_processor.motor_names == motor_names

        # Test functionality after loading
        robot_action = {"joint1.pos": 1.0, "joint2.pos": 2.0, "joint3.pos": 3.0}
        policy_action = loaded_processor.action(robot_action)
        torch.testing.assert_close(policy_action, torch.tensor([1.0, 2.0, 3.0]))


def test_save_and_load_policy_to_robot():
    """Test saving and loading PolicyActionToRobotActionProcessorStep."""
    motor_names = ["motor_a", "motor_b"]
    processor = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)
    pipeline = DataProcessorPipeline([processor], name="TestPolicyToRobot")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)

        # Load pipeline
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            "testpolicytorobot.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        loaded_processor = loaded_pipeline.steps[0]
        assert isinstance(loaded_processor, PolicyActionToRobotActionProcessorStep)
        assert loaded_processor.motor_names == motor_names

        policy_action = torch.tensor([10.0, 20.0])
        robot_action = loaded_processor.action(policy_action)
        assert robot_action["motor_a.pos"] == pytest.approx(10.0)
        assert robot_action["motor_b.pos"] == pytest.approx(20.0)


# Integration and chaining tests


def test_round_trip_conversion():
    """Test that robot->policy->robot conversion preserves values."""
    motor_names = ["joint1", "joint2", "joint3"]
    robot_to_policy = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)
    policy_to_robot = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    original_robot_action = {
        "joint1.pos": 1.5,
        "joint2.pos": -2.3,
        "joint3.pos": 0.7,
    }

    policy_action = robot_to_policy.action(original_robot_action)
    final_robot_action = policy_to_robot.action(policy_action)

    for key in original_robot_action:
        original_val = original_robot_action[key]
        final_val = final_robot_action[key]
        if isinstance(final_val, torch.Tensor):
            final_val = final_val.item()
        assert final_val == pytest.approx(original_val, abs=1e-6)


def test_chained_processors_in_pipeline():
    """Test both processors chained in a pipeline."""
    motor_names = ["joint1", "joint2"]
    robot_to_policy = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)
    policy_to_robot = PolicyActionToRobotActionProcessorStep(motor_names=motor_names)

    pipeline = DataProcessorPipeline(
        [robot_to_policy, policy_to_robot],
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.steps[0], RobotActionToPolicyActionProcessorStep)
    assert isinstance(pipeline.steps[1], PolicyActionToRobotActionProcessorStep)


def test_robot_to_policy_features_contract(policy_feature_factory):
    """Test feature transformation maintains proper typing contract."""
    processor = RobotActionToPolicyActionProcessorStep(motor_names=["j1", "j2"])
    features = {
        PipelineFeatureType.ACTION: {
            "j1.pos": policy_feature_factory(FeatureType.ACTION, (1,)),
            "j2.pos": policy_feature_factory(FeatureType.ACTION, (1,)),
            "other": policy_feature_factory(FeatureType.ENV, (3,)),
        }
    }

    out = processor.transform_features(features.copy())

    assert_contract_is_typed(out)

    assert ACTION in out[PipelineFeatureType.ACTION]
    action_feature = out[PipelineFeatureType.ACTION][ACTION]
    assert action_feature.type == FeatureType.ACTION
    assert action_feature.shape == (2,)


def test_policy_to_robot_features_contract(policy_feature_factory):
    """Test feature transformation maintains proper typing contract."""
    processor = PolicyActionToRobotActionProcessorStep(motor_names=["m1", "m2", "m3"])
    features = {
        PipelineFeatureType.ACTION: {
            ACTION: policy_feature_factory(FeatureType.ACTION, (3,)),
            "other": policy_feature_factory(FeatureType.ENV, (1,)),
        }
    }

    out = processor.transform_features(features.copy())

    assert_contract_is_typed(out)

    for motor in ["m1", "m2", "m3"]:
        key = f"{motor}.pos"
        assert key in out[PipelineFeatureType.ACTION]
        motor_feature = out[PipelineFeatureType.ACTION][key]
        assert motor_feature.type == FeatureType.ACTION
        assert motor_feature.shape == (1,)


def test_empty_motor_names_list():
    """Test behavior with empty motor names list."""
    processor = RobotActionToPolicyActionProcessorStep(motor_names=[])

    robot_action = {}
    policy_action = processor.action(robot_action)

    assert isinstance(policy_action, torch.Tensor)
    assert policy_action.shape == (0,)


def test_empty_motor_names_list_policy_to_robot():
    """Test PolicyActionToRobotActionProcessorStep with empty motor names."""
    processor = PolicyActionToRobotActionProcessorStep(motor_names=[])

    policy_action = torch.tensor([])
    robot_action = processor.action(policy_action)

    assert isinstance(robot_action, dict)
    assert len(robot_action) == 0


def test_very_long_motor_names():
    """Test with many motor names."""
    motor_names = [f"joint_{i}" for i in range(100)]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    robot_action = {f"joint_{i}.pos": float(i) for i in range(100)}
    policy_action = processor.action(robot_action)

    assert policy_action.shape == (100,)
    expected = torch.tensor([float(i) for i in range(100)])
    torch.testing.assert_close(policy_action, expected)


def test_special_characters_in_motor_names():
    """Test with special characters in motor names."""
    motor_names = ["motor-1", "motor_2", "motor.3"]
    processor = RobotActionToPolicyActionProcessorStep(motor_names=motor_names)

    robot_action = {
        "motor-1.pos": 1.0,
        "motor_2.pos": 2.0,
        "motor.3.pos": 3.0,
    }

    policy_action = processor.action(robot_action)
    torch.testing.assert_close(policy_action, torch.tensor([1.0, 2.0, 3.0]))
