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

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import DataProcessorPipeline, DeviceProcessorStep, TransitionKey
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def test_basic_functionality():
    """Test basic device processor functionality on CPU."""
    processor = DeviceProcessorStep(device="cpu")

    # Create a transition with CPU tensors
    observation = {OBS_STATE: torch.randn(10), OBS_IMAGE: torch.randn(3, 224, 224)}
    action = torch.randn(5)
    reward = torch.tensor(1.0)
    done = torch.tensor(False)
    truncated = torch.tensor(False)

    transition = create_transition(
        observation=observation, action=action, reward=reward, done=done, truncated=truncated
    )

    result = processor(transition)

    # Check that all tensors are on CPU
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cpu"
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].device.type == "cpu"
    assert result[TransitionKey.ACTION].device.type == "cpu"
    assert result[TransitionKey.REWARD].device.type == "cpu"
    assert result[TransitionKey.DONE].device.type == "cpu"
    assert result[TransitionKey.TRUNCATED].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_functionality():
    """Test device processor functionality on CUDA."""
    processor = DeviceProcessorStep(device="cuda")

    # Create a transition with CPU tensors
    observation = {OBS_STATE: torch.randn(10), OBS_IMAGE: torch.randn(3, 224, 224)}
    action = torch.randn(5)
    reward = torch.tensor(1.0)
    done = torch.tensor(False)
    truncated = torch.tensor(False)

    transition = create_transition(
        observation=observation, action=action, reward=reward, done=done, truncated=truncated
    )

    result = processor(transition)

    # Check that all tensors are on CUDA
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.type == "cuda"
    assert result[TransitionKey.REWARD].device.type == "cuda"
    assert result[TransitionKey.DONE].device.type == "cuda"
    assert result[TransitionKey.TRUNCATED].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_specific_cuda_device():
    """Test device processor with specific CUDA device."""
    processor = DeviceProcessorStep(device="cuda:0")

    observation = {OBS_STATE: torch.randn(10)}
    action = torch.randn(5)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.index == 0
    assert result[TransitionKey.ACTION].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.index == 0


def test_non_tensor_values():
    """Test that non-tensor values are preserved."""
    processor = DeviceProcessorStep(device="cpu")

    observation = {
        OBS_STATE: torch.randn(10),
        "observation.metadata": {"key": "value"},  # Non-tensor data
        "observation.list": [1, 2, 3],  # Non-tensor data
    }
    action = torch.randn(5)
    info = {"episode": 1, "step": 42}

    transition = create_transition(observation=observation, action=action, info=info)

    result = processor(transition)

    # Check tensors are processed
    assert isinstance(result[TransitionKey.OBSERVATION][OBS_STATE], torch.Tensor)
    assert isinstance(result[TransitionKey.ACTION], torch.Tensor)

    # Check non-tensor values are preserved
    assert result[TransitionKey.OBSERVATION]["observation.metadata"] == {"key": "value"}
    assert result[TransitionKey.OBSERVATION]["observation.list"] == [1, 2, 3]
    assert result[TransitionKey.INFO] == {"episode": 1, "step": 42}


def test_none_values():
    """Test handling of None values."""
    processor = DeviceProcessorStep(device="cpu")

    # Test with None observation
    transition = create_transition(observation=None, action=torch.randn(5))
    result = processor(transition)
    assert result[TransitionKey.OBSERVATION] is None
    assert result[TransitionKey.ACTION].device.type == "cpu"

    # Test with None action
    transition = create_transition(observation={OBS_STATE: torch.randn(10)}, action=None)
    result = processor(transition)
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cpu"
    assert result[TransitionKey.ACTION] is None


def test_empty_observation():
    """Test handling of empty observation dictionary."""
    processor = DeviceProcessorStep(device="cpu")

    transition = create_transition(observation={}, action=torch.randn(5))
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION] == {}
    assert result[TransitionKey.ACTION].device.type == "cpu"


def test_scalar_tensors():
    """Test handling of scalar tensors."""
    processor = DeviceProcessorStep(device="cpu")

    observation = {"observation.scalar": torch.tensor(1.5)}
    action = torch.tensor(2.0)
    reward = torch.tensor(0.5)

    transition = create_transition(observation=observation, action=action, reward=reward)

    result = processor(transition)

    assert result[TransitionKey.OBSERVATION]["observation.scalar"].item() == 1.5
    assert result[TransitionKey.ACTION].item() == 2.0
    assert result[TransitionKey.REWARD].item() == 0.5


def test_dtype_preservation():
    """Test that tensor dtypes are preserved."""
    processor = DeviceProcessorStep(device="cpu")

    observation = {
        "observation.float32": torch.randn(5, dtype=torch.float32),
        "observation.float64": torch.randn(5, dtype=torch.float64),
        "observation.int32": torch.randint(0, 10, (5,), dtype=torch.int32),
        "observation.bool": torch.tensor([True, False, True], dtype=torch.bool),
    }
    action = torch.randn(3, dtype=torch.float16)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION]["observation.float32"].dtype == torch.float32
    assert result[TransitionKey.OBSERVATION]["observation.float64"].dtype == torch.float64
    assert result[TransitionKey.OBSERVATION]["observation.int32"].dtype == torch.int32
    assert result[TransitionKey.OBSERVATION]["observation.bool"].dtype == torch.bool
    assert result[TransitionKey.ACTION].dtype == torch.float16


def test_shape_preservation():
    """Test that tensor shapes are preserved."""
    processor = DeviceProcessorStep(device="cpu")

    observation = {
        "observation.1d": torch.randn(10),
        "observation.2d": torch.randn(5, 10),
        "observation.3d": torch.randn(3, 224, 224),
        "observation.4d": torch.randn(2, 3, 224, 224),
    }
    action = torch.randn(2, 5, 3)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION]["observation.1d"].shape == (10,)
    assert result[TransitionKey.OBSERVATION]["observation.2d"].shape == (5, 10)
    assert result[TransitionKey.OBSERVATION]["observation.3d"].shape == (3, 224, 224)
    assert result[TransitionKey.OBSERVATION]["observation.4d"].shape == (2, 3, 224, 224)
    assert result[TransitionKey.ACTION].shape == (2, 5, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_devices():
    """Test handling of tensors already on different devices."""
    processor = DeviceProcessorStep(device="cuda")

    # Create tensors on different devices
    observation = {
        "observation.cpu": torch.randn(5),  # CPU
        "observation.cuda": torch.randn(5).cuda(),  # Already on CUDA
    }
    action = torch.randn(3).cuda()  # Already on CUDA

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # All should be on CUDA
    assert result[TransitionKey.OBSERVATION]["observation.cpu"].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION]["observation.cuda"].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.type == "cuda"


def test_non_blocking_flag():
    """Test that non_blocking flag is set correctly."""
    # CPU processor should have non_blocking=False
    cpu_processor = DeviceProcessorStep(device="cpu")
    assert cpu_processor.non_blocking is False

    if torch.cuda.is_available():
        # CUDA processor should have non_blocking=True
        cuda_processor = DeviceProcessorStep(device="cuda")
        assert cuda_processor.non_blocking is True

        cuda_0_processor = DeviceProcessorStep(device="cuda:0")
        assert cuda_0_processor.non_blocking is True


def test_serialization_methods():
    """Test get_config, state_dict, and load_state_dict methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DeviceProcessorStep(device=device)

    # Test get_config
    config = processor.get_config()
    assert config == {"device": device, "float_dtype": None}

    # Test state_dict (should be empty)
    state = processor.state_dict()
    assert state == {}

    # Test load_state_dict (should be no-op)
    processor.load_state_dict({})
    assert processor.device == device

    # Test reset (should be no-op)
    processor.reset()
    assert processor.device == device


def test_features():
    """Test that features returns features unchanged."""
    processor = DeviceProcessorStep(device="cpu")

    features = {
        PipelineFeatureType.OBSERVATION: {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
        PipelineFeatureType.ACTION: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
    }

    result = processor.transform_features(features)
    assert result == features
    assert result is features  # Should return the same object


def test_integration_with_robot_processor():
    """Test integration with RobotProcessor."""
    from lerobot.processor import AddBatchDimensionProcessorStep
    from lerobot.utils.constants import OBS_STATE

    # Create a pipeline with DeviceProcessorStep
    device_processor = DeviceProcessorStep(device="cpu")
    batch_processor = AddBatchDimensionProcessorStep()

    processor = DataProcessorPipeline(
        steps=[batch_processor, device_processor],
        name="test_pipeline",
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    # Create test data
    observation = {OBS_STATE: torch.randn(10)}
    action = torch.randn(5)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Check that tensors are batched and on correct device
    # The result has TransitionKey.OBSERVATION as the key, with observation.state inside
    assert result[TransitionKey.OBSERVATION][OBS_STATE].shape[0] == 1  # Batched
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cpu"
    assert result[TransitionKey.ACTION].shape[0] == 1  # Batched
    assert result[TransitionKey.ACTION].device.type == "cpu"


def test_save_and_load_pretrained():
    """Test saving and loading processor with DeviceProcessorStep."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = DeviceProcessorStep(device=device, float_dtype="float16")
    robot_processor = DataProcessorPipeline(steps=[processor], name="device_test_processor")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        robot_processor.save_pretrained(tmpdir)

        # Load
        loaded_processor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="device_test_processor.json"
        )

        assert len(loaded_processor.steps) == 1
        loaded_device_processor = loaded_processor.steps[0]
        assert isinstance(loaded_device_processor, DeviceProcessorStep)
        # Use getattr to access attributes safely
        assert (
            getattr(loaded_device_processor, "device", None) == device.split(":")[0]
        )  # Device normalizes cuda:0 to cuda
        assert getattr(loaded_device_processor, "float_dtype", None) == "float16"


def test_registry_functionality():
    """Test that DeviceProcessorStep is properly registered."""
    from lerobot.processor import ProcessorStepRegistry

    # Check that DeviceProcessorStep is registered
    registered_class = ProcessorStepRegistry.get("device_processor")
    assert registered_class is DeviceProcessorStep


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_performance_with_large_tensors():
    """Test performance with large tensors and non_blocking flag."""
    processor = DeviceProcessorStep(device="cuda")

    # Create large tensors
    observation = {
        "observation.large_image": torch.randn(10, 3, 512, 512),  # Large image batch
        "observation.features": torch.randn(10, 2048),  # Large feature vector
    }
    action = torch.randn(10, 100)  # Large action space

    transition = create_transition(observation=observation, action=action)

    # Process should not raise any errors
    result = processor(transition)

    # Verify all tensors are on CUDA
    assert result[TransitionKey.OBSERVATION]["observation.large_image"].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION]["observation.features"].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.type == "cuda"


def test_reward_done_truncated_types():
    """Test handling of different types for reward, done, and truncated."""
    processor = DeviceProcessorStep(device="cpu")

    # Test with scalar values (not tensors)
    transition = create_transition(
        observation={OBS_STATE: torch.randn(5)},
        action=torch.randn(3),
        reward=1.0,  # float
        done=False,  # bool
        truncated=True,  # bool
    )

    result = processor(transition)

    # Non-tensor values should be preserved as-is
    assert result[TransitionKey.REWARD] == 1.0
    assert result[TransitionKey.DONE] is False
    assert result[TransitionKey.TRUNCATED] is True

    # Test with tensor values
    transition = create_transition(
        observation={OBS_STATE: torch.randn(5)},
        action=torch.randn(3),
        reward=torch.tensor(1.0),
        done=torch.tensor(False),
        truncated=torch.tensor(True),
    )

    result = processor(transition)

    # Tensor values should be moved to device
    assert isinstance(result[TransitionKey.REWARD], torch.Tensor)
    assert isinstance(result[TransitionKey.DONE], torch.Tensor)
    assert isinstance(result[TransitionKey.TRUNCATED], torch.Tensor)
    assert result[TransitionKey.REWARD].device.type == "cpu"
    assert result[TransitionKey.DONE].device.type == "cpu"
    assert result[TransitionKey.TRUNCATED].device.type == "cpu"


def test_complementary_data_preserved():
    """Test that complementary_data is preserved unchanged."""
    processor = DeviceProcessorStep(device="cpu")

    complementary_data = {
        "task": "pick_object",
        "episode_id": 42,
        "metadata": {"sensor": "camera_1"},
        "observation_is_pad": torch.tensor([False, False, True]),  # This should be moved to device
    }

    transition = create_transition(
        observation={OBS_STATE: torch.randn(5)}, complementary_data=complementary_data
    )

    result = processor(transition)

    # Check that complementary_data is preserved
    assert TransitionKey.COMPLEMENTARY_DATA in result
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "pick_object"
    assert result[TransitionKey.COMPLEMENTARY_DATA]["episode_id"] == 42
    assert result[TransitionKey.COMPLEMENTARY_DATA]["metadata"] == {"sensor": "camera_1"}
    # Note: Currently DeviceProcessorStep doesn't process tensors in complementary_data
    # This is intentional as complementary_data is typically metadata


def test_float_dtype_conversion():
    """Test float dtype conversion functionality."""
    processor = DeviceProcessorStep(device="cpu", float_dtype="float16")

    # Create tensors of different types
    observation = {
        "observation.float32": torch.randn(5, dtype=torch.float32),
        "observation.float64": torch.randn(5, dtype=torch.float64),
        "observation.int32": torch.randint(0, 10, (5,), dtype=torch.int32),
        "observation.int64": torch.randint(0, 10, (5,), dtype=torch.int64),
        "observation.bool": torch.tensor([True, False, True], dtype=torch.bool),
    }
    action = torch.randn(3, dtype=torch.float32)
    reward = torch.tensor(1.0, dtype=torch.float32)

    transition = create_transition(observation=observation, action=action, reward=reward)
    result = processor(transition)

    # Check that float tensors are converted to float16
    assert result[TransitionKey.OBSERVATION]["observation.float32"].dtype == torch.float16
    assert result[TransitionKey.OBSERVATION]["observation.float64"].dtype == torch.float16
    assert result[TransitionKey.ACTION].dtype == torch.float16
    assert result[TransitionKey.REWARD].dtype == torch.float16

    # Check that non-float tensors are preserved
    assert result[TransitionKey.OBSERVATION]["observation.int32"].dtype == torch.int32
    assert result[TransitionKey.OBSERVATION]["observation.int64"].dtype == torch.int64
    assert result[TransitionKey.OBSERVATION]["observation.bool"].dtype == torch.bool


def test_float_dtype_none():
    """Test that when float_dtype is None, no dtype conversion occurs."""
    processor = DeviceProcessorStep(device="cpu", float_dtype=None)

    observation = {
        "observation.float32": torch.randn(5, dtype=torch.float32),
        "observation.float64": torch.randn(5, dtype=torch.float64),
        "observation.int32": torch.randint(0, 10, (5,), dtype=torch.int32),
    }
    action = torch.randn(3, dtype=torch.float64)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Check that dtypes are preserved when float_dtype is None
    assert result[TransitionKey.OBSERVATION]["observation.float32"].dtype == torch.float32
    assert result[TransitionKey.OBSERVATION]["observation.float64"].dtype == torch.float64
    assert result[TransitionKey.OBSERVATION]["observation.int32"].dtype == torch.int32
    assert result[TransitionKey.ACTION].dtype == torch.float64


def test_float_dtype_bfloat16():
    """Test conversion to bfloat16."""
    processor = DeviceProcessorStep(device="cpu", float_dtype="bfloat16")

    observation = {OBS_STATE: torch.randn(5, dtype=torch.float32)}
    action = torch.randn(3, dtype=torch.float64)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.bfloat16
    assert result[TransitionKey.ACTION].dtype == torch.bfloat16


def test_float_dtype_float64():
    """Test conversion to float64."""
    processor = DeviceProcessorStep(device="cpu", float_dtype="float64")

    observation = {OBS_STATE: torch.randn(5, dtype=torch.float16)}
    action = torch.randn(3, dtype=torch.float32)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float64
    assert result[TransitionKey.ACTION].dtype == torch.float64


def test_float_dtype_invalid():
    """Test that invalid float_dtype raises ValueError."""
    with pytest.raises(ValueError, match="Invalid float_dtype 'invalid_dtype'"):
        DeviceProcessorStep(device="cpu", float_dtype="invalid_dtype")


def test_float_dtype_aliases():
    """Test that dtype aliases work correctly."""
    # Test 'half' alias for float16
    processor_half = DeviceProcessorStep(device="cpu", float_dtype="half")
    assert processor_half._target_float_dtype == torch.float16

    # Test 'float' alias for float32
    processor_float = DeviceProcessorStep(device="cpu", float_dtype="float")
    assert processor_float._target_float_dtype == torch.float32

    # Test 'double' alias for float64
    processor_double = DeviceProcessorStep(device="cpu", float_dtype="double")
    assert processor_double._target_float_dtype == torch.float64


def test_float_dtype_with_mixed_tensors():
    """Test float dtype conversion with mixed tensor types."""
    processor = DeviceProcessorStep(device="cpu", float_dtype="float32")

    observation = {
        OBS_IMAGE: torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),  # Should not convert
        OBS_STATE: torch.randn(10, dtype=torch.float64),  # Should convert
        "observation.mask": torch.tensor([True, False, True], dtype=torch.bool),  # Should not convert
        "observation.indices": torch.tensor([1, 2, 3], dtype=torch.long),  # Should not convert
    }
    action = torch.randn(5, dtype=torch.float16)  # Should convert

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Check conversions
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].dtype == torch.uint8  # Unchanged
    assert result[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float32  # Converted
    assert result[TransitionKey.OBSERVATION]["observation.mask"].dtype == torch.bool  # Unchanged
    assert result[TransitionKey.OBSERVATION]["observation.indices"].dtype == torch.long  # Unchanged
    assert result[TransitionKey.ACTION].dtype == torch.float32  # Converted


def test_float_dtype_serialization():
    """Test that float_dtype is properly serialized in get_config."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DeviceProcessorStep(device=device, float_dtype="float16")
    config = processor.get_config()

    assert config == {"device": device, "float_dtype": "float16"}

    # Test with None float_dtype
    processor_none = DeviceProcessorStep(device="cpu", float_dtype=None)
    config_none = processor_none.get_config()

    assert config_none == {"device": "cpu", "float_dtype": None}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_float_dtype_with_cuda():
    """Test float dtype conversion combined with CUDA device."""
    processor = DeviceProcessorStep(device="cuda", float_dtype="float16")

    # Create tensors on CPU with different dtypes
    observation = {
        "observation.float32": torch.randn(5, dtype=torch.float32),
        "observation.int64": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    action = torch.randn(3, dtype=torch.float64)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Check that tensors are on CUDA and float types are converted
    assert result[TransitionKey.OBSERVATION]["observation.float32"].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION]["observation.float32"].dtype == torch.float16

    assert result[TransitionKey.OBSERVATION]["observation.int64"].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION]["observation.int64"].dtype == torch.int64  # Unchanged

    assert result[TransitionKey.ACTION].device.type == "cuda"
    assert result[TransitionKey.ACTION].dtype == torch.float16


def test_complementary_data_index_fields():
    """Test processing of index and task_index fields in complementary_data."""
    processor = DeviceProcessorStep(device="cpu")

    # Create transition with index and task_index in complementary_data
    complementary_data = {
        "task": ["pick_cube"],
        "index": torch.tensor([42], dtype=torch.int64),
        "task_index": torch.tensor([3], dtype=torch.int64),
        "episode_id": 123,  # Non-tensor field
    }
    transition = create_transition(
        observation={OBS_STATE: torch.randn(1, 7)},
        action=torch.randn(1, 4),
        complementary_data=complementary_data,
    )

    result = processor(transition)

    # Check that tensors in complementary_data are processed
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Check index tensor
    assert isinstance(processed_comp_data["index"], torch.Tensor)
    assert processed_comp_data["index"].device.type == "cpu"
    assert torch.equal(processed_comp_data["index"], complementary_data["index"])

    # Check task_index tensor
    assert isinstance(processed_comp_data["task_index"], torch.Tensor)
    assert processed_comp_data["task_index"].device.type == "cpu"
    assert torch.equal(processed_comp_data["task_index"], complementary_data["task_index"])

    # Check non-tensor fields remain unchanged
    assert processed_comp_data["task"] == ["pick_cube"]
    assert processed_comp_data["episode_id"] == 123


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_complementary_data_index_fields_cuda():
    """Test moving index and task_index fields to CUDA."""
    processor = DeviceProcessorStep(device="cuda:0")

    # Create CPU tensors
    complementary_data = {
        "index": torch.tensor([100, 101], dtype=torch.int64),
        "task_index": torch.tensor([5], dtype=torch.int64),
    }
    transition = create_transition(complementary_data=complementary_data)

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Check tensors moved to CUDA
    assert processed_comp_data["index"].device.type == "cuda"
    assert processed_comp_data["index"].device.index == 0
    assert processed_comp_data["task_index"].device.type == "cuda"
    assert processed_comp_data["task_index"].device.index == 0


def test_complementary_data_without_index_fields():
    """Test that complementary_data without index/task_index fields works correctly."""
    processor = DeviceProcessorStep(device="cpu")

    complementary_data = {
        "task": ["navigate"],
        "episode_id": 456,
    }
    transition = create_transition(complementary_data=complementary_data)

    result = processor(transition)

    # Should process without errors and preserve non-tensor fields
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["task"] == ["navigate"]
    assert processed_comp_data["episode_id"] == 456


def test_complementary_data_mixed_tensors():
    """Test complementary_data with mix of tensors and non-tensors."""
    processor = DeviceProcessorStep(device="cpu")

    complementary_data = {
        "task": ["pick_and_place"],
        "index": torch.tensor([42], dtype=torch.int64),
        "task_index": torch.tensor([3], dtype=torch.int64),
        "metrics": [1.0, 2.0, 3.0],  # List, not tensor
        "config": {"speed": "fast"},  # Dict
        "episode_id": 789,  # Int
    }
    transition = create_transition(complementary_data=complementary_data)

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Check tensors are processed
    assert isinstance(processed_comp_data["index"], torch.Tensor)
    assert isinstance(processed_comp_data["task_index"], torch.Tensor)

    # Check non-tensors remain unchanged
    assert processed_comp_data["task"] == ["pick_and_place"]
    assert processed_comp_data["metrics"] == [1.0, 2.0, 3.0]
    assert processed_comp_data["config"] == {"speed": "fast"}
    assert processed_comp_data["episode_id"] == 789


def test_complementary_data_float_dtype_conversion():
    """Test that float dtype conversion doesn't affect int tensors in complementary_data."""
    processor = DeviceProcessorStep(device="cpu", float_dtype="float16")

    complementary_data = {
        "index": torch.tensor([42], dtype=torch.int64),
        "task_index": torch.tensor([3], dtype=torch.int64),
        "float_tensor": torch.tensor([1.5, 2.5], dtype=torch.float32),  # Should be converted
    }
    transition = create_transition(complementary_data=complementary_data)

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Int tensors should keep their dtype
    assert processed_comp_data["index"].dtype == torch.int64
    assert processed_comp_data["task_index"].dtype == torch.int64

    # Float tensor should be converted
    assert processed_comp_data["float_tensor"].dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_complementary_data_full_pipeline_cuda():
    """Test full transition with complementary_data on CUDA."""
    processor = DeviceProcessorStep(device="cuda:0", float_dtype="float16")

    # Create full transition with mixed CPU tensors
    observation = {OBS_STATE: torch.randn(1, 7, dtype=torch.float32)}
    action = torch.randn(1, 4, dtype=torch.float32)
    reward = torch.tensor(1.5, dtype=torch.float32)
    done = torch.tensor(False)
    complementary_data = {
        "task": ["reach_target"],
        "index": torch.tensor([1000], dtype=torch.int64),
        "task_index": torch.tensor([10], dtype=torch.int64),
    }

    transition = create_transition(
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        complementary_data=complementary_data,
    )

    result = processor(transition)

    # Check all components moved to CUDA
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.type == "cuda"
    assert result[TransitionKey.REWARD].device.type == "cuda"
    assert result[TransitionKey.DONE].device.type == "cuda"

    # Check complementary_data tensors
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["index"].device.type == "cuda"
    assert processed_comp_data["task_index"].device.type == "cuda"

    # Check float conversion happened for float tensors
    assert result[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float16
    assert result[TransitionKey.ACTION].dtype == torch.float16
    assert result[TransitionKey.REWARD].dtype == torch.float16

    # Check int tensors kept their dtype
    assert processed_comp_data["index"].dtype == torch.int64
    assert processed_comp_data["task_index"].dtype == torch.int64


def test_complementary_data_empty():
    """Test empty complementary_data handling."""
    processor = DeviceProcessorStep(device="cpu")

    transition = create_transition(
        observation={OBS_STATE: torch.randn(1, 7)},
        complementary_data={},
    )

    result = processor(transition)

    # Should have empty dict
    assert result[TransitionKey.COMPLEMENTARY_DATA] == {}


def test_complementary_data_none():
    """Test None complementary_data handling."""
    processor = DeviceProcessorStep(device="cpu")

    transition = create_transition(
        observation={OBS_STATE: torch.randn(1, 7)},
        complementary_data=None,
    )

    result = processor(transition)

    # Complementary data should not be in the result (same as input)
    assert result[TransitionKey.COMPLEMENTARY_DATA] == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_preserves_gpu_placement():
    """Test that DeviceProcessorStep preserves GPU placement when tensor is already on GPU."""
    processor = DeviceProcessorStep(device="cuda:0")

    # Create tensors already on GPU
    observation = {
        OBS_STATE: torch.randn(10).cuda(),  # Already on GPU
        OBS_IMAGE: torch.randn(3, 224, 224).cuda(),  # Already on GPU
    }
    action = torch.randn(5).cuda()  # Already on GPU

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Check that tensors remain on their original GPU
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.type == "cuda"

    # Verify no unnecessary copies were made (same data pointer)
    assert torch.equal(result[TransitionKey.OBSERVATION][OBS_STATE], observation[OBS_STATE])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_multi_gpu_preservation():
    """Test that DeviceProcessorStep preserves placement on different GPUs in multi-GPU setup."""
    # Test 1: GPU-to-GPU preservation (cuda:0 config, cuda:1 input)
    processor_gpu = DeviceProcessorStep(device="cuda:0")

    # Create tensors on cuda:1 (simulating Accelerate placement)
    cuda1_device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(10).to(cuda1_device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(cuda1_device),
    }
    action = torch.randn(5).to(cuda1_device)

    transition = create_transition(observation=observation, action=action)
    result = processor_gpu(transition)

    # Check that tensors remain on cuda:1 (not moved to cuda:0)
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device == cuda1_device
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].device == cuda1_device
    assert result[TransitionKey.ACTION].device == cuda1_device

    # Test 2: GPU-to-CPU should move to CPU (not preserve GPU)
    processor_cpu = DeviceProcessorStep(device="cpu")

    transition_gpu = create_transition(
        observation={OBS_STATE: torch.randn(10).cuda()}, action=torch.randn(5).cuda()
    )
    result_cpu = processor_cpu(transition_gpu)

    # Check that tensors are moved to CPU
    assert result_cpu[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cpu"
    assert result_cpu[TransitionKey.ACTION].device.type == "cpu"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_multi_gpu_with_cpu_tensors():
    """Test that CPU tensors are moved to configured device even in multi-GPU context."""
    # Processor configured for cuda:1
    processor = DeviceProcessorStep(device="cuda:1")

    # Mix of CPU and GPU tensors
    observation = {
        "observation.cpu": torch.randn(10),  # CPU tensor
        "observation.gpu0": torch.randn(10).cuda(0),  # Already on cuda:0
        "observation.gpu1": torch.randn(10).cuda(1),  # Already on cuda:1
    }
    action = torch.randn(5)  # CPU tensor

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # CPU tensor should move to configured device (cuda:1)
    assert result[TransitionKey.OBSERVATION]["observation.cpu"].device.type == "cuda"
    assert result[TransitionKey.OBSERVATION]["observation.cpu"].device.index == 1
    assert result[TransitionKey.ACTION].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.index == 1

    # GPU tensors should stay on their original devices
    assert result[TransitionKey.OBSERVATION]["observation.gpu0"].device.index == 0
    assert result[TransitionKey.OBSERVATION]["observation.gpu1"].device.index == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_multi_gpu_with_float_dtype():
    """Test float dtype conversion works correctly with multi-GPU preservation."""
    processor = DeviceProcessorStep(device="cuda:0", float_dtype="float16")

    # Create float tensors on different GPUs
    observation = {
        "observation.gpu0": torch.randn(5, dtype=torch.float32).cuda(0),
        "observation.gpu1": torch.randn(5, dtype=torch.float32).cuda(1),
        "observation.cpu": torch.randn(5, dtype=torch.float32),  # CPU
    }

    transition = create_transition(observation=observation)
    result = processor(transition)

    # Check device placement
    assert result[TransitionKey.OBSERVATION]["observation.gpu0"].device.index == 0
    assert result[TransitionKey.OBSERVATION]["observation.gpu1"].device.index == 1
    assert result[TransitionKey.OBSERVATION]["observation.cpu"].device.index == 0  # Moved to cuda:0

    # Check dtype conversion happened for all
    assert result[TransitionKey.OBSERVATION]["observation.gpu0"].dtype == torch.float16
    assert result[TransitionKey.OBSERVATION]["observation.gpu1"].dtype == torch.float16
    assert result[TransitionKey.OBSERVATION]["observation.cpu"].dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_simulated_accelerate_scenario():
    """Test a scenario simulating how Accelerate would use the processor."""
    # Simulate different processes getting different GPU assignments
    for gpu_id in range(min(torch.cuda.device_count(), 2)):
        # Each "process" has a processor configured for cuda:0
        # but data comes in already placed on the process's GPU
        processor = DeviceProcessorStep(device="cuda:0")

        # Simulate data already placed by Accelerate
        device = torch.device(f"cuda:{gpu_id}")
        observation = {OBS_STATE: torch.randn(1, 10).to(device)}
        action = torch.randn(1, 5).to(device)

        transition = create_transition(observation=observation, action=action)
        result = processor(transition)

        # Verify data stays on the GPU where Accelerate placed it
        assert result[TransitionKey.OBSERVATION][OBS_STATE].device == device
        assert result[TransitionKey.ACTION].device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_policy_processor_integration():
    """Test integration with policy processors - input on GPU, output on CPU."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.utils.constants import ACTION, OBS_STATE

    # Create features and stats
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
    }

    stats = {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        ACTION: {"mean": torch.zeros(5), "std": torch.ones(5)},
    }

    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}

    # Create input processor (preprocessor) that moves to GPU
    input_processor = DataProcessorPipeline(
        steps=[
            NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device="cuda"),
        ],
        name="test_preprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    # Create output processor (postprocessor) that moves to CPU
    output_processor = DataProcessorPipeline(
        steps=[
            DeviceProcessorStep(device="cpu"),
            UnnormalizerProcessorStep(features={ACTION: features[ACTION]}, norm_map=norm_map, stats=stats),
        ],
        name="test_postprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    # Test data on CPU
    observation = {OBS_STATE: torch.randn(10)}
    action = torch.randn(5)
    transition = create_transition(observation=observation, action=action)

    # Process through input processor
    input_result = input_processor(transition)

    # Verify tensors are on GPU and batched
    # The result has TransitionKey.OBSERVATION as the key, with observation.state inside
    assert input_result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert input_result[TransitionKey.OBSERVATION][OBS_STATE].shape[0] == 1
    assert input_result[TransitionKey.ACTION].device.type == "cuda"
    assert input_result[TransitionKey.ACTION].shape[0] == 1

    # Simulate model output on GPU
    model_output = create_transition(action=torch.randn(1, 5).cuda())

    # Process through output processor
    output_result = output_processor(model_output)

    # Verify action is back on CPU and unnormalized
    assert output_result[TransitionKey.ACTION].device.type == "cpu"
    assert output_result[TransitionKey.ACTION].shape == (1, 5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_float64_compatibility():
    """Test MPS device compatibility with float64 tensors (automatic conversion to float32)."""
    processor = DeviceProcessorStep(device="mps")

    # Create tensors with different dtypes, including float64 which MPS doesn't support
    observation = {
        "observation.float64": torch.randn(5, dtype=torch.float64),  # Should be converted to float32
        "observation.float32": torch.randn(5, dtype=torch.float32),  # Should remain float32
        "observation.float16": torch.randn(5, dtype=torch.float16),  # Should remain float16
        "observation.int64": torch.randint(0, 10, (5,), dtype=torch.int64),  # Should remain int64
        "observation.bool": torch.tensor([True, False, True], dtype=torch.bool),  # Should remain bool
    }
    action = torch.randn(3, dtype=torch.float64)  # Should be converted to float32
    reward = torch.tensor(1.0, dtype=torch.float64)  # Should be converted to float32
    done = torch.tensor(False, dtype=torch.bool)  # Should remain bool
    truncated = torch.tensor(True, dtype=torch.bool)  # Should remain bool

    transition = create_transition(
        observation=observation, action=action, reward=reward, done=done, truncated=truncated
    )

    result = processor(transition)

    # Check that all tensors are on MPS device
    assert result[TransitionKey.OBSERVATION]["observation.float64"].device.type == "mps"
    assert result[TransitionKey.OBSERVATION]["observation.float32"].device.type == "mps"
    assert result[TransitionKey.OBSERVATION]["observation.float16"].device.type == "mps"
    assert result[TransitionKey.OBSERVATION]["observation.int64"].device.type == "mps"
    assert result[TransitionKey.OBSERVATION]["observation.bool"].device.type == "mps"
    assert result[TransitionKey.ACTION].device.type == "mps"
    assert result[TransitionKey.REWARD].device.type == "mps"
    assert result[TransitionKey.DONE].device.type == "mps"
    assert result[TransitionKey.TRUNCATED].device.type == "mps"

    # Check that float64 tensors were automatically converted to float32
    assert result[TransitionKey.OBSERVATION]["observation.float64"].dtype == torch.float32
    assert result[TransitionKey.ACTION].dtype == torch.float32
    assert result[TransitionKey.REWARD].dtype == torch.float32

    # Check that other dtypes were preserved
    assert result[TransitionKey.OBSERVATION]["observation.float32"].dtype == torch.float32
    assert result[TransitionKey.OBSERVATION]["observation.float16"].dtype == torch.float16
    assert result[TransitionKey.OBSERVATION]["observation.int64"].dtype == torch.int64
    assert result[TransitionKey.OBSERVATION]["observation.bool"].dtype == torch.bool
    assert result[TransitionKey.DONE].dtype == torch.bool
    assert result[TransitionKey.TRUNCATED].dtype == torch.bool


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_float64_with_complementary_data():
    """Test MPS float64 conversion with complementary_data tensors."""
    processor = DeviceProcessorStep(device="mps")

    # Create complementary_data with float64 tensors
    complementary_data = {
        "task": ["pick_object"],
        "index": torch.tensor([42], dtype=torch.int64),  # Should remain int64
        "task_index": torch.tensor([3], dtype=torch.int64),  # Should remain int64
        "float64_tensor": torch.tensor([1.5, 2.5], dtype=torch.float64),  # Should convert to float32
        "float32_tensor": torch.tensor([3.5], dtype=torch.float32),  # Should remain float32
    }

    transition = create_transition(
        observation={OBS_STATE: torch.randn(5, dtype=torch.float64)},
        action=torch.randn(3, dtype=torch.float64),
        complementary_data=complementary_data,
    )

    result = processor(transition)

    # Check that all tensors are on MPS device
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "mps"
    assert result[TransitionKey.ACTION].device.type == "mps"

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["index"].device.type == "mps"
    assert processed_comp_data["task_index"].device.type == "mps"
    assert processed_comp_data["float64_tensor"].device.type == "mps"
    assert processed_comp_data["float32_tensor"].device.type == "mps"

    # Check dtype conversions
    assert result[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float32  # Converted
    assert result[TransitionKey.ACTION].dtype == torch.float32  # Converted
    assert processed_comp_data["float64_tensor"].dtype == torch.float32  # Converted
    assert processed_comp_data["float32_tensor"].dtype == torch.float32  # Unchanged
    assert processed_comp_data["index"].dtype == torch.int64  # Unchanged
    assert processed_comp_data["task_index"].dtype == torch.int64  # Unchanged

    # Check non-tensor data preserved
    assert processed_comp_data["task"] == ["pick_object"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_with_explicit_float_dtype():
    """Test MPS device with explicit float_dtype setting."""
    # Test that explicit float_dtype still works on MPS
    processor = DeviceProcessorStep(device="mps", float_dtype="float16")

    observation = {
        "observation.float64": torch.randn(
            5, dtype=torch.float64
        ),  # First converted to float32, then to float16
        "observation.float32": torch.randn(5, dtype=torch.float32),  # Converted to float16
        "observation.int32": torch.randint(0, 10, (5,), dtype=torch.int32),  # Should remain int32
    }
    action = torch.randn(3, dtype=torch.float64)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Check device placement
    assert result[TransitionKey.OBSERVATION]["observation.float64"].device.type == "mps"
    assert result[TransitionKey.OBSERVATION]["observation.float32"].device.type == "mps"
    assert result[TransitionKey.OBSERVATION]["observation.int32"].device.type == "mps"
    assert result[TransitionKey.ACTION].device.type == "mps"

    # Check that all float tensors end up as float16 (the target dtype)
    assert result[TransitionKey.OBSERVATION]["observation.float64"].dtype == torch.float16
    assert result[TransitionKey.OBSERVATION]["observation.float32"].dtype == torch.float16
    assert result[TransitionKey.ACTION].dtype == torch.float16

    # Check that non-float tensors are preserved
    assert result[TransitionKey.OBSERVATION]["observation.int32"].dtype == torch.int32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_serialization():
    """Test that MPS device processor can be serialized and loaded correctly."""
    processor = DeviceProcessorStep(device="mps", float_dtype="float32")

    # Test get_config
    config = processor.get_config()
    assert config == {"device": "mps", "float_dtype": "float32"}

    # Test state_dict (should be empty)
    state = processor.state_dict()
    assert state == {}

    # Test load_state_dict (should be no-op)
    processor.load_state_dict({})
    assert processor.device == "mps"
