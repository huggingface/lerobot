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

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType
from lerobot.processor import (
    DataProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from tests.conftest import assert_contract_is_typed


def test_basic_renaming():
    """Test basic key renaming functionality."""
    rename_map = {
        "old_key1": "new_key1",
        "old_key2": "new_key2",
    }
    processor = RenameObservationsProcessorStep(rename_map=rename_map)

    observation = {
        "old_key1": torch.tensor([1.0, 2.0]),
        "old_key2": np.array([3.0, 4.0]),
        "unchanged_key": "keep_me",
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check renamed keys
    assert "new_key1" in processed_obs
    assert "new_key2" in processed_obs
    assert "old_key1" not in processed_obs
    assert "old_key2" not in processed_obs

    # Check values are preserved
    torch.testing.assert_close(processed_obs["new_key1"], torch.tensor([1.0, 2.0]))
    np.testing.assert_array_equal(processed_obs["new_key2"], np.array([3.0, 4.0]))

    # Check unchanged key is preserved
    assert processed_obs["unchanged_key"] == "keep_me"


def test_empty_rename_map():
    """Test processor with empty rename map (should pass through unchanged)."""
    processor = RenameObservationsProcessorStep(rename_map={})

    observation = {
        "key1": torch.tensor([1.0]),
        "key2": "value2",
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # All keys should be unchanged
    assert processed_obs.keys() == observation.keys()
    torch.testing.assert_close(processed_obs["key1"], observation["key1"])
    assert processed_obs["key2"] == observation["key2"]


def test_none_observation():
    """Test processor with None observation."""
    processor = RenameObservationsProcessorStep(rename_map={"old": "new"})

    transition = create_transition(observation={})
    result = processor(transition)

    # Should return transition unchanged
    assert result == transition


def test_overlapping_rename():
    """Test renaming when new names might conflict."""
    rename_map = {
        "a": "b",
        "b": "c",  # This creates a potential conflict
    }
    processor = RenameObservationsProcessorStep(rename_map=rename_map)

    observation = {
        "a": 1,
        "b": 2,
        "x": 3,
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that renaming happens correctly
    assert "a" not in processed_obs
    assert processed_obs["b"] == 1  # 'a' renamed to 'b'
    assert processed_obs["c"] == 2  # original 'b' renamed to 'c'
    assert processed_obs["x"] == 3


def test_partial_rename():
    """Test renaming only some keys."""
    rename_map = {
        OBS_STATE: "observation.proprio_state",
        "pixels": OBS_IMAGE,
    }
    processor = RenameObservationsProcessorStep(rename_map=rename_map)

    observation = {
        OBS_STATE: torch.randn(10),
        "pixels": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        "reward": 1.0,
        "info": {"episode": 1},
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check renamed keys
    assert "observation.proprio_state" in processed_obs
    assert OBS_IMAGE in processed_obs
    assert OBS_STATE not in processed_obs
    assert "pixels" not in processed_obs

    # Check unchanged keys
    assert processed_obs["reward"] == 1.0
    assert processed_obs["info"] == {"episode": 1}


def test_get_config():
    """Test configuration serialization."""
    rename_map = {
        "old1": "new1",
        "old2": "new2",
    }
    processor = RenameObservationsProcessorStep(rename_map=rename_map)

    config = processor.get_config()
    assert config == {"rename_map": rename_map}


def test_state_dict():
    """Test state dict (should be empty for RenameProcessorStep)."""
    processor = RenameObservationsProcessorStep(rename_map={"old": "new"})

    state = processor.state_dict()
    assert state == {}

    # Load state dict should work even with empty dict
    processor.load_state_dict({})


def test_integration_with_robot_processor():
    """Test integration with RobotProcessor pipeline."""
    rename_map = {
        "agent_pos": OBS_STATE,
        "pixels": OBS_IMAGE,
    }
    rename_processor = RenameObservationsProcessorStep(rename_map=rename_map)

    pipeline = DataProcessorPipeline(
        [rename_processor], to_transition=identity_transition, to_output=identity_transition
    )

    observation = {
        "agent_pos": np.array([1.0, 2.0, 3.0]),
        "pixels": np.zeros((32, 32, 3), dtype=np.uint8),
        "other_data": "preserve_me",
    }
    transition = create_transition(
        observation=observation, reward=0.5, done=False, truncated=False, info={}, complementary_data={}
    )

    result = pipeline(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check renaming worked through pipeline
    assert OBS_STATE in processed_obs
    assert OBS_IMAGE in processed_obs
    assert "agent_pos" not in processed_obs
    assert "pixels" not in processed_obs
    assert processed_obs["other_data"] == "preserve_me"

    # Check other transition elements unchanged
    assert result[TransitionKey.REWARD] == 0.5
    assert result[TransitionKey.DONE] is False


def test_save_and_load_pretrained():
    """Test saving and loading processor with RobotProcessor."""
    rename_map = {
        "old_state": OBS_STATE,
        "old_image": OBS_IMAGE,
    }
    processor = RenameObservationsProcessorStep(rename_map=rename_map)
    pipeline = DataProcessorPipeline([processor], name="TestRenameProcessorStep")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)

        # Check files were created
        config_path = (
            Path(tmp_dir) / "testrenameprocessorstep.json"
        )  # Based on name="TestRenameProcessorStep"
        assert config_path.exists()

        # No state files should be created for RenameProcessorStep
        state_files = list(Path(tmp_dir).glob("*.safetensors"))
        assert len(state_files) == 0

        # Load pipeline
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="testrenameprocessorstep.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        assert loaded_pipeline.name == "TestRenameProcessorStep"
        assert len(loaded_pipeline) == 1

        # Check that loaded processor works correctly
        loaded_processor = loaded_pipeline.steps[0]
        assert isinstance(loaded_processor, RenameObservationsProcessorStep)
        assert loaded_processor.rename_map == rename_map

        # Test functionality after loading
        observation = {"old_state": [1, 2, 3], "old_image": "image_data"}
        transition = create_transition(observation=observation)

        result = loaded_pipeline(transition)
        processed_obs = result[TransitionKey.OBSERVATION]

        assert OBS_STATE in processed_obs
        assert OBS_IMAGE in processed_obs
        assert processed_obs[OBS_STATE] == [1, 2, 3]
        assert processed_obs[OBS_IMAGE] == "image_data"


def test_registry_functionality():
    """Test that RenameProcessorStep is properly registered."""
    # Check that it's registered
    assert "rename_observations_processor" in ProcessorStepRegistry.list()

    # Get from registry
    retrieved_class = ProcessorStepRegistry.get("rename_observations_processor")
    assert retrieved_class is RenameObservationsProcessorStep

    # Create instance from registry
    instance = retrieved_class(rename_map={"old": "new"})
    assert isinstance(instance, RenameObservationsProcessorStep)
    assert instance.rename_map == {"old": "new"}


def test_registry_based_save_load():
    """Test save/load using registry name instead of module path."""
    processor = RenameObservationsProcessorStep(rename_map={"key1": "renamed_key1"})
    pipeline = DataProcessorPipeline(
        [processor], to_transition=identity_transition, to_output=identity_transition
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save and load
        pipeline.save_pretrained(tmp_dir)

        # Verify config uses registry name
        import json

        with open(Path(tmp_dir) / "dataprocessorpipeline.json") as f:  # Default name is "RobotProcessor"
            config = json.load(f)

        assert "registry_name" in config["steps"][0]
        assert config["steps"][0]["registry_name"] == "rename_observations_processor"
        assert "class" not in config["steps"][0]  # Should use registry, not module path

        # Load should work
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir, config_filename="dataprocessorpipeline.json"
        )
        loaded_processor = loaded_pipeline.steps[0]
        assert isinstance(loaded_processor, RenameObservationsProcessorStep)
        assert loaded_processor.rename_map == {"key1": "renamed_key1"}


def test_chained_rename_processors():
    """Test multiple RenameProcessorSteps in a pipeline."""
    # First processor: rename raw keys to intermediate format
    processor1 = RenameObservationsProcessorStep(
        rename_map={
            "pos": "agent_position",
            "img": "camera_image",
        }
    )

    # Second processor: rename to final format
    processor2 = RenameObservationsProcessorStep(
        rename_map={
            "agent_position": OBS_STATE,
            "camera_image": OBS_IMAGE,
        }
    )

    pipeline = DataProcessorPipeline(
        [processor1, processor2], to_transition=identity_transition, to_output=identity_transition
    )

    observation = {
        "pos": np.array([1.0, 2.0]),
        "img": "image_data",
        "extra": "keep_me",
    }
    transition = create_transition(observation=observation)

    # Step through to see intermediate results
    results = list(pipeline.step_through(transition))

    # After first processor
    assert "agent_position" in results[1][TransitionKey.OBSERVATION]
    assert "camera_image" in results[1][TransitionKey.OBSERVATION]

    # After second processor
    final_obs = results[2][TransitionKey.OBSERVATION]
    assert OBS_STATE in final_obs
    assert OBS_IMAGE in final_obs
    assert final_obs["extra"] == "keep_me"

    # Original keys should be gone
    assert "pos" not in final_obs
    assert "img" not in final_obs
    assert "agent_position" not in final_obs
    assert "camera_image" not in final_obs


def test_nested_observation_rename():
    """Test renaming with nested observation structures."""
    rename_map = {
        f"{OBS_IMAGES}.left": "observation.camera.left_view",
        f"{OBS_IMAGES}.right": "observation.camera.right_view",
        "observation.proprio": "observation.proprioception",
    }
    processor = RenameObservationsProcessorStep(rename_map=rename_map)

    observation = {
        f"{OBS_IMAGES}.left": torch.randn(3, 64, 64),
        f"{OBS_IMAGES}.right": torch.randn(3, 64, 64),
        "observation.proprio": torch.randn(7),
        "observation.gripper": torch.tensor([0.0]),  # Not renamed
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check renames
    assert "observation.camera.left_view" in processed_obs
    assert "observation.camera.right_view" in processed_obs
    assert "observation.proprioception" in processed_obs

    # Check unchanged key
    assert "observation.gripper" in processed_obs

    # Check old keys removed
    assert f"{OBS_IMAGES}.left" not in processed_obs
    assert f"{OBS_IMAGES}.right" not in processed_obs
    assert "observation.proprio" not in processed_obs


def test_value_types_preserved():
    """Test that various value types are preserved during renaming."""
    rename_map = {"old_tensor": "new_tensor", "old_array": "new_array", "old_scalar": "new_scalar"}
    processor = RenameObservationsProcessorStep(rename_map=rename_map)

    tensor_value = torch.randn(3, 3)
    array_value = np.random.rand(2, 2)

    observation = {
        "old_tensor": tensor_value,
        "old_array": array_value,
        "old_scalar": 42,
        "old_string": "hello",
        "old_dict": {"nested": "value"},
        "old_list": [1, 2, 3],
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that values and types are preserved
    assert torch.equal(processed_obs["new_tensor"], tensor_value)
    assert np.array_equal(processed_obs["new_array"], array_value)
    assert processed_obs["new_scalar"] == 42
    assert processed_obs["old_string"] == "hello"
    assert processed_obs["old_dict"] == {"nested": "value"}
    assert processed_obs["old_list"] == [1, 2, 3]


def test_features_basic_renaming(policy_feature_factory):
    processor = RenameObservationsProcessorStep(rename_map={"a": "x", "b": "y"})
    features = {
        PipelineFeatureType.OBSERVATION: {
            "a": policy_feature_factory(FeatureType.VISUAL, (2,)),
            "b": policy_feature_factory(FeatureType.VISUAL, (3,)),
            "c": policy_feature_factory(FeatureType.VISUAL, (1,)),
        },
    }

    out = processor.transform_features(features.copy())

    # Values preserved and typed
    assert out[PipelineFeatureType.OBSERVATION]["x"] == features[PipelineFeatureType.OBSERVATION]["a"]
    assert out[PipelineFeatureType.OBSERVATION]["y"] == features[PipelineFeatureType.OBSERVATION]["b"]
    assert out[PipelineFeatureType.OBSERVATION]["c"] == features[PipelineFeatureType.OBSERVATION]["c"]

    assert_contract_is_typed(out)
    # Input not mutated
    assert set(features[PipelineFeatureType.OBSERVATION]) == {"a", "b", "c"}


def test_features_overlapping_keys(policy_feature_factory):
    # Overlapping renames: both 'a' and 'b' exist. 'a'->'b', 'b'->'c'
    processor = RenameObservationsProcessorStep(rename_map={"a": "b", "b": "c"})
    features = {
        PipelineFeatureType.OBSERVATION: {
            "a": policy_feature_factory(FeatureType.VISUAL, (1,)),
            "b": policy_feature_factory(FeatureType.VISUAL, (2,)),
        },
    }
    out = processor.transform_features(features)

    assert set(out[PipelineFeatureType.OBSERVATION]) == {"b", "c"}
    assert (
        out[PipelineFeatureType.OBSERVATION]["b"] == features[PipelineFeatureType.OBSERVATION]["a"]
    )  # 'a' renamed to'b'
    assert (
        out[PipelineFeatureType.OBSERVATION]["c"] == features[PipelineFeatureType.OBSERVATION]["b"]
    )  # 'b' renamed to 'c'
    assert_contract_is_typed(out)


def test_features_chained_processors(policy_feature_factory):
    # Chain two rename processors at the contract level
    processor1 = RenameObservationsProcessorStep(rename_map={"pos": "agent_position", "img": "camera_image"})
    processor2 = RenameObservationsProcessorStep(
        rename_map={"agent_position": OBS_STATE, "camera_image": OBS_IMAGE}
    )
    pipeline = DataProcessorPipeline([processor1, processor2])

    spec = {
        PipelineFeatureType.OBSERVATION: {
            "pos": policy_feature_factory(FeatureType.VISUAL, (7,)),
            "img": policy_feature_factory(FeatureType.VISUAL, (3, 64, 64)),
            "extra": policy_feature_factory(FeatureType.VISUAL, (1,)),
        },
    }
    out = pipeline.transform_features(initial_features=spec)

    assert set(out[PipelineFeatureType.OBSERVATION]) == {OBS_STATE, OBS_IMAGE, "extra"}
    assert out[PipelineFeatureType.OBSERVATION][OBS_STATE] == spec[PipelineFeatureType.OBSERVATION]["pos"]
    assert out[PipelineFeatureType.OBSERVATION][OBS_IMAGE] == spec[PipelineFeatureType.OBSERVATION]["img"]
    assert out[PipelineFeatureType.OBSERVATION]["extra"] == spec[PipelineFeatureType.OBSERVATION]["extra"]
    assert_contract_is_typed(out)


def test_rename_stats_basic():
    orig = {
        OBS_STATE: {"mean": np.array([0.0]), "std": np.array([1.0])},
        ACTION: {"mean": np.array([0.0])},
    }
    mapping = {OBS_STATE: "observation.robot_state"}
    renamed = rename_stats(orig, mapping)
    assert "observation.robot_state" in renamed and OBS_STATE not in renamed
    # Ensure deep copy: mutate original and verify renamed unaffected
    orig[OBS_STATE]["mean"][0] = 42.0
    assert renamed["observation.robot_state"]["mean"][0] != 42.0
