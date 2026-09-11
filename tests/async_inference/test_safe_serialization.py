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

"""Tests for the safe serialization module."""

import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.safe_serialization import (
    deserialize_actions,
    deserialize_observation,
    deserialize_policy_config,
    serialize_actions,
    serialize_observation,
    serialize_policy_config,
)
from lerobot.configs.types import FeatureType, PolicyFeature


class TestPolicyConfigSerialization:
    def test_roundtrip(self):
        config = RemotePolicyConfig(
            policy_type="act",
            pretrained_name_or_path="user/model",
            lerobot_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            },
            actions_per_chunk=50,
            device="cuda",
            rename_map={"old_key": "new_key"},
        )
        data = serialize_policy_config(config)
        restored = deserialize_policy_config(data)

        assert restored.policy_type == config.policy_type
        assert restored.pretrained_name_or_path == config.pretrained_name_or_path
        assert restored.actions_per_chunk == config.actions_per_chunk
        assert restored.device == config.device
        assert restored.rename_map == config.rename_map

        for key in config.lerobot_features:
            assert key in restored.lerobot_features
            assert restored.lerobot_features[key].type == config.lerobot_features[key].type
            assert restored.lerobot_features[key].shape == config.lerobot_features[key].shape

    def test_rejects_invalid_type(self):
        data = serialize_observation(
            TimedObservation(timestamp=0.0, timestep=0, observation={"task": "test"})
        )
        with pytest.raises(ValueError, match="Expected RemotePolicyConfig"):
            deserialize_policy_config(data)

    def test_rejects_pickle_bytes(self):
        """Verify that raw pickle bytes are rejected."""
        import pickle

        raw_pickle = pickle.dumps({"policy_type": "act"})
        with pytest.raises(Exception):
            deserialize_policy_config(raw_pickle)


class TestObservationSerialization:
    def test_roundtrip_with_tensors(self):
        obs = TimedObservation(
            timestamp=1234567890.123,
            timestep=42,
            observation={
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "task": "pick up the cup",
            },
            must_go=True,
        )
        data = serialize_observation(obs)
        restored = deserialize_observation(data)

        assert restored.get_timestamp() == obs.get_timestamp()
        assert restored.get_timestep() == obs.get_timestep()
        assert restored.must_go == obs.must_go
        assert restored.get_observation()["task"] == "pick up the cup"
        assert torch.allclose(
            restored.get_observation()["observation.state"],
            obs.get_observation()["observation.state"],
        )

    def test_roundtrip_with_numpy(self):
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        obs = TimedObservation(
            timestamp=1.0,
            timestep=0,
            observation={
                "observation.images.front": image,
                "task": "test",
            },
        )
        data = serialize_observation(obs)
        restored = deserialize_observation(data)

        restored_image = restored.get_observation()["observation.images.front"]
        assert isinstance(restored_image, np.ndarray)
        np.testing.assert_array_equal(restored_image, image)

    def test_rejects_invalid_type(self):
        data = serialize_policy_config(
            RemotePolicyConfig(
                policy_type="act",
                pretrained_name_or_path="m",
                lerobot_features={},
                actions_per_chunk=1,
            )
        )
        with pytest.raises(ValueError, match="Expected TimedObservation"):
            deserialize_observation(data)


class TestActionsSerialization:
    def test_roundtrip(self):
        actions = [
            TimedAction(timestamp=1.0, timestep=0, action=torch.tensor([0.1, 0.2])),
            TimedAction(timestamp=1.033, timestep=1, action=torch.tensor([0.3, 0.4])),
            TimedAction(timestamp=1.066, timestep=2, action=torch.tensor([0.5, 0.6])),
        ]
        data = serialize_actions(actions)
        restored = deserialize_actions(data)

        assert len(restored) == len(actions)
        for orig, rest in zip(actions, restored):
            assert rest.get_timestamp() == orig.get_timestamp()
            assert rest.get_timestep() == orig.get_timestep()
            assert torch.allclose(rest.get_action(), orig.get_action())

    def test_empty_actions(self):
        data = serialize_actions([])
        restored = deserialize_actions(data)
        assert restored == []

    def test_rejects_invalid_type(self):
        obs = TimedObservation(timestamp=0.0, timestep=0, observation={"task": "t"})
        data = serialize_observation(obs)
        with pytest.raises(ValueError, match="Expected TimedActions"):
            deserialize_actions(data)
