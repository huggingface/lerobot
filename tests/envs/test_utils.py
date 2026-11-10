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

"""Env helpers: which checkpoints a simulator admits, and how its observations reach a policy."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.envs.utils import prepare_canonical_observation, validate_checkpoint
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep
from tests.fixtures.toy_env import ToyEnvConfig


class _Declares(ProcessorStep):
    """An env processor step that outputs one feature of its own shape, e.g. a decoded action or a built state."""

    def __init__(self, kind: PipelineFeatureType, feature: PolicyFeature, key: str):
        self.kind, self.feature, self.key = kind, feature, key

    def __call__(self, transition):
        return transition

    def get_config(self):
        return {"key": self.key, "shape": self.feature.shape}

    def transform_features(self, features):
        return {**features, self.kind: {**features[self.kind], self.key: self.feature}}


def _decodes_action(*shape: int) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline(
        steps=[_Declares(PipelineFeatureType.ACTION, PolicyFeature(FeatureType.ACTION, shape), "action")]
    )


def test_checkpoint_admission():
    env = ToyEnvConfig().create_env()
    descriptor = env.descriptor
    env.close()
    config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    validate_checkpoint(descriptor, config, {})
    # Acquisition resolution is independent of policy-side image preparation.
    for shape in ((3, 256, 256), (3, 480, 640)):
        config.input_features["observation.image"] = PolicyFeature(FeatureType.VISUAL, shape)
        validate_checkpoint(descriptor, config, {})
    for shape in ((4, 256, 256), (256, 256)):
        config.input_features["observation.image"] = PolicyFeature(FeatureType.VISUAL, shape)
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_checkpoint(descriptor, config, {})
    # A checkpoint camera named differently is admitted through --rename_map, as for any robot.
    del config.input_features["observation.image"]
    config.input_features["observation.images.front"] = PolicyFeature(FeatureType.VISUAL, (3, 8, 8))
    with pytest.raises(ValueError, match="missing from the simulator"):
        validate_checkpoint(descriptor, config, {})
    rename_map = {"observation.image": "observation.images.front"}
    validate_checkpoint(descriptor, config, rename_map)
    # Placeholder cameras are the policy's own (its empty_cameras setting): admission skips them.
    config.input_features["observation.images.empty_camera_0"] = PolicyFeature(
        FeatureType.VISUAL, (3, 480, 640)
    )
    with pytest.raises(ValueError, match="missing from the simulator"):
        validate_checkpoint(descriptor, config, rename_map)
    config.empty_cameras = 1
    validate_checkpoint(descriptor, config, rename_map)
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (3,))
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_checkpoint(descriptor, config, rename_map)
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (2,))
    config.output_features["action"] = PolicyFeature(FeatureType.ACTION, (7,))
    with pytest.raises(ValueError, match="action dimension"):
        validate_checkpoint(descriptor, config, rename_map)
    # A checkpoint is admitted on what its env processors produce: the action they command ...
    validate_checkpoint(descriptor, config, rename_map, env_postprocessor=_decodes_action(2))
    with pytest.raises(ValueError, match="command \\(3,\\) and the simulator expects \\(2,\\)"):
        validate_checkpoint(descriptor, config, rename_map, env_postprocessor=_decodes_action(3))
    # ... and the observation they build, renamed after them.
    config.output_features["action"] = PolicyFeature(FeatureType.ACTION, (2,))
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (5,))
    state = PolicyFeature(FeatureType.STATE, (5,))
    builds_state = PolicyProcessorPipeline(
        steps=[_Declares(PipelineFeatureType.OBSERVATION, state, "observation.state")]
    )
    validate_checkpoint(descriptor, config, rename_map, env_preprocessor=builds_state)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_checkpoint(descriptor, config, rename_map)


def test_prepare_canonical_observation():
    obs = {
        "observation.image": np.full((2, 8, 8, 3), 255, np.uint8),
        "observation.state": np.zeros((2, 2), np.float32),
    }
    converted = prepare_canonical_observation(obs)
    assert converted["observation.image"].shape == (2, 3, 8, 8)
    assert converted["observation.image"].max() == 1
    assert converted["observation.state"].dtype == torch.float32
