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

"""Which checkpoints a simulator admits."""

from types import SimpleNamespace

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.utils import validate_checkpoint
from tests.fixtures.toy_env import ToyEnvConfig


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
