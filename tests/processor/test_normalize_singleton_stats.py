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

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep


@pytest.mark.parametrize("step_type", [NormalizerProcessorStep, UnnormalizerProcessorStep])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (2,)])
def test_checkpoint_stat_shapes_survive_device_conversion(step_type, shape):
    step = step_type(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
    )
    state = {"action.mean": torch.full(shape, 0.5), "action.std": torch.ones(shape)}
    step.load_state_dict(state)
    step.to(device="cpu")
    for key, expected in state.items():
        torch.testing.assert_close(step.state_dict()[key], expected)


@pytest.mark.parametrize("step_type", [NormalizerProcessorStep, UnnormalizerProcessorStep])
def test_grayscale_checkpoint_shape_survives_conversion(step_type):
    key = "observation.images.camera"
    step = step_type(
        features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 8, 8))},
        norm_map={FeatureType.VISUAL: NormalizationMode.MEAN_STD},
    )
    step.load_state_dict({f"{key}.mean": torch.ones(1, 1, 1)})
    step.to(device="cpu")
    assert step.state_dict()[f"{key}.mean"].shape == (1, 1, 1)
