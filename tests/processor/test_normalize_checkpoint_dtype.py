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


@pytest.mark.parametrize("processor_type", [NormalizerProcessorStep, UnnormalizerProcessorStep])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_checkpoint_stats_respect_configured_dtype(processor_type, dtype):
    processor = processor_type(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        dtype=dtype,
    )
    mean = torch.tensor([1.0000000001, 2.0], dtype=torch.float64)
    processor.load_state_dict({"action.mean": mean, "action.std": torch.ones(2, dtype=torch.float64)})
    saved = processor.state_dict()
    assert saved["action.mean"].dtype == dtype
    torch.testing.assert_close(saved["action.mean"], mean.to(dtype), rtol=0, atol=0)
    processor.to(dtype=dtype)
    torch.testing.assert_close(processor.state_dict()["action.mean"], mean.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("processor_type", [NormalizerProcessorStep, UnnormalizerProcessorStep])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_visual_checkpoint_retains_broadcast_shape(processor_type, dtype):
    processor = processor_type(
        features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))},
        norm_map={FeatureType.VISUAL: NormalizationMode.MEAN_STD},
        dtype=dtype,
    )
    processor.load_state_dict({"observation.image.mean": torch.tensor([0.1, 0.2, 0.3])})
    actual = processor.state_dict()["observation.image.mean"]
    assert actual.dtype == dtype
    assert actual.shape == (3, 1, 1)


def test_explicit_stats_still_override_checkpoint():
    processor = NormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": {"mean": [2.0, 3.0]}},
        dtype=torch.float64,
    )
    processor.load_state_dict({"action.mean": torch.zeros(2)})
    torch.testing.assert_close(
        processor.state_dict()["action.mean"], torch.tensor([2.0, 3.0], dtype=torch.float64)
    )
