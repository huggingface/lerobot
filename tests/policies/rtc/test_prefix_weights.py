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

import pytest
import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


@pytest.mark.parametrize("time", [0.0, 0.2, 0.75, 1.0])
def test_explicit_prefix_weights_preserve_reference_guidance_and_leave_horizon_edges_alone(time):
    processor = RTCProcessor(RTCConfig(max_guidance_weight=3, execution_horizon=3))
    x = torch.linspace(-1, 1, 28).reshape(2, 7, 2)
    prev = torch.full_like(x, 0.3)
    weights = torch.tensor([0, 0, 1, 2 / 3, 1 / 3, 0, 0])

    def denoise(value):
        return 0.2 * value.sin() + value * 0.1

    expected = denoise(x)
    expected[:, 2:5] = processor.denoise_step(x[:, 2:5], prev[:, 2:5], 1, time, denoise)
    actual = processor.denoise_step(x, prev, 1, time, denoise, prefix_weights=weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-7)


@pytest.mark.parametrize("shape", [(7, 1), (1, 7, 1), (6,), ()])
def test_explicit_prefix_weights_reject_ambiguous_or_wrong_length_masks(shape):
    processor = RTCProcessor(RTCConfig())
    with pytest.raises(ValueError, match="prefix_weights.*shape"):
        processor.denoise_step(
            torch.zeros(2, 7, 2),
            torch.ones(2, 7, 2),
            1,
            0.5,
            lambda x: x * 0.2,
            prefix_weights=torch.ones(shape),
        )
