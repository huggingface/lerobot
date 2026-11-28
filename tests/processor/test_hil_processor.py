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
import torch
from torch import nn

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import RewardClassifierProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


class DeviceCheckingClassifier(nn.Module):
    """Stand-in for `Classifier` that fails like conv2d does when inputs and weights sit on different devices."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def predict_reward(self, batch, threshold=0.5):
        for key, image in batch.items():
            if image.device != self.weight.device:
                raise RuntimeError(
                    f"{key}: input(device='{image.device}') and weight(device='{self.weight.device}') "
                    "must be on the same device"
                )
        return torch.tensor([1.0])


def test_reward_classifier_moves_images_to_classifier_device():
    # The `meta` device stands in for an accelerator (MPS, CUDA) so the test runs on CPU-only CI.
    step = RewardClassifierProcessorStep(device="meta")
    step.reward_classifier = DeviceCheckingClassifier().to("meta")

    observation = {OBS_IMAGE: torch.rand(1, 3, 8, 8), OBS_STATE: torch.rand(1, 6)}
    result = step(create_transition(observation=observation, reward=0.0, done=False))

    assert result[TransitionKey.REWARD] == step.success_reward
    assert result[TransitionKey.DONE] is True
    # Only the classifier input is moved; the observation passed down the pipeline keeps its device.
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].device.type == "cpu"
