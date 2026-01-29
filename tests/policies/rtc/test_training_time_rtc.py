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

"""Tests for training-time RTC helpers."""

import torch

from lerobot.configs.types import RTCTrainingDelayDistribution
from lerobot.policies.rtc.configuration_rtc import RTCTrainingConfig
from lerobot.policies.rtc.training_time import apply_rtc_training_time, sample_rtc_delay


def test_rtc_training_config_defaults():
    config = RTCTrainingConfig()
    assert config.enabled is False
    assert config.min_delay == 0
    assert config.max_delay == 0
    assert config.delay_distribution == RTCTrainingDelayDistribution.UNIFORM
    assert config.exp_decay == 1.0


def test_sample_rtc_delay_uniform_range():
    cfg = RTCTrainingConfig(enabled=True, min_delay=1, max_delay=4)
    delays = sample_rtc_delay(cfg, batch_size=100, device=torch.device("cpu"))
    assert delays.min().item() >= 1
    assert delays.max().item() <= 4


def test_apply_rtc_training_time_prefix_mask():
    time = torch.tensor([0.5])
    delays = torch.tensor([2])
    time_tokens, postfix_mask = apply_rtc_training_time(time, delays, seq_len=4)
    assert time_tokens.shape == (1, 4)
    assert postfix_mask.shape == (1, 4)
    # Delay=2 means the first two steps are prefix (time forced to 0.0) and only the last two are postfix.
    assert torch.allclose(time_tokens[0], torch.tensor([0.0, 0.0, 0.5, 0.5]))
    assert torch.equal(postfix_mask[0], torch.tensor([False, False, True, True]))
