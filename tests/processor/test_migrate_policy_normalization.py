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

import torch

from lerobot.processor.migrate_policy_normalization import extract_normalization_stats


def test_extracts_dataset_prefixed_state_stats():
    state_dict = {
        "normalize_inputs.so100_buffer_observation_state.mean": torch.tensor([1.0, 2.0]),
        "normalize_inputs.so100_buffer_observation_state.std": torch.tensor([3.0, 4.0]),
        "normalize_inputs.so100-blue_buffer_observation_state.mean": torch.tensor([5.0, 6.0]),
        "normalize_inputs.so100-blue_buffer_observation_state.std": torch.tensor([7.0, 8.0]),
    }

    stats = extract_normalization_stats(state_dict)

    torch.testing.assert_close(stats["so100.buffer.observation.state"]["mean"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(stats["so100.buffer.observation.state"]["std"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(stats["so100-blue.buffer.observation.state"]["mean"], torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(stats["so100-blue.buffer.observation.state"]["std"], torch.tensor([7.0, 8.0]))
