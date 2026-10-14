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

from lerobot.policies.act.configuration_act import ACTConfig


@pytest.mark.parametrize("value", [0, -1])
def test_act_rejects_nonpositive_chunk_size(value):
    with pytest.raises(ValueError, match="chunk_size.*positive"):
        ACTConfig(device="cpu", chunk_size=value, n_action_steps=value)


@pytest.mark.parametrize("value", [0, -1])
def test_act_rejects_nonpositive_action_steps(value):
    with pytest.raises(ValueError, match="n_action_steps.*positive"):
        ACTConfig(device="cpu", n_action_steps=value)


@pytest.mark.parametrize("chunk_size,n_action_steps", [(1, 1), (20, 1), (20, 20)])
def test_act_accepts_positive_horizons(chunk_size, n_action_steps):
    ACTConfig(device="cpu", chunk_size=chunk_size, n_action_steps=n_action_steps)


def test_act_still_rejects_action_steps_exceeding_chunk_size():
    with pytest.raises(ValueError, match="chunk size is the upper bound"):
        ACTConfig(device="cpu", chunk_size=1, n_action_steps=2)


def test_act_still_accepts_single_step_temporal_ensemble():
    ACTConfig(device="cpu", chunk_size=1, n_action_steps=1, temporal_ensemble_coeff=0.01)
