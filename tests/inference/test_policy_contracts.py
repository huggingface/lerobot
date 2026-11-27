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

"""Regression checks at the checkpoint/model/canonical action boundary."""

from dataclasses import replace

import pytest
import torch

from lerobot.inference import ExecutionMode
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from tests.inference.policy_helpers import (
    ConformingPolicy,
    PaddedPolicy,
    make_runner,
    observation,
    tiny_config,
)

NAMES = ("shoulder.pos", "elbow.pos", "gripper.pos")


@pytest.mark.parametrize("names", [("elbow.pos", "shoulder.pos", "gripper.pos"), ()])
def test_checkpoint_order_is_enforced_without_relative_actions(names):
    config = tiny_config()
    config.action_feature_names = list(NAMES)
    with pytest.raises(ValueError, match="checkpoint action_feature_names"):
        make_runner(ConformingPolicy(config), action_names=names)


def test_state_permutation_is_rejected_even_without_checkpoint_names_or_relative_actions():
    with pytest.raises(ValueError, match="State and action component order"):
        make_runner(ConformingPolicy(tiny_config()), state_names=NAMES[::-1])


def test_checkpoint_order_accepts_matching_canonical_metadata():
    config = tiny_config()
    config.action_feature_names = list(NAMES)
    assert make_runner(ConformingPolicy(config)).predict(observation()).canonical_actions.shape == (3, 3)


def test_padded_prediction_uses_real_evo1_crop_before_canonical_validation():
    runner = make_runner(PaddedPolicy(tiny_config()), padded=True)
    chunk = runner.predict(observation())
    expected = torch.arange(48, dtype=torch.float32).reshape(8, 6)[:3, :3]
    expected = expected * 2 + torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(chunk.canonical_actions, expected)
    assert chunk.model_actions is None
    assert runner.capabilities.model_action_dim == 6


def test_padded_rtc_keeps_all_model_columns_in_successor_prefix():
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    policy = PaddedPolicy(config)
    runner = make_runner(policy, padded=True, modes=(ExecutionMode.RTC_GUIDED,))
    first = runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)
    assert first.model_actions.shape == (8, 6)
    assert first.canonical_actions.shape == (8, 3)
    second = runner.predict(
        replace(observation(), capture_time=11),
        mode=ExecutionMode.RTC_GUIDED,
        inference_delay=1,
        model_continuation=first.model_actions[-4:],
        canonical_continuation=first.canonical_actions[-4:],
    )
    torch.testing.assert_close(policy.last_kwargs["prev_chunk_left_over"], first.model_actions[-4:])
    assert second.model_actions.shape == (8, 6)
    policy.width = 7
    with pytest.raises(ValueError, match="width changed"):
        runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)


def test_width_changing_relative_rtc_requires_explicit_adapter():
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    runner = make_runner(PaddedPolicy(config), padded=True, relative=True, modes=(ExecutionMode.RTC_GUIDED,))
    with pytest.raises(ValueError, match="different model/canonical widths"):
        runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)
